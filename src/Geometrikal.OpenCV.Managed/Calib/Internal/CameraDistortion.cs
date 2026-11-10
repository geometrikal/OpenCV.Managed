// This implementation is ported from the OpenCV distortion model helpers.
//
// Original sources:
//   modules/geometry/src/distortion_model.hpp (computeTiltProjectionMatrix)
//   modules/core/include/opencv2/core/matx.inl.hpp (Matx_MatMulOp)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Distortion coefficient parsing, 3 × 3 matrix helpers with the OpenCV <c>Matx</c> operation
/// order, and the tilted-sensor projection matrices.
/// </summary>
internal static class CameraDistortion
{
    /// <summary>Validates a coefficient count (0, 4, 5, 8, 12 or 14) and returns 14 coefficients padded with zero.</summary>
    public static double[] Expand(ReadOnlySpan<double> coefficients, string paramName, bool allowEmpty = true)
    {
        int n = coefficients.Length;
        if (!((n == 0 && allowEmpty) || n == 4 || n == 5 || n == 8 || n == 12 || n == 14))
        {
            throw new ArgumentException("Distortion coefficients must have 4, 5, 8, 12 or 14 elements.", paramName);
        }

        var k = new double[14];
        coefficients.CopyTo(k);
        return k;
    }

    /// <summary>Reads a 3 × 3 single-channel matrix in row-major order.</summary>
    public static double[] Read3x3(Mat<double> m, string paramName)
    {
        ArgumentNullException.ThrowIfNull(m, paramName);
        if (m.Width != 3 || m.Height != 3 || m.Channels != 1)
        {
            throw new ArgumentException("The matrix must be 3 × 3 with one channel.", paramName);
        }

        var a = new double[9];
        for (int r = 0; r < 3; r++)
        {
            ReadOnlySpan<double> row = m.GetReadOnlyRowSpan(r);
            a[r * 3] = row[0];
            a[r * 3 + 1] = row[1];
            a[r * 3 + 2] = row[2];
        }

        return a;
    }

    /// <summary>Reads the left 3 × 3 block of a 3 × 3 or 3 × 4 single-channel matrix.</summary>
    public static double[] ReadLeft3x3(Mat<double> m, string paramName)
    {
        ArgumentNullException.ThrowIfNull(m, paramName);
        if (m.Height != 3 || (m.Width != 3 && m.Width != 4) || m.Channels != 1)
        {
            throw new ArgumentException("The matrix must be 3 × 3 or 3 × 4 with one channel.", paramName);
        }

        return Read3x3(m.Width == 3 ? m : m.Roi(0, 0, 3, 3), paramName);
    }

    /// <summary>Wraps a row-major 3 × 3 array as a matrix.</summary>
    public static Mat<double> ToMat(double[] a) => new((double[])a.Clone(), 3, 3);

    /// <summary><c>Matx33d * Matx33d</c>: each element sums from 0 in increasing k order.</summary>
    public static double[] Mul(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var c = new double[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double s = 0;
                s += a[i * 3] * b[j];
                s += a[i * 3 + 1] * b[3 + j];
                s += a[i * 3 + 2] * b[6 + j];
                c[i * 3 + j] = s;
            }
        }

        return c;
    }

    /// <summary><c>Matx33d * Vec3d(x, y, 1)</c> for one row: <c>((0 + m0·x) + m1·y) + m2·1</c>.</summary>
    public static double MulRow(ReadOnlySpan<double> m, int row, double x, double y)
    {
        double s = 0;
        s += m[row * 3] * x;
        s += m[row * 3 + 1] * y;
        s += m[row * 3 + 2] * 1.0;
        return s;
    }

    /// <summary>Returns the identity as a row-major array.</summary>
    public static double[] Identity() => [1, 0, 0, 0, 1, 0, 0, 0, 1];

    /// <summary>
    /// OpenCV <c>computeTiltProjectionMatrix&lt;double&gt;</c>. Each output is optional.
    /// </summary>
    public static void TiltProjection(
        double tauX,
        double tauY,
        double[]? matTilt,
        double[]? dTiltdTauX = null,
        double[]? dTiltdTauY = null,
        double[]? invMatTilt = null)
    {
        double cTauX = Math.Cos(tauX);
        double sTauX = Math.Sin(tauX);
        double cTauY = Math.Cos(tauY);
        double sTauY = Math.Sin(tauY);
        double[] matRotX = [1, 0, 0, 0, cTauX, sTauX, 0, -sTauX, cTauX];
        double[] matRotY = [cTauY, 0, -sTauY, 0, 1, 0, sTauY, 0, cTauY];
        double[] matRotXY = Mul(matRotY, matRotX);
        double[] matProjZ = [matRotXY[8], 0, -matRotXY[2], 0, matRotXY[8], -matRotXY[5], 0, 0, 1];
        if (matTilt is not null)
        {
            Mul(matProjZ, matRotXY).CopyTo(matTilt, 0);
        }

        if (dTiltdTauX is not null)
        {
            double[] dRotXY = Mul(matRotY, [0, 0, 0, 0, -sTauX, cTauX, 0, -cTauX, -sTauX]);
            double[] dProjZ = [dRotXY[8], 0, -dRotXY[2], 0, dRotXY[8], -dRotXY[5], 0, 0, 0];
            double[] t1 = Mul(matProjZ, dRotXY);
            double[] t2 = Mul(dProjZ, matRotXY);
            for (int i = 0; i < 9; i++)
            {
                dTiltdTauX[i] = t1[i] + t2[i];
            }
        }

        if (dTiltdTauY is not null)
        {
            double[] dRotXY = Mul([-sTauY, 0, -cTauY, 0, 0, 0, cTauY, 0, -sTauY], matRotX);
            double[] dProjZ = [dRotXY[8], 0, -dRotXY[2], 0, dRotXY[8], -dRotXY[5], 0, 0, 0];
            double[] t1 = Mul(matProjZ, dRotXY);
            double[] t2 = Mul(dProjZ, matRotXY);
            for (int i = 0; i < 9; i++)
            {
                dTiltdTauY[i] = t1[i] + t2[i];
            }
        }

        if (invMatTilt is not null)
        {
            double inv = 1.0 / matRotXY[8];
            double[] invProjZ = [inv, 0, inv * matRotXY[2], 0, inv, inv * matRotXY[5], 0, 0, 1];
            double[] rt = [matRotXY[0], matRotXY[3], matRotXY[6], matRotXY[1], matRotXY[4], matRotXY[7], matRotXY[2], matRotXY[5], matRotXY[8]];
            Mul(rt, invProjZ).CopyTo(invMatTilt, 0);
        }
    }
}
