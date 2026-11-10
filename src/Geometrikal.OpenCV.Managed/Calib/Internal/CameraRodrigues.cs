// This implementation is ported from the OpenCV Rodrigues transform.
//
// Original source:
//   modules/geometry/src/calibration_base.cpp (cv::Rodrigues)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/calibration_base.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Rotation vector to matrix and matrix to vector conversions in <c>double</c>.</summary>
internal static class CameraRodrigues
{
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>
    /// Rotation vector to rotation matrix. <paramref name="jacobian"/> (3 × 9, row-major), when
    /// not null, receives <c>∂R/∂r</c>.
    /// </summary>
    public static double[] VectorToMatrix(double rx, double ry, double rz, double[]? jacobian)
    {
        double theta = Math.Sqrt(rx * rx + ry * ry + rz * rz);
        if (theta < DblEpsilon)
        {
            if (jacobian is not null)
            {
                Array.Clear(jacobian);
                jacobian[5] = jacobian[15] = jacobian[19] = -1;
                jacobian[7] = jacobian[11] = jacobian[21] = 1;
            }

            return CameraDistortion.Identity();
        }

        double c = Math.Cos(theta);
        double s = Math.Sin(theta);
        double c1 = 1.0 - c;
        double itheta = theta != 0 ? 1.0 / theta : 0.0;
        rx *= itheta;
        ry *= itheta;
        rz *= itheta;

        double[] rrt = [rx * rx, rx * ry, rx * rz, rx * ry, ry * ry, ry * rz, rx * rz, ry * rz, rz * rz];
        double[] rX = [0, -rz, ry, rz, 0, -rx, -ry, rx, 0];
        double[] eye = CameraDistortion.Identity();
        var r = new double[9];
        for (int k = 0; k < 9; k++)
        {
            r[k] = c * eye[k] + c1 * rrt[k] + s * rX[k];
        }

        if (jacobian is not null)
        {
            double[] drrt =
            [
                rx + rx, ry, rz, ry, 0, 0, rz, 0, 0,
                0, rx, 0, rx, ry + ry, rz, 0, rz, 0,
                0, 0, rx, 0, 0, ry, rx, ry, rz + rz,
            ];
            double[] dRx =
            [
                0, 0, 0, 0, 0, -1, 0, 1, 0,
                0, 0, 1, 0, 0, 0, -1, 0, 0,
                0, -1, 0, 1, 0, 0, 0, 0, 0,
            ];
            for (int i = 0; i < 3; i++)
            {
                double ri = i == 0 ? rx : i == 1 ? ry : rz;
                double a0 = -s * ri, a1 = (s - 2 * c1 * itheta) * ri, a2 = c1 * itheta;
                double a3 = (c - s * itheta) * ri, a4 = s * itheta;
                for (int k = 0; k < 9; k++)
                {
                    jacobian[i * 9 + k] = a0 * eye[k] + a1 * rrt[k] + a2 * drrt[i * 9 + k] + a3 * rX[k] + a4 * dRx[i * 9 + k];
                }
            }
        }

        return r;
    }

    /// <summary>
    /// Rotation matrix to rotation vector. <paramref name="jacobian"/> (3 × 9, row-major, before the
    /// final transpose to 9 × 3), when not null, receives <c>∂r/∂R</c>.
    /// </summary>
    public static (double X, double Y, double Z) MatrixToVector(double[] src, double[]? jacobian)
    {
        for (int i = 0; i < 9; i++)
        {
            double e = src[i];
            if (!(e >= -100 && e < 100))
            {
                if (jacobian is not null)
                {
                    Array.Clear(jacobian);
                }

                return (0, 0, 0);
            }
        }

        var svd = new Svd<double>(CameraDistortion.ToMat(src));
        double[] u = CameraDistortion.Read3x3(svd.U, "u");
        double[] vt = CameraDistortion.Read3x3(svd.Vt, "vt");
        double[] rm = CameraDistortion.Mul(u, vt);

        double rx = rm[7] - rm[5], ry = rm[2] - rm[6], rz = rm[3] - rm[1];
        double s = Math.Sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
        double c = (rm[0] + rm[4] + rm[8] - 1) * 0.5;
        c = c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c;
        double theta = Math.Acos(c);

        if (s < 1e-5)
        {
            if (c > 0)
            {
                rx = ry = rz = 0;
            }
            else
            {
                double t = (rm[0] + 1) * 0.5;
                rx = Math.Sqrt(t < 0.0 ? 0.0 : t);
                t = (rm[4] + 1) * 0.5;
                ry = Math.Sqrt(t < 0.0 ? 0.0 : t) * (rm[1] < 0 ? -1.0 : 1.0);
                t = (rm[8] + 1) * 0.5;
                rz = Math.Sqrt(t < 0.0 ? 0.0 : t) * (rm[2] < 0 ? -1.0 : 1.0);
                if (Math.Abs(rx) < Math.Abs(ry) && Math.Abs(rx) < Math.Abs(rz) && (rm[5] > 0) != (ry * rz > 0))
                {
                    rz = -rz;
                }

                theta /= Math.Sqrt(rx * rx + ry * ry + rz * rz);
                rx *= theta;
                ry *= theta;
                rz *= theta;
            }

            if (jacobian is not null)
            {
                Array.Clear(jacobian);
                if (c > 0)
                {
                    jacobian[5] = jacobian[15] = jacobian[19] = -0.5;
                    jacobian[7] = jacobian[11] = jacobian[21] = 0.5;
                }
            }
        }
        else
        {
            double vth = 1 / (2 * s);
            if (jacobian is not null)
            {
                double dthetaDtr = -1.0 / s;
                double dvthDtheta = -vth * c / s;
                double d1 = 0.5 * dvthDtheta * dthetaDtr;
                double d2 = 0.5 * dthetaDtr;
                double[] dvardR =
                [
                    0, 0, 0, 0, 0, 1, 0, -1, 0,
                    0, 0, -1, 0, 0, 0, 1, 0, 0,
                    0, 1, 0, -1, 0, 0, 0, 0, 0,
                    d1, 0, 0, 0, d1, 0, 0, 0, d1,
                    d2, 0, 0, 0, d2, 0, 0, 0, d2,
                ];
                double[] dvar2dvar =
                [
                    vth, 0, 0, rx, 0,
                    0, vth, 0, ry, 0,
                    0, 0, vth, rz, 0,
                    0, 0, 0, 0, 1,
                ];
                double[] domegadvar2 =
                [
                    theta, 0, 0, rx * vth,
                    0, theta, 0, ry * vth,
                    0, 0, theta, rz * vth,
                ];
                Mat<double> t0 = LinearAlgebra.Gemm(new Mat<double>(domegadvar2, 4, 3), new Mat<double>(dvar2dvar, 5, 4), 1, null, 0);
                Mat<double> j = LinearAlgebra.Gemm(t0, new Mat<double>(dvardR, 9, 5), 1, null, 0);
                j.GetReadOnlySpan().CopyTo(jacobian);
                Swap(jacobian, 1, 3);
                Swap(jacobian, 2, 6);
                Swap(jacobian, 5, 7);
                Swap(jacobian, 10, 12);
                Swap(jacobian, 11, 15);
                Swap(jacobian, 14, 16);
                Swap(jacobian, 19, 21);
                Swap(jacobian, 20, 24);
                Swap(jacobian, 23, 25);
            }

            vth *= theta;
            rx *= vth;
            ry *= vth;
            rz *= vth;
        }

        return (rx, ry, rz);
    }

    private static void Swap(double[] a, int i, int j) => (a[i], a[j]) = (a[j], a[i]);
}
