// This implementation is ported from the OpenCV 5.0 Rodrigues conversion (without the Jacobian).
//
// Original source:
//   modules/geometry/src/calibration_base.cpp (cv::Rodrigues)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>OpenCV <c>Rodrigues</c> between a rotation vector and a 3 × 3 rotation matrix.</summary>
internal static class Rodrigues
{
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>Rotation vector to a row-major rotation matrix, in <c>double</c>.</summary>
    public static void ToMatrix(double rx, double ry, double rz, Span<double> dst)
    {
        double theta = Math.Sqrt((rx * rx) + (ry * ry) + (rz * rz));
        if (theta < DblEpsilon)
        {
            dst.Clear();
            dst[0] = dst[4] = dst[8] = 1;
            return;
        }

        double c = Math.Cos(theta);
        double s = Math.Sin(theta);
        double c1 = 1.0 - c;
        double itheta = theta != 0 ? 1.0 / theta : 0.0;
        double x = rx * itheta, y = ry * itheta, z = rz * itheta;

        Span<double> rrt = stackalloc double[9] { x * x, x * y, x * z, x * y, y * y, y * z, x * z, y * z, z * z };
        Span<double> rx_ = stackalloc double[9] { 0, -z, y, z, 0, -x, -y, x, 0 };
        for (int k = 0; k < 9; k++)
        {
            double eye = (k == 0 || k == 4 || k == 8) ? 1.0 : 0.0;
            dst[k] = ((c * eye) + (c1 * rrt[k])) + (s * rx_[k]);
        }
    }

    /// <summary>Rotation matrix (row-major, already converted to <c>double</c>) to a rotation vector.</summary>
    public static void ToVector(ReadOnlySpan<double> src, out double rx, out double ry, out double rz)
    {
        for (int k = 0; k < 9; k++)
        {
            double v = src[k];
            if (double.IsNaN(v) || v < -100 || v >= 100)
            {
                rx = ry = rz = 0;
                return;
            }
        }

        var m = new Mat<double>(src.ToArray(), 3, 3);
        LinearAlgebra.SVDecomp(m, out _, out Mat<double> u, out Mat<double> vt);
        Span<double> r = stackalloc double[9];
        ReadOnlySpan<double> us = u.GetReadOnlySpan(), vs = vt.GetReadOnlySpan();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double acc = 0;
                for (int k = 0; k < 3; k++)
                {
                    acc += us[(i * 3) + k] * vs[(k * 3) + j];
                }

                r[(i * 3) + j] = acc;
            }
        }

        double px = r[7] - r[5], py = r[2] - r[6], pz = r[3] - r[1];
        double s = Math.Sqrt(((px * px) + (py * py) + (pz * pz)) * 0.25);
        double c = (r[0] + r[4] + r[8] - 1) * 0.5;
        c = c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c;
        double theta = Math.Acos(c);

        if (s < 1e-5)
        {
            if (c > 0)
            {
                px = py = pz = 0;
            }
            else
            {
                double t = (r[0] + 1) * 0.5;
                px = Math.Sqrt(Math.Max(t, 0.0));
                t = (r[4] + 1) * 0.5;
                py = Math.Sqrt(Math.Max(t, 0.0)) * (r[1] < 0 ? -1.0 : 1.0);
                t = (r[8] + 1) * 0.5;
                pz = Math.Sqrt(Math.Max(t, 0.0)) * (r[2] < 0 ? -1.0 : 1.0);
                if (Math.Abs(px) < Math.Abs(py) && Math.Abs(px) < Math.Abs(pz) && (r[5] > 0) != (py * pz > 0))
                {
                    pz = -pz;
                }

                theta /= Math.Sqrt((px * px) + (py * py) + (pz * pz));
                px *= theta;
                py *= theta;
                pz *= theta;
            }
        }
        else
        {
            double vth = 1 / (2 * s);
            vth *= theta;
            px *= vth;
            py *= vth;
            pz *= vth;
        }

        rx = px;
        ry = py;
        rz = pz;
    }
}
