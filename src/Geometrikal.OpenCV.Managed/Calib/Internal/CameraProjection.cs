// This implementation is ported from the OpenCV projectPoints.
//
// Original source:
//   modules/geometry/src/calibration_base.cpp (cv::projectPoints)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/calibration_base.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Pinhole projection with distortion and the optional jacobian of <c>projectPoints</c>, in the
/// OpenCV <c>double</c> operation order.
/// </summary>
internal static class CameraProjection
{
    private const float FltEpsilon = 1.1920929e-7f;

    /// <summary>
    /// Projects <paramref name="objectPoints"/>. <paramref name="jacobian"/>, when not null, has
    /// 2·N rows and <c>10 + ktotal</c> columns: ∂p/∂r, ∂p/∂t, ∂p/∂f, ∂p/∂c, ∂p/∂k.
    /// </summary>
    public static void Project(
        ReadOnlySpan<Point3d> objectPoints,
        double[] r,
        double[] dRdr,
        double[] t,
        double[] a,
        ReadOnlySpan<double> distCoeffs,
        double aspectRatio,
        Span<Point2d> imagePoints,
        Mat<double>? jacobian)
    {
        int ktotal = distCoeffs.Length;
        var k = new double[14];
        distCoeffs.CopyTo(k);
        double[] matTilt = CameraDistortion.Identity();
        double[] dTiltdTauX = [0, 0, 0, 0, 0, 0, 0, -1, 0];
        double[] dTiltdTauY = [0, 0, 0, 0, 0, 0, 1, 0, 0];
        if (k[12] != 0 || k[13] != 0)
        {
            CameraDistortion.TiltProjection(k[12], k[13], matTilt, dTiltdTauX, dTiltdTauY);
        }

        bool fixedAspectRatio = aspectRatio > FltEpsilon;
        double fx = a[0], fy = a[4], cx = a[2], cy = a[5];
        if (fixedAspectRatio)
        {
            fx = fy * aspectRatio;
        }

        Span<double> dMatTilt = stackalloc double[4];
        Span<double> dxdt = stackalloc double[3];
        Span<double> dydt = stackalloc double[3];
        Span<double> dx0dr = stackalloc double[3];
        Span<double> dy0dr = stackalloc double[3];
        Span<double> dz0dr = stackalloc double[3];

        for (int i = 0; i < objectPoints.Length; i++)
        {
            double X = objectPoints[i].X, Y = objectPoints[i].Y, Z = objectPoints[i].Z;
            double x = r[0] * X + r[1] * Y + r[2] * Z + t[0];
            double y = r[3] * X + r[4] * Y + r[5] * Z + t[1];
            double z = r[6] * X + r[7] * Y + r[8] * Z + t[2];

            z = z != 0 ? 1.0 / z : 1;
            x *= z;
            y *= z;

            double r2 = x * x + y * y;
            double r4 = r2 * r2;
            double r6 = r4 * r2;
            double a1 = 2 * x * y;
            double a2 = r2 + 2 * x * x;
            double a3 = r2 + 2 * y * y;
            double cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
            double icdist2 = 1.0 / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
            double xd0 = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
            double yd0 = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;

            double vt0 = CameraDistortion.MulRow(matTilt, 0, xd0, yd0);
            double vt1 = CameraDistortion.MulRow(matTilt, 1, xd0, yd0);
            double vt2 = CameraDistortion.MulRow(matTilt, 2, xd0, yd0);
            double invProj = vt2 != 0 ? 1.0 / vt2 : 1;
            double xd = invProj * vt0;
            double yd = invProj * vt1;

            imagePoints[i] = new Point2d(xd * fx + cx, yd * fy + cy);

            if (jacobian is null)
            {
                continue;
            }

            Span<double> row0 = jacobian.GetRowSpan(2 * i);
            Span<double> row1 = jacobian.GetRowSpan(2 * i + 1);

            row0[8] = 1;
            row0[9] = 0;
            row1[8] = 0;
            row1[9] = 1;

            if (fixedAspectRatio)
            {
                row0[6] = 0;
                row0[7] = xd * aspectRatio;
                row1[6] = 0;
                row1[7] = yd;
            }
            else
            {
                row0[6] = xd;
                row0[7] = 0;
                row1[6] = 0;
                row1[7] = yd;
            }

            double[] vecTilt = [vt0, vt1, vt2];
            for (int rr = 0; rr < 2; rr++)
            {
                for (int cc = 0; cc < 2; cc++)
                {
                    dMatTilt[rr * 2 + cc] = matTilt[rr * 3 + cc] * vecTilt[2] - matTilt[6 + cc] * vecTilt[rr];
                }
            }

            double invProjSquare = invProj * invProj;
            for (int e = 0; e < 4; e++)
            {
                dMatTilt[e] *= invProjSquare;
            }

            Span<double> dk0 = row0[10..];
            Span<double> dk1 = row1[10..];
            if (ktotal > 0)
            {
                SetDk(dMatTilt, dk0, dk1, 0, x * icdist2 * r2, y * icdist2 * r2, fx, fy);
                SetDk(dMatTilt, dk0, dk1, 1, x * icdist2 * r4, y * icdist2 * r4, fx, fy);
            }

            if (ktotal > 2)
            {
                SetDk(dMatTilt, dk0, dk1, 2, a1, a3, fx, fy);
                SetDk(dMatTilt, dk0, dk1, 3, a2, a1, fx, fy);
                if (ktotal > 4)
                {
                    SetDk(dMatTilt, dk0, dk1, 4, x * icdist2 * r6, y * icdist2 * r6, fx, fy);
                    if (ktotal > 5)
                    {
                        SetDk(dMatTilt, dk0, dk1, 5, x * cdist * -icdist2 * icdist2 * r2, y * cdist * -icdist2 * icdist2 * r2, fx, fy);
                        SetDk(dMatTilt, dk0, dk1, 6, x * cdist * -icdist2 * icdist2 * r4, y * cdist * -icdist2 * icdist2 * r4, fx, fy);
                        SetDk(dMatTilt, dk0, dk1, 7, x * cdist * -icdist2 * icdist2 * r6, y * cdist * -icdist2 * icdist2 * r6, fx, fy);
                        if (ktotal > 8)
                        {
                            SetDk(dMatTilt, dk0, dk1, 8, r2, 0, fx, fy);
                            SetDk(dMatTilt, dk0, dk1, 9, r4, 0, fx, fy);
                            SetDk(dMatTilt, dk0, dk1, 10, 0, r2, fx, fy);
                            SetDk(dMatTilt, dk0, dk1, 11, 0, r4, fx, fy);
                            if (ktotal > 12)
                            {
                                double d0 = CameraDistortion.MulRow(dTiltdTauX, 0, xd0, yd0);
                                double d1 = CameraDistortion.MulRow(dTiltdTauX, 1, xd0, yd0);
                                double d2 = CameraDistortion.MulRow(dTiltdTauX, 2, xd0, yd0);
                                dk0[12] = fx * invProjSquare * (d0 * vecTilt[2] - d2 * vecTilt[0]);
                                dk1[12] = fy * invProjSquare * (d1 * vecTilt[2] - d2 * vecTilt[1]);
                                d0 = CameraDistortion.MulRow(dTiltdTauY, 0, xd0, yd0);
                                d1 = CameraDistortion.MulRow(dTiltdTauY, 1, xd0, yd0);
                                d2 = CameraDistortion.MulRow(dTiltdTauY, 2, xd0, yd0);
                                dk0[13] = fx * invProjSquare * (d0 * vecTilt[2] - d2 * vecTilt[0]);
                                dk1[13] = fy * invProjSquare * (d1 * vecTilt[2] - d2 * vecTilt[1]);
                            }
                        }
                    }
                }
            }

            dxdt[0] = z;
            dxdt[1] = 0;
            dxdt[2] = -x * z;
            dydt[0] = 0;
            dydt[1] = z;
            dydt[2] = -y * z;
            for (int j = 0; j < 3; j++)
            {
                double dr2dt = 2 * x * dxdt[j] + 2 * y * dydt[j];
                double dcdistDt = k[0] * dr2dt + 2 * k[1] * r2 * dr2dt + 3 * k[4] * r4 * dr2dt;
                double dicdist2Dt = -icdist2 * icdist2 * (k[5] * dr2dt + 2 * k[6] * r2 * dr2dt + 3 * k[7] * r4 * dr2dt);
                double da1dt = 2 * (x * dydt[j] + y * dxdt[j]);
                double dmxdt = dxdt[j] * cdist * icdist2 + x * dcdistDt * icdist2 + x * cdist * dicdist2Dt +
                               k[2] * da1dt + k[3] * (dr2dt + 4 * x * dxdt[j]) + k[8] * dr2dt + 2 * r2 * k[9] * dr2dt;
                double dmydt = dydt[j] * cdist * icdist2 + y * dcdistDt * icdist2 + y * cdist * dicdist2Dt +
                               k[2] * (dr2dt + 4 * y * dydt[j]) + k[3] * da1dt + k[10] * dr2dt + 2 * r2 * k[11] * dr2dt;
                MulTilt(dMatTilt, dmxdt, dmydt, out double ex, out double ey);
                row0[3 + j] = fx * ex;
                row1[3 + j] = fy * ey;
            }

            dx0dr[0] = X * dRdr[0] + Y * dRdr[1] + Z * dRdr[2];
            dx0dr[1] = X * dRdr[9] + Y * dRdr[10] + Z * dRdr[11];
            dx0dr[2] = X * dRdr[18] + Y * dRdr[19] + Z * dRdr[20];
            dy0dr[0] = X * dRdr[3] + Y * dRdr[4] + Z * dRdr[5];
            dy0dr[1] = X * dRdr[12] + Y * dRdr[13] + Z * dRdr[14];
            dy0dr[2] = X * dRdr[21] + Y * dRdr[22] + Z * dRdr[23];
            dz0dr[0] = X * dRdr[6] + Y * dRdr[7] + Z * dRdr[8];
            dz0dr[1] = X * dRdr[15] + Y * dRdr[16] + Z * dRdr[17];
            dz0dr[2] = X * dRdr[24] + Y * dRdr[25] + Z * dRdr[26];
            for (int j = 0; j < 3; j++)
            {
                double dxdr = z * (dx0dr[j] - x * dz0dr[j]);
                double dydr = z * (dy0dr[j] - y * dz0dr[j]);
                double dr2dr = 2 * x * dxdr + 2 * y * dydr;
                double dcdistDr = (k[0] + 2 * k[1] * r2 + 3 * k[4] * r4) * dr2dr;
                double dicdist2Dr = -icdist2 * icdist2 * (k[5] + 2 * k[6] * r2 + 3 * k[7] * r4) * dr2dr;
                double da1dr = 2 * (x * dydr + y * dxdr);
                double dmxdr = dxdr * cdist * icdist2 + x * dcdistDr * icdist2 + x * cdist * dicdist2Dr +
                               k[2] * da1dr + k[3] * (dr2dr + 4 * x * dxdr) + (k[8] + 2 * r2 * k[9]) * dr2dr;
                double dmydr = dydr * cdist * icdist2 + y * dcdistDr * icdist2 + y * cdist * dicdist2Dr +
                               k[2] * (dr2dr + 4 * y * dydr) + k[3] * da1dr + (k[10] + 2 * r2 * k[11]) * dr2dr;
                MulTilt(dMatTilt, dmxdr, dmydr, out double ex, out double ey);
                row0[j] = fx * ex;
                row1[j] = fy * ey;
            }
        }
    }

    private static void SetDk(ReadOnlySpan<double> dMatTilt, Span<double> dk0, Span<double> dk1, int index, double vx, double vy, double fx, double fy)
    {
        MulTilt(dMatTilt, vx, vy, out double ex, out double ey);
        dk0[index] = fx * ex;
        dk1[index] = fy * ey;
    }

    /// <summary><c>Matx22d * Vec2d</c>: each element sums from 0.</summary>
    private static void MulTilt(ReadOnlySpan<double> m, double vx, double vy, out double ex, out double ey)
    {
        double s = 0;
        s += m[0] * vx;
        s += m[1] * vy;
        ex = s;
        s = 0;
        s += m[2] * vx;
        s += m[3] * vy;
        ey = s;
    }
}
