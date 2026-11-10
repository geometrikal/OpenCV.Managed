// This implementation is ported from the OpenCV iterative PnP solver.
//
// Original sources:
//   modules/geometry/src/calibration_base.cpp (findExtrinsicCameraParams2)
//   modules/geometry/src/solvepnp.cpp (solvePnP, solvePnPGeneric with SOLVEPNP_ITERATIVE)
//   modules/geometry/src/fundam.cpp (findHomography with method 0: HomographyEstimatorCallback::runKernel
//     and the Levenberg-Marquardt refinement)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <c>findExtrinsicCameraParams2</c>: initial pose from a homography (planar object) or a DLT
/// (non-planar object), then Levenberg-Marquardt refinement of the reprojection error.
/// </summary>
internal static class CameraPnP
{
    private const double DblEpsilon = 2.220446049250313e-16;
    private const double FltEpsilon = 1.1920928955078125e-7;
    private const int MaxIter = 20;

    public static void FindExtrinsic(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        double[] param,
        bool useExtrinsicGuess)
    {
        int count = objectPoints.Length;
        if (count != imagePoints.Length)
        {
            throw new ArgumentException("The object and image point counts must be equal.", nameof(imagePoints));
        }

        if (!(count >= 4 || (count == 3 && useExtrinsicGuess)))
        {
            throw new ArgumentException("At least 4 points (3 with an extrinsic guess) are necessary.", nameof(objectPoints));
        }

        double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        Point3d[] m3 = objectPoints.ToArray();
        Point2d[] m2 = imagePoints.ToArray();
        double[] dist = distCoeffs.ToArray();
        CameraDistortion.Expand(dist, nameof(distCoeffs));

        Point2d[] mn = Undistortion.UndistortPoints(m2, cameraMatrix, dist, null, new Mat<double>(CameraDistortion.Identity(), 3, 3));

        if (!useExtrinsicGuess)
        {
            InitialPose(m3, mn, param);
        }

        CameraLevMarqCallback callback = ReprojectionCallback(m3, m2, a, dist);
        new CameraLevMarq(param, 2 * count, callback, MaxIter, true).Optimize();
    }

    /// <summary>
    /// Creates the <c>LevMarq</c> callback of <c>findExtrinsicCameraParams2</c> and <c>solvePnPRefineLM</c>:
    /// the errors are the projections minus the image points; the jacobian has the ∂p/∂r and ∂p/∂t columns.
    /// </summary>
    public static CameraLevMarqCallback ReprojectionCallback(Point3d[] m3, Point2d[] m2, double[] a, double[] dist)
    {
        int count = m3.Length;
        return new CameraLevMarqCallback((x, err, jac) =>
        {
            var rv = new Point3d(x[0], x[1], x[2]);
            var tv = new Point3d(x[3], x[4], x[5]);
            Point2d[] proj = ArrayPool<Point2d>.Shared.Rent(count);
            PooledMat<double>? full = jac is null ? null : PooledMat<double>.Rent(10 + dist.Length, 2 * count, 1);
            try
            {
                full?.Mat.Clear();
                var dRdr = new double[27];
                double[] r = CameraRodrigues.VectorToMatrix(rv.X, rv.Y, rv.Z, dRdr);
                CameraProjection.Project(m3, r, dRdr, [tv.X, tv.Y, tv.Z], a, dist, 0, proj.AsSpan(0, count), full?.Mat);
                Span<double> e = err.GetSpan();
                for (int i = 0; i < count; i++)
                {
                    e[2 * i] = proj[i].X - m2[i].X;
                    e[2 * i + 1] = proj[i].Y - m2[i].Y;
                }

                if (jac is not null)
                {
                    for (int row = 0; row < 2 * count; row++)
                    {
                        full!.Value.Mat.GetReadOnlyRowSpan(row)[..6].CopyTo(jac.GetRowSpan(row));
                    }
                }
            }
            finally
            {
                ArrayPool<Point2d>.Shared.Return(proj);
                full?.Dispose();
            }

            return true;
        });

    }

    private static void InitialPose(Point3d[] m3, Point2d[] mn, double[] param)
    {
        int count = m3.Length;
        var mat = new Mat<double>(3, count);
        for (int i = 0; i < count; i++)
        {
            Span<double> row = mat.GetRowSpan(i);
            row[0] = m3[i].X;
            row[1] = m3[i].Y;
            row[2] = m3[i].Z;
        }

        Scalar mean = Reductions.Mean(new Mat<double>(mat.ToArray(), 1, count, 3));
        double[] mc = [mean[0], mean[1], mean[2]];
        var mm = new Mat<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double s = 0;
                for (int k = 0; k < count; k++)
                {
                    s += (mat.At(k, i) - mc[i]) * (mat.At(k, j) - mc[j]);
                }

                mm.At(i, j) = s;
            }
        }

        LinearAlgebra.SVDecomp(mm, out Mat<double> w, out _, out Mat<double> vtm);
        double[] wv = w.ToArray();
        double[] v = CameraDistortion.Read3x3(vtm, "v");
        double[] rm;
        double[] t = new double[3];

        if (wv[2] / wv[1] < 1e-3)
        {
            double[] rt = v[2] * v[2] + v[5] * v[5] < 1e-10 ? CameraDistortion.Identity() : (double[])v.Clone();
            if (LinearAlgebra.Determinant(CameraDistortion.ToMat(rt)) < 0)
            {
                for (int i = 0; i < 9; i++)
                {
                    rt[i] *= -1.0;
                }
            }

            Mat<double> tt = LinearAlgebra.Gemm(CameraDistortion.ToMat(rt), new Mat<double>(mc, 3, 1), -1, null, 0, GemmFlags.BTranspose);
            double[] tp = tt.ToArray();
            var mxy = new Point2f[count];
            var mnf = new Point2f[count];
            for (int i = 0; i < count; i++)
            {
                double x = rt[0] * m3[i].X + rt[1] * m3[i].Y + rt[2] * m3[i].Z + tp[0];
                double y = rt[3] * m3[i].X + rt[4] * m3[i].Y + rt[5] * m3[i].Z + tp[1];
                mxy[i] = new Point2f((float)x, (float)y);
                mnf[i] = new Point2f((float)mn[i].X, (float)mn[i].Y);
            }

            double[]? h = FindHomography(mxy, mnf);
            if (h is not null && h.All(e => e >= -double.MaxValue && e <= double.MaxValue))
            {
                double h1Norm = Math.Sqrt(h[0] * h[0] + h[3] * h[3] + h[6] * h[6]);
                double h2Norm = Math.Sqrt(h[1] * h[1] + h[4] * h[4] + h[7] * h[7]);
                double s1 = 1.0 / MacroMax(h1Norm, DblEpsilon);
                double s2 = 1.0 / MacroMax(h2Norm, DblEpsilon);
                double s3 = 2.0 / MacroMax(h1Norm + h2Norm, DblEpsilon);
                for (int r = 0; r < 3; r++)
                {
                    h[r * 3] *= s1;
                    h[r * 3 + 1] *= s2;
                    t[r] = h[r * 3 + 2] * s3;
                }

                h[2] = h[3] * h[7] - h[6] * h[4];
                h[5] = h[6] * h[1] - h[0] * h[7];
                h[8] = h[0] * h[4] - h[3] * h[1];

                (double rx, double ry, double rz) = CameraRodrigues.MatrixToVector(h, null);
                double[] hr = CameraRodrigues.VectorToMatrix(rx, ry, rz, null);
                Mat<double> ht = LinearAlgebra.Gemm(CameraDistortion.ToMat(hr), new Mat<double>(tp, 1, 3), 1, null, 0);
                double[] htv = ht.ToArray();
                for (int i = 0; i < 3; i++)
                {
                    t[i] += htv[i];
                }

                rm = CameraDistortion.Read3x3(LinearAlgebra.Gemm(CameraDistortion.ToMat(hr), CameraDistortion.ToMat(rt), 1, null, 0), "r");
            }
            else
            {
                rm = CameraDistortion.Identity();
                t = new double[3];
            }
        }
        else
        {
            if (count < 6)
            {
                throw new ArgumentException("The DLT algorithm needs at least 6 points for a non-planar object.", nameof(m3));
            }

            var l = new Mat<double>(12, 2 * count);
            for (int i = 0; i < count; i++)
            {
                Span<double> l0 = l.GetRowSpan(2 * i);
                Span<double> l1 = l.GetRowSpan(2 * i + 1);
                double x = -mn[i].X, y = -mn[i].Y;
                l0[0] = l1[4] = m3[i].X;
                l0[1] = l1[5] = m3[i].Y;
                l0[2] = l1[6] = m3[i].Z;
                l0[3] = l1[7] = 1.0;
                l0[4] = l0[5] = l0[6] = l0[7] = 0.0;
                l1[0] = l1[1] = l1[2] = l1[3] = 0.0;
                l0[8] = x * m3[i].X;
                l0[9] = x * m3[i].Y;
                l0[10] = x * m3[i].Z;
                l0[11] = x;
                l1[8] = y * m3[i].X;
                l1[9] = y * m3[i].Y;
                l1[10] = y * m3[i].Z;
                l1[11] = y;
            }

            Mat<double> ll = LinearAlgebra.MulTransposed(l, true);
            LinearAlgebra.SVDecomp(ll, out _, out _, out Mat<double> lv);
            var rrt = new double[12];
            lv.GetReadOnlyRowSpan(11).CopyTo(rrt);
            double[] rr = [rrt[0], rrt[1], rrt[2], rrt[4], rrt[5], rrt[6], rrt[8], rrt[9], rrt[10]];
            if (LinearAlgebra.Determinant(CameraDistortion.ToMat(rr)) < 0)
            {
                for (int i = 0; i < 12; i++)
                {
                    rrt[i] *= -1.0;
                }

                rr = [rrt[0], rrt[1], rrt[2], rrt[4], rrt[5], rrt[6], rrt[8], rrt[9], rrt[10]];
            }

            double sc = Math.Sqrt(NormL2Sqr(rrt.AsSpan(0, 3)) + NormL2Sqr(rrt.AsSpan(4, 3)) + NormL2Sqr(rrt.AsSpan(8, 3)));
            if (!(Math.Abs(sc) > DblEpsilon))
            {
                throw new InvalidOperationException("The DLT pose is degenerate.");
            }

            LinearAlgebra.SVDecomp(CameraDistortion.ToMat(rr), out _, out Mat<double> u, out Mat<double> vt);
            rm = CameraDistortion.Read3x3(LinearAlgebra.Gemm(u, vt, 1, null, 0), "r");
            double scale = Math.Sqrt(NormL2Sqr(rm)) / sc;
            t = [rrt[3] * scale, rrt[7] * scale, rrt[11] * scale];
        }

        (double vx, double vy, double vz) = CameraRodrigues.MatrixToVector(rm, null);
        param[0] = vx;
        param[1] = vy;
        param[2] = vz;
        param[3] = t[0];
        param[4] = t[1];
        param[5] = t[2];
    }

    /// <summary>
    /// <c>findHomography</c> with method 0: normalized DLT from the eigenvector of the smallest
    /// eigenvalue, then 10 Levenberg-Marquardt iterations when there are more than 4 points.
    /// Returns null when the points are degenerate.
    /// </summary>
    private static double[]? FindHomography(Point2f[] src, Point2f[] dst)
    {
        int count = src.Length;
        double cmx = 0, cmy = 0, cMx = 0, cMy = 0;
        for (int i = 0; i < count; i++)
        {
            cmx += dst[i].X;
            cmy += dst[i].Y;
            cMx += src[i].X;
            cMy += src[i].Y;
        }

        cmx /= count;
        cmy /= count;
        cMx /= count;
        cMy /= count;

        double smx = 0, smy = 0, sMx = 0, sMy = 0;
        for (int i = 0; i < count; i++)
        {
            smx += Math.Abs(dst[i].X - cmx);
            smy += Math.Abs(dst[i].Y - cmy);
            sMx += Math.Abs(src[i].X - cMx);
            sMy += Math.Abs(src[i].Y - cMy);
        }

        if (Math.Abs(smx) < DblEpsilon || Math.Abs(smy) < DblEpsilon || Math.Abs(sMx) < DblEpsilon || Math.Abs(sMy) < DblEpsilon)
        {
            return null;
        }

        smx = count / smx;
        smy = count / smy;
        sMx = count / sMx;
        sMy = count / sMy;

        double[] invHnorm = [1.0 / smx, 0, cmx, 0, 1.0 / smy, cmy, 0, 0, 1];
        double[] hnorm2 = [sMx, 0, -cMx * sMx, 0, sMy, -cMy * sMy, 0, 0, 1];
        var ltl = new Mat<double>(9, 9);
        Span<double> lx = stackalloc double[9];
        Span<double> ly = stackalloc double[9];
        for (int i = 0; i < count; i++)
        {
            double x = (dst[i].X - cmx) * smx, y = (dst[i].Y - cmy) * smy;
            double bx = (src[i].X - cMx) * sMx, by = (src[i].Y - cMy) * sMy;
            lx[0] = bx; lx[1] = by; lx[2] = 1; lx[3] = 0; lx[4] = 0; lx[5] = 0; lx[6] = -x * bx; lx[7] = -x * by; lx[8] = -x;
            ly[0] = 0; ly[1] = 0; ly[2] = 0; ly[3] = bx; ly[4] = by; ly[5] = 1; ly[6] = -y * bx; ly[7] = -y * by; ly[8] = -y;
            for (int j = 0; j < 9; j++)
            {
                Span<double> row = ltl.GetRowSpan(j);
                for (int k = j; k < 9; k++)
                {
                    row[k] += lx[j] * lx[k] + ly[j] * ly[k];
                }
            }
        }

        LinearAlgebra.CompleteSymm(ltl);
        LinearAlgebra.Eigen(ltl, out _, out Mat<double> v);
        var h0 = new double[9];
        v.GetReadOnlyRowSpan(8).CopyTo(h0);
        Mat<double> htemp = LinearAlgebra.Gemm(CameraDistortion.ToMat(invHnorm), CameraDistortion.ToMat(h0), 1, null, 0);
        double[] h = LinearAlgebra.Gemm(htemp, CameraDistortion.ToMat(hnorm2), 1, null, 0).ToArray();
        Scale(h);

        if (count > 4)
        {
            var callback = new CameraLevMarqCallback((p, err, jac) =>
            {
                Span<double> e = err.GetSpan();
                for (int i = 0; i < count; i++)
                {
                    double mx = src[i].X, my = src[i].Y;
                    double ww = p[6] * mx + p[7] * my + p[8];
                    ww = Math.Abs(ww) > DblEpsilon ? 1.0 / ww : 0;
                    double xi = (p[0] * mx + p[1] * my + p[2]) * ww;
                    double yi = (p[3] * mx + p[4] * my + p[5]) * ww;
                    e[i * 2] = xi - dst[i].X;
                    e[i * 2 + 1] = yi - dst[i].Y;
                    if (jac is not null)
                    {
                        Span<double> j0 = jac.GetRowSpan(2 * i);
                        Span<double> j1 = jac.GetRowSpan(2 * i + 1);
                        j0.Clear();
                        j1.Clear();
                        j0[0] = mx * ww; j0[1] = my * ww; j0[2] = ww;
                        j0[6] = -mx * ww * xi; j0[7] = -my * ww * xi; j0[8] = -ww * xi;
                        j1[3] = mx * ww; j1[4] = my * ww; j1[5] = ww;
                        j1[6] = -mx * ww * yi; j1[7] = -my * ww * yi; j1[8] = -ww * yi;
                    }
                }

                return true;
            });
            new CameraLevMarq(h, 2 * count, callback, 10, true).Optimize();
            Scale(h);
        }

        return h;
    }

    /// <summary><c>convertTo(H, type, scaleFor(H(2, 2)))</c>.</summary>
    private static void Scale(double[] h)
    {
        double s = Math.Abs(h[8]) > FltEpsilon ? 1.0 / h[8] : 1.0;
        for (int i = 0; i < 9; i++)
        {
            h[i] *= s;
        }
    }

    private static double MacroMax(double a, double b) => a < b ? b : a;

    /// <summary>
    /// <c>NormL2_SIMD&lt;double, double&gt;</c> with two lanes: blocks of 8 in four lane
    /// accumulators, then the remaining squares in order.
    /// </summary>
    private static double NormL2Sqr(ReadOnlySpan<double> v)
    {
        int j = 0;
        double s = 0;
        double r00a = 0, r00b = 0, r01a = 0, r01b = 0, r10a = 0, r10b = 0, r11a = 0, r11b = 0;
        for (; j <= v.Length - 8; j += 8)
        {
            r00a = v[j] * v[j] + r00a;
            r00b = v[j + 1] * v[j + 1] + r00b;
            r01a = v[j + 2] * v[j + 2] + r01a;
            r01b = v[j + 3] * v[j + 3] + r01b;
            r10a = v[j + 4] * v[j + 4] + r10a;
            r10b = v[j + 5] * v[j + 5] + r10b;
            r11a = v[j + 6] * v[j + 6] + r11a;
            r11b = v[j + 7] * v[j + 7] + r11b;
        }

        s += (((r00a + r01a) + r10a) + r11a) + (((r00b + r01b) + r10b) + r11b);
        for (; j < v.Length; j++)
        {
            s += v[j] * v[j];
        }

        return s;
    }
}
