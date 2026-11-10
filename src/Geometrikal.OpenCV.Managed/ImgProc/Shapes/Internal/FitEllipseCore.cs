// This implementation is ported from OpenCV's ellipse fitting.
//
// Original source:
//   modules/geometry/src/shapedescr.cpp (getOfs, fitEllipseNoDirect, fitEllipse, fitEllipseAMS,
//   fitEllipseDirect)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/shapedescr.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// The three OpenCV ellipse fits: the general conic least-squares fit (<c>fitEllipse</c>), the
/// Approximate Mean Square fit (<c>fitEllipseAMS</c>) and the Direct least-squares fit
/// (<c>fitEllipseDirect</c>). The input is single-precision points (OpenCV converts integer
/// points to float).
/// </summary>
internal static class FitEllipseCore
{
    private const float FltEpsilon = 1.1920928955078125e-7f;
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>OpenCV <c>fitEllipse</c>: the Direct fit for five points, else the general fit.</summary>
    public static RotatedRect Fit(ReadOnlySpan<Point2f> points)
    {
        CheckCount(points.Length);
        return points.Length == 5 ? Direct(points) : NoDirect(points);
    }

    /// <summary>
    /// <c>getOfs</c>: a random offset from <c>theRNG()</c>. The y draw comes first, as the
    /// MSVC build of OpenCV evaluates the constructor arguments right to left.
    /// </summary>
    private static Point2f GetOfs(float eps)
    {
        ref CvRng rng = ref CvRng.TheRng;
        float y = rng.Uniform(-eps, eps);
        float x = rng.Uniform(-eps, eps);
        return new Point2f(x, y);
    }

    private static void CheckCount(int n)
    {
        if (n < 5)
        {
            throw new ArgumentException("There should be at least 5 points to fit the ellipse.", "points");
        }
    }

    /// <summary><c>fitEllipseNoDirect</c>.</summary>
    public static RotatedRect NoDirect(ReadOnlySpan<Point2f> points)
    {
        int n = points.Length;
        CheckCount(n);
        Point2f[] rentedPts = ArrayPool<Point2f>.Shared.Rent(n);
        double[] rented = ArrayPool<double>.Shared.Rent(n * 14);
        try
        {
            return NoDirect(points, rentedPts.AsSpan(0, n), rented.AsSpan(0, n * 14));
        }
        finally
        {
            ArrayPool<double>.Shared.Return(rented);
            ArrayPool<Point2f>.Shared.Return(rentedPts);
        }
    }

    private static RotatedRect NoDirect(ReadOnlySpan<Point2f> points, Span<Point2f> ptsf, Span<double> buffer)
    {
        int n = points.Length;
        const double minEps = 1e-8;
        float cx = 0, cy = 0;
        for (int i = 0; i < n; i++)
        {
            ptsf[i] = points[i];
            cx += points[i].X;
            cy += points[i].Y;
        }

        cx /= n;
        cy /= n;

        double s = 0;
        for (int i = 0; i < n; i++)
        {
            float px = ptsf[i].X - cx;
            float py = ptsf[i].Y - cy;
            s += MathF.Abs(px) + MathF.Abs(py);
        }

        double scale = 100.0 / (s > FltEpsilon ? s : FltEpsilon);

        Span<double> a = buffer[..(n * 5)];
        Span<double> b = buffer.Slice(n * 5, n);
        Span<double> ut = buffer.Slice(n * 6, n * 5);
        Span<double> w = stackalloc double[5];
        Span<double> vt = stackalloc double[25];
        Span<double> gfp = stackalloc double[5];
        Span<double> rp = stackalloc double[5];

        FillConicRows(ptsf, cx, cy, scale, a, b);
        FitMatrix.SvDecomp(a, n, 5, w, ut, vt);
        if (w[0] * FltEpsilon > w[4])
        {
            float eps = (float)(s / (n * 2) * 1e-3);
            for (int i = 0; i < n; i++)
            {
                Point2f o = GetOfs(eps);
                ptsf[i] = new Point2f(ptsf[i].X + o.X, ptsf[i].Y + o.Y);
            }

            FillConicRows(ptsf, cx, cy, scale, a, b);
            FitMatrix.SvDecomp(a, n, 5, w, ut, vt);
        }

        FitMatrix.SvBackSubst(n, 5, w, ut, vt, b, gfp);

        Span<double> a2 = stackalloc double[4];
        Span<double> b2 = stackalloc double[2];
        a2[0] = 2 * gfp[0];
        a2[1] = a2[2] = gfp[2];
        a2[3] = 2 * gfp[1];
        b2[0] = gfp[3];
        b2[1] = gfp[4];
        FitMatrix.SolveSvd(a2, 2, 2, b2, rp);

        Span<double> a3 = buffer.Slice(n * 11, n * 3);
        for (int i = 0; i < n; i++)
        {
            float qx = ptsf[i].X - cx;
            float qy = ptsf[i].Y - cy;
            double px = qx * scale;
            double py = qy * scale;
            b[i] = 1.0;
            a3[i * 3] = (px - rp[0]) * (px - rp[0]);
            a3[(i * 3) + 1] = (py - rp[1]) * (py - rp[1]);
            a3[(i * 3) + 2] = (px - rp[0]) * (py - rp[1]);
        }

        FitMatrix.SolveSvd(a3, n, 3, b, gfp[..3]);

        rp[4] = -0.5 * Math.Atan2(gfp[2], gfp[1] - gfp[0]);
        double t = Math.Abs(gfp[2]) > minEps ? gfp[2] / Math.Sin(-2.0 * rp[4]) : gfp[1] - gfp[0];
        rp[2] = Math.Abs(gfp[0] + gfp[1] - t);
        if (rp[2] > minEps)
        {
            rp[2] = Math.Sqrt(2.0 / rp[2]);
        }

        rp[3] = Math.Abs(gfp[0] + gfp[1] + t);
        if (rp[3] > minEps)
        {
            rp[3] = Math.Sqrt(2.0 / rp[3]);
        }

        float centerX = (float)(rp[0] / scale) + cx;
        float centerY = (float)(rp[1] / scale) + cy;
        float width = (float)(rp[2] * 2 / scale);
        float height = (float)(rp[3] * 2 / scale);
        float angle = 0;
        if (width > height)
        {
            (width, height) = (height, width);
            angle = (float)(90 + (rp[4] * 180 / Math.PI));
        }

        if (angle < -180)
        {
            angle += 360;
        }

        if (angle > 360)
        {
            angle -= 360;
        }

        return new RotatedRect(new Point2f(centerX, centerY), new Size2f(width, height), angle);
    }

    private static void FillConicRows(ReadOnlySpan<Point2f> ptsf, float cx, float cy, double scale, Span<double> a, Span<double> b)
    {
        for (int i = 0; i < ptsf.Length; i++)
        {
            float qx = ptsf[i].X - cx;
            float qy = ptsf[i].Y - cy;
            double px = qx * scale;
            double py = qy * scale;
            b[i] = 10000.0;
            a[i * 5] = -px * px;
            a[(i * 5) + 1] = -py * py;
            a[(i * 5) + 2] = -px * py;
            a[(i * 5) + 3] = px;
            a[(i * 5) + 4] = py;
        }
    }

    /// <summary>
    /// Fills the design matrix rows (x², xy, y², x, y, 1) of the AMS and Direct fits and returns
    /// <c>DM = AᵀA / n</c>. Each point draws one offset from <c>theRNG()</c>, also when
    /// <paramref name="eps"/> is 0, as OpenCV does.
    /// </summary>
    private static double[] Scatter(ReadOnlySpan<Point2f> points, double cx, double cy, bool floatCenter, double scale, float eps)
    {
        int n = points.Length;
        double[] rented = ArrayPool<double>.Shared.Rent(n * 6);
        Span<double> a = rented.AsSpan(0, n * 6);
        for (int i = 0; i < n; i++)
        {
            Point2f p = points[i];
            Point2f delta = GetOfs(eps);
            double px, py;
            if (floatCenter)
            {
                px = (p.X + delta.X - (float)cx) * scale;
                py = (p.Y + delta.Y - (float)cy) * scale;
            }
            else
            {
                px = ((p.X + delta.X) - cx) * scale;
                py = ((p.Y + delta.Y) - cy) * scale;
            }

            a[i * 6] = px * px;
            a[(i * 6) + 1] = px * py;
            a[(i * 6) + 2] = py * py;
            a[(i * 6) + 3] = px;
            a[(i * 6) + 4] = py;
            a[(i * 6) + 5] = 1.0;
        }

        var dm = new double[36];
        MatrixSolvers.MulTransposed(a, n, 6, dm);
        ArrayPool<double>.Shared.Return(rented);
        double inv = 1.0 / n;
        for (int i = 0; i < 36; i++)
        {
            dm[i] *= inv;
        }

        return dm;
    }

    /// <summary>OpenCV <c>fitEllipseAMS</c>.</summary>
    public static RotatedRect Ams(ReadOnlySpan<Point2f> points)
    {
        int n = points.Length;
        CheckCount(n);
        float eps = 0;

        float cx = 0, cy = 0;
        for (int i = 0; i < n; i++)
        {
            cx += points[i].X;
            cy += points[i].Y;
        }

        cx /= (float)n;
        cy /= (float)n;

        double s = 0;
        for (int i = 0; i < n; i++)
        {
            s += MathF.Abs(points[i].X - cx) + MathF.Abs(points[i].Y - cy);
        }

        double scale = 100.0 / (s > FltEpsilon ? s : (double)FltEpsilon);

        double[] dm = null!;
        var mm = new double[25];
        int iter;
        for (iter = 0; iter < 2; iter++)
        {
            dm = Scatter(points, cx, cy, true, scale, eps);
            double dnm = (dm[17] * (dm[5] + dm[17])) - (dm[11] * dm[11]);
            double ddm = 4.0 * (dm[5] + dm[17]) * ((dm[5] * dm[17]) - (dm[11] * dm[11]));
            double ddmm = 2.0 * (dm[5] + dm[17]) * ((dm[5] * dm[17]) - (dm[11] * dm[11]));

            mm[0] = ((((-dm[0] + dm[2] + (dm[5] * dm[5])) * (dm[11] * dm[11])) + ((((-2 * dm[1]) * dm[11]) + (dm[5] * (dm[0]
                    - (dm[5] * dm[5]) + (dm[11] * dm[11])))) * dm[17])) + ((dm[0] - (dm[5] * dm[5])) * (dm[17] * dm[17]))) / ddm;
            mm[1] = (((dm[11] * dm[11]) * (-dm[1] + dm[8] + (dm[5] * dm[11]))) + (((dm[1] * dm[5]) - (((dm[5] * dm[5]) + (2 * dm[7])) * dm[11])
                    + (dm[11] * dm[11] * dm[11])) * dm[17]) + ((dm[1] - (dm[5] * dm[11])) * (dm[17] * dm[17]))) / ddm;
            mm[2] = ((-2 * dm[8] * dm[11] * dm[17]) - (dm[5] * (dm[17] * dm[17]) * (dm[5] + dm[17])) + (dm[2] * dnm)
                    + ((dm[11] * dm[11]) * (dm[14] + (dm[17] * (dm[5] + dm[17]))))) / ddm;
            mm[3] = ((dm[11] * ((dm[11] * dm[15]) - (2 * dm[9] * dm[17]))) + (dm[3] * dnm)) / ddm;
            mm[4] = ((dm[11] * ((dm[11] * dm[16]) - (2 * dm[10] * dm[17]))) + (dm[4] * dnm)) / ddm;
            mm[5] = (-(dm[2] * dm[5] * dm[11]) + (((2 * dm[1] * dm[5]) - (dm[0] * dm[11])) * dm[17])) / ddmm;
            mm[6] = (-(dm[1] * dm[11] * dm[17]) + (dm[5] * (-(dm[8] * dm[11]) + (2 * dm[7] * dm[17])))) / ddmm;
            mm[7] = (-(dm[2] * dm[11] * dm[17]) + (dm[5] * (-(dm[11] * dm[14]) + (2 * dm[8] * dm[17])))) / ddmm;
            mm[8] = (-(dm[3] * dm[11] * dm[17]) + (dm[5] * (-(dm[11] * dm[15]) + (2 * dm[9] * dm[17])))) / ddmm;
            mm[9] = (-(dm[4] * dm[11] * dm[17]) + (dm[5] * (-(dm[11] * dm[16]) + (2 * dm[10] * dm[17])))) / ddmm;
            mm[10] = ((-2 * dm[1] * dm[5] * dm[11]) + ((dm[0] + (dm[5] * dm[5])) * (dm[11] * dm[11])) + (dm[5] * (-(dm[5] * dm[5])
                    + (dm[11] * dm[11])) * dm[17]) - ((dm[5] * dm[5]) * (dm[17] * dm[17])) + (dm[2] * (-(dm[11] * dm[11]) + (dm[5] * (dm[5] + dm[17]))))) / ddm;
            mm[11] = (((dm[5] * dm[5]) * (dm[8] - (dm[11] * dm[17]))) + ((dm[11] * dm[11]) * (dm[1] - dm[8] + (dm[11] * dm[17])))
                    + (dm[5] * ((dm[8] * dm[17]) + (dm[11] * ((-2 * dm[7]) + (dm[11] * dm[11]) - (dm[17] * dm[17])))))) / ddm;
            mm[12] = (((dm[5] * dm[5]) * (dm[14] - (dm[17] * dm[17]))) + ((dm[11] * dm[11]) * (dm[2] - dm[14] + (dm[17] * dm[17])))
                    + (dm[5] * ((-2 * dm[8] * dm[11]) + (dm[17] * ((dm[11] * dm[11]) + dm[14] - (dm[17] * dm[17])))))) / ddm;
            mm[13] = (((dm[11] * dm[11]) * (dm[3] - dm[15])) + ((dm[5] * dm[5]) * dm[15]) + (dm[5] * ((-2 * dm[9] * dm[11]) + (dm[15] * dm[17])))) / ddm;
            mm[14] = (((dm[11] * dm[11]) * (dm[4] - dm[16])) + ((dm[5] * dm[5]) * dm[16]) + (dm[5] * ((-2 * dm[10] * dm[11]) + (dm[16] * dm[17])))) / ddm;
            mm[15] = dm[3];
            mm[16] = dm[9];
            mm[17] = dm[15];
            mm[18] = dm[21];
            mm[19] = dm[22];
            mm[20] = dm[4];
            mm[21] = dm[10];
            mm[22] = dm[16];
            mm[23] = dm[22];
            mm[24] = dm[28];

            double npow = (double)n * (double)n;
            npow = npow * npow * (double)n;
            if (Math.Abs(FitMatrix.LuDeterminant(mm, 5)) > 1.0e-10 / npow)
            {
                break;
            }

            eps = (float)(s / (n * 2) * 1e-2);
        }

        if (iter >= 2)
        {
            return NoDirect(points);
        }

        Span<double> eval = stackalloc double[5];
        Span<double> ev = stackalloc double[25];
        FitMatrix.EigenNonSymmetric(mm, 5, eval, ev);

        int minpos = 0;
        double normMinpos = Math.Sqrt((ev[0] * ev[0]) + (ev[1] * ev[1]) + (ev[2] * ev[2]) + (ev[3] * ev[3]) + (ev[4] * ev[4]));
        double normEValMinpos = eval[0] * normMinpos;
        for (int i = 1; i < 5; i++)
        {
            int r = i * 5;
            double normi = Math.Sqrt((ev[r] * ev[r]) + (ev[r + 1] * ev[r + 1]) + (ev[r + 2] * ev[r + 2]) + (ev[r + 3] * ev[r + 3]) + (ev[r + 4] * ev[r + 4]));
            double normEVali = eval[i] * normi;
            if (normEVali < normEValMinpos)
            {
                minpos = i;
                normMinpos = normi;
                normEValMinpos = normEVali;
            }
        }

        Span<double> pv = stackalloc double[5];
        for (int k = 0; k < 5; k++)
        {
            pv[k] = ev[(minpos * 5) + k] / normMinpos;
        }

        Span<double> cf = stackalloc double[6];
        pv.CopyTo(cf);
        cf[5] = (-pv[0] * dm[5]) - (pv[1] * dm[11]) - (cf[2] * dm[17]);

        bool isEllipse = (cf[0] < 0 &&
                          cf[2] < (cf[1] * cf[1]) / (4.0 * cf[0]) &&
                          cf[5] > (-(cf[2] * (cf[3] * cf[3])) + (cf[1] * cf[3] * cf[4]) - (cf[0] * (cf[4] * cf[4]))) /
                                  ((cf[1] * cf[1]) - (4 * cf[0] * cf[2]))) ||
                         (cf[0] > 0 &&
                          cf[2] > (cf[1] * cf[1]) / (4.0 * cf[0]) &&
                          cf[5] < (-(cf[2] * (cf[3] * cf[3])) + (cf[1] * cf[3] * cf[4]) - (cf[0] * (cf[4] * cf[4]))) /
                                  ((cf[1] * cf[1]) - (4 * cf[0] * cf[2])));
        if (!isEllipse)
        {
            return Direct(points);
        }

        double u1 = (pv[2] * pv[3] * pv[3]) - (pv[1] * pv[3] * pv[4]) + (pv[0] * pv[4] * pv[4]) + (pv[1] * pv[1] * cf[5]);
        double u2 = pv[0] * pv[2] * cf[5];
        double l1 = Math.Sqrt((pv[1] * pv[1]) + ((pv[0] - pv[2]) * (pv[0] - pv[2])));
        double l2 = pv[0] + pv[2];
        double l3 = (pv[1] * pv[1]) - (4.0 * pv[0] * pv[2]);
        double p1 = (2.0 * pv[2] * pv[3]) - (pv[1] * pv[4]);
        double p2 = (2.0 * pv[0] * pv[4]) - (pv[1] * pv[3]);

        double x0 = (p1 / l3 / scale) + cx;
        double y0 = (p2 / l3 / scale) + cy;
        double ea = Math.Sqrt(2.0) * Math.Sqrt((u1 - (4.0 * u2)) / ((l1 - l2) * l3)) / scale;
        double eb = Math.Sqrt(2.0) * Math.Sqrt(-1.0 * ((u1 - (4.0 * u2)) / ((l1 + l2) * l3))) / scale;
        double theta = Theta(pv);

        float width = (float)(2.0 * ea);
        float height = (float)(2.0 * eb);
        float angle;
        if (width > height)
        {
            (width, height) = (height, width);
            angle = (float)(90 + (theta * 180 / Math.PI));
        }
        else
        {
            angle = (float)(theta * 180 / Math.PI % 180.0);
        }

        return new RotatedRect(new Point2f((float)x0, (float)y0), new Size2f(width, height), angle);
    }

    /// <summary>OpenCV <c>fitEllipseDirect</c>.</summary>
    public static RotatedRect Direct(ReadOnlySpan<Point2f> points)
    {
        int n = points.Length;
        CheckCount(n);
        float eps = 0;

        double cx = 0, cy = 0;
        for (int i = 0; i < n; i++)
        {
            cx += points[i].X;
            cy += points[i].Y;
        }

        cx /= n;
        cy /= n;

        double s = 0;
        for (int i = 0; i < n; i++)
        {
            s += Math.Abs(points[i].X - cx) + Math.Abs(points[i].Y - cy);
        }

        double scale = 100.0 / (s > FltEpsilon ? s : (double)FltEpsilon);

        double[] dm = null!;
        var tm = new double[9];
        var mm = new double[9];
        Span<double> eval = stackalloc double[3];
        Span<double> ev = stackalloc double[9];
        Span<double> cond = stackalloc double[3];
        double ts = 0;
        int sel = 0;
        int iter;
        for (iter = 0; iter < 2; iter++)
        {
            dm = Scatter(points, cx, cy, false, scale, eps);

            tm[0] = (dm[5] * dm[23] * dm[28]) - (dm[5] * dm[22] * dm[29]) - (dm[4] * dm[23] * dm[34]) +
                    (dm[3] * dm[29] * dm[34]) + (dm[4] * dm[22] * dm[35]) - (dm[3] * dm[28] * dm[35]);
            tm[1] = (dm[11] * dm[23] * dm[28]) - (dm[11] * dm[22] * dm[29]) - (dm[10] * dm[23] * dm[34]) +
                    (dm[9] * dm[29] * dm[34]) + (dm[10] * dm[22] * dm[35]) - (dm[9] * dm[28] * dm[35]);
            tm[2] = (dm[17] * dm[23] * dm[28]) - (dm[17] * dm[22] * dm[29]) - (dm[16] * dm[23] * dm[34]) +
                    (dm[15] * dm[29] * dm[34]) + (dm[16] * dm[22] * dm[35]) - (dm[15] * dm[28] * dm[35]);
            tm[3] = (dm[5] * dm[21] * dm[29]) - (dm[5] * dm[23] * dm[27]) + (dm[4] * dm[23] * dm[33]) -
                    (dm[3] * dm[29] * dm[33]) - (dm[4] * dm[21] * dm[35]) + (dm[3] * dm[27] * dm[35]);
            tm[4] = (dm[11] * dm[21] * dm[29]) - (dm[11] * dm[23] * dm[27]) + (dm[10] * dm[23] * dm[33]) -
                    (dm[9] * dm[29] * dm[33]) - (dm[10] * dm[21] * dm[35]) + (dm[9] * dm[27] * dm[35]);
            tm[5] = (dm[17] * dm[21] * dm[29]) - (dm[17] * dm[23] * dm[27]) + (dm[16] * dm[23] * dm[33]) -
                    (dm[15] * dm[29] * dm[33]) - (dm[16] * dm[21] * dm[35]) + (dm[15] * dm[27] * dm[35]);
            tm[6] = (dm[5] * dm[22] * dm[27]) - (dm[5] * dm[21] * dm[28]) - (dm[4] * dm[22] * dm[33]) +
                    (dm[3] * dm[28] * dm[33]) + (dm[4] * dm[21] * dm[34]) - (dm[3] * dm[27] * dm[34]);
            tm[7] = (dm[11] * dm[22] * dm[27]) - (dm[11] * dm[21] * dm[28]) - (dm[10] * dm[22] * dm[33]) +
                    (dm[9] * dm[28] * dm[33]) + (dm[10] * dm[21] * dm[34]) - (dm[9] * dm[27] * dm[34]);
            tm[8] = (dm[17] * dm[22] * dm[27]) - (dm[17] * dm[21] * dm[28]) - (dm[16] * dm[22] * dm[33]) +
                    (dm[15] * dm[28] * dm[33]) + (dm[16] * dm[21] * dm[34]) - (dm[15] * dm[27] * dm[34]);

            ts = -(dm[23] * dm[28] * dm[33]) + (dm[22] * dm[29] * dm[33]) + (dm[23] * dm[27] * dm[34]) -
                 (dm[21] * dm[29] * dm[34]) - (dm[22] * dm[27] * dm[35]) + (dm[21] * dm[28] * dm[35]);

            if (Math.Abs(ts) < DblEpsilon)
            {
                eps = (float)(s / (n * 2) * 1e-2);
                continue;
            }

            mm[0] = (dm[12] + (((dm[15] * tm[0]) + (dm[16] * tm[3]) + (dm[17] * tm[6])) / ts)) / 2.0;
            mm[1] = (dm[13] + (((dm[15] * tm[1]) + (dm[16] * tm[4]) + (dm[17] * tm[7])) / ts)) / 2.0;
            mm[2] = (dm[14] + (((dm[15] * tm[2]) + (dm[16] * tm[5]) + (dm[17] * tm[8])) / ts)) / 2.0;
            mm[3] = -dm[6] - (((dm[9] * tm[0]) + (dm[10] * tm[3]) + (dm[11] * tm[6])) / ts);
            mm[4] = -dm[7] - (((dm[9] * tm[1]) + (dm[10] * tm[4]) + (dm[11] * tm[7])) / ts);
            mm[5] = -dm[8] - (((dm[9] * tm[2]) + (dm[10] * tm[5]) + (dm[11] * tm[8])) / ts);
            mm[6] = (dm[0] + (((dm[3] * tm[0]) + (dm[4] * tm[3]) + (dm[5] * tm[6])) / ts)) / 2.0;
            mm[7] = (dm[1] + (((dm[3] * tm[1]) + (dm[4] * tm[4]) + (dm[5] * tm[7])) / ts)) / 2.0;
            mm[8] = (dm[2] + (((dm[3] * tm[2]) + (dm[4] * tm[5]) + (dm[5] * tm[8])) / ts)) / 2.0;

            FitMatrix.EigenNonSymmetric(mm, 3, eval, ev);

            cond[0] = (4.0 * ev[0] * ev[2]) - (ev[1] * ev[1]);
            cond[1] = (4.0 * ev[3] * ev[5]) - (ev[4] * ev[4]);
            cond[2] = (4.0 * ev[6] * ev[8]) - (ev[7] * ev[7]);
            if (cond[0] < cond[1])
            {
                sel = cond[1] < cond[2] ? 2 : 1;
            }
            else
            {
                sel = cond[0] < cond[2] ? 2 : 0;
            }

            double v0 = ev[sel * 3], v1 = ev[(sel * 3) + 1], v2 = ev[(sel * 3) + 2];
            double vnorm2 = (v0 * v0) + (v1 * v1) + (v2 * v2);
            if (cond[sel] > 1e-6 * vnorm2)
            {
                break;
            }

            eps = (float)(s / (n * 2) * 1e-2);
        }

        if (iter >= 2)
        {
            return NoDirect(points);
        }

        double e0 = ev[sel * 3], e1 = ev[(sel * 3) + 1], e2 = ev[(sel * 3) + 2];
        double norm = Math.Sqrt((e0 * e0) + (e1 * e1) + (e2 * e2));
        if ((e0 < 0.0 ? -1 : 1) * (e1 < 0.0 ? -1 : 1) * (e2 < 0.0 ? -1 : 1) <= 0.0)
        {
            norm = -1.0 * norm;
        }

        Span<double> pv = stackalloc double[3];
        pv[0] = e0 / norm;
        pv[1] = e1 / norm;
        pv[2] = e2 / norm;

        double q0 = ((tm[0] * pv[0]) + (tm[1] * pv[1]) + (tm[2] * pv[2])) / ts;
        double q1 = ((tm[3] * pv[0]) + (tm[4] * pv[1]) + (tm[5] * pv[2])) / ts;
        double q2 = ((tm[6] * pv[0]) + (tm[7] * pv[1]) + (tm[8] * pv[2])) / ts;

        double u1 = (pv[2] * q0 * q0) - (pv[1] * q0 * q1) + (pv[0] * q1 * q1) + (pv[1] * pv[1] * q2);
        double u2 = pv[0] * pv[2] * q2;
        double l1 = Math.Sqrt((pv[1] * pv[1]) + ((pv[0] - pv[2]) * (pv[0] - pv[2])));
        double l2 = pv[0] + pv[2];
        double l3 = (pv[1] * pv[1]) - (4 * pv[0] * pv[2]);
        double p1 = (2 * pv[2] * q0) - (pv[1] * q1);
        double p2 = (2 * pv[0] * q1) - (pv[1] * q0);

        double x0 = (p1 / l3 / scale) + cx;
        double y0 = (p2 / l3 / scale) + cy;
        double ea = Math.Sqrt(2.0) * Math.Sqrt((u1 - (4.0 * u2)) / ((l1 - l2) * l3)) / scale;
        double eb = Math.Sqrt(2.0) * Math.Sqrt(-1.0 * ((u1 - (4.0 * u2)) / ((l1 + l2) * l3))) / scale;
        double theta = Theta(pv);

        float width = (float)(2.0 * ea);
        float height = (float)(2.0 * eb);
        float angle;
        if (width > height)
        {
            (width, height) = (height, width);
            angle = (float)((90 + (theta * 180 / Math.PI)) % 180.0);
        }
        else
        {
            angle = (float)(theta * 180 / Math.PI % 180.0);
        }

        return new RotatedRect(new Point2f((float)x0, (float)y0), new Size2f(width, height), angle);
    }

    private static double Theta(ReadOnlySpan<double> pv)
    {
        if (pv[1] == 0)
        {
            return pv[0] < pv[2] ? 0 : Math.PI / 2.0;
        }

        return (Math.PI / 2.0) + (0.5 * Math.Atan2(pv[1], pv[0] - pv[2]));
    }
}
