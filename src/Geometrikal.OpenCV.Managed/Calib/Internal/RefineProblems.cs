// This implementation is ported from the OpenCV refinement callbacks of the 2-D estimators.
//
// Original sources:
//   modules/geometry/src/ptsetreg.cpp (affine2DRefineCallback, affinePartial2dRefineCallback,
//   translation2dRefineCallback)
//   modules/geometry/src/fundam.cpp (homographyRefineCallback)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Residuals of a full affine model with parameters (a, b, c, d, e, f).</summary>
internal readonly struct AffineRefineProblem(Point2f[] src, Point2f[] dst, int count) : ILevMarqProblem
{
    public int ParamCount => 6;

    public int ErrorCount => 2 * count;

    public void Compute(ReadOnlySpan<double> h, Span<double> err, Span<double> jac)
    {
        double h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4], h5 = h[5];
        ReadOnlySpan<Point2f> s = src.AsSpan(0, count), d = dst.AsSpan(0, count);
        Span<double> e = err[..(2 * count)];
        Span<double> jc = jac.IsEmpty ? default : jac[..(12 * count)];
        for (int i = 0; i < s.Length; i++)
        {
            double mx = s[i].X, my = s[i].Y;
            double xi = (h0 * mx) + (h1 * my) + h2;
            double yi = (h3 * mx) + (h4 * my) + h5;
            e[2 * i] = xi - d[i].X;
            e[(2 * i) + 1] = yi - d[i].Y;
            if (!jc.IsEmpty)
            {
                int o = 12 * i;
                jc[o] = mx;
                jc[o + 1] = my;
                jc[o + 2] = 1.0;
                jc[o + 9] = mx;
                jc[o + 10] = my;
                jc[o + 11] = 1.0;
            }
        }
    }
}

/// <summary>Residuals of a partial affine model with parameters (a, b, tx, ty).</summary>
internal readonly struct AffinePartialRefineProblem(Point2f[] src, Point2f[] dst, int count) : ILevMarqProblem
{
    public int ParamCount => 4;

    public int ErrorCount => 2 * count;

    public void Compute(ReadOnlySpan<double> h, Span<double> err, Span<double> jac)
    {
        double h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3];
        ReadOnlySpan<Point2f> s = src.AsSpan(0, count), d = dst.AsSpan(0, count);
        Span<double> e = err[..(2 * count)];
        Span<double> jc = jac.IsEmpty ? default : jac[..(8 * count)];
        for (int i = 0; i < s.Length; i++)
        {
            double mx = s[i].X, my = s[i].Y;
            double xi = (h0 * mx) - (h1 * my) + h2;
            double yi = (h1 * mx) + (h0 * my) + h3;
            e[2 * i] = xi - d[i].X;
            e[(2 * i) + 1] = yi - d[i].Y;
            if (!jc.IsEmpty)
            {
                int o = 8 * i;
                jc[o] = mx;
                jc[o + 1] = -my;
                jc[o + 2] = 1.0;
                jc[o + 3] = 0.0;
                jc[o + 4] = my;
                jc[o + 5] = mx;
                jc[o + 6] = 0.0;
                jc[o + 7] = 1.0;
            }
        }
    }
}

/// <summary>Residuals of a translation with parameters (tx, ty).</summary>
internal readonly struct TranslationRefineProblem(Point2f[] src, Point2f[] dst, int count) : ILevMarqProblem
{
    public int ParamCount => 2;

    public int ErrorCount => 2 * count;

    public void Compute(ReadOnlySpan<double> h, Span<double> err, Span<double> jac)
    {
        for (int i = 0; i < count; i++)
        {
            double mx = src[i].X, my = src[i].Y;
            double xi = mx + h[0];
            double yi = my + h[1];
            err[2 * i] = xi - dst[i].X;
            err[(2 * i) + 1] = yi - dst[i].Y;
            if (!jac.IsEmpty)
            {
                jac[4 * i] = 1.0;
                jac[(4 * i) + 3] = 1.0;
            }
        }
    }
}

/// <summary>Residuals of a homography with 9 parameters.</summary>
internal readonly struct HomographyRefineProblem(Point2f[] src, Point2f[] dst, int count) : ILevMarqProblem
{
    public int ParamCount => 9;

    public int ErrorCount => 2 * count;

    public void Compute(ReadOnlySpan<double> h, Span<double> err, Span<double> jac)
    {
        const double DblEpsilon = 2.220446049250313e-16;
        double h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4], h5 = h[5], h6 = h[6], h7 = h[7], h8 = h[8];
        ReadOnlySpan<Point2f> s = src.AsSpan(0, count), d = dst.AsSpan(0, count);
        Span<double> e = err[..(2 * count)];
        Span<double> jc = jac.IsEmpty ? default : jac[..(18 * count)];
        for (int i = 0; i < s.Length; i++)
        {
            double mx = s[i].X, my = s[i].Y;
            double ww = (h6 * mx) + (h7 * my) + h8;
            ww = Math.Abs(ww) > DblEpsilon ? 1.0 / ww : 0;
            double xi = ((h0 * mx) + (h1 * my) + h2) * ww;
            double yi = ((h3 * mx) + (h4 * my) + h5) * ww;
            e[2 * i] = xi - d[i].X;
            e[(2 * i) + 1] = yi - d[i].Y;
            if (!jc.IsEmpty)
            {
                int o = 18 * i;
                jc[o] = mx * ww;
                jc[o + 1] = my * ww;
                jc[o + 2] = ww;
                jc[o + 6] = -mx * ww * xi;
                jc[o + 7] = -my * ww * xi;
                jc[o + 8] = -ww * xi;
                jc[o + 12] = mx * ww;
                jc[o + 13] = my * ww;
                jc[o + 14] = ww;
                jc[o + 15] = -mx * ww * yi;
                jc[o + 16] = -my * ww * yi;
                jc[o + 17] = -ww * yi;
            }
        }
    }
}
