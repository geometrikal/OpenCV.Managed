// This implementation is ported from OpenCV phase correlation.
//
// Original sources:
//   modules/imgproc/src/phasecorr.cpp (magSpectrums, fftShift, weightedCentroid, createHanningWindow)
//   modules/core/src/dxt.cpp (divSpectrums)
//   modules/core/include/opencv2/core/hal/intrin_math.hpp (v_sincos_default_64f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Helpers of OpenCV <c>phaseCorrelate</c> for 1-channel CCS spectra of float or double type.
/// </summary>
internal static class PhaseCorrKernels
{
    /// <summary>
    /// OpenCV <c>magSpectrums</c> for a 1-channel CCS spectrum: the magnitude of each packed
    /// complex value at its real slot, zero at its imaginary slot. Float spectra compute the
    /// magnitude in double. <paramref name="dst"/> must not be <paramref name="src"/>.
    /// </summary>
    public static void MagSpectrums<T>(Mat<T> src, Mat<T> dst)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int rows = src.Height, cols = src.Width;
        dst.Clear();
        bool is1d = rows == 1 || (cols == 1 && src.IsContinuous);
        if (is1d)
        {
            cols = cols + rows - 1;
            rows = 1;
        }

        int j0 = 1;
        int j1 = cols - (cols % 2 == 0 ? 1 : 0);

        if (!is1d)
        {
            for (int k = 0; k < (cols % 2 != 0 ? 1 : 2); k++)
            {
                int c = k == 1 ? cols - 1 : 0;
                dst.At(0, c) = T.Abs(src.At(0, c));
                if (rows % 2 == 0)
                {
                    dst.At(rows - 1, c) = T.Abs(src.At(rows - 1, c));
                }

                for (int j = 1; j <= rows - 2; j += 2)
                {
                    dst.At(j, c) = Mag(src.At(j, c), src.At(j + 1, c));
                }
            }
        }

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<T> s = is1d ? src.GetReadOnlySpan() : src.GetReadOnlyRowSpan(r);
            Span<T> d = is1d ? dst.GetSpan() : dst.GetRowSpan(r);
            if (is1d)
            {
                d[0] = T.Abs(s[0]);
                if (cols % 2 == 0)
                {
                    d[j1] = T.Abs(s[j1]);
                }
            }

            for (int j = j0; j < j1; j += 2)
            {
                d[j] = Mag(s[j], s[j + 1]);
            }
        }
    }

    private static T Mag<T>(T re, T im)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        double a = double.CreateTruncating(re), b = double.CreateTruncating(im);
        return T.CreateTruncating(Math.Sqrt((a * a) + (b * b)));
    }

    /// <summary>
    /// OpenCV <c>divSpectrums(a, b, c, 0, false)</c> for 1-channel CCS spectra: <c>a / b</c> per
    /// packed complex value with <c>eps</c> (<c>FLT_EPSILON</c> or <c>DBL_EPSILON</c>) added to the
    /// denominators. Float spectra compute in double. <paramref name="c"/> can be <paramref name="a"/>.
    /// </summary>
    public static void DivSpectrums<T>(Mat<T> a, Mat<T> b, Mat<T> c)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int rows = a.Height, cols = a.Width;
        bool is1d = rows == 1 || (cols == 1 && a.IsContinuous && b.IsContinuous);
        if (is1d)
        {
            cols = cols + rows - 1;
            rows = 1;
        }

        int j0 = 1;
        int j1 = cols - (cols % 2 == 0 ? 1 : 0);
        bool isFloat = typeof(T) == typeof(float);
        T eps = isFloat ? T.CreateTruncating(1.1920928955078125E-07) : T.CreateTruncating(2.220446049250313E-16);
        double deps = double.CreateTruncating(eps);

        if (!is1d)
        {
            for (int k = 0; k < (cols % 2 != 0 ? 1 : 2); k++)
            {
                int x = k == 1 ? cols - 1 : 0;
                c.At(0, x) = a.At(0, x) / (b.At(0, x) + eps);
                if (rows % 2 == 0)
                {
                    c.At(rows - 1, x) = a.At(rows - 1, x) / (b.At(rows - 1, x) + eps);
                }

                for (int j = 1; j <= rows - 2; j += 2)
                {
                    Div(a.At(j, x), a.At(j + 1, x), b.At(j, x), b.At(j + 1, x), deps, out c.At(j, x), out c.At(j + 1, x));
                }
            }
        }

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<T> sa = is1d ? a.GetReadOnlySpan() : a.GetReadOnlyRowSpan(r);
            ReadOnlySpan<T> sb = is1d ? b.GetReadOnlySpan() : b.GetReadOnlyRowSpan(r);
            Span<T> d = is1d ? c.GetSpan() : c.GetRowSpan(r);
            if (is1d)
            {
                d[0] = sa[0] / (sb[0] + eps);
                if (cols % 2 == 0)
                {
                    d[j1] = sa[j1] / (sb[j1] + eps);
                }
            }

            for (int j = j0; j < j1; j += 2)
            {
                Div(sa[j], sa[j + 1], sb[j], sb[j + 1], deps, out d[j], out d[j + 1]);
            }
        }
    }

    private static void Div<T>(T are, T aim, T bre, T bim, double eps, out T re, out T im)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        double ar = double.CreateTruncating(are), ai = double.CreateTruncating(aim);
        double br = double.CreateTruncating(bre), bi = double.CreateTruncating(bim);
        double denom = (br * br) + (bi * bi) + eps;
        double r = (ar * br) + (ai * bi);
        double i = (ai * br) - (ar * bi);
        re = T.CreateTruncating(r / denom);
        im = T.CreateTruncating(i / denom);
    }

    /// <summary>OpenCV <c>fftShift</c> (phasecorr.cpp): swaps the quadrants, or the halves of a vector, in place. <paramref name="copy"/> is a scratch matrix of the same shape.</summary>
    public static void FftShift<T>(Mat<T> m, Mat<T> copy)
        where T : unmanaged
    {
        int rows = m.Height, cols = m.Width;
        if (rows == 1 && cols == 1)
        {
            return;
        }

        int xMid = cols >> 1, yMid = rows >> 1;
        m.CopyTo(copy);
        if (xMid == 0 || yMid == 0)
        {
            if (cols == 1)
            {
                throw new NotSupportedException("OpenCV phaseCorrelate fails for single-column inputs.");
            }

            int isOdd = cols % 2;
            ReadOnlySpan<T> s = copy.GetReadOnlyRowSpan(0);
            Span<T> d = m.GetRowSpan(0);
            int first = xMid + isOdd;
            s[first..].CopyTo(d[..xMid]);
            s[..first].CopyTo(d[xMid..]);
            return;
        }

        int isXodd = cols % 2, isYodd = rows % 2;
        int qx = xMid + isXodd, qy = yMid + isYodd;

        // Quadrant q0 (qx×qy) moves to (xMid, yMid); q1 to (0, yMid); q2 to (xMid, 0); q3 to (0, 0).
        for (int y = 0; y < rows; y++)
        {
            ReadOnlySpan<T> s = copy.GetReadOnlyRowSpan(y);
            int dy = y < qy ? y + yMid : y - qy;
            Span<T> d = m.GetRowSpan(dy);
            s[..qx].CopyTo(d[xMid..]);
            s[qx..].CopyTo(d[..xMid]);
        }
    }

    /// <summary>
    /// OpenCV <c>weightedCentroid</c>: centroid of the <paramref name="box"/> window around
    /// <paramref name="peak"/>, clipped to the matrix, with double sums in row order.
    /// </summary>
    public static Point2d WeightedCentroid<T>(Mat<T> src, Point peak, Size box, out double response)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int minr = Math.Max(peak.Y - (box.Height >> 1), 0);
        int maxr = Math.Min(peak.Y + (box.Height >> 1), src.Height - 1);
        int minc = Math.Max(peak.X - (box.Width >> 1), 0);
        int maxc = Math.Min(peak.X + (box.Width >> 1), src.Width - 1);

        double cx = 0, cy = 0, sum = 0;
        for (int y = minr; y <= maxr; y++)
        {
            ReadOnlySpan<T> row = src.GetReadOnlyRowSpan(y);
            for (int x = minc; x <= maxc; x++)
            {
                double v = double.CreateTruncating(row[x]);
                cx += x * v;
                cy += y * v;
                sum += v;
            }
        }

        response = sum;
        sum += 2.220446049250313E-16;
        return new Point2d(cx / sum, cy / sum);
    }

    /// <summary>
    /// OpenCV <c>v_cos</c> for <c>v_float64</c> (<c>v_sincos_default_64f</c>, Cephes polynomial)
    /// without fused multiply-add, as the SSE3 baseline build evaluates it.
    /// </summary>
    public static double SimdCos(double x)
    {
        const double fopi = 1.2732395447351626861510701069801148;
        const double minusDp1 = -7.853981554508209228515625E-1;
        const double minusDp2 = -7.94662735614792836714E-9;
        const double minusDp3 = -3.06161699786838294307E-17;

        if (double.IsNaN(x) || double.IsInfinity(x))
        {
            return double.NaN;
        }

        double vx = Math.Abs(x);
        double vy = vx * fopi;
        long emm2 = (int)vy;
        emm2 = (emm2 + 1) & ~1L;
        vy = emm2;
        bool polyMask = (emm2 & 2) == 0;

        vx = (vy * minusDp1) + vx;
        vx = (vy * minusDp2) + vx;
        vx = (vy * minusDp3) + vx;

        bool signCos = ((emm2 - 2) & 4) == 0;
        double xx = vx * vx;

        double y1 = (-1.13585365213876817300E-11 * xx) + 2.08757008419747316778E-9;
        y1 = (y1 * xx) + -2.75573141792967388112E-7;
        y1 = (y1 * xx) + 2.48015872888517045348E-5;
        y1 = (y1 * xx) + -1.38888888888730564116E-3;
        y1 = (y1 * xx) + 4.16666666666665929218E-2;
        y1 = (y1 * xx) + -0.5;
        y1 = (y1 * xx) + 1.0;

        double y2 = (1.58962301576546568060E-10 * xx) + -2.50507477628578072866E-8;
        y2 = (y2 * xx) + 2.75573136213857245213E-6;
        y2 = (y2 * xx) + -1.98412698295895385996E-4;
        y2 = (y2 * xx) + 8.33333333332211858878E-3;
        y2 = (y2 * xx) + -1.66666666666666307295E-1;
        y2 *= xx;
        y2 = (y2 * vx) + vx;

        double ycos = polyMask ? y1 : y2;
        return signCos ? -ycos : ycos;
    }
}
