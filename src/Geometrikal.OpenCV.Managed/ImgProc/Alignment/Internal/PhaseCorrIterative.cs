// This implementation is ported from OpenCV iterative phase correlation.
//
// Original sources:
//   modules/imgproc/src/phasecorr_iterative.cpp (phaseCorrelateIterative and helpers)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV 5.0 <c>phaseCorrelateIterative</c>.</summary>
internal static class PhaseCorrIterative
{
    private const int L2USize = 223;
    private const double L1RatioBase = 0.45;
    private const double L1RatioStep = 0.025;

    public static Point2d Run<T>(Mat<T> src1, Mat<T> src2, int l2Size, int maxIters)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int w = src1.Width, h = src1.Height;
        using PooledMat<T> image1Pool = PooledMat<T>.Rent(w, h, 1);
        using PooledMat<T> image2Pool = PooledMat<T>.Rent(w, h, 1);
        using PooledMat<T> dft1Pool = PooledMat<T>.Rent(w, h, 2);
        using PooledMat<T> dft2Pool = PooledMat<T>.Rent(w, h, 2);
        Mat<T> image1 = image1Pool.Mat, image2 = image2Pool.Mat, dft1 = dft1Pool.Mat, dft2 = dft2Pool.Mat;
        Mat<T> window = PhaseCorrelation.CreateHanningWindow<T>(new Size(w, h));
        Arithmetic.Multiply(src1, window, image1);
        Arithmetic.Multiply(src2, window, image2);
        DftEngine<T>.Dft(image1, dft1, inv: false, rowTransform: false, scaled: false, nonzeroRows: 0);
        DftEngine<T>.Dft(image2, dft2, inv: false, rowTransform: false, scaled: false, nonzeroRows: 0);

        CrossPowerSpectrum(dft1, dft2, dft1);
        Mat<T> l3 = image1;
        DftEngine<T>.Dft(dft1, l3, inv: true, rowTransform: false, scaled: true, nonzeroRows: 0);
        FftShift(l3);
        var l3Mid = new Point2d(l3.Width / 2, l3.Height / 2);
        Point l3PeakI = Reductions.MinMaxLoc(l3).MaxLoc;
        var l3Peak = new Point2d(l3PeakI.X, l3PeakI.Y);

        while (IsOutOfBounds(l3Peak, l3, l2Size))
        {
            l2Size -= 2;
            if (l2Size < 3)
            {
                return Sub(l3Peak, l3Mid);
            }
        }

        Mat<T> l2 = l3.Roi(l3PeakI.X - (l2Size / 2), l3PeakI.Y - (l2Size / 2), l2Size, l2Size);
        Mat<T> l2u = Geometry.Resize(l2, new Size(L2USize, L2USize), 0, 0, InterpolationFlags.Linear);
        var l2uMid = new Point2d(l2u.Width / 2, l2u.Height / 2);

        for (double l1Ratio = L1RatioBase; L1Size(l2u.Width, l1Ratio) > 0; l1Ratio -= L1RatioStep)
        {
            Point2d l2uPeak = l2uMid;
            int l1Size = L1Size(l2u.Width, l1Ratio);
            var l1Mid = new Point2d(l1Size / 2, l1Size / 2);

            for (int iter = 0; iter < maxIters; ++iter)
            {
                if (IsOutOfBounds(l2uPeak, l2u, l1Size))
                {
                    break;
                }

                Mat<T> l1 = l2u.Roi((int)(l2uPeak.X - (l1Size / 2)), (int)(l2uPeak.Y - (l1Size / 2)), l1Size, l1Size);
                Point2d l1Peak = PeakSubpixel(l1);
                l2uPeak = new Point2d(
                    l2uPeak.X + Math.Round(l1Peak.X - l1Mid.X, MidpointRounding.AwayFromZero),
                    l2uPeak.Y + Math.Round(l1Peak.Y - l1Mid.Y, MidpointRounding.AwayFromZero));

                if (Math.Abs(l1Peak.X - l1Mid.X) < 0.5 && Math.Abs(l1Peak.Y - l1Mid.Y) < 0.5)
                {
                    double scale = (double)L2USize / l2Size;
                    double ox = (l2uPeak.X - l2uMid.X + l1Peak.X - l1Mid.X) / scale;
                    double oy = (l2uPeak.Y - l2uMid.Y + l1Peak.Y - l1Mid.Y) / scale;
                    return new Point2d(l3Peak.X - l3Mid.X + ox, l3Peak.Y - l3Mid.Y + oy);
                }
            }
        }

        return SubpixelShift(l3, l3Peak, l3Mid, l2Size);
    }

    private static Point2d Sub(Point2d a, Point2d b) => new(a.X - b.X, a.Y - b.Y);

    /// <summary>
    /// Normalized cross-power spectrum of two complex spectra, in the element type.
    /// <paramref name="cps"/> can be <paramref name="a"/> or <paramref name="b"/>.
    /// </summary>
    private static void CrossPowerSpectrum<T>(Mat<T> a, Mat<T> b, Mat<T> cps)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        for (int y = 0; y < a.Height; y++)
        {
            ReadOnlySpan<T> pa = a.GetReadOnlyRowSpan(y);
            ReadOnlySpan<T> pb = b.GetReadOnlyRowSpan(y);
            Span<T> d = cps.GetRowSpan(y);
            for (int i = 0; i < pa.Length; i += 2)
            {
                T re = (pa[i] * pb[i]) + (pa[i + 1] * pb[i + 1]);
                T im = (pa[i] * pb[i + 1]) - (pa[i + 1] * pb[i]);
                T mag = T.Sqrt((re * re) + (im * im));
                d[i] = re / mag;
                d[i + 1] = im / mag;
            }
        }
    }

    /// <summary>The quadrant swap of phasecorr_iterative.cpp; an odd last row or column stays in place.</summary>
    private static void FftShift<T>(Mat<T> m)
        where T : unmanaged
    {
        int cx = m.Width / 2, cy = m.Height / 2;
        for (int y = 0; y < cy; y++)
        {
            Span<T> r0 = m.GetRowSpan(y);
            Span<T> r1 = m.GetRowSpan(y + cy);
            for (int x = 0; x < cx; x++)
            {
                (r0[x], r1[x + cx]) = (r1[x + cx], r0[x]);
                (r0[x + cx], r1[x]) = (r1[x], r0[x + cx]);
            }
        }
    }

    private static bool IsOutOfBounds<T>(Point2d peak, Mat<T> mat, int size)
        where T : unmanaged
        => peak.X - (size / 2) < 0 || peak.Y - (size / 2) < 0 || peak.X + (size / 2) >= mat.Width || peak.Y + (size / 2) >= mat.Height;

    private static int L1Size(int l2uSize, double l1Ratio)
    {
        int l1 = (int)Math.Floor(l1Ratio * l2uSize);
        return (l1 % 2) != 0 ? l1 : l1 + 1;
    }

    private static Point2d PeakSubpixel<T>(Mat<T> mat)
        where T : unmanaged
    {
        Moments m = ShapeDescriptors.Moments(mat);
        return new Point2d(m.M10 / m.M00, m.M01 / m.M00);
    }

    private static Point2d SubpixelShift<T>(Mat<T> l3, Point2d l3Peak, Point2d l3Mid, int l2Size)
        where T : unmanaged
    {
        while (IsOutOfBounds(l3Peak, l3, l2Size))
        {
            l2Size -= 2;
            if (l2Size < 3)
            {
                return Sub(l3Peak, l3Mid);
            }
        }

        Mat<T> l2 = l3.Roi((int)(l3Peak.X - (l2Size / 2)), (int)(l3Peak.Y - (l2Size / 2)), l2Size, l2Size);
        Point2d l2Peak = PeakSubpixel(l2);
        var l2Mid = new Point2d(l2.Width / 2, l2.Height / 2);
        return new Point2d(l3Peak.X - l3Mid.X + l2Peak.X - l2Mid.X, l3Peak.Y - l3Mid.Y + l2Peak.Y - l2Mid.Y);
    }
}
