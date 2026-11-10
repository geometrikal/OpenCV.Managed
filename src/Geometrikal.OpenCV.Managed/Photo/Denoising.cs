// This implementation is ported from the OpenCV non-local means denoising.
//
// Original sources:
//   modules/photo/src/denoising.cpp (fastNlMeansDenoising, fastNlMeansDenoisingColored)
//   modules/photo/src/fast_nlmeans_denoising_invoker.hpp
//   modules/photo/src/fast_nlmeans_denoising_invoker_commons.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Photo;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Non-local means denoising, as OpenCV <c>fastNlMeansDenoising</c>.</summary>
public static class Denoising
{
    /// <summary>
    /// Denoises an image with the non-local means algorithm, as OpenCV <c>fastNlMeansDenoising</c>.
    /// </summary>
    /// <param name="source">Image with 1 to 4 channels: <c>byte</c>, or <c>ushort</c> with <see cref="NormType.L1"/>.</param>
    /// <param name="h">Filter strength: one value for all channels, or one value per channel.</param>
    /// <param name="templateWindowSize">Side of the compared patches (odd; even values are reduced by one).</param>
    /// <param name="searchWindowSize">Side of the search window (odd; even values are reduced by one).</param>
    /// <param name="normType"><see cref="NormType.L2"/> (byte only) or <see cref="NormType.L1"/>.</param>
    /// <returns>A new denoised image.</returns>
    /// <remarks>
    /// Patch distances are exact integer sums over the template window (borders reflect, as
    /// <c>BORDER_REFLECT_101</c>). Each average distance is approximated by a binary shift and
    /// mapped to a fixed-point weight from a table of <c>cvRound(mult · exp(−d / (h²·cn)))</c>
    /// (L1: <c>exp(−d² / (h²·cn))</c>); weights below 0.001 of the multiplier are zero. The output is
    /// the rounded integer weighted mean. The result does not depend on the thread count.
    /// </remarks>
    public static Mat<T> FastNlMeansDenoising<T>(Mat<T> source, ReadOnlySpan<float> h, int templateWindowSize = 7, int searchWindowSize = 21, NormType normType = NormType.L2)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        int cn = source.Channels;
        if (h.Length != 1 && h.Length != cn)
        {
            throw new ArgumentException("h must have one value or one value per channel.", nameof(h));
        }

        if (cn is < 1 or > 4)
        {
            throw new NotSupportedException("Only 1, 2, 3 and 4 channels are supported.");
        }

        if (normType == NormType.L2)
        {
            if (typeof(T) != typeof(byte))
            {
                throw new NotSupportedException("Only byte images are supported for NORM_L2.");
            }
        }
        else if (normType == NormType.L1)
        {
            if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort))
            {
                throw new NotSupportedException("Only byte and ushort images are supported for NORM_L1.");
            }
        }
        else
        {
            throw new NotSupportedException("Only NORM_L2 and NORM_L1 are supported.");
        }

        return typeof(T) == typeof(byte)
            ? Unsafe.As<Mat<T>>(Run(Unsafe.As<Mat<byte>>(source), h.ToArray(), templateWindowSize, searchWindowSize, normType == NormType.L2, 255, int.MaxValue))
            : Unsafe.As<Mat<T>>(Run(Unsafe.As<Mat<ushort>>(source), h.ToArray(), templateWindowSize, searchWindowSize, false, 65535, long.MaxValue));
    }

    /// <inheritdoc cref="FastNlMeansDenoising{T}(Mat{T}, ReadOnlySpan{float}, int, int, NormType)"/>
    public static Mat<T> FastNlMeansDenoising<T>(Mat<T> source, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21, NormType normType = NormType.L2)
        where T : unmanaged =>
        FastNlMeansDenoising(source, [h], templateWindowSize, searchWindowSize, normType);

    /// <summary>
    /// Denoises a colour image, as OpenCV <c>fastNlMeansDenoisingColored</c>: the image is converted to
    /// Lab (<c>COLOR_LBGR2Lab</c>), L is denoised with <paramref name="h"/> and the a and b channels
    /// together with <paramref name="hColor"/>, and the result is converted back.
    /// </summary>
    /// <param name="source">Byte image with 3 (BGR) or 4 (BGRA) channels.</param>
    /// <param name="h">Filter strength for the lightness.</param>
    /// <param name="hColor">Filter strength for the colour components.</param>
    /// <param name="templateWindowSize">Side of the compared patches.</param>
    /// <param name="searchWindowSize">Side of the search window.</param>
    /// <returns>A new denoised image with the source channel count (alpha is the type maximum).</returns>
    public static Mat<byte> FastNlMeansDenoisingColored(Mat<byte> source, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels is not (3 or 4))
        {
            throw new NotSupportedException("Type of input image should be CV_8UC3 or CV_8UC4.");
        }

        var lab = ColorConversions.CvtColor(source, ColorConversionCode.Lbgr2Lab);
        int w = source.Width, ht = source.Height;
        var l = new Mat<byte>(w, ht);
        var ab = new Mat<byte>(w, ht, 2);
        for (int y = 0; y < ht; y++)
        {
            var s = lab.GetReadOnlyRowSpan(y);
            var lr = l.GetRowSpan(y);
            var abr = ab.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                lr[x] = s[3 * x];
                abr[2 * x] = s[(3 * x) + 1];
                abr[(2 * x) + 1] = s[(3 * x) + 2];
            }
        }

        l = FastNlMeansDenoising(l, h, templateWindowSize, searchWindowSize);
        ab = FastNlMeansDenoising(ab, hColor, templateWindowSize, searchWindowSize);
        for (int y = 0; y < ht; y++)
        {
            var d = lab.GetRowSpan(y);
            var lr = l.GetReadOnlyRowSpan(y);
            var abr = ab.GetReadOnlyRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                d[3 * x] = lr[x];
                d[(3 * x) + 1] = abr[2 * x];
                d[(3 * x) + 2] = abr[(2 * x) + 1];
            }
        }

        return ColorConversions.CvtColor(lab, ColorConversionCode.Lab2Lbgr, source.Channels);
    }

    private static Mat<T> Run<T>(Mat<T> source, float[] h, int templateWindowSize, int searchWindowSize, bool squared, int sampleMax, long itMax)
        where T : unmanaged
    {
        int cn = source.Channels;
        int th = templateWindowSize / 2, sh = searchWindowSize / 2;
        int tws = (th * 2) + 1, sws = (sh * 2) + 1;
        int border = sh + th;
        var ext = Borders.CopyMakeBorder(source, border, border, border, border, BorderType.Reflect101);

        long maxEstimate = (long)sws * sws * sampleMax;
        int fixedPointMult = (int)Math.Min(itMax / maxEstimate, int.MaxValue);
        int tsq = tws * tws;
        int shift = 0;
        while (1 << shift < tsq)
        {
            shift++;
        }

        double multiplier = (double)(1 << shift) / tsq;
        int maxDist = squared ? sampleMax * sampleMax * cn : sampleMax * cn;
        int almostMaxDist = (int)((maxDist / multiplier) + 1);
        int nw = h.Length;
        var table = new int[nw][];
        for (int c = 0; c < nw; c++)
        {
            table[c] = new int[almostMaxDist];
            float hh = h[c] * h[c] * cn;
            for (int k = 0; k < almostMaxDist; k++)
            {
                double dist = k * multiplier;
                double w = Math.Exp(-(squared ? dist : dist * dist) / hh);
                if (double.IsNaN(w))
                {
                    w = 1.0;
                }

                int weight = ScalarConverter.CvRound(fixedPointMult * w);
                table[c][k] = weight < 0.001 * fixedPointMult ? 0 : weight;
            }
        }

        int width = source.Width, height = source.Height;
        var dst = new Mat<T>(width, height, cn, clear: false);
        var job = new Job<T>(ext, dst, table, th, sh, border, shift, squared);
        ParallelRows.For(height, (long)width * sws * sws * 4, job.Rows);
        return dst;
    }

    private sealed class Job<T>
        where T : unmanaged
    {
        private readonly Mat<T> _ext;
        private readonly Mat<T> _dst;
        private readonly int[][] _table;
        private readonly int _th;
        private readonly int _sh;
        private readonly int _border;
        private readonly int _shift;
        private readonly bool _squared;

        public Job(Mat<T> ext, Mat<T> dst, int[][] table, int th, int sh, int border, int shift, bool squared)
        {
            _ext = ext;
            _dst = dst;
            _table = table;
            _th = th;
            _sh = sh;
            _border = border;
            _shift = shift;
            _squared = squared;
        }

        public void Rows(int r0, int r1)
        {
            int width = _dst.Width, cn = _dst.Channels, nw = _table.Length;
            int th = _th, tws = (2 * th) + 1, rows = r1 - r0;
            int bandRows = rows + (2 * th);
            var est = new long[rows * width * cn];
            var wsum = new long[rows * width * nw];
            var hsum = new int[bandRows * width];
            var dist = new int[width + (2 * th)];
            var colSum = new int[width];

            for (int oy = -_sh; oy <= _sh; oy++)
            {
                for (int ox = -_sh; ox <= _sh; ox++)
                {
                    // Horizontal template sums of the pixel distances for every band row.
                    for (int k = 0; k < bandRows; k++)
                    {
                        int ya = _border + r0 - th + k;
                        var a = _ext.GetReadOnlyRowSpan(ya);
                        var b = _ext.GetReadOnlyRowSpan(ya + oy);
                        int xa0 = _border - th;
                        for (int x = 0; x < dist.Length; x++)
                        {
                            dist[x] = Distance(a, b, (xa0 + x) * cn, (xa0 + x + ox) * cn, cn);
                        }

                        int s = 0;
                        for (int t = 0; t < tws; t++)
                        {
                            s += dist[t];
                        }

                        var hrow = hsum.AsSpan(k * width, width);
                        hrow[0] = s;
                        for (int x = 1; x < width; x++)
                        {
                            s += dist[x + tws - 1] - dist[x - 1];
                            hrow[x] = s;
                        }
                    }

                    // Vertical template sums, weights and accumulation.
                    Array.Clear(colSum);
                    for (int k = 0; k < tws - 1; k++)
                    {
                        Add(colSum, hsum.AsSpan(k * width, width));
                    }

                    for (int r = 0; r < rows; r++)
                    {
                        Add(colSum, hsum.AsSpan((r + tws - 1) * width, width));
                        var p = _ext.GetReadOnlyRowSpan(_border + r0 + r + oy);
                        int pBase = (_border + ox) * cn;
                        int eBase = r * width * cn, wBase = r * width * nw;
                        for (int x = 0; x < width; x++)
                        {
                            int almost = colSum[x] >> _shift;
                            for (int c = 0; c < cn; c++)
                            {
                                int weight = _table[nw == 1 ? 0 : c][almost];
                                est[eBase + (x * cn) + c] += (long)weight * Sample(p, pBase + (x * cn) + c);
                            }

                            if (nw == 1)
                            {
                                wsum[wBase + x] += _table[0][almost];
                            }
                            else
                            {
                                for (int c = 0; c < nw; c++)
                                {
                                    wsum[wBase + (x * nw) + c] += _table[c][almost];
                                }
                            }
                        }

                        Sub(colSum, hsum.AsSpan(r * width, width));
                    }
                }
            }

            for (int r = 0; r < rows; r++)
            {
                var o = _dst.GetRowSpan(r0 + r);
                for (int x = 0; x < width; x++)
                {
                    for (int c = 0; c < cn; c++)
                    {
                        long ws = wsum[(r * width * nw) + (x * nw) + (nw == 1 ? 0 : c)];
                        long e = est[(r * width * cn) + (x * cn) + c];
                        o[(x * cn) + c] = Saturate((e + (ws / 2)) / ws);
                    }
                }
            }
        }

        private static void Add(Span<int> acc, ReadOnlySpan<int> row)
        {
            for (int x = 0; x < acc.Length; x++)
            {
                acc[x] += row[x];
            }
        }

        private static void Sub(Span<int> acc, ReadOnlySpan<int> row)
        {
            for (int x = 0; x < acc.Length; x++)
            {
                acc[x] -= row[x];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int Distance(ReadOnlySpan<T> a, ReadOnlySpan<T> b, int ia, int ib, int cn)
        {
            int d = 0;
            for (int c = 0; c < cn; c++)
            {
                int diff = Sample(a, ia + c) - Sample(b, ib + c);
                d += _squared ? diff * diff : Math.Abs(diff);
            }

            return d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Sample(ReadOnlySpan<T> row, int i) =>
            typeof(T) == typeof(byte) ? Unsafe.As<T, byte>(ref Unsafe.AsRef(in row[i])) : Unsafe.As<T, ushort>(ref Unsafe.AsRef(in row[i]));

        private static T Saturate(long v)
        {
            if (typeof(T) == typeof(byte))
            {
                byte b = (byte)Math.Clamp(v, 0, 255);
                return Unsafe.As<byte, T>(ref b);
            }

            ushort u = (ushort)Math.Clamp(v, 0, 65535);
            return Unsafe.As<ushort, T>(ref u);
        }
    }
}
