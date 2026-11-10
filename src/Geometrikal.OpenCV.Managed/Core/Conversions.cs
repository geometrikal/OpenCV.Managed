// This implementation is ported from the OpenCV conversion and channel functions.
//
// Original sources:
//   modules/core/src/convert.dispatch.cpp, convert.simd.hpp (Mat::convertTo)
//   modules/core/src/convert_scale.dispatch.cpp, convert_scale.simd.hpp (convertScaleAbs,
//     scaled conversion work types)
//   modules/core/src/split.dispatch.cpp, merge.dispatch.cpp
//   modules/core/src/lut.cpp
//   modules/core/src/norm.dispatch.cpp (normalize)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>Mat::convertTo</c>, <c>convertScaleAbs</c>, <c>split</c>,
/// <c>merge</c>, <c>LUT</c> and <c>normalize</c>. No operation modifies its sources.
/// </summary>
public static class Conversions
{
    /// <summary>C <c>DBL_EPSILON</c>; <see cref="double.Epsilon"/> is the smallest subnormal instead.</summary>
    internal const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>
    /// Converts the element type with optional scaling, as OpenCV
    /// <c>src.convertTo(dst, type, alpha, beta)</c>: <c>dst = saturate_cast(src * alpha + beta)</c>.
    /// </summary>
    /// <remarks>
    /// Without scaling (<c>alpha = 1</c>, <c>beta = 0</c>) each element converts with
    /// <c>saturate_cast</c>. With scaling, the computation uses float (<c>(float)x * (float)alpha
    /// + (float)beta</c>) except for these pairs, which use double: any source to
    /// <c>double</c>, <c>int</c> to <c>int</c>, and <c>double</c> to <c>int</c> or
    /// <c>float</c>. A <c>double</c> source with float work converts each element to float first.
    /// Float-to-integer conversions round half to even; NaN and values outside the <c>int</c>
    /// range give the minimum of the destination type.
    /// </remarks>
    public static Mat<TDst> ConvertTo<TSrc, TDst>(Mat<TSrc> source, double alpha = 1, double beta = 0)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Saturate.EnsureSupported<TSrc>();
        Saturate.EnsureSupported<TDst>();
        var dst = new Mat<TDst>(source.Width, source.Height, source.Channels, clear: false);
        ConvertInto(source, dst, alpha, beta, null);
        return dst;
    }

    /// <summary>
    /// <see cref="ConvertTo{TSrc, TDst}"/> into <paramref name="dst"/>; where <paramref name="mask"/>
    /// is zero, <paramref name="dst"/> keeps its values.
    /// </summary>
    internal static void ConvertInto<TSrc, TDst>(Mat<TSrc> source, Mat<TDst> dst, double alpha, double beta, Mat<byte>? mask)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        bool noScale = Math.Abs(alpha - 1) < DblEpsilon && Math.Abs(beta) < DblEpsilon;
        if (noScale)
        {
            Elementwise.Unary(source, dst, mask, default(ConvertOp<TSrc, TDst>));
        }
        else if (UsesDoubleWork<TSrc, TDst>())
        {
            Elementwise.Unary(source, dst, mask, new ConvertScaleDoubleOp<TSrc, TDst>(alpha, beta));
        }
        else
        {
            Elementwise.Unary(source, dst, mask, new ConvertScaleFloatOp<TSrc, TDst>((float)alpha, (float)beta));
        }
    }

    /// <summary>
    /// Normalizes the norm or the value range of a matrix, as OpenCV
    /// <c>normalize(src, dst, alpha, beta, normType, dtype, mask)</c>, and returns a new matrix.
    /// Elements outside <paramref name="mask"/> are 0.
    /// </summary>
    /// <param name="source">Source matrix.</param>
    /// <param name="alpha">Target norm, or one bound of the range for <see cref="NormType.MinMax"/>.</param>
    /// <param name="beta">Other bound of the range for <see cref="NormType.MinMax"/>; ignored otherwise.</param>
    /// <param name="normType"><see cref="NormType.MinMax"/>, <see cref="NormType.Inf"/>, <see cref="NormType.L1"/> or <see cref="NormType.L2"/>.</param>
    /// <param name="mask">Optional single-channel mask of the source size.</param>
    /// <remarks>
    /// <para><see cref="NormType.MinMax"/>: with <c>dmin = min(alpha, beta)</c>,
    /// <c>dmax = max(alpha, beta)</c> and the source extremes <c>smin</c>, <c>smax</c> of the selected
    /// elements, <c>scale = (dmax - dmin) * (smax - smin &gt; DBL_EPSILON ? 1 / (smax - smin) : 0)</c>
    /// and <c>shift = dmin - smin * scale</c>. For a <c>float</c> destination, <c>scale</c> is rounded to
    /// float and <c>shift = (float)dmin - (float)(smin * scale)</c>. NaN elements are skipped. A
    /// multi-channel source uses all channels and does not accept a mask.</para>
    /// <para>Norms: <c>scale = norm &gt; DBL_EPSILON ? alpha / norm : 0</c>, <c>shift = 0</c>, with the norm of
    /// <see cref="Reductions.Norm{T}(Mat{T}, NormType, Mat{byte})"/>.</para>
    /// <para>The result is <see cref="ConvertTo{TSrc, TDst}"/> with <c>scale</c> and <c>shift</c>.</para>
    /// </remarks>
    public static Mat<TDst> Normalize<TSrc, TDst>(
        Mat<TSrc> source, double alpha = 1, double beta = 0, NormType normType = NormType.L2, Mat<byte>? mask = null)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<TDst>(source.Width, source.Height, source.Channels, clear: mask is not null);
        Normalize(source, dst, alpha, beta, normType, mask);
        return dst;
    }

    /// <summary>
    /// <see cref="Normalize{TSrc, TDst}(Mat{TSrc}, double, double, NormType, Mat{byte})"/> into
    /// <paramref name="destination"/>. Where <paramref name="mask"/> is zero,
    /// <paramref name="destination"/> keeps its values, as OpenCV <c>copyTo(dst, mask)</c>.
    /// </summary>
    public static void Normalize<TSrc, TDst>(
        Mat<TSrc> source, Mat<TDst> destination, double alpha, double beta, NormType normType, Mat<byte>? mask = null)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        Saturate.EnsureSupported<TSrc>();
        Saturate.EnsureSupported<TDst>();
        Elementwise.ValidateSameShape(source, destination, nameof(destination));
        Elementwise.ValidateMask(source, mask);

        double scale;
        double shift;
        if (normType == NormType.MinMax)
        {
            if (source.Channels > 1 && mask is not null)
            {
                throw new ArgumentException("A multi-channel source with a mask is not supported for MinMax.", nameof(mask));
            }

            Mat<TSrc> flat = source.Channels == 1
                ? source
                : new Mat<TSrc>(source.DangerousGetArray(), source.RowLength, source.Height, 1, source.Stride, source.DangerousGetOffset());
            MinMaxLocResult mm = Reductions.MinMaxLoc(flat, mask);
            double smin = mm.MinVal;
            double smax = mm.MaxVal;
            double dmin = Math.Min(alpha, beta);
            double dmax = Math.Max(alpha, beta);
            scale = (dmax - dmin) * (smax - smin > DblEpsilon ? 1.0 / (smax - smin) : 0);
            if (typeof(TDst) == typeof(float))
            {
                scale = (float)scale;
                shift = (float)dmin - (float)(smin * scale);
            }
            else
            {
                shift = dmin - (smin * scale);
            }
        }
        else if (normType is NormType.L1 or NormType.L2 or NormType.Inf)
        {
            scale = Reductions.Norm(source, normType, mask);
            scale = scale > DblEpsilon ? alpha / scale : 0.0;
            shift = 0;
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(normType));
        }

        ConvertInto(source, destination, scale, shift, mask);
    }

    /// <summary>
    /// Scales, takes the absolute value and converts to bytes, as OpenCV
    /// <c>convertScaleAbs</c>: <c>dst = saturate_cast&lt;byte&gt;(|src * alpha + beta|)</c>,
    /// computed in float for every source type.
    /// </summary>
    public static Mat<byte> ConvertScaleAbs<T>(Mat<T> source, double alpha = 1, double beta = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Saturate.EnsureSupported<T>();
        var dst = new Mat<byte>(source.Width, source.Height, source.Channels, clear: false);
        Elementwise.Unary(source, dst, null, new ConvertScaleAbsOp<T>((float)alpha, (float)beta));
        return dst;
    }

    /// <summary>Copies each channel into its own single-channel matrix, as OpenCV <c>split</c>.</summary>
    public static Mat<T>[] Split<T>(Mat<T> source)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        int cn = source.Channels;
        var planes = new Mat<T>[cn];
        for (int c = 0; c < cn; c++)
        {
            planes[c] = new Mat<T>(source.Width, source.Height, 1, clear: false);
        }

        ParallelRows.For(
            source.Height,
            Elementwise.WorkPerRow<T>(source.RowLength),
            (y0, y1) => SplitRows(source, planes, y0, y1));
        return planes;
    }

    private static void SplitRows<T>(Mat<T> source, Mat<T>[] planes, int y0, int y1)
        where T : unmanaged
    {
        int cn = source.Channels;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<T> row = source.GetReadOnlyRowSpan(y);
            Span<T> d0 = planes[0].GetRowSpan(y);
            Span<T> d1 = cn > 1 ? planes[1].GetRowSpan(y) : default;
            Span<T> d2 = cn > 2 ? planes[2].GetRowSpan(y) : default;
            Span<T> d3 = cn > 3 ? planes[3].GetRowSpan(y) : default;
            int x = cn is 3 or 4 ? ChannelShuffle.Split(row, d0, d1, d2, d3, cn) : 0;
            if (cn <= 4)
            {
                for (int i = x * cn; x < d0.Length; x++, i += cn)
                {
                    d0[x] = row[i];
                    if (cn > 1)
                    {
                        d1[x] = row[i + 1];
                    }

                    if (cn > 2)
                    {
                        d2[x] = row[i + 2];
                    }

                    if (cn > 3)
                    {
                        d3[x] = row[i + 3];
                    }
                }

                continue;
            }

            for (int c = 0; c < cn; c++)
            {
                Span<T> d = planes[c].GetRowSpan(y);
                for (int xx = 0, i = c; xx < d.Length; xx++, i += cn)
                {
                    d[xx] = row[i];
                }
            }
        }
    }

    /// <summary>
    /// Interleaves the channels of several matrices of the same size into one matrix, as OpenCV
    /// <c>merge</c>. Each input may have several channels; the channel counts add up.
    /// </summary>
    public static Mat<T> Merge<T>(params Mat<T>[] planes)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(planes);
        if (planes.Length == 0)
        {
            throw new ArgumentException("At least one matrix is required.", nameof(planes));
        }

        int width = planes[0].Width;
        int height = planes[0].Height;
        int cn = 0;
        foreach (Mat<T> p in planes)
        {
            ArgumentNullException.ThrowIfNull(p, nameof(planes));
            if (p.Width != width || p.Height != height)
            {
                throw new ArgumentException("All matrices must have the same size.", nameof(planes));
            }

            cn += p.Channels;
        }

        var dst = new Mat<T>(width, height, cn, clear: false);
        ParallelRows.For(
            height,
            Elementwise.WorkPerRow<T>(dst.RowLength),
            (y0, y1) => MergeRows(planes, dst, y0, y1));
        return dst;
    }

    private static void MergeRows<T>(Mat<T>[] planes, Mat<T> dst, int y0, int y1)
        where T : unmanaged
    {
        int cn = dst.Channels;
        bool singlePlanes = planes.Length == cn;
        for (int y = y0; y < y1; y++)
        {
            Span<T> d = dst.GetRowSpan(y);
            int x = 0;
            if (singlePlanes && cn is 3 or 4)
            {
                ReadOnlySpan<T> s0 = planes[0].GetReadOnlyRowSpan(y);
                ReadOnlySpan<T> s1 = planes[1].GetReadOnlyRowSpan(y);
                ReadOnlySpan<T> s2 = planes[2].GetReadOnlyRowSpan(y);
                ReadOnlySpan<T> s3 = cn == 4 ? planes[3].GetReadOnlyRowSpan(y) : default;
                x = ChannelShuffle.Merge(s0, s1, s2, s3, d, cn);
            }

            int offset = 0;
            foreach (Mat<T> p in planes)
            {
                ReadOnlySpan<T> s = p.GetReadOnlyRowSpan(y);
                int pcn = p.Channels;
                for (int xx = x; xx < dst.Width; xx++)
                {
                    for (int c = 0; c < pcn; c++)
                    {
                        d[(xx * cn) + offset + c] = s[(xx * pcn) + c];
                    }
                }

                offset += pcn;
            }
        }
    }

    /// <summary>
    /// Maps every byte through a 256-entry lookup table, as OpenCV <c>LUT</c>. A single-channel
    /// table applies to all channels; a table with the source channel count maps channel
    /// <c>c</c> through table channel <c>c</c>.
    /// </summary>
    /// <param name="source">Source matrix of bytes.</param>
    /// <param name="table">Table with 256 pixels and 1 or <c>source.Channels</c> channels.</param>
    /// <returns>A new matrix with the table element type and the source channel count.</returns>
    public static Mat<T> Lut<T>(Mat<byte> source, Mat<T> table)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(table);
        if (table.PixelCount != 256 || (table.Channels != 1 && table.Channels != source.Channels))
        {
            throw new ArgumentException(
                "The table must have 256 pixels and 1 channel or the source channel count.", nameof(table));
        }

        T[] lut = table.ToArray();
        int cn = source.Channels;
        bool perChannel = table.Channels != 1;
        var dst = new Mat<T>(source.Width, source.Height, cn, clear: false);
        ParallelRows.For(
            source.Height,
            source.RowLength,
            (y0, y1) =>
            {
                for (int y = y0; y < y1; y++)
                {
                    ReadOnlySpan<byte> s = source.GetReadOnlyRowSpan(y);
                    Span<T> d = dst.GetRowSpan(y);
                    if (!perChannel)
                    {
                        for (int i = 0; i < s.Length; i++)
                        {
                            d[i] = lut[s[i]];
                        }
                    }
                    else
                    {
                        for (int i = 0, c = 0; i < s.Length; i++, c = c + 1 == cn ? 0 : c + 1)
                        {
                            d[i] = lut[(s[i] * cn) + c];
                        }
                    }
                }
            });
        return dst;
    }

    private static bool UsesDoubleWork<TSrc, TDst>()
        where TSrc : unmanaged
        where TDst : unmanaged
        => typeof(TDst) == typeof(double) ||
           (typeof(TSrc) == typeof(int) && typeof(TDst) == typeof(int)) ||
           (typeof(TSrc) == typeof(double) && (typeof(TDst) == typeof(int) || typeof(TDst) == typeof(float)));
}
