// This implementation is ported from the OpenCV reduction functions.
//
// Original sources:
//   modules/core/src/sum.dispatch.cpp, sum.simd.hpp (sum)
//   modules/core/src/mean.dispatch.cpp, mean.simd.hpp (mean, meanStdDev)
//   modules/core/src/minmax.dispatch.cpp (minMaxIdx, minMaxLoc)
//   modules/core/src/count_non_zero.dispatch.cpp (countNonZero)
//   modules/core/src/norm.dispatch.cpp, norm.simd.hpp (norm)
//   modules/core/include/opencv2/core/base.hpp (cv_abs, cv_absdiff)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Per-row partial results. Rows are processed in parallel; partials are combined in row
/// order, so results do not depend on the thread count.
/// </summary>
internal static class ReductionKernels
{
    /// <summary>
    /// Per-channel sums, sums of squares (when <paramref name="squares"/> is set) and the count
    /// of selected pixels.
    /// </summary>
    public static void SumRows<T>(
        Mat<T> src, Mat<byte>? mask, bool squares, out double[] sum, out double[] sqsum, out long count)
        where T : unmanaged
    {
        int cn = src.Channels;
        int h = src.Height;
        var rowSum = new double[h * cn];
        var rowSq = new double[h * cn];
        var rowCount = new long[h];
        ParallelRows.For(
            h,
            Elementwise.WorkPerRow<T>(src.RowLength),
            (y0, y1) => SumStripe(src, mask, squares, rowSum, rowSq, rowCount, y0, y1));

        sum = new double[cn];
        sqsum = new double[cn];
        count = 0;
        for (int y = 0; y < h; y++)
        {
            for (int c = 0; c < cn; c++)
            {
                sum[c] += rowSum[(y * cn) + c];
                sqsum[c] += rowSq[(y * cn) + c];
            }

            count += rowCount[y];
        }
    }

    private static void SumStripe<T>(
        Mat<T> src, Mat<byte>? mask, bool squares, double[] rowSum, double[] rowSq, long[] rowCount, int y0, int y1)
        where T : unmanaged
    {
        int cn = src.Channels;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<T> row = src.GetReadOnlyRowSpan(y);
            Span<double> s = rowSum.AsSpan(y * cn, cn);
            Span<double> q = rowSq.AsSpan(y * cn, cn);
            if (mask is not null)
            {
                rowCount[y] = MaskedSums(row, mask.GetReadOnlyRowSpan(y), cn, squares, s, q);
                continue;
            }

            rowCount[y] = src.Width;
            if (cn == 1 && typeof(T) == typeof(byte))
            {
                (s[0], q[0]) = ByteSums(MemoryMarshal.Cast<T, byte>(row), squares);
            }
            else if (cn == 1)
            {
                double a = 0, b = 0;
                for (int i = 0; i < row.Length; i++)
                {
                    double v = Saturate.ToDouble(row[i]);
                    a += v;
                    b += v * v;
                }

                s[0] = a;
                q[0] = squares ? b : 0;
            }
            else
            {
                InterleavedSums(row, cn, squares, s, q);
            }
        }
    }

    private static void InterleavedSums<T>(ReadOnlySpan<T> row, int cn, bool squares, Span<double> s, Span<double> q)
        where T : unmanaged
    {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0, q0 = 0, q1 = 0, q2 = 0, q3 = 0;
        for (int i = 0; i <= row.Length - cn; i += cn)
        {
            double v0 = Saturate.ToDouble(row[i]);
            s0 += v0;
            q0 += v0 * v0;
            if (cn > 1)
            {
                double v1 = Saturate.ToDouble(row[i + 1]);
                s1 += v1;
                q1 += v1 * v1;
            }

            if (cn > 2)
            {
                double v2 = Saturate.ToDouble(row[i + 2]);
                s2 += v2;
                q2 += v2 * v2;
            }

            if (cn > 3)
            {
                double v3 = Saturate.ToDouble(row[i + 3]);
                s3 += v3;
                q3 += v3 * v3;
            }
        }

        ReadOnlySpan<double> sv = [s0, s1, s2, s3];
        ReadOnlySpan<double> qv = [q0, q1, q2, q3];
        for (int c = 0; c < cn; c++)
        {
            s[c] = sv[c];
            q[c] = squares ? qv[c] : 0;
        }
    }

    private static long MaskedSums<T>(ReadOnlySpan<T> row, ReadOnlySpan<byte> m, int cn, bool squares, Span<double> s, Span<double> q)
        where T : unmanaged
    {
        long n = 0;
        for (int x = 0; x < m.Length; x++)
        {
            if (m[x] == 0)
            {
                continue;
            }

            n++;
            for (int c = 0; c < cn; c++)
            {
                double v = Saturate.ToDouble(row[(x * cn) + c]);
                s[c] += v;
                if (squares)
                {
                    q[c] += v * v;
                }
            }
        }

        return n;
    }

    /// <summary>Exact sum and sum of squares of a byte row (row lengths below 2^16 fit in 32-bit lanes).</summary>
    private static (double Sum, double Squares) ByteSums(ReadOnlySpan<byte> row, bool squares)
    {
        int i = 0;
        ulong total = 0, sq = 0;
        if (Sse2.IsSupported)
        {
            i = ByteSumsSse2(row, squares, out total, out sq);
        }
        else if (Vector.IsHardwareAccelerated)
        {
            var acc = Vector<uint>.Zero;
            var accSq = Vector<uint>.Zero;
            int limit = Math.Min(row.Length, 65536) - Vector<byte>.Count;
            for (; i <= limit; i += Vector<byte>.Count)
            {
                Vector.Widen(new Vector<byte>(row[i..]), out Vector<ushort> lo, out Vector<ushort> hi);
                Vector.Widen(lo + hi, out Vector<uint> a, out Vector<uint> b);
                acc += a + b;
                if (squares)
                {
                    Vector.Widen(lo * lo, out Vector<uint> l0, out Vector<uint> l1);
                    Vector.Widen(hi * hi, out Vector<uint> h0, out Vector<uint> h1);
                    accSq += (l0 + l1) + (h0 + h1);
                }
            }

            total = SumLanes(acc);
            sq = SumLanes(accSq);
        }

        for (; i < row.Length; i++)
        {
            total += row[i];
            sq += (uint)(row[i] * row[i]);
        }

        return (total, squares ? sq : 0);
    }

    /// <summary>
    /// Sums with <c>psadbw</c> against zero; squares with <c>pmaddwd</c>, flushed to 64 bits
    /// every 4096 blocks so the 32-bit lanes cannot overflow.
    /// </summary>
    private static int ByteSumsSse2(ReadOnlySpan<byte> row, bool squares, out ulong total, out ulong sq)
    {
        ref byte r = ref MemoryMarshal.GetReference(row);
        Vector128<ulong> acc = Vector128<ulong>.Zero;
        Vector128<byte> zero = Vector128<byte>.Zero;
        sq = 0;
        int i = 0;
        while (i <= row.Length - 16)
        {
            int end = Math.Min(row.Length - 16, i + (4096 * 16));
            Vector128<int> accSq = Vector128<int>.Zero;
            for (; i <= end; i += 16)
            {
                Vector128<byte> v = Vector128.LoadUnsafe(ref r, (nuint)i);
                acc += Sse2.SumAbsoluteDifferences(v, zero).AsUInt64();
                if (squares)
                {
                    Vector128<short> lo = Sse2.UnpackLow(v, zero).AsInt16();
                    Vector128<short> hi = Sse2.UnpackHigh(v, zero).AsInt16();
                    accSq += Sse2.MultiplyAddAdjacent(lo, lo) + Sse2.MultiplyAddAdjacent(hi, hi);
                }
            }

            sq += SumLanes(accSq);
        }

        total = acc.GetElement(0) + acc.GetElement(1);
        return i;
    }

    private static ulong SumLanes(Vector128<int> v)
        => (ulong)v.AsUInt32().GetElement(0) + v.AsUInt32().GetElement(1) + v.AsUInt32().GetElement(2) + v.AsUInt32().GetElement(3);

    private static ulong SumLanes(Vector<uint> v)
    {
        ulong s = 0;
        for (int k = 0; k < Vector<uint>.Count; k++)
        {
            s += v[k];
        }

        return s;
    }

    public static int CountNonZero<T>(Mat<T> src)
        where T : unmanaged
    {
        var rowCount = new int[src.Height];
        ParallelRows.For(
            src.Height,
            Elementwise.WorkPerRow<T>(src.RowLength),
            (y0, y1) => CountStripe(src, rowCount, y0, y1));

        int total = 0;
        foreach (int n in rowCount)
        {
            total += n;
        }

        return total;
    }

    private static void CountStripe<T>(Mat<T> src, int[] rowCount, int y0, int y1)
        where T : unmanaged
    {
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<T> row = src.GetReadOnlyRowSpan(y);
            int n = 0;
            int i = 0;
            if (typeof(T) == typeof(byte) && Sse2.IsSupported)
            {
                ref byte b = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(row));
                Vector128<ulong> zeros = Vector128<ulong>.Zero;
                Vector128<byte> one = Vector128.Create((byte)1);
                for (; i <= row.Length - 16; i += 16)
                {
                    Vector128<byte> z = Vector128.Equals(Vector128.LoadUnsafe(ref b, (nuint)i), Vector128<byte>.Zero) & one;
                    zeros += Sse2.SumAbsoluteDifferences(z, Vector128<byte>.Zero).AsUInt64();
                }

                n = i - (int)(zeros.GetElement(0) + zeros.GetElement(1));
            }
            else if (typeof(T) == typeof(byte) && Vector128.IsHardwareAccelerated)
            {
                var b = MemoryMarshal.Cast<T, byte>(row);
                for (; i <= b.Length - Vector128<byte>.Count; i += Vector128<byte>.Count)
                {
                    uint zeros = Vector128.Equals(Vector128.Create(b[i..]), Vector128<byte>.Zero).ExtractMostSignificantBits();
                    n += Vector128<byte>.Count - BitOperations.PopCount(zeros);
                }
            }

            for (; i < row.Length; i++)
            {
                if (Saturate.ToDouble(row[i]) != 0)
                {
                    n++;
                }
            }

            rowCount[y] = n;
        }
    }

    /// <summary>
    /// Minimum and maximum of a single-channel matrix: strict comparisons in row-major order
    /// (first occurrence wins), NaN skipped.
    /// </summary>
    public static MinMaxLocResult MinMaxLoc<T>(Mat<T> src, Mat<byte>? mask)
        where T : unmanaged
    {
        int h = src.Height;
        var rows = new (double Min, int MinX, double Max, int MaxX)[h];
        ParallelRows.For(
            h,
            Elementwise.WorkPerRow<T>(src.RowLength),
            (y0, y1) => MinMaxStripe(src, mask, rows, y0, y1));

        double minVal = double.PositiveInfinity, maxVal = double.NegativeInfinity;
        Point minLoc = new(-1, -1), maxLoc = new(-1, -1);
        for (int y = 0; y < h; y++)
        {
            var r = rows[y];
            if (r.MinX >= 0 && (minLoc.X < 0 || r.Min < minVal))
            {
                minVal = r.Min;
                minLoc = new Point(r.MinX, y);
            }

            if (r.MaxX >= 0 && (maxLoc.X < 0 || r.Max > maxVal))
            {
                maxVal = r.Max;
                maxLoc = new Point(r.MaxX, y);
            }
        }

        if (mask is not null)
        {
            if (Reductions.CountNonZero(mask) == 0)
            {
                return new MinMaxLocResult(0, 0, new Point(-1, -1), new Point(-1, -1));
            }

            return new MinMaxLocResult(minVal, maxVal, minLoc, maxLoc);
        }

        return new MinMaxLocResult(
            minVal,
            maxVal,
            minLoc.X < 0 ? new Point(0, 0) : minLoc,
            maxLoc.X < 0 ? new Point(0, 0) : maxLoc);
    }

    private static void MinMaxStripe<T>(Mat<T> src, Mat<byte>? mask, (double, int, double, int)[] rows, int y0, int y1)
        where T : unmanaged
    {
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<T> row = src.GetReadOnlyRowSpan(y);
            if (mask is null && typeof(T) == typeof(byte) && Vector.IsHardwareAccelerated)
            {
                rows[y] = ByteRowMinMax(MemoryMarshal.Cast<T, byte>(row));
                continue;
            }

            if (mask is null && typeof(T) == typeof(float) && Vector.IsHardwareAccelerated)
            {
                rows[y] = FloatRowMinMax(MemoryMarshal.Cast<T, float>(row));
                continue;
            }

            ReadOnlySpan<byte> m = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            double min = double.PositiveInfinity, max = double.NegativeInfinity;
            int minX = -1, maxX = -1;
            for (int x = 0; x < row.Length; x++)
            {
                if (mask is not null && m[x] == 0)
                {
                    continue;
                }

                double v = Saturate.ToDouble(row[x]);
                if (v < min || (minX < 0 && v == min))
                {
                    min = v;
                    minX = x;
                }

                if (v > max || (maxX < 0 && v == max))
                {
                    max = v;
                    maxX = x;
                }
            }

            rows[y] = (min, minX, max, maxX);
        }
    }

    private static (double, int, double, int) ByteRowMinMax(ReadOnlySpan<byte> row)
    {
        int i = 0;
        byte min = 255, max = 0;
        if (row.Length >= Vector<byte>.Count)
        {
            var vmin = new Vector<byte>(255);
            var vmax = Vector<byte>.Zero;
            for (; i <= row.Length - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                var v = new Vector<byte>(row[i..]);
                vmin = Vector.Min(vmin, v);
                vmax = Vector.Max(vmax, v);
            }

            for (int k = 0; k < Vector<byte>.Count; k++)
            {
                min = Math.Min(min, vmin[k]);
                max = Math.Max(max, vmax[k]);
            }
        }

        for (; i < row.Length; i++)
        {
            min = Math.Min(min, row[i]);
            max = Math.Max(max, row[i]);
        }

        return (min, row.IndexOf(min), max, row.IndexOf(max));
    }

    /// <summary>NaN lanes are replaced by +∞ for the minimum and −∞ for the maximum.</summary>
    private static (double, int, double, int) FloatRowMinMax(ReadOnlySpan<float> row)
    {
        int i = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        if (row.Length >= Vector<float>.Count)
        {
            var vmin = new Vector<float>(float.PositiveInfinity);
            var vmax = new Vector<float>(float.NegativeInfinity);
            for (; i <= row.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var v = new Vector<float>(row[i..]);
                Vector<int> ok = Vector.Equals(v, v);
                vmin = Vector.Min(vmin, Vector.ConditionalSelect(ok, v, new Vector<float>(float.PositiveInfinity)));
                vmax = Vector.Max(vmax, Vector.ConditionalSelect(ok, v, new Vector<float>(float.NegativeInfinity)));
            }

            for (int k = 0; k < Vector<float>.Count; k++)
            {
                min = MathF.Min(min, vmin[k]);
                max = MathF.Max(max, vmax[k]);
            }
        }

        for (; i < row.Length; i++)
        {
            float v = row[i];
            if (v < min)
            {
                min = v;
            }

            if (v > max)
            {
                max = v;
            }
        }

        int minX = FirstEqual(row, min);
        int maxX = FirstEqual(row, max);
        return (minX >= 0 ? row[minX] : min, minX, maxX >= 0 ? row[maxX] : max, maxX);
    }

    /// <summary>First index whose value equals <paramref name="value"/> (+0 equals −0).</summary>
    private static int FirstEqual(ReadOnlySpan<float> row, float value)
    {
        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] == value)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>Unscaled norm of one matrix, or of the difference when <paramref name="b"/> is set.</summary>
    public static double Norm<T>(Mat<T> a, Mat<T>? b, int normType, Mat<byte>? mask)
        where T : unmanaged
    {
        var rows = new double[a.Height];
        ParallelRows.For(
            a.Height,
            Elementwise.WorkPerRow<T>(a.RowLength),
            (y0, y1) => NormStripe(a, b, normType, mask, rows, y0, y1));

        double result = 0;
        foreach (double r in rows)
        {
            if (normType == 1)
            {
                if (r > result)
                {
                    result = r;
                }
            }
            else
            {
                result += r;
            }
        }

        return result;
    }

    private static void NormStripe<T>(Mat<T> a, Mat<T>? b, int normType, Mat<byte>? mask, double[] rows, int y0, int y1)
        where T : unmanaged
    {
        int cn = a.Channels;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<T> ra = a.GetReadOnlyRowSpan(y);
            ReadOnlySpan<T> rb = b is null ? default : b.GetReadOnlyRowSpan(y);
            if (mask is null && Vector.IsHardwareAccelerated)
            {
                if (typeof(T) == typeof(byte))
                {
                    rows[y] = ByteNorm(MemoryMarshal.Cast<T, byte>(ra), MemoryMarshal.Cast<T, byte>(rb), b is not null, normType);
                    continue;
                }

                if (typeof(T) == typeof(float))
                {
                    rows[y] = FloatNorm(MemoryMarshal.Cast<T, float>(ra), MemoryMarshal.Cast<T, float>(rb), b is not null, normType);
                    continue;
                }
            }

            ReadOnlySpan<byte> m = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            double acc = 0;
            for (int i = 0; i < ra.Length; i++)
            {
                if (mask is not null && m[i / cn] == 0)
                {
                    continue;
                }

                double v = b is null ? Abs(ra[i]) : AbsDiff(ra[i], rb[i]);
                switch (normType)
                {
                    case 1:
                        if (v < 0)
                        {
                            // The wrapped int difference int.MinValue; the INF norm reads it as unsigned.
                            v = 2147483648.0;
                        }

                        if (v > acc)
                        {
                            acc = v;
                        }

                        break;
                    case 2:
                        acc += v;
                        break;
                    default:
                        // The unmasked float kernel squares the float |a - b|; the masked one subtracts in double.
                        double d = b is null
                            ? Saturate.ToDouble(ra[i])
                            : typeof(T) == typeof(float) && mask is null
                                ? v
                                : Saturate.ToDouble(ra[i]) - Saturate.ToDouble(rb[i]);
                        acc += d * d;
                        break;
                }
            }

            rows[y] = acc;
        }
    }

    /// <summary>Exact byte norms; partial sums fit 32-bit lanes for rows below 2^15 elements.</summary>
    private static double ByteNorm(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, bool diff, int normType)
    {
        if (Sse2.IsSupported && normType != 1)
        {
            return ByteNormSse2(a, b, diff, normType);
        }

        int i = 0;
        ulong acc = 0;
        int limit = Math.Min(a.Length, 32768) - Vector<byte>.Count;
        var vacc = Vector<uint>.Zero;
        var vmax = Vector<byte>.Zero;
        for (; i <= limit; i += Vector<byte>.Count)
        {
            var v = new Vector<byte>(a[i..]);
            if (diff)
            {
                var w = new Vector<byte>(b[i..]);
                v = Vector.Max(v, w) - Vector.Min(v, w);
            }

            if (normType == 1)
            {
                vmax = Vector.Max(vmax, v);
                continue;
            }

            Vector.Widen(v, out Vector<ushort> lo, out Vector<ushort> hi);
            if (normType == 2)
            {
                Vector.Widen(lo + hi, out Vector<uint> p, out Vector<uint> q);
                vacc += p + q;
            }
            else
            {
                Vector.Widen(lo * lo, out Vector<uint> l0, out Vector<uint> l1);
                Vector.Widen(hi * hi, out Vector<uint> h0, out Vector<uint> h1);
                vacc += (l0 + l1) + (h0 + h1);
            }
        }

        uint max = 0;
        for (int k = 0; k < Vector<byte>.Count; k++)
        {
            max = Math.Max(max, vmax[k]);
        }

        acc = SumLanes(vacc);
        for (; i < a.Length; i++)
        {
            uint v = diff ? (uint)Math.Abs(a[i] - b[i]) : a[i];
            max = Math.Max(max, v);
            acc += normType == 2 ? v : v * v;
        }

        return normType == 1 ? max : acc;
    }

    /// <summary>L1 with <c>psadbw</c> (on both rows for differences); L2 with <c>pmaddwd</c>.</summary>
    private static double ByteNormSse2(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, bool diff, int normType)
    {
        ref byte ra = ref MemoryMarshal.GetReference(a);
        ref byte rb = ref MemoryMarshal.GetReference(b);
        Vector128<byte> zero = Vector128<byte>.Zero;
        Vector128<ulong> acc = Vector128<ulong>.Zero;
        ulong sq = 0;
        int i = 0;
        while (i <= a.Length - 16)
        {
            int end = Math.Min(a.Length - 16, i + (4096 * 16));
            Vector128<int> accSq = Vector128<int>.Zero;
            for (; i <= end; i += 16)
            {
                Vector128<byte> v = Vector128.LoadUnsafe(ref ra, (nuint)i);
                Vector128<byte> w = diff ? Vector128.LoadUnsafe(ref rb, (nuint)i) : zero;
                if (normType == 2)
                {
                    acc += Sse2.SumAbsoluteDifferences(v, w).AsUInt64();
                    continue;
                }

                Vector128<byte> dv = Vector128.Max(v, w) - Vector128.Min(v, w);
                Vector128<short> lo = Sse2.UnpackLow(dv, zero).AsInt16();
                Vector128<short> hi = Sse2.UnpackHigh(dv, zero).AsInt16();
                accSq += Sse2.MultiplyAddAdjacent(lo, lo) + Sse2.MultiplyAddAdjacent(hi, hi);
            }

            sq += SumLanes(accSq);
        }

        ulong total = normType == 2 ? acc.GetElement(0) + acc.GetElement(1) : sq;
        for (; i < a.Length; i++)
        {
            uint v = diff ? (uint)Math.Abs(a[i] - b[i]) : a[i];
            total += normType == 2 ? v : v * v;
        }

        return total;
    }

    /// <summary>
    /// Float norms in double. The infinity norm ignores NaN; the L2 difference squares the
    /// float <c>|a - b|</c>.
    /// </summary>
    private static double FloatNorm(ReadOnlySpan<float> a, ReadOnlySpan<float> b, bool diff, int normType)
    {
        int i = 0;
        var acc0 = Vector<double>.Zero;
        var acc1 = Vector<double>.Zero;
        var vmax = Vector<float>.Zero;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var v = new Vector<float>(a[i..]);
            if (diff)
            {
                v -= new Vector<float>(b[i..]);
            }

            v = Vector.Abs(v);
            if (normType == 1)
            {
                vmax = Vector.Max(vmax, Vector.ConditionalSelect(Vector.Equals(v, v), v, Vector<float>.Zero));
                continue;
            }

            Vector.Widen(v, out Vector<double> d0, out Vector<double> d1);
            if (normType == 2)
            {
                acc0 += d0;
                acc1 += d1;
            }
            else
            {
                acc0 += d0 * d0;
                acc1 += d1 * d1;
            }
        }

        double acc = Vector.Sum(acc0 + acc1);
        float max = 0;
        for (int k = 0; k < Vector<float>.Count; k++)
        {
            max = MathF.Max(max, vmax[k]);
        }

        for (; i < a.Length; i++)
        {
            float v = MathF.Abs(diff ? a[i] - b[i] : a[i]);
            if (normType == 1)
            {
                if (v > max)
                {
                    max = v;
                }
            }
            else
            {
                acc += normType == 2 ? v : (double)v * v;
            }
        }

        return normType == 1 ? max : acc;
    }

    /// <summary><c>cv_abs</c>: <c>int.MinValue</c> gives 2147483648.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Abs<T>(T v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return MathF.Abs(Saturate.ToFloat(v));
        }

        return Math.Abs(Saturate.ToDouble(v));
    }

    /// <summary>
    /// <c>cv_absdiff</c>: exact for 8/16-bit, <c>abs</c> of the wrapped difference for
    /// <see cref="int"/>, and <c>|a - b|</c> in the element type for floating types.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double AbsDiff<T>(T a, T b)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return MathF.Abs(Saturate.ToFloat(a) - Saturate.ToFloat(b));
        }

        if (typeof(T) == typeof(int))
        {
            return Saturate.ToInt(ElementMath.AbsDiff(a, b));
        }

        return Math.Abs(Saturate.ToDouble(a) - Saturate.ToDouble(b));
    }
}
