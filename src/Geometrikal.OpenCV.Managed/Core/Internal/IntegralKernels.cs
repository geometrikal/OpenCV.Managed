// This implementation is ported from the OpenCV integral image functions.
//
// Original sources:
//   modules/imgproc/src/sumpixels.dispatch.cpp (hal::integral_, integral)
//   modules/imgproc/src/sumpixels.simd.hpp (Integral_SIMD)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Integral images. Every output element is <c>out[y + 1][x] = out[y][x] + prefix_y[x]</c>, where
/// <c>prefix_y</c> is the running row sum of OpenCV's scalar loop. The row prefix and the vertical
/// addition are separate passes; each element sees the same operations as in OpenCV.
/// </summary>
/// <remarks>
/// <para>Integer row prefixes are computed exactly (in <c>int</c> with SIMD, or in <c>long</c>) and
/// converted once. This equals the sequential OpenCV sum when every partial sum is exact in the
/// output type: always for <c>int</c> (wrapping) and <c>double</c>, and below 2^24 for
/// <c>float</c>. Other prefixes run the sequential float or double loop.</para>
/// <para>Rows are independent in the prefix pass and columns are independent in the vertical
/// pass. Large images with a sequential float prefix run both passes in parallel; other images
/// fuse the passes per row on one thread. The tilted sum is a direct transcription and runs on one
/// thread.</para>
/// </remarks>
internal static class IntegralKernels
{
    private const int ColumnChunk = 512;
    private const int MaxLongChannels = 64;

    /// <summary>Selects the 16-lane byte prefix. Tests clear it to run the 4-lane path on AVX-512 hardware.</summary>
    internal static bool UseAvx512 = Avx512F.IsSupported && Vector512.IsHardwareAccelerated;

    private enum PrefixKind
    {
        Int32Simd,
        Long,
        Sequential,
    }

    public static void Run<TSrc, TSum, TSq>(Mat<TSrc> src, Mat<TSum> sum, Mat<TSq>? sqsum, Mat<TSum>? tilted)
        where TSrc : unmanaged, INumber<TSrc>
        where TSum : unmanaged, INumber<TSum>
        where TSq : unmanaged, INumber<TSq>
    {
        if (tilted is not null)
        {
            Tilted(src, sum, sqsum, tilted);
            return;
        }

        int cn = src.Channels;
        int rowLength = src.RowLength;
        PrefixKind sumKind = Kind<TSrc, TSum>(src.Width, squares: false, cn);
        PrefixKind sqKind = Kind<TSrc, TSq>(src.Width, squares: true, cn);
        sum.GetRowSpan(0).Clear();
        sqsum?.GetRowSpan(0).Clear();

        long elements = (long)rowLength * src.Height;
        // Exact integer prefixes are cheap; the vertical pass is memory-bound and a second pass
        // over the output costs more than threads save. Only the sequential float prefix gains.
        bool sequential = sumKind == PrefixKind.Sequential || (sqsum is not null && sqKind == PrefixKind.Sequential);
        bool parallel = sequential && CvParallel.MaxDegreeOfParallelism > 1 && elements >= 1 << 20;
        if (!parallel)
        {
            Rows(src, sum, sqsum, sumKind, sqKind, 0, src.Height, vertical: true);
            return;
        }

        ParallelRows.For(src.Height, rowLength, (y0, y1) => Rows(src, sum, sqsum, sumKind, sqKind, y0, y1, vertical: false));
        int outLength = sum.RowLength;
        int chunks = (outLength + ColumnChunk - 1) / ColumnChunk;
        ParallelRows.For(
            chunks,
            (long)src.Height * ColumnChunk / Math.Max(1, Vector<TSum>.Count),
            (c0, c1) => VerticalChunks(sum, sqsum, c0, c1));
    }

    /// <summary>Selects how the row prefix of <typeparamref name="TOut"/> is computed exactly.</summary>
    private static PrefixKind Kind<TSrc, TOut>(int width, bool squares, int cn)
        where TSrc : unmanaged
        where TOut : unmanaged
    {
        double max = typeof(TSrc) == typeof(byte) ? 255
            : typeof(TSrc) == typeof(ushort) ? 65535
            : typeof(TSrc) == typeof(short) ? 32768
            : -1;
        if (max < 0 || cn > MaxLongChannels)
        {
            return PrefixKind.Sequential;
        }

        double rowMax = (squares ? max * max : max) * width;
        bool exactInOut = typeof(TOut) == typeof(int)
            || (typeof(TOut) == typeof(float) && rowMax <= 16777216.0)
            || (typeof(TOut) == typeof(double) && rowMax <= 9007199254740992.0);
        if (!exactInOut)
        {
            return PrefixKind.Sequential;
        }

        bool fitsInt = typeof(TOut) == typeof(int) || rowMax <= int.MaxValue;
        return typeof(TSrc) == typeof(byte) && cn == 1 && fitsInt && Sse41.IsSupported
            ? PrefixKind.Int32Simd
            : PrefixKind.Long;
    }

    private static void Rows<TSrc, TSum, TSq>(
        Mat<TSrc> src, Mat<TSum> sum, Mat<TSq>? sqsum, PrefixKind sumKind, PrefixKind sqKind, int y0, int y1, bool vertical)
        where TSrc : unmanaged, INumber<TSrc>
        where TSum : unmanaged, INumber<TSum>
        where TSq : unmanaged, INumber<TSq>
    {
        int cn = src.Channels;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<TSrc> s = src.GetReadOnlyRowSpan(y);
            Row(s, sum.GetRowSpan(y + 1), vertical ? sum.GetReadOnlyRowSpan(y) : default, cn, squares: false, sumKind);
            if (sqsum is not null)
            {
                Row(s, sqsum.GetRowSpan(y + 1), vertical ? sqsum.GetReadOnlyRowSpan(y) : default, cn, squares: true, sqKind);
            }
        }
    }

    /// <summary>
    /// Writes the zero column and the running row sum of each channel to <paramref name="o"/>
    /// (length <c>(width + 1) * cn</c>). A non-empty <paramref name="above"/> also adds the row
    /// above: <c>o[i] = above[i] + prefix[i]</c>.
    /// </summary>
    private static void Row<TSrc, TOut>(
        ReadOnlySpan<TSrc> s, Span<TOut> o, ReadOnlySpan<TOut> above, int cn, bool squares, PrefixKind kind)
        where TSrc : unmanaged, INumber<TSrc>
        where TOut : unmanaged, INumber<TOut>
    {
        int n = s.Length;
        ref TSrc sr = ref MemoryMarshal.GetReference(s);
        ref TOut or = ref MemoryMarshal.GetReference(o);
        for (int c = 0; c < cn; c++)
        {
            Unsafe.Add(ref or, c) = TOut.Zero;
        }

        ref TOut dst = ref Unsafe.Add(ref or, cn);
        bool fused = !above.IsEmpty;
        switch (kind)
        {
            case PrefixKind.Int32Simd:
                PrefixBytes(
                    ref Unsafe.As<TSrc, byte>(ref sr), n, squares, ref dst,
                    ref fused ? ref Unsafe.Add(ref MemoryMarshal.GetReference(above), cn) : ref dst, fused);
                return;

            case PrefixKind.Long:
                if (cn == 1)
                {
                    long acc = 0;
                    for (int i = 0; i < n; i++)
                    {
                        long v = long.CreateTruncating(Unsafe.Add(ref sr, i));
                        acc += squares ? v * v : v;
                        Unsafe.Add(ref dst, i) = TOut.CreateTruncating(acc);
                    }
                }
                else
                {
                    Span<long> accs = stackalloc long[MaxLongChannels];
                    accs.Clear();
                    for (int i = 0, c = 0; i < n; i++, c = c + 1 == cn ? 0 : c + 1)
                    {
                        long v = long.CreateTruncating(Unsafe.Add(ref sr, i));
                        accs[c] += squares ? v * v : v;
                        Unsafe.Add(ref dst, i) = TOut.CreateTruncating(accs[c]);
                    }
                }

                break;

            default:
            {
                // dst[i - cn] reads the zero column for the first pixel.
                ref TOut prev = ref or;
                for (int i = 0; i < n; i++)
                {
                    TOut v = TOut.CreateTruncating(Unsafe.Add(ref sr, i));
                    Unsafe.Add(ref dst, i) = Unsafe.Add(ref prev, i) + (squares ? v * v : v);
                }

                break;
            }
        }

        if (fused)
        {
            VerticalAdd(above[cn..], o[cn..]);
        }
    }

    /// <summary>
    /// Running sum of bytes (or of their squares) in <c>int</c>, converted to
    /// <typeparamref name="TOut"/> and, if <paramref name="fused"/>, added to the row above. Per 4
    /// lanes, two shifted additions form the in-register prefix, then the previous total is added.
    /// Integer addition is exact, with wrap as in OpenCV; the conversion is exact by the choice of
    /// <see cref="PrefixKind.Int32Simd"/>.
    /// </summary>
    private static void PrefixBytes<TOut>(ref byte s, int n, bool squares, ref TOut dst, ref TOut above, bool fused)
        where TOut : unmanaged, INumber<TOut>
    {
        int i = 0;
        int acc;
        if (UseAvx512)
        {
            Vector512<int> run = Vector512<int>.Zero;
            for (; i <= n - 16; i += 16)
            {
                run = Block512(Avx512F.ConvertToVector512Int32(Vector128.LoadUnsafe(ref s, (nuint)i)), run, squares, ref dst, ref above, fused, i);
            }

            acc = run.ToScalar();
        }
        else
        {
            Vector128<int> run = Vector128<int>.Zero;
            for (; i <= n - 16; i += 16)
            {
                Vector128<byte> b = Vector128.LoadUnsafe(ref s, (nuint)i);
                run = Block(Sse41.ConvertToVector128Int32(b), run, squares, ref dst, ref above, fused, i);
                run = Block(Sse41.ConvertToVector128Int32(Sse2.ShiftRightLogical128BitLane(b, 4)), run, squares, ref dst, ref above, fused, i + 4);
                run = Block(Sse41.ConvertToVector128Int32(Sse2.ShiftRightLogical128BitLane(b, 8)), run, squares, ref dst, ref above, fused, i + 8);
                run = Block(Sse41.ConvertToVector128Int32(Sse2.ShiftRightLogical128BitLane(b, 12)), run, squares, ref dst, ref above, fused, i + 12);
            }

            acc = run.ToScalar();
        }

        for (; i < n; i++)
        {
            int v = Unsafe.Add(ref s, i);
            acc += squares ? v * v : v;
            TOut t = TOut.CreateTruncating(acc);
            Unsafe.Add(ref dst, i) = fused ? Unsafe.Add(ref above, i) + t : t;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Block<TOut>(
        Vector128<int> v, Vector128<int> run, bool squares, ref TOut dst, ref TOut above, bool fused, int i)
        where TOut : unmanaged
    {
        if (squares)
        {
            v = Sse41.MultiplyLow(v, v);
        }

        v += Sse2.ShiftLeftLogical128BitLane(v, 4);
        v += Sse2.ShiftLeftLogical128BitLane(v, 8);
        v += run;
        if (typeof(TOut) == typeof(int))
        {
            ref int d = ref Unsafe.As<TOut, int>(ref dst);
            (fused ? Vector128.LoadUnsafe(ref Unsafe.As<TOut, int>(ref above), (nuint)i) + v : v).StoreUnsafe(ref d, (nuint)i);
        }
        else if (typeof(TOut) == typeof(float))
        {
            ref float d = ref Unsafe.As<TOut, float>(ref dst);
            Vector128<float> f = Sse2.ConvertToVector128Single(v);
            (fused ? Vector128.LoadUnsafe(ref Unsafe.As<TOut, float>(ref above), (nuint)i) + f : f).StoreUnsafe(ref d, (nuint)i);
        }
        else
        {
            ref double d = ref Unsafe.As<TOut, double>(ref dst);
            ref double a = ref Unsafe.As<TOut, double>(ref above);
            Vector128<double> lo = Sse2.ConvertToVector128Double(v);
            Vector128<double> hi = Sse2.ConvertToVector128Double(Sse2.Shuffle(v, 0xEE));
            (fused ? Vector128.LoadUnsafe(ref a, (nuint)i) + lo : lo).StoreUnsafe(ref d, (nuint)i);
            (fused ? Vector128.LoadUnsafe(ref a, (nuint)(i + 2)) + hi : hi).StoreUnsafe(ref d, (nuint)(i + 2));
        }

        return Sse2.Shuffle(v, 0xFF);
    }

    /// <summary>
    /// <see cref="Block{TOut}"/> for 16 lanes: four shifted additions (<c>valignd</c>) form the
    /// in-register prefix. Returns the last prefix in every lane.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<int> Block512<TOut>(
        Vector512<int> v, Vector512<int> run, bool squares, ref TOut dst, ref TOut above, bool fused, int i)
        where TOut : unmanaged
    {
        if (squares)
        {
            v *= v;
        }

        Vector512<int> zero = Vector512<int>.Zero;
        v += Avx512F.AlignRight32(v, zero, 15);
        v += Avx512F.AlignRight32(v, zero, 14);
        v += Avx512F.AlignRight32(v, zero, 12);
        v += Avx512F.AlignRight32(v, zero, 8);
        v += run;
        if (typeof(TOut) == typeof(int))
        {
            ref int d = ref Unsafe.As<TOut, int>(ref dst);
            (fused ? Vector512.LoadUnsafe(ref Unsafe.As<TOut, int>(ref above), (nuint)i) + v : v).StoreUnsafe(ref d, (nuint)i);
        }
        else if (typeof(TOut) == typeof(float))
        {
            ref float d = ref Unsafe.As<TOut, float>(ref dst);
            Vector512<float> f = Vector512.ConvertToSingle(v);
            (fused ? Vector512.LoadUnsafe(ref Unsafe.As<TOut, float>(ref above), (nuint)i) + f : f).StoreUnsafe(ref d, (nuint)i);
        }
        else
        {
            ref double d = ref Unsafe.As<TOut, double>(ref dst);
            ref double a = ref Unsafe.As<TOut, double>(ref above);
            Vector512<double> lo = Avx512F.ConvertToVector512Double(v.GetLower());
            Vector512<double> hi = Avx512F.ConvertToVector512Double(v.GetUpper());
            (fused ? Vector512.LoadUnsafe(ref a, (nuint)i) + lo : lo).StoreUnsafe(ref d, (nuint)i);
            (fused ? Vector512.LoadUnsafe(ref a, (nuint)(i + 8)) + hi : hi).StoreUnsafe(ref d, (nuint)(i + 8));
        }

        return Avx512F.PermuteVar16x32(v, Vector512.Create(15));
    }

    /// <summary><c>row[i] = above[i] + row[i]</c>; each lane performs the scalar addition.</summary>
    private static void VerticalAdd<T>(ReadOnlySpan<T> above, Span<T> row)
        where T : unmanaged, INumber<T>
    {
        int n = row.Length;
        ref T a = ref MemoryMarshal.GetReference(above);
        ref T r = ref MemoryMarshal.GetReference(row);
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<T>.Count;
            for (; i <= n - vc; i += vc)
            {
                (Vector.LoadUnsafe(ref a, (nuint)i) + Vector.LoadUnsafe(ref r, (nuint)i)).StoreUnsafe(ref r, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref r, i) = Unsafe.Add(ref a, i) + Unsafe.Add(ref r, i);
        }
    }

    private static void VerticalChunks<TSum, TSq>(Mat<TSum> sum, Mat<TSq>? sqsum, int c0, int c1)
        where TSum : unmanaged, INumber<TSum>
        where TSq : unmanaged, INumber<TSq>
    {
        int start = c0 * ColumnChunk;
        int end = Math.Min(c1 * ColumnChunk, sum.RowLength);
        for (int y = 1; y < sum.Height; y++)
        {
            VerticalAdd(sum.GetReadOnlyRowSpan(y - 1)[start..end], sum.GetRowSpan(y)[start..end]);
            if (sqsum is not null)
            {
                VerticalAdd(sqsum.GetReadOnlyRowSpan(y - 1)[start..end], sqsum.GetRowSpan(y)[start..end]);
            }
        }
    }

    /// <summary>Transcription of the tilted branch of <c>hal::integral_</c>.</summary>
    private static void Tilted<TSrc, TSum, TSq>(Mat<TSrc> src, Mat<TSum> sum, Mat<TSq>? sqsum, Mat<TSum> tilted)
        where TSrc : unmanaged, INumber<TSrc>
        where TSum : unmanaged, INumber<TSum>
        where TSq : unmanaged, INumber<TSq>
    {
        int cn = src.Channels;
        int width = src.RowLength;
        int height = src.Height;
        sum.GetRowSpan(0).Clear();
        tilted.GetRowSpan(0).Clear();
        sqsum?.GetRowSpan(0).Clear();
        bool sq = sqsum is not null;
        TSum[] bufArray = ArrayPool<TSum>.Shared.Rent(width + cn);
        try
        {
            Span<TSum> buf = bufArray.AsSpan(0, width + cn);
            {
                ReadOnlySpan<TSrc> s0 = src.GetReadOnlyRowSpan(0);
                Span<TSum> sumRow = sum.GetRowSpan(1);
                Span<TSum> tRow = tilted.GetRowSpan(1);
                Span<TSq> qRow = sq ? sqsum!.GetRowSpan(1) : default;
                for (int k = 0; k < cn; k++)
                {
                    sumRow[k] = TSum.Zero;
                    tRow[k] = TSum.Zero;
                    TSum s = TSum.Zero;
                    TSq q = TSq.Zero;
                    for (int x = 0; x < width; x += cn)
                    {
                        TSrc it = s0[x + k];
                        TSum t = TSum.CreateTruncating(it);
                        tRow[cn + x + k] = t;
                        buf[x + k] = t;
                        s += t;
                        TSq iq = TSq.CreateTruncating(it);
                        q += iq * iq;
                        sumRow[cn + x + k] = s;
                        if (sq)
                        {
                            qRow[cn + x + k] = q;
                        }
                    }

                    if (width == cn)
                    {
                        buf[cn + k] = TSum.Zero;
                    }

                    if (sq)
                    {
                        qRow[k] = TSq.Zero;
                    }
                }
            }

            for (int y = 1; y < height; y++)
            {
                ReadOnlySpan<TSrc> sr = src.GetReadOnlyRowSpan(y);
                ReadOnlySpan<TSum> sumUp = sum.GetReadOnlyRowSpan(y);
                Span<TSum> sumRow = sum.GetRowSpan(y + 1);
                ReadOnlySpan<TSum> tUp = tilted.GetReadOnlyRowSpan(y);
                Span<TSum> tRow = tilted.GetRowSpan(y + 1);
                ReadOnlySpan<TSq> qUp = sq ? sqsum!.GetReadOnlyRowSpan(y) : default;
                Span<TSq> qRow = sq ? sqsum!.GetRowSpan(y + 1) : default;
                for (int k = 0; k < cn; k++)
                {
                    TSrc it = sr[k];
                    TSum t0 = TSum.CreateTruncating(it);
                    TSum s = t0;
                    TSq iq = TSq.CreateTruncating(it);
                    TSq tq0 = iq * iq;
                    TSq q = tq0;

                    sumRow[k] = TSum.Zero;
                    if (sq)
                    {
                        qRow[k] = TSq.Zero;
                    }

                    tRow[k] = tUp[cn + k];
                    sumRow[cn + k] = sumUp[cn + k] + t0;
                    if (sq)
                    {
                        qRow[cn + k] = qUp[cn + k] + tq0;
                    }

                    tRow[cn + k] = tUp[cn + k] + t0 + buf[cn + k];

                    int x = cn;
                    for (; x < width - cn; x += cn)
                    {
                        TSum t1 = buf[x + k];
                        buf[x - cn + k] = t1 + t0;
                        it = sr[x + k];
                        t0 = TSum.CreateTruncating(it);
                        iq = TSq.CreateTruncating(it);
                        tq0 = iq * iq;
                        s += t0;
                        q += tq0;
                        sumRow[cn + x + k] = sumUp[cn + x + k] + s;
                        if (sq)
                        {
                            qRow[cn + x + k] = qUp[cn + x + k] + q;
                        }

                        t1 += buf[x + cn + k] + t0 + tUp[x + k];
                        tRow[cn + x + k] = t1;
                    }

                    if (width > cn)
                    {
                        TSum t1 = buf[x + k];
                        buf[x - cn + k] = t1 + t0;
                        it = sr[x + k];
                        t0 = TSum.CreateTruncating(it);
                        iq = TSq.CreateTruncating(it);
                        tq0 = iq * iq;
                        s += t0;
                        q += tq0;
                        sumRow[cn + x + k] = sumUp[cn + x + k] + s;
                        if (sq)
                        {
                            qRow[cn + x + k] = qUp[cn + x + k] + q;
                        }

                        tRow[cn + x + k] = t0 + t1 + tUp[x + k];
                        buf[x + k] = t0;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<TSum>.Shared.Return(bufArray);
        }
    }
}
