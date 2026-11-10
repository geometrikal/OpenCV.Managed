// This implementation is ported from the OpenCV box filter and squared box filter.
//
// Original sources:
//   modules/imgproc/src/box_filter.dispatch.cpp (boxFilter, blur, sqrBoxFilter)
//   modules/imgproc/src/box_filter.simd.hpp (RowSum, SqrRowSum, ColumnSum,
//     getRowSumFilter, getColumnSumFilter, createBoxFilter, BlockSum)
//
// The sum-type ladder follows createBoxFilter: byte→byte with kernel area
// ≤ 256 accumulates in ushort, other byte-source combinations in int32
// (double for normalized areas beyond 2^23), float sources in double.
// sqrBoxFilter accumulates squares in int32 for byte sources and in double
// otherwise. OpenCV 5's BlockSum fast path (float, ksize ≤ 5) is
// arithmetic-identical to the RowSum/ColumnSum recurrence implemented here.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Box filter and squared box filter replicating OpenCV's RowSum/SqrRowSum/ColumnSum pipeline.
/// </summary>
/// <remarks>
/// The int32 column sums of OpenCV scale with a float32 factor in the vector body and with a
/// double factor in the scalar tail. The vector body covers the first <c>8·⌊n/8⌋</c> row
/// elements for byte and short outputs and the first <c>4·⌊n/4⌋</c> elements for float and int
/// outputs, for every SIMD width. The port applies the same split.
/// </remarks>
internal static class BoxFilterEngine
{
    private const int FixedShift = 23;

    private const int ColumnBlock = 32;

    /// <summary>Fewest stripes for which the two-pass parallel double-sum box filter is faster than the fused form.</summary>
    private const int MinBoxStripes = 4;

    public static Mat<TDst> Apply<TSrc, TDst>(
        Mat<TSrc> source,
        Size ksize,
        Point anchor,
        bool normalize,
        BorderType borderType)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<TDst>(source.Width, source.Height, source.Channels, clear: false);
        ApplyInto(source, ksize, anchor, normalize, borderType, dst);
        return dst;
    }

    /// <summary>
    /// <see cref="Apply{TSrc, TDst}"/> into <paramref name="dst"/>, which has the source size and
    /// channel count and does not share memory with the source.
    /// </summary>
    public static void ApplyInto<TSrc, TDst>(
        Mat<TSrc> source,
        Size ksize,
        Point anchor,
        bool normalize,
        BorderType borderType,
        Mat<TDst> dst)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        (int kw, int kh, int ax, int ay, double scale) = Prologue(source, ksize, anchor, normalize, borderType, normalize);
        long area = (long)kw * kh;

        if (typeof(TSrc) == typeof(byte)
            && (typeof(TDst) == typeof(byte) || typeof(TDst) == typeof(short) || typeof(TDst) == typeof(float)))
        {
            var src = (Mat<byte>)(object)source;
            if (typeof(TDst) == typeof(byte) && area <= 256)
            {
                using PooledMat<byte> pooled = Borders.CopyMakeBorderPooled(
                    src, ay, kh - ay - 1, ax, kw - ax - 1, borderType, default);
                BoxUInt16ToByte(pooled.Mat, (Mat<byte>)(object)dst, kw, kh, scale);
                return;
            }

            if (!normalize || area <= 1 << 23)
            {
                IntSums(src, dst, kw, kh, ax, ay, scale, borderType, square: false);
                return;
            }

            DoubleSums(src, null, dst, kw, kh, ax, ay, scale, borderType, RowKind.Sum);
            return;
        }

        if (typeof(TSrc) == typeof(float) && typeof(TDst) == typeof(float))
        {
            var src = (Mat<float>)(object)source;
            DoubleSums(src, null, dst, kw, kh, ax, ay, scale, borderType, kw <= 5 && kh <= 5 ? RowKind.Block : RowKind.Sum);
            return;
        }

        throw new NotSupportedException(
            $"Unsupported box filter type combination ({typeof(TSrc).Name} → {typeof(TDst).Name}). " +
            "Supported: byte→byte, byte→short, byte→float, float→float.");
    }

    public static Mat<TDst> ApplySqr<TSrc, TDst>(
        Mat<TSrc> source,
        Size ksize,
        Point anchor,
        bool normalize,
        BorderType borderType)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        if (typeof(TSrc) == typeof(byte)
            && (typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double) || typeof(TDst) == typeof(int)))
        {
            (int kw, int kh, int ax, int ay, double scale) = Prologue(source, ksize, anchor, normalize && borderType != BorderType.Constant, borderType, normalize);
            var dst = new Mat<TDst>(source.Width, source.Height, source.Channels, clear: false);
            IntSums((Mat<byte>)(object)source, dst, kw, kh, ax, ay, scale, borderType, square: true);
            return dst;
        }

        if (typeof(TSrc) == typeof(float) && (typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double)))
        {
            (int kw, int kh, int ax, int ay, double scale) = Prologue(source, ksize, anchor, normalize && borderType != BorderType.Constant, borderType, normalize);
            var src = (Mat<float>)(object)source;
            var dst = new Mat<TDst>(src.Width, src.Height, src.Channels, clear: false);
            DoubleSums(src, null, dst, kw, kh, ax, ay, scale, borderType, RowKind.Square);
            return dst;
        }

        ArgumentNullException.ThrowIfNull(source);
        throw new NotSupportedException(
            $"Unsupported sqrBoxFilter type combination ({typeof(TSrc).Name} → {typeof(TDst).Name}). " +
            "Supported: byte→float, byte→double, byte→int, float→float, float→double.");
    }

    /// <summary>
    /// <c>boxFilter(src·factor, dst, CV_32F, (k, k), (-1, -1), normalize=true, borderType)</c> for
    /// single-channel float planes of equal size. A null <paramref name="factor"/> filters
    /// <paramref name="src"/>; otherwise each input element is the float product
    /// <c>src·factor</c>. The border type carries no <c>BORDER_ISOLATED</c> bit, so OpenCV 5
    /// does not shrink the kernel for single-row or single-column planes.
    /// </summary>
    public static void FloatBox(Mat<float> src, Mat<float>? factor, Mat<float> dst, int k, BorderType borderType)
        => DoubleSums(src, factor, dst, k, k, k / 2, k / 2, 1.0 / ((double)k * k), borderType, k <= 5 ? RowKind.Block : RowKind.Sum);

    /// <summary>
    /// Runs <see cref="FloatBox"/> for every job. Images whose single box filter splits into fewer
    /// than <see cref="MinBoxStripes"/> stripes run whole filters in parallel instead; each filter
    /// is then the single-threaded form, so the result is the same.
    /// </summary>
    public static void FloatBoxes((Mat<float> Src, Mat<float>? Factor, Mat<float> Dst)[] jobs, int k, BorderType borderType)
    {
        Mat<float> first = jobs[0].Src;
        if (jobs.Length == 1 || ParallelRows.Stripes(first.Height + k - 1, DoubleRowWork(first.RowLength)) >= MinBoxStripes)
        {
            foreach ((Mat<float> src, Mat<float>? factor, Mat<float> dst) in jobs)
            {
                FloatBox(src, factor, dst, k, borderType);
            }

            return;
        }

        ParallelRows.ForEachSerial(jobs.Length, i => FloatBox(jobs[i].Src, jobs[i].Factor, jobs[i].Dst, k, borderType));
    }

    /// <summary>Row sum rule of the double-sum paths.</summary>
    private enum RowKind
    {
        /// <summary>RowSum: direct sums for ksize 3 and 5, running sums otherwise.</summary>
        Sum,

        /// <summary>SqrRowSum: running sums of squares.</summary>
        Square,

        /// <summary>
        /// BlockSum (OpenCV 5, same-type float filters with both kernel sides ≤ 5): direct sums
        /// for every kernel width; widths 1, 2 and 4 start from 0.
        /// </summary>
        Block,
    }

    private static (int Kw, int Kh, int Ax, int Ay, double Scale) Prologue<TSrc>(
        Mat<TSrc> source, Size ksize, Point anchor, bool shrink, BorderType borderType, bool normalize)
        where TSrc : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (ksize.Width <= 0 || ksize.Height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ksize), ksize, "The kernel size must be positive.");
        }

        if (borderType == BorderType.Wrap)
        {
            throw new NotSupportedException(
                "BORDER_WRAP is not supported by OpenCV's filter engine and is likewise rejected here.");
        }

        if (borderType == BorderType.Transparent)
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "BorderType.Transparent is not supported by filters.");
        }

        // boxFilter tests `borderType != BORDER_CONSTANT` with the BORDER_ISOLATED bit still set, so
        // for isolated matrices (this port's model) its collapse applies to every border type.
        // sqrBoxFilter rejects BORDER_ISOLATED and collapses only for non-constant borders.
        if (shrink)
        {
            if (source.Height == 1)
            {
                ksize = new Size(ksize.Width, 1);
            }

            if (source.Width == 1)
            {
                ksize = new Size(1, ksize.Height);
            }
        }

        int kw = ksize.Width;
        int kh = ksize.Height;
        int ax = anchor.X < 0 ? kw / 2 : anchor.X;
        int ay = anchor.Y < 0 ? kh / 2 : anchor.Y;
        if (ax >= kw || ay >= kh)
        {
            throw new ArgumentOutOfRangeException(nameof(anchor), anchor, "The anchor must lie inside the kernel.");
        }

        return (kw, kh, ax, ay, normalize ? 1.0 / ((double)kw * kh) : 1.0);
    }

    // ------------------------------------------------------- ushort sums

    /// <summary>
    /// ColumnSum&lt;ushort, uchar&gt; over RowSum&lt;uchar, ushort&gt;. A scaled sum is divided with
    /// OpenCV's fixed-point rule <c>(s + divDelta) * divScale &gt;&gt; 23</c>, which rounds exact
    /// halves up (box_filter.simd.hpp:438-457, 577-583).
    /// </summary>
    private static void BoxUInt16ToByte(Mat<byte> padded, Mat<byte> dst, int kw, int kh, double scale)
    {
        int cn = dst.Channels;
        int dstHeight = dst.Height;
        int width = dst.Width * cn;
        bool haveScale = scale != 1;
        uint divScale = 1;
        uint divDelta = 0;
        if (haveScale)
        {
            int d = ScalarConverter.CvRound(1.0 / scale);
            double scalef = (double)(1 << FixedShift) / d;
            divScale = (uint)Math.Floor(scalef);
            scalef -= divScale;
            divDelta = (uint)(d / 2);
            if (scalef < 0.5)
            {
                divDelta++;
            }
            else
            {
                divScale++;
            }
        }

        ushort[] rowSums = ArrayPool<ushort>.Shared.Rent(width * padded.Height);
        try
        {
            ParallelRows.For(
                padded.Height,
                (long)width * kw / Vector<ushort>.Count,
                (y0, y1) => RowSumsUInt16(padded, rowSums, width, kw, cn, y0, y1));

            // Each stripe seeds its running sum from the kh - 1 rows above it; integer sums are exact.
            ParallelRows.For(
                dstHeight,
                (long)width * 4 / Vector<ushort>.Count,
                (y0, y1) => ColumnRowsUInt16(rowSums, dst, width, kh, haveScale, divScale, divDelta, y0, y1));
        }
        finally
        {
            ArrayPool<ushort>.Shared.Return(rowSums);
        }
    }

    private static void RowSumsUInt16(Mat<byte> padded, ushort[] rowSums, int width, int kw, int cn, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            RowSumUInt16(padded.GetReadOnlyRowSpan(y), rowSums.AsSpan(y * width, width), kw, cn);
        }
    }

    private static void ColumnRowsUInt16(
        ushort[] rowSums, Mat<byte> dst, int width, int kh, bool haveScale, uint divScale, uint divDelta, int y0, int y1)
    {
        ushort[] sumBuffer = ArrayPool<ushort>.Shared.Rent(width);
        try
        {
            Span<ushort> sum = sumBuffer.AsSpan(0, width);
            sum.Clear();
            for (int r = y0; r < y0 + kh - 1; r++)
            {
                AddUInt16(sum, rowSums.AsSpan(r * width, width), sum);
            }

            ref ushort rows = ref rowSums[0];
            ref ushort sumRef = ref MemoryMarshal.GetReference(sum);
            for (int y = y0; y < y1; y++)
            {
                ColumnStepUInt16(
                    ref sumRef,
                    ref Unsafe.Add(ref rows, (y + kh - 1) * width),
                    ref Unsafe.Add(ref rows, y * width),
                    ref MemoryMarshal.GetReference(dst.GetRowSpan(y)),
                    width,
                    haveScale,
                    divScale,
                    divDelta);
            }
        }
        finally
        {
            ArrayPool<ushort>.Shared.Return(sumBuffer);
        }
    }

    /// <summary>
    /// One output row: <c>s = SUM + Sp; D = cast(s); SUM = s - Sm</c>. Sums are at most
    /// <c>256 * 255</c>, so <c>s + divDelta</c> fits in 16 bits and the product in 32 bits.
    /// </summary>
    private static void ColumnStepUInt16(
        ref ushort sum, ref ushort sp, ref ushort sm, ref byte d, int width, bool haveScale, uint divScale, uint divDelta)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int half = Vector<ushort>.Count;
            var ds = new Vector<uint>(divScale);
            var dd = new Vector<ushort>((ushort)divDelta);
            var max = new Vector<ushort>(255);
            for (; i <= width - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                Vector<ushort> s0 = Vector.LoadUnsafe(ref sum, (nuint)i) + Vector.LoadUnsafe(ref sp, (nuint)i);
                Vector<ushort> s1 = Vector.LoadUnsafe(ref sum, (nuint)(i + half)) + Vector.LoadUnsafe(ref sp, (nuint)(i + half));
                Vector<ushort> r0, r1;
                if (haveScale)
                {
                    Vector.Widen(s0 + dd, out Vector<uint> a0, out Vector<uint> a1);
                    Vector.Widen(s1 + dd, out Vector<uint> b0, out Vector<uint> b1);
                    r0 = Vector.Narrow(
                        Vector.ShiftRightLogical(a0 * ds, FixedShift), Vector.ShiftRightLogical(a1 * ds, FixedShift));
                    r1 = Vector.Narrow(
                        Vector.ShiftRightLogical(b0 * ds, FixedShift), Vector.ShiftRightLogical(b1 * ds, FixedShift));
                }
                else
                {
                    r0 = Vector.Min(s0, max);
                    r1 = Vector.Min(s1, max);
                }

                Vector.Narrow(r0, r1).StoreUnsafe(ref d, (nuint)i);
                (s0 - Vector.LoadUnsafe(ref sm, (nuint)i)).StoreUnsafe(ref sum, (nuint)i);
                (s1 - Vector.LoadUnsafe(ref sm, (nuint)(i + half))).StoreUnsafe(ref sum, (nuint)(i + half));
            }
        }

        for (; i < width; i++)
        {
            int s0 = Unsafe.Add(ref sum, i) + Unsafe.Add(ref sp, i);
            Unsafe.Add(ref d, i) = haveScale
                ? (byte)(((uint)s0 + divDelta) * divScale >> FixedShift)
                : (byte)Math.Min(s0, 255);
            Unsafe.Add(ref sum, i) = (ushort)(s0 - Unsafe.Add(ref sm, i));
        }
    }

    private static void RowSumUInt16(ReadOnlySpan<byte> row, Span<ushort> dst, int kw, int cn)
    {
        ref byte s = ref MemoryMarshal.GetReference(row);
        ref ushort d = ref MemoryMarshal.GetReference(dst);
        int n = dst.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int half = Vector<ushort>.Count;
            for (; i <= n - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                Vector<ushort> lo = Vector<ushort>.Zero;
                Vector<ushort> hi = Vector<ushort>.Zero;
                for (int k = 0; k < kw; k++)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)(i + (k * cn))), out Vector<ushort> wl, out Vector<ushort> wh);
                    lo += wl;
                    hi += wh;
                }

                lo.StoreUnsafe(ref d, (nuint)i);
                hi.StoreUnsafe(ref d, (nuint)(i + half));
            }
        }

        for (; i < n; i++)
        {
            int sum = 0;
            for (int k = 0; k < kw; k++)
            {
                sum += Unsafe.Add(ref s, i + (k * cn));
            }

            Unsafe.Add(ref d, i) = (ushort)sum;
        }
    }

    private static void AddUInt16(ReadOnlySpan<ushort> a, ReadOnlySpan<ushort> b, Span<ushort> result)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= a.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
            {
                (new Vector<ushort>(a[i..]) + new Vector<ushort>(b[i..])).CopyTo(result[i..]);
            }
        }

        for (; i < a.Length; i++)
        {
            result[i] = (ushort)(a[i] + b[i]);
        }
    }

    // ---------------------------------------------------------- int sums

    /// <summary>
    /// RowSum&lt;uchar, int&gt; or SqrRowSum&lt;uchar, int&gt; followed by ColumnSum&lt;int, TDst&gt;.
    /// Integer sums are exact, so the stripes of the column pass seed their own running sums.
    /// </summary>
    private static void IntSums<TDst>(
        Mat<byte> src, Mat<TDst> dst, int kw, int kh, int ax, int ay, double scale, BorderType borderType, bool square)
        where TDst : unmanaged
    {
        using PooledMat<byte> pooled = Borders.CopyMakeBorderPooled(
            src, ay, kh - ay - 1, ax, kw - ax - 1, borderType, default);
        Mat<byte> padded = pooled.Mat;
        int cn = src.Channels;
        int width = src.Width * cn;
        int[] rows = ArrayPool<int>.Shared.Rent(width * padded.Height);
        try
        {
            ParallelRows.For(
                padded.Height,
                (long)width * kw / Vector<int>.Count,
                (y0, y1) => IntRowSums(padded, rows, width, kw, cn, square, y0, y1));

            ParallelRows.For(
                src.Height,
                (long)width * 4 / Vector<int>.Count,
                (y0, y1) => IntColumnRows(rows, dst, width, kh, scale, y0, y1));
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rows);
        }
    }

    private static void IntRowSums(Mat<byte> padded, int[] rows, int width, int kw, int cn, bool square, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            ref byte s = ref MemoryMarshal.GetReference(padded.GetReadOnlyRowSpan(y));
            ref int d = ref rows[y * width];
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int vi = Vector<int>.Count;
                for (; i <= width - Vector<byte>.Count; i += Vector<byte>.Count)
                {
                    Vector<int> a0 = default, a1 = default, a2 = default, a3 = default;
                    for (int k = 0; k < kw; k++)
                    {
                        FilterVectors.WidenToInt32(
                            Vector.LoadUnsafe(ref s, (nuint)(i + (k * cn))),
                            out Vector<int> w0, out Vector<int> w1, out Vector<int> w2, out Vector<int> w3);
                        if (square)
                        {
                            a0 += w0 * w0;
                            a1 += w1 * w1;
                            a2 += w2 * w2;
                            a3 += w3 * w3;
                        }
                        else
                        {
                            a0 += w0;
                            a1 += w1;
                            a2 += w2;
                            a3 += w3;
                        }
                    }

                    a0.StoreUnsafe(ref d, (nuint)i);
                    a1.StoreUnsafe(ref d, (nuint)(i + vi));
                    a2.StoreUnsafe(ref d, (nuint)(i + (2 * vi)));
                    a3.StoreUnsafe(ref d, (nuint)(i + (3 * vi)));
                }
            }

            for (; i < width; i++)
            {
                int sum = 0;
                for (int k = 0; k < kw; k++)
                {
                    int v = Unsafe.Add(ref s, i + (k * cn));
                    sum += square ? v * v : v;
                }

                Unsafe.Add(ref d, i) = sum;
            }
        }
    }

    private static void IntColumnRows<TDst>(int[] rows, Mat<TDst> dst, int width, int kh, double scale, int y0, int y1)
        where TDst : unmanaged
    {
        int[] sumBuffer = ArrayPool<int>.Shared.Rent(width);
        try
        {
            Span<int> sum = sumBuffer.AsSpan(0, width);
            sum.Clear();
            for (int r = y0; r < y0 + kh - 1; r++)
            {
                ReadOnlySpan<int> sp = rows.AsSpan(r * width, width);
                for (int i = 0; i < width; i++)
                {
                    sum[i] += sp[i];
                }
            }

            ref int rowsRef = ref rows[0];
            ref int sumRef = ref sumBuffer[0];
            for (int y = y0; y < y1; y++)
            {
                IntColumnStep(
                    ref sumRef,
                    ref Unsafe.Add(ref rowsRef, (y + kh - 1) * width),
                    ref Unsafe.Add(ref rowsRef, y * width),
                    ref MemoryMarshal.GetReference(dst.GetRowSpan(y)),
                    width,
                    scale);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(sumBuffer);
        }
    }

    /// <summary>
    /// One output row of ColumnSum&lt;int, TDst&gt;: <c>s = SUM + Sp; D = cast(s·scale); SUM = s − Sm</c>.
    /// Elements before the split use <c>(float)s·(float)scale</c>; the others use
    /// <c>s·scale</c> in double.
    /// </summary>
    private static void IntColumnStep<TDst>(ref int sum, ref int sp, ref int sm, ref TDst d, int width, double scale)
        where TDst : unmanaged
    {
        bool haveScale = scale != 1;
        float fscale = (float)scale;
        int split = typeof(TDst) == typeof(byte) || typeof(TDst) == typeof(short) ? width & ~7
            : typeof(TDst) == typeof(float) || typeof(TDst) == typeof(int) ? width & ~3
            : 0;
        int vectorEnd = haveScale ? split : width;
        int i = 0;
        if (Vector.IsHardwareAccelerated && typeof(TDst) != typeof(double))
        {
            int vc = Vector<int>.Count;
            var vs = new Vector<float>(fscale);
            if (typeof(TDst) == typeof(float))
            {
                ref float df = ref Unsafe.As<TDst, float>(ref d);
                for (; i <= vectorEnd - vc; i += vc)
                {
                    Vector<int> s = Vector.LoadUnsafe(ref sum, (nuint)i) + Vector.LoadUnsafe(ref sp, (nuint)i);
                    Vector<float> f = Vector.ConvertToSingle(s);
                    (haveScale ? f * vs : f).StoreUnsafe(ref df, (nuint)i);
                    (s - Vector.LoadUnsafe(ref sm, (nuint)i)).StoreUnsafe(ref sum, (nuint)i);
                }
            }
            else if (typeof(TDst) == typeof(int))
            {
                ref int di = ref Unsafe.As<TDst, int>(ref d);
                for (; i <= vectorEnd - vc; i += vc)
                {
                    Vector<int> s = Vector.LoadUnsafe(ref sum, (nuint)i) + Vector.LoadUnsafe(ref sp, (nuint)i);
                    (haveScale ? FilterVectors.CvRound(Vector.ConvertToSingle(s) * vs) : s).StoreUnsafe(ref di, (nuint)i);
                    (s - Vector.LoadUnsafe(ref sm, (nuint)i)).StoreUnsafe(ref sum, (nuint)i);
                }
            }
            else if (typeof(TDst) == typeof(short))
            {
                ref short ds = ref Unsafe.As<TDst, short>(ref d);
                for (; i <= vectorEnd - (2 * vc); i += 2 * vc)
                {
                    Vector<int> s0 = Vector.LoadUnsafe(ref sum, (nuint)i) + Vector.LoadUnsafe(ref sp, (nuint)i);
                    Vector<int> s1 = Vector.LoadUnsafe(ref sum, (nuint)(i + vc)) + Vector.LoadUnsafe(ref sp, (nuint)(i + vc));
                    FilterVectors.SaturateToInt16(Scaled(s0, vs, haveScale), Scaled(s1, vs, haveScale)).StoreUnsafe(ref ds, (nuint)i);
                    (s0 - Vector.LoadUnsafe(ref sm, (nuint)i)).StoreUnsafe(ref sum, (nuint)i);
                    (s1 - Vector.LoadUnsafe(ref sm, (nuint)(i + vc))).StoreUnsafe(ref sum, (nuint)(i + vc));
                }
            }
            else if (typeof(TDst) == typeof(byte))
            {
                ref byte db = ref Unsafe.As<TDst, byte>(ref d);
                for (; i <= vectorEnd - (4 * vc); i += 4 * vc)
                {
                    Vector<int> s0 = Vector.LoadUnsafe(ref sum, (nuint)i) + Vector.LoadUnsafe(ref sp, (nuint)i);
                    Vector<int> s1 = Vector.LoadUnsafe(ref sum, (nuint)(i + vc)) + Vector.LoadUnsafe(ref sp, (nuint)(i + vc));
                    Vector<int> s2 = Vector.LoadUnsafe(ref sum, (nuint)(i + (2 * vc))) + Vector.LoadUnsafe(ref sp, (nuint)(i + (2 * vc)));
                    Vector<int> s3 = Vector.LoadUnsafe(ref sum, (nuint)(i + (3 * vc))) + Vector.LoadUnsafe(ref sp, (nuint)(i + (3 * vc)));
                    FilterVectors.SaturateToByte(
                        Scaled(s0, vs, haveScale), Scaled(s1, vs, haveScale), Scaled(s2, vs, haveScale), Scaled(s3, vs, haveScale))
                        .StoreUnsafe(ref db, (nuint)i);
                    (s0 - Vector.LoadUnsafe(ref sm, (nuint)i)).StoreUnsafe(ref sum, (nuint)i);
                    (s1 - Vector.LoadUnsafe(ref sm, (nuint)(i + vc))).StoreUnsafe(ref sum, (nuint)(i + vc));
                    (s2 - Vector.LoadUnsafe(ref sm, (nuint)(i + (2 * vc)))).StoreUnsafe(ref sum, (nuint)(i + (2 * vc)));
                    (s3 - Vector.LoadUnsafe(ref sm, (nuint)(i + (3 * vc)))).StoreUnsafe(ref sum, (nuint)(i + (3 * vc)));
                }
            }
        }

        for (; i < width; i++)
        {
            int s0 = Unsafe.Add(ref sum, i) + Unsafe.Add(ref sp, i);
            TDst value;
            if (!haveScale)
            {
                value = Saturate.FromInt<TDst>(s0);
            }
            else if (i < split)
            {
                value = Saturate.FromFloat<TDst>(s0 * fscale);
            }
            else
            {
                value = Saturate.FromDouble<TDst>(s0 * scale);
            }

            Unsafe.Add(ref d, i) = value;
            Unsafe.Add(ref sum, i) = s0 - Unsafe.Add(ref sm, i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Scaled(Vector<int> s, Vector<float> scale, bool haveScale)
        => haveScale ? FilterVectors.CvRound(Vector.ConvertToSingle(s) * scale) : s;

    // ------------------------------------------------------- double sums

    /// <summary>
    /// RowSum&lt;T, double&gt;, SqrRowSum&lt;T, double&gt; or the BlockSum row sums, followed by
    /// ColumnSum&lt;double, TDst&gt; (BlockSum has the same column recurrence). The double running
    /// sums depend on their history, so the column pass splits the columns, not the rows,
    /// between threads.
    /// </summary>
    private static void DoubleSums<TSrc, TDst>(
        Mat<TSrc> src, Mat<float>? factor, Mat<TDst> dst, int kw, int kh, int ax, int ay, double scale, BorderType borderType, RowKind kind)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        int cn = src.Channels;
        int width = src.Width * cn;
        int paddedHeight = src.Height + kh - 1;
        long rowWork = DoubleRowWork(width);

        // The parallel form makes two passes through a full-size row-sum image; it pays only with
        // several stripes.
        if (ParallelRows.Stripes(paddedHeight, rowWork) < MinBoxStripes)
        {
            FusedDoubleSums(src, factor, dst, kw, kh, ax, ay, scale, borderType, kind);
            return;
        }

        double[] rows = ArrayPool<double>.Shared.Rent(width * paddedHeight);
        try
        {
            var job = new DoubleJob<TSrc>(src, factor, rows, paddedHeight, kw, ax, ay, borderType, kind);
            ParallelRows.For(paddedHeight, rowWork, (y0, y1) => DoubleRowSums(job, y0, y1));

            int blocks = (width + ColumnBlock - 1) / ColumnBlock;
            ParallelRows.For(
                blocks,
                (long)ColumnBlock * src.Height / Vector<double>.Count,
                (b0, b1) => DoubleColumnBlocks(rows, dst, width, kh, scale, b0 * ColumnBlock, Math.Min(b1 * ColumnBlock, width)));
        }
        finally
        {
            ArrayPool<double>.Shared.Return(rows);
        }
    }

    /// <summary>Work units of one padded row of the double-sum row pass (a running sum per element).</summary>
    private static long DoubleRowWork(int width) => (long)width * 2;

    /// <summary>
    /// Single-threaded form of <see cref="DoubleSums{TSrc, TDst}"/>: the row sums live in a ring
    /// of <c>kh + 3</c> rows, so they stay in cache. The arithmetic is the same.
    /// </summary>
    private static void FusedDoubleSums<TSrc, TDst>(
        Mat<TSrc> src, Mat<float>? factor, Mat<TDst> dst, int kw, int kh, int ax, int ay, double scale, BorderType borderType, RowKind kind)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        int width = src.Width * src.Channels;
        int paddedHeight = src.Height + kh - 1;
        int ringRows = kh + 3;
        double[] ring = ArrayPool<double>.Shared.Rent(width * ringRows);
        double[] sumBuffer = ArrayPool<double>.Shared.Rent(width);
        try
        {
            var job = new DoubleJob<TSrc>(src, factor, ring, ringRows, kw, ax, ay, borderType, kind);
            using var scratch = new RowScratch<TSrc>(job);
            Span<double> sum = sumBuffer.AsSpan(0, width);
            sum.Clear();
            int computed = 0;
            for (int y = 0; y < src.Height; y++)
            {
                int needed = y + kh;
                if (computed < needed)
                {
                    int end = Math.Min(Math.Max(needed, computed + 4), paddedHeight);
                    DoubleRowSums(job, computed, end, scratch);
                    computed = end;
                }

                if (y == 0)
                {
                    for (int r = 0; r < kh - 1; r++)
                    {
                        ReadOnlySpan<double> sp = ring.AsSpan((r % ringRows) * width, width);
                        for (int i = 0; i < width; i++)
                        {
                            sum[i] += sp[i];
                        }
                    }
                }

                DoubleColumnRow(
                    ref sumBuffer[0],
                    ref ring[((y + kh - 1) % ringRows) * width],
                    ref ring[(y % ringRows) * width],
                    ref MemoryMarshal.GetReference(dst.GetRowSpan(y)),
                    width,
                    scale);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(sumBuffer);
            ArrayPool<double>.Shared.Return(ring);
        }
    }

    /// <summary>
    /// Row sums in <see cref="Rows"/>: padded row y occupies slot <c>y % RingRows</c>.
    /// </summary>
    private sealed record DoubleJob<TSrc>(
        Mat<TSrc> Src, Mat<float>? Factor, double[] Rows, int RingRows, int Kw, int Ax, int Ay, BorderType Border, RowKind Kind)
        where TSrc : unmanaged;

    private static void DoubleRowSums<TSrc>(DoubleJob<TSrc> job, int y0, int y1)
        where TSrc : unmanaged
    {
        using var scratch = new RowScratch<TSrc>(job);
        DoubleRowSums(job, y0, y1, scratch);
    }

    /// <summary>Pooled line buffers and the border column map of one row-sum worker.</summary>
    private sealed class RowScratch<TSrc> : IDisposable
        where TSrc : unmanaged
    {
        public RowScratch(DoubleJob<TSrc> job)
        {
            int w = job.Src.Width, cn = job.Src.Channels, kw = job.Kw;
            PaddedWidth = (w + kw - 1) * cn;
            Lines = ArrayPool<float>.Shared.Rent(4 * PaddedWidth);
            Work = ArrayPool<double>.Shared.Rent(4 * (PaddedWidth + (w * cn)));
            ColMap = ArrayPool<int>.Shared.Rent(kw);
            Left = job.Ax;
            Right = kw - 1 - job.Ax;
            for (int x = 0; x < Left; x++)
            {
                ColMap[x] = Borders.BorderInterpolate(x - Left, w, job.Border);
            }

            for (int x = 0; x < Right; x++)
            {
                ColMap[Left + x] = Borders.BorderInterpolate(w + x, w, job.Border);
            }
        }

        public int PaddedWidth { get; }

        public float[] Lines { get; }

        public double[] Work { get; }

        public int[] ColMap { get; }

        public int Left { get; }

        public int Right { get; }

        public void Dispose()
        {
            ArrayPool<int>.Shared.Return(ColMap);
            ArrayPool<double>.Shared.Return(Work);
            ArrayPool<float>.Shared.Return(Lines);
        }
    }

    private static void DoubleRowSums<TSrc>(DoubleJob<TSrc> job, int y0, int y1, RowScratch<TSrc> scratch)
        where TSrc : unmanaged
    {
        int cn = job.Src.Channels;
        int kw = job.Kw;
        int width = job.Src.Width * cn;
        int paddedWidth = scratch.PaddedWidth;
        float[] lineBuffer = scratch.Lines;

        // Running sums form one dependency chain per row; four rows run interleaved.
        bool running = job.Kind == RowKind.Square || (job.Kind == RowKind.Sum && kw != 3 && kw != 5);
        int group = running ? 4 : 1;
        for (int y = y0; y < y1; y += group)
        {
            int count = Math.Min(group, y1 - y);
            for (int r = 0; r < count; r++)
            {
                PrepareLine(job, lineBuffer.AsSpan(r * paddedWidth, paddedWidth), scratch.ColMap, y + r, scratch.Left, scratch.Right);
            }

            if (count == 4)
            {
                int ring = job.RingRows;
                RunningRowSums4(
                    lineBuffer,
                    scratch.Work,
                    paddedWidth,
                    job.Rows,
                    (y % ring) * width,
                    ((y + 1) % ring) * width,
                    ((y + 2) % ring) * width,
                    ((y + 3) % ring) * width,
                    width,
                    kw,
                    cn,
                    job.Kind == RowKind.Square);
                continue;
            }

            for (int r = 0; r < count; r++)
            {
                ReadOnlySpan<float> line = lineBuffer.AsSpan(r * paddedWidth, paddedWidth);
                Span<double> d = job.Rows.AsSpan(((y + r) % job.RingRows) * width, width);
                if (job.Kind == RowKind.Square)
                {
                    SqrRowSum(line, d, kw, cn);
                }
                else
                {
                    RowSum(line, d, kw, cn, direct: job.Kind == RowKind.Block);
                }
            }
        }
    }

    /// <summary>
    /// Builds padded row <paramref name="y"/> (border-extended source row, or zeros for a constant
    /// border row). The row sums of a zero row are +0, as OpenCV's.
    /// </summary>
    private static void PrepareLine<TSrc>(DoubleJob<TSrc> job, Span<float> line, int[] colMap, int y, int left, int right)
        where TSrc : unmanaged
    {
        Mat<TSrc> src = job.Src;
        int cn = src.Channels;
        int w = src.Width;
        int sy = Borders.BorderInterpolate(y - job.Ay, src.Height, job.Border);
        if (sy < 0)
        {
            line.Clear();
            return;
        }

        Span<float> inner = line.Slice(left * cn, w * cn);
        LoadLine(src, job.Factor, sy, inner);
        for (int x = 0; x < left; x++)
        {
            CopyPixel(inner, colMap[x], line.Slice(x * cn, cn));
        }

        for (int x = 0; x < right; x++)
        {
            CopyPixel(inner, colMap[left + x], line.Slice((left + w + x) * cn, cn));
        }
    }

    /// <summary>
    /// The running RowSum or SqrRowSum of four consecutive padded rows, interleaved. Each row
    /// keeps OpenCV's operation sequence.
    /// </summary>
    private static void RunningRowSums4(
        float[] lines, double[] work, int paddedWidth, double[] rows, int o0, int o1, int o2, int o3, int width, int kw, int cn, bool square)
    {
        // Widening to double and the differences run vectorized; only the running additions are
        // sequential. (double)b − (double)a and b·b − a·a give the same values as in OpenCV.
        int kszCn = kw * cn;
        int last = width - cn;
        for (int r = 0; r < 4; r++)
        {
            Span<double> dl = work.AsSpan(r * paddedWidth, paddedWidth);
            WidenLine(lines.AsSpan(r * paddedWidth, paddedWidth), dl, square);
            SubtractShifted(dl, work.AsSpan((4 * paddedWidth) + (r * width), width), kszCn, last);
        }

        ref double l0 = ref work[0];
        ref double l1 = ref Unsafe.Add(ref l0, paddedWidth);
        ref double l2 = ref Unsafe.Add(ref l0, 2 * paddedWidth);
        ref double l3 = ref Unsafe.Add(ref l0, 3 * paddedWidth);
        ref double f0 = ref Unsafe.Add(ref l0, 4 * paddedWidth);
        ref double f1 = ref Unsafe.Add(ref f0, width);
        ref double f2 = ref Unsafe.Add(ref f0, 2 * width);
        ref double f3 = ref Unsafe.Add(ref f0, 3 * width);
        ref double d0 = ref rows[o0];
        ref double d1 = ref rows[o1];
        ref double d2 = ref rows[o2];
        ref double d3 = ref rows[o3];
        for (int c = 0; c < cn; c++)
        {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = c; i < kszCn; i += cn)
            {
                s0 += Unsafe.Add(ref l0, i);
                s1 += Unsafe.Add(ref l1, i);
                s2 += Unsafe.Add(ref l2, i);
                s3 += Unsafe.Add(ref l3, i);
            }

            Unsafe.Add(ref d0, c) = s0;
            Unsafe.Add(ref d1, c) = s1;
            Unsafe.Add(ref d2, c) = s2;
            Unsafe.Add(ref d3, c) = s3;
            for (int i = c; i < last; i += cn)
            {
                s0 += Unsafe.Add(ref f0, i);
                s1 += Unsafe.Add(ref f1, i);
                s2 += Unsafe.Add(ref f2, i);
                s3 += Unsafe.Add(ref f3, i);
                Unsafe.Add(ref d0, i + cn) = s0;
                Unsafe.Add(ref d1, i + cn) = s1;
                Unsafe.Add(ref d2, i + cn) = s2;
                Unsafe.Add(ref d3, i + cn) = s3;
            }
        }
    }

    /// <summary><c>d[i] = (double)s[i]</c>, or its square.</summary>
    private static void WidenLine(ReadOnlySpan<float> s, Span<double> d, bool square)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            ref float sr = ref MemoryMarshal.GetReference(s);
            ref double dr = ref MemoryMarshal.GetReference(d);
            int vf = Vector<float>.Count, vd = Vector<double>.Count;
            for (; i <= s.Length - vf; i += vf)
            {
                Vector.Widen(Vector.LoadUnsafe(ref sr, (nuint)i), out Vector<double> a, out Vector<double> b);
                if (square)
                {
                    a *= a;
                    b *= b;
                }

                a.StoreUnsafe(ref dr, (nuint)i);
                b.StoreUnsafe(ref dr, (nuint)(i + vd));
            }
        }

        for (; i < s.Length; i++)
        {
            double v = s[i];
            d[i] = square ? v * v : v;
        }
    }

    /// <summary><c>f[i] = l[i + shift] − l[i]</c> for <c>i &lt; count</c>.</summary>
    private static void SubtractShifted(ReadOnlySpan<double> l, Span<double> f, int shift, int count)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            ref double lr = ref MemoryMarshal.GetReference(l);
            ref double fr = ref MemoryMarshal.GetReference(f);
            for (; i <= count - Vector<double>.Count; i += Vector<double>.Count)
            {
                (Vector.LoadUnsafe(ref lr, (nuint)(i + shift)) - Vector.LoadUnsafe(ref lr, (nuint)i)).StoreUnsafe(ref fr, (nuint)i);
            }
        }

        for (; i < count; i++)
        {
            f[i] = l[i + shift] - l[i];
        }
    }

    private static void LoadLine<TSrc>(Mat<TSrc> src, Mat<float>? factor, int y, Span<float> line)
        where TSrc : unmanaged
    {
        if (typeof(TSrc) == typeof(float))
        {
            ReadOnlySpan<float> s = MemoryMarshal.Cast<TSrc, float>(src.GetReadOnlyRowSpan(y));
            if (factor is null)
            {
                s.CopyTo(line);
                return;
            }

            ReadOnlySpan<float> f = factor.GetReadOnlyRowSpan(y);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                for (; i <= line.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    (new Vector<float>(s[i..]) * new Vector<float>(f[i..])).CopyTo(line[i..]);
                }
            }

            for (; i < line.Length; i++)
            {
                line[i] = s[i] * f[i];
            }

            return;
        }

        ReadOnlySpan<byte> b = MemoryMarshal.Cast<TSrc, byte>(src.GetReadOnlyRowSpan(y));
        for (int i = 0; i < line.Length; i++)
        {
            line[i] = b[i];
        }
    }

    private static void CopyPixel(ReadOnlySpan<float> inner, int sx, Span<float> pixel)
    {
        if (sx < 0)
        {
            pixel.Clear();
        }
        else
        {
            inner.Slice(sx * pixel.Length, pixel.Length).CopyTo(pixel);
        }
    }

    /// <summary>
    /// RowSum&lt;float, double&gt;: direct left-associated sums for ksize 3 and 5, per-channel
    /// running sums <c>s += (S[i + k·cn] − S[i])</c> otherwise.
    /// </summary>
    private static void RowSum(ReadOnlySpan<float> s, Span<double> d, int kw, int cn, bool direct)
    {
        int n = d.Length;
        if (kw == 3 || kw == 5 || direct)
        {
            // ksize 3 and 5 start from the first element; the other BlockSum widths start from 0.
            int k0 = kw == 3 || kw == 5 ? 1 : 0;
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                ref float sr = ref MemoryMarshal.GetReference(s);
                ref double dr = ref MemoryMarshal.GetReference(d);
                int vf = Vector<float>.Count, vd = Vector<double>.Count;
                for (; i <= n - vf; i += vf)
                {
                    Vector<double> a0 = Vector<double>.Zero, a1 = Vector<double>.Zero;
                    if (k0 == 1)
                    {
                        Vector.Widen(Vector.LoadUnsafe(ref sr, (nuint)i), out a0, out a1);
                    }

                    for (int k = k0; k < kw; k++)
                    {
                        Vector.Widen(Vector.LoadUnsafe(ref sr, (nuint)(i + (k * cn))), out Vector<double> b0, out Vector<double> b1);
                        a0 += b0;
                        a1 += b1;
                    }

                    a0.StoreUnsafe(ref dr, (nuint)i);
                    a1.StoreUnsafe(ref dr, (nuint)(i + vd));
                }
            }

            for (; i < n; i++)
            {
                double acc = k0 == 1 ? s[i] : 0.0;
                for (int k = k0; k < kw; k++)
                {
                    acc += s[i + (k * cn)];
                }

                d[i] = acc;
            }

            return;
        }

        int kszCn = kw * cn;
        int last = n - cn;
        for (int c = 0; c < cn; c++)
        {
            double sum = 0;
            for (int i = c; i < kszCn; i += cn)
            {
                sum += s[i];
            }

            d[c] = sum;
            for (int i = c; i < last; i += cn)
            {
                sum += (double)s[i + kszCn] - s[i];
                d[i + cn] = sum;
            }
        }
    }

    /// <summary>SqrRowSum&lt;float, double&gt;: <c>s += v1·v1 − v0·v0</c> per channel.</summary>
    private static void SqrRowSum(ReadOnlySpan<float> s, Span<double> d, int kw, int cn)
    {
        int n = d.Length;
        int kszCn = kw * cn;
        int last = n - cn;
        for (int c = 0; c < cn; c++)
        {
            double sum = 0;
            for (int i = c; i < kszCn; i += cn)
            {
                double v = s[i];
                sum += v * v;
            }

            d[c] = sum;
            for (int i = c; i < last; i += cn)
            {
                double v0 = s[i], v1 = s[i + kszCn];
                sum += (v1 * v1) - (v0 * v0);
                d[i + cn] = sum;
            }
        }
    }

    /// <summary>ColumnSum&lt;double, TDst&gt; on the columns <c>[c0, c1)</c>.</summary>
    private static void DoubleColumnBlocks<TDst>(double[] rows, Mat<TDst> dst, int width, int kh, double scale, int c0, int c1)
        where TDst : unmanaged
    {
        int n = c1 - c0;
        bool haveScale = scale != 1;
        double[] sumBuffer = ArrayPool<double>.Shared.Rent(n);
        try
        {
            Span<double> sum = sumBuffer.AsSpan(0, n);
            sum.Clear();
            for (int r = 0; r < kh - 1; r++)
            {
                ReadOnlySpan<double> sp = rows.AsSpan((r * width) + c0, n);
                for (int i = 0; i < n; i++)
                {
                    sum[i] += sp[i];
                }
            }

            ref double rowsRef = ref rows[0];
            for (int y = 0; y < dst.Height; y++)
            {
                DoubleColumnRow(
                    ref sumBuffer[0],
                    ref Unsafe.Add(ref rowsRef, ((y + kh - 1) * width) + c0),
                    ref Unsafe.Add(ref rowsRef, (y * width) + c0),
                    ref Unsafe.Add(ref MemoryMarshal.GetReference(dst.GetRowSpan(y)), c0),
                    n,
                    scale);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(sumBuffer);
        }
    }

    /// <summary>One output row of ColumnSum&lt;double, TDst&gt;: <c>s = SUM + Sp; D = cast(s·scale); SUM = s − Sm</c>.</summary>
    private static void DoubleColumnRow<TDst>(ref double sum, ref double sp, ref double sm, ref TDst d, int n, double scale)
        where TDst : unmanaged
    {
        bool haveScale = scale != 1;
        int i = 0;
        if (Vector.IsHardwareAccelerated && (typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double)))
        {
            var vscale = new Vector<double>(scale);
            int vd = Vector<double>.Count;
            for (; i <= n - (2 * vd); i += 2 * vd)
            {
                Vector<double> s0 = Vector.LoadUnsafe(ref sum, (nuint)i) + Vector.LoadUnsafe(ref sp, (nuint)i);
                Vector<double> s1 = Vector.LoadUnsafe(ref sum, (nuint)(i + vd)) + Vector.LoadUnsafe(ref sp, (nuint)(i + vd));
                Vector<double> o0 = haveScale ? s0 * vscale : s0;
                Vector<double> o1 = haveScale ? s1 * vscale : s1;
                if (typeof(TDst) == typeof(float))
                {
                    Vector.Narrow(o0, o1).StoreUnsafe(ref Unsafe.As<TDst, float>(ref d), (nuint)i);
                }
                else
                {
                    o0.StoreUnsafe(ref Unsafe.As<TDst, double>(ref d), (nuint)i);
                    o1.StoreUnsafe(ref Unsafe.As<TDst, double>(ref d), (nuint)(i + vd));
                }

                (s0 - Vector.LoadUnsafe(ref sm, (nuint)i)).StoreUnsafe(ref sum, (nuint)i);
                (s1 - Vector.LoadUnsafe(ref sm, (nuint)(i + vd))).StoreUnsafe(ref sum, (nuint)(i + vd));
            }
        }

        for (; i < n; i++)
        {
            double s0 = Unsafe.Add(ref sum, i) + Unsafe.Add(ref sp, i);
            Unsafe.Add(ref d, i) = Saturate.FromDouble<TDst>(haveScale ? s0 * scale : s0);
            Unsafe.Add(ref sum, i) = s0 - Unsafe.Add(ref sm, i);
        }
    }
}
