// This implementation is ported from the OpenCV moments implementation.
//
// Original source:
//   modules/geometry/src/moments.cpp (cv::moments, momentsInTile, MomentsInTile_SIMD)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Raster moments of a single-channel image over 32 x 32 tiles, as OpenCV <c>cv::moments</c>.
/// </summary>
/// <remarks>
/// <para>OpenCV computes the ten spatial moments of each tile in local coordinates, then adds the
/// tiles in row-major tile order with the shift formulas in <c>double</c>. This port computes the
/// tiles in parallel (one tile row per work item) into a buffer and adds them on one thread in the
/// same order, so the result does not depend on the thread count.</para>
/// <para>8-bit, 16-bit and binary tiles have exact integer moments: OpenCV accumulates them in
/// <c>int</c> or <c>int64</c> without overflow. The port sums the pixels of each column over the
/// rows of a tile row with weights <c>1, y, y², y³</c> (<c>Vector&lt;int&gt;</c> for 8-bit and
/// binary sources), then reduces each tile with weights <c>x^j</c>. The sums are exact, so the
/// order differs from OpenCV without a change in the result.</para>
/// <para>32F and 64F tiles accumulate in <c>double</c> in the OpenCV order: per row along x, then
/// per tile along y.</para>
/// </remarks>
internal static class DescrImageMoments
{
    private const int TileSize = 32;

    public static Moments Compute<T>(Mat<T> image, bool binary)
        where T : unmanaged
    {
        int width = image.Width, height = image.Height;
        int xtiles = (width + TileSize - 1) / TileSize;
        int ytiles = (height + TileSize - 1) / TileSize;
        int length = xtiles * ytiles * 10;
        double[] tiles = ArrayPool<double>.Shared.Rent(length);
        try
        {
            bool exact = binary || typeof(T) != typeof(float) && typeof(T) != typeof(double);
            long work = (long)width * TileSize;
            if (exact && (binary || typeof(T) == typeof(byte)))
            {
                work /= Vector<int>.Count;
            }

            ParallelRows.For(ytiles, work, (start, end) => TileRows(image, binary, tiles, start, end));

            double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m11 = 0, m02 = 0, m30 = 0, m21 = 0, m12 = 0, m03 = 0;
            const double s = 1.0 / 255;
            Span<double> mom = stackalloc double[10];
            for (int ty = 0; ty < ytiles; ty++)
            {
                int y = ty * TileSize;
                for (int tx = 0; tx < xtiles; tx++)
                {
                    int x = tx * TileSize;
                    tiles.AsSpan(((ty * xtiles) + tx) * 10, 10).CopyTo(mom);
                    if (binary)
                    {
                        for (int k = 0; k < 10; k++)
                        {
                            mom[k] *= s;
                        }
                    }

                    double xm = x * mom[0], ym = y * mom[0];
                    m00 += mom[0];
                    m10 += mom[1] + xm;
                    m01 += mom[2] + ym;
                    m20 += mom[3] + (x * ((mom[1] * 2) + xm));
                    m11 += mom[4] + (x * (mom[2] + ym)) + (y * mom[1]);
                    m02 += mom[5] + (y * ((mom[2] * 2) + ym));
                    m30 += mom[6] + (x * ((3.0 * mom[3]) + (x * ((3.0 * mom[1]) + xm))));
                    m21 += mom[7] + (x * ((2 * (mom[4] + (y * mom[1]))) + (x * (mom[2] + ym)))) + (y * mom[3]);
                    m12 += mom[8] + (y * ((2 * (mom[4] + (x * mom[2]))) + (y * (mom[1] + xm)))) + (x * mom[5]);
                    m03 += mom[9] + (y * ((3.0 * mom[5]) + (y * ((3.0 * mom[2]) + ym))));
                }
            }

            return new Moments(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(tiles);
        }
    }

    private static void TileRows<T>(Mat<T> image, bool binary, double[] tiles, int startTileRow, int endTileRow)
        where T : unmanaged
    {
        if (binary || typeof(T) == typeof(byte))
        {
            IntegerTileRows(image, binary, tiles, startTileRow, endTileRow);
        }
        else if (typeof(T) == typeof(ushort) || typeof(T) == typeof(short))
        {
            Integer16TileRows(image, tiles, startTileRow, endTileRow);
        }
        else
        {
            FloatTileRows(image, tiles, startTileRow, endTileRow);
        }
    }

    /// <summary>
    /// 8-bit and binary tiles. Column sums fit in <c>int</c>: <c>sum(255 * y³) &lt;= 62.7e6</c>
    /// over 32 rows, and each tile moment is at most 2.01e9.
    /// </summary>
    private static void IntegerTileRows<T>(Mat<T> image, bool binary, double[] tiles, int startTileRow, int endTileRow)
        where T : unmanaged
    {
        int width = image.Width, height = image.Height;
        int xtiles = (width + TileSize - 1) / TileSize;
        int[] sums = ArrayPool<int>.Shared.Rent(4 * width);
        byte[]? binRow = typeof(T) == typeof(byte) ? null : ArrayPool<byte>.Shared.Rent(width);
        try
        {
            Span<int> s0 = sums.AsSpan(0, width), s1 = sums.AsSpan(width, width);
            Span<int> s2 = sums.AsSpan(2 * width, width), s3 = sums.AsSpan(3 * width, width);
            for (int ty = startTileRow; ty < endTileRow; ty++)
            {
                sums.AsSpan(0, 4 * width).Clear();
                int y0 = ty * TileSize;
                int th = Math.Min(TileSize, height - y0);
                for (int yl = 0; yl < th; yl++)
                {
                    ReadOnlySpan<byte> row;
                    bool mapNonZero;
                    if (binRow is null)
                    {
                        row = MemoryMarshal.Cast<T, byte>(image.GetReadOnlyRowSpan(y0 + yl));
                        mapNonZero = binary;
                    }
                    else
                    {
                        NonZeroRow(image.GetReadOnlyRowSpan(y0 + yl), binRow);
                        row = binRow.AsSpan(0, width);
                        mapNonZero = false;
                    }

                    AccumulateColumns8u(row, mapNonZero, yl, s0, s1, s2, s3);
                }

                for (int tx = 0; tx < xtiles; tx++)
                {
                    int x0 = tx * TileSize;
                    int tw = Math.Min(TileSize, width - x0);
                    long m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0, m8 = 0, m9 = 0;
                    for (int xl = 0; xl < tw; xl++)
                    {
                        long c0 = s0[x0 + xl], c1 = s1[x0 + xl], c2 = s2[x0 + xl], c3 = s3[x0 + xl];
                        long xx = xl * xl, xxx = xx * xl;
                        m0 += c0;
                        m1 += xl * c0;
                        m3 += xx * c0;
                        m6 += xxx * c0;
                        m2 += c1;
                        m4 += xl * c1;
                        m7 += xx * c1;
                        m5 += c2;
                        m8 += xl * c2;
                        m9 += c3;
                    }

                    Span<double> t = tiles.AsSpan(((ty * xtiles) + tx) * 10, 10);
                    t[0] = m0;
                    t[1] = m1;
                    t[2] = m2;
                    t[3] = m3;
                    t[4] = m4;
                    t[5] = m5;
                    t[6] = m6;
                    t[7] = m7;
                    t[8] = m8;
                    t[9] = m9;
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(sums);
            if (binRow is not null)
            {
                ArrayPool<byte>.Shared.Return(binRow);
            }
        }
    }

    /// <summary>
    /// Adds <c>p</c>, <c>p*y</c>, <c>p*y²</c> and <c>p*y³</c> to the column sums. With
    /// <paramref name="mapNonZero"/>, <c>p</c> is 255 for a nonzero byte and 0 otherwise
    /// (OpenCV <c>compare(src, 0, CMP_NE)</c>).
    /// </summary>
    private static void AccumulateColumns8u(ReadOnlySpan<byte> row, bool mapNonZero, int y, Span<int> s0, Span<int> s1, Span<int> s2, Span<int> s3)
    {
        int width = row.Length;
        int yy = y * y, yyy = yy * y;
        ref byte src = ref MemoryMarshal.GetReference(row);
        ref int r0 = ref MemoryMarshal.GetReference(s0);
        ref int r1 = ref MemoryMarshal.GetReference(s1);
        ref int r2 = ref MemoryMarshal.GetReference(s2);
        ref int r3 = ref MemoryMarshal.GetReference(s3);
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int n = Vector<byte>.Count, ni = Vector<int>.Count;
            for (; x <= width - n; x += n)
            {
                Vector<byte> v = Vector.LoadUnsafe(ref src, (nuint)x);
                if (mapNonZero)
                {
                    v = ~Vector.Equals(v, Vector<byte>.Zero);
                }

                Vector.Widen(v, out Vector<ushort> lo, out Vector<ushort> hi);
                Vector.Widen(lo, out Vector<uint> a, out Vector<uint> b);
                Vector.Widen(hi, out Vector<uint> c, out Vector<uint> d);
                Add4(ref r0, ref r1, ref r2, ref r3, x, Vector.AsVectorInt32(a), y, yy, yyy);
                Add4(ref r0, ref r1, ref r2, ref r3, x + ni, Vector.AsVectorInt32(b), y, yy, yyy);
                Add4(ref r0, ref r1, ref r2, ref r3, x + (2 * ni), Vector.AsVectorInt32(c), y, yy, yyy);
                Add4(ref r0, ref r1, ref r2, ref r3, x + (3 * ni), Vector.AsVectorInt32(d), y, yy, yyy);
            }
        }

        for (; x < width; x++)
        {
            int p = Unsafe.Add(ref src, x);
            if (mapNonZero && p != 0)
            {
                p = 255;
            }

            Unsafe.Add(ref r0, x) += p;
            Unsafe.Add(ref r1, x) += p * y;
            Unsafe.Add(ref r2, x) += p * yy;
            Unsafe.Add(ref r3, x) += p * yyy;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add4(ref int r0, ref int r1, ref int r2, ref int r3, int x, Vector<int> p, int y, int yy, int yyy)
    {
        nuint i = (nuint)x;
        (Vector.LoadUnsafe(ref r0, i) + p).StoreUnsafe(ref r0, i);
        (Vector.LoadUnsafe(ref r1, i) + (p * y)).StoreUnsafe(ref r1, i);
        (Vector.LoadUnsafe(ref r2, i) + (p * yy)).StoreUnsafe(ref r2, i);
        (Vector.LoadUnsafe(ref r3, i) + (p * yyy)).StoreUnsafe(ref r3, i);
    }

    /// <summary>Writes 255 for each nonzero element of <paramref name="src"/> and 0 otherwise.</summary>
    private static void NonZeroRow<T>(ReadOnlySpan<T> src, byte[] dst)
        where T : unmanaged
    {
        if (typeof(T) == typeof(ushort))
        {
            ReadOnlySpan<ushort> s = MemoryMarshal.Cast<T, ushort>(src);
            for (int x = 0; x < s.Length; x++)
            {
                dst[x] = s[x] != 0 ? (byte)255 : (byte)0;
            }
        }
        else if (typeof(T) == typeof(short))
        {
            ReadOnlySpan<short> s = MemoryMarshal.Cast<T, short>(src);
            for (int x = 0; x < s.Length; x++)
            {
                dst[x] = s[x] != 0 ? (byte)255 : (byte)0;
            }
        }
        else if (typeof(T) == typeof(float))
        {
            // NaN compares unequal to 0 and maps to 255; -0 maps to 0.
            ReadOnlySpan<float> s = MemoryMarshal.Cast<T, float>(src);
            for (int x = 0; x < s.Length; x++)
            {
                dst[x] = s[x] != 0 ? (byte)255 : (byte)0;
            }
        }
        else
        {
            ReadOnlySpan<double> s = MemoryMarshal.Cast<T, double>(src);
            for (int x = 0; x < s.Length; x++)
            {
                dst[x] = s[x] != 0 ? (byte)255 : (byte)0;
            }
        }
    }

    /// <summary>
    /// 16-bit tiles. OpenCV <c>momentsInTile&lt;ushort|short, int, int64&gt;</c> sums are exact;
    /// the column sums use <c>long</c> because <c>sum(65535 * y³)</c> exceeds <c>int</c>.
    /// </summary>
    private static void Integer16TileRows<T>(Mat<T> image, double[] tiles, int startTileRow, int endTileRow)
        where T : unmanaged
    {
        int width = image.Width, height = image.Height;
        int xtiles = (width + TileSize - 1) / TileSize;
        long[] sums = ArrayPool<long>.Shared.Rent(4 * width);
        try
        {
            for (int ty = startTileRow; ty < endTileRow; ty++)
            {
                sums.AsSpan(0, 4 * width).Clear();
                int y0 = ty * TileSize;
                int th = Math.Min(TileSize, height - y0);
                for (int yl = 0; yl < th; yl++)
                {
                    long yy = yl * yl, yyy = yy * yl;
                    if (typeof(T) == typeof(ushort))
                    {
                        ReadOnlySpan<ushort> row = MemoryMarshal.Cast<T, ushort>(image.GetReadOnlyRowSpan(y0 + yl));
                        for (int x = 0; x < width; x++)
                        {
                            long p = row[x];
                            sums[x] += p;
                            sums[width + x] += p * yl;
                            sums[(2 * width) + x] += p * yy;
                            sums[(3 * width) + x] += p * yyy;
                        }
                    }
                    else
                    {
                        ReadOnlySpan<short> row = MemoryMarshal.Cast<T, short>(image.GetReadOnlyRowSpan(y0 + yl));
                        for (int x = 0; x < width; x++)
                        {
                            long p = row[x];
                            sums[x] += p;
                            sums[width + x] += p * yl;
                            sums[(2 * width) + x] += p * yy;
                            sums[(3 * width) + x] += p * yyy;
                        }
                    }
                }

                for (int tx = 0; tx < xtiles; tx++)
                {
                    int x0 = tx * TileSize;
                    int tw = Math.Min(TileSize, width - x0);
                    long m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0, m8 = 0, m9 = 0;
                    for (int xl = 0; xl < tw; xl++)
                    {
                        int x = x0 + xl;
                        long c0 = sums[x], c1 = sums[width + x], c2 = sums[(2 * width) + x], c3 = sums[(3 * width) + x];
                        long xx = xl * xl, xxx = xx * xl;
                        m0 += c0;
                        m1 += xl * c0;
                        m3 += xx * c0;
                        m6 += xxx * c0;
                        m2 += c1;
                        m4 += xl * c1;
                        m7 += xx * c1;
                        m5 += c2;
                        m8 += xl * c2;
                        m9 += c3;
                    }

                    Span<double> t = tiles.AsSpan(((ty * xtiles) + tx) * 10, 10);
                    t[0] = m0;
                    t[1] = m1;
                    t[2] = m2;
                    t[3] = m3;
                    t[4] = m4;
                    t[5] = m5;
                    t[6] = m6;
                    t[7] = m7;
                    t[8] = m8;
                    t[9] = m9;
                }
            }
        }
        finally
        {
            ArrayPool<long>.Shared.Return(sums);
        }
    }

    /// <summary>
    /// 32F and 64F tiles: OpenCV <c>momentsInTile&lt;T, double, double&gt;</c> in its operation order.
    /// </summary>
    /// <remarks>
    /// With AVX, lane i of a <c>Vector256&lt;double&gt;</c> holds the row sums of row i of a
    /// 4-row group. Each lane runs the scalar operations of its row in the scalar order, so the
    /// sums are bit-identical. Two groups run interleaved to hide the add latency.
    /// </remarks>
    private static void FloatTileRows<T>(Mat<T> image, double[] tiles, int startTileRow, int endTileRow)
        where T : unmanaged
    {
        int width = image.Width, height = image.Height;
        int xtiles = (width + TileSize - 1) / TileSize;
        T[] buffer = image.DangerousGetArray();
        int offset = image.DangerousGetOffset();
        int stride = image.Stride;
        Span<double> rowSums = stackalloc double[4 * TileSize];
        ref double sx0 = ref rowSums[0];
        ref double sx1 = ref rowSums[TileSize];
        ref double sx2 = ref rowSums[2 * TileSize];
        ref double sx3 = ref rowSums[3 * TileSize];
        for (int ty = startTileRow; ty < endTileRow; ty++)
        {
            int y0 = ty * TileSize;
            int th = Math.Min(TileSize, height - y0);
            for (int tx = 0; tx < xtiles; tx++)
            {
                int x0 = tx * TileSize;
                int tw = Math.Min(TileSize, width - x0);
                int y = 0;
                if (Avx.IsSupported)
                {
                    for (; y + 8 <= th; y += 8)
                    {
                        ref T row = ref buffer[offset + ((y0 + y) * stride) + x0];
                        RowSums8(ref row, stride, tw, out var a0, out var a1, out var a2, out var a3, out var b0, out var b1, out var b2, out var b3);
                        a0.StoreUnsafe(ref sx0, (nuint)y);
                        a1.StoreUnsafe(ref sx1, (nuint)y);
                        a2.StoreUnsafe(ref sx2, (nuint)y);
                        a3.StoreUnsafe(ref sx3, (nuint)y);
                        b0.StoreUnsafe(ref sx0, (nuint)(y + 4));
                        b1.StoreUnsafe(ref sx1, (nuint)(y + 4));
                        b2.StoreUnsafe(ref sx2, (nuint)(y + 4));
                        b3.StoreUnsafe(ref sx3, (nuint)(y + 4));
                    }
                }

                for (; y < th; y++)
                {
                    ref T row = ref buffer[offset + ((y0 + y) * stride) + x0];
                    double x0s = 0, x1s = 0, x2s = 0, x3s = 0;
                    for (int x = 0; x < tw; x++)
                    {
                        double p = Load(ref row, x);
                        double xp = x * p;
                        x0s += p;
                        x1s += xp;
                        double xxp = xp * x;
                        x2s += xxp;
                        x3s += xxp * x;
                    }

                    Unsafe.Add(ref sx0, y) = x0s;
                    Unsafe.Add(ref sx1, y) = x1s;
                    Unsafe.Add(ref sx2, y) = x2s;
                    Unsafe.Add(ref sx3, y) = x3s;
                }

                double m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0, m8 = 0, m9 = 0;
                for (y = 0; y < th; y++)
                {
                    double x0s = Unsafe.Add(ref sx0, y), x1s = Unsafe.Add(ref sx1, y);
                    double x2s = Unsafe.Add(ref sx2, y), x3s = Unsafe.Add(ref sx3, y);
                    double py = y * x0s;
                    double sy = y * y;
                    m9 += py * sy;
                    m8 += x1s * sy;
                    m7 += x2s * y;
                    m6 += x3s;
                    m5 += x0s * sy;
                    m4 += x1s * y;
                    m3 += x2s;
                    m2 += py;
                    m1 += x1s;
                    m0 += x0s;
                }

                Span<double> t = tiles.AsSpan(((ty * xtiles) + tx) * 10, 10);
                t[0] = m0;
                t[1] = m1;
                t[2] = m2;
                t[3] = m3;
                t[4] = m4;
                t[5] = m5;
                t[6] = m6;
                t[7] = m7;
                t[8] = m8;
                t[9] = m9;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Load<T>(ref T row, int x)
        where T : unmanaged
        => typeof(T) == typeof(float)
            ? Unsafe.As<T, float>(ref Unsafe.Add(ref row, x))
            : Unsafe.As<T, double>(ref Unsafe.Add(ref row, x));

    /// <summary>Row sums of 8 rows; lane i of the a vectors is row i, lane i of the b vectors is row 4 + i.</summary>
    private static void RowSums8<T>(
        ref T row,
        int stride,
        int tw,
        out Vector256<double> a0,
        out Vector256<double> a1,
        out Vector256<double> a2,
        out Vector256<double> a3,
        out Vector256<double> b0,
        out Vector256<double> b1,
        out Vector256<double> b2,
        out Vector256<double> b3)
        where T : unmanaged
    {
        ref T rowB = ref Unsafe.Add(ref row, 4 * stride);
        Vector256<double> s0 = default, s1 = default, s2 = default, s3 = default;
        Vector256<double> t0 = default, t1 = default, t2 = default, t3 = default;
        int x = 0;
        for (; x + 4 <= tw; x += 4)
        {
            Transpose4(ref row, stride, x, out var pa0, out var pa1, out var pa2, out var pa3);
            Transpose4(ref rowB, stride, x, out var pb0, out var pb1, out var pb2, out var pb3);
            Vector256<double> xd = Vector256.Create((double)x);
            Step(ref s0, ref s1, ref s2, ref s3, pa0, xd);
            Step(ref t0, ref t1, ref t2, ref t3, pb0, xd);
            xd = Vector256.Create((double)(x + 1));
            Step(ref s0, ref s1, ref s2, ref s3, pa1, xd);
            Step(ref t0, ref t1, ref t2, ref t3, pb1, xd);
            xd = Vector256.Create((double)(x + 2));
            Step(ref s0, ref s1, ref s2, ref s3, pa2, xd);
            Step(ref t0, ref t1, ref t2, ref t3, pb2, xd);
            xd = Vector256.Create((double)(x + 3));
            Step(ref s0, ref s1, ref s2, ref s3, pa3, xd);
            Step(ref t0, ref t1, ref t2, ref t3, pb3, xd);
        }

        for (; x < tw; x++)
        {
            Vector256<double> xd = Vector256.Create((double)x);
            Vector256<double> pa = Vector256.Create(
                Load(ref row, x),
                Load(ref row, x + stride),
                Load(ref row, x + (2 * stride)),
                Load(ref row, x + (3 * stride)));
            Vector256<double> pb = Vector256.Create(
                Load(ref rowB, x),
                Load(ref rowB, x + stride),
                Load(ref rowB, x + (2 * stride)),
                Load(ref rowB, x + (3 * stride)));
            Step(ref s0, ref s1, ref s2, ref s3, pa, xd);
            Step(ref t0, ref t1, ref t2, ref t3, pb, xd);
        }

        (a0, a1, a2, a3, b0, b1, b2, b3) = (s0, s1, s2, s3, t0, t1, t2, t3);
    }

    /// <summary>One scalar step <c>x0s += p; x1s += x*p; x2s += x*p*x; x3s += x*p*x*x</c> per lane, without FMA.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Step(ref Vector256<double> s0, ref Vector256<double> s1, ref Vector256<double> s2, ref Vector256<double> s3, Vector256<double> p, Vector256<double> x)
    {
        Vector256<double> xp = Avx.Multiply(x, p);
        s0 = Avx.Add(s0, p);
        s1 = Avx.Add(s1, xp);
        Vector256<double> xxp = Avx.Multiply(xp, x);
        s2 = Avx.Add(s2, xxp);
        s3 = Avx.Add(s3, Avx.Multiply(xxp, x));
    }

    /// <summary>Loads elements x..x+3 of 4 rows; <paramref name="c0"/>..<paramref name="c3"/> are the columns, lane i is row i.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4<T>(ref T row, int stride, int x, out Vector256<double> c0, out Vector256<double> c1, out Vector256<double> c2, out Vector256<double> c3)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            ref float r = ref Unsafe.As<T, float>(ref Unsafe.Add(ref row, x));
            Vector128<float> a = Vector128.LoadUnsafe(ref r);
            Vector128<float> b = Vector128.LoadUnsafe(ref r, (nuint)stride);
            Vector128<float> c = Vector128.LoadUnsafe(ref r, (nuint)(2 * stride));
            Vector128<float> d = Vector128.LoadUnsafe(ref r, (nuint)(3 * stride));
            Vector128<float> ab0 = Sse.UnpackLow(a, b), cd0 = Sse.UnpackLow(c, d);
            Vector128<float> ab1 = Sse.UnpackHigh(a, b), cd1 = Sse.UnpackHigh(c, d);
            c0 = Avx.ConvertToVector256Double(Sse.MoveLowToHigh(ab0, cd0));
            c1 = Avx.ConvertToVector256Double(Sse.MoveHighToLow(cd0, ab0));
            c2 = Avx.ConvertToVector256Double(Sse.MoveLowToHigh(ab1, cd1));
            c3 = Avx.ConvertToVector256Double(Sse.MoveHighToLow(cd1, ab1));
        }
        else
        {
            ref double r = ref Unsafe.As<T, double>(ref Unsafe.Add(ref row, x));
            Vector256<double> a = Vector256.LoadUnsafe(ref r);
            Vector256<double> b = Vector256.LoadUnsafe(ref r, (nuint)stride);
            Vector256<double> c = Vector256.LoadUnsafe(ref r, (nuint)(2 * stride));
            Vector256<double> d = Vector256.LoadUnsafe(ref r, (nuint)(3 * stride));
            Vector256<double> ab0 = Avx.UnpackLow(a, b), ab1 = Avx.UnpackHigh(a, b);
            Vector256<double> cd0 = Avx.UnpackLow(c, d), cd1 = Avx.UnpackHigh(c, d);
            c0 = Avx.Permute2x128(ab0, cd0, 0x20);
            c1 = Avx.Permute2x128(ab1, cd1, 0x20);
            c2 = Avx.Permute2x128(ab0, cd0, 0x31);
            c3 = Avx.Permute2x128(ab1, cd1, 0x31);
        }
    }
}
