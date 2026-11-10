// This implementation is ported from the OpenCV stack blur.
//
// Original source:
//   modules/imgproc/src/stackblur.cpp (stackBlur, ParallelStackBlurRow,
//     ParallelStackBlurColumn, opRow, opColumn, opComputeDiff)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/stackblur.cpp
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
/// OpenCV <c>stackBlur</c> for byte and float images with the arithmetic of a 128-bit SIMD
/// build and one column stripe (OpenCV with one thread).
/// </summary>
/// <remarks>
/// <para>Byte rows and columns use <c>(s·mul) &gt;&gt; shr</c> with a wrapping 32-bit multiply
/// in the vector part and <c>(byte)(s·(1/(r+1)²))</c> (truncation) in the scalar part. The
/// vector part of the column pass covers the first <c>16·⌊W·cn/16⌋</c> elements; the vector
/// part of a small-kernel row covers <c>16·⌊(W·cn − 2r·cn)/16⌋</c> elements from <c>r·cn</c>.
/// Float rows with a 3-tap kernel use <c>((x1 + x1) + x0) + x2</c> in the vector part
/// (<c>4·⌊(W·cn − 2·cn)/4⌋</c> elements) and <c>x1·2 + (x2 + x0)</c> elsewhere.</para>
/// <para>OpenCV splits the column pass into one stripe per thread, and each stripe has its own
/// vector/scalar split, so OpenCV's output depends on the thread count. The port always uses
/// the split of one stripe and is independent of the thread count.</para>
/// </remarks>
internal static class StackBlurEngine
{
    private const int MaxRadius = 254;
    private const int MinColumnBlockByte = 64;
    private const int MinColumnBlockFloat = 16;

    private static readonly ushort[] MulTable =
    [
        512, 512, 456, 512, 328, 456, 335, 512, 405, 328, 271, 456, 388, 335, 292, 512,
        454, 405, 364, 328, 298, 271, 496, 456, 420, 388, 360, 335, 312, 292, 273, 512,
        482, 454, 428, 405, 383, 364, 345, 328, 312, 298, 284, 271, 259, 496, 475, 456,
        437, 420, 404, 388, 374, 360, 347, 335, 323, 312, 302, 292, 282, 273, 265, 512,
        497, 482, 468, 454, 441, 428, 417, 405, 394, 383, 373, 364, 354, 345, 337, 328,
        320, 312, 305, 298, 291, 284, 278, 271, 265, 259, 507, 496, 485, 475, 465, 456,
        446, 437, 428, 420, 412, 404, 396, 388, 381, 374, 367, 360, 354, 347, 341, 335,
        329, 323, 318, 312, 307, 302, 297, 292, 287, 282, 278, 273, 269, 265, 261, 512,
        505, 497, 489, 482, 475, 468, 461, 454, 447, 441, 435, 428, 422, 417, 411, 405,
        399, 394, 389, 383, 378, 373, 368, 364, 359, 354, 350, 345, 341, 337, 332, 328,
        324, 320, 316, 312, 309, 305, 301, 298, 294, 291, 287, 284, 281, 278, 274, 271,
        268, 265, 262, 259, 257, 507, 501, 496, 491, 485, 480, 475, 470, 465, 460, 456,
        451, 446, 442, 437, 433, 428, 424, 420, 416, 412, 408, 404, 400, 396, 392, 388,
        385, 381, 377, 374, 370, 367, 363, 360, 357, 354, 350, 347, 344, 341, 338, 335,
        332, 329, 326, 323, 320, 318, 315, 312, 310, 307, 304, 302, 299, 297, 294, 292,
        289, 287, 285, 282, 280, 278, 275, 273, 271, 269, 267, 265, 263, 261, 259,
    ];

    private static readonly byte[] ShrTable =
    [
        9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17,
        17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
        22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
        22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
    ];

    public static Mat<T> Apply<T>(Mat<T> src, Size ksize)
        where T : unmanaged
    {
        int w = src.Width, h = src.Height;
        int kw = ksize.Width, kh = ksize.Height;
        if (kw > w)
        {
            kw = w % 2 == 0 ? Math.Max(1, w - 1) : w;
        }

        if (kh > h)
        {
            kh = h % 2 == 0 ? Math.Max(1, h - 1) : h;
        }

        if (typeof(T) == typeof(byte) && kh != 1 && !ParallelRows.IsSerial(h, (long)w * src.Channels * 8))
        {
            return (Mat<T>)(object)ApplyByteParallel((Mat<byte>)(object)src, kw, kh);
        }

        Mat<T> dst;
        if (kw == 1)
        {
            dst = src.Clone();
            if (kh == 1)
            {
                return dst;
            }
        }
        else
        {
            dst = new Mat<T>(w, h, src.Channels, clear: false);
            int rw = kw / 2;
            long work = (long)w * src.Channels * (kw <= 9 ? kw : 8);
            if (!ParallelRows.IsSerial(h, work))
            {
                PageTouch.Rows(dst);
            }

            if (typeof(T) == typeof(byte))
            {
                var s = (Mat<byte>)(object)src;
                var d = (Mat<byte>)(object)dst;
                ParallelRows.For(h, work, (y0, y1) => RowsByte(s, d, rw, y0, y1));
            }
            else
            {
                var s = (Mat<float>)(object)src;
                var d = (Mat<float>)(object)dst;
                ParallelRows.For(h, work, (y0, y1) => RowsFloat(s, d, rw, y0, y1));
            }
        }

        if (kh != 1)
        {
            int rh = kh / 2;
            int width = w * src.Channels;

            // Narrow blocks give small images enough stripes; columns are independent, so the block
            // width does not change the result.
            int block = typeof(T) == typeof(byte) ? MinColumnBlockByte : MinColumnBlockFloat;
            int blocks = (width + block - 1) / block;
            long work = (long)block * h * 8;
            if (typeof(T) == typeof(byte))
            {
                var d = (Mat<byte>)(object)dst;
                ParallelRows.For(blocks, work, (b0, b1) => ColumnsByte(d, d, rh, b0 * block, Math.Min(b1 * block, width), 0, h));
            }
            else
            {
                var d = (Mat<float>)(object)dst;
                ParallelRows.For(blocks, work, (b0, b1) => ColumnsFloat(d, rh, b0 * block, Math.Min(b1 * block, width)));
            }
        }

        return dst;
    }

    /// <summary>
    /// The byte filter with the row pass into a pooled image and the column pass in tiles of
    /// <see cref="MinColumnBlockByte"/> columns and row bands, so that small images give every
    /// thread work. The result equals the single-stripe result (see <see cref="ColumnsByte"/>).
    /// </summary>
    private static Mat<byte> ApplyByteParallel(Mat<byte> src, int kw, int kh)
    {
        int w = src.Width, h = src.Height, cn = src.Channels;
        int width = w * cn;
        var dst = new Mat<byte>(w, h, cn, clear: false);
        PageTouch.Rows(dst);
        PooledMat<byte> pooled = default;
        try
        {
            Mat<byte> mid = src;
            if (kw != 1)
            {
                pooled = PooledMat<byte>.Rent(w, h, cn);
                mid = pooled.Mat;
                int rw = kw / 2;
                Mat<byte> rows = mid;
                ParallelRows.For(h, (long)width * (kw <= 9 ? kw : 8), (y0, y1) => RowsByte(src, rows, rw, y0, y1));
            }

            int rh = kh / 2;
            int blocks = (width + MinColumnBlockByte - 1) / MinColumnBlockByte;
            int dop = CvParallel.MaxDegreeOfParallelism;
            int bands = Math.Clamp(((2 * dop) + blocks - 1) / blocks, 1, Math.Max(1, h / (2 * kh)));
            int items = blocks * bands;
            Mat<byte> input = mid;
            ParallelRows.For(items, (long)MinColumnBlockByte * h * 8 / bands, (i0, i1) =>
            {
                for (int item = i0; item < i1; item++)
                {
                    int b = item % blocks, band = item / blocks;
                    int c0 = b * MinColumnBlockByte;
                    ColumnsByte(input, dst, rh, c0, Math.Min(c0 + MinColumnBlockByte, width), (int)((long)h * band / bands), (int)((long)h * (band + 1) / bands));
                }
            });
        }
        finally
        {
            if (kw != 1)
            {
                pooled.Dispose();
            }
        }

        return dst;
    }

    private static float MulVal(int radius) => 1.0f / ((radius + 1) * (radius + 1));

    /// <summary>
    /// The byte division of the vector part: wrapping <c>s·mul</c>, arithmetic shift, int32→int16
    /// saturation, reinterpretation as uint16 and uint16→uint8 saturation (negative values give 255).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte SimdDivide(int s, int mul, int shr)
    {
        int q = unchecked(s * mul) >> shr;
        return q < 0 ? (byte)255 : (byte)Math.Min(q, 255);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> SimdDivide(Vector<int> s, Vector<int> mul, int shr)
    {
        Vector<int> q = Vector.ShiftRightArithmetic(s * mul, shr);
        return Vector.ConditionalSelect(Vector.LessThan(q, Vector<int>.Zero), new Vector<int>(255), Vector.Min(q, new Vector<int>(255)));
    }

    // ------------------------------------------------------------ byte rows

    private static void RowsByte(Mat<byte> src, Mat<byte> dst, int radius, int y0, int y1)
    {
        int cn = src.Channels;
        int w = src.Width;
        int ksize = (2 * radius) + 1;
        if (ksize <= 9 && w > ksize)
        {
            for (int y = y0; y < y1; y++)
            {
                SmallRowByte(src.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), radius, cn, w);
            }

            return;
        }

        if (UseTriangularRows(radius, w))
        {
            if (radius <= 15 && Vector.IsHardwareAccelerated && w * cn >= Vector<byte>.Count)
            {
                TriangularRowsByte16(src, dst, radius, y0, y1);
            }
            else
            {
                TriangularRowsByte(src, dst, radius, y0, y1);
            }

            return;
        }

        LargeRows<byte, int>(src, dst, radius, y0, y1);
    }

    private static void SmallRowByte(ReadOnlySpan<byte> s, Span<byte> d, int radius, int cn, int w)
    {
        int ksize = (2 * radius) + 1;
        float mulVal = MulVal(radius);
        Span<int> kv = stackalloc int[ksize];
        for (int i = 0; i < ksize; i++)
        {
            kv[i] = i <= radius ? i + 1 : (2 * radius) - i + 1;
        }

        ReadOnlySpan<int> kx = kv[radius..];
        int wm = w - 1;
        for (int i = 0; i < radius; i++)
        {
            for (int ci = 0; ci < cn; ci++)
            {
                int sum = 0;
                for (int k = 0; k < ksize; k++)
                {
                    sum += s[(Math.Max(k - radius + i, 0) * cn) + ci] * kv[k];
                }

                d[(i * cn) + ci] = (byte)(int)(sum * mulVal);
            }
        }

        int widthCn = (w - radius) * cn;
        int wc = radius * cn;
        int mul = MulTable[radius], shr = ShrTable[radius];
        int simdEnd = wc + ((widthCn - wc) & ~15);
        ref byte sr = ref MemoryMarshal.GetReference(s);
        ref byte dr = ref MemoryMarshal.GetReference(d);
        if (Vector.IsHardwareAccelerated)
        {
            int vb = Vector<byte>.Count;
            var vmul = new Vector<int>(mul);
            for (; wc <= simdEnd - vb; wc += vb)
            {
                FilterVectors.WidenToInt32(Vector.LoadUnsafe(ref sr, (nuint)wc), out Vector<int> a0, out Vector<int> a1, out Vector<int> a2, out Vector<int> a3);
                var k0 = new Vector<int>(kx[0]);
                a0 *= k0;
                a1 *= k0;
                a2 *= k0;
                a3 *= k0;
                for (int k = 1; k <= radius; k++)
                {
                    FilterVectors.WidenToInt32(Vector.LoadUnsafe(ref sr, (nuint)(wc - (k * cn))), out Vector<int> l0, out Vector<int> l1, out Vector<int> l2, out Vector<int> l3);
                    FilterVectors.WidenToInt32(Vector.LoadUnsafe(ref sr, (nuint)(wc + (k * cn))), out Vector<int> r0, out Vector<int> r1, out Vector<int> r2, out Vector<int> r3);
                    var kk = new Vector<int>(kx[k]);
                    a0 += (l0 + r0) * kk;
                    a1 += (l1 + r1) * kk;
                    a2 += (l2 + r2) * kk;
                    a3 += (l3 + r3) * kk;
                }

                FilterVectors.SaturateToByte(
                    SimdDivide(a0, vmul, shr), SimdDivide(a1, vmul, shr), SimdDivide(a2, vmul, shr), SimdDivide(a3, vmul, shr))
                    .StoreUnsafe(ref dr, (nuint)wc);
            }
        }

        for (; wc < widthCn; wc++)
        {
            int sum = Unsafe.Add(ref sr, wc) * kx[0];
            for (int k = 1; k <= radius; k++)
            {
                sum += (Unsafe.Add(ref sr, wc + (k * cn)) + Unsafe.Add(ref sr, wc - (k * cn))) * kx[k];
            }

            Unsafe.Add(ref dr, wc) = wc < simdEnd ? SimdDivide(sum, mul, shr) : (byte)(int)(sum * mulVal);
        }

        for (int i = wc / cn; i < w; i++)
        {
            for (int ci = 0; ci < cn; ci++)
            {
                int sum = 0;
                for (int k = 0; k < ksize; k++)
                {
                    sum += s[(Math.Min(k - radius + i, wm) * cn) + ci] * kv[k];
                }

                d[(i * cn) + ci] = (byte)(int)(sum * mulVal);
            }
        }
    }

    // ----------------------------------------------------------- float rows

    private static void RowsFloat(Mat<float> src, Mat<float> dst, int radius, int y0, int y1)
    {
        int cn = src.Channels;
        int w = src.Width;
        int ksize = (2 * radius) + 1;
        if (ksize <= 9 && w > ksize)
        {
            for (int y = y0; y < y1; y++)
            {
                SmallRowFloat(src.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), radius, cn, w);
            }

            return;
        }

        LargeRows<float, float>(src, dst, radius, y0, y1);
    }

    private static void SmallRowFloat(ReadOnlySpan<float> s, Span<float> d, int radius, int cn, int w)
    {
        int ksize = (2 * radius) + 1;
        float mulVal = MulVal(radius);
        Span<int> kv = stackalloc int[ksize];
        for (int i = 0; i < ksize; i++)
        {
            kv[i] = i <= radius ? i + 1 : (2 * radius) - i + 1;
        }

        ReadOnlySpan<int> kx = kv[radius..];
        int wm = w - 1;
        for (int i = 0; i < radius; i++)
        {
            for (int ci = 0; ci < cn; ci++)
            {
                float sum = 0;
                for (int k = 0; k < ksize; k++)
                {
                    sum += s[(Math.Max(k - radius + i, 0) * cn) + ci] * (float)kv[k];
                }

                d[(i * cn) + ci] = sum * mulVal;
            }
        }

        int widthCn = (w - radius) * cn;
        int wc = radius * cn;
        int simdEnd = wc + ((widthCn - wc) & ~3);
        ref float sr = ref MemoryMarshal.GetReference(s);
        ref float dr = ref MemoryMarshal.GetReference(d);
        if (Vector.IsHardwareAccelerated)
        {
            int vf = Vector<float>.Count;
            var vm = new Vector<float>(mulVal);
            int end = ksize == 3 ? simdEnd : widthCn;
            for (; wc <= end - vf; wc += vf)
            {
                Vector<float> x = Vector.LoadUnsafe(ref sr, (nuint)wc);
                Vector<float> acc;
                if (ksize == 3)
                {
                    acc = ((x + x) + Vector.LoadUnsafe(ref sr, (nuint)(wc - cn))) + Vector.LoadUnsafe(ref sr, (nuint)(wc + cn));
                }
                else
                {
                    acc = x * new Vector<float>(kx[0]);
                    for (int k = 1; k <= radius; k++)
                    {
                        acc += (Vector.LoadUnsafe(ref sr, (nuint)(wc - (k * cn))) + Vector.LoadUnsafe(ref sr, (nuint)(wc + (k * cn)))) * new Vector<float>(kx[k]);
                    }
                }

                (acc * vm).StoreUnsafe(ref dr, (nuint)wc);
            }
        }

        for (; wc < widthCn; wc++)
        {
            float x = Unsafe.Add(ref sr, wc);
            float sum;
            if (ksize == 3 && wc < simdEnd)
            {
                sum = ((x + x) + Unsafe.Add(ref sr, wc - cn)) + Unsafe.Add(ref sr, wc + cn);
            }
            else
            {
                sum = x * kx[0];
                for (int k = 1; k <= radius; k++)
                {
                    sum += (Unsafe.Add(ref sr, wc + (k * cn)) + Unsafe.Add(ref sr, wc - (k * cn))) * kx[k];
                }
            }

            Unsafe.Add(ref dr, wc) = sum * mulVal;
        }

        for (int i = wc / cn; i < w; i++)
        {
            for (int ci = 0; ci < cn; ci++)
            {
                float sum = 0;
                for (int k = 0; k < ksize; k++)
                {
                    sum += s[(Math.Min(k - radius + i, wm) * cn) + ci] * (float)kv[k];
                }

                d[(i * cn) + ci] = sum * mulVal;
            }
        }
    }

    /// <summary>Selects the vectorized triangular sums for the byte running-sum row pass.</summary>
    internal static bool TriangularRowsEnabled { get; set; } = true;

    private static bool UseTriangularRows(int radius, int w) => TriangularRowsEnabled && radius <= 32 && w >= (2 * radius) + 3;

    /// <summary>
    /// The byte running-sum row pass computed directly. Its int32 sums are exact and equal the
    /// triangular sum <c>Σ (r + 1 − |j|)·x[clamp(i + j)]</c>, which is a box of width r + 1
    /// applied twice to the replicate-padded row. The output rounds <c>(float)sum·mulVal</c> as
    /// OpenCV does.
    /// </summary>
    private static void TriangularRowsByte(Mat<byte> src, Mat<byte> dst, int radius, int y0, int y1)
    {
        int cn = src.Channels;
        int w = src.Width;
        int n = w * cn;
        int padded = (w + (2 * radius)) * cn;
        int boxLen = (w + radius) * cn;
        float mulVal = MulVal(radius);
        int[] pBuf = ArrayPool<int>.Shared.Rent(padded);
        int[] bBuf = ArrayPool<int>.Shared.Rent(boxLen);
        int[] tBuf = ArrayPool<int>.Shared.Rent(n);
        try
        {
            ref int pr = ref pBuf[0];
            ref int br = ref bBuf[0];
            ref int tr = ref tBuf[0];
            int vi = Vector<int>.Count;
            for (int y = y0; y < y1; y++)
            {
                ReadOnlySpan<byte> s = src.GetReadOnlyRowSpan(y);
                for (int x = 0; x < radius; x++)
                {
                    for (int c = 0; c < cn; c++)
                    {
                        Unsafe.Add(ref pr, (x * cn) + c) = s[c];
                        Unsafe.Add(ref pr, ((radius + w + x) * cn) + c) = s[((w - 1) * cn) + c];
                    }
                }

                ref int mid = ref Unsafe.Add(ref pr, radius * cn);
                int e = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    ref byte sr = ref MemoryMarshal.GetReference(s);
                    for (; e <= n - Vector<byte>.Count; e += Vector<byte>.Count)
                    {
                        FilterVectors.WidenToInt32(Vector.LoadUnsafe(ref sr, (nuint)e), out Vector<int> a0, out Vector<int> a1, out Vector<int> a2, out Vector<int> a3);
                        a0.StoreUnsafe(ref mid, (nuint)e);
                        a1.StoreUnsafe(ref mid, (nuint)(e + vi));
                        a2.StoreUnsafe(ref mid, (nuint)(e + (2 * vi)));
                        a3.StoreUnsafe(ref mid, (nuint)(e + (3 * vi)));
                    }
                }

                for (; e < n; e++)
                {
                    Unsafe.Add(ref mid, e) = s[e];
                }

                BoxSum(ref pr, ref br, boxLen, radius + 1, cn);
                BoxSum(ref br, ref tr, n, radius + 1, cn);

                ref byte d = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    var vm = new Vector<float>(mulVal);
                    for (; i <= n - Vector<byte>.Count; i += Vector<byte>.Count)
                    {
                        FilterVectors.SaturateToByte(
                            FilterVectors.CvRound(Vector.ConvertToSingle(Vector.LoadUnsafe(ref tr, (nuint)i)) * vm),
                            FilterVectors.CvRound(Vector.ConvertToSingle(Vector.LoadUnsafe(ref tr, (nuint)(i + vi))) * vm),
                            FilterVectors.CvRound(Vector.ConvertToSingle(Vector.LoadUnsafe(ref tr, (nuint)(i + (2 * vi)))) * vm),
                            FilterVectors.CvRound(Vector.ConvertToSingle(Vector.LoadUnsafe(ref tr, (nuint)(i + (3 * vi)))) * vm))
                            .StoreUnsafe(ref d, (nuint)i);
                    }
                }

                for (; i < n; i++)
                {
                    Unsafe.Add(ref d, i) = Saturate.FromFloat<byte>(Unsafe.Add(ref tr, i) * mulVal);
                }

                RunningTail(s, ref tr, ref d, radius, cn, w, mulVal);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(tBuf);
            ArrayPool<int>.Shared.Return(bBuf);
            ArrayPool<int>.Shared.Return(pBuf);
        }
    }

    /// <summary>
    /// <see cref="TriangularRowsByte"/> in 16-bit lanes. For radius ≤ 15 the triangular sums are at
    /// most <c>255·16² = 65280</c>, so the uint16 box sums are exact and equal the int32 sums.
    /// </summary>
    private static void TriangularRowsByte16(Mat<byte> src, Mat<byte> dst, int radius, int y0, int y1)
    {
        int cn = src.Channels;
        int w = src.Width;
        int n = w * cn;
        int padded = (w + (2 * radius)) * cn;
        int boxLen = (w + radius) * cn;
        int len = radius + 1;
        float mulVal = MulVal(radius);
        int vb = Vector<byte>.Count, vs = Vector<ushort>.Count, vi = Vector<int>.Count;
        ushort[] pBuf = ArrayPool<ushort>.Shared.Rent(padded);
        ushort[] bBuf = ArrayPool<ushort>.Shared.Rent(boxLen);
        int[] tBuf = ArrayPool<int>.Shared.Rent(n);
        try
        {
            ref ushort pr = ref pBuf[0];
            ref ushort br = ref bBuf[0];
            ref int tr = ref tBuf[0];
            var vm = new Vector<float>(mulVal);
            for (int y = y0; y < y1; y++)
            {
                ReadOnlySpan<byte> s = src.GetReadOnlyRowSpan(y);
                ref byte sr = ref MemoryMarshal.GetReference(s);
                for (int x = 0; x < radius; x++)
                {
                    for (int c = 0; c < cn; c++)
                    {
                        Unsafe.Add(ref pr, (x * cn) + c) = s[c];
                        Unsafe.Add(ref pr, ((radius + w + x) * cn) + c) = s[((w - 1) * cn) + c];
                    }
                }

                ref ushort mid = ref Unsafe.Add(ref pr, radius * cn);
                int e = 0;
                for (; e <= n - vb; e += vb)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref sr, (nuint)e), out Vector<ushort> lo, out Vector<ushort> hi);
                    lo.StoreUnsafe(ref mid, (nuint)e);
                    hi.StoreUnsafe(ref mid, (nuint)(e + vs));
                }

                for (; e < n; e++)
                {
                    Unsafe.Add(ref mid, e) = Unsafe.Add(ref sr, e);
                }

                for (e = 0; e < boxLen; e += vs)
                {
                    // The final block overlaps the previous one; it rewrites equal values.
                    e = Math.Min(e, boxLen - vs);
                    Vector<ushort> a = Vector.LoadUnsafe(ref pr, (nuint)e);
                    for (int t = 1; t < len; t++)
                    {
                        a += Vector.LoadUnsafe(ref pr, (nuint)(e + (t * cn)));
                    }

                    a.StoreUnsafe(ref br, (nuint)e);
                }

                ref byte d = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
                int i = 0;
                for (; i <= n - vb; i += vb)
                {
                    Vector<ushort> t0 = Vector.LoadUnsafe(ref br, (nuint)i);
                    Vector<ushort> t1 = Vector.LoadUnsafe(ref br, (nuint)(i + vs));
                    for (int t = 1; t < len; t++)
                    {
                        t0 += Vector.LoadUnsafe(ref br, (nuint)(i + (t * cn)));
                        t1 += Vector.LoadUnsafe(ref br, (nuint)(i + (t * cn) + vs));
                    }

                    Vector.Widen(t0, out Vector<uint> u0, out Vector<uint> u1);
                    Vector.Widen(t1, out Vector<uint> u2, out Vector<uint> u3);
                    FilterVectors.SaturateToByte(
                        FilterVectors.CvRound(Vector.ConvertToSingle(Vector.AsVectorInt32(u0)) * vm),
                        FilterVectors.CvRound(Vector.ConvertToSingle(Vector.AsVectorInt32(u1)) * vm),
                        FilterVectors.CvRound(Vector.ConvertToSingle(Vector.AsVectorInt32(u2)) * vm),
                        FilterVectors.CvRound(Vector.ConvertToSingle(Vector.AsVectorInt32(u3)) * vm))
                        .StoreUnsafe(ref d, (nuint)i);
                }

                for (; i < n; i++)
                {
                    Unsafe.Add(ref d, i) = Saturate.FromFloat<byte>(TriangularSum(ref br, i, len, cn) * mulVal);
                }

                // RunningTail starts from the exact sums of pixels w − r − 3 and w − r − 2.
                int t0Index = (w - radius - 3) * cn;
                for (int k = t0Index; k < t0Index + (2 * cn); k++)
                {
                    Unsafe.Add(ref tr, k) = TriangularSum(ref br, k, len, cn);
                }

                RunningTail(s, ref tr, ref d, radius, cn, w, mulVal);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(tBuf);
            ArrayPool<ushort>.Shared.Return(bBuf);
            ArrayPool<ushort>.Shared.Return(pBuf);
        }
    }

    private static int TriangularSum(ref ushort box, int i, int len, int cn)
    {
        int sum = 0;
        for (int t = 0; t < len; t++)
        {
            sum += Unsafe.Add(ref box, i + (t * cn));
        }

        return sum;
    }

    /// <summary>
    /// The last <c>r + 3</c> pixels of a row. OpenCV's difference buffer stops following the
    /// replicate border near the right end (the <c>dist</c> rule of stackblur.cpp:595-606), so these
    /// pixels repeat OpenCV's recurrence <c>sum += dv; dv += diff[k] − diff[k − (r+1)·cn]</c>,
    /// starting from the exact triangular sums of pixels <c>i0</c> and <c>i0 + 1</c>.
    /// </summary>
    private static void RunningTail(ReadOnlySpan<byte> s, ref int tr, ref byte d, int radius, int cn, int w, float mulVal)
    {
        int i0 = w - radius - 3;
        int cnr1 = cn * (radius + 1);
        int widthCn = (w - radius - 1) * cn;
        int tailStart = (radius * cn) + widthCn;
        Span<int> q0s = stackalloc int[radius];
        int q0 = widthCn, dist = w - (widthCn / cn);
        for (int r = 0; r < radius; r++)
        {
            q0s[r] = q0;
            if (dist >= r)
            {
                q0 += cn;
                dist--;
            }
        }

        int last = (w - 1) * cn;
        for (int c = 0; c < cn; c++)
        {
            int sum = Unsafe.Add(ref tr, (i0 * cn) + c);
            int dv = Unsafe.Add(ref tr, ((i0 + 1) * cn) + c) - sum;
            for (int i = i0; i < w; i++)
            {
                Unsafe.Add(ref d, (i * cn) + c) = Saturate.FromFloat<byte>(sum * mulVal);
                sum += dv;
                dv += TailDiff(s, q0s, cnr1 + (i * cn) + c, radius, cn, widthCn, tailStart, last)
                    - TailDiff(s, q0s, (i * cn) + c, radius, cn, widthCn, tailStart, last);
            }
        }
    }

    /// <summary>Element <paramref name="e"/> of OpenCV's difference buffer (middle and tail part).</summary>
    private static int TailDiff(ReadOnlySpan<byte> s, ReadOnlySpan<int> q0s, int e, int radius, int cn, int widthCn, int tailStart, int last)
    {
        int m = e - (radius * cn);
        if (m < widthCn)
        {
            return s[m + (cn * (radius + 1))] - s[m];
        }

        int t = e - tailStart;
        if (t < radius * cn)
        {
            int c = t % cn;
            return s[last + c] - s[q0s[t / cn] + c];
        }

        return 0;
    }

    /// <summary><c>dst[e] = Σ_{t &lt; len} src[e + t·cn]</c> for <c>e &lt; count</c> (exact int32).</summary>
    private static void BoxSum(ref int src, ref int dst, int count, int len, int cn)
    {
        int e = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vi = Vector<int>.Count;
            for (; e <= count - (2 * vi); e += 2 * vi)
            {
                Vector<int> a0 = Vector.LoadUnsafe(ref src, (nuint)e);
                Vector<int> a1 = Vector.LoadUnsafe(ref src, (nuint)(e + vi));
                for (int t = 1; t < len; t++)
                {
                    a0 += Vector.LoadUnsafe(ref src, (nuint)(e + (t * cn)));
                    a1 += Vector.LoadUnsafe(ref src, (nuint)(e + (t * cn) + vi));
                }

                a0.StoreUnsafe(ref dst, (nuint)e);
                a1.StoreUnsafe(ref dst, (nuint)(e + vi));
            }
        }

        for (; e < count; e++)
        {
            int acc = 0;
            for (int t = 0; t < len; t++)
            {
                acc += Unsafe.Add(ref src, e + (t * cn));
            }

            Unsafe.Add(ref dst, e) = acc;
        }
    }

    // ----------------------------------------------------------- large rows

    /// <summary>
    /// The running-sum row pass (kernel above 9 or image not wider than the kernel) for the rows
    /// <c>[y0, y1)</c>. Single-channel rows run four at a time; each row keeps OpenCV's sequence.
    /// </summary>
    private static void LargeRows<T, TBuf>(Mat<T> src, Mat<T> dst, int radius, int y0, int y1)
        where T : unmanaged
        where TBuf : unmanaged, INumber<TBuf>
    {
        int cn = src.Channels;
        int w = src.Width;
        int ksize = (2 * radius) + 1;
        int stride = (cn * (w + ksize)) + (2 * cn);
        TBuf[] buffer = ArrayPool<TBuf>.Shared.Rent(4 * stride);
        try
        {
            for (int y = y0; y < y1; y += 4)
            {
                int count = Math.Min(4, y1 - y);
                for (int r = 0; r < count; r++)
                {
                    SetupLargeRow(src.GetReadOnlyRowSpan(y + r), buffer.AsSpan(r * stride, stride), radius, cn, w);
                }

                if (cn == 1 && count == 4)
                {
                    if (typeof(T) == typeof(byte))
                    {
                        EmitLargeRows4Byte((int[])(object)buffer, stride, (Mat<byte>)(object)dst, y, radius, w);
                    }
                    else
                    {
                        EmitLargeRows4(buffer, stride, dst, y, radius, w);
                    }

                    continue;
                }

                for (int r = 0; r < count; r++)
                {
                    EmitLargeRow(buffer.AsSpan(r * stride, stride), dst.GetRowSpan(y + r), radius, cn, w);
                }
            }
        }
        finally
        {
            ArrayPool<TBuf>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Fills <c>diffVal</c> (elements <c>[0, cn)</c>), <c>sum</c> (<c>[cn, 2cn)</c>) and the
    /// difference buffer (from <c>2cn</c>) of one row, up to the output loop of
    /// ParallelStackBlurRow (stackblur.cpp:560-618).
    /// </summary>
    private static void SetupLargeRow<T, TBuf>(ReadOnlySpan<T> s, Span<TBuf> buf, int radius, int cn, int w)
        where T : unmanaged
        where TBuf : unmanaged, INumber<TBuf>
    {
        buf.Clear();
        Span<TBuf> diffVal = buf[..cn];
        Span<TBuf> sum = buf.Slice(cn, cn);
        Span<TBuf> diff = buf[(2 * cn)..];
        int wm = w - 1;
        int cnr1 = cn * (radius + 1);
        int widthCn = (w - radius - 1) * cn;
        TBuf radiusMul = TBuf.CreateTruncating((radius + 2) * (radius + 1) / 2);
        for (int ci = 0; ci < cn; ci++)
        {
            sum[ci] += V<T, TBuf>(s[ci]) * radiusMul;
        }

        int p0 = 0;
        for (int i = 0; i < radius; i++)
        {
            if (i < wm)
            {
                p0 += cn;
            }

            TBuf weight = TBuf.CreateTruncating(radius - i);
            for (int ci = 0; ci < cn; ci++)
            {
                TBuf v = V<T, TBuf>(s[p0 + ci]);
                diff[(i * cn) + ci] = v - V<T, TBuf>(s[ci]);
                diffVal[ci] += diff[(i * cn) + ci];
                sum[ci] += v * weight;
            }
        }

        int baseIndex = radius * cn;
        int index = 0;
        ref T sr = ref MemoryMarshal.GetReference(s);
        ref TBuf dr = ref Unsafe.Add(ref MemoryMarshal.GetReference(diff), baseIndex);
        for (; index < widthCn; index++)
        {
            Unsafe.Add(ref dr, index) = V<T, TBuf>(Unsafe.Add(ref sr, index + cnr1)) - V<T, TBuf>(Unsafe.Add(ref sr, index));
        }

        int pos = baseIndex + index;
        int q0 = index, q1 = (w - 1) * cn;
        int dist = w - (index / cn);
        for (int r = 0; r < radius; r++, pos += cn)
        {
            for (int ci = 0; ci < cn; ci++)
            {
                diff[pos + ci] = V<T, TBuf>(s[q1 + ci]) - V<T, TBuf>(s[q0 + ci]);
            }

            if (dist >= r)
            {
                q0 += cn;
                dist--;
            }
        }

        for (int ci = 0; ci < cn; ci++)
        {
            diffVal[ci] += diff[baseIndex + ci];
        }
    }

    /// <summary>The output loop of one row; channels are independent and run one after another.</summary>
    private static void EmitLargeRow<T, TBuf>(Span<TBuf> buf, Span<T> d, int radius, int cn, int w)
        where T : unmanaged
        where TBuf : unmanaged, INumber<TBuf>
    {
        float mulVal = MulVal(radius);
        int cnr1 = cn * (radius + 1);
        ref TBuf df = ref buf[2 * cn];
        ref T dr = ref MemoryMarshal.GetReference(d);
        int d0 = (radius * cn) + cn;
        if (cn == 3)
        {
            TBuf s0 = buf[3], s1 = buf[4], s2 = buf[5], v0 = buf[0], v1 = buf[1], v2 = buf[2];
            for (int i = 0, k = d0; i < w; i++, k += 3)
            {
                ref T o = ref Unsafe.Add(ref dr, 3 * i);
                o = Out<T, TBuf>(s0, mulVal);
                Unsafe.Add(ref o, 1) = Out<T, TBuf>(s1, mulVal);
                Unsafe.Add(ref o, 2) = Out<T, TBuf>(s2, mulVal);
                s0 += v0;
                s1 += v1;
                s2 += v2;
                v0 += Unsafe.Add(ref df, k) - Unsafe.Add(ref df, k - cnr1);
                v1 += Unsafe.Add(ref df, k + 1) - Unsafe.Add(ref df, k + 1 - cnr1);
                v2 += Unsafe.Add(ref df, k + 2) - Unsafe.Add(ref df, k + 2 - cnr1);
            }

            return;
        }

        for (int ci = 0; ci < cn; ci++)
        {
            TBuf s0 = buf[cn + ci], dv = buf[ci];
            int k = d0 + ci;
            for (int i = 0; i < w; i++, k += cn)
            {
                Unsafe.Add(ref dr, (i * cn) + ci) = Out<T, TBuf>(s0, mulVal);
                s0 += dv;
                dv += Unsafe.Add(ref df, k) - Unsafe.Add(ref df, k - cnr1);
            }
        }
    }

    /// <summary>The output loops of four single-channel rows, interleaved.</summary>
    private static void EmitLargeRows4<T, TBuf>(TBuf[] buffer, int stride, Mat<T> dst, int y, int radius, int w)
        where T : unmanaged
        where TBuf : unmanaged, INumber<TBuf>
    {
        float mulVal = MulVal(radius);
        int cnr1 = radius + 1;
        ref TBuf b0 = ref buffer[0];
        ref TBuf b1 = ref buffer[stride];
        ref TBuf b2 = ref buffer[2 * stride];
        ref TBuf b3 = ref buffer[3 * stride];
        ref T o0 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
        ref T o1 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y + 1));
        ref T o2 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y + 2));
        ref T o3 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y + 3));
        TBuf s0 = Unsafe.Add(ref b0, 1), s1 = Unsafe.Add(ref b1, 1), s2 = Unsafe.Add(ref b2, 1), s3 = Unsafe.Add(ref b3, 1);
        TBuf v0 = b0, v1 = b1, v2 = b2, v3 = b3;
        int k = 2 + radius + 1;
        for (int i = 0; i < w; i++, k++)
        {
            Unsafe.Add(ref o0, i) = Out<T, TBuf>(s0, mulVal);
            Unsafe.Add(ref o1, i) = Out<T, TBuf>(s1, mulVal);
            Unsafe.Add(ref o2, i) = Out<T, TBuf>(s2, mulVal);
            Unsafe.Add(ref o3, i) = Out<T, TBuf>(s3, mulVal);
            s0 += v0;
            s1 += v1;
            s2 += v2;
            s3 += v3;
            v0 += Unsafe.Add(ref b0, k) - Unsafe.Add(ref b0, k - cnr1);
            v1 += Unsafe.Add(ref b1, k) - Unsafe.Add(ref b1, k - cnr1);
            v2 += Unsafe.Add(ref b2, k) - Unsafe.Add(ref b2, k - cnr1);
            v3 += Unsafe.Add(ref b3, k) - Unsafe.Add(ref b3, k - cnr1);
        }
    }

    /// <summary><see cref="EmitLargeRows4{T, TBuf}"/> for byte rows with int32 sums.</summary>
    private static void EmitLargeRows4Byte(int[] buffer, int stride, Mat<byte> dst, int y, int radius, int w)
    {
        float mulVal = MulVal(radius);
        int cnr1 = radius + 1;
        ref int b0 = ref buffer[0];
        ref int b1 = ref buffer[stride];
        ref int b2 = ref buffer[2 * stride];
        ref int b3 = ref buffer[3 * stride];
        ref byte o0 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
        ref byte o1 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y + 1));
        ref byte o2 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y + 2));
        ref byte o3 = ref MemoryMarshal.GetReference(dst.GetRowSpan(y + 3));
        int s0 = Unsafe.Add(ref b0, 1), s1 = Unsafe.Add(ref b1, 1), s2 = Unsafe.Add(ref b2, 1), s3 = Unsafe.Add(ref b3, 1);
        int v0 = b0, v1 = b1, v2 = b2, v3 = b3;
        int k = 2 + radius + 1;
        for (int i = 0; i < w; i++, k++)
        {
            Unsafe.Add(ref o0, i) = RoundByte(s0 * mulVal);
            Unsafe.Add(ref o1, i) = RoundByte(s1 * mulVal);
            Unsafe.Add(ref o2, i) = RoundByte(s2 * mulVal);
            Unsafe.Add(ref o3, i) = RoundByte(s3 * mulVal);
            s0 += v0;
            s1 += v1;
            s2 += v2;
            s3 += v3;
            v0 += Unsafe.Add(ref b0, k) - Unsafe.Add(ref b0, k - cnr1);
            v1 += Unsafe.Add(ref b1, k) - Unsafe.Add(ref b1, k - cnr1);
            v2 += Unsafe.Add(ref b2, k) - Unsafe.Add(ref b2, k - cnr1);
            v3 += Unsafe.Add(ref b3, k) - Unsafe.Add(ref b3, k - cnr1);
        }
    }

    /// <summary><c>saturate_cast&lt;uchar&gt;(float)</c>; adding and subtracting 1.5·2^23 rounds half to even for 0 ≤ v &lt; 2^22.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte RoundByte(float v)
        => v >= 0f && v < 255.5f ? (byte)(int)((v + 12582912f) - 12582912f) : Saturate.FromFloat<byte>(v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TBuf V<T, TBuf>(T v)
        where T : unmanaged
        where TBuf : unmanaged, INumber<TBuf>
    {
        if (typeof(T) == typeof(byte) && typeof(TBuf) == typeof(int))
        {
            int x = Unsafe.As<T, byte>(ref v);
            return Unsafe.As<int, TBuf>(ref x);
        }

        if (typeof(T) == typeof(float) && typeof(TBuf) == typeof(float))
        {
            return Unsafe.As<T, TBuf>(ref v);
        }

        return typeof(T) == typeof(byte) ? TBuf.CreateTruncating(Unsafe.As<T, byte>(ref v)) : TBuf.CreateTruncating(Unsafe.As<T, float>(ref v));
    }

    /// <summary><c>saturate_cast&lt;T&gt;(sum·mulVal)</c> with the float product of OpenCV.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Out<T, TBuf>(TBuf sum, float mulVal)
        where T : unmanaged
        where TBuf : unmanaged, INumber<TBuf>
    {
        float v = (typeof(TBuf) == typeof(int) ? Unsafe.As<TBuf, int>(ref sum) : Unsafe.As<TBuf, float>(ref sum)) * mulVal;
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<float, T>(ref v);
        }

        byte b = RoundByte(v);
        return Unsafe.As<byte, T>(ref b);
    }

    // -------------------------------------------------------------- columns

    /// <summary>
    /// ParallelStackBlurColumn on the elements <c>[c0, c1)</c> of the rows <c>[y0, y1)</c>. The vector
    /// formula applies to elements below <c>16·⌊W·cn/16⌋</c> when the radius has a table entry.
    /// </summary>
    /// <remarks>
    /// The int32 state at row <c>i</c> is exact: <c>sum = Σ (r + 1 − |j|)·v(i + j)</c>,
    /// <c>sumOut = Σ_{j ≤ 0} v(i + j)</c>, <c>sumIn = Σ_{j &gt; 0} v(i + j)</c> over the clamped rows
    /// <c>v</c>, and stack slot <c>(i + j + r) mod ksize</c> holds <c>v(i + j)</c>. A stripe can
    /// therefore start at any row. <paramref name="output"/> can be <paramref name="input"/> when
    /// <paramref name="y0"/> is 0 and <paramref name="y1"/> is the height: every row is read
    /// before it is written.
    /// </remarks>
    private static void ColumnsByte(Mat<byte> input, Mat<byte> output, int radius, int c0, int c1, int y0, int y1)
    {
        int n = c1 - c0;
        int h = input.Height, hm = h - 1;
        int ksize = (2 * radius) + 1;
        int width = input.Width * input.Channels;
        float mulVal = MulVal(radius);
        bool table = radius <= MaxRadius;
        int mul = table ? MulTable[radius] : 0, shr = table ? ShrTable[radius] : 0;
        int simdEnd = table && mul != 0 && shr != 0 ? width & ~15 : 0;
        int vectorEnd = Math.Clamp(simdEnd - c0, 0, n);

        int[] sums = ArrayPool<int>.Shared.Rent(3 * n);
        byte[] stackArray = ArrayPool<byte>.Shared.Rent(ksize * n);
        try
        {
            Span<int> all = sums.AsSpan(0, 3 * n);
            all.Clear();
            ref int sum = ref sums[0];
            ref int sumIn = ref sums[n];
            ref int sumOut = ref sums[2 * n];
            ref byte stack = ref stackArray[0];

            for (int j = -radius; j <= radius; j++)
            {
                ref byte rr = ref Unsafe.Add(ref MemoryMarshal.GetReference(input.GetRowSpan(Math.Clamp(y0 + j, 0, hm))), c0);
                ref byte slot = ref Unsafe.Add(ref stack, ((y0 + j + radius) % ksize) * n);
                int weight = radius + 1 - Math.Abs(j);
                ref int side = ref j <= 0 ? ref sumOut : ref sumIn;
                for (int k = 0; k < n; k++)
                {
                    int v = Unsafe.Add(ref rr, k);
                    Unsafe.Add(ref slot, k) = (byte)v;
                    Unsafe.Add(ref sum, k) += v * weight;
                    Unsafe.Add(ref side, k) += v;
                }
            }

            int sp = (radius + y0) % ksize;
            int yp = Math.Min(y0 + radius, hm);
            var vmul = new Vector<int>(mul);
            var vmulVal = new Vector<float>(mulVal);
            for (int i = y0; i < y1; i++)
            {
                int stackStart = sp + ksize - radius;
                if (stackStart >= ksize)
                {
                    stackStart -= ksize;
                }

                int sp1 = sp + 1 < ksize ? sp + 1 : 0;
                if (yp < hm)
                {
                    yp++;
                }

                ref byte src = ref Unsafe.Add(ref MemoryMarshal.GetReference(input.GetRowSpan(yp)), c0);
                ref byte dst = ref Unsafe.Add(ref MemoryMarshal.GetReference(output.GetRowSpan(i)), c0);
                ref byte ss = ref Unsafe.Add(ref stack, stackStart * n);
                ref byte s1 = ref Unsafe.Add(ref stack, sp1 * n);
                int k = 0;
                int vb = Vector.IsHardwareAccelerated ? Vector<byte>.Count : int.MaxValue / 2;
                for (; k <= vectorEnd - vb; k += vb)
                {
                    ColumnStepByte(ref sum, ref sumIn, ref sumOut, ref ss, ref s1, ref src, ref dst, k, true, vmul, shr, vmulVal);
                }

                k = ScalarColumnsByte(ref sum, ref sumIn, ref sumOut, ref ss, ref s1, ref src, ref dst, k, vectorEnd, vectorEnd, mul, shr, mulVal);
                for (; k <= n - vb; k += vb)
                {
                    ColumnStepByte(ref sum, ref sumIn, ref sumOut, ref ss, ref s1, ref src, ref dst, k, false, vmul, shr, vmulVal);
                }

                ScalarColumnsByte(ref sum, ref sumIn, ref sumOut, ref ss, ref s1, ref src, ref dst, k, n, vectorEnd, mul, shr, mulVal);
                sp = sp + 1 < ksize ? sp + 1 : 0;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(stackArray);
            ArrayPool<int>.Shared.Return(sums);
        }
    }

    private static int ScalarColumnsByte(
        ref int sum, ref int sumIn, ref int sumOut, ref byte ss, ref byte s1, ref byte src, ref byte dst, int k, int end, int vectorEnd, int mul, int shr, float mulVal)
    {
        for (; k < end; k++)
        {
            int s0 = Unsafe.Add(ref sum, k);
            Unsafe.Add(ref dst, k) = k < vectorEnd ? SimdDivide(s0, mul, shr) : (byte)(int)(s0 * mulVal);
            s0 -= Unsafe.Add(ref sumOut, k);
            int so = Unsafe.Add(ref sumOut, k) - Unsafe.Add(ref ss, k);
            int v = Unsafe.Add(ref src, k);
            Unsafe.Add(ref ss, k) = (byte)v;
            int si = Unsafe.Add(ref sumIn, k) + v;
            Unsafe.Add(ref sum, k) = s0 + si;
            int t = Unsafe.Add(ref s1, k);
            Unsafe.Add(ref sumOut, k) = so + t;
            Unsafe.Add(ref sumIn, k) = si - t;
        }

        return k;
    }

    /// <summary>One block of <c>Vector&lt;byte&gt;.Count</c> columns of the byte column step.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ColumnStepByte(
        ref int sum, ref int sumIn, ref int sumOut, ref byte ss, ref byte s1, ref byte src, ref byte dst, int k, bool simd, Vector<int> vmul, int shr, Vector<float> mulVal)
    {
        int vi = Vector<int>.Count;
        FilterVectors.WidenToInt32(Vector.LoadUnsafe(ref ss, (nuint)k), out Vector<int> o0, out Vector<int> o1, out Vector<int> o2, out Vector<int> o3);
        Vector<byte> srcBytes = Vector.LoadUnsafe(ref src, (nuint)k);
        FilterVectors.WidenToInt32(srcBytes, out Vector<int> n0, out Vector<int> n1, out Vector<int> n2, out Vector<int> n3);
        FilterVectors.WidenToInt32(Vector.LoadUnsafe(ref s1, (nuint)k), out Vector<int> t0, out Vector<int> t1, out Vector<int> t2, out Vector<int> t3);
        Vector<int> r0 = Step(ref sum, ref sumIn, ref sumOut, k, o0, n0, t0, simd, vmul, shr, mulVal);
        Vector<int> r1 = Step(ref sum, ref sumIn, ref sumOut, k + vi, o1, n1, t1, simd, vmul, shr, mulVal);
        Vector<int> r2 = Step(ref sum, ref sumIn, ref sumOut, k + (2 * vi), o2, n2, t2, simd, vmul, shr, mulVal);
        Vector<int> r3 = Step(ref sum, ref sumIn, ref sumOut, k + (3 * vi), o3, n3, t3, simd, vmul, shr, mulVal);
        srcBytes.StoreUnsafe(ref ss, (nuint)k);
        FilterVectors.SaturateToByte(r0, r1, r2, r3).StoreUnsafe(ref dst, (nuint)k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Step(
        ref int sum, ref int sumIn, ref int sumOut, int k, Vector<int> stackOut, Vector<int> src, Vector<int> stackNext, bool simd, Vector<int> vmul, int shr, Vector<float> mulVal)
    {
        Vector<int> s = Vector.LoadUnsafe(ref sum, (nuint)k);
        Vector<int> si = Vector.LoadUnsafe(ref sumIn, (nuint)k);
        Vector<int> so = Vector.LoadUnsafe(ref sumOut, (nuint)k);
        Vector<int> result = simd ? SimdDivide(s, vmul, shr) : Vector.ConvertToInt32(Vector.ConvertToSingle(s) * mulVal);
        s -= so;
        so -= stackOut;
        si += src;
        (s + si).StoreUnsafe(ref sum, (nuint)k);
        (so + stackNext).StoreUnsafe(ref sumOut, (nuint)k);
        (si - stackNext).StoreUnsafe(ref sumIn, (nuint)k);
        return result;
    }

    private static void ColumnsFloat(Mat<float> img, int radius, int c0, int c1)
    {
        int n = c1 - c0;
        int h = img.Height, hm = h - 1;
        int ksize = (2 * radius) + 1;
        float mulVal = MulVal(radius);
        float[] sums = ArrayPool<float>.Shared.Rent(3 * n);
        float[] stackArray = ArrayPool<float>.Shared.Rent(ksize * n);
        try
        {
            sums.AsSpan(0, 3 * n).Clear();
            ref float sum = ref sums[0];
            ref float sumIn = ref sums[n];
            ref float sumOut = ref sums[2 * n];
            ref float stack = ref stackArray[0];

            ref float row0 = ref Unsafe.Add(ref MemoryMarshal.GetReference(img.GetRowSpan(0)), c0);
            for (int i = 0; i <= radius; i++)
            {
                for (int k = 0; k < n; k++)
                {
                    float v = Unsafe.Add(ref row0, k);
                    Unsafe.Add(ref stack, (i * n) + k) = v;
                    Unsafe.Add(ref sum, k) += v * (i + 1);
                    Unsafe.Add(ref sumOut, k) += v;
                }
            }

            int row = 0;
            for (int i = 1; i <= radius; i++)
            {
                if (i <= hm)
                {
                    row++;
                }

                ref float rr = ref Unsafe.Add(ref MemoryMarshal.GetReference(img.GetRowSpan(row)), c0);
                for (int k = 0; k < n; k++)
                {
                    float v = Unsafe.Add(ref rr, k);
                    Unsafe.Add(ref stack, ((i + radius) * n) + k) = v;
                    Unsafe.Add(ref sum, k) += v * (radius - i + 1);
                    Unsafe.Add(ref sumIn, k) += v;
                }
            }

            int sp = radius;
            int yp = Math.Min(radius, hm);
            var vm = new Vector<float>(mulVal);
            for (int i = 0; i < h; i++)
            {
                int stackStart = sp + ksize - radius;
                if (stackStart >= ksize)
                {
                    stackStart -= ksize;
                }

                int sp1 = sp + 1 < ksize ? sp + 1 : 0;
                if (yp < hm)
                {
                    yp++;
                }

                ref float src = ref Unsafe.Add(ref MemoryMarshal.GetReference(img.GetRowSpan(yp)), c0);
                ref float dst = ref Unsafe.Add(ref MemoryMarshal.GetReference(img.GetRowSpan(i)), c0);
                ref float ss = ref Unsafe.Add(ref stack, stackStart * n);
                ref float s1 = ref Unsafe.Add(ref stack, sp1 * n);
                int k = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    int vf = Vector<float>.Count;
                    for (; k <= n - vf; k += vf)
                    {
                        Vector<float> s = Vector.LoadUnsafe(ref sum, (nuint)k);
                        Vector<float> si = Vector.LoadUnsafe(ref sumIn, (nuint)k);
                        Vector<float> so = Vector.LoadUnsafe(ref sumOut, (nuint)k);
                        (s * vm).StoreUnsafe(ref dst, (nuint)k);
                        s -= so;
                        so -= Vector.LoadUnsafe(ref ss, (nuint)k);
                        Vector<float> v = Vector.LoadUnsafe(ref src, (nuint)k);
                        v.StoreUnsafe(ref ss, (nuint)k);
                        si += v;
                        (s + si).StoreUnsafe(ref sum, (nuint)k);
                        Vector<float> t = Vector.LoadUnsafe(ref s1, (nuint)k);
                        (so + t).StoreUnsafe(ref sumOut, (nuint)k);
                        (si - t).StoreUnsafe(ref sumIn, (nuint)k);
                    }
                }

                for (; k < n; k++)
                {
                    float s0 = Unsafe.Add(ref sum, k);
                    Unsafe.Add(ref dst, k) = s0 * mulVal;
                    s0 -= Unsafe.Add(ref sumOut, k);
                    float so = Unsafe.Add(ref sumOut, k) - Unsafe.Add(ref ss, k);
                    float v = Unsafe.Add(ref src, k);
                    Unsafe.Add(ref ss, k) = v;
                    float si = Unsafe.Add(ref sumIn, k) + v;
                    Unsafe.Add(ref sum, k) = s0 + si;
                    float t = Unsafe.Add(ref s1, k);
                    Unsafe.Add(ref sumOut, k) = so + t;
                    Unsafe.Add(ref sumIn, k) = si - t;
                }

                sp = sp + 1 < ksize ? sp + 1 : 0;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(stackArray);
            ArrayPool<float>.Shared.Return(sums);
        }
    }
}
