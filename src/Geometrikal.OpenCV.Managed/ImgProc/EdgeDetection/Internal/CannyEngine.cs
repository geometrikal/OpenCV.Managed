// This implementation is ported from the OpenCV Canny edge detector.
//
// Original sources:
//   modules/imgproc/src/canny.cpp (Canny, parallelCanny, finalPass)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Canny core (<c>parallelCanny</c>, global hysteresis and <c>finalPass</c>) on integer
/// thresholds. The edge map has one ring of value 1 around the image; pixel
/// <c>(i, j)</c> is at <c>(i + 1) * stride + j + 1</c>. Map values: 0 weak candidate,
/// 1 not an edge, 2 edge. Rows are split into stripes of at least two rows. Each stripe
/// computes magnitudes in a three-row ring, runs non-maximum suppression and a local
/// hysteresis that writes only inside its own rows; pixels on the stripe border rows are
/// collected and expanded by one serial pass. The output is the union of the 8-connected
/// components of candidate pixels that contain an edge pixel, so it does not depend on
/// the stripe split or the traversal order.
/// </summary>
internal static class CannyEngine
{
    /// <summary><c>round(tan(22.5°) · 2^15)</c>.</summary>
    private const int Tg22 = 13573;

    /// <summary>Minimum image rows per stripe.</summary>
    private const int MinStripeRows = 4;

    /// <summary>Minimum elements (pixels times channels) per stripe that justify a thread.</summary>
    private const int MinStripeElements = 1 << 14;

    /// <summary>
    /// Canny on an 8-bit image with any channel count. Aperture 3 computes the Sobel
    /// derivatives inside each stripe; other apertures compute them for the whole image first.
    /// </summary>
    public static Mat<byte> FromImage(Mat<byte> image, int aperture, bool l2, int low, int high)
    {
        if (aperture == 3)
        {
            return Run(new Context(image, null, null, l2, low, high));
        }

        using PooledMat<short> dx = PooledMat<short>.Rent(image.Width, image.Height, image.Channels);
        using PooledMat<short> dy = PooledMat<short>.Rent(image.Width, image.Height, image.Channels);
        double scale = aperture == 7 ? 1.0 / 16 : 1.0;
        Derivatives.SobelInto(image, 1, 0, aperture, scale, 0, BorderType.Replicate, dx.Mat);
        Derivatives.SobelInto(image, 0, 1, aperture, scale, 0, BorderType.Replicate, dy.Mat);
        return Run(new Context(null, dx.Mat, dy.Mat, l2, low, high));
    }

    /// <summary>Canny on precomputed <c>short</c> derivatives.</summary>
    public static Mat<byte> FromDerivatives(Mat<short> dx, Mat<short> dy, bool l2, int low, int high)
        => Run(new Context(null, dx, dy, l2, low, high));

    private static Mat<byte> Run(Context ctx)
    {
        int width = ctx.Width;
        int height = ctx.Height;
        int stride = ctx.Stride;
        int mapLength = ctx.MapLength;
        byte[] map = ArrayPool<byte>.Shared.Rent(mapLength);
        try
        {
            ctx.Map = map;
            map.AsSpan(0, stride).Fill(1);
            map.AsSpan((height + 1) * stride, stride).Fill(1);

            int stripes = StripeCount(width, height, ctx.Channels);
            ctx.Stripes = stripes;
            ctx.Peaks = new int[stripes][];
            ctx.PeakCounts = new int[stripes];
            if (stripes == 1)
            {
                Stripe(ctx, 0);
            }
            else
            {
                ParallelRows.For(stripes, 1L << 40, (s0, s1) => Stripes(ctx, s0, s1));
            }

            GlobalHysteresis(ctx);

            var dst = new Mat<byte>(width, height, 1, clear: false);
            ParallelRows.For(height, width / Vector<byte>.Count, (y0, y1) => FinalPass(map, stride, dst, y0, y1));
            return dst;
        }
        finally
        {
            ctx.ReturnPeaks();
            ArrayPool<byte>.Shared.Return(map);
        }
    }

    /// <summary>
    /// Stripe count: at most the thread limit, at least <see cref="MinStripeRows"/> rows
    /// and <see cref="MinStripeElements"/> elements per stripe.
    /// </summary>
    private static int StripeCount(int width, int height, int channels)
    {
        long byPixels = (long)width * height * channels / MinStripeElements;
        long byRows = height / MinStripeRows;
        return (int)Math.Max(1, Math.Min(CvParallel.MaxDegreeOfParallelism, Math.Min(byPixels, byRows)));
    }

    private static void Stripes(Context ctx, int s0, int s1)
    {
        for (int s = s0; s < s1; s++)
        {
            Stripe(ctx, s);
        }
    }

    /// <summary>One <c>parallelCanny</c> body over image rows of stripe <paramref name="index"/>.</summary>
    private static void Stripe(Context ctx, int index)
    {
        int width = ctx.Width;
        int height = ctx.Height;
        int cn = ctx.Channels;
        int stride = ctx.Stride;
        int low = ctx.Low;
        int high = ctx.High;
        bool l2 = ctx.L2;
        byte[] map = ctx.Map!;
        Mat<byte>? image = ctx.Image;
        Mat<short>? dxMat = ctx.Dx;
        Mat<short>? dyMat = ctx.Dy;

        int start = (int)((long)height * index / ctx.Stripes);
        int end = (int)((long)height * (index + 1) / ctx.Stripes);
        int rowStart = Math.Max(0, start - 1);
        int rowEnd = Math.Min(height, end + 1);
        int n = width * cn;
        bool select = cn > 1;
        bool ownRows = select || image is not null;
        int ringWidth = width + 2;

        int[] ring = ArrayPool<int>.Shared.Rent(3 * ringWidth);
        int[]? magRaw = select ? ArrayPool<int>.Shared.Rent(n) : null;
        short[]? rawDx = select && image is not null ? ArrayPool<short>.Shared.Rent(n) : null;
        short[]? rawDy = select && image is not null ? ArrayPool<short>.Shared.Rent(n) : null;
        short[]? selDx = ownRows ? ArrayPool<short>.Shared.Rent(2 * width) : null;
        short[]? selDy = ownRows ? ArrayPool<short>.Shared.Rent(2 * width) : null;
        int[] stack = ArrayPool<int>.Shared.Rent(Math.Max(256, (end - start) * width / 8));
        int sp = 0;
        try
        {
            // Offsets of column 0 in the ring rows; column -1 and column width are zero.
            int p = 1, a = ringWidth + 1, nn = (2 * ringWidth) + 1;
            if (rowStart == start)
            {
                Array.Clear(ring, nn - 1, ringWidth);
            }
            else
            {
                ring[nn - 1] = ring[nn + width] = 0;
            }

            ring[a - 1] = ring[a + width] = ring[p - 1] = ring[p + width] = 0;
            int selA = 0, selN = width;

            for (int i = rowStart; i <= end; i++)
            {
                (nn, a) = (a, nn);
                (nn, p) = (p, nn);

                if (i < rowEnd)
                {
                    if (ownRows)
                    {
                        (selA, selN) = (selN, selA);
                    }

                    ref int magN = ref ring[nn];
                    if (!select)
                    {
                        ref short rdx = ref image is not null ? ref selDx![selN] : ref MemoryMarshal.GetReference(dxMat!.GetReadOnlyRowSpan(i));
                        ref short rdy = ref image is not null ? ref selDy![selN] : ref MemoryMarshal.GetReference(dyMat!.GetReadOnlyRowSpan(i));
                        if (image is not null)
                        {
                            SobelGradient3x3.Row(image, i, BorderType.Replicate, selDx.AsSpan(selN, width), selDy.AsSpan(selN, width));
                        }

                        Magnitude(ref rdx, ref rdy, ref magN, n, l2);
                    }
                    else
                    {
                        ReadOnlySpan<short> rdx, rdy;
                        if (image is not null)
                        {
                            SobelGradient3x3.Row(image, i, BorderType.Replicate, rawDx.AsSpan(0, n), rawDy.AsSpan(0, n));
                            rdx = rawDx.AsSpan(0, n);
                            rdy = rawDy.AsSpan(0, n);
                        }
                        else
                        {
                            rdx = dxMat!.GetReadOnlyRowSpan(i);
                            rdy = dyMat!.GetReadOnlyRowSpan(i);
                        }

                        Magnitude(ref MemoryMarshal.GetReference(rdx), ref MemoryMarshal.GetReference(rdy), ref magRaw![0], n, l2);
                        SelectChannel(magRaw, rdx, rdy, ring.AsSpan(nn, width), selDx.AsSpan(selN, width), selDy.AsSpan(selN, width), cn);
                        ring[nn + width] = 0;
                    }

                    if (i <= start)
                    {
                        continue;
                    }
                }
                else
                {
                    Array.Clear(ring, nn - 1, ringWidth);
                    if (ownRows)
                    {
                        (selA, selN) = (selN, selA);
                    }
                }

                // Non-maximum suppression of image row i - 1 (map row i).
                int mapRow = (i * stride) + 1;
                map[mapRow - 1] = 1;
                map[mapRow + width] = 1;
                ref short dxRow = ref ownRows ? ref selDx![selA] : ref MemoryMarshal.GetReference(dxMat!.GetReadOnlyRowSpan(i - 1));
                ref short dyRow = ref ownRows ? ref selDy![selA] : ref MemoryMarshal.GetReference(dyMat!.GetReadOnlyRowSpan(i - 1));
                SuppressRow(ref ring[p], ref ring[a], ref ring[nn], ref dxRow, ref dyRow, map, mapRow, width, low, high, ref stack, ref sp);
            }

            // Local hysteresis; pixels outside [lower, upper) are stripe border pixels.
            int lower = rowStart == 0 ? 0 : (start + 2) * stride;
            int upper = rowEnd == height ? ctx.MapLength : end * stride;
            uint span = (uint)(upper - lower);
            int[] peaks = ArrayPool<int>.Shared.Rent(Math.Max(1, 2 * width));
            int peakCount = 0;
            while (sp > 0)
            {
                int m = stack[--sp];
                if ((uint)(m - lower) < span)
                {
                    PushIfCandidate(map, m - stride - 1, ref stack, ref sp);
                    PushIfCandidate(map, m - stride, ref stack, ref sp);
                    PushIfCandidate(map, m - stride + 1, ref stack, ref sp);
                    PushIfCandidate(map, m - 1, ref stack, ref sp);
                    PushIfCandidate(map, m + 1, ref stack, ref sp);
                    PushIfCandidate(map, m + stride - 1, ref stack, ref sp);
                    PushIfCandidate(map, m + stride, ref stack, ref sp);
                    PushIfCandidate(map, m + stride + 1, ref stack, ref sp);
                }
                else
                {
                    peaks[peakCount++] = m;
                    int step = m < lower ? stride : -stride;
                    PushIfCandidate(map, m - 1, ref stack, ref sp);
                    PushIfCandidate(map, m + 1, ref stack, ref sp);
                    PushIfCandidate(map, m + step - 1, ref stack, ref sp);
                    PushIfCandidate(map, m + step, ref stack, ref sp);
                    PushIfCandidate(map, m + step + 1, ref stack, ref sp);
                }
            }

            ctx.Peaks![index] = peaks;
            ctx.PeakCounts![index] = peakCount;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(stack);
            ArrayPool<int>.Shared.Return(ring);
            if (magRaw is not null)
            {
                ArrayPool<int>.Shared.Return(magRaw);
            }

            if (rawDx is not null)
            {
                ArrayPool<short>.Shared.Return(rawDx);
                ArrayPool<short>.Shared.Return(rawDy!);
            }

            if (selDx is not null)
            {
                ArrayPool<short>.Shared.Return(selDx);
                ArrayPool<short>.Shared.Return(selDy!);
            }
        }
    }

    /// <summary>
    /// L1: <c>|dx| + |dy|</c> with <c>|−32768| = 32768</c>. L2: <c>dx² + dy²</c> with int32 wrap.
    /// </summary>
    private static void Magnitude(ref short dx, ref short dy, ref int mag, int n, bool l2)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vi = Vector<int>.Count;
            for (; i <= n - Vector<short>.Count; i += Vector<short>.Count)
            {
                Vector.Widen(Vector.LoadUnsafe(ref dx, (nuint)i), out Vector<int> x0, out Vector<int> x1);
                Vector.Widen(Vector.LoadUnsafe(ref dy, (nuint)i), out Vector<int> y0, out Vector<int> y1);
                if (l2)
                {
                    ((x0 * x0) + (y0 * y0)).StoreUnsafe(ref mag, (nuint)i);
                    ((x1 * x1) + (y1 * y1)).StoreUnsafe(ref mag, (nuint)(i + vi));
                }
                else
                {
                    (Vector.Abs(x0) + Vector.Abs(y0)).StoreUnsafe(ref mag, (nuint)i);
                    (Vector.Abs(x1) + Vector.Abs(y1)).StoreUnsafe(ref mag, (nuint)(i + vi));
                }
            }
        }

        for (; i < n; i++)
        {
            int x = Unsafe.Add(ref dx, i);
            int y = Unsafe.Add(ref dy, i);
            Unsafe.Add(ref mag, i) = l2 ? unchecked((x * x) + (y * y)) : Math.Abs(x) + Math.Abs(y);
        }
    }

    /// <summary>Per pixel, the first channel with the largest magnitude supplies magnitude, dx and dy.</summary>
    private static void SelectChannel(
        int[] magRaw, ReadOnlySpan<short> dx, ReadOnlySpan<short> dy, Span<int> mag, Span<short> sdx, Span<short> sdy, int cn)
    {
        for (int j = 0, jn = 0; j < mag.Length; j++, jn += cn)
        {
            int maxIdx = jn;
            for (int k = 1; k < cn; k++)
            {
                if (magRaw[jn + k] > magRaw[maxIdx])
                {
                    maxIdx = jn + k;
                }
            }

            mag[j] = magRaw[maxIdx];
            sdx[j] = dx[maxIdx];
            sdy[j] = dy[maxIdx];
        }
    }

    /// <summary>
    /// Non-maximum suppression of one row. Pixels with magnitude above <paramref name="low"/>
    /// that are peaks along the quantized gradient direction become 0 (weak) or 2 (edge, pushed);
    /// all other pixels become 1. Blocks of 16 pixels with no candidate cost one compare.
    /// </summary>
    private static void SuppressRow(
        ref int magP,
        ref int magA,
        ref int magN,
        ref short dx,
        ref short dy,
        byte[] map,
        int mapRow,
        int width,
        int low,
        int high,
        ref int[] stack,
        ref int sp)
    {
        ref byte mapRef = ref map[mapRow];
        int j = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<byte> ones = Vector128.Create((byte)1);
            for (; j <= width - 16; j += 16)
            {
                ones.StoreUnsafe(ref mapRef, (nuint)j);
                uint mask = CandidateMask(ref Unsafe.Add(ref magA, j), low);
                while (mask != 0)
                {
                    int k = j + BitOperations.TrailingZeroCount(mask);
                    mask &= mask - 1;
                    int m = Unsafe.Add(ref magA, k);
                    if (IsPeak(ref magP, ref magA, ref magN, k, m, Unsafe.Add(ref dx, k), Unsafe.Add(ref dy, k)))
                    {
                        Mark(map, mapRow + k, m, high, ref stack, ref sp);
                    }
                }
            }
        }

        for (; j < width; j++)
        {
            int m = Unsafe.Add(ref magA, j);
            if (m > low && IsPeak(ref magP, ref magA, ref magN, j, m, Unsafe.Add(ref dx, j), Unsafe.Add(ref dy, j)))
            {
                Mark(map, mapRow + j, m, high, ref stack, ref sp);
            }
            else
            {
                Unsafe.Add(ref mapRef, j) = 1;
            }
        }
    }

    /// <summary>Bit <c>l</c> is set when <c>mag[l] &gt; low</c>, for 16 magnitudes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint CandidateMask(ref int mag, int low)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            Vector256<int> vlow = Vector256.Create(low);
            return Vector256.GreaterThan(Vector256.LoadUnsafe(ref mag), vlow).ExtractMostSignificantBits()
                | (Vector256.GreaterThan(Vector256.LoadUnsafe(ref mag, 8), vlow).ExtractMostSignificantBits() << 8);
        }

        Vector128<int> low4 = Vector128.Create(low);
        return Vector128.GreaterThan(Vector128.LoadUnsafe(ref mag), low4).ExtractMostSignificantBits()
            | (Vector128.GreaterThan(Vector128.LoadUnsafe(ref mag, 4), low4).ExtractMostSignificantBits() << 4)
            | (Vector128.GreaterThan(Vector128.LoadUnsafe(ref mag, 8), low4).ExtractMostSignificantBits() << 8)
            | (Vector128.GreaterThan(Vector128.LoadUnsafe(ref mag, 12), low4).ExtractMostSignificantBits() << 12);
    }

    /// <summary>
    /// OpenCV's direction test with <c>TG22</c>: horizontal when <c>|dy|·2^15 &lt; |dx|·TG22</c>,
    /// vertical when <c>|dy|·2^15 &gt; |dx|·(TG22 + 2^16)</c> (int32 wrap for <c>|dx| ≥ 27146</c>),
    /// else diagonal. The comparisons are strict toward left/up and both diagonal neighbours.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsPeak(ref int magP, ref int magA, ref int magN, int k, int m, short xs, short ys)
    {
        int x = Math.Abs((int)xs);
        int y = Math.Abs((int)ys) << 15;
        int tg22x = x * Tg22;
        if (y < tg22x)
        {
            return m > Unsafe.Add(ref magA, k - 1) && m >= Unsafe.Add(ref magA, k + 1);
        }

        int tg67x = unchecked(tg22x + (x << 16));
        if (y > tg67x)
        {
            return m > Unsafe.Add(ref magP, k) && m >= Unsafe.Add(ref magN, k);
        }

        int s = (xs ^ ys) < 0 ? -1 : 1;
        return m > Unsafe.Add(ref magP, k - s) && m > Unsafe.Add(ref magN, k + s);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mark(byte[] map, int index, int m, int high, ref int[] stack, ref int sp)
    {
        if (m > high)
        {
            map[index] = 2;
            Push(ref stack, ref sp, index);
        }
        else
        {
            map[index] = 0;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PushIfCandidate(byte[] map, int index, ref int[] stack, ref int sp)
    {
        if (map[index] == 0)
        {
            map[index] = 2;
            Push(ref stack, ref sp, index);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Push(ref int[] stack, ref int sp, int index)
    {
        if (sp == stack.Length)
        {
            Grow(ref stack);
        }

        stack[sp++] = index;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Grow(ref int[] stack)
    {
        int[] larger = ArrayPool<int>.Shared.Rent(stack.Length * 2);
        stack.AsSpan().CopyTo(larger);
        ArrayPool<int>.Shared.Return(stack);
        stack = larger;
    }

    /// <summary>Serial 8-neighbour expansion from the stripe border pixels.</summary>
    private static void GlobalHysteresis(Context ctx)
    {
        int total = 0;
        for (int s = 0; s < ctx.Stripes; s++)
        {
            total += ctx.PeakCounts![s];
        }

        if (total == 0)
        {
            return;
        }

        byte[] map = ctx.Map!;
        int stride = ctx.Stride;
        int[] stack = ArrayPool<int>.Shared.Rent(Math.Max(256, total * 2));
        int sp = 0;
        try
        {
            for (int s = 0; s < ctx.Stripes; s++)
            {
                ctx.Peaks![s].AsSpan(0, ctx.PeakCounts![s]).CopyTo(stack.AsSpan(sp));
                sp += ctx.PeakCounts[s];
            }

            while (sp > 0)
            {
                int m = stack[--sp];
                PushIfCandidate(map, m - stride - 1, ref stack, ref sp);
                PushIfCandidate(map, m - stride, ref stack, ref sp);
                PushIfCandidate(map, m - stride + 1, ref stack, ref sp);
                PushIfCandidate(map, m - 1, ref stack, ref sp);
                PushIfCandidate(map, m + 1, ref stack, ref sp);
                PushIfCandidate(map, m + stride - 1, ref stack, ref sp);
                PushIfCandidate(map, m + stride, ref stack, ref sp);
                PushIfCandidate(map, m + stride + 1, ref stack, ref sp);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(stack);
        }
    }

    /// <summary><c>dst = map == 2 ? 255 : 0</c>.</summary>
    private static void FinalPass(byte[] map, int stride, Mat<byte> dst, int y0, int y1)
    {
        int width = dst.Width;
        for (int y = y0; y < y1; y++)
        {
            ref byte m = ref map[((y + 1) * stride) + 1];
            ref byte d = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
            int j = 0;
            if (Vector.IsHardwareAccelerated && width >= Vector<byte>.Count)
            {
                var two = new Vector<byte>(2);
                for (; j < width; j += Vector<byte>.Count)
                {
                    // The final block overlaps the previous one; it rewrites equal values.
                    j = Math.Min(j, width - Vector<byte>.Count);
                    Vector.Equals(Vector.LoadUnsafe(ref m, (nuint)j), two).StoreUnsafe(ref d, (nuint)j);
                }
            }

            for (; j < width; j++)
            {
                Unsafe.Add(ref d, j) = Unsafe.Add(ref m, j) == 2 ? (byte)255 : (byte)0;
            }
        }
    }

    private sealed class Context
    {
        public Context(Mat<byte>? image, Mat<short>? dx, Mat<short>? dy, bool l2, int low, int high)
        {
            Image = image;
            Dx = dx;
            Dy = dy;
            Width = image?.Width ?? dx!.Width;
            Height = image?.Height ?? dx!.Height;
            Channels = image?.Channels ?? dx!.Channels;
            L2 = l2;
            Low = low;
            High = high;
            Stride = Width + 2;
            MapLength = checked((Height + 2) * Stride);
        }

        public Mat<byte>? Image { get; }

        public Mat<short>? Dx { get; }

        public Mat<short>? Dy { get; }

        public int Width { get; }

        public int Height { get; }

        public int Channels { get; }

        public bool L2 { get; }

        public int Low { get; }

        public int High { get; }

        public int Stride { get; }

        public int MapLength { get; }

        public byte[]? Map { get; set; }

        public int Stripes { get; set; }

        public int[][]? Peaks { get; set; }

        public int[]? PeakCounts { get; set; }

        public void ReturnPeaks()
        {
            if (Peaks is null)
            {
                return;
            }

            foreach (int[] peaks in Peaks)
            {
                if (peaks is not null)
                {
                    ArrayPool<int>.Shared.Return(peaks);
                }
            }
        }
    }
}
