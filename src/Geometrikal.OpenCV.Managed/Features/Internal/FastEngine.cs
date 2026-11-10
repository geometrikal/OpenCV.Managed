// This implementation is ported from the OpenCV FAST detector.
//
// Original sources:
//   modules/features/src/fast.cpp (FAST_t, FAST)
//   modules/features/src/fast_score.cpp (makeOffsets, cornerScore)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>FAST_t</c> for one 8-bit single-channel image: candidate test, contiguous-arc test,
/// corner score and 3x3 non-maximum suppression.
/// </summary>
internal static unsafe class FastEngine
{
    private static readonly int[] Offsets16 =
    {
        0, 3, 1, 3, 2, 2, 3, 1, 3, 0, 3, -1, 2, -2, 1, -3,
        0, -3, -1, -3, -2, -2, -3, -1, -3, 0, -3, 1, -2, 2, -1, 3,
    };

    private static readonly int[] Offsets12 =
    {
        0, 2, 1, 2, 2, 1, 2, 0, 2, -1, 1, -2,
        0, -2, -1, -2, -2, -1, -2, 0, -2, 1, -1, 2,
    };

    private static readonly int[] Offsets8 =
    {
        0, 1, 1, 1, 1, 0, 1, -1,
        0, -1, -1, -1, -1, 0, -1, 1,
    };

    /// <summary>Circle size of a detector type.</summary>
    internal static int PatternSize(FastDetectorType type) => type switch
    {
        FastDetectorType.Type5_8 => 8,
        FastDetectorType.Type7_12 => 12,
        FastDetectorType.Type9_16 => 16,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown FAST detector type."),
    };

    /// <summary>OpenCV <c>makeOffsets</c>: 25 element offsets; entries from the circle size repeat the circle.</summary>
    internal static void MakeOffsets(Span<int> pixel, int rowStride, int patternSize)
    {
        int[] offsets = patternSize == 16 ? Offsets16 : patternSize == 12 ? Offsets12 : Offsets8;
        int k = 0;
        for (; k < patternSize; k++)
        {
            pixel[k] = offsets[2 * k] + (offsets[(2 * k) + 1] * rowStride);
        }

        for (; k < 25; k++)
        {
            pixel[k] = pixel[k - patternSize];
        }
    }

    /// <summary>
    /// Detects FAST corners of <paramref name="img"/> in OpenCV order: row-major, rows from 3 to
    /// <c>height − 4</c>.
    /// </summary>
    /// <returns>A new array of the keypoints.</returns>
    internal static KeyPoint[] Detect(Mat<byte> img, int threshold, bool nonmaxSuppression, FastDetectorType type)
    {
        var buffer = KeyPointBuffer.Rent();
        try
        {
            DetectInto(img, threshold, nonmaxSuppression, type, ref buffer);
            return buffer.Span.ToArray();
        }
        finally
        {
            buffer.Return();
        }
    }

    /// <summary>
    /// Detects FAST corners of <paramref name="img"/> into <paramref name="keypoints"/> (cleared
    /// first) in OpenCV order: row-major, rows from 3 to <c>height − 4</c>.
    /// </summary>
    internal static void Detect(Mat<byte> img, List<KeyPoint> keypoints, int threshold, bool nonmaxSuppression, FastDetectorType type)
    {
        keypoints.Clear();
        var buffer = KeyPointBuffer.Rent();
        try
        {
            DetectInto(img, threshold, nonmaxSuppression, type, ref buffer);
            keypoints.AddRange(buffer.Span);
        }
        finally
        {
            buffer.Return();
        }
    }

    /// <summary>
    /// Detects FAST corners of <paramref name="img"/> into <paramref name="keypoints"/> (cleared
    /// first) in OpenCV order: row-major, rows from 3 to <c>height − 4</c>.
    /// </summary>
    internal static void DetectInto(Mat<byte> img, int threshold, bool nonmaxSuppression, FastDetectorType type, ref KeyPointBuffer keypoints)
    {
        keypoints.Clear();
        int patternSize = PatternSize(type);
        int rows = img.Height;
        int cols = img.Width;
        if (rows < 7 || cols < 7)
        {
            return;
        }

        threshold = Math.Clamp(threshold, 0, 255);
        int outRows = rows - 6;
        long workPerRow = (long)cols * 24 / Vector<byte>.Count;
        if (ParallelRows.IsSerial(outRows, workPerRow))
        {
            Stripe(img, patternSize, threshold, nonmaxSuppression, 3, outRows + 3, ref keypoints);
            return;
        }

        var stripes = new List<(int Start, KeyPointBuffer Keypoints)>();
        try
        {
            ParallelRows.For(outRows, workPerRow, (start, end) =>
            {
                var local = KeyPointBuffer.Rent();
                try
                {
                    Stripe(img, patternSize, threshold, nonmaxSuppression, start + 3, end + 3, ref local);
                }
                catch
                {
                    local.Return();
                    throw;
                }

                lock (stripes)
                {
                    stripes.Add((start, local));
                }
            });

            stripes.Sort((a, b) => a.Start.CompareTo(b.Start));
            foreach ((_, KeyPointBuffer local) in stripes)
            {
                keypoints.AddRange(local.Span);
            }
        }
        finally
        {
            foreach ((_, KeyPointBuffer local) in stripes)
            {
                local.Return();
            }
        }
    }

    /// <summary>
    /// Emits the keypoints of rows <c>[y0, y1)</c>. The stripe recomputes the score rows above
    /// and below, so the output does not depend on the stripe split.
    /// </summary>
    private static void Stripe(Mat<byte> img, int patternSize, int threshold, bool nonmax, int y0, int y1, ref KeyPointBuffer keypoints)
    {
        int cols = img.Width;
        int stride = img.Stride;
        int* pixel = stackalloc int[25];
        MakeOffsets(new Span<int>(pixel, 25), stride, patternSize);
        byte[] scoreRows = ArrayPool<byte>.Shared.Rent(3 * cols);
        int[] cornerRows = ArrayPool<int>.Shared.Rent(3 * (cols + 1));
        try
        {
            Array.Clear(scoreRows, 0, 3 * cols);
            fixed (byte* data0 = img.DangerousGetArray())
            fixed (byte* scores = scoreRows)
            fixed (int* corners = cornerRows)
            {
                byte* data = data0 + img.DangerousGetOffset();
                Run(data, stride, img.Height, cols, pixel, patternSize, threshold, nonmax, scores, corners, y0, y1, ref keypoints);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(cornerRows);
            ArrayPool<byte>.Shared.Return(scoreRows);
        }
    }

    private static void Run(
        byte* data,
        int stride,
        int rows,
        int cols,
        int* pixel,
        int patternSize,
        int threshold,
        bool nonmax,
        byte* scores,
        int* corners,
        int y0,
        int y1,
        ref KeyPointBuffer keypoints)
    {
        int cpStride = cols + 1;
        short* scoreBuf = stackalloc short[32];
        int iStart = Math.Max(3, y0 - 1);
        for (int i = iStart; i <= y1; i++)
        {
            int ring = i - iStart;
            byte* curr = scores + ((ring % 3) * cols);
            int* cornerpos = corners + ((ring % 3) * cpStride) + 1;
            new Span<byte>(curr, cols).Clear();
            int ncorners = 0;

            if (i < rows - 3)
            {
                byte* row = data + ((long)i * stride);
                int j = 3;
                if (Vector.IsHardwareAccelerated && cols - 6 >= Vector<byte>.Count)
                {
                    ncorners = VectorRow(row, cols, pixel, patternSize, threshold, nonmax, curr, cornerpos, ncorners);
                    j = cols - 3;
                }

                for (; j < cols - 3; j++)
                {
                    if (IsCorner(row + j, pixel, patternSize, threshold))
                    {
                        cornerpos[ncorners++] = j;
                        if (nonmax)
                        {
                            curr[j] = (byte)CornerScore(row + j, pixel, patternSize, scoreBuf);
                        }
                    }
                }
            }

            cornerpos[-1] = ncorners;
            if (i - 1 < y0)
            {
                continue;
            }

            // Ring slot (ring + 1) % 3 holds row i − 2, or the cleared initial row when i − 2 < 3.
            byte* prev = scores + (((ring + 2) % 3) * cols);
            byte* pprev = scores + (((ring + 1) % 3) * cols);
            cornerpos = corners + (((ring + 2) % 3) * cpStride) + 1;
            ncorners = cornerpos[-1];
            for (int k = 0; k < ncorners; k++)
            {
                int j = cornerpos[k];
                int score = prev[j];
                if (!nonmax ||
                    (score > prev[j + 1] && score > prev[j - 1] &&
                     score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1] &&
                     score > curr[j - 1] && score > curr[j] && score > curr[j + 1]))
                {
                    keypoints.Add(new KeyPoint(j, i - 1, 7f, -1f, score));
                }
            }
        }
    }

    /// <summary>
    /// Scalar OpenCV corner decision: the paired candidate test of <c>FAST_t</c> and the
    /// contiguous-arc test. For 8 and 12 pixel circles the paired test is stricter than the arc
    /// test, as in OpenCV.
    /// </summary>
    private static bool IsCorner(byte* p, int* pixel, int patternSize, int threshold)
    {
        int v = p[0];
        int lo = v - threshold;
        int hi = v + threshold;
        int d = Tab(p[pixel[0]], lo, hi) | Tab(p[pixel[8]], lo, hi);
        if (d == 0)
        {
            return false;
        }

        d &= Tab(p[pixel[2]], lo, hi) | Tab(p[pixel[10]], lo, hi);
        d &= Tab(p[pixel[4]], lo, hi) | Tab(p[pixel[12]], lo, hi);
        d &= Tab(p[pixel[6]], lo, hi) | Tab(p[pixel[14]], lo, hi);
        if (d == 0)
        {
            return false;
        }

        d &= Tab(p[pixel[1]], lo, hi) | Tab(p[pixel[9]], lo, hi);
        d &= Tab(p[pixel[3]], lo, hi) | Tab(p[pixel[11]], lo, hi);
        d &= Tab(p[pixel[5]], lo, hi) | Tab(p[pixel[13]], lo, hi);
        d &= Tab(p[pixel[7]], lo, hi) | Tab(p[pixel[15]], lo, hi);

        int k2 = patternSize / 2;
        int n = patternSize + k2 + 1;
        if ((d & 1) != 0)
        {
            int count = 0;
            for (int k = 0; k < n; k++)
            {
                if (p[pixel[k]] < lo)
                {
                    if (++count > k2)
                    {
                        return true;
                    }
                }
                else
                {
                    count = 0;
                }
            }
        }

        if ((d & 2) != 0)
        {
            int count = 0;
            for (int k = 0; k < n; k++)
            {
                if (p[pixel[k]] > hi)
                {
                    if (++count > k2)
                    {
                        return true;
                    }
                }
                else
                {
                    count = 0;
                }
            }
        }

        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Tab(int x, int lo, int hi) => x < lo ? 1 : x > hi ? 2 : 0;

    /// <summary>
    /// Vector form of <see cref="IsCorner"/> over a whole row. Each lane evaluates the same
    /// paired test and arc test, so the corner set equals the scalar set. The final block
    /// overlaps the previous block; lanes already processed are skipped.
    /// </summary>
    private static int VectorRow(byte* row, int cols, int* pixel, int patternSize, int threshold, bool nonmax, byte* curr, int* cornerpos, int ncorners)
    {
        int w = Vector<byte>.Count;
        int jEnd = cols - 3;
        int k2 = patternSize / 2;
        int n = patternSize + k2 + 1;
        var tv = new Vector<byte>((byte)threshold);
        var tMax = new Vector<byte>((byte)(255 - threshold));
        var kv = new Vector<byte>((byte)k2);
        byte* lanes = stackalloc byte[w];
        short* scoreBuf = stackalloc short[32];

        for (int jb = 3; ; jb += w)
        {
            bool last = jb + w >= jEnd;
            int j0 = last ? jEnd - w : jb;
            byte* p = row + j0;
            Vector<byte> v = Unsafe.ReadUnaligned<Vector<byte>>(p);
            Vector<byte> vHi = Vector.Min(v, tMax) + tv;
            Vector<byte> vLo = Vector.Max(v, tv) - tv;

            Vector<byte> dark = Pair(p, pixel, 0, 8, vLo, vHi, out Vector<byte> bright);
            dark &= Pair(p, pixel, 2, 10, vLo, vHi, out Vector<byte> b);
            bright &= b;
            dark &= Pair(p, pixel, 4, 12, vLo, vHi, out b);
            bright &= b;
            dark &= Pair(p, pixel, 6, 14, vLo, vHi, out b);
            bright &= b;
            if ((dark | bright) != Vector<byte>.Zero)
            {
                dark &= Pair(p, pixel, 1, 9, vLo, vHi, out b);
                bright &= b;
                dark &= Pair(p, pixel, 3, 11, vLo, vHi, out b);
                bright &= b;
                dark &= Pair(p, pixel, 5, 13, vLo, vHi, out b);
                bright &= b;
                dark &= Pair(p, pixel, 7, 15, vLo, vHi, out b);
                bright &= b;
            }

            if ((dark | bright) != Vector<byte>.Zero)
            {
                Vector<byte> c0 = Vector<byte>.Zero;
                Vector<byte> c1 = Vector<byte>.Zero;
                Vector<byte> max0 = Vector<byte>.Zero;
                Vector<byte> max1 = Vector<byte>.Zero;
                for (int k = 0; k < n; k++)
                {
                    Vector<byte> x = Unsafe.ReadUnaligned<Vector<byte>>(p + pixel[k]);
                    Vector<byte> m0 = Vector.LessThan(x, vLo);
                    Vector<byte> m1 = Vector.GreaterThan(x, vHi);
                    c0 = (c0 - m0) & m0;
                    c1 = (c1 - m1) & m1;
                    max0 = Vector.Max(max0, c0);
                    max1 = Vector.Max(max1, c1);
                }

                Vector<byte> corner = (dark & Vector.GreaterThan(max0, kv)) | (bright & Vector.GreaterThan(max1, kv));
                ulong bits = LaneMask(corner, lanes, w) & (ulong.MaxValue << (jb - j0));
                while (bits != 0)
                {
                    int j = j0 + BitOperations.TrailingZeroCount(bits);
                    bits &= bits - 1;
                    cornerpos[ncorners++] = j;
                    if (nonmax)
                    {
                        curr[j] = (byte)CornerScore(row + j, pixel, patternSize, scoreBuf);
                    }
                }
            }

            if (last)
            {
                return ncorners;
            }
        }
    }

    /// <summary>Bit <c>l</c> is set when lane <c>l</c> of <paramref name="m"/> is all ones.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong LaneMask(Vector<byte> m, byte* lanes, int w)
    {
        if (w == 32)
        {
            return m.AsVector256().ExtractMostSignificantBits();
        }

        if (w == 16)
        {
            return m.AsVector128().ExtractMostSignificantBits();
        }

        if (w == 64)
        {
            return m.AsVector512().ExtractMostSignificantBits();
        }

        Unsafe.WriteUnaligned(lanes, m);
        ulong bits = 0;
        for (int l = 0; l < w; l++)
        {
            bits |= (ulong)(lanes[l] & 1) << l;
        }

        return bits;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<byte> Pair(byte* p, int* pixel, int a, int b, Vector<byte> vLo, Vector<byte> vHi, out Vector<byte> bright)
    {
        Vector<byte> xa = Unsafe.ReadUnaligned<Vector<byte>>(p + pixel[a]);
        Vector<byte> xb = Unsafe.ReadUnaligned<Vector<byte>>(p + pixel[b]);
        bright = Vector.GreaterThan(xa, vHi) | Vector.GreaterThan(xb, vHi);
        return Vector.LessThan(xa, vLo) | Vector.LessThan(xb, vLo);
    }

    /// <summary>
    /// OpenCV <c>cornerScore</c> of the 128-bit SIMD build: the largest value over all arcs of
    /// <c>K + 1</c> circle pixels of <c>max(min d, −max d)</c>, minus 1, where
    /// <c>d = centre − circle pixel</c>. The SIMD form does not use the detection threshold.
    /// </summary>
    internal static int CornerScore(byte* p, int* pixel, int patternSize)
    {
        short* d = stackalloc short[32];
        return CornerScore(p, pixel, patternSize, d);
    }

    /// <inheritdoc cref="CornerScore(byte*, int*, int)"/>
    /// <param name="d">Scratch buffer of at least 32 elements.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CornerScore(byte* p, int* pixel, int patternSize, short* d)
    {
        int k2 = patternSize / 2;
        int v = p[0];
        for (int k = 0; k < 25; k++)
        {
            d[k] = (short)(v - p[pixel[k]]);
        }

        if (Vector256.IsHardwareAccelerated)
        {
            // Lane s holds the arc that starts at circle pixel s; lanes s >= patternSize repeat lanes s − patternSize.
            Vector256<short> mn = Vector256.Load(d);
            Vector256<short> mx = mn;
            for (int k = 1; k <= k2; k++)
            {
                Vector256<short> x = Vector256.Load(d + k);
                mn = Vector256.Min(mn, x);
                mx = Vector256.Max(mx, x);
            }

            Vector256<short> r = Vector256.Max(mn, -mx);
            Vector128<short> h = Vector128.Max(r.GetLower(), r.GetUpper());
            h = Vector128.Max(h, Vector128.Shuffle(h, Vector128.Create((short)4, 5, 6, 7, 0, 1, 2, 3)));
            h = Vector128.Max(h, Vector128.Shuffle(h, Vector128.Create((short)2, 3, 0, 1, 2, 3, 0, 1)));
            h = Vector128.Max(h, Vector128.Shuffle(h, Vector128.Create((short)1, 0, 1, 0, 1, 0, 1, 0)));
            return h.ToScalar() - 1;
        }

        int best = int.MinValue;
        for (int s = 0; s < patternSize; s++)
        {
            int mn = d[s];
            int mx = d[s];
            for (int k = s + 1; k <= s + k2; k++)
            {
                int t = d[k];
                mn = Math.Min(mn, t);
                mx = Math.Max(mx, t);
            }

            best = Math.Max(best, Math.Max(mn, -mx));
        }

        return best - 1;
    }

    /// <summary>A growable keypoint array from <see cref="ArrayPool{T}"/>; elements from <see cref="Count"/> on are undefined.</summary>
    internal struct KeyPointBuffer
    {
        private KeyPoint[] _items;

        public int Count { get; private set; }

        public readonly Span<KeyPoint> Span => _items.AsSpan(0, Count);

        public static KeyPointBuffer Rent() => new() { _items = ArrayPool<KeyPoint>.Shared.Rent(1024) };

        public void Clear() => Count = 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(KeyPoint keypoint)
        {
            if (Count == _items.Length)
            {
                Grow(Count + 1);
            }

            _items[Count++] = keypoint;
        }

        public void AddRange(ReadOnlySpan<KeyPoint> keypoints)
        {
            if (Count + keypoints.Length > _items.Length)
            {
                Grow(Count + keypoints.Length);
            }

            keypoints.CopyTo(_items.AsSpan(Count));
            Count += keypoints.Length;
        }

        public readonly void Return() => ArrayPool<KeyPoint>.Shared.Return(_items);

        private void Grow(int minimum)
        {
            KeyPoint[] larger = ArrayPool<KeyPoint>.Shared.Rent(Math.Max(minimum, 2 * _items.Length));
            Span.CopyTo(larger);
            ArrayPool<KeyPoint>.Shared.Return(_items);
            _items = larger;
        }
    }
}
