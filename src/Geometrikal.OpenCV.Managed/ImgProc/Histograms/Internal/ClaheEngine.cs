// This implementation is ported from the OpenCV CLAHE implementation.
//
// Original source:
//   modules/imgproc/src/clahe.cpp (CLAHE_CalcLut_Body, CLAHE_Interpolation_Body, CLAHE_Impl::apply)
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
/// CLAHE for <c>byte</c> and <c>ushort</c> images. Phase 1 computes one lookup table per tile;
/// phase 2 interpolates the four nearest tables per pixel. Phase 2 starts after phase 1 ends,
/// so the source and destination may be the same matrix.
/// </summary>
internal static class ClaheEngine
{
    /// <summary>Selects the 64-pixel byte blend. Tests clear it to run the <see cref="Vector{T}"/> path.</summary>
    internal static bool UseVector512 = Vector512.IsHardwareAccelerated;

    public static void Apply<T>(Mat<T> src, Mat<T> dst, double clipLimit, int tilesX, int tilesY, int bitShift)
        where T : unmanaged
    {
        bool is8u = typeof(T) == typeof(byte);
        int histSize = (is8u ? 256 : 65536) >> bitShift;
        int width = src.Width;
        int height = src.Height;

        PooledMat<T>? ext = null;
        Mat<T> lutSource = src;
        if (width % tilesX != 0 || height % tilesY != 0)
        {
            // OpenCV pads a divisible dimension by a whole tile as well.
            ext = Borders.CopyMakeBorderPooled(
                src, 0, tilesY - (height % tilesY), 0, tilesX - (width % tilesX), BorderType.Reflect101);
            lutSource = ext.Value.Mat;
        }

        try
        {
            int tileWidth = lutSource.Width / tilesX;
            int tileHeight = lutSource.Height / tilesY;
            int tileArea = tileWidth * tileHeight;
            float lutScale = (float)(histSize - 1) / tileArea;
            int clip = 0;
            if (clipLimit > 0.0)
            {
                clip = Math.Max((int)(clipLimit * tileArea / histSize), 1);
            }

            int tiles = tilesX * tilesY;
            T[] lut = ArrayPool<T>.Shared.Rent(tiles * histSize);
            try
            {
                // Measured cost per pixel in vector-operation units: 4 for counting, 8 for gather and blend.
                var tile = new TileParams(tileWidth, tileHeight, tilesX, clip, lutScale, histSize, bitShift);
                ParallelRows.For(
                    tiles,
                    4L * (tileArea + histSize),
                    (k0, k1) => TileLuts(lutSource, lut, tile, k0, k1));

                ClaheColumns cols = ClaheColumns.Create(width, tileWidth, tilesX, histSize);
                try
                {
                    var interp = new InterpParams(tileHeight, tilesX, tilesY, histSize, bitShift);
                    if (!ParallelRows.IsSerial(height, width * 8L) && !ReferenceEquals(src.DangerousGetArray(), dst.DangerousGetArray()))
                    {
                        PageTouch.Rows(dst);
                    }

                    ParallelRows.For(
                        height,
                        width * 8L,
                        (y0, y1) => Interpolate(src, dst, lut, cols, interp, y0, y1));
                }
                finally
                {
                    cols.Return();
                }
            }
            finally
            {
                ArrayPool<T>.Shared.Return(lut);
            }
        }
        finally
        {
            ext?.Dispose();
        }
    }

    private readonly record struct TileParams(
        int Width, int Height, int TilesX, int Clip, float LutScale, int HistSize, int Shift);

    private readonly record struct InterpParams(int TileHeight, int TilesX, int TilesY, int HistSize, int Shift);

    /// <summary>
    /// Per-column interpolation data, as the OpenCV interpolation body constructor. The arrays are
    /// rented and can be longer than the width.
    /// </summary>
    private sealed class ClaheColumns
    {
        private ClaheColumns(int width)
        {
            Ind1 = ArrayPool<int>.Shared.Rent(width);
            Ind2 = ArrayPool<int>.Shared.Rent(width);
            RunStart = ArrayPool<int>.Shared.Rent(width + 1);
            Xa = ArrayPool<float>.Shared.Rent(width);
            Xa1 = ArrayPool<float>.Shared.Rent(width);
        }

        public int[] Ind1 { get; }

        public int[] Ind2 { get; }

        public float[] Xa { get; }

        public float[] Xa1 { get; }

        /// <summary>Start columns of runs with constant <c>(tx1, tx2)</c>, plus <c>width</c> at index <see cref="RunCount"/>.</summary>
        public int[] RunStart { get; }

        public int RunCount { get; private set; }

        public static ClaheColumns Create(int width, int tileWidth, int tilesX, int histSize)
        {
            var c = new ClaheColumns(width);
            int[] ind1 = c.Ind1;
            int[] ind2 = c.Ind2;
            float[] xa = c.Xa;
            float[] xa1 = c.Xa1;
            int runs = 0;
            float invTw = 1.0f / tileWidth;
            for (int x = 0; x < width; x++)
            {
                float txf = (x * invTw) - 0.5f;
                int tx1 = ScalarConverter.CvFloor(txf);
                int tx2 = tx1 + 1;
                xa[x] = txf - tx1;
                xa1[x] = 1.0f - xa[x];
                tx1 = Math.Max(tx1, 0);
                tx2 = Math.Min(tx2, tilesX - 1);
                ind1[x] = tx1 * histSize;
                ind2[x] = tx2 * histSize;
                if (x == 0 || ind1[x] != ind1[x - 1] || ind2[x] != ind2[x - 1])
                {
                    c.RunStart[runs++] = x;
                }
            }

            c.RunStart[runs] = width;
            c.RunCount = runs;
            return c;
        }

        public void Return()
        {
            ArrayPool<int>.Shared.Return(Ind1);
            ArrayPool<int>.Shared.Return(Ind2);
            ArrayPool<int>.Shared.Return(RunStart);
            ArrayPool<float>.Shared.Return(Xa);
            ArrayPool<float>.Shared.Return(Xa1);
        }
    }

    private static void TileLuts<T>(Mat<T> src, T[] lut, TileParams p, int k0, int k1)
        where T : unmanaged
    {
        int subHists = typeof(T) == typeof(byte) ? 4 : 1;
        int[] histBuffer = ArrayPool<int>.Shared.Rent(subHists * p.HistSize);
        try
        {
            ref T origin = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(src.DangerousGetArray()), src.DangerousGetOffset());
            ref int hist = ref MemoryMarshal.GetArrayDataReference(histBuffer);
            for (int k = k0; k < k1; k++)
            {
                int ty = k / p.TilesX;
                int tx = k % p.TilesX;
                ref T tile = ref Unsafe.Add(ref origin, ((nint)(ty * p.Height) * src.Stride) + (tx * p.Width));
                histBuffer.AsSpan(0, subHists * p.HistSize).Clear();
                TileHistogram(ref tile, src.Stride, p.Width, p.Height, ref hist, p.HistSize, p.Shift);
                int batch = MergeClip(ref hist, p.HistSize, subHists, p.Clip);
                TableFromCounts(ref hist, p.HistSize, batch, p.LutScale, ref lut[k * p.HistSize]);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(histBuffer);
        }
    }

    /// <summary>
    /// Counts <c>v &gt;&gt; shift</c> into zeroed bins. Byte tiles count into four interleaved
    /// sub-histograms at <c>hist + j * histSize</c>.
    /// </summary>
    private static void TileHistogram<T>(ref T tile, int stride, int w, int h, ref int hist, int histSize, int shift)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            ref int h1 = ref Unsafe.Add(ref hist, histSize);
            ref int h2 = ref Unsafe.Add(ref hist, 2 * histSize);
            ref int h3 = ref Unsafe.Add(ref hist, 3 * histSize);
            for (int y = 0; y < h; y++)
            {
                ref byte p = ref Unsafe.As<T, byte>(ref Unsafe.Add(ref tile, (nint)y * stride));
                int x = 0;
                for (; x <= w - 4; x += 4)
                {
                    Unsafe.Add(ref hist, Unsafe.Add(ref p, x) >> shift)++;
                    Unsafe.Add(ref h1, Unsafe.Add(ref p, x + 1) >> shift)++;
                    Unsafe.Add(ref h2, Unsafe.Add(ref p, x + 2) >> shift)++;
                    Unsafe.Add(ref h3, Unsafe.Add(ref p, x + 3) >> shift)++;
                }

                for (; x < w; x++)
                {
                    Unsafe.Add(ref hist, Unsafe.Add(ref p, x) >> shift)++;
                }
            }

            return;
        }

        for (int y = 0; y < h; y++)
        {
            ref ushort p = ref Unsafe.As<T, ushort>(ref Unsafe.Add(ref tile, (nint)y * stride));
            for (int x = 0; x < w; x++)
            {
                Unsafe.Add(ref hist, Unsafe.Add(ref p, x) >> shift)++;
            }
        }
    }

    /// <summary>
    /// Adds the sub-histograms into the first one and clips it at <paramref name="clipLimit"/>
    /// (no clipping if not positive). Adds the clip remainder one count at a time and returns the
    /// count that every bin still receives. The arithmetic is integer, so the order is free.
    /// </summary>
    private static int MergeClip(ref int hist, int histSize, int subHists, int clipLimit)
    {
        bool clip = clipLimit > 0;
        int clipped = 0;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            var vclip = new Vector<int>(clipLimit);
            Vector<int> excess = Vector<int>.Zero;
            for (; i <= histSize - vc; i += vc)
            {
                Vector<int> v = Vector.LoadUnsafe(ref hist, (nuint)i);
                if (subHists == 4)
                {
                    v += Vector.LoadUnsafe(ref hist, (nuint)(i + histSize))
                        + Vector.LoadUnsafe(ref hist, (nuint)(i + (2 * histSize)))
                        + Vector.LoadUnsafe(ref hist, (nuint)(i + (3 * histSize)));
                }

                if (clip)
                {
                    excess += Vector.Max(v - vclip, Vector<int>.Zero);
                    v = Vector.Min(v, vclip);
                }

                v.StoreUnsafe(ref hist, (nuint)i);
            }

            clipped = Vector.Sum(excess);
        }

        for (; i < histSize; i++)
        {
            int v = Unsafe.Add(ref hist, i);
            if (subHists == 4)
            {
                v += Unsafe.Add(ref hist, i + histSize) + Unsafe.Add(ref hist, i + (2 * histSize)) + Unsafe.Add(ref hist, i + (3 * histSize));
            }

            if (clip && v > clipLimit)
            {
                clipped += v - clipLimit;
                v = clipLimit;
            }

            Unsafe.Add(ref hist, i) = v;
        }

        if (!clip)
        {
            return 0;
        }

        int redistBatch = clipped / histSize;
        int residual = clipped - (redistBatch * histSize);
        if (residual != 0)
        {
            int residualStep = Math.Max(histSize / residual, 1);
            for (int k = 0; k < histSize && residual > 0; k += residualStep, residual--)
            {
                Unsafe.Add(ref hist, k)++;
            }
        }

        return redistBatch;
    }

    /// <summary>
    /// <c>lut[i] = saturate_cast&lt;T&gt;((float)cumsum[i] * lutScale)</c>, where every bin first
    /// receives <paramref name="batch"/> more counts.
    /// </summary>
    private static void TableFromCounts<T>(ref int hist, int histSize, int batch, float lutScale, ref T lut)
        where T : unmanaged
    {
        int sum = 0;
        for (int i = 0; i < histSize; i++)
        {
            sum += Unsafe.Add(ref hist, i) + batch;
            Unsafe.Add(ref hist, i) = sum;
        }

        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            var scale = new Vector<float>(lutScale);
            if (typeof(T) == typeof(byte))
            {
                ref byte o = ref Unsafe.As<T, byte>(ref lut);
                for (; x <= histSize - (4 * vc); x += 4 * vc)
                {
                    FilterVectors.SaturateToByte(
                        Round(Vector.LoadUnsafe(ref hist, (nuint)x), scale),
                        Round(Vector.LoadUnsafe(ref hist, (nuint)(x + vc)), scale),
                        Round(Vector.LoadUnsafe(ref hist, (nuint)(x + (2 * vc))), scale),
                        Round(Vector.LoadUnsafe(ref hist, (nuint)(x + (3 * vc))), scale)).StoreUnsafe(ref o, (nuint)x);
                }
            }
            else
            {
                ref ushort o = ref Unsafe.As<T, ushort>(ref lut);
                var vmax = new Vector<int>(65535);
                for (; x <= histSize - (2 * vc); x += 2 * vc)
                {
                    Vector<int> a = Vector.Min(Round(Vector.LoadUnsafe(ref hist, (nuint)x), scale), vmax);
                    Vector<int> b = Vector.Min(Round(Vector.LoadUnsafe(ref hist, (nuint)(x + vc)), scale), vmax);
                    Vector.Narrow(Vector.AsVectorUInt32(a), Vector.AsVectorUInt32(b)).StoreUnsafe(ref o, (nuint)x);
                }
            }
        }

        for (; x < histSize; x++)
        {
            Unsafe.Add(ref lut, x) = Saturate.FromFloat<T>(Unsafe.Add(ref hist, x) * lutScale);
        }
    }

    /// <summary>
    /// <c>cvRound(cum * scale)</c> per lane. Counts are non-negative and at most the tile area, so
    /// the product is finite, non-negative and in the <c>int</c> range.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Round(Vector<int> cum, Vector<float> scale)
        => Vector.ConvertToInt32(Vector.Round(Vector.ConvertToSingle(cum) * scale));

    private static void Interpolate<T>(Mat<T> src, Mat<T> dst, T[] lut, ClaheColumns cols, InterpParams p, int y0, int y1)
        where T : unmanaged
    {
        int w = src.Width;
        T[] gather = ArrayPool<T>.Shared.Rent(4 * w);
        try
        {
            Span<T> g11 = gather.AsSpan(0, w);
            Span<T> g12 = gather.AsSpan(w, w);
            Span<T> g21 = gather.AsSpan(2 * w, w);
            Span<T> g22 = gather.AsSpan(3 * w, w);
            float invTh = 1.0f / p.TileHeight;
            for (int y = y0; y < y1; y++)
            {
                float tyf = (y * invTh) - 0.5f;
                int ty1 = ScalarConverter.CvFloor(tyf);
                int ty2 = ty1 + 1;
                float ya = tyf - ty1;
                float ya1 = 1.0f - ya;
                ty1 = Math.Max(ty1, 0);
                ty2 = Math.Min(ty2, p.TilesY - 1);
                int plane1 = ty1 * p.TilesX * p.HistSize;
                int plane2 = ty2 * p.TilesX * p.HistSize;

                ReadOnlySpan<T> s = src.GetReadOnlyRowSpan(y);
                Gather(s, lut, plane1, plane2, cols, p, g11, g12, g21, g22);
                Blend(g11, g12, g21, g22, cols, ya, ya1, p.Shift, dst.GetRowSpan(y));
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(gather);
        }
    }

    /// <summary>Reads the four table values of every pixel of one row.</summary>
    private static void Gather<T>(
        ReadOnlySpan<T> s, T[] lut, int plane1, int plane2, ClaheColumns cols, InterpParams p,
        Span<T> g11, Span<T> g12, Span<T> g21, Span<T> g22)
        where T : unmanaged
    {
        int w = s.Length;
        if (typeof(T) == typeof(byte) && p.Shift == 0 && ByteLut.UseVbmi && w >= Vector512<byte>.Count)
        {
            GatherVbmi(
                MemoryMarshal.Cast<T, byte>(s), MemoryMarshal.Cast<T, byte>(lut.AsSpan()), plane1, plane2, cols,
                MemoryMarshal.Cast<T, byte>(g11), MemoryMarshal.Cast<T, byte>(g12),
                MemoryMarshal.Cast<T, byte>(g21), MemoryMarshal.Cast<T, byte>(g22));
            return;
        }

        ref T sr = ref MemoryMarshal.GetReference(s);
        ref T l1 = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(lut), plane1);
        ref T l2 = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(lut), plane2);
        ref int ind1 = ref MemoryMarshal.GetArrayDataReference(cols.Ind1);
        ref int ind2 = ref MemoryMarshal.GetArrayDataReference(cols.Ind2);
        ref T a = ref MemoryMarshal.GetReference(g11);
        ref T b = ref MemoryMarshal.GetReference(g12);
        ref T c = ref MemoryMarshal.GetReference(g21);
        ref T d = ref MemoryMarshal.GetReference(g22);
        int shift = p.Shift;
        for (int x = 0; x < w; x++)
        {
            int v = ToInt(Unsafe.Add(ref sr, x)) >> shift;
            int k1 = Unsafe.Add(ref ind1, x) + v;
            int k2 = Unsafe.Add(ref ind2, x) + v;
            Unsafe.Add(ref a, x) = Unsafe.Add(ref l1, k1);
            Unsafe.Add(ref b, x) = Unsafe.Add(ref l1, k2);
            Unsafe.Add(ref c, x) = Unsafe.Add(ref l2, k1);
            Unsafe.Add(ref d, x) = Unsafe.Add(ref l2, k2);
        }
    }

    /// <summary>
    /// Byte table reads with AVX-512 VBMI. Columns form runs with the same tile pair; each run
    /// loads its four 256-byte tables once and maps 64 pixels per step. A step that crosses a run
    /// boundary stores only the lanes inside the run.
    /// </summary>
    private static void GatherVbmi(
        ReadOnlySpan<byte> s, ReadOnlySpan<byte> lut, int plane1, int plane2, ClaheColumns cols,
        Span<byte> g11, Span<byte> g12, Span<byte> g21, Span<byte> g22)
    {
        const int Vc = 64;
        int w = s.Length;
        ref byte sr = ref MemoryMarshal.GetReference(s);
        ref byte lr = ref MemoryMarshal.GetReference(lut);
        ref byte a = ref MemoryMarshal.GetReference(g11);
        ref byte b = ref MemoryMarshal.GetReference(g12);
        ref byte c = ref MemoryMarshal.GetReference(g21);
        ref byte d = ref MemoryMarshal.GetReference(g22);
        int[] runs = cols.RunStart;
        for (int r = 0; r < cols.RunCount; r++)
        {
            int x0 = runs[r];
            int x1 = runs[r + 1];
            Tables t11 = Tables.Load(ref Unsafe.Add(ref lr, plane1 + cols.Ind1[x0]));
            Tables t12 = Tables.Load(ref Unsafe.Add(ref lr, plane1 + cols.Ind2[x0]));
            Tables t21 = Tables.Load(ref Unsafe.Add(ref lr, plane2 + cols.Ind1[x0]));
            Tables t22 = Tables.Load(ref Unsafe.Add(ref lr, plane2 + cols.Ind2[x0]));
            int x = x0;
            while (x < x1)
            {
                // A step at the run end overlaps the previous step and stays inside the row.
                int start = Math.Min(Math.Min(x, x1 - Vc), w - Vc);
                start = Math.Max(start, 0);
                Vector512<byte> idx = Vector512.LoadUnsafe(ref sr, (nuint)start);
                Vector512<byte> hi = Vector512.GreaterThan(idx, Vector512.Create((byte)127));
                StoreRange(t11.Lookup(idx, hi), ref a, start, x, x1);
                StoreRange(t12.Lookup(idx, hi), ref b, start, x, x1);
                StoreRange(t21.Lookup(idx, hi), ref c, start, x, x1);
                StoreRange(t22.Lookup(idx, hi), ref d, start, x, x1);
                x = start + Vc;
            }
        }
    }

    /// <summary>Stores the lanes of <paramref name="v"/> that fall in <c>[from, to)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreRange(Vector512<byte> v, ref byte dst, int start, int from, int to)
    {
        if (from == start && to - start >= Vector512<byte>.Count)
        {
            v.StoreUnsafe(ref dst, (nuint)start);
            return;
        }

        Vector512<byte> lane = Vector512<byte>.Indices;
        Vector512<byte> keep = Vector512.GreaterThanOrEqual(lane, Vector512.Create((byte)(from - start)))
            & Vector512.LessThan(lane, Vector512.Create((byte)Math.Min(to - start, 64)));
        Vector512<byte> old = Vector512.LoadUnsafe(ref dst, (nuint)start);
        Vector512.ConditionalSelect(keep, v, old).StoreUnsafe(ref dst, (nuint)start);
    }

    /// <summary>A 256-byte table in four 64-byte registers.</summary>
    private readonly struct Tables
    {
        private readonly Vector512<byte> _t0;
        private readonly Vector512<byte> _t1;
        private readonly Vector512<byte> _t2;
        private readonly Vector512<byte> _t3;

        private Tables(Vector512<byte> t0, Vector512<byte> t1, Vector512<byte> t2, Vector512<byte> t3)
        {
            _t0 = t0;
            _t1 = t1;
            _t2 = t2;
            _t3 = t3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tables Load(ref byte t)
            => new(Vector512.LoadUnsafe(ref t), Vector512.LoadUnsafe(ref t, 64), Vector512.LoadUnsafe(ref t, 128), Vector512.LoadUnsafe(ref t, 192));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector512<byte> Lookup(Vector512<byte> idx, Vector512<byte> hiMask)
            => Vector512.ConditionalSelect(
                hiMask,
                Avx512Vbmi.PermuteVar64x8x2(_t2, idx, _t3),
                Avx512Vbmi.PermuteVar64x8x2(_t0, idx, _t1));
    }

    /// <summary>
    /// <c>res = (l11 * xa1 + l12 * xa) * ya1 + (l21 * xa1 + l22 * xa) * ya</c> in float, then
    /// <c>saturate_cast&lt;T&gt;(res) &lt;&lt; shift</c>. Each lane performs the scalar operations in
    /// the same order. <c>res</c> is finite and in <c>[0, max]</c> up to rounding, so the vector
    /// paths round and convert without the out-of-range handling of <c>cvRound</c>.
    /// </summary>
    private static void Blend<T>(
        ReadOnlySpan<T> g11, ReadOnlySpan<T> g12, ReadOnlySpan<T> g21, ReadOnlySpan<T> g22, ClaheColumns cols,
        float ya, float ya1, int shift, Span<T> dst)
        where T : unmanaged
    {
        int w = dst.Length;
        ref float xa = ref MemoryMarshal.GetArrayDataReference(cols.Xa);
        ref float xa1 = ref MemoryMarshal.GetArrayDataReference(cols.Xa1);
        int x = 0;
        int max = typeof(T) == typeof(byte) ? 255 : 65535;
        if (typeof(T) == typeof(byte) && UseVector512 && w >= Vector512<byte>.Count)
        {
            Blend512(
                ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g11)),
                ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g12)),
                ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g21)),
                ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g22)),
                ref xa, ref xa1, ya, ya1, shift, ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(dst)), w);
            return;
        }

        if (Vector.IsHardwareAccelerated)
        {
            var vya = new Vector<float>(ya);
            var vya1 = new Vector<float>(ya1);
            var vmax = new Vector<int>(max);
            int fc = Vector<float>.Count;
            if (typeof(T) == typeof(byte))
            {
                ref byte a = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g11));
                ref byte b = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g12));
                ref byte c = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g21));
                ref byte d = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(g22));
                ref byte o = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(dst));
                int bc = Vector<byte>.Count;
                for (; x <= w - bc; x += bc)
                {
                    FilterVectors.WidenToSingle(Vector.LoadUnsafe(ref a, (nuint)x), out var a0, out var a1, out var a2, out var a3);
                    FilterVectors.WidenToSingle(Vector.LoadUnsafe(ref b, (nuint)x), out var b0, out var b1, out var b2, out var b3);
                    FilterVectors.WidenToSingle(Vector.LoadUnsafe(ref c, (nuint)x), out var c0, out var c1, out var c2, out var c3);
                    FilterVectors.WidenToSingle(Vector.LoadUnsafe(ref d, (nuint)x), out var d0, out var d1, out var d2, out var d3);
                    Vector<int> r0 = Lane(a0, b0, c0, d0, ref xa, ref xa1, x, vya, vya1, vmax, shift);
                    Vector<int> r1 = Lane(a1, b1, c1, d1, ref xa, ref xa1, x + fc, vya, vya1, vmax, shift);
                    Vector<int> r2 = Lane(a2, b2, c2, d2, ref xa, ref xa1, x + (2 * fc), vya, vya1, vmax, shift);
                    Vector<int> r3 = Lane(a3, b3, c3, d3, ref xa, ref xa1, x + (3 * fc), vya, vya1, vmax, shift);
                    FilterVectors.SaturateToByte(r0, r1, r2, r3).StoreUnsafe(ref o, (nuint)x);
                }
            }
            else
            {
                ref ushort a = ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(g11));
                ref ushort b = ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(g12));
                ref ushort c = ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(g21));
                ref ushort d = ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(g22));
                ref ushort o = ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(dst));
                int uc = Vector<ushort>.Count;
                for (; x <= w - uc; x += uc)
                {
                    Widen(Vector.LoadUnsafe(ref a, (nuint)x), out var a0, out var a1);
                    Widen(Vector.LoadUnsafe(ref b, (nuint)x), out var b0, out var b1);
                    Widen(Vector.LoadUnsafe(ref c, (nuint)x), out var c0, out var c1);
                    Widen(Vector.LoadUnsafe(ref d, (nuint)x), out var d0, out var d1);
                    Vector<int> r0 = Lane(a0, b0, c0, d0, ref xa, ref xa1, x, vya, vya1, vmax, shift);
                    Vector<int> r1 = Lane(a1, b1, c1, d1, ref xa, ref xa1, x + fc, vya, vya1, vmax, shift);
                    Vector.Narrow(Vector.AsVectorUInt32(r0), Vector.AsVectorUInt32(r1)).StoreUnsafe(ref o, (nuint)x);
                }
            }
        }

        for (; x < w; x++)
        {
            float x1 = Unsafe.Add(ref xa1, x);
            float x0 = Unsafe.Add(ref xa, x);
            float res = (((ToInt(g11[x]) * x1) + (ToInt(g12[x]) * x0)) * ya1) + (((ToInt(g21[x]) * x1) + (ToInt(g22[x]) * x0)) * ya);
            int r = Math.Clamp(ScalarConverter.CvRound(res), 0, max);
            dst[x] = FromInt<T>((r << shift) & max);
        }
    }

    /// <summary>
    /// The byte blend with 64 pixels per step. The final step overlaps the previous one; the
    /// inputs are the gathered rows, so the overlap rewrites equal values.
    /// </summary>
    private static void Blend512(
        ref byte a, ref byte b, ref byte c, ref byte d, ref float xa, ref float xa1, float ya, float ya1, int shift,
        ref byte o, int w)
    {
        const int Vc = 64;
        var vya = Vector512.Create(ya);
        var vya1 = Vector512.Create(ya1);
        for (int x = 0; x < w; x += Vc)
        {
            x = Math.Min(x, w - Vc);
            Widen(Vector512.LoadUnsafe(ref a, (nuint)x), out var a0, out var a1, out var a2, out var a3);
            Widen(Vector512.LoadUnsafe(ref b, (nuint)x), out var b0, out var b1, out var b2, out var b3);
            Widen(Vector512.LoadUnsafe(ref c, (nuint)x), out var c0, out var c1, out var c2, out var c3);
            Widen(Vector512.LoadUnsafe(ref d, (nuint)x), out var d0, out var d1, out var d2, out var d3);
            Vector512<int> r0 = Lane(a0, b0, c0, d0, ref xa, ref xa1, x, vya, vya1);
            Vector512<int> r1 = Lane(a1, b1, c1, d1, ref xa, ref xa1, x + 16, vya, vya1);
            Vector512<int> r2 = Lane(a2, b2, c2, d2, ref xa, ref xa1, x + 32, vya, vya1);
            Vector512<int> r3 = Lane(a3, b3, c3, d3, ref xa, ref xa1, x + 48, vya, vya1);
            Vector512<byte> r = Vector512.Narrow(
                Vector512.Narrow(Clamp255(r0).AsUInt32(), Clamp255(r1).AsUInt32()),
                Vector512.Narrow(Clamp255(r2).AsUInt32(), Clamp255(r3).AsUInt32()));
            if (shift != 0)
            {
                r = Vector512.ShiftLeft(r, shift);
            }

            r.StoreUnsafe(ref o, (nuint)x);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<int> Clamp255(Vector512<int> v)
        => Vector512.Min(Vector512.Max(v, Vector512<int>.Zero), Vector512.Create(255));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<int> Lane(
        Vector512<float> l11, Vector512<float> l12, Vector512<float> l21, Vector512<float> l22, ref float xa, ref float xa1,
        int x, Vector512<float> ya, Vector512<float> ya1)
    {
        Vector512<float> x1 = Vector512.LoadUnsafe(ref xa1, (nuint)x);
        Vector512<float> x0 = Vector512.LoadUnsafe(ref xa, (nuint)x);
        Vector512<float> res = (((l11 * x1) + (l12 * x0)) * ya1) + (((l21 * x1) + (l22 * x0)) * ya);
        return Vector512.ConvertToInt32(Vector512.Round(res));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Widen(
        Vector512<byte> v, out Vector512<float> f0, out Vector512<float> f1, out Vector512<float> f2, out Vector512<float> f3)
    {
        (Vector512<ushort> lo, Vector512<ushort> hi) = Vector512.Widen(v);
        (Vector512<uint> u0, Vector512<uint> u1) = Vector512.Widen(lo);
        (Vector512<uint> u2, Vector512<uint> u3) = Vector512.Widen(hi);
        f0 = Vector512.ConvertToSingle(u0.AsInt32());
        f1 = Vector512.ConvertToSingle(u1.AsInt32());
        f2 = Vector512.ConvertToSingle(u2.AsInt32());
        f3 = Vector512.ConvertToSingle(u3.AsInt32());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Lane(
        Vector<float> l11, Vector<float> l12, Vector<float> l21, Vector<float> l22, ref float xa, ref float xa1, int x,
        Vector<float> ya, Vector<float> ya1, Vector<int> max, int shift)
    {
        Vector<float> x1 = Vector.LoadUnsafe(ref xa1, (nuint)x);
        Vector<float> x0 = Vector.LoadUnsafe(ref xa, (nuint)x);
        Vector<float> res = (((l11 * x1) + (l12 * x0)) * ya1) + (((l21 * x1) + (l22 * x0)) * ya);
        Vector<int> r = Vector.Min(Vector.Max(Vector.ConvertToInt32(Vector.Round(res)), Vector<int>.Zero), max);
        return shift == 0 ? r : Vector.ShiftLeft(r, shift) & max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Widen(Vector<ushort> v, out Vector<float> lo, out Vector<float> hi)
    {
        Vector.Widen(v, out Vector<uint> u0, out Vector<uint> u1);
        lo = Vector.ConvertToSingle(Vector.AsVectorInt32(u0));
        hi = Vector.ConvertToSingle(Vector.AsVectorInt32(u1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ToInt<T>(T v)
        where T : unmanaged
        => typeof(T) == typeof(byte) ? Unsafe.As<T, byte>(ref v) : Unsafe.As<T, ushort>(ref v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T FromInt<T>(int v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            byte b = (byte)v;
            return Unsafe.As<byte, T>(ref b);
        }

        ushort u = (ushort)v;
        return Unsafe.As<ushort, T>(ref u);
    }
}
