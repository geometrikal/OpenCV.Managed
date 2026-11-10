// This implementation is ported from the OpenCV table-driven remap kernels.
//
// Original sources:
//   modules/imgproc/src/imgwarp.cpp (remapNearest, remapBilinear, remapLanczos4, FixedPtCast,
//     Cast)
//   OpenCV 4.5.5 modules/imgproc/src/imgwarp.cpp (remapBicubic)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Element arithmetic of one depth of the remap kernels: work type, weight type and output cast.</summary>
internal interface IRemapOps<T, TWork, TWeight>
    where T : unmanaged
    where TWork : unmanaged, INumber<TWork>
    where TWeight : unmanaged
{
    /// <summary>Weight of one tap that represents 1 (<c>INTER_REMAP_COEF_SCALE</c> or 1).</summary>
    static abstract TWork One { get; }

    static abstract TWork Load(T v);

    static abstract TWork Weight(TWeight w);

    static abstract T Cast(TWork v);

    /// <summary>OpenCV 5.0 renormalisation <c>(WT)(t0 * (float)totalInitial / total)</c>.</summary>
    static abstract TWork Renormalize(TWork t0, TWork totalInitial, TWork total);
}

/// <summary>8-bit: <c>int</c> sums of 16-bit weights, <c>FixedPtCast&lt;int, uchar, 15&gt;</c>.</summary>
internal readonly struct RemapOps8u : IRemapOps<byte, int, short>
{
    public static int One => InterTables.CoefScale;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Load(byte v) => v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Weight(short w) => w;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Cast(int v) => (byte)Math.Clamp((v + (1 << 14)) >> 15, 0, 255);

    public static int Renormalize(int t0, int totalInitial, int total) => (int)(t0 * (float)totalInitial / total);
}

/// <summary>16-bit and 32-bit float: <c>float</c> sums, <c>saturate_cast</c>.</summary>
internal readonly struct RemapOpsFloat<T> : IRemapOps<T, float, float>
    where T : unmanaged
{
    public static float One => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Load(T v) => Saturate.ToFloat(v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Weight(float w) => w;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Cast(float v) => Saturate.FromFloat<T>(v);

    public static float Renormalize(float t0, float totalInitial, float total) => t0 * totalInitial / total;
}

/// <summary>64-bit float: <c>double</c> sums of single-precision weights.</summary>
internal readonly struct RemapOpsDouble : IRemapOps<double, double, float>
{
    public static double One => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Load(double v) => v;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Weight(float w) => w;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Cast(double v) => v;

    public static double Renormalize(double t0, double totalInitial, double total) => t0 * (float)totalInitial / total;
}

/// <summary>Source image, border rule and constant of one remap call.</summary>
internal readonly struct RemapSource<T>
    where T : unmanaged
{
    public readonly T[] Array;
    public readonly int Offset;
    public readonly int Step;
    public readonly int Width;
    public readonly int Height;
    public readonly int Cn;
    public readonly BorderType Border;
    public readonly T[] Constant;

    public RemapSource(Mat<T> src, BorderType border, Scalar value)
    {
        Array = src.DangerousGetArray();
        Offset = src.DangerousGetOffset();
        Step = src.Stride;
        Width = src.Width;
        Height = src.Height;
        Cn = src.Channels;
        Border = border;
        Constant = new T[Cn];
        for (int k = 0; k < Cn; k++)
        {
            Constant[k] = ScalarConverter.SaturateCast<T>(value[k & 3]);
        }
    }

    /// <summary>Element <c>(y, x * cn + k)</c>; the caller keeps the indices in range.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref T At(int y, int e) => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Array), Offset + (y * Step) + e);
}

/// <summary>
/// Row kernels of the table-driven remap. <c>xy</c> holds the integer source coordinates of each
/// destination pixel and <c>fxy</c> the table index <c>(y fraction) * 32 + (x fraction)</c>.
/// With <c>relative</c>, the destination coordinates <c>(x0 + dx, y)</c> are added to <c>xy</c>.
/// </summary>
internal static class RemapKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Clip(int x, int a, int b) => x >= a ? (x < b ? x : b - 1) : a;

    public static void Nearest<T>(in RemapSource<T> s, Span<T> dst, ReadOnlySpan<short> xy, int x0, int y, bool relative)
        where T : unmanaged
    {
        int cn = s.Cn;
        int width = dst.Length / cn;
        int offY = relative ? y : 0;
        for (int dx = 0; dx < width; dx++)
        {
            int sx = xy[2 * dx] + (relative ? x0 + dx : 0);
            int sy = xy[(2 * dx) + 1] + offY;
            Span<T> d = dst.Slice(dx * cn, cn);
            if ((uint)sx < (uint)s.Width && (uint)sy < (uint)s.Height)
            {
                CopyPixel(ref s.At(sy, sx * cn), d);
                continue;
            }

            switch (s.Border)
            {
                case BorderType.Transparent:
                    break;
                case BorderType.Constant:
                    s.Constant.CopyTo(d);
                    break;
                case BorderType.Replicate:
                    CopyPixel(ref s.At(Clip(sy, 0, s.Height), Clip(sx, 0, s.Width) * cn), d);
                    break;
                default:
                    CopyPixel(ref s.At(Borders.BorderInterpolate(sy, s.Height, s.Border), Borders.BorderInterpolate(sx, s.Width, s.Border) * cn), d);
                    break;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void CopyPixel<T>(ref T src, Span<T> d)
        where T : unmanaged
    {
        for (int k = 0; k < d.Length; k++)
        {
            d[k] = Unsafe.Add(ref src, k);
        }
    }

    public static void Bilinear<T, TWork, TWeight, TOps>(
        in RemapSource<T> s, Span<T> dst, ReadOnlySpan<short> xy, ReadOnlySpan<ushort> fxy, TWeight[] table, int x0, int y, bool relative)
        where T : unmanaged
        where TWork : unmanaged, INumber<TWork>
        where TWeight : unmanaged
        where TOps : struct, IRemapOps<T, TWork, TWeight>
    {
        int cn = s.Cn;
        int width = dst.Length / cn;
        uint w1 = (uint)Math.Max(s.Width - 1, 0), h1 = (uint)Math.Max(s.Height - 1, 0);
        int offY = relative ? y : 0;
        int step = s.Step;
        bool constant = s.Border == BorderType.Constant;
        ref T src0 = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(s.Array), s.Offset);
        ref TWeight tab = ref MemoryMarshal.GetArrayDataReference(table);
        ref short xyr = ref MemoryMarshal.GetReference(xy);
        ref ushort fr = ref MemoryMarshal.GetReference(fxy);
        ref T d = ref MemoryMarshal.GetReference(dst);
        for (int dx = 0; dx < width; dx++)
        {
            int sx = Unsafe.Add(ref xyr, 2 * dx) + (relative ? x0 + dx : 0);
            int sy = Unsafe.Add(ref xyr, (2 * dx) + 1) + offY;
            ref TWeight wr = ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx) * 4);
            TWork wa = TOps.Weight(wr), wb = TOps.Weight(Unsafe.Add(ref wr, 1));
            TWork wc = TOps.Weight(Unsafe.Add(ref wr, 2)), wd = TOps.Weight(Unsafe.Add(ref wr, 3));
            if ((uint)sx < w1 && (uint)sy < h1)
            {
                ref T p = ref Unsafe.Add(ref src0, (sy * step) + (sx * cn));
                ref T q = ref Unsafe.Add(ref p, step);
                ref T o = ref Unsafe.Add(ref d, dx * cn);
                for (int k = 0; k < cn; k++)
                {
                    Unsafe.Add(ref o, k) = TOps.Cast((TOps.Load(Unsafe.Add(ref p, k)) * wa) + (TOps.Load(Unsafe.Add(ref p, k + cn)) * wb)
                        + (TOps.Load(Unsafe.Add(ref q, k)) * wc) + (TOps.Load(Unsafe.Add(ref q, k + cn)) * wd));
                }

                continue;
            }

            if (constant && (sx >= s.Width || sx + 1 < 0 || sy >= s.Height || sy + 1 < 0))
            {
                for (int k = 0; k < cn; k++)
                {
                    Unsafe.Add(ref d, (dx * cn) + k) = s.Constant[k];
                }

                continue;
            }

            BilinearOutlier<T, TWork, TWeight, TOps>(s, dst.Slice(dx * cn, cn), sx, sy, wa, wb, wc, wd);
        }
    }

    /// <summary><see cref="Bilinear{T, TWork, TWeight, TOps}"/> for 8-bit images with a reference-based inlier loop.</summary>
    public static void Bilinear8u(
        in RemapSource<byte> s, Span<byte> dst, ReadOnlySpan<short> xy, ReadOnlySpan<ushort> fxy, short[] table, int x0, int y, bool relative)
    {
        int cn = s.Cn;
        int width = dst.Length / cn;
        uint w1 = (uint)Math.Max(s.Width - 1, 0), h1 = (uint)Math.Max(s.Height - 1, 0);
        int offY = relative ? y : 0;
        int offX = relative ? x0 : 0;
        int step = s.Step;
        ref byte src0 = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(s.Array), s.Offset);
        ref short tab = ref MemoryMarshal.GetArrayDataReference(table);
        ref short xyr = ref MemoryMarshal.GetReference(xy);
        ref ushort fr = ref MemoryMarshal.GetReference(fxy);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        bool simd = cn == 1 && Ssse3.IsSupported;
        bool simd3 = cn == 3 && Ssse3.IsSupported;
        Vector128<byte> shufA = Vector128.Create((byte)0, 0x80, 3, 0x80, 8, 0x80, 11, 0x80, 1, 0x80, 4, 0x80, 9, 0x80, 12, 0x80);
        Vector128<byte> shufB = Vector128.Create((byte)2, 0x80, 5, 0x80, 10, 0x80, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        bool constant = s.Border == BorderType.Constant;
        for (int dx = 0; dx < width; dx++)
        {
            if (simd && dx <= width - 4 && Inliers4(ref xyr, dx, relative ? offX : int.MinValue, offY, w1, h1))
            {
                Bilinear4(ref src0, step, ref tab, ref xyr, ref fr, ref d, dx, relative ? offX : int.MinValue, offY);
                dx += 3;
                continue;
            }

            int sx = Unsafe.Add(ref xyr, 2 * dx) + (relative ? offX + dx : 0);
            int sy = Unsafe.Add(ref xyr, (2 * dx) + 1) + offY;
            ref short w = ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx) * 4);
            int wa = w, wb = Unsafe.Add(ref w, 1), wc = Unsafe.Add(ref w, 2), wd = Unsafe.Add(ref w, 3);
            if ((uint)sx < w1 && (uint)sy < h1)
            {
                // Bilinear weights are non-negative and sum to 32768, so the result needs no saturation.
                ref byte p = ref Unsafe.Add(ref src0, (sy * step) + (sx * cn));
                ref byte q = ref Unsafe.Add(ref p, step);
                ref byte o = ref Unsafe.Add(ref d, dx * cn);
                if (simd3)
                {
                    Bilinear3(ref p, ref q, ref w, ref o, shufA, shufB);
                    continue;
                }

                if (cn == 3)
                {
                    o = (byte)(((p * wa) + (Unsafe.Add(ref p, 3) * wb) + (q * wc) + (Unsafe.Add(ref q, 3) * wd) + (1 << 14)) >> 15);
                    Unsafe.Add(ref o, 1) = (byte)(((Unsafe.Add(ref p, 1) * wa) + (Unsafe.Add(ref p, 4) * wb) + (Unsafe.Add(ref q, 1) * wc) + (Unsafe.Add(ref q, 4) * wd) + (1 << 14)) >> 15);
                    Unsafe.Add(ref o, 2) = (byte)(((Unsafe.Add(ref p, 2) * wa) + (Unsafe.Add(ref p, 5) * wb) + (Unsafe.Add(ref q, 2) * wc) + (Unsafe.Add(ref q, 5) * wd) + (1 << 14)) >> 15);
                    continue;
                }

                for (int k = 0; k < cn; k++)
                {
                    int v = (Unsafe.Add(ref p, k) * wa) + (Unsafe.Add(ref p, k + cn) * wb) + (Unsafe.Add(ref q, k) * wc) + (Unsafe.Add(ref q, k + cn) * wd);
                    Unsafe.Add(ref o, k) = (byte)((v + (1 << 14)) >> 15);
                }

                continue;
            }

            if (constant && (sx >= s.Width || sx + 1 < 0 || sy >= s.Height || sy + 1 < 0))
            {
                for (int k = 0; k < cn; k++)
                {
                    Unsafe.Add(ref d, (dx * cn) + k) = s.Constant[k];
                }

                continue;
            }

            BilinearOutlier<byte, int, short, RemapOps8u>(s, dst.Slice(dx * cn, cn), sx, sy, wa, wb, wc, wd);
        }
    }

    /// <summary>
    /// One 3-channel 8-bit inlier pixel from the 2 × 2 neighbourhood at <paramref name="p"/> and
    /// <paramref name="q"/> and the weights at <paramref name="w"/>; the integer sums equal the scalar sums.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Bilinear3(ref byte p, ref byte q, ref short w, ref byte o, Vector128<byte> shufA, Vector128<byte> shufB)
    {
        ulong pv = Unsafe.ReadUnaligned<uint>(ref p) | ((ulong)Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 4)) << 32);
        ulong qv = Unsafe.ReadUnaligned<uint>(ref q) | ((ulong)Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref q, 4)) << 32);
        Vector128<byte> v = Vector128.Create(pv, qv).AsByte();
        Vector128<short> weights = Vector128.Create(Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<short, byte>(ref w))).AsInt16();
        Vector128<int> a = Sse2.MultiplyAddAdjacent(Ssse3.Shuffle(v, shufA).AsInt16(), weights);
        Vector128<int> b = Sse2.MultiplyAddAdjacent(Ssse3.Shuffle(v, shufB).AsInt16(), weights);
        Vector128<int> sum = Sse2.ShiftRightArithmetic(Ssse3.HorizontalAdd(a, b) + Vector128.Create(1 << 14), 15);
        Vector128<short> s16 = Sse2.PackSignedSaturate(sum, sum);
        uint packed = Sse2.PackUnsignedSaturate(s16, s16).AsUInt32().ToScalar();
        Unsafe.WriteUnaligned(ref o, (ushort)packed);
        Unsafe.Add(ref o, 2) = (byte)(packed >> 16);
    }

    /// <summary>True when the four pixels from <paramref name="dx"/> have their 2 × 2 neighbourhood inside the source.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Inliers4(ref short xy, int dx, int offX, int offY, uint w1, uint h1)
    {
        int rx = offX == int.MinValue ? 0 : offX + dx;
        int rx1 = offX == int.MinValue ? 0 : 1;
        return (uint)(Unsafe.Add(ref xy, 2 * dx) + rx) < w1 && (uint)(Unsafe.Add(ref xy, (2 * dx) + 1) + offY) < h1
            && (uint)(Unsafe.Add(ref xy, (2 * dx) + 2) + rx + rx1) < w1 && (uint)(Unsafe.Add(ref xy, (2 * dx) + 3) + offY) < h1
            && (uint)(Unsafe.Add(ref xy, (2 * dx) + 4) + rx + (2 * rx1)) < w1 && (uint)(Unsafe.Add(ref xy, (2 * dx) + 5) + offY) < h1
            && (uint)(Unsafe.Add(ref xy, (2 * dx) + 6) + rx + (3 * rx1)) < w1 && (uint)(Unsafe.Add(ref xy, (2 * dx) + 7) + offY) < h1;
    }

    /// <summary>
    /// Four single-channel 8-bit inlier pixels: the 2 × 2 neighbourhoods and the 16-bit weights
    /// are combined with <c>pmaddwd</c>; the integer sums equal the scalar sums.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Bilinear4(ref byte src0, int step, ref short tab, ref short xy, ref ushort fr, ref byte d, int dx, int offX, int offY)
    {
        int rx = offX == int.MinValue ? 0 : offX + dx;
        int rx1 = offX == int.MinValue ? 0 : 1;
        uint n0 = Pair(ref src0, step, (Unsafe.Add(ref xy, (2 * dx) + 1) + offY) * step + Unsafe.Add(ref xy, 2 * dx) + rx);
        uint n1 = Pair(ref src0, step, (Unsafe.Add(ref xy, (2 * dx) + 3) + offY) * step + Unsafe.Add(ref xy, (2 * dx) + 2) + rx + rx1);
        uint n2 = Pair(ref src0, step, (Unsafe.Add(ref xy, (2 * dx) + 5) + offY) * step + Unsafe.Add(ref xy, (2 * dx) + 4) + rx + (2 * rx1));
        uint n3 = Pair(ref src0, step, (Unsafe.Add(ref xy, (2 * dx) + 7) + offY) * step + Unsafe.Add(ref xy, (2 * dx) + 6) + rx + (3 * rx1));
        Vector128<byte> px = Vector128.Create(n0, n1, n2, n3).AsByte();
        Vector128<short> lo = Sse2.UnpackLow(px, Vector128<byte>.Zero).AsInt16();
        Vector128<short> hi = Sse2.UnpackHigh(px, Vector128<byte>.Zero).AsInt16();
        Vector128<short> w01 = Vector128.Create(
            Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<short, byte>(ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx) * 4))),
            Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<short, byte>(ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx + 1) * 4)))).AsInt16();
        Vector128<short> w23 = Vector128.Create(
            Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<short, byte>(ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx + 2) * 4))),
            Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<short, byte>(ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx + 3) * 4)))).AsInt16();
        Vector128<int> sum = Ssse3.HorizontalAdd(Sse2.MultiplyAddAdjacent(lo, w01), Sse2.MultiplyAddAdjacent(hi, w23));
        sum = Sse2.ShiftRightArithmetic(sum + Vector128.Create(1 << 14), 15);
        Vector128<short> s16 = Sse2.PackSignedSaturate(sum, sum);
        Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, dx), Sse2.PackUnsignedSaturate(s16, s16).AsUInt32().ToScalar());
    }

    /// <summary>The bytes <c>p[0], p[1], p[step], p[step + 1]</c> in one 32-bit value.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Pair(ref byte src0, int step, int offset)
    {
        ref byte p = ref Unsafe.Add(ref src0, offset);
        return Unsafe.ReadUnaligned<ushort>(ref p) | ((uint)Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, step)) << 16);
    }

    private static void BilinearOutlier<T, TWork, TWeight, TOps>(
        in RemapSource<T> s, Span<T> d, int sx, int sy, TWork wa, TWork wb, TWork wc, TWork wd)
        where T : unmanaged
        where TWork : unmanaged, INumber<TWork>
        where TWeight : unmanaged
        where TOps : struct, IRemapOps<T, TWork, TWeight>
    {
        int cn = s.Cn;
        int width = s.Width, height = s.Height;
        if (s.Border == BorderType.Transparent)
        {
            // OpenCV 5.0: a pixel whose base lies in the image uses the available taps with renormalised weights.
            if (sx >= 0 && sx <= width - 1 && sy >= 0 && sy <= height - 1)
            {
                bool right = sx < width - 1, down = sy < height - 1;
                TWork total = TWork.Zero + wa;
                if (right)
                {
                    total += wb;
                }

                if (down)
                {
                    total += wc;
                }

                if (right && down)
                {
                    total += wd;
                }

                if (total == TWork.Zero)
                {
                    return;
                }

                TWork totalInitial = wa + wb + wc + wd;
                ref T p = ref s.At(sy, sx * cn);
                for (int k = 0; k < cn; k++)
                {
                    TWork t0 = TWork.Zero + (TOps.Load(Unsafe.Add(ref p, k)) * wa);
                    if (right)
                    {
                        t0 += TOps.Load(Unsafe.Add(ref p, k + cn)) * wb;
                    }

                    if (down)
                    {
                        t0 += TOps.Load(Unsafe.Add(ref p, s.Step + k)) * wc;
                    }

                    if (right && down)
                    {
                        t0 += TOps.Load(Unsafe.Add(ref p, s.Step + k + cn)) * wd;
                    }

                    d[k] = TOps.Cast(TOps.Renormalize(t0, totalInitial, total));
                }
            }

            return;
        }

        if (s.Border == BorderType.Constant && (sx >= width || sx + 1 < 0 || sy >= height || sy + 1 < 0))
        {
            s.Constant.CopyTo(d);
            return;
        }

        int sx0, sx1, sy0, sy1;
        if (s.Border == BorderType.Replicate)
        {
            sx0 = Clip(sx, 0, width);
            sx1 = Clip(sx + 1, 0, width);
            sy0 = Clip(sy, 0, height);
            sy1 = Clip(sy + 1, 0, height);
        }
        else
        {
            sx0 = Borders.BorderInterpolate(sx, width, s.Border);
            sx1 = Borders.BorderInterpolate(sx + 1, width, s.Border);
            sy0 = Borders.BorderInterpolate(sy, height, s.Border);
            sy1 = Borders.BorderInterpolate(sy + 1, height, s.Border);
        }

        for (int k = 0; k < cn; k++)
        {
            T v0 = sx0 >= 0 && sy0 >= 0 ? s.At(sy0, (sx0 * cn) + k) : s.Constant[k];
            T v1 = sx1 >= 0 && sy0 >= 0 ? s.At(sy0, (sx1 * cn) + k) : s.Constant[k];
            T v2 = sx0 >= 0 && sy1 >= 0 ? s.At(sy1, (sx0 * cn) + k) : s.Constant[k];
            T v3 = sx1 >= 0 && sy1 >= 0 ? s.At(sy1, (sx1 * cn) + k) : s.Constant[k];
            d[k] = TOps.Cast((TOps.Load(v0) * wa) + (TOps.Load(v1) * wb) + (TOps.Load(v2) * wc) + (TOps.Load(v3) * wd));
        }
    }

    /// <summary>
    /// <c>remapBicubic</c> (4.5.5) and <c>remapLanczos4</c>: <paramref name="k"/> = 4 or 8 taps per
    /// axis, the source window starts at <c>xy - (k/2 - 1)</c>. Inlier sums are row sums of
    /// left-to-right products, added row by row (Lanczos4 adds the first row to 0).
    /// </summary>
    public static void Separable<T, TWork, TWeight, TOps>(
        in RemapSource<T> s, Span<T> dst, ReadOnlySpan<short> xy, ReadOnlySpan<ushort> fxy, TWeight[] table, int k, int x0, int y, bool relative)
        where T : unmanaged
        where TWork : unmanaged, INumber<TWork>
        where TWeight : unmanaged
        where TOps : struct, IRemapOps<T, TWork, TWeight>
    {
        int cn = s.Cn;
        int width = dst.Length / cn;
        int half = (k / 2) - 1;
        int kk = k * k;
        uint width1 = (uint)Math.Max(s.Width - (k - 1), 0), height1 = (uint)Math.Max(s.Height - (k - 1), 0);
        int offY = relative ? y : 0;
        int step = s.Step;
        ref T src0 = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(s.Array), s.Offset);
        ref TWeight tab = ref MemoryMarshal.GetArrayDataReference(table);
        ref short xyr = ref MemoryMarshal.GetReference(xy);
        ref ushort fr = ref MemoryMarshal.GetReference(fxy);
        ref T d = ref MemoryMarshal.GetReference(dst);
        for (int dx = 0; dx < width; dx++)
        {
            int sx = Unsafe.Add(ref xyr, 2 * dx) - half + (relative ? x0 + dx : 0);
            int sy = Unsafe.Add(ref xyr, (2 * dx) + 1) - half + offY;
            ref TWeight w = ref Unsafe.Add(ref tab, Unsafe.Add(ref fr, dx) * kk);
            if ((uint)sx < width1 && (uint)sy < height1)
            {
                ref T p0 = ref Unsafe.Add(ref src0, (sy * step) + (sx * cn));
                for (int c = 0; c < cn; c++)
                {
                    ref T p = ref Unsafe.Add(ref p0, c);
                    TWork sum = k == 8 ? TWork.Zero + RowSum<T, TWork, TWeight, TOps>(ref p, ref w, cn, k) : RowSum<T, TWork, TWeight, TOps>(ref p, ref w, cn, k);
                    for (int r = 1; r < k; r++)
                    {
                        sum += RowSum<T, TWork, TWeight, TOps>(ref Unsafe.Add(ref p, r * step), ref Unsafe.Add(ref w, r * k), cn, k);
                    }

                    Unsafe.Add(ref d, (dx * cn) + c) = TOps.Cast(sum);
                }

                continue;
            }

            SeparableOutlier<T, TWork, TWeight, TOps>(s, dst.Slice(dx * cn, cn), sx, sy, ref w, k);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TWork RowSum<T, TWork, TWeight, TOps>(ref T p, ref TWeight w, int cn, int k)
        where T : unmanaged
        where TWork : unmanaged, INumber<TWork>
        where TWeight : unmanaged
        where TOps : struct, IRemapOps<T, TWork, TWeight>
    {
        TWork acc = (TOps.Load(p) * TOps.Weight(w)) + (TOps.Load(Unsafe.Add(ref p, cn)) * TOps.Weight(Unsafe.Add(ref w, 1)))
            + (TOps.Load(Unsafe.Add(ref p, 2 * cn)) * TOps.Weight(Unsafe.Add(ref w, 2))) + (TOps.Load(Unsafe.Add(ref p, 3 * cn)) * TOps.Weight(Unsafe.Add(ref w, 3)));
        if (k == 8)
        {
            acc = acc + (TOps.Load(Unsafe.Add(ref p, 4 * cn)) * TOps.Weight(Unsafe.Add(ref w, 4))) + (TOps.Load(Unsafe.Add(ref p, 5 * cn)) * TOps.Weight(Unsafe.Add(ref w, 5)))
                + (TOps.Load(Unsafe.Add(ref p, 6 * cn)) * TOps.Weight(Unsafe.Add(ref w, 6))) + (TOps.Load(Unsafe.Add(ref p, 7 * cn)) * TOps.Weight(Unsafe.Add(ref w, 7)));
        }

        return acc;
    }

    private static void SeparableOutlier<T, TWork, TWeight, TOps>(in RemapSource<T> s, Span<T> d, int sx, int sy, ref TWeight wr, int k)
        where T : unmanaged
        where TWork : unmanaged, INumber<TWork>
        where TWeight : unmanaged
        where TOps : struct, IRemapOps<T, TWork, TWeight>
    {
        int cn = s.Cn;
        int half = (k / 2) - 1;
        BorderType border1 = s.Border == BorderType.Transparent ? BorderType.Reflect101 : s.Border;
        if (s.Border == BorderType.Transparent &&
            ((uint)(sx + half) >= (uint)s.Width || (uint)(sy + half) >= (uint)s.Height))
        {
            return;
        }

        if (border1 == BorderType.Constant && (sx >= s.Width || sx + k <= 0 || sy >= s.Height || sy + k <= 0))
        {
            s.Constant.CopyTo(d);
            return;
        }

        Span<int> xs = stackalloc int[8];
        Span<int> ys = stackalloc int[8];
        for (int i = 0; i < k; i++)
        {
            xs[i] = Borders.BorderInterpolate(sx + i, s.Width, border1);
            ys[i] = Borders.BorderInterpolate(sy + i, s.Height, border1);
        }

        for (int c = 0; c < cn; c++)
        {
            TWork cv = TOps.Load(s.Constant[c]);
            TWork sum = cv * TOps.One;
            for (int i = 0; i < k; i++)
            {
                if (ys[i] < 0)
                {
                    continue;
                }

                for (int j = 0; j < k; j++)
                {
                    if (xs[j] >= 0)
                    {
                        sum += (TOps.Load(s.At(ys[i], (xs[j] * cn) + c)) - cv) * TOps.Weight(Unsafe.Add(ref wr, (i * k) + j));
                    }
                }
            }

            d[c] = TOps.Cast(sum);
        }
    }
}
