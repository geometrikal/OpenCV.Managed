// This implementation is ported from the OpenCV colour conversion kernels.
//
// Original sources:
//   modules/imgproc/src/color.simd_helpers.hpp (constants, CV_DESCALE)
//   modules/imgproc/src/color_rgb.simd.hpp (RGB2RGB, RGB2Gray, Gray2RGB)
//   modules/imgproc/src/color_hsv.simd.hpp (RGB2HSV_b, RGB2HSV_f, HSV2RGB_b, HSV2RGB_f)
//   modules/imgproc/src/color_lab.cpp (RGB2XYZ_i, RGB2XYZ_f, XYZ2RGB_i, XYZ2RGB_f,
//     RGB2Lab_b, RGB2Lab_f, Lab2RGBinteger, Lab2RGBfloat)
//   modules/imgproc/src/color_yuv.simd.hpp (RGB2YCrCb_i, RGB2YCrCb_f, YCrCb2RGB_i, YCrCb2RGB_f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Converts one row of <c>width</c> pixels.</summary>
internal interface IColorRow<T>
    where T : unmanaged
{
    void Apply(ReadOnlySpan<T> src, Span<T> dst, int width);
}

/// <summary>
/// Shared constants. Kernels whose OpenCV vector body and scalar tail use different float
/// formulas apply the vector formula to the first <c>width / B * B</c> pixels of each row and
/// the scalar formula to the rest, with the block size <c>B</c> of a 128-bit vector build.
/// </summary>
internal static class ColorMath
{
    public const float FltEpsilon = 1.1920929e-7f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Descale(int x, int n) => (x + (1 << (n - 1))) >> n;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Clip(float v) => v < 0f ? 0f : v <= 1f ? v : 1f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float VClip(float v) => MathF.Max(0f, MathF.Min(v, 1f));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte SatByte(int v) => (byte)Math.Clamp(v, 0, 255);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte RoundByte(float v) => Saturate.FromFloat<byte>(v);

    public static T MaxValue<T>()
        where T : unmanaged
        => typeof(T) == typeof(float) ? Saturate.FromFloat<T>(1f) : Saturate.FromInt<T>((int)Saturate.MaxValue<T>());

    public static T Half<T>()
        where T : unmanaged
        => typeof(T) == typeof(float) ? Saturate.FromFloat<T>(0.5f) : Saturate.FromInt<T>(((int)Saturate.MaxValue<T>() + 1) / 2);
}

/// <summary>Channel reorder and alpha add/remove (<c>RGB2RGB</c>).</summary>
internal readonly struct SwapRow<T>(int scn, int dcn, int blueIdx) : IColorRow<T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> src, Span<T> dst, int width)
    {
        T alpha = ColorMath.MaxValue<T>();
        int bi = blueIdx;
        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x0 = ColorVectorKernels.Swap(src, dst, width, scn, dcn, bi);
        for (int x = x0, i = x0 * scn, j = x0 * dcn; x < width; x++, i += scn, j += dcn)
        {
            T t0 = Unsafe.Add(ref s, i), t1 = Unsafe.Add(ref s, i + 1), t2 = Unsafe.Add(ref s, i + 2);
            Unsafe.Add(ref d, j + bi) = t0;
            Unsafe.Add(ref d, j + 1) = t1;
            Unsafe.Add(ref d, j + (bi ^ 2)) = t2;
            if (dcn == 4)
            {
                Unsafe.Add(ref d, j + 3) = scn == 4 ? Unsafe.Add(ref s, i + 3) : alpha;
            }
        }
    }
}

/// <summary><c>RGB2Gray</c> for 8- and 16-bit: 15-bit fixed-point weights.</summary>
internal readonly struct GrayIntRow<T>(int scn, int blueIdx) : IColorRow<T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> src, Span<T> dst, int width)
    {
        const int ry = 9798, gy = 19235, by = 3735;
        int c0 = blueIdx == 0 ? by : ry;
        int c2 = blueIdx == 0 ? ry : by;
        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x0 = ColorVectorKernels.GrayInt(src, dst, width, scn, c0, gy, c2);
        for (int x = x0, i = x0 * scn; x < width; x++, i += scn)
        {
            uint sum = (uint)((Saturate.ToInt(Unsafe.Add(ref s, i)) * c0) +
                              (Saturate.ToInt(Unsafe.Add(ref s, i + 1)) * gy) +
                              (Saturate.ToInt(Unsafe.Add(ref s, i + 2)) * c2));
            Unsafe.Add(ref d, x) = Saturate.FromInt<T>((int)((sum + (1u << 14)) >> 15));
        }
    }
}

internal readonly struct GrayFloatRow(int scn, int blueIdx) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        float c0 = blueIdx == 0 ? 0.114f : 0.299f;
        float c2 = blueIdx == 0 ? 0.299f : 0.114f;
        const float c1 = 0.587f;
        int x0 = ColorVectorKernels.GrayFloat(src, dst, width, scn, c0, c1, c2);
        for (int x = x0, s = x0 * scn; x < width; x++, s += scn)
        {
            dst[x] = (src[s] * c0) + (src[s + 1] * c1) + (src[s + 2] * c2);
        }
    }
}

internal readonly struct GrayToColorRow<T>(int dcn) : IColorRow<T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> src, Span<T> dst, int width)
    {
        T alpha = ColorMath.MaxValue<T>();
        int x0 = ColorVectorKernels.GrayToColor(src, dst, width, dcn);
        for (int x = x0, d = x0 * dcn; x < width; x++, d += dcn)
        {
            T v = src[x];
            dst[d] = dst[d + 1] = dst[d + 2] = v;
            if (dcn == 4)
            {
                dst[d + 3] = alpha;
            }
        }
    }
}

/// <summary><c>RGB2HSV_b</c>: 12-bit fixed point with division tables.</summary>
internal readonly struct HsvByteRow(int scn, int blueIdx, int hrange) : IColorRow<byte>
{
    private static readonly int[] SDiv = Table(255, 1.0);
    private static readonly int[] HDiv180 = Table(180, 6.0);
    private static readonly int[] HDiv256 = Table(256, 6.0);

    public void Apply(ReadOnlySpan<byte> src, Span<byte> dst, int width)
    {
        const int shift = 12;
        int[] hdiv = hrange == 180 ? HDiv180 : HDiv256;
        int[] sdiv = SDiv;
        int hr = hrange;
        int bi = blueIdx;
        ref byte sp = ref MemoryMarshal.GetReference(src);
        ref byte dp = ref MemoryMarshal.GetReference(dst);
        ref int sd = ref MemoryMarshal.GetArrayDataReference(sdiv);
        ref int hd = ref MemoryMarshal.GetArrayDataReference(hdiv);
        int x0 = ColorVectorKernels.HsvByte(src, dst, width, scn, bi, hr, sdiv, hdiv);
        for (int x = x0, s = x0 * scn, d = x0 * 3; x < width; x++, s += scn, d += 3)
        {
            int b = Unsafe.Add(ref sp, s + bi), g = Unsafe.Add(ref sp, s + 1), r = Unsafe.Add(ref sp, s + (bi ^ 2));
            int v = BranchFree.Max(b, BranchFree.Max(g, r));
            int vmin = BranchFree.Min(b, BranchFree.Min(g, r));
            int diff = v - vmin;
            int vr = ((v ^ r) - 1) >> 31;
            int vg = ((v ^ g) - 1) >> 31;
            int sat = ((diff * Unsafe.Add(ref sd, v)) + (1 << (shift - 1))) >> shift;
            int h = (vr & (g - b)) + (~vr & ((vg & (b - r + (2 * diff))) + (~vg & (r - g + (4 * diff)))));
            h = ((h * Unsafe.Add(ref hd, diff)) + (1 << (shift - 1))) >> shift;
            h += (h >> 31) & hr;
            Unsafe.Add(ref dp, d) = (byte)BranchFree.Min(h, 255);
            Unsafe.Add(ref dp, d + 1) = (byte)sat;
            Unsafe.Add(ref dp, d + 2) = (byte)v;
        }
    }

    private static int[] Table(int numerator, double factor)
    {
        var t = new int[256];
        for (int i = 1; i < 256; i++)
        {
            t[i] = ScalarConverter.CvRound((numerator << 12) / (factor * i));
        }

        return t;
    }
}

/// <summary><c>RGB2HSV_f</c>: hue in degrees.</summary>
internal readonly struct HsvFloatRow(int scn, int blueIdx) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        int x0 = ColorVectorKernels.HsvFloat(src, dst, width, scn, blueIdx);
        for (int x = x0, s = x0 * scn, d = x0 * 3; x < width; x++, s += scn, d += 3)
        {
            float b = src[s + blueIdx], g = src[s + 1], r = src[s + (blueIdx ^ 2)];
            float v = r, vmin = r;
            if (v < g) v = g;
            if (v < b) v = b;
            if (vmin > g) vmin = g;
            if (vmin > b) vmin = b;
            float diff = v - vmin;
            float sat = diff / (MathF.Abs(v) + ColorMath.FltEpsilon);
            diff = (float)(60.0 / (diff + ColorMath.FltEpsilon));
            float h = v == r ? (g - b) * diff : v == g ? ((b - r) * diff) + 120f : ((r - g) * diff) + 240f;
            if (h < 0)
            {
                h += 360f;
            }

            dst[d] = h;
            dst[d + 1] = sat;
            dst[d + 2] = v;
        }
    }
}

/// <summary>Integer minimum and maximum without branches, for unpredictable comparisons.</summary>
internal static class BranchFree
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Max(int a, int b)
    {
        int d = a - b;
        return a - (d & (d >> 31));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Min(int a, int b)
    {
        int d = a - b;
        return b + (d & (d >> 31));
    }
}

internal static class HsvMath
{
    /// <summary><c>HSV2RGB_native</c> (OpenCV 5.0 sector wrap).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ToRgb(float h, float s, float v, float hscale, out float b, out float g, out float r)
    {
        if (s == 0)
        {
            b = g = r = v;
            return;
        }

        h *= hscale;
        int sector = (int)MathF.Floor(h);
        h -= sector;
        sector %= 6;
        sector += (sector >> 31) & 6;
        Vector128<float> tab = Vector128.Create(v, v * (1f - s), v * (1f - (s * h)), v * (1f - (s * (1f - h))));
        ReadOnlySpan<byte> sectorData = [1, 3, 0, 1, 0, 2, 3, 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 0];
        int k = sector * 3;
        b = tab.GetElement(sectorData[k]);
        g = tab.GetElement(sectorData[k + 1]);
        r = tab.GetElement(sectorData[k + 2]);
    }
}

/// <summary>
/// <c>HSV2RGB_b</c>: the vector body (blocks of 16 pixels) truncates <c>x * 255</c>; the scalar
/// tail rounds it.
/// </summary>
internal readonly struct HsvToRgbByteRow(int dcn, int blueIdx, int hrange) : IColorRow<byte>
{
    public void Apply(ReadOnlySpan<byte> src, Span<byte> dst, int width)
    {
        float hscale = 6.0f / hrange;
        int vectorEnd = width / 16 * 16;
        int x0 = ColorVectorKernels.HsvToRgbByte(src, dst, vectorEnd, dcn, blueIdx, hscale);
        for (int x = x0, s = x0 * 3, d = x0 * dcn; x < width; x++, s += 3, d += dcn)
        {
            float h = src[s];
            float sat = src[s + 1] * (1.0f / 255.0f);
            float v = src[s + 2] * (1.0f / 255.0f);
            HsvMath.ToRgb(h, sat, v, hscale, out float b, out float g, out float r);
            if (x < vectorEnd)
            {
                dst[d + blueIdx] = ColorMath.SatByte((int)(b * 255.0f));
                dst[d + 1] = ColorMath.SatByte((int)(g * 255.0f));
                dst[d + (blueIdx ^ 2)] = ColorMath.SatByte((int)(r * 255.0f));
            }
            else
            {
                dst[d + blueIdx] = ColorMath.RoundByte(b * 255.0f);
                dst[d + 1] = ColorMath.RoundByte(g * 255.0f);
                dst[d + (blueIdx ^ 2)] = ColorMath.RoundByte(r * 255.0f);
            }

            if (dcn == 4)
            {
                dst[d + 3] = 255;
            }
        }
    }
}

internal readonly struct HsvToRgbFloatRow(int dcn, int blueIdx) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        const float hscale = 6f / 360f;
        int x0 = ColorVectorKernels.HsvToRgbFloat(src, dst, width, dcn, blueIdx);
        for (int x = x0, s = x0 * 3, d = x0 * dcn; x < width; x++, s += 3, d += dcn)
        {
            HsvMath.ToRgb(src[s], src[s + 1], src[s + 2], hscale, out float b, out float g, out float r);
            dst[d + blueIdx] = b;
            dst[d + 1] = g;
            dst[d + (blueIdx ^ 2)] = r;
            if (dcn == 4)
            {
                dst[d + 3] = 1f;
            }
        }
    }
}

internal static class XyzCoefficients
{
    private static readonly int[] RgbToXyzInt = [1689, 1465, 739, 871, 2929, 296, 79, 488, 3892];
    private static readonly int[] XyzToRgbInt = [13273, -6296, -2042, -3970, 7684, 170, 228, -836, 4331];

    /// <summary>Forward coefficients per output row; columns follow the source channel order.</summary>
    public static T[] Forward<T>(int blueIdx, Func<int, T> pick)
    {
        var c = new T[9];
        for (int i = 0; i < 9; i++)
        {
            c[i] = pick(i);
        }

        if (blueIdx == 0)
        {
            (c[0], c[2]) = (c[2], c[0]);
            (c[3], c[5]) = (c[5], c[3]);
            (c[6], c[8]) = (c[8], c[6]);
        }

        return c;
    }

    /// <summary>Inverse coefficients; row <c>k</c> produces destination channel <c>k</c>.</summary>
    public static T[] Inverse<T>(int blueIdx, Func<int, T> pick)
    {
        var c = new T[9];
        for (int i = 0; i < 9; i++)
        {
            c[i] = pick(i);
        }

        if (blueIdx == 0)
        {
            (c[0], c[6]) = (c[6], c[0]);
            (c[1], c[7]) = (c[7], c[1]);
            (c[2], c[8]) = (c[8], c[2]);
        }

        return c;
    }

    public static int[] ForwardInt(int blueIdx) => Forward(blueIdx, i => RgbToXyzInt[i]);

    public static int[] InverseInt(int blueIdx) => Inverse(blueIdx, i => XyzToRgbInt[i]);

    public static float[] ForwardFloat(int blueIdx) => Forward(blueIdx, i => (float)LabTables.SRgb2XyzD65[i]);

    public static float[] InverseFloat(int blueIdx) => Inverse(blueIdx, i => (float)LabTables.Xyz2SRgbD65[i]);
}

/// <summary>3x3 integer transform with 12-bit coefficients (<c>RGB2XYZ_i</c>, <c>XYZ2RGB_i</c>).</summary>
internal readonly struct MatrixIntRow<T>(int scn, int dcn, int[] c) : IColorRow<T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> src, Span<T> dst, int width)
    {
        T alpha = ColorMath.MaxValue<T>();
        int x0 = ColorVectorKernels.MatrixInt(src, dst, width, scn, dcn, c);
        for (int x = x0, s = x0 * scn, d = x0 * dcn; x < width; x++, s += scn, d += dcn)
        {
            int s0 = Saturate.ToInt(src[s]), s1 = Saturate.ToInt(src[s + 1]), s2 = Saturate.ToInt(src[s + 2]);
            dst[d] = Saturate.FromInt<T>(ColorMath.Descale((s0 * c[0]) + (s1 * c[1]) + (s2 * c[2]), 12));
            dst[d + 1] = Saturate.FromInt<T>(ColorMath.Descale((s0 * c[3]) + (s1 * c[4]) + (s2 * c[5]), 12));
            dst[d + 2] = Saturate.FromInt<T>(ColorMath.Descale((s0 * c[6]) + (s1 * c[7]) + (s2 * c[8]), 12));
            if (dcn == 4)
            {
                dst[d + 3] = alpha;
            }
        }
    }
}

/// <summary>
/// 3x3 float transform (<c>RGB2XYZ_f</c>, <c>XYZ2RGB_f</c>). Vector body (blocks of 4):
/// <c>s0*c0 + (s1*c1 + s2*c2)</c>; scalar tail: <c>(s0*c0 + s1*c1) + s2*c2</c>.
/// </summary>
internal readonly struct MatrixFloatRow(int scn, int dcn, float[] c) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        int vectorEnd = width / 4 * 4;
        int x0 = ColorVectorKernels.MatrixFloat(src, dst, vectorEnd, scn, dcn, c);
        for (int x = x0, s = x0 * scn, d = x0 * dcn; x < width; x++, s += scn, d += dcn)
        {
            float s0 = src[s], s1 = src[s + 1], s2 = src[s + 2];
            if (x < vectorEnd)
            {
                dst[d] = (s0 * c[0]) + ((s1 * c[1]) + (s2 * c[2]));
                dst[d + 1] = (s0 * c[3]) + ((s1 * c[4]) + (s2 * c[5]));
                dst[d + 2] = (s0 * c[6]) + ((s1 * c[7]) + (s2 * c[8]));
            }
            else
            {
                dst[d] = (s0 * c[0]) + (s1 * c[1]) + (s2 * c[2]);
                dst[d + 1] = (s0 * c[3]) + (s1 * c[4]) + (s2 * c[5]);
                dst[d + 2] = (s0 * c[6]) + (s1 * c[7]) + (s2 * c[8]);
            }

            if (dcn == 4)
            {
                dst[d + 3] = 1f;
            }
        }
    }
}

/// <summary><c>RGB2YCrCb_i</c>: 14-bit fixed point; output order Y, Cr, Cb.</summary>
internal readonly struct YCrCbIntRow<T>(int scn, int blueIdx) : IColorRow<T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> src, Span<T> dst, int width)
    {
        const int shift = 14;
        int c0 = blueIdx == 0 ? 1868 : 4899, c1 = 9617, c2 = blueIdx == 0 ? 4899 : 1868;
        int delta = Saturate.ToInt(ColorMath.Half<T>()) << shift;
        int x0 = ColorVectorKernels.YCrCbInt(src, dst, width, scn, blueIdx);
        for (int x = x0, s = x0 * scn, d = x0 * 3; x < width; x++, s += scn, d += 3)
        {
            int s0 = Saturate.ToInt(src[s]), s1 = Saturate.ToInt(src[s + 1]), s2 = Saturate.ToInt(src[s + 2]);
            int y = ColorMath.Descale((s0 * c0) + (s1 * c1) + (s2 * c2), shift);
            int r = blueIdx == 0 ? s2 : s0;
            int b = blueIdx == 0 ? s0 : s2;
            int cr = ColorMath.Descale(((r - y) * 11682) + delta, shift);
            int cb = ColorMath.Descale(((b - y) * 9241) + delta, shift);
            dst[d] = Saturate.FromInt<T>(y);
            dst[d + 1] = Saturate.FromInt<T>(cr);
            dst[d + 2] = Saturate.FromInt<T>(cb);
        }
    }
}

/// <summary>
/// <c>RGB2YCrCb_f</c>. Vector body (blocks of 4): <c>Y = s0*c0 + (s1*c1 + s2*c2)</c>; scalar
/// tail: <c>Y = (s0*c0 + s1*c1) + s2*c2</c>.
/// </summary>
internal readonly struct YCrCbFloatRow(int scn, int blueIdx) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        float c0 = blueIdx == 0 ? 0.114f : 0.299f, c1 = 0.587f, c2 = blueIdx == 0 ? 0.299f : 0.114f;
        int vectorEnd = width / 4 * 4;
        int x0 = ColorVectorKernels.YCrCbFloat(src, dst, vectorEnd, scn, blueIdx);
        for (int x = x0, s = x0 * scn, d = x0 * 3; x < width; x++, s += scn, d += 3)
        {
            float s0 = src[s], s1 = src[s + 1], s2 = src[s + 2];
            float y = x < vectorEnd ? (s0 * c0) + ((s1 * c1) + (s2 * c2)) : (s0 * c0) + (s1 * c1) + (s2 * c2);
            float r = blueIdx == 0 ? s2 : s0;
            float b = blueIdx == 0 ? s0 : s2;
            dst[d] = y;
            dst[d + 1] = ((r - y) * 0.713f) + 0.5f;
            dst[d + 2] = ((b - y) * 0.564f) + 0.5f;
        }
    }
}

/// <summary><c>YCrCb2RGB_i</c>: 14-bit fixed point.</summary>
internal readonly struct YCrCbToRgbIntRow<T>(int dcn, int blueIdx) : IColorRow<T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> src, Span<T> dst, int width)
    {
        const int shift = 14;
        int half = Saturate.ToInt(ColorMath.Half<T>());
        T alpha = ColorMath.MaxValue<T>();
        int x0 = ColorVectorKernels.YCrCbToRgbInt(src, dst, width, dcn, blueIdx);
        for (int x = x0, s = x0 * 3, d = x0 * dcn; x < width; x++, s += 3, d += dcn)
        {
            int y = Saturate.ToInt(src[s]);
            int cr = Saturate.ToInt(src[s + 1]) - half;
            int cb = Saturate.ToInt(src[s + 2]) - half;
            int b = y + ColorMath.Descale(cb * 29049, shift);
            int g = y + ColorMath.Descale((cb * -5636) + (cr * -11698), shift);
            int r = y + ColorMath.Descale(cr * 22987, shift);
            dst[d + blueIdx] = Saturate.FromInt<T>(b);
            dst[d + 1] = Saturate.FromInt<T>(g);
            dst[d + (blueIdx ^ 2)] = Saturate.FromInt<T>(r);
            if (dcn == 4)
            {
                dst[d + 3] = alpha;
            }
        }
    }
}

internal readonly struct YCrCbToRgbFloatRow(int dcn, int blueIdx) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        int x0 = ColorVectorKernels.YCrCbToRgbFloat(src, dst, width, dcn, blueIdx);
        for (int x = x0, s = x0 * 3, d = x0 * dcn; x < width; x++, s += 3, d += dcn)
        {
            float y = src[s], cr = src[s + 1] - 0.5f, cb = src[s + 2] - 0.5f;
            dst[d + blueIdx] = y + (cb * 1.773f);
            dst[d + 1] = y + (cb * -0.344f) + (cr * -0.714f);
            dst[d + (blueIdx ^ 2)] = y + (cr * 1.403f);
            if (dcn == 4)
            {
                dst[d + 3] = 1f;
            }
        }
    }
}

/// <summary><c>RGB2Lab_b</c>: gamma table, 12-bit matrix and cube-root table.</summary>
internal readonly struct LabByteRow(int scn, int blueIdx, bool srgb) : IColorRow<byte>
{
    public void Apply(ReadOnlySpan<byte> src, Span<byte> dst, int width)
    {
        LabTables.Tables t = LabTables.Get;
        int[] c = LabCoefficients.ForwardInt(blueIdx);
        ushort[] tab = srgb ? t.SRgbGammaTabB : t.LinearGammaTabB;
        ushort[] cbrt = t.LabCbrtTabB;
        const int lscale = ((116 * 255) + 50) / 100;
        const int lshift = -(((16 * 255 * (1 << LabTables.LabShift2)) + 50) / 100);
        int x0 = LabVectorKernels.RgbToLabByte(src, dst, width, scn, c, tab, cbrt);
        for (int x = x0, s = x0 * scn, d = x0 * 3; x < width; x++, s += scn, d += 3)
        {
            int r = tab[src[s]], g = tab[src[s + 1]], b = tab[src[s + 2]];
            int fx = cbrt[ColorMath.Descale((r * c[0]) + (g * c[1]) + (b * c[2]), LabTables.LabShift)];
            int fy = cbrt[ColorMath.Descale((r * c[3]) + (g * c[4]) + (b * c[5]), LabTables.LabShift)];
            int fz = cbrt[ColorMath.Descale((r * c[6]) + (g * c[7]) + (b * c[8]), LabTables.LabShift)];
            int l = ColorMath.Descale((lscale * fy) + lshift, LabTables.LabShift2);
            int a = ColorMath.Descale((500 * (fx - fy)) + (128 * (1 << LabTables.LabShift2)), LabTables.LabShift2);
            int bb = ColorMath.Descale((200 * (fy - fz)) + (128 * (1 << LabTables.LabShift2)), LabTables.LabShift2);
            dst[d] = ColorMath.SatByte(l);
            dst[d + 1] = ColorMath.SatByte(a);
            dst[d + 2] = ColorMath.SatByte(bb);
        }
    }
}

/// <summary><c>Lab2RGBinteger</c>.</summary>
internal readonly struct LabToRgbByteRow(int dcn, int blueIdx, bool srgb) : IColorRow<byte>
{
    public void Apply(ReadOnlySpan<byte> src, Span<byte> dst, int width)
    {
        LabTables.Tables t = LabTables.Get;
        int[] c = LabCoefficients.InverseInt(blueIdx);
        const int shift = LabTables.LabShift + (14 - LabTables.InvGammaShift);
        const int max = LabTables.InvGammaTabSize - 1;
        int x0 = LabVectorKernels.LabToRgbByte(src, dst, width, dcn, srgb, c, t);
        for (int x = x0, s = x0 * 3, d = x0 * dcn; x < width; x++, s += 3, d += dcn)
        {
            int ll = src[s], aa = src[s + 1], bb = src[s + 2];
            int y = t.LabToYFB[ll * 2];
            int ify = t.LabToYFB[(ll * 2) + 1];
            int adiv = (((5 * aa * 53687) + (1 << 7)) >> 13) - (128 * LabTables.LabBase / 500);
            int bdiv = (((bb * 41943) + (1 << 4)) >> 9) - (128 * LabTables.LabBase / 200) + 1;
            int xx = t.AbToXzB[ify + adiv - LabTables.MinAbValue];
            int zz = t.AbToXzB[ify - bdiv - LabTables.MinAbValue];
            int ro = Math.Clamp(ColorMath.Descale((c[0] * xx) + (c[1] * y) + (c[2] * zz), shift), 0, max);
            int go = Math.Clamp(ColorMath.Descale((c[3] * xx) + (c[4] * y) + (c[5] * zz), shift), 0, max);
            int bo = Math.Clamp(ColorMath.Descale((c[6] * xx) + (c[7] * y) + (c[8] * zz), shift), 0, max);
            if (srgb)
            {
                ro = t.SRgbInvGammaTabB[ro];
                go = t.SRgbInvGammaTabB[go];
                bo = t.SRgbInvGammaTabB[bo];
            }
            else
            {
                ro = ((ro << 8) - ro) >> LabTables.InvGammaShift;
                go = ((go << 8) - go) >> LabTables.InvGammaShift;
                bo = ((bo << 8) - bo) >> LabTables.InvGammaShift;
            }

            dst[d] = ColorMath.SatByte(bo);
            dst[d + 1] = ColorMath.SatByte(go);
            dst[d + 2] = ColorMath.SatByte(ro);
            if (dcn == 4)
            {
                dst[d + 3] = 255;
            }
        }
    }
}

/// <summary>
/// <c>RGB2Lab_f</c>. sRGB sources use the trilinear table on 14-bit quantized values. Linear
/// sources: vector body (blocks of 8) uses the cube-root spline and
/// <c>R*c0 + (G*c1 + B*c2)</c>; scalar tail uses <c>cubeRoot</c> and
/// <c>(R*c0 + G*c1) + B*c2</c>.
/// </summary>
internal readonly struct LabFloatRow(int scn, int blueIdx, bool srgb) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        LabTables.Tables t = LabTables.Get;
        if (srgb)
        {
            int xs = LabVectorKernels.SRgbToLabFloat(src, dst, width, scn, blueIdx, t);
            for (int x = xs, s = xs * scn, d = xs * 3; x < width; x++, s += scn, d += 3)
            {
                float r = ColorMath.Clip(src[s + blueIdx]);
                float g = ColorMath.Clip(src[s + 1]);
                float b = ColorMath.Clip(src[s + (blueIdx ^ 2)]);
                int ir = ScalarConverter.CvRound(r * LabTables.LabBase);
                int ig = ScalarConverter.CvRound(g * LabTables.LabBase);
                int ib = ScalarConverter.CvRound(b * LabTables.LabBase);
                LabTables.TrilinearInterpolate(ir, ig, ib, t.RgbToLabLut, t.TrilinearWeights, out int il, out int ia, out int iab);
                dst[d] = il * 1.0f / LabTables.LabBase * 100.0f;
                dst[d + 1] = (ia * 1.0f / LabTables.LabBase * 256.0f) - 128.0f;
                dst[d + 2] = (iab * 1.0f / LabTables.LabBase * 256.0f) - 128.0f;
            }

            return;
        }

        float[] c = LabCoefficients.ForwardFloat(blueIdx);
        float bias = 16f / 116f;
        int vectorEnd = width / 8 * 8;
        int x0 = LabVectorKernels.LinearToLabFloat(src, dst, vectorEnd, scn, c, t);
        for (int x = x0, s = x0 * scn, d = x0 * 3; x < width; x++, s += scn, d += 3)
        {
            float l, a, bb;
            if (x < vectorEnd)
            {
                float r = ColorMath.VClip(src[s]), g = ColorMath.VClip(src[s + 1]), b = ColorMath.VClip(src[s + 2]);
                float xx = (r * c[0]) + ((g * c[1]) + (b * c[2]));
                float yy = (r * c[3]) + ((g * c[4]) + (b * c[5]));
                float zz = (r * c[6]) + ((g * c[7]) + (b * c[8]));
                float fx = LabTables.SplineInterpolate(xx * LabTables.LabCbrtTabScale, t.LabCbrtTab, LabTables.CbrtTabSize);
                float fy = LabTables.SplineInterpolate(yy * LabTables.LabCbrtTabScale, t.LabCbrtTab, LabTables.CbrtTabSize);
                float fz = LabTables.SplineInterpolate(zz * LabTables.LabCbrtTabScale, t.LabCbrtTab, LabTables.CbrtTabSize);
                l = yy > 0.008856f ? (116f * fy) + -16f : 903.3f * yy;
                a = 500f * (fx - fy);
                bb = 200f * (fy - fz);
            }
            else
            {
                float r = ColorMath.Clip(src[s]), g = ColorMath.Clip(src[s + 1]), b = ColorMath.Clip(src[s + 2]);
                float xx = (r * c[0]) + (g * c[1]) + (b * c[2]);
                float yy = (r * c[3]) + (g * c[4]) + (b * c[5]);
                float zz = (r * c[6]) + (g * c[7]) + (b * c[8]);
                float fx = xx > 0.008856f ? LabTables.CubeRoot(xx) : (7.787f * xx) + bias;
                float fy = yy > 0.008856f ? LabTables.CubeRoot(yy) : (7.787f * yy) + bias;
                float fz = zz > 0.008856f ? LabTables.CubeRoot(zz) : (7.787f * zz) + bias;
                l = yy > 0.008856f ? (116f * fy) - 16f : 903.3f * yy;
                a = 500f * (fx - fy);
                bb = 200f * (fy - fz);
            }

            dst[d] = l;
            dst[d + 1] = a;
            dst[d + 2] = bb;
        }
    }
}

/// <summary>
/// <c>Lab2RGBfloat</c>. The vector body (blocks of 8) multiplies by reciprocals and computes
/// <c>c0*x + (c1*y + c2*z)</c>; the scalar tail divides and computes <c>(c0*x + c1*y) + c2*z</c>.
/// </summary>
internal readonly struct LabToRgbFloatRow(int dcn, int blueIdx, bool srgb) : IColorRow<float>
{
    public void Apply(ReadOnlySpan<float> src, Span<float> dst, int width)
    {
        LabTables.Tables t = LabTables.Get;
        float[] c = LabCoefficients.InverseFloat(blueIdx);
        const float lThresh = 8f;
        float fThresh = 6f / 29f;
        float bias = 16.0f / 116.0f;
        float inv903 = 1f / 903.3f, inv116 = 1f / 116.0f, inv500 = 1f / 500f, ninv200 = -1f / 200f, inv7787 = 1f / 7.787f;
        int vectorEnd = width / 8 * 8;
        int x0 = LabVectorKernels.LabToRgbFloat(src, dst, vectorEnd, dcn, srgb, c, t);
        for (int x = x0, s = x0 * 3, d = x0 * dcn; x < width; x++, s += 3, d += dcn)
        {
            float li = src[s], ai = src[s + 1], bi = src[s + 2];
            float ro, go, bo;
            if (x < vectorEnd)
            {
                float ylo = li * inv903;
                float fylo = (7.787f * ylo) + bias;
                float fyhi = (li + 16.0f) * inv116;
                float yhi = fyhi * fyhi * fyhi;
                bool lo = li <= lThresh;
                float y = lo ? ylo : yhi;
                float fy = lo ? fylo : fyhi;
                float fx = (ai * inv500) + fy;
                float fz = (bi * ninv200) + fy;
                fx = fx <= fThresh ? (fx - bias) * inv7787 : fx * fx * fx;
                fz = fz <= fThresh ? (fz - bias) * inv7787 : fz * fz * fz;
                ro = ColorMath.VClip((c[0] * fx) + ((c[1] * y) + (c[2] * fz)));
                go = ColorMath.VClip((c[3] * fx) + ((c[4] * y) + (c[5] * fz)));
                bo = ColorMath.VClip((c[6] * fx) + ((c[7] * y) + (c[8] * fz)));
            }
            else
            {
                float y, fy;
                if (li <= lThresh)
                {
                    y = li / 903.3f;
                    fy = (7.787f * y) + bias;
                }
                else
                {
                    fy = (li + 16.0f) / 116.0f;
                    y = fy * fy * fy;
                }

                float fx = (ai / 500.0f) + fy;
                float fz = fy - (bi / 200.0f);
                fx = fx <= fThresh ? (fx - bias) / 7.787f : fx * fx * fx;
                fz = fz <= fThresh ? (fz - bias) / 7.787f : fz * fz * fz;
                ro = ColorMath.Clip((c[0] * fx) + (c[1] * y) + (c[2] * fz));
                go = ColorMath.Clip((c[3] * fx) + (c[4] * y) + (c[5] * fz));
                bo = ColorMath.Clip((c[6] * fx) + (c[7] * y) + (c[8] * fz));
            }

            if (srgb)
            {
                ro = LabTables.SplineInterpolate(ro * LabTables.GammaTabScale, t.SRgbInvGammaTab, LabTables.GammaTabSize);
                go = LabTables.SplineInterpolate(go * LabTables.GammaTabScale, t.SRgbInvGammaTab, LabTables.GammaTabSize);
                bo = LabTables.SplineInterpolate(bo * LabTables.GammaTabScale, t.SRgbInvGammaTab, LabTables.GammaTabSize);
            }

            dst[d] = ro;
            dst[d + 1] = go;
            dst[d + 2] = bo;
            if (dcn == 4)
            {
                dst[d + 3] = 1f;
            }
        }
    }
}

internal static class LabCoefficients
{
    /// <summary>
    /// <c>RGB2Lab_b</c> coefficients: <c>cvRound(4096 * M / white)</c>; column order follows the
    /// source channels.
    /// </summary>
    public static int[] ForwardInt(int blueIdx)
    {
        var c = new int[9];
        double[] m = LabTables.SRgb2XyzD65;
        for (int i = 0; i < 3; i++)
        {
            double w = LabTables.D65[i];
            c[(i * 3) + (blueIdx ^ 2)] = ScalarConverter.CvRound(4096.0 * m[i * 3] / w);
            c[(i * 3) + 1] = ScalarConverter.CvRound(4096.0 * m[(i * 3) + 1] / w);
            c[(i * 3) + blueIdx] = ScalarConverter.CvRound(4096.0 * m[(i * 3) + 2] / w);
        }

        return c;
    }

    /// <summary><c>RGB2Lab_f</c> coefficients: <c>(float)(scale * M)</c>.</summary>
    public static float[] ForwardFloat(int blueIdx)
    {
        var c = new float[9];
        double[] m = LabTables.SRgb2XyzD65;
        double[] scale = [1.0 / LabTables.D65[0], 1.0, 1.0 / LabTables.D65[2]];
        for (int i = 0; i < 3; i++)
        {
            c[(i * 3) + (blueIdx ^ 2)] = (float)(scale[i] * m[i * 3]);
            c[(i * 3) + 1] = (float)(scale[i] * m[(i * 3) + 1]);
            c[(i * 3) + blueIdx] = (float)(scale[i] * m[(i * 3) + 2]);
        }

        return c;
    }

    /// <summary><c>Lab2RGBinteger</c> coefficients: <c>cvRound(4096 * M' * white)</c>.</summary>
    public static int[] InverseInt(int blueIdx)
    {
        var c = new int[9];
        double[] m = LabTables.Xyz2SRgbD65;
        for (int i = 0; i < 3; i++)
        {
            double w = LabTables.D65[i];
            c[i + (blueIdx * 3)] = ScalarConverter.CvRound(4096.0 * m[i] * w);
            c[i + 3] = ScalarConverter.CvRound(4096.0 * m[i + 3] * w);
            c[i + ((blueIdx ^ 2) * 3)] = ScalarConverter.CvRound(4096.0 * m[i + 6] * w);
        }

        return c;
    }

    /// <summary><c>Lab2RGBfloat</c> coefficients: <c>(float)(M' * white)</c>.</summary>
    public static float[] InverseFloat(int blueIdx)
    {
        var c = new float[9];
        double[] m = LabTables.Xyz2SRgbD65;
        for (int i = 0; i < 3; i++)
        {
            double w = LabTables.D65[i];
            c[i + ((blueIdx ^ 2) * 3)] = (float)(m[i] * w);
            c[i + 3] = (float)(m[i + 3] * w);
            c[i + (blueIdx * 3)] = (float)(m[i + 6] * w);
        }

        return c;
    }
}
