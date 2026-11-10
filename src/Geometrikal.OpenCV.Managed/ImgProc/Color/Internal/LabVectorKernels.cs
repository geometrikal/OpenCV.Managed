// This implementation is ported from the OpenCV Lab conversion vector kernels.
//
// Original sources:
//   modules/imgproc/src/color_lab.cpp (RGB2Lab_b, RGB2Lab_f, Lab2RGBfloat, Lab2RGBinteger,
//     splineInterpolate, trilinearPackedInterpolate)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// <see cref="Vector128{T}"/> bodies of the Lab kernels. Table reads stay per lane, as in
/// OpenCV (<c>v_lut</c>); the arithmetic between them is vectorized with the operation order of
/// the kernel that would otherwise process each pixel.
/// </summary>
internal static class LabVectorKernels
{
    /// <summary>
    /// sRGB float to Lab through the trilinear table: 8 corner values and 8 weights are
    /// contiguous <see cref="short"/>, so each channel is one 16-bit multiply-add.
    /// </summary>
    public static int SRgbToLabFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int scn, int blueIdx, LabTables.Tables t)
    {
        if (!Ssse3.IsSupported)
        {
            return 0;
        }

        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        ref short lut = ref MemoryMarshal.GetArrayDataReference(t.RgbToLabLut);
        ref short wts = ref MemoryMarshal.GetArrayDataReference(t.TrilinearWeights);
        var scale = Vector128.Create((float)LabTables.LabBase);
        Span<int> idx = stackalloc int[8];
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out _);
            Vector128<int> cx = RoundToInt(Clip((blueIdx == 0 ? p0 : p2).AsSingle()) * scale);
            Vector128<int> cy = RoundToInt(Clip(p1.AsSingle()) * scale);
            Vector128<int> cz = RoundToInt(Clip((blueIdx == 0 ? p2 : p0).AsSingle()) * scale);
            Vector128<int> lutIdx = (Vector128.ShiftRightArithmetic(cx, 9) * 24)
                                    + (Vector128.ShiftRightArithmetic(cy, 9) * (24 * LabTables.LutDim))
                                    + (Vector128.ShiftRightArithmetic(cz, 9) * (24 * LabTables.LutDim * LabTables.LutDim));
            var mask = Vector128.Create(LabTables.TrilinearBase - 1);
            Vector128<int> wIdx = ((Vector128.ShiftRightArithmetic(cx, 5) & mask)
                                   + ((Vector128.ShiftRightArithmetic(cy, 5) & mask) * LabTables.TrilinearBase)
                                   + ((Vector128.ShiftRightArithmetic(cz, 5) & mask) * (LabTables.TrilinearBase * LabTables.TrilinearBase))) * 8;
            lutIdx.StoreUnsafe(ref MemoryMarshal.GetReference(idx));
            wIdx.StoreUnsafe(ref MemoryMarshal.GetReference(idx), 4);

            Vector128<int> l = Interpolate(ref lut, ref wts, idx, 0);
            Vector128<int> a = Interpolate(ref lut, ref wts, idx, 8);
            Vector128<int> b = Interpolate(ref lut, ref wts, idx, 16);
            Vector128<float> lf = Vector128.ConvertToSingle(l) * 1.0f / LabTables.LabBase * 100.0f;
            Vector128<float> af = (Vector128.ConvertToSingle(a) * 1.0f / LabTables.LabBase * 256.0f) - Vector128.Create(128.0f);
            Vector128<float> bf = (Vector128.ConvertToSingle(b) * 1.0f / LabTables.LabBase * 256.0f) - Vector128.Create(128.0f);
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * 3), 3, lf.AsByte(), af.AsByte(), bf.AsByte(), default);
        }

        return x;
    }

    /// <summary>Descaled trilinear sums of one channel (<paramref name="channel"/> 0, 8 or 16) for 4 pixels.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Interpolate(ref short lut, ref short wts, Span<int> idx, int channel)
    {
        Vector128<int> s0 = Dot(ref lut, ref wts, idx[0] + channel, idx[4]);
        Vector128<int> s1 = Dot(ref lut, ref wts, idx[1] + channel, idx[5]);
        Vector128<int> s2 = Dot(ref lut, ref wts, idx[2] + channel, idx[6]);
        Vector128<int> s3 = Dot(ref lut, ref wts, idx[3] + channel, idx[7]);
        Vector128<int> sum = Ssse3.HorizontalAdd(Ssse3.HorizontalAdd(s0, s1), Ssse3.HorizontalAdd(s2, s3));
        return Vector128.ShiftRightArithmetic(sum + Vector128.Create(1 << 11), 12);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Dot(ref short lut, ref short wts, int lutAt, int wAt)
        => Sse2.MultiplyAddAdjacent(
            Vector128.LoadUnsafe(ref lut, (nuint)lutAt),
            Vector128.LoadUnsafe(ref wts, (nuint)wAt));

    /// <summary>OpenCV's <c>clip</c> macro: <c>v &lt; 0 ? 0 : v &lt;= 1 ? v : 1</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Clip(Vector128<float> v)
        => Vector128.ConditionalSelect(
            Vector128.LessThan(v, Vector128<float>.Zero),
            Vector128<float>.Zero,
            Vector128.ConditionalSelect(Vector128.LessThanOrEqual(v, Vector128<float>.One), v, Vector128<float>.One));

    /// <summary><c>max(0, min(v, 1))</c> of the vector bodies.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> VClip(Vector128<float> v)
        => Vector128.Max(Vector128<float>.Zero, Vector128.Min(v, Vector128<float>.One));

    /// <summary>Round half to even, for values inside the <see cref="int"/> range.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> RoundToInt(Vector128<float> v) => Vector128.ConvertToInt32(Vector128.Round(v));

    /// <summary>
    /// <c>splineInterpolate</c> for 4 lanes: per-lane coefficient load, 4x4 transpose, then
    /// <c>((t3*x + t2)*x + t1)*x + t0</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Spline(Vector128<float> x, float[] tab, int n)
    {
        Vector128<int> ix = Vector128.Min(Vector128.Max(Vector128.ConvertToInt32(x), Vector128<int>.Zero), Vector128.Create(n - 1));
        x -= Vector128.ConvertToSingle(ix);
        ref float t = ref MemoryMarshal.GetArrayDataReference(tab);
        Vector128<float> r0 = Vector128.LoadUnsafe(ref t, (nuint)(ix.GetElement(0) * 4));
        Vector128<float> r1 = Vector128.LoadUnsafe(ref t, (nuint)(ix.GetElement(1) * 4));
        Vector128<float> r2 = Vector128.LoadUnsafe(ref t, (nuint)(ix.GetElement(2) * 4));
        Vector128<float> r3 = Vector128.LoadUnsafe(ref t, (nuint)(ix.GetElement(3) * 4));
        Transpose(ref r0, ref r1, ref r2, ref r3);
        return (((((r3 * x) + r2) * x) + r1) * x) + r0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose(ref Vector128<float> a, ref Vector128<float> b, ref Vector128<float> c, ref Vector128<float> d)
    {
        if (Sse.IsSupported)
        {
            Vector128<float> t0 = Sse.UnpackLow(a, b);
            Vector128<float> t1 = Sse.UnpackLow(c, d);
            Vector128<float> t2 = Sse.UnpackHigh(a, b);
            Vector128<float> t3 = Sse.UnpackHigh(c, d);
            a = Sse.MoveLowToHigh(t0, t1);
            b = Sse.MoveHighToLow(t1, t0);
            c = Sse.MoveLowToHigh(t2, t3);
            d = Sse.MoveHighToLow(t3, t2);
            return;
        }

        (a, b, c, d) = (
            Vector128.Create(a.GetElement(0), b.GetElement(0), c.GetElement(0), d.GetElement(0)),
            Vector128.Create(a.GetElement(1), b.GetElement(1), c.GetElement(1), d.GetElement(1)),
            Vector128.Create(a.GetElement(2), b.GetElement(2), c.GetElement(2), d.GetElement(2)),
            Vector128.Create(a.GetElement(3), b.GetElement(3), c.GetElement(3), d.GetElement(3)));
    }

    /// <summary>Linear float RGB to Lab, vector-body formulas; the caller limits the width to the vector part.</summary>
    public static int LinearToLabFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int scn, float[] c, LabTables.Tables t)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        var tabScale = Vector128.Create(LabTables.LabCbrtTabScale);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out _);
            Vector128<float> r = VClip(p0.AsSingle()), g = VClip(p1.AsSingle()), b = VClip(p2.AsSingle());
            Vector128<float> xx = (r * c[0]) + ((g * c[1]) + (b * c[2]));
            Vector128<float> yy = (r * c[3]) + ((g * c[4]) + (b * c[5]));
            Vector128<float> zz = (r * c[6]) + ((g * c[7]) + (b * c[8]));
            Vector128<float> fx = Spline(xx * tabScale, t.LabCbrtTab, LabTables.CbrtTabSize);
            Vector128<float> fy = Spline(yy * tabScale, t.LabCbrtTab, LabTables.CbrtTabSize);
            Vector128<float> fz = Spline(zz * tabScale, t.LabCbrtTab, LabTables.CbrtTabSize);
            Vector128<float> l = Vector128.ConditionalSelect(
                Vector128.GreaterThan(yy, Vector128.Create(0.008856f)),
                (fy * 116f) + Vector128.Create(-16f),
                yy * 903.3f);
            Vector128<float> a = (fx - fy) * 500f;
            Vector128<float> bb = (fy - fz) * 200f;
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * 3), 3, l.AsByte(), a.AsByte(), bb.AsByte(), default);
        }

        return x;
    }

    /// <summary>Lab to float RGB, vector-body formulas; the caller limits the width to the vector part.</summary>
    public static int LabToRgbFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int dcn, bool srgb, float[] c, LabTables.Tables t)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        float bias = 16.0f / 116.0f, inv903 = 1f / 903.3f, inv116 = 1f / 116.0f, inv500 = 1f / 500f, ninv200 = -1f / 200f, inv7787 = 1f / 7.787f;
        var fThresh = Vector128.Create(6f / 29f);
        var vbias = Vector128.Create(bias);
        Vector128<byte> alpha = Vector128.Create(1f).AsByte();
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * 3), 3, out var p0, out var p1, out var p2, out _);
            Vector128<float> li = p0.AsSingle(), ai = p1.AsSingle(), bi = p2.AsSingle();
            Vector128<float> ylo = li * inv903;
            Vector128<float> fylo = (ylo * 7.787f) + vbias;
            Vector128<float> fyhi = (li + Vector128.Create(16.0f)) * inv116;
            Vector128<float> yhi = fyhi * fyhi * fyhi;
            Vector128<float> lo = Vector128.LessThanOrEqual(li, Vector128.Create(8f));
            Vector128<float> y = Vector128.ConditionalSelect(lo, ylo, yhi);
            Vector128<float> fy = Vector128.ConditionalSelect(lo, fylo, fyhi);
            Vector128<float> fx = (ai * inv500) + fy;
            Vector128<float> fz = (bi * ninv200) + fy;
            fx = Vector128.ConditionalSelect(Vector128.LessThanOrEqual(fx, fThresh), (fx - vbias) * inv7787, fx * fx * fx);
            fz = Vector128.ConditionalSelect(Vector128.LessThanOrEqual(fz, fThresh), (fz - vbias) * inv7787, fz * fz * fz);
            Vector128<float> ro = VClip((fx * c[0]) + ((y * c[1]) + (fz * c[2])));
            Vector128<float> go = VClip((fx * c[3]) + ((y * c[4]) + (fz * c[5])));
            Vector128<float> bo = VClip((fx * c[6]) + ((y * c[7]) + (fz * c[8])));
            if (srgb)
            {
                ro = Spline(ro * LabTables.GammaTabScale, t.SRgbInvGammaTab, LabTables.GammaTabSize);
                go = Spline(go * LabTables.GammaTabScale, t.SRgbInvGammaTab, LabTables.GammaTabSize);
                bo = Spline(bo * LabTables.GammaTabScale, t.SRgbInvGammaTab, LabTables.GammaTabSize);
            }

            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * dcn), dcn, ro.AsByte(), go.AsByte(), bo.AsByte(), alpha);
        }

        return x;
    }

    /// <summary><c>RGB2Lab_b</c>: per-lane gamma and cube-root table reads, vector arithmetic.</summary>
    public static int RgbToLabByte(ReadOnlySpan<byte> src, Span<byte> dst, int width, int scn, int[] c, ushort[] gamma, ushort[] cbrt)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        Span<int> buf = stackalloc int[48];
        ref int bufRef = ref MemoryMarshal.GetReference(buf);
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        ref ushort gt = ref MemoryMarshal.GetArrayDataReference(gamma);
        ref ushort ct = ref MemoryMarshal.GetArrayDataReference(cbrt);
        const int lscale = ((116 * 255) + 50) / 100;
        const int lshift = -(((16 * 255 * (1 << LabTables.LabShift2)) + 50) / 100);
        var abias = Vector128.Create(128 * (1 << LabTables.LabShift2));
        Span<Vector128<int>> lab = stackalloc Vector128<int>[12];
        int x = 0;
        for (; x <= width - 16; x += 16)
        {
            ref byte block = ref Unsafe.Add(ref s, x * scn);
            for (int i = 0; i < 16; i++)
            {
                ref byte px = ref Unsafe.Add(ref block, i * scn);
                Unsafe.Add(ref bufRef, i) = Unsafe.Add(ref gt, px);
                Unsafe.Add(ref bufRef, 16 + i) = Unsafe.Add(ref gt, Unsafe.Add(ref px, 1));
                Unsafe.Add(ref bufRef, 32 + i) = Unsafe.Add(ref gt, Unsafe.Add(ref px, 2));
            }

            for (int q = 0; q < 16; q += 4)
            {
                Vector128<int> r = Vector128.LoadUnsafe(ref bufRef, (nuint)q);
                Vector128<int> g = Vector128.LoadUnsafe(ref bufRef, (nuint)(16 + q));
                Vector128<int> b = Vector128.LoadUnsafe(ref bufRef, (nuint)(32 + q));
                DescaleIndex((r * c[0]) + (g * c[1]) + (b * c[2])).StoreUnsafe(ref bufRef, (nuint)q);
                DescaleIndex((r * c[3]) + (g * c[4]) + (b * c[5])).StoreUnsafe(ref bufRef, (nuint)(16 + q));
                DescaleIndex((r * c[6]) + (g * c[7]) + (b * c[8])).StoreUnsafe(ref bufRef, (nuint)(32 + q));
            }

            for (int i = 0; i < 48; i++)
            {
                Unsafe.Add(ref bufRef, i) = Unsafe.Add(ref ct, Unsafe.Add(ref bufRef, i));
            }

            for (int q = 0, k = 0; q < 16; q += 4, k++)
            {
                Vector128<int> fx = Vector128.LoadUnsafe(ref bufRef, (nuint)q);
                Vector128<int> fy = Vector128.LoadUnsafe(ref bufRef, (nuint)(16 + q));
                Vector128<int> fz = Vector128.LoadUnsafe(ref bufRef, (nuint)(32 + q));
                lab[k] = Descale15((fy * lscale) + Vector128.Create(lshift));
                lab[4 + k] = Descale15(((fx - fy) * 500) + abias);
                lab[8 + k] = Descale15(((fy - fz) * 200) + abias);
            }

            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * 3), 3,
                ColorVectors.SaturateToByte(lab[0], lab[1], lab[2], lab[3]),
                ColorVectors.SaturateToByte(lab[4], lab[5], lab[6], lab[7]),
                ColorVectors.SaturateToByte(lab[8], lab[9], lab[10], lab[11]),
                default);
        }

        return x;

        static Vector128<int> DescaleIndex(Vector128<int> v)
            => Vector128.ShiftRightArithmetic(v + Vector128.Create(1 << (LabTables.LabShift - 1)), LabTables.LabShift);

        static Vector128<int> Descale15(Vector128<int> v)
            => Vector128.ShiftRightArithmetic(v + Vector128.Create(1 << (LabTables.LabShift2 - 1)), LabTables.LabShift2);
    }

    /// <summary><c>Lab2RGBinteger</c>: per-lane table reads, vector arithmetic.</summary>
    public static int LabToRgbByte(ReadOnlySpan<byte> src, Span<byte> dst, int width, int dcn, bool srgb, int[] c, LabTables.Tables t)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        Span<int> buf = stackalloc int[64];
        ref int bufRef = ref MemoryMarshal.GetReference(buf);
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        ref ushort yf = ref MemoryMarshal.GetArrayDataReference(t.LabToYFB);
        ref int xz = ref MemoryMarshal.GetArrayDataReference(t.AbToXzB);
        ref ushort inv = ref MemoryMarshal.GetArrayDataReference(t.SRgbInvGammaTabB);
        const int shift = LabTables.LabShift + (14 - LabTables.InvGammaShift);
        var max = Vector128.Create(LabTables.InvGammaTabSize - 1);
        Vector128<byte> alpha = Vector128.Create((byte)255);
        Span<Vector128<int>> rgb = stackalloc Vector128<int>[12];
        int x = 0;
        for (; x <= width - 16; x += 16)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * 3), 3, out _, out var pa, out var pb, out _);
            ColorVectors.WidenToInt32(pa, out var a0, out var a1, out var a2, out var a3);
            ColorVectors.WidenToInt32(pb, out var b0, out var b1, out var b2, out var b3);
            for (int i = 0; i < 16; i++)
            {
                int ll = Unsafe.Add(ref s, ((x + i) * 3)) * 2;
                Unsafe.Add(ref bufRef, i) = Unsafe.Add(ref yf, ll);
                Unsafe.Add(ref bufRef, 16 + i) = Unsafe.Add(ref yf, ll + 1);
            }

            Prepare(ref bufRef, a0, b0, 0);
            Prepare(ref bufRef, a1, b1, 4);
            Prepare(ref bufRef, a2, b2, 8);
            Prepare(ref bufRef, a3, b3, 12);
            for (int i = 32; i < 64; i++)
            {
                Unsafe.Add(ref bufRef, i) = Unsafe.Add(ref xz, Unsafe.Add(ref bufRef, i) - LabTables.MinAbValue);
            }

            for (int q = 0, k = 0; q < 16; q += 4, k++)
            {
                Vector128<int> y = Vector128.LoadUnsafe(ref bufRef, (nuint)q);
                Vector128<int> xx = Vector128.LoadUnsafe(ref bufRef, (nuint)(32 + q));
                Vector128<int> zz = Vector128.LoadUnsafe(ref bufRef, (nuint)(48 + q));
                rgb[k] = Clamp(Descale((xx * c[0]) + (y * c[1]) + (zz * c[2])), max);
                rgb[4 + k] = Clamp(Descale((xx * c[3]) + (y * c[4]) + (zz * c[5])), max);
                rgb[8 + k] = Clamp(Descale((xx * c[6]) + (y * c[7]) + (zz * c[8])), max);
            }

            if (srgb)
            {
                for (int k = 0; k < 12; k++)
                {
                    Vector128<int> v = rgb[k];
                    rgb[k] = Vector128.Create(
                        (int)Unsafe.Add(ref inv, v.GetElement(0)), Unsafe.Add(ref inv, v.GetElement(1)),
                        Unsafe.Add(ref inv, v.GetElement(2)), Unsafe.Add(ref inv, v.GetElement(3)));
                }
            }
            else
            {
                for (int k = 0; k < 12; k++)
                {
                    rgb[k] = Vector128.ShiftRightArithmetic(Vector128.ShiftLeft(rgb[k], 8) - rgb[k], LabTables.InvGammaShift);
                }
            }

            Vector128<byte> ro = ColorVectors.SaturateToByte(rgb[0], rgb[1], rgb[2], rgb[3]);
            Vector128<byte> go = ColorVectors.SaturateToByte(rgb[4], rgb[5], rgb[6], rgb[7]);
            Vector128<byte> bo = ColorVectors.SaturateToByte(rgb[8], rgb[9], rgb[10], rgb[11]);
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * dcn), dcn, bo, go, ro, alpha);
        }

        return x;

        static void Prepare(ref int bufRef, Vector128<int> aa, Vector128<int> bb, int q)
        {
            Vector128<int> ify = Vector128.LoadUnsafe(ref bufRef, (nuint)(16 + q));
            Vector128<int> adiv = Vector128.ShiftRightArithmetic((aa * (5 * 53687)) + Vector128.Create(1 << 7), 13)
                                  - Vector128.Create(128 * LabTables.LabBase / 500);
            Vector128<int> bdiv = Vector128.ShiftRightArithmetic((bb * 41943) + Vector128.Create(1 << 4), 9)
                                  - Vector128.Create((128 * LabTables.LabBase / 200) - 1);
            (ify + adiv).StoreUnsafe(ref bufRef, (nuint)(32 + q));
            (ify - bdiv).StoreUnsafe(ref bufRef, (nuint)(48 + q));
        }

        static Vector128<int> Descale(Vector128<int> v)
            => Vector128.ShiftRightArithmetic(v + Vector128.Create(1 << (shift - 1)), shift);

        static Vector128<int> Clamp(Vector128<int> v, Vector128<int> max) => Vector128.Min(Vector128.Max(v, Vector128<int>.Zero), max);
    }
}
