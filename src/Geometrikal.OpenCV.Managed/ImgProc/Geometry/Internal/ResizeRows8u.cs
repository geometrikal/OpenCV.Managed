// This implementation is ported from the OpenCV separable resize for 8-bit images.
//
// Original source:
//   modules/imgproc/src/resize.cpp (HResizeLinear, HResizeCubic, HResizeLanczos4 with
//     INTER_RESIZE_COEF_SCALE, VResizeLinear<uchar, int, short>, VResizeCubicVec_32s8u,
//     VResizeLanczos4, FixedPtCast)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// 8-bit separable resize. The horizontal pass is exact integer arithmetic, so each destination
/// element is <c>Σ S[tap]·coef</c> over precomputed taps; border taps are resolved into the tap
/// table. With SSSE3 the taps of four elements are gathered by byte shuffles from 16-byte loads
/// and summed with 16-bit multiply-adds.
/// </summary>
internal sealed class ResizeRows8uPlan : IDisposable
{
    public ResizeRows8uPlan(ResizeCoefficients c, int srcWidth, int dstWidth, int cn)
    {
        K = c.Ksize;
        Width = dstWidth * cn;
        int swidth = srcWidth * cn;
        Taps = ArrayPool<int>.Shared.Rent(Width * K);
        Coefs = ArrayPool<short>.Shared.Rent(Width * K);
        BuildTaps(c, swidth, cn);

        int steps = Width / 4;
        int v = K / 2;
        if (Ssse3.IsSupported && steps > 0)
        {
            Bases = ArrayPool<int>.Shared.Rent(steps * v);
            Masks = ArrayPool<Vector128<byte>>.Shared.Rent(steps * v);
            Weights = ArrayPool<Vector128<short>>.Shared.Rent(steps * v);
            Steps = steps;
            BuildShuffles(swidth, v);
        }
    }

    public int K { get; }

    public int Width { get; }

    public int[] Taps { get; }

    public short[] Coefs { get; }

    public int Steps { get; }

    public int[]? Bases { get; }

    public Vector128<byte>[]? Masks { get; }

    public Vector128<short>[]? Weights { get; }

    public void Dispose()
    {
        ArrayPool<int>.Shared.Return(Taps);
        ArrayPool<short>.Shared.Return(Coefs);
        if (Bases is not null)
        {
            ArrayPool<int>.Shared.Return(Bases);
            ArrayPool<Vector128<byte>>.Shared.Return(Masks!);
            ArrayPool<Vector128<short>>.Shared.Return(Weights!);
        }
    }

    private void BuildTaps(ResizeCoefficients c, int swidth, int cn)
    {
        int k = K;
        int xminE = c.XMin * cn, xmaxE = c.XMax * cn;
        int off = k == 2 ? 0 : (k == 4 ? -cn : -3 * cn);
        for (int e = 0; e < Width; e++)
        {
            int sx = c.XOffsets[e];
            int o = e * k;
            if (k == 2)
            {
                if (e < xmaxE)
                {
                    Taps[o] = sx;
                    Taps[o + 1] = sx + cn;
                    Coefs[o] = c.IAlpha[o];
                    Coefs[o + 1] = c.IAlpha[o + 1];
                }
                else
                {
                    Taps[o] = Taps[o + 1] = sx;
                    Coefs[o] = ResizeCoefficients.CoefScale;
                    Coefs[o + 1] = 0;
                }

                continue;
            }

            bool border = e < xminE || e >= xmaxE;
            for (int j = 0; j < k; j++)
            {
                int sxj = sx + off + (j * cn);
                if (border)
                {
                    while (sxj < 0)
                    {
                        sxj += cn;
                    }

                    while (sxj >= swidth)
                    {
                        sxj -= cn;
                    }
                }

                Taps[o + j] = sxj;
                Coefs[o + j] = c.IAlpha[o + j];
            }
        }
    }

    private void BuildShuffles(int swidth, int v)
    {
        int k = K;
        int perVector = 8 / k;
        Span<byte> mask = stackalloc byte[16];
        Span<short> w = stackalloc short[8];
        for (int s = 0; s < Steps; s++)
        {
            bool ok = true;
            for (int i = 0; i < v && ok; i++)
            {
                int e0 = (s * 4) + (i * perVector);
                int lo = int.MaxValue, hi = int.MinValue;
                for (int t = 0; t < 8; t++)
                {
                    int tap = Taps[(e0 * k) + t];
                    lo = Math.Min(lo, tap);
                    hi = Math.Max(hi, tap);
                }

                int baseOffset = Math.Min(lo, swidth - 16);
                if (baseOffset < 0 || hi - baseOffset >= 16)
                {
                    ok = false;
                    break;
                }

                for (int t = 0; t < 8; t++)
                {
                    mask[2 * t] = (byte)(Taps[(e0 * k) + t] - baseOffset);
                    mask[(2 * t) + 1] = 0x80;
                    w[t] = Coefs[(e0 * k) + t];
                }

                Bases![(s * v) + i] = baseOffset;
                Masks![(s * v) + i] = Vector128.Create((ReadOnlySpan<byte>)mask);
                Weights![(s * v) + i] = Vector128.Create((ReadOnlySpan<short>)w);
            }

            if (!ok)
            {
                Bases![s * v] = -1;
            }
        }
    }
}

/// <summary>8-bit rows of the separable resize (work type <c>int</c>, weights scaled by 2048).</summary>
internal readonly struct ResizeRows8u : IResizeRows<byte, int>
{
    private const float VectorScale = 1f / (ResizeCoefficients.CoefScale * ResizeCoefficients.CoefScale);

    private readonly ResizeRows8uPlan _plan;
    private readonly short[] _beta;

    public ResizeRows8u(ResizeRows8uPlan plan, ResizeCoefficients c)
    {
        _plan = plan;
        _beta = c.IBeta;
    }

    public void Horizontal(ref byte src, ref int dst)
    {
        ResizeRows8uPlan p = _plan;
        int k = p.K;
        int width = p.Width;
        ref int taps = ref MemoryMarshal.GetArrayDataReference(p.Taps);
        ref short coefs = ref MemoryMarshal.GetArrayDataReference(p.Coefs);
        int e = 0;
        if (p.Bases is not null && Ssse3.IsSupported)
        {
            int v = k / 2;
            ref int bases = ref MemoryMarshal.GetArrayDataReference(p.Bases);
            ref Vector128<byte> masks = ref MemoryMarshal.GetArrayDataReference(p.Masks!);
            ref Vector128<short> weights = ref MemoryMarshal.GetArrayDataReference(p.Weights!);
            for (int s = 0; s < p.Steps; s++, e += 4)
            {
                int q = s * v;
                if (Unsafe.Add(ref bases, q) < 0)
                {
                    Scalar(ref src, ref dst, ref taps, ref coefs, k, e, e + 4);
                    continue;
                }

                Vector128<int> r = Madd(ref src, ref bases, ref masks, ref weights, q);
                if (k >= 4)
                {
                    r = Ssse3.HorizontalAdd(r, Madd(ref src, ref bases, ref masks, ref weights, q + 1));
                    if (k == 8)
                    {
                        Vector128<int> r2 = Ssse3.HorizontalAdd(
                            Madd(ref src, ref bases, ref masks, ref weights, q + 2),
                            Madd(ref src, ref bases, ref masks, ref weights, q + 3));
                        r = Ssse3.HorizontalAdd(r, r2);
                    }
                }

                r.StoreUnsafe(ref dst, (nuint)e);
            }
        }

        Scalar(ref src, ref dst, ref taps, ref coefs, k, e, width);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Madd(ref byte src, ref int bases, ref Vector128<byte> masks, ref Vector128<short> weights, int q)
    {
        Vector128<byte> raw = Vector128.LoadUnsafe(ref src, (nuint)Unsafe.Add(ref bases, q));
        return Sse2.MultiplyAddAdjacent(Ssse3.Shuffle(raw, Unsafe.Add(ref masks, q)).AsInt16(), Unsafe.Add(ref weights, q));
    }

    private static void Scalar(ref byte src, ref int dst, ref int taps, ref short coefs, int k, int e0, int e1)
    {
        for (int e = e0; e < e1; e++)
        {
            int o = e * k;
            int sum = 0;
            for (int j = 0; j < k; j++)
            {
                sum += Unsafe.Add(ref src, Unsafe.Add(ref taps, o + j)) * Unsafe.Add(ref coefs, o + j);
            }

            Unsafe.Add(ref dst, e) = sum;
        }
    }

    public void Vertical(ref int buffer, ReadOnlySpan<int> rows, int dy, ref byte dst)
    {
        int k = _plan.K;
        ReadOnlySpan<short> b = _beta.AsSpan(dy * k, k);
        int width = _plan.Width;
        if (k == 2)
        {
            VerticalLinear(ref Unsafe.Add(ref buffer, rows[0]), ref Unsafe.Add(ref buffer, rows[1]), b[0], b[1], ref dst, width);
        }
        else if (k == 4)
        {
            VerticalCubic(ref buffer, rows, b, ref dst, width);
        }
        else
        {
            VerticalInteger(ref buffer, rows, b, ref dst, 0, width);
        }
    }

    /// <summary><c>((b0·(S0 &gt;&gt; 4)) &gt;&gt; 16) + ((b1·(S1 &gt;&gt; 4)) &gt;&gt; 16) + 2) &gt;&gt; 2</c>.</summary>
    private static void VerticalLinear(ref int s0, ref int s1, short b0, short b1, ref byte dst, int width)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            var vb0 = new Vector<int>(b0);
            var vb1 = new Vector<int>(b1);
            for (; x <= width - (4 * vc); x += 4 * vc)
            {
                FilterVectors.SaturateToByte(
                    Linear(ref s0, ref s1, vb0, vb1, x),
                    Linear(ref s0, ref s1, vb0, vb1, x + vc),
                    Linear(ref s0, ref s1, vb0, vb1, x + (2 * vc)),
                    Linear(ref s0, ref s1, vb0, vb1, x + (3 * vc))).StoreUnsafe(ref dst, (nuint)x);
            }
        }

        for (; x < width; x++)
        {
            Unsafe.Add(ref dst, x) = (byte)((((b0 * (Unsafe.Add(ref s0, x) >> 4)) >> 16) + ((b1 * (Unsafe.Add(ref s1, x) >> 4)) >> 16) + 2) >> 2);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Linear(ref int s0, ref int s1, Vector<int> b0, Vector<int> b1, int x)
    {
        Vector<int> a = Vector.ShiftRightArithmetic(b0 * Vector.ShiftRightArithmetic(Vector.LoadUnsafe(ref s0, (nuint)x), 4), 16);
        Vector<int> c = Vector.ShiftRightArithmetic(b1 * Vector.ShiftRightArithmetic(Vector.LoadUnsafe(ref s1, (nuint)x), 4), 16);
        return Vector.ShiftRightArithmetic(a + c + new Vector<int>(2), 2);
    }

    /// <summary>
    /// Elements below <c>width / 8 * 8</c> use the 128-bit vector body of
    /// <c>VResizeCubicVec_32s8u</c> (single precision, summed right to left); the rest use the
    /// integer formula.
    /// </summary>
    private static void VerticalCubic(ref int buffer, ReadOnlySpan<int> rows, ReadOnlySpan<short> b, ref byte dst, int width)
    {
        ref int s0 = ref Unsafe.Add(ref buffer, rows[0]);
        ref int s1 = ref Unsafe.Add(ref buffer, rows[1]);
        ref int s2 = ref Unsafe.Add(ref buffer, rows[2]);
        ref int s3 = ref Unsafe.Add(ref buffer, rows[3]);
        float f0 = b[0] * VectorScale, f1 = b[1] * VectorScale, f2 = b[2] * VectorScale, f3 = b[3] * VectorScale;
        int bodyEnd = width / 8 * 8;
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<float>.Count;
            var v0 = new Vector<float>(f0);
            var v1 = new Vector<float>(f1);
            var v2 = new Vector<float>(f2);
            var v3 = new Vector<float>(f3);
            for (; x <= bodyEnd - (4 * vc); x += 4 * vc)
            {
                FilterVectors.SaturateToByte(
                    CubicVec(ref s0, ref s1, ref s2, ref s3, v0, v1, v2, v3, x),
                    CubicVec(ref s0, ref s1, ref s2, ref s3, v0, v1, v2, v3, x + vc),
                    CubicVec(ref s0, ref s1, ref s2, ref s3, v0, v1, v2, v3, x + (2 * vc)),
                    CubicVec(ref s0, ref s1, ref s2, ref s3, v0, v1, v2, v3, x + (3 * vc))).StoreUnsafe(ref dst, (nuint)x);
            }
        }

        for (; x < bodyEnd; x++)
        {
            float v = (Unsafe.Add(ref s0, x) * f0) + ((Unsafe.Add(ref s1, x) * f1) + ((Unsafe.Add(ref s2, x) * f2) + (Unsafe.Add(ref s3, x) * f3)));
            Unsafe.Add(ref dst, x) = (byte)Math.Clamp(ScalarConverter.CvRound(v), 0, 255);
        }

        Span<int> r = stackalloc int[4];
        rows.CopyTo(r);
        VerticalInteger(ref buffer, r, b, ref dst, bodyEnd, width);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> CubicVec(
        ref int s0, ref int s1, ref int s2, ref int s3, Vector<float> b0, Vector<float> b1, Vector<float> b2, Vector<float> b3, int x)
    {
        Vector<float> a3 = Vector.ConvertToSingle(Vector.LoadUnsafe(ref s3, (nuint)x)) * b3;
        Vector<float> a2 = (Vector.ConvertToSingle(Vector.LoadUnsafe(ref s2, (nuint)x)) * b2) + a3;
        Vector<float> a1 = (Vector.ConvertToSingle(Vector.LoadUnsafe(ref s1, (nuint)x)) * b1) + a2;
        Vector<float> a0 = (Vector.ConvertToSingle(Vector.LoadUnsafe(ref s0, (nuint)x)) * b0) + a1;
        return FilterVectors.CvRound(a0);
    }

    /// <summary><c>saturate_cast&lt;uchar&gt;((Σ S_k·b_k + (1 &lt;&lt; 21)) &gt;&gt; 22)</c> with wrapping 32-bit sums.</summary>
    private static void VerticalInteger(ref int buffer, ReadOnlySpan<int> rows, ReadOnlySpan<short> b, ref byte dst, int x0, int width)
    {
        int k = b.Length;
        Span<int> r = stackalloc int[8];
        Span<int> bi = stackalloc int[8];
        rows.CopyTo(r);
        for (int i = 0; i < k; i++)
        {
            bi[i] = b[i];
        }

        int x = x0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            Span<Vector<int>> vb = stackalloc Vector<int>[8];
            for (int i = 0; i < k; i++)
            {
                vb[i] = new Vector<int>(bi[i]);
            }

            for (; x <= width - (4 * vc); x += 4 * vc)
            {
                FilterVectors.SaturateToByte(
                    IntegerVec(ref buffer, r, vb, k, x),
                    IntegerVec(ref buffer, r, vb, k, x + vc),
                    IntegerVec(ref buffer, r, vb, k, x + (2 * vc)),
                    IntegerVec(ref buffer, r, vb, k, x + (3 * vc))).StoreUnsafe(ref dst, (nuint)x);
            }
        }

        for (; x < width; x++)
        {
            int sum = 0;
            for (int i = 0; i < k; i++)
            {
                sum = unchecked(sum + (Unsafe.Add(ref buffer, r[i] + x) * bi[i]));
            }

            Unsafe.Add(ref dst, x) = (byte)Math.Clamp(unchecked(sum + (1 << 21)) >> 22, 0, 255);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> IntegerVec(ref int buffer, Span<int> r, Span<Vector<int>> vb, int k, int x)
    {
        Vector<int> sum = Vector.LoadUnsafe(ref buffer, (nuint)(r[0] + x)) * vb[0];
        for (int i = 1; i < k; i++)
        {
            sum += Vector.LoadUnsafe(ref buffer, (nuint)(r[i] + x)) * vb[i];
        }

        return Vector.ShiftRightArithmetic(sum + new Vector<int>(1 << 21), 22);
    }
}
