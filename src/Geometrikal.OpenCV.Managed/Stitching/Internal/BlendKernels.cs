// This implementation is ported from the OpenCV stitching blenders.
//
// Original source:
//   modules/stitching/src/blenders.cpp (Blender::feed, FeatherBlender::feed, MultiBandBlender::feed,
//   normalizeUsingWeightMap)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Per-row accumulation and normalisation loops of the blenders on 3-channel <c>short</c> images.</summary>
internal static class BlendKernels
{
    public const float WeightEps = 1e-5f;

    /// <summary>C++ <c>static_cast&lt;short&gt;(float)</c> on x64: <c>cvttss2si</c> to 32 bits, then the low 16 bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static short TruncToShort(float v)
        => unchecked((short)(v is >= -2147483648f and < 2147483648f ? (int)v : int.MinValue));

    /// <summary>
    /// <c>v = min((float)((double)v · sharpness), 1)</c> in place, as OpenCV <c>multiply</c> by a
    /// scalar (computed in <c>double</c>) followed by <c>threshold(..., 1, 1, THRESH_TRUNC)</c>.
    /// </summary>
    public static void ScaleAndTruncate(Mat<float> map, float sharpness)
    {
        ParallelRows.For(map.Height, map.Width / Vector<float>.Count + 1, (y0, y1) =>
        {
            double scale = sharpness;
            var vScale = new Vector<double>(scale);
            var one = Vector<float>.One;
            int n = Vector<float>.Count;
            for (int y = y0; y < y1; y++)
            {
                Span<float> r = map.GetRowSpan(y);
                int x = 0;
                for (; x <= r.Length - n; x += n)
                {
                    Vector.Widen(new Vector<float>(r.Slice(x)), out Vector<double> lo, out Vector<double> hi);
                    // Narrow rounds to nearest, as the C++ double to float conversion.
                    Vector<float> v = Vector.Narrow(lo * vScale, hi * vScale);
                    Vector.ConditionalSelect(Vector.GreaterThan(v, one), one, v).CopyTo(r.Slice(x));
                }

                for (; x < r.Length; x++)
                {
                    float v = (float)(r[x] * scale);
                    r[x] = v > 1f ? 1f : v;
                }
            }
        });
    }

    /// <summary>OpenCV <c>Blender::feed</c>: copies masked pixels and ORs the mask.</summary>
    public static void FeedNo(Mat<short> img, Mat<byte> mask, Mat<short> dst, Mat<byte> dstMask, int dx, int dy)
    {
        for (int y = 0; y < img.Height; y++)
        {
            ReadOnlySpan<short> s = img.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> m = mask.GetReadOnlyRowSpan(y);
            Span<short> d = dst.GetRowSpan(dy + y).Slice(dx * 3, img.Width * 3);
            Span<byte> dm = dstMask.GetRowSpan(dy + y).Slice(dx, img.Width);
            for (int x = 0; x < m.Length; x++)
            {
                if (m[x] != 0)
                {
                    d[3 * x] = s[3 * x];
                    d[(3 * x) + 1] = s[(3 * x) + 1];
                    d[(3 * x) + 2] = s[(3 * x) + 2];
                }

                dm[x] |= m[x];
            }
        }
    }

    /// <summary>
    /// <c>dst += (short)(src · w)</c> per channel and <c>dstWeight += w</c>, as the float-weight loops of
    /// <c>FeatherBlender::feed</c> and <c>MultiBandBlender::feed</c>. <paramref name="dst"/> and
    /// <paramref name="dstWeight"/> are views of the size of <paramref name="src"/>.
    /// </summary>
    public static void AccumulateFloat(Mat<short> src, Mat<float> weight, Mat<short> dst, Mat<float> dstWeight)
    {
        int width = src.Width;
        ParallelRows.For(src.Height, width * 4L / Vector<float>.Count + 1, (y0, y1) => AccumulateFloatRows(src, weight, dst, dstWeight, y0, y1));
    }

    private static void AccumulateFloatRows(Mat<short> src, Mat<float> weight, Mat<short> dst, Mat<float> dstWeight, int y0, int y1)
    {
        float[] expanded = ArrayPool<float>.Shared.Rent(src.RowLength);
        try
        {
            for (int y = y0; y < y1; y++)
            {
                ReadOnlySpan<short> s = src.GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> w = weight.GetReadOnlyRowSpan(y);
                Span<short> d = dst.GetRowSpan(y);
                Span<float> dw = dstWeight.GetRowSpan(y);
                Span<float> e = expanded.AsSpan(0, s.Length);
                Expand(w, e, 0f, addEps: false);
                int i = MultiplyAddVector(s, e, d);
                for (; i < s.Length; i++)
                {
                    d[i] = unchecked((short)(d[i] + TruncToShort(s[i] * e[i])));
                }

                AddWeights(w, dw);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(expanded);
        }
    }

    /// <summary>Repeats each weight (plus <paramref name="eps"/> when <paramref name="addEps"/>) for the 3 channels of its pixel.</summary>
    private static void Expand(ReadOnlySpan<float> w, Span<float> e, float eps, bool addEps)
    {
        int x = 0;
        if (Avx2.IsSupported)
        {
            Vector256<int> i0 = Vector256.Create(0, 0, 0, 1, 1, 1, 2, 2);
            Vector256<int> i1 = Vector256.Create(2, 3, 3, 3, 4, 4, 4, 5);
            Vector256<int> i2 = Vector256.Create(5, 5, 6, 6, 6, 7, 7, 7);
            Vector256<float> vEps = Vector256.Create(eps);
            ref float wr = ref MemoryMarshal.GetReference(w);
            ref float er = ref MemoryMarshal.GetReference(e);
            for (; x <= w.Length - 8; x += 8)
            {
                Vector256<float> v = Vector256.LoadUnsafe(ref wr, (nuint)x);
                if (addEps)
                {
                    v += vEps;
                }

                Avx2.PermuteVar8x32(v, i0).StoreUnsafe(ref er, (nuint)(3 * x));
                Avx2.PermuteVar8x32(v, i1).StoreUnsafe(ref er, (nuint)((3 * x) + 8));
                Avx2.PermuteVar8x32(v, i2).StoreUnsafe(ref er, (nuint)((3 * x) + 16));
            }
        }

        for (; x < w.Length; x++)
        {
            float v = addEps ? w[x] + eps : w[x];
            e[3 * x] = v;
            e[(3 * x) + 1] = v;
            e[(3 * x) + 2] = v;
        }
    }

    private static void AddWeights(ReadOnlySpan<float> w, Span<float> dw)
    {
        int n = Vector<float>.Count;
        int x = 0;
        for (; x <= w.Length - n; x += n)
        {
            (new Vector<float>(dw.Slice(x)) + new Vector<float>(w.Slice(x))).CopyTo(dw.Slice(x));
        }

        for (; x < w.Length; x++)
        {
            dw[x] += w[x];
        }
    }

    /// <summary>
    /// <c>d += (short)(int)((float)s · e)</c> in lanes; returns the first element not processed.
    /// The products stay below 2^31 in magnitude (|s| ≤ 32768, 0 ≤ e ≤ 1), where the lane
    /// truncation equals the C++ conversion.
    /// </summary>
    private static int MultiplyAddVector(ReadOnlySpan<short> s, ReadOnlySpan<float> e, Span<short> d)
    {
        int n = Vector<short>.Count;
        int nf = Vector<float>.Count;
        int i = 0;
        for (; i <= s.Length - n; i += n)
        {
            Vector.Widen(new Vector<short>(s.Slice(i)), out Vector<int> s0, out Vector<int> s1);
            Vector<int> p0 = Vector.ConvertToInt32(Vector.ConvertToSingle(s0) * new Vector<float>(e.Slice(i)));
            Vector<int> p1 = Vector.ConvertToInt32(Vector.ConvertToSingle(s1) * new Vector<float>(e.Slice(i + nf)));
            // Narrow keeps the low 16 bits and short lanes add with wrap-around, as the C++ code.
            (new Vector<short>(d.Slice(i)) + Vector.Narrow(p0, p1)).CopyTo(d.Slice(i));
        }

        return i;
    }

    /// <summary>
    /// <c>dst += (short)((src · w) &gt;&gt; 8)</c> per channel and <c>dstWeight += w</c>, as the
    /// <c>CV_16S</c> weight loop of <c>MultiBandBlender::feed</c>.
    /// </summary>
    public static void AccumulateShort(Mat<short> src, Mat<short> weight, Mat<short> dst, Mat<short> dstWeight)
    {
        ParallelRows.For(src.Height, src.Width * 4L, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                ReadOnlySpan<short> s = src.GetReadOnlyRowSpan(y);
                ReadOnlySpan<short> w = weight.GetReadOnlyRowSpan(y);
                Span<short> d = dst.GetRowSpan(y);
                Span<short> dw = dstWeight.GetRowSpan(y);
                for (int x = 0; x < w.Length; x++)
                {
                    int wx = w[x];
                    int i = 3 * x;
                    d[i] = unchecked((short)(d[i] + (short)((s[i] * wx) >> 8)));
                    d[i + 1] = unchecked((short)(d[i + 1] + (short)((s[i + 1] * wx) >> 8)));
                    d[i + 2] = unchecked((short)(d[i + 2] + (short)((s[i + 2] * wx) >> 8)));
                    dw[x] = unchecked((short)(dw[x] + wx));
                }
            }
        });
    }

    /// <summary>OpenCV <c>normalizeUsingWeightMap</c> with a <c>CV_32F</c> weight map.</summary>
    /// <remarks>
    /// The quotients stay below 2^31 in magnitude: each value is a sum of truncated products
    /// <c>v · w</c> with |v| ≤ 32768, so |value / (Σw + eps)| ≤ 32768. The lane truncation then
    /// equals the C++ conversion.
    /// </remarks>
    public static void NormalizeFloat(Mat<float> weight, Mat<short> src)
    {
        ParallelRows.For(src.Height, src.Width * 2L, (y0, y1) => NormalizeFloatRows(weight, src, y0, y1));
    }

    private static void NormalizeFloatRows(Mat<float> weight, Mat<short> src, int y0, int y1)
    {
        float[] expanded = ArrayPool<float>.Shared.Rent(src.RowLength);
        int n = Vector<short>.Count;
        int nf = Vector<float>.Count;
        try
        {
            for (int y = y0; y < y1; y++)
            {
                Span<short> r = src.GetRowSpan(y);
                Span<float> e = expanded.AsSpan(0, r.Length);
                Expand(weight.GetReadOnlyRowSpan(y), e, WeightEps, addEps: true);
                int i = 0;
                for (; i <= r.Length - n; i += n)
                {
                    Vector.Widen(new Vector<short>(r.Slice(i)), out Vector<int> v0, out Vector<int> v1);
                    Vector<int> q0 = Vector.ConvertToInt32(Vector.ConvertToSingle(v0) / new Vector<float>(e.Slice(i)));
                    Vector<int> q1 = Vector.ConvertToInt32(Vector.ConvertToSingle(v1) / new Vector<float>(e.Slice(i + nf)));
                    Vector.Narrow(q0, q1).CopyTo(r.Slice(i));
                }

                for (; i < r.Length; i++)
                {
                    r[i] = TruncToShort(r[i] / e[i]);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(expanded);
        }
    }

    /// <summary>OpenCV <c>normalizeUsingWeightMap</c> with a <c>CV_16S</c> weight map.</summary>
    public static void NormalizeShort(Mat<short> weight, Mat<short> src)
    {
        ParallelRows.For(src.Height, src.Width * 8L, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                Span<short> r = src.GetRowSpan(y);
                ReadOnlySpan<short> w = weight.GetReadOnlyRowSpan(y);
                for (int x = 0; x < w.Length; x++)
                {
                    int div = w[x] + 1;
                    int i = 3 * x;
                    r[i] = unchecked((short)((r[i] << 8) / div));
                    r[i + 1] = unchecked((short)((r[i + 1] << 8) / div));
                    r[i + 2] = unchecked((short)((r[i + 2] << 8) / div));
                }
            }
        });
    }

    /// <summary>Destination mask <c>weight &gt; eps</c> (float) or <c>weight &gt; 0</c> (short), as OpenCV <c>compare(..., WEIGHT_EPS, CMP_GT)</c>.</summary>
    public static Mat<byte> WeightMask<T>(Mat<T> weight)
        where T : unmanaged
    {
        var mask = new Mat<byte>(weight.Width, weight.Height, 1, clear: false);
        ParallelRows.For(weight.Height, weight.Width / 8 + 1, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                Span<byte> m = mask.GetRowSpan(y);
                if (typeof(T) == typeof(float))
                {
                    ReadOnlySpan<float> w = MemoryMarshal.Cast<T, float>(weight.GetReadOnlyRowSpan(y));
                    int n = Vector<float>.Count;
                    int x = 0;
                    if (Vector256.IsHardwareAccelerated && n == 8)
                    {
                        var eps = Vector256.Create(WeightEps);
                        ref float wr = ref MemoryMarshal.GetReference(w);
                        ref byte mr = ref MemoryMarshal.GetReference(m);
                        for (; x <= m.Length - 8; x += 8)
                        {
                            Vector256<int> gt = Vector256.GreaterThan(Vector256.LoadUnsafe(ref wr, (nuint)x), eps).AsInt32();
                            Vector128<short> s16 = Vector128.Narrow(gt.GetLower(), gt.GetUpper());
                            Vector128<sbyte> s8 = Vector128.Narrow(s16, s16);
                            Unsafe.WriteUnaligned(ref Unsafe.Add(ref mr, x), s8.AsUInt64().ToScalar());
                        }
                    }

                    for (; x < m.Length; x++)
                    {
                        m[x] = w[x] > WeightEps ? (byte)255 : (byte)0;
                    }
                }
                else
                {
                    ReadOnlySpan<short> w = MemoryMarshal.Cast<T, short>(weight.GetReadOnlyRowSpan(y));
                    for (int x = 0; x < m.Length; x++)
                    {
                        m[x] = w[x] > 0 ? (byte)255 : (byte)0;
                    }
                }
            }
        });

        return mask;
    }

    /// <summary>OpenCV <c>Blender::blend</c>: zeroes the pixels outside the mask.</summary>
    public static void ZeroOutsideMask(Mat<short> dst, Mat<byte> mask)
    {
        ParallelRows.For(dst.Height, dst.Width / 8 + 1, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                Span<short> d = dst.GetRowSpan(y);
                ReadOnlySpan<byte> m = mask.GetReadOnlyRowSpan(y);
                int x = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    ref byte mr = ref MemoryMarshal.GetReference(m);
                    for (; x <= m.Length - 32; x += 32)
                    {
                        if (Vector256.EqualsAll(Vector256.LoadUnsafe(ref mr, (nuint)x), Vector256<byte>.AllBitsSet))
                        {
                            continue;
                        }

                        ZeroRange(d, m, x, x + 32);
                    }
                }

                ZeroRange(d, m, x, m.Length);
            }
        });
    }

    private static void ZeroRange(Span<short> d, ReadOnlySpan<byte> m, int x0, int x1)
    {
        for (int x = x0; x < x1; x++)
        {
            if (m[x] == 0)
            {
                d[3 * x] = 0;
                d[(3 * x) + 1] = 0;
                d[(3 * x) + 2] = 0;
            }
        }
    }
}
