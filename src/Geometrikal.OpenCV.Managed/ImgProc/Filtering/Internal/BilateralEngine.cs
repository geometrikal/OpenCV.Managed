// This implementation is ported from the OpenCV bilateral filter.
//
// Original sources:
//   modules/imgproc/src/bilateral_filter.dispatch.cpp (bilateralFilter_8u, bilateralFilter_32f)
//   modules/imgproc/src/bilateral_filter.simd.hpp (BilateralFilter_8u_Invoker,
//     BilateralFilter_32f_Invoker)
//   modules/core/include/opencv2/core/hal/intrin_math.hpp (v_exp_default_32f)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV 5.0 <c>bilateralFilter</c> with the arithmetic of the SSE2 dispatch: the vector
/// formula on the first <c>L</c> columns of each row and the scalar formula on the others, with
/// <c>L = 16·⌊W/16⌋</c> for 8-bit images and <c>L = 8·⌊W/8⌋</c> for float images.
/// </summary>
/// <remarks>
/// The columns <c>[0, L)</c> run in AVX2 blocks when the hardware supports AVX2 and otherwise in
/// a scalar emulation of the same formula. The result does not depend on the vector width or on
/// the thread count.
/// </remarks>
internal static class BilateralEngine
{
    private const double SigmaEpsilon = 1e-6;
    private const int ExpBinsPerChannel = 1 << 12;
    private const double FltEpsilon = 1.1920928955078125e-07;

    /// <summary>Selects the AVX2 blocks for the vector-formula columns; tests clear it to run the scalar emulation.</summary>
    internal static bool VectorBodyEnabled { get; set; } = true;

    private static bool UseVectorBody => Avx2.IsSupported && VectorBodyEnabled;

    /// <summary>Accumulation order of the unrolled 8-bit 1-channel lanes for 13 taps (radius 2).</summary>
    private static readonly int[] Order13 = [0, 12, 1, 2, 3, 9, 10, 11, 4, 5, 6, 7, 8];

    public static Mat<byte> Apply(Mat<byte> src, int d, double sigmaColor, double sigmaSpace, BorderType borderType)
    {
        if (sigmaColor <= SigmaEpsilon || sigmaSpace <= SigmaEpsilon)
        {
            return src.Clone();
        }

        int cn = src.Channels;
        int radius = Radius(d, sigmaSpace);
        float gaussColor = (float)(-0.5 / (sigmaColor * sigmaColor));
        float gaussSpace = (float)(-0.5 / (sigmaSpace * sigmaSpace));
        float[] colorWeight = ColorWeights8u(gaussColor, cn);

        using PooledMat<byte> planes = PadPlanar(src, radius, borderType);
        int stride = src.Width + (2 * radius);
        int planeSize = (src.Height + (2 * radius)) * stride;
        (float[] spaceWeight, int[] spaceOfs) = SpaceTable(radius, stride, gaussSpace, skipCentre: false);
        int maxk = spaceWeight.Length;
        int[] laneOrder = cn == 1 && maxk == 13 ? Order13 : Identity(maxk);

        var dst = new Mat<byte>(src.Width, src.Height, cn, clear: false);
        var job = new Job<byte>(planes.Mat, planeSize, stride, radius, spaceOfs, spaceWeight, colorWeight, [], laneOrder, dst, 0, 0);
        long work = (long)src.Width * cn * maxk;
        if (cn == 1)
        {
            ParallelRows.For(src.Height, work, (y0, y1) => Rows8uC1(job, y0, y1));
        }
        else
        {
            ParallelRows.For(src.Height, work, (y0, y1) => Rows8uC3(job, y0, y1));
        }

        return dst;
    }

    public static Mat<float> Apply(Mat<float> src, int d, double sigmaColor, double sigmaSpace, BorderType borderType)
    {
        if (sigmaColor <= SigmaEpsilon || sigmaSpace <= SigmaEpsilon)
        {
            return src.Clone();
        }

        int cn = src.Channels;
        int radius = Radius(d, sigmaSpace);
        double gaussColor = -0.5 / (sigmaColor * sigmaColor);
        double gaussSpace = -0.5 / (sigmaSpace * sigmaSpace);

        var flat = new Mat<float>(src.DangerousGetArray(), src.Width * cn, src.Height, 1, src.Stride, src.DangerousGetOffset());
        MinMaxLocResult mm = Reductions.MinMaxLoc(flat);
        if (Math.Abs(mm.MinVal - mm.MaxVal) < FltEpsilon)
        {
            return src.Clone();
        }

        float len = (float)(mm.MaxVal - mm.MinVal) * cn;
        int bins = ExpBinsPerChannel * cn;
        float scaleIndex = bins / len;
        float[] lut = ExpLut(bins, scaleIndex, gaussColor);

        using PooledMat<float> planes = PadPlanar(src, radius, borderType);
        int stride = src.Width + (2 * radius);
        int planeSize = (src.Height + (2 * radius)) * stride;
        (float[] spaceWeight, int[] spaceOfs) = SpaceTable(radius, stride, gaussSpace, skipCentre: true);
        int maxk = spaceWeight.Length;

        var dst = new Mat<float>(src.Width, src.Height, cn, clear: false);
        var job = new Job<float>(planes.Mat, planeSize, stride, radius, spaceOfs, spaceWeight, lut, LutPairs(lut), [], dst, scaleIndex, bins);
        long work = (long)src.Width * cn * maxk * 2;
        if (cn == 1)
        {
            ParallelRows.For(src.Height, work, (y0, y1) => Rows32fC1(job, y0, y1));
        }
        else
        {
            ParallelRows.For(src.Height, work, (y0, y1) => Rows32fC3(job, y0, y1));
        }

        return dst;
    }

    private static int Radius(int d, double sigmaSpace)
        => Math.Max(d <= 0 ? ScalarConverter.CvRound(sigmaSpace * 1.5) : d / 2, 1);

    private static int[] Identity(int n)
    {
        var order = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }

        return order;
    }

    // ------------------------------------------------------------- tables

    /// <summary>
    /// 8-bit colour weights (bilateral_filter.dispatch.cpp:194-219): <c>v_exp((i·i)·gcc)</c> in
    /// 4-lane blocks while <c>i &lt; 256·cn − 4</c>, then <c>expf((float)(i·i)·gcc)</c>.
    /// </summary>
    internal static float[] ColorWeights8u(float gaussColor, int cn)
    {
        int n = 256 * cn;
        var weights = new float[n];
        int i = 0;
        for (; i < n - 4; i += 4)
        {
            for (int l = 0; l < 4; l++)
            {
                float fi = i + l;
                weights[i + l] = VExp(fi * fi * gaussColor);
            }
        }

        for (; i < n; i++)
        {
            weights[i] = MathF.Exp((float)(i * i) * gaussColor);
        }

        return weights;
    }

    /// <summary>One lane of <c>v_exp_default_32f</c> with <c>v_fma</c> as a multiply and an add.</summary>
    internal static float VExp(float x)
    {
        float xx = x > -88.3762626647949f ? x : -88.3762626647949f;
        xx = xx < 89f ? xx : 89f;
        float e = (xx * 1.44269504088896341f) + 0.5f;
        int m = (int)MathF.Floor(e);
        float fm = m;
        int pow2 = (m + 0x7f) << 23;
        xx = (fm * -6.93359375E-1f) + xx;
        xx = (fm * 2.12194440E-4f) + xx;
        float x2 = xx * xx;
        float y = (xx * 1.9875691500E-4f) + 1.3981999507E-3f;
        y = (y * xx) + 8.3334519073E-3f;
        y = (y * xx) + 4.1665795894E-2f;
        y = (y * xx) + 1.6666665459E-1f;
        y = (y * xx) + 5.0000001201E-1f;
        y = (y * x2) + xx;
        y += 1f;
        return float.IsNaN(x) ? BitConverter.Int32BitsToSingle(0x7fc00000) : y * BitConverter.Int32BitsToSingle(pow2);
    }

    /// <summary>Float colour LUT (bilateral_filter.dispatch.cpp:300-310); zero after the first zero.</summary>
    private static float[] ExpLut(int bins, float scaleIndex, double gaussColor)
    {
        var lut = new float[bins + 2];
        float last = 1f;
        for (int i = 0; i < bins + 2; i++)
        {
            if (last > 0f)
            {
                double val = (float)i / scaleIndex;
                lut[i] = (float)Math.Exp(val * val * gaussColor);
                last = lut[i];
            }
        }

        return lut;
    }

    /// <summary>
    /// Space weights and offsets in row-major order over the disc <c>√(i² + j²) ≤ radius</c>:
    /// <c>(float)exp(r·r·gsc)</c> with the rounded square root r.
    /// </summary>
    private static (float[] Weights, int[] Offsets) SpaceTable(int radius, int stride, double gaussSpace, bool skipCentre)
    {
        var weights = new List<float>();
        var offsets = new List<int>();
        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                double r = Math.Sqrt(((double)i * i) + ((double)j * j));
                if (r > radius || (skipCentre && i == 0 && j == 0))
                {
                    continue;
                }

                weights.Add((float)Math.Exp(r * r * gaussSpace));
                offsets.Add((i * stride) + j);
            }
        }

        return ([.. weights], [.. offsets]);
    }

    /// <summary>
    /// <c>copyMakeBorder(src, r, r, r, r, borderType)</c> (constant value 0) written as one padded
    /// plane per channel, plane after plane. Rows are filled in parallel.
    /// </summary>
    private static PooledMat<T> PadPlanar<T>(Mat<T> src, int r, BorderType borderType)
        where T : unmanaged
    {
        if (borderType == BorderType.Transparent)
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "BorderType.Transparent is not supported by filters.");
        }

        int w = src.Width, h = src.Height, cn = src.Channels;
        int pw = w + (2 * r), ph = h + (2 * r);
        var colMap = new int[pw];
        for (int x = 0; x < pw; x++)
        {
            colMap[x] = Borders.BorderInterpolate(x - r, w, borderType);
        }

        PooledMat<T> planes = PooledMat<T>.Rent(pw, ph * cn, 1);
        Mat<T> dst = planes.Mat;
        ParallelRows.For(ph, (long)pw * cn, (y0, y1) => PadRows(src, dst, colMap, r, borderType, y0, y1));
        return planes;
    }

    private static void PadRows<T>(Mat<T> src, Mat<T> dst, int[] colMap, int r, BorderType borderType, int y0, int y1)
        where T : unmanaged
    {
        int w = src.Width, h = src.Height, cn = src.Channels;
        int ph = h + (2 * r);
        for (int y = y0; y < y1; y++)
        {
            int sy = Borders.BorderInterpolate(y - r, h, borderType);
            for (int c = 0; c < cn; c++)
            {
                Span<T> d = dst.GetRowSpan((c * ph) + y);
                if (sy < 0)
                {
                    d.Clear();
                    continue;
                }

                ReadOnlySpan<T> s = src.GetReadOnlyRowSpan(sy);
                if (cn == 1)
                {
                    s.CopyTo(d.Slice(r, w));
                }
                else
                {
                    for (int x = 0; x < w; x++)
                    {
                        d[r + x] = s[(x * cn) + c];
                    }
                }

                for (int x = 0; x < r; x++)
                {
                    int sx = colMap[x];
                    d[x] = sx < 0 ? default : s[(sx * cn) + c];
                    sx = colMap[r + w + x];
                    d[r + w + x] = sx < 0 ? default : s[(sx * cn) + c];
                }
            }
        }
    }

    private sealed record Job<T>(
        Mat<T> Planes,
        int PlaneSize,
        int Stride,
        int Radius,
        int[] Ofs,
        float[] Sw,
        float[] Table,
        float[] Pairs,
        int[] LaneOrder,
        Mat<T> Dst,
        float ScaleIndex,
        int Bins)
        where T : unmanaged;

    // ------------------------------------------------------------- 8-bit

    private static unsafe void Rows8uC1(Job<byte> job, int y0, int y1)
    {
        int w = job.Dst.Width;
        int lanes = w & ~15;
        ref byte src = ref MemoryMarshal.GetReference(job.Planes.GetReadOnlySpan());
        int[] ofs = job.Ofs;
        float[] sw = job.Sw;
        int[] order = job.LaneOrder;
        fixed (float* cw = job.Table)
        {
            for (int y = y0; y < y1; y++)
            {
                int centre = ((y + job.Radius) * job.Stride) + job.Radius;
                ref byte d = ref MemoryMarshal.GetReference(job.Dst.GetRowSpan(y));
                int j = 0;
                if (UseVectorBody)
                {
                    for (; j <= lanes - 32; j += 32)
                    {
                        Block8uC1x2(ref src, centre + j, ofs, sw, order, cw, ref Unsafe.Add(ref d, j));
                    }

                    for (; j < lanes; j += 16)
                    {
                        Block8uC1(ref src, centre + j, ofs, sw, order, cw).StoreUnsafe(ref d, (nuint)j);
                    }
                }

                for (; j < lanes; j++)
                {
                    Unsafe.Add(ref d, j) = Pixel8uC1(ref src, centre + j, ofs, sw, cw, order);
                }

                for (; j < w; j++)
                {
                    Unsafe.Add(ref d, j) = Pixel8uC1(ref src, centre + j, ofs, sw, cw, null);
                }
            }
        }
    }

    private static unsafe byte Pixel8uC1(ref byte src, int p, int[] ofs, float[] sw, float* cw, int[]? order)
    {
        int c = Unsafe.Add(ref src, p);
        float wsum = 0f, sum = 0f;
        for (int kk = 0; kk < ofs.Length; kk++)
        {
            int k = order is null ? kk : order[kk];
            int v = Unsafe.Add(ref src, p + ofs[k]);
            float wk = sw[k] * cw[Math.Abs(v - c)];
            wsum += wk;
            sum += v * wk;
        }

        return (byte)ScalarConverter.CvRound(sum / wsum);
    }

    private static unsafe Vector128<byte> Block8uC1(ref byte src, int p, int[] ofs, float[] sw, int[] order, float* cw)
    {
        Vector128<byte> cv = Vector128.LoadUnsafe(ref src, (nuint)p);
        Vector256<int> c0 = Avx2.ConvertToVector256Int32(cv);
        Vector256<int> c1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(cv, 8));
        Vector256<float> ws0 = Vector256<float>.Zero, ws1 = Vector256<float>.Zero;
        Vector256<float> s0 = Vector256<float>.Zero, s1 = Vector256<float>.Zero;
        for (int kk = 0; kk < order.Length; kk++)
        {
            int k = order[kk];
            Vector128<byte> v = Vector128.LoadUnsafe(ref src, (nuint)(p + ofs[k]));
            Vector256<int> v0 = Avx2.ConvertToVector256Int32(v);
            Vector256<int> v1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(v, 8));
            var kw = Vector256.Create(sw[k]);
            Vector256<float> w0 = kw * Gather(cw, Avx2.Abs(v0 - c0).AsInt32());
            Vector256<float> w1 = kw * Gather(cw, Avx2.Abs(v1 - c1).AsInt32());
            ws0 += w0;
            s0 = (Avx.ConvertToVector256Single(v0) * w0) + s0;
            ws1 += w1;
            s1 = (Avx.ConvertToVector256Single(v1) * w1) + s1;
        }

        return PackBytes(Avx.ConvertToVector256Int32(s0 / ws0), Avx.ConvertToVector256Int32(s1 / ws1));
    }

    /// <summary><see cref="Block8uC1"/> for 32 columns, with four independent accumulator chains.</summary>
    private static unsafe void Block8uC1x2(ref byte src, int p, int[] ofs, float[] sw, int[] order, float* cw, ref byte d)
    {
        Load16(ref src, p, out Vector256<int> c0, out Vector256<int> c1);
        Load16(ref src, p + 16, out Vector256<int> c2, out Vector256<int> c3);
        Vector256<float> ws0 = default, ws1 = default, ws2 = default, ws3 = default;
        Vector256<float> s0 = default, s1 = default, s2 = default, s3 = default;
        for (int kk = 0; kk < order.Length; kk++)
        {
            int k = order[kk];
            int q = p + ofs[k];
            Load16(ref src, q, out Vector256<int> v0, out Vector256<int> v1);
            Load16(ref src, q + 16, out Vector256<int> v2, out Vector256<int> v3);
            var kw = Vector256.Create(sw[k]);
            Vector256<float> w0 = kw * Gather(cw, Avx2.Abs(v0 - c0).AsInt32());
            Vector256<float> w1 = kw * Gather(cw, Avx2.Abs(v1 - c1).AsInt32());
            Vector256<float> w2 = kw * Gather(cw, Avx2.Abs(v2 - c2).AsInt32());
            Vector256<float> w3 = kw * Gather(cw, Avx2.Abs(v3 - c3).AsInt32());
            ws0 += w0;
            s0 = (Avx.ConvertToVector256Single(v0) * w0) + s0;
            ws1 += w1;
            s1 = (Avx.ConvertToVector256Single(v1) * w1) + s1;
            ws2 += w2;
            s2 = (Avx.ConvertToVector256Single(v2) * w2) + s2;
            ws3 += w3;
            s3 = (Avx.ConvertToVector256Single(v3) * w3) + s3;
        }

        PackBytes(Avx.ConvertToVector256Int32(s0 / ws0), Avx.ConvertToVector256Int32(s1 / ws1)).StoreUnsafe(ref d);
        PackBytes(Avx.ConvertToVector256Int32(s2 / ws2), Avx.ConvertToVector256Int32(s3 / ws3)).StoreUnsafe(ref d, 16);
    }

    /// <summary>Interleaved <c>(lut[i], lut[i + 1])</c> pairs, so one 64-bit load fetches both table entries.</summary>
    private static float[] LutPairs(float[] lut)
    {
        var pairs = new float[2 * (lut.Length - 1)];
        for (int i = 0; i < lut.Length - 1; i++)
        {
            pairs[2 * i] = lut[i];
            pairs[(2 * i) + 1] = lut[i + 1];
        }

        return pairs;
    }

    /// <summary><c>lut[idx]</c> and <c>lut[idx + 1]</c> for eight indices from the pair table.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void GatherPairs(float* pairs, Vector256<int> idx, out Vector256<float> l0, out Vector256<float> l1)
    {
        double* pp = (double*)pairs;
        Vector128<int> lo = idx.GetLower(), hi = idx.GetUpper();
        Vector256<float> a = Vector256.Create(pp[lo.ToScalar()], pp[Sse41.Extract(lo, 1)], pp[Sse41.Extract(lo, 2)], pp[Sse41.Extract(lo, 3)]).AsSingle();
        Vector256<float> b = Vector256.Create(pp[hi.ToScalar()], pp[Sse41.Extract(hi, 1)], pp[Sse41.Extract(hi, 2)], pp[Sse41.Extract(hi, 3)]).AsSingle();
        l0 = Avx2.Permute4x64(Avx.Shuffle(a, b, 0b10_00_10_00).AsDouble(), 0b11_01_10_00).AsSingle();
        l1 = Avx2.Permute4x64(Avx.Shuffle(a, b, 0b11_01_11_01).AsDouble(), 0b11_01_10_00).AsSingle();
    }

    /// <summary>
    /// Eight table loads; the indices are in range. Scalar loads are faster than
    /// <c>VPGATHERDD</c> on processors with the gather data sampling mitigation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> Gather(float* table, Vector256<int> idx)
    {
        Vector128<int> lo = idx.GetLower(), hi = idx.GetUpper();
        return Vector256.Create(
            Vector128.Create(table[lo.ToScalar()], table[Sse41.Extract(lo, 1)], table[Sse41.Extract(lo, 2)], table[Sse41.Extract(lo, 3)]),
            Vector128.Create(table[hi.ToScalar()], table[Sse41.Extract(hi, 1)], table[Sse41.Extract(hi, 2)], table[Sse41.Extract(hi, 3)]));
    }

    /// <summary>Packs 16 ints in [0, 255] into bytes, in element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> PackBytes(Vector256<int> a, Vector256<int> b)
    {
        Vector256<short> s = Vector256.Narrow(a, b);
        return Vector128.Narrow(s.GetLower().AsUInt16(), s.GetUpper().AsUInt16());
    }

    private static unsafe void Rows8uC3(Job<byte> job, int y0, int y1)
    {
        int w = job.Dst.Width;
        int lanes = w & ~15;
        ref byte src = ref MemoryMarshal.GetReference(job.Planes.GetReadOnlySpan());
        int[] ofs = job.Ofs;
        float[] sw = job.Sw;
        int plane = job.PlaneSize;
        byte[] rowBuffer = System.Buffers.ArrayPool<byte>.Shared.Rent(3 * w);
        try
        {
            fixed (float* cw = job.Table)
            {
                for (int y = y0; y < y1; y++)
                {
                    int centre = ((y + job.Radius) * job.Stride) + job.Radius;
                    ref byte ob = ref rowBuffer[0];
                    int j = 0;
                    if (UseVectorBody)
                    {
                        for (; j < lanes; j += 16)
                        {
                            Block8uC3(ref src, centre + j, plane, ofs, sw, cw, ref ob, j, w);
                        }
                    }

                    for (; j < w; j++)
                    {
                        Pixel8uC3(ref src, centre + j, plane, ofs, sw, cw, ref ob, j, w);
                    }

                    Span<byte> d = job.Dst.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        d[3 * x] = rowBuffer[x];
                        d[(3 * x) + 1] = rowBuffer[w + x];
                        d[(3 * x) + 2] = rowBuffer[(2 * w) + x];
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(rowBuffer);
        }
    }

    private static unsafe void Pixel8uC3(ref byte src, int p, int plane, int[] ofs, float[] sw, float* cw, ref byte o, int j, int w)
    {
        int rb = Unsafe.Add(ref src, p), rg = Unsafe.Add(ref src, p + plane), rr = Unsafe.Add(ref src, p + (2 * plane));
        float wsum = 0f, sumB = 0f, sumG = 0f, sumR = 0f;
        for (int k = 0; k < ofs.Length; k++)
        {
            int q = p + ofs[k];
            int b = Unsafe.Add(ref src, q), g = Unsafe.Add(ref src, q + plane), r = Unsafe.Add(ref src, q + (2 * plane));
            float wk = sw[k] * cw[Math.Abs(b - rb) + Math.Abs(g - rg) + Math.Abs(r - rr)];
            wsum += wk;
            sumB += b * wk;
            sumG += g * wk;
            sumR += r * wk;
        }

        wsum = 1f / wsum;
        Unsafe.Add(ref o, j) = (byte)ScalarConverter.CvRound(sumB * wsum);
        Unsafe.Add(ref o, w + j) = (byte)ScalarConverter.CvRound(sumG * wsum);
        Unsafe.Add(ref o, (2 * w) + j) = (byte)ScalarConverter.CvRound(sumR * wsum);
    }

    private static unsafe void Block8uC3(ref byte src, int p, int plane, int[] ofs, float[] sw, float* cw, ref byte o, int j, int w)
    {
        Load16(ref src, p, out Vector256<int> cb0, out Vector256<int> cb1);
        Load16(ref src, p + plane, out Vector256<int> cg0, out Vector256<int> cg1);
        Load16(ref src, p + (2 * plane), out Vector256<int> cr0, out Vector256<int> cr1);
        Vector256<float> ws0 = default, ws1 = default;
        Vector256<float> sb0 = default, sb1 = default, sg0 = default, sg1 = default, sr0 = default, sr1 = default;
        for (int k = 0; k < ofs.Length; k++)
        {
            int q = p + ofs[k];
            Load16(ref src, q, out Vector256<int> b0, out Vector256<int> b1);
            Load16(ref src, q + plane, out Vector256<int> g0, out Vector256<int> g1);
            Load16(ref src, q + (2 * plane), out Vector256<int> r0, out Vector256<int> r1);
            Vector256<int> d0 = (Avx2.Abs(b0 - cb0) + Avx2.Abs(g0 - cg0) + Avx2.Abs(r0 - cr0)).AsInt32();
            Vector256<int> d1 = (Avx2.Abs(b1 - cb1) + Avx2.Abs(g1 - cg1) + Avx2.Abs(r1 - cr1)).AsInt32();
            var kw = Vector256.Create(sw[k]);
            Vector256<float> w0 = kw * Gather(cw, d0);
            Vector256<float> w1 = kw * Gather(cw, d1);
            ws0 += w0;
            ws1 += w1;
            sb0 = (Avx.ConvertToVector256Single(b0) * w0) + sb0;
            sb1 = (Avx.ConvertToVector256Single(b1) * w1) + sb1;
            sg0 = (Avx.ConvertToVector256Single(g0) * w0) + sg0;
            sg1 = (Avx.ConvertToVector256Single(g1) * w1) + sg1;
            sr0 = (Avx.ConvertToVector256Single(r0) * w0) + sr0;
            sr1 = (Avx.ConvertToVector256Single(r1) * w1) + sr1;
        }

        Vector256<float> one = Vector256.Create(1f);
        Vector256<float> i0 = one / ws0, i1 = one / ws1;
        PackBytes(Avx.ConvertToVector256Int32(i0 * sb0), Avx.ConvertToVector256Int32(i1 * sb1)).StoreUnsafe(ref o, (nuint)j);
        PackBytes(Avx.ConvertToVector256Int32(i0 * sg0), Avx.ConvertToVector256Int32(i1 * sg1)).StoreUnsafe(ref o, (nuint)(w + j));
        PackBytes(Avx.ConvertToVector256Int32(i0 * sr0), Avx.ConvertToVector256Int32(i1 * sr1)).StoreUnsafe(ref o, (nuint)((2 * w) + j));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Load16(ref byte src, int p, out Vector256<int> lo, out Vector256<int> hi)
    {
        Vector128<byte> v = Vector128.LoadUnsafe(ref src, (nuint)p);
        lo = Avx2.ConvertToVector256Int32(v);
        hi = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(v, 8));
    }

    // -------------------------------------------------------------- float

    private static unsafe void Rows32fC1(Job<float> job, int y0, int y1)
    {
        int w = job.Dst.Width;
        int lanes = w & ~7;
        ref float src = ref MemoryMarshal.GetReference(job.Planes.GetReadOnlySpan());
        fixed (float* lut = job.Table)
        fixed (float* pairs = job.Pairs)
        {
            for (int y = y0; y < y1; y++)
            {
                int centre = ((y + job.Radius) * job.Stride) + job.Radius;
                ref float d = ref MemoryMarshal.GetReference(job.Dst.GetRowSpan(y));
                int j = 0;
                if (UseVectorBody)
                {
                    if (Avx512F.IsSupported)
                    {
                        for (; j <= lanes - 32; j += 32)
                        {
                            Block32fC1x512(ref src, centre + j, job, pairs, ref Unsafe.Add(ref d, j));
                        }
                    }

                    for (; j <= lanes - 16; j += 16)
                    {
                        Block32fC1x2(ref src, centre + j, job, pairs, ref Unsafe.Add(ref d, j));
                    }

                    for (; j < lanes; j += 8)
                    {
                        Block32fC1(ref src, centre + j, job, pairs).StoreUnsafe(ref d, (nuint)j);
                    }
                }

                for (; j < lanes; j++)
                {
                    Unsafe.Add(ref d, j) = LanePixel32fC1(ref src, centre + j, job, lut);
                }

                for (; j < w; j++)
                {
                    Unsafe.Add(ref d, j) = TailPixel32fC1(ref src, centre + j, job, lut);
                }
            }
        }
    }

    /// <summary>The lane index rule: <c>min(trunc(alpha), N)</c>; an overflowed truncation maps to N.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int LaneIndex(float alpha, int bins)
    {
        int idx = alpha >= -2147483648f && alpha < 2147483648f ? (int)alpha : int.MinValue;
        idx = Math.Min(idx, bins);
        return idx < 0 ? bins : idx;
    }

    /// <summary>
    /// The scalar index rule <c>cvFloor(min(alpha, N))</c>. An index outside <c>[0, N]</c> occurs
    /// only for infinite input, where OpenCV reads outside its table; it maps to N.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int TailIndex(float alpha, int bins)
    {
        int idx = ScalarConverter.CvFloor(alpha);
        return (uint)idx > (uint)bins ? bins : idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<int> LaneIndex(Vector512<float> alpha, Vector512<int> bins)
    {
        Vector512<int> idx = Vector512.Min(Avx512F.ConvertToVector512Int32WithTruncation(alpha), bins);
        return Vector512.ConditionalSelect(Vector512.LessThan(idx, Vector512<int>.Zero), bins, idx);
    }

    /// <summary>Sixteen <c>(lut[i], lut[i + 1])</c> pairs, as <see cref="GatherPairs(float*, Vector256{int}, out Vector256{float}, out Vector256{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void GatherPairs(float* pairs, Vector512<int> idx, out Vector512<float> l0, out Vector512<float> l1)
    {
        GatherPairs(pairs, idx.GetLower(), out Vector256<float> a0, out Vector256<float> a1);
        GatherPairs(pairs, idx.GetUpper(), out Vector256<float> b0, out Vector256<float> b1);
        l0 = Vector512.Create(a0, b0);
        l1 = Vector512.Create(a1, b1);
    }

    /// <summary><see cref="Block32fC1"/> for 32 columns in 512-bit lanes (same per-lane operations).</summary>
    private static unsafe void Block32fC1x512(ref float src, int p, Job<float> job, float* pairs, ref float d)
    {
        int[] ofs = job.Ofs;
        float[] sw = job.Sw;
        var sindex = Vector512.Create(job.ScaleIndex);
        var bins = Vector512.Create(job.Bins);
        Vector512<float> one = Vector512.Create(1f);
        Vector512<float> absMask = Vector512.Create(0x7fffffff).AsSingle();
        Vector512<float> c0 = Vector512.LoadUnsafe(ref src, (nuint)p);
        Vector512<float> c1 = Vector512.LoadUnsafe(ref src, (nuint)(p + 16));
        Vector512<float> ok0 = Vector512.Equals(c0, c0), ok1 = Vector512.Equals(c1, c1);
        Vector512<float> ws0 = default, s0 = default, ws1 = default, s1 = default;
        for (int k = 0; k < ofs.Length; k++)
        {
            var kw = Vector512.Create(sw[k]);
            int q = p + ofs[k];
            Vector512<float> v0 = Vector512.LoadUnsafe(ref src, (nuint)q);
            Vector512<float> v1 = Vector512.LoadUnsafe(ref src, (nuint)(q + 16));
            Vector512<float> vOk0 = Vector512.Equals(v0, v0), vOk1 = Vector512.Equals(v1, v1);
            Vector512<float> a0 = (((v0 - c0) & absMask) * sindex) & ok0 & vOk0;
            Vector512<float> a1 = (((v1 - c1) & absMask) * sindex) & ok1 & vOk1;
            Vector512<int> i0 = LaneIndex(a0, bins), i1 = LaneIndex(a1, bins);
            a0 -= Avx512F.ConvertToVector512Single(i0);
            a1 -= Avx512F.ConvertToVector512Single(i1);
            GatherPairs(pairs, i0, out Vector512<float> l00, out Vector512<float> l01);
            GatherPairs(pairs, i1, out Vector512<float> l10, out Vector512<float> l11);
            Vector512<float> w0 = (kw * ((l01 * a0) + (l00 * (one - a0)))) & vOk0;
            Vector512<float> w1 = (kw * ((l11 * a1) + (l10 * (one - a1)))) & vOk1;
            ws0 += w0;
            s0 = ((v0 & vOk0) * w0) + s0;
            ws1 += w1;
            s1 = ((v1 & vOk1) * w1) + s1;
        }

        ((s0 + (c0 & ok0)) / (ws0 + (one & ok0))).StoreUnsafe(ref d);
        ((s1 + (c1 & ok1)) / (ws1 + (one & ok1))).StoreUnsafe(ref d, 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> LaneIndex(Vector256<float> alpha, Vector256<int> bins)
    {
        Vector256<int> idx = Avx2.Min(Avx.ConvertToVector256Int32WithTruncation(alpha), bins);
        return Vector256.ConditionalSelect(Vector256.LessThan(idx, Vector256<int>.Zero), bins, idx);
    }

    private static unsafe float LanePixel32fC1(ref float src, int p, Job<float> job, float* lut)
    {
        float c = Unsafe.Add(ref src, p);
        bool cOk = !float.IsNaN(c);
        float wsum = 0f, sum = 0f;
        int[] ofs = job.Ofs;
        for (int k = 0; k < ofs.Length; k++)
        {
            float v = Unsafe.Add(ref src, p + ofs[k]);
            bool vOk = !float.IsNaN(v);
            float alpha = cOk && vOk ? MathF.Abs(v - c) * job.ScaleIndex : 0f;
            int idx = LaneIndex(alpha, job.Bins);
            alpha -= idx;
            float wk = vOk ? job.Sw[k] * ((lut[idx + 1] * alpha) + (lut[idx] * (1f - alpha))) : 0f;
            wsum += wk;
            sum = ((vOk ? v : 0f) * wk) + sum;
        }

        return (sum + (cOk ? c : 0f)) / (wsum + (cOk ? 1f : 0f));
    }

    private static unsafe float TailPixel32fC1(ref float src, int p, Job<float> job, float* lut)
    {
        float c = Unsafe.Add(ref src, p);
        bool cNaN = float.IsNaN(c);
        float wsum = 0f, sum = 0f;
        int[] ofs = job.Ofs;
        float bins = job.Bins;
        for (int k = 0; k < ofs.Length; k++)
        {
            float v = Unsafe.Add(ref src, p + ofs[k]);
            if (float.IsNaN(v))
            {
                continue;
            }

            float a = MathF.Abs(v - c) * job.ScaleIndex;
            float alpha = bins < a ? bins : a;
            float wk;
            if (cNaN)
            {
                wk = job.Sw[k] * 1f;
            }
            else
            {
                int idx = TailIndex(alpha, job.Bins);
                alpha -= idx;
                wk = job.Sw[k] * (lut[idx] + (alpha * (lut[idx + 1] - lut[idx])));
            }

            wsum += wk;
            sum += v * wk;
        }

        return cNaN ? sum / wsum : (sum + c) / (wsum + 1f);
    }

    private static unsafe Vector256<float> Block32fC1(ref float src, int p, Job<float> job, float* pairs)
    {
        int[] ofs = job.Ofs;
        float[] sw = job.Sw;
        var sindex = Vector256.Create(job.ScaleIndex);
        var bins = Vector256.Create(job.Bins);
        Vector256<float> one = Vector256.Create(1f);
        Vector256<float> absMask = Vector256.Create(0x7fffffff).AsSingle();
        Vector256<float> c = Vector256.LoadUnsafe(ref src, (nuint)p);
        Vector256<float> cOk = Avx.CompareOrdered(c, c);
        Vector256<float> ws = default, s = default;
        for (int k = 0; k < ofs.Length; k++)
        {
            Vector256<float> v = Vector256.LoadUnsafe(ref src, (nuint)(p + ofs[k]));
            Vector256<float> vOk = Avx.CompareOrdered(v, v);
            Vector256<float> alpha = (((v - c) & absMask) * sindex) & cOk & vOk;
            Vector256<int> idx = LaneIndex(alpha, bins);
            alpha -= Avx.ConvertToVector256Single(idx);
            GatherPairs(pairs, idx, out Vector256<float> l0, out Vector256<float> l1);
            Vector256<float> wk = (Vector256.Create(sw[k]) * ((l1 * alpha) + (l0 * (one - alpha)))) & vOk;
            ws += wk;
            s = ((v & vOk) * wk) + s;
        }

        return (s + (c & cOk)) / (ws + (one & cOk));
    }

    /// <summary><see cref="Block32fC1"/> for 16 columns, with two independent accumulator chains.</summary>
    private static unsafe void Block32fC1x2(ref float src, int p, Job<float> job, float* pairs, ref float d)
    {
        int[] ofs = job.Ofs;
        float[] sw = job.Sw;
        var sindex = Vector256.Create(job.ScaleIndex);
        var bins = Vector256.Create(job.Bins);
        Vector256<float> one = Vector256.Create(1f);
        Vector256<float> absMask = Vector256.Create(0x7fffffff).AsSingle();
        Vector256<float> c0 = Vector256.LoadUnsafe(ref src, (nuint)p);
        Vector256<float> c1 = Vector256.LoadUnsafe(ref src, (nuint)(p + 8));
        Vector256<float> ok0 = Avx.CompareOrdered(c0, c0), ok1 = Avx.CompareOrdered(c1, c1);
        Vector256<float> ws0 = default, s0 = default, ws1 = default, s1 = default;
        for (int k = 0; k < ofs.Length; k++)
        {
            var kw = Vector256.Create(sw[k]);
            int q = p + ofs[k];
            Vector256<float> v0 = Vector256.LoadUnsafe(ref src, (nuint)q);
            Vector256<float> v1 = Vector256.LoadUnsafe(ref src, (nuint)(q + 8));
            Vector256<float> vOk0 = Avx.CompareOrdered(v0, v0), vOk1 = Avx.CompareOrdered(v1, v1);
            Vector256<float> a0 = (((v0 - c0) & absMask) * sindex) & ok0 & vOk0;
            Vector256<float> a1 = (((v1 - c1) & absMask) * sindex) & ok1 & vOk1;
            Vector256<int> i0 = LaneIndex(a0, bins), i1 = LaneIndex(a1, bins);
            a0 -= Avx.ConvertToVector256Single(i0);
            a1 -= Avx.ConvertToVector256Single(i1);
            GatherPairs(pairs, i0, out Vector256<float> l00, out Vector256<float> l01);
            GatherPairs(pairs, i1, out Vector256<float> l10, out Vector256<float> l11);
            Vector256<float> w0 = (kw * ((l01 * a0) + (l00 * (one - a0)))) & vOk0;
            Vector256<float> w1 = (kw * ((l11 * a1) + (l10 * (one - a1)))) & vOk1;
            ws0 += w0;
            s0 = ((v0 & vOk0) * w0) + s0;
            ws1 += w1;
            s1 = ((v1 & vOk1) * w1) + s1;
        }

        ((s0 + (c0 & ok0)) / (ws0 + (one & ok0))).StoreUnsafe(ref d);
        ((s1 + (c1 & ok1)) / (ws1 + (one & ok1))).StoreUnsafe(ref d, 8);
    }

    private static unsafe void Rows32fC3(Job<float> job, int y0, int y1)
    {
        int w = job.Dst.Width;
        int lanes = w & ~7;
        int plane = job.PlaneSize;
        ref float src = ref MemoryMarshal.GetReference(job.Planes.GetReadOnlySpan());
        float[] rowBuffer = System.Buffers.ArrayPool<float>.Shared.Rent(3 * w);
        try
        {
            fixed (float* lut = job.Table)
            fixed (float* pairs = job.Pairs)
            {
                for (int y = y0; y < y1; y++)
                {
                    int centre = ((y + job.Radius) * job.Stride) + job.Radius;
                    ref float o = ref rowBuffer[0];
                    int j = 0;
                    if (UseVectorBody)
                    {
                        for (; j < lanes; j += 8)
                        {
                            Block32fC3(ref src, centre + j, plane, job, pairs, ref o, j, w);
                        }
                    }

                    for (; j < lanes; j++)
                    {
                        LanePixel32fC3(ref src, centre + j, plane, job, lut, ref o, j, w);
                    }

                    for (; j < w; j++)
                    {
                        TailPixel32fC3(ref src, centre + j, plane, job, lut, ref o, j, w);
                    }

                    Span<float> d = job.Dst.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        d[3 * x] = rowBuffer[x];
                        d[(3 * x) + 1] = rowBuffer[w + x];
                        d[(3 * x) + 2] = rowBuffer[(2 * w) + x];
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(rowBuffer);
        }
    }

    private static unsafe void LanePixel32fC3(ref float src, int p, int plane, Job<float> job, float* lut, ref float o, int j, int w)
    {
        float rb = Unsafe.Add(ref src, p), rg = Unsafe.Add(ref src, p + plane), rr = Unsafe.Add(ref src, p + (2 * plane));
        bool cOk = !float.IsNaN(rb) && !float.IsNaN(rg) && !float.IsNaN(rr);
        float wsum = 0f, sumB = 0f, sumG = 0f, sumR = 0f;
        int[] ofs = job.Ofs;
        for (int k = 0; k < ofs.Length; k++)
        {
            int q = p + ofs[k];
            float b = Unsafe.Add(ref src, q), g = Unsafe.Add(ref src, q + plane), r = Unsafe.Add(ref src, q + (2 * plane));
            bool vOk = !float.IsNaN(b) && !float.IsNaN(g) && !float.IsNaN(r);
            float alpha = cOk && vOk ? (MathF.Abs(b - rb) + MathF.Abs(g - rg) + MathF.Abs(r - rr)) * job.ScaleIndex : 0f;
            int idx = LaneIndex(alpha, job.Bins);
            alpha -= idx;
            float wk = vOk ? job.Sw[k] * ((lut[idx + 1] * alpha) + (lut[idx] * (1f - alpha))) : 0f;
            wsum += wk;
            sumB = ((vOk ? b : 0f) * wk) + sumB;
            sumG = ((vOk ? g : 0f) * wk) + sumG;
            sumR = ((vOk ? r : 0f) * wk) + sumR;
        }

        float inv = 1f / (wsum + (cOk ? 1f : 0f));
        Unsafe.Add(ref o, j) = (sumB + (cOk ? rb : 0f)) * inv;
        Unsafe.Add(ref o, w + j) = (sumG + (cOk ? rg : 0f)) * inv;
        Unsafe.Add(ref o, (2 * w) + j) = (sumR + (cOk ? rr : 0f)) * inv;
    }

    private static unsafe void TailPixel32fC3(ref float src, int p, int plane, Job<float> job, float* lut, ref float o, int j, int w)
    {
        float rb = Unsafe.Add(ref src, p), rg = Unsafe.Add(ref src, p + plane), rr = Unsafe.Add(ref src, p + (2 * plane));
        bool cNaN = float.IsNaN(rb) || float.IsNaN(rg) || float.IsNaN(rr);
        float wsum = 0f, sumB = 0f, sumG = 0f, sumR = 0f;
        int[] ofs = job.Ofs;
        float bins = job.Bins;
        for (int k = 0; k < ofs.Length; k++)
        {
            int q = p + ofs[k];
            float b = Unsafe.Add(ref src, q), g = Unsafe.Add(ref src, q + plane), r = Unsafe.Add(ref src, q + (2 * plane));
            if (float.IsNaN(b) || float.IsNaN(g) || float.IsNaN(r))
            {
                continue;
            }

            float wk;
            if (cNaN)
            {
                wk = job.Sw[k] * 1f;
            }
            else
            {
                float a = (MathF.Abs(b - rb) + MathF.Abs(g - rg) + MathF.Abs(r - rr)) * job.ScaleIndex;
                float alpha = bins < a ? bins : a;
                int idx = TailIndex(alpha, job.Bins);
                alpha -= idx;
                wk = job.Sw[k] * (lut[idx] + (alpha * (lut[idx + 1] - lut[idx])));
            }

            wsum += wk;
            sumB += b * wk;
            sumG += g * wk;
            sumR += r * wk;
        }

        if (cNaN)
        {
            wsum = 1f / wsum;
            Unsafe.Add(ref o, j) = sumB * wsum;
            Unsafe.Add(ref o, w + j) = sumG * wsum;
            Unsafe.Add(ref o, (2 * w) + j) = sumR * wsum;
        }
        else
        {
            wsum = 1f / (wsum + 1f);
            Unsafe.Add(ref o, j) = (sumB + rb) * wsum;
            Unsafe.Add(ref o, w + j) = (sumG + rg) * wsum;
            Unsafe.Add(ref o, (2 * w) + j) = (sumR + rr) * wsum;
        }
    }

    private static unsafe void Block32fC3(ref float src, int p, int plane, Job<float> job, float* pairs, ref float o, int j, int w)
    {
        int[] ofs = job.Ofs;
        float[] sw = job.Sw;
        var sindex = Vector256.Create(job.ScaleIndex);
        var bins = Vector256.Create(job.Bins);
        Vector256<float> one = Vector256.Create(1f);
        Vector256<float> absMask = Vector256.Create(0x7fffffff).AsSingle();
        Vector256<float> cb = Vector256.LoadUnsafe(ref src, (nuint)p);
        Vector256<float> cg = Vector256.LoadUnsafe(ref src, (nuint)(p + plane));
        Vector256<float> cr = Vector256.LoadUnsafe(ref src, (nuint)(p + (2 * plane)));
        Vector256<float> cOk = Avx.CompareOrdered(cb, cb) & Avx.CompareOrdered(cg, cg) & Avx.CompareOrdered(cr, cr);
        Vector256<float> ws = default, sb = default, sg = default, sr = default;
        for (int k = 0; k < ofs.Length; k++)
        {
            int q = p + ofs[k];
            Vector256<float> b = Vector256.LoadUnsafe(ref src, (nuint)q);
            Vector256<float> g = Vector256.LoadUnsafe(ref src, (nuint)(q + plane));
            Vector256<float> r = Vector256.LoadUnsafe(ref src, (nuint)(q + (2 * plane)));
            Vector256<float> vOk = Avx.CompareOrdered(b, b) & Avx.CompareOrdered(g, g) & Avx.CompareOrdered(r, r);
            Vector256<float> dsum = ((b - cb) & absMask) + ((g - cg) & absMask) + ((r - cr) & absMask);
            Vector256<float> alpha = (dsum * sindex) & cOk & vOk;
            Vector256<int> idx = LaneIndex(alpha, bins);
            alpha -= Avx.ConvertToVector256Single(idx);
            GatherPairs(pairs, idx, out Vector256<float> l0, out Vector256<float> l1);
            Vector256<float> wk = (Vector256.Create(sw[k]) * ((l1 * alpha) + (l0 * (one - alpha)))) & vOk;
            ws += wk;
            sb = ((b & vOk) * wk) + sb;
            sg = ((g & vOk) * wk) + sg;
            sr = ((r & vOk) * wk) + sr;
        }

        Vector256<float> inv = one / (ws + (one & cOk));
        ((sb + (cb & cOk)) * inv).StoreUnsafe(ref o, (nuint)j);
        ((sg + (cg & cOk)) * inv).StoreUnsafe(ref o, (nuint)(w + j));
        ((sr + (cr & cOk)) * inv).StoreUnsafe(ref o, (nuint)((2 * w) + j));
    }
}
