// This implementation is ported from the OpenCV SIFT descriptor.
//
// Original sources:
//   modules/features/src/sift.dispatch.cpp (unpackOctave, calcDescriptorsComputer)
//   modules/features/src/sift.simd.hpp (calcSIFTDescriptor)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>SIFT descriptor computation of OpenCV <c>calcSIFTDescriptor</c>.</summary>
internal static class SiftDescriptor
{
    /// <summary>Descriptor length: 4 × 4 cells × 8 orientation bins.</summary>
    public const int Length = SiftConstants.DescrWidth * SiftConstants.DescrWidth * SiftConstants.DescrHistBins;

    /// <summary>OpenCV <c>unpackOctave</c>.</summary>
    public static void UnpackOctave(KeyPoint kpt, out int octave, out int layer, out float scale)
    {
        octave = kpt.Octave & 255;
        layer = (kpt.Octave >> 8) & 255;
        octave = octave < 128 ? octave : (-128 | octave);
        scale = octave >= 0 ? 1f / (1 << octave) : (float)(1 << -octave);
    }

    /// <summary>
    /// OpenCV <c>calcDescriptors</c>: one descriptor row per keypoint, as values rounded to
    /// <c>[0, 255]</c>. <paramref name="floatDst"/> or <paramref name="byteDst"/> receives the rows.
    /// </summary>
    public static void Compute(
        Mat<float>[] gpyr,
        IReadOnlyList<KeyPoint> keypoints,
        int nOctaveLayers,
        int firstOctave,
        Mat<float>? floatDst,
        Mat<byte>? byteDst)
    {
        ParallelRows.For(keypoints.Count, 1 << 14, (start, end) =>
        {
            using var scratch = new Scratch();
            for (int i = start; i < end; i++)
            {
                KeyPoint kpt = keypoints[i];
                UnpackOctave(kpt, out int octave, out int layer, out float scale);
                if (!(octave >= firstOctave && layer <= nOctaveLayers + 2))
                {
                    throw new ArgumentException("A keypoint octave or layer is outside the pyramid.", nameof(keypoints));
                }

                float size = kpt.Size * scale;
                float ptx = kpt.Pt.X * scale;
                float pty = kpt.Pt.Y * scale;
                Mat<float> img = gpyr[((octave - firstOctave) * (nOctaveLayers + 3)) + layer];

                float angle = 360f - kpt.Angle;
                if (MathF.Abs(angle - 360f) < SiftConstants.FltEpsilon)
                {
                    angle = 0f;
                }

                Span<float> raw = scratch.RawDst;
                CalcRaw(img, ptx, pty, angle, size * 0.5f, scratch);
                if (floatDst is not null)
                {
                    Finish(raw, floatDst.GetRowSpan(i), default);
                }
                else
                {
                    Finish(raw, default, byteDst!.GetRowSpan(i));
                }
            }
        });
    }

    /// <summary>Per-thread buffers of <see cref="CalcRaw"/>. X, Y, Ori, W, RBin and CBin are pooled arrays.</summary>
    private sealed class Scratch : IDisposable
    {
        public float[] X = ArrayPool<float>.Shared.Rent(1024);
        public float[] Y = ArrayPool<float>.Shared.Rent(1024);
        public float[] Ori = ArrayPool<float>.Shared.Rent(1024);
        public float[] W = ArrayPool<float>.Shared.Rent(1024);
        public float[] RBin = ArrayPool<float>.Shared.Rent(1024);
        public float[] CBin = ArrayPool<float>.Shared.Rent(1024);
        public float[] Hist = new float[(SiftConstants.DescrWidth + 2) * (SiftConstants.DescrWidth + 2) * (SiftConstants.DescrHistBins + 2)];
        public float[] RawDst = new float[Length];

        public void Ensure(int len)
        {
            if (X.Length < len)
            {
                Dispose();
                X = ArrayPool<float>.Shared.Rent(len);
                Y = ArrayPool<float>.Shared.Rent(len);
                Ori = ArrayPool<float>.Shared.Rent(len);
                W = ArrayPool<float>.Shared.Rent(len);
                RBin = ArrayPool<float>.Shared.Rent(len);
                CBin = ArrayPool<float>.Shared.Rent(len);
            }
        }

        public void Dispose()
        {
            ArrayPool<float>.Shared.Return(X);
            ArrayPool<float>.Shared.Return(Y);
            ArrayPool<float>.Shared.Return(Ori);
            ArrayPool<float>.Shared.Return(W);
            ArrayPool<float>.Shared.Return(RBin);
            ArrayPool<float>.Shared.Return(CBin);
        }
    }

    /// <summary>Histogram part of OpenCV <c>calcSIFTDescriptor</c>; leaves the unnormalised descriptor in <c>RawDst</c>.</summary>
    private static void CalcRaw(Mat<float> img, float ptfX, float ptfY, float ori, float scl, Scratch s)
    {
        const int d = SiftConstants.DescrWidth;
        const int n = SiftConstants.DescrHistBins;
        int ptX = ScalarConverter.CvRound(ptfX);
        int ptY = ScalarConverter.CvRound(ptfY);
        float cosT = MathF.Cos(ori * (float)(Math.PI / 180));
        float sinT = MathF.Sin(ori * (float)(Math.PI / 180));
        const float binsPerRad = n / 360f;
        const float expScale = -1f / (d * d * 0.5f);
        float histWidth = SiftConstants.DescrSclFctr * scl;
        int radius = ScalarConverter.CvRound(histWidth * 1.4142135623730951f * (d + 1) * 0.5f);
        radius = Math.Min(radius, (int)Math.Sqrt(((double)img.Width * img.Width) + ((double)img.Height * img.Height)));
        cosT /= histWidth;
        sinT /= histWidth;

        int len = ((radius * 2) + 1) * ((radius * 2) + 1);
        int rows = img.Height;
        int cols = img.Width;
        s.Ensure(len);
        float[] xs = s.X;
        float[] ys = s.Y;
        float[] w = s.W;
        float[] rb = s.RBin;
        float[] cb = s.CBin;
        float[] ori2 = s.Ori;
        float[] hist = s.Hist;
        Array.Clear(hist);

        float[] data = img.DangerousGetArray();
        int off = img.DangerousGetOffset();
        int stride = img.Stride;
        int k = 0;
        int colLo = Math.Max(-radius, 1 - ptX);
        int colHi = Math.Min(radius, cols - 2 - ptX);
        int rowLo = Math.Max(-radius, 1 - ptY);
        int rowHi = Math.Min(radius, rows - 2 - ptY);
        for (int i = rowLo; i <= rowHi; i++)
        {
            int r = ptY + i;
            int rowBase = off + (r * stride) + ptX;

            // A superset of the columns inside the rotated d x d window; the exact test follows.
            int jLo = colLo, jHi = colHi;
            NarrowRange(sinT, i * cosT, ref jLo, ref jHi);
            NarrowRange(cosT, -(i * sinT), ref jLo, ref jHi);
            int j = jLo;
            if (Vector.IsHardwareAccelerated)
            {
                k = SampleRowVector(data, rowBase, stride, i, ref j, jHi, cosT, sinT, s, k);
            }

            for (; j <= jHi; j++)
            {
                float cRot = (j * cosT) - (i * sinT);
                float rRot = (j * sinT) + (i * cosT);
                float rbin = rRot + (d / 2) - 0.5f;
                float cbin = cRot + (d / 2) - 0.5f;

                if (rbin > -1 && rbin < d && cbin > -1 && cbin < d)
                {
                    int p = rowBase + j;
                    xs[k] = data[p + 1] - data[p - 1];
                    ys[k] = data[p - stride] - data[p + stride];
                    rb[k] = rbin;
                    cb[k] = cbin;
                    w[k] = ((cRot * cRot) + (rRot * rRot)) * expScale;
                    k++;
                }
            }
        }

        int lenLeft = k;
        FastMath.FastAtan32fDegrees(ys.AsSpan(0, lenLeft), xs.AsSpan(0, lenLeft), ori2.AsSpan(0, lenLeft));
        float[] mag = ys;
        FastMath.Magnitude32f(xs.AsSpan(0, lenLeft), ys.AsSpan(0, lenLeft), mag.AsSpan(0, lenLeft));
        FastMath.Exp32f(w.AsSpan(0, lenLeft), w.AsSpan(0, lenLeft));

        ref float histRef = ref MemoryMarshal.GetArrayDataReference(hist);
        k = 0;
        if (Vector.IsHardwareAccelerated)
        {
            k = AccumulateVector(s, lenLeft, ori, ref histRef);
        }

        for (; k < lenLeft; k++)
        {
            float rbin = rb[k];
            float cbin = cb[k];
            float obin = (ori2[k] - ori) * binsPerRad;
            float m = mag[k] * w[k];

            int r0 = (int)MathF.Floor(rbin);
            int c0 = (int)MathF.Floor(cbin);
            int o0 = (int)MathF.Floor(obin);
            rbin -= r0;
            cbin -= c0;
            obin -= o0;

            if (o0 < 0)
            {
                o0 += n;
            }

            if (o0 >= n)
            {
                o0 -= n;
            }

            float vR1 = m * rbin, vR0 = m - vR1;
            float vRc11 = vR1 * cbin, vRc10 = vR1 - vRc11;
            float vRc01 = vR0 * cbin, vRc00 = vR0 - vRc01;
            float vRco111 = vRc11 * obin, vRco110 = vRc11 - vRco111;
            float vRco101 = vRc10 * obin, vRco100 = vRc10 - vRco101;
            float vRco011 = vRc01 * obin, vRco010 = vRc01 - vRco011;
            float vRco001 = vRc00 * obin, vRco000 = vRc00 - vRco001;

            int idx = ((((r0 + 1) * (d + 2)) + c0 + 1) * (n + 2)) + o0;
            // idx is in [0, 288) because r0, c0 are in [-1, 3] and o0 is in [0, n).
            ref float h = ref Unsafe.Add(ref histRef, idx);
            h += vRco000;
            Unsafe.Add(ref h, 1) += vRco001;
            Unsafe.Add(ref h, n + 2) += vRco010;
            Unsafe.Add(ref h, n + 3) += vRco011;
            Unsafe.Add(ref h, (d + 2) * (n + 2)) += vRco100;
            Unsafe.Add(ref h, ((d + 2) * (n + 2)) + 1) += vRco101;
            Unsafe.Add(ref h, (d + 3) * (n + 2)) += vRco110;
            Unsafe.Add(ref h, ((d + 3) * (n + 2)) + 1) += vRco111;
        }

        float[] rawDst = s.RawDst;
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                int idx = ((((i + 1) * (d + 2)) + (j + 1)) * (n + 2));
                hist[idx] += hist[idx + n];
                hist[idx + 1] += hist[idx + n + 1];
                for (k = 0; k < n; k++)
                {
                    rawDst[(((i * d) + j) * n) + k] = hist[idx + k];
                }
            }
        }
    }

    private static readonly Vector<float> Iota = CreateIota();

    private static Vector<float> CreateIota()
    {
        Span<float> v = stackalloc float[Vector<float>.Count];
        for (int i = 0; i < v.Length; i++)
        {
            v[i] = i;
        }

        return new Vector<float>(v);
    }

    /// <summary>
    /// Vector form of the sample loop for columns <c>j</c> of one row; each lane uses the scalar
    /// operation order. Returns the new sample count and advances <paramref name="j"/>.
    /// </summary>
    private static int SampleRowVector(float[] data, int rowBase, int stride, int i, ref int j, int jHi, float cosT, float sinT, Scratch s, int k)
    {
        const float expScale = -1f / (SiftConstants.DescrWidth * SiftConstants.DescrWidth * 0.5f);
        int vc = Vector<float>.Count;
        var vCos = new Vector<float>(cosT);
        var vSin = new Vector<float>(sinT);
        var iSin = new Vector<float>(i * sinT);
        var iCos = new Vector<float>(i * cosT);
        var half = new Vector<float>(SiftConstants.DescrWidth / 2);
        var lo = new Vector<float>(-1f);
        var hi = new Vector<float>(SiftConstants.DescrWidth);
        float[] xs = s.X, ys = s.Y, rb = s.RBin, cb = s.CBin, w = s.W;
        for (; j + vc - 1 <= jHi; j += vc)
        {
            Vector<float> jf = new Vector<float>(j) + Iota;
            Vector<float> cRot = (jf * vCos) - iSin;
            Vector<float> rRot = (jf * vSin) + iCos;
            Vector<float> rbin = (rRot + half) - new Vector<float>(0.5f);
            Vector<float> cbin = (cRot + half) - new Vector<float>(0.5f);
            Vector<int> inside = Vector.GreaterThan(rbin, lo) & Vector.LessThan(rbin, hi) &
                                 Vector.GreaterThan(cbin, lo) & Vector.LessThan(cbin, hi);
            if (inside == Vector<int>.Zero)
            {
                continue;
            }

            int p = rowBase + j;
            Vector<float> dx = new Vector<float>(data, p + 1) - new Vector<float>(data, p - 1);
            Vector<float> dy = new Vector<float>(data, p - stride) - new Vector<float>(data, p + stride);
            Vector<float> wv = ((cRot * cRot) + (rRot * rRot)) * new Vector<float>(expScale);
            uint bits = SiftExtrema.MaskBits(inside);
            while (bits != 0)
            {
                int l = BitOperations.TrailingZeroCount(bits);
                bits &= bits - 1;
                xs[k] = dx[l];
                ys[k] = dy[l];
                rb[k] = rbin[l];
                cb[k] = cbin[l];
                w[k] = wv[l];
                k++;
            }
        }

        return k;
    }

    /// <summary>
    /// Vector form of the trilinear histogram update, as the OpenCV vector loop: the bins and
    /// weights are computed in lanes, then the lanes update the histogram in sample order.
    /// Returns the number of samples done.
    /// </summary>
    private static int AccumulateVector(Scratch s, int lenLeft, float ori, ref float histRef)
    {
        const int d = SiftConstants.DescrWidth;
        const int n = SiftConstants.DescrHistBins;
        int vc = Vector<float>.Count;
        var vOri = new Vector<float>(ori);
        var vBins = new Vector<float>(n / 360f);
        var vn = new Vector<int>(n);
        float[] rb = s.RBin, cb = s.CBin, oriArr = s.Ori, mag = s.Y, w = s.W;
        Span<int> idxBuf = stackalloc int[vc];
        Span<float> wb = stackalloc float[8 * vc];
        int k = 0;
        for (; k + vc <= lenLeft; k += vc)
        {
            var rbin = new Vector<float>(rb, k);
            var cbin = new Vector<float>(cb, k);
            Vector<float> obin = (new Vector<float>(oriArr, k) - vOri) * vBins;
            Vector<float> m = new Vector<float>(mag, k) * new Vector<float>(w, k);

            Vector<int> r0 = Vector.ConvertToInt32(Vector.Floor(rbin));
            Vector<int> c0 = Vector.ConvertToInt32(Vector.Floor(cbin));
            Vector<int> o0 = Vector.ConvertToInt32(Vector.Floor(obin));
            rbin -= Vector.ConvertToSingle(r0);
            cbin -= Vector.ConvertToSingle(c0);
            obin -= Vector.ConvertToSingle(o0);
            o0 = Vector.ConditionalSelect(Vector.LessThan(o0, Vector<int>.Zero), o0 + vn, o0);
            o0 = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(o0, vn), o0 - vn, o0);

            Vector<float> vR1 = m * rbin, vR0 = m - vR1;
            Vector<float> vRc11 = vR1 * cbin, vRc10 = vR1 - vRc11;
            Vector<float> vRc01 = vR0 * cbin, vRc00 = vR0 - vRc01;
            Vector<float> vRco111 = vRc11 * obin;
            Vector<float> vRco101 = vRc10 * obin;
            Vector<float> vRco011 = vRc01 * obin;
            Vector<float> vRco001 = vRc00 * obin;
            (vRc00 - vRco001).CopyTo(wb);
            vRco001.CopyTo(wb[vc..]);
            (vRc01 - vRco011).CopyTo(wb[(2 * vc)..]);
            vRco011.CopyTo(wb[(3 * vc)..]);
            (vRc10 - vRco101).CopyTo(wb[(4 * vc)..]);
            vRco101.CopyTo(wb[(5 * vc)..]);
            (vRc11 - vRco111).CopyTo(wb[(6 * vc)..]);
            vRco111.CopyTo(wb[(7 * vc)..]);
            Vector<int> idx = (((((r0 + Vector<int>.One) * new Vector<int>(d + 2)) + c0 + Vector<int>.One) * new Vector<int>(n + 2)) + o0);
            idx.CopyTo(idxBuf);

            for (int l = 0; l < vc; l++)
            {
                ref float h = ref Unsafe.Add(ref histRef, idxBuf[l]);
                h += wb[l];
                Unsafe.Add(ref h, 1) += wb[vc + l];
                Unsafe.Add(ref h, n + 2) += wb[(2 * vc) + l];
                Unsafe.Add(ref h, n + 3) += wb[(3 * vc) + l];
                Unsafe.Add(ref h, (d + 2) * (n + 2)) += wb[(4 * vc) + l];
                Unsafe.Add(ref h, ((d + 2) * (n + 2)) + 1) += wb[(5 * vc) + l];
                Unsafe.Add(ref h, (d + 3) * (n + 2)) += wb[(6 * vc) + l];
                Unsafe.Add(ref h, ((d + 3) * (n + 2)) + 1) += wb[(7 * vc) + l];
            }
        }

        return k;
    }

    /// <summary>
    /// Narrows <c>[lo, hi]</c> to the integers <c>j</c> with <c>a*j + b</c> in <c>(-2.5, 2.5)</c>
    /// (bin coordinates in <c>(-1, d)</c>), with a margin of two steps for rounding.
    /// </summary>
    private static void NarrowRange(float a, float b, ref int lo, ref int hi)
    {
        if (MathF.Abs(a) < 1e-3f)
        {
            return;
        }

        double t0 = (-2.5 - 0.01 - b) / a;
        double t1 = (2.5 + 0.01 - b) / a;
        if (t0 > t1)
        {
            (t0, t1) = (t1, t0);
        }

        lo = Math.Max(lo, (int)Math.Max(Math.Floor(t0) - 2, int.MinValue / 2));
        hi = Math.Min(hi, (int)Math.Min(Math.Ceiling(t1) + 2, int.MaxValue / 2));
    }

    /// <summary>
    /// Normalisation and conversion of OpenCV <c>calcSIFTDescriptor</c>. The first squared norm
    /// uses four lane accumulators and the SSE <c>v_reduce_sum</c> order
    /// <c>(a0 + a2) + (a1 + a3)</c>; the second squared norm is sequential.
    /// </summary>
    private static void Finish(Span<float> rawDst, Span<float> floatDst, Span<byte> byteDst)
    {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int k = 0; k < Length; k += 4)
        {
            a0 += rawDst[k] * rawDst[k];
            a1 += rawDst[k + 1] * rawDst[k + 1];
            a2 += rawDst[k + 2] * rawDst[k + 2];
            a3 += rawDst[k + 3] * rawDst[k + 3];
        }

        float nrm2 = (a0 + a2) + (a1 + a3);
        float thr = MathF.Sqrt(nrm2) * SiftConstants.DescrMagThr;

        nrm2 = 0;
        for (int i = 0; i < Length; i++)
        {
            float val = StdMin(rawDst[i], thr);
            rawDst[i] = val;
            nrm2 += val * val;
        }

        nrm2 = SiftConstants.IntDescrFctr / StdMax(MathF.Sqrt(nrm2), SiftConstants.FltEpsilon);

        if (!floatDst.IsEmpty)
        {
            for (int k = 0; k < Length; k++)
            {
                floatDst[k] = ToByte(rawDst[k] * nrm2);
            }
        }
        else
        {
            for (int k = 0; k < Length; k++)
            {
                byteDst[k] = ToByte(rawDst[k] * nrm2);
            }
        }
    }

    /// <summary>OpenCV <c>saturate_cast&lt;uchar&gt;(float)</c>.</summary>
    private static byte ToByte(float v) => (byte)Math.Clamp(ScalarConverter.CvRound(v), 0, 255);

    private static float StdMin(float a, float b) => b < a ? b : a;

    private static float StdMax(float a, float b) => a < b ? b : a;
}
