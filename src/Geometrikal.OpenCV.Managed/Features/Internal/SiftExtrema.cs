// This implementation is ported from the OpenCV SIFT detector.
//
// Original sources:
//   modules/features/src/sift.dispatch.cpp (SIFT_Impl::findScaleSpaceExtrema)
//   modules/features/src/sift.simd.hpp (findScaleSpaceExtremaT, adjustLocalExtrema,
//   calcOrientationHist)
//   modules/core/include/opencv2/core/operations.hpp (Matx_FastSolveOp<_Tp, 3, 3, 1>)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Scale-space extrema detection of OpenCV SIFT.</summary>
internal static class SiftExtrema
{
    private readonly struct Params
    {
        public Params(Mat<float>[] gauss, Mat<float>[] dog, int nOctaveLayers, float contrastThreshold, float edgeThreshold, float sigma, int threshold)
        {
            Gauss = gauss;
            Dog = dog;
            NOctaveLayers = nOctaveLayers;
            ContrastThreshold = contrastThreshold;
            EdgeThreshold = edgeThreshold;
            Sigma = sigma;
            Threshold = threshold;
        }

        public Mat<float>[] Gauss { get; }

        public Mat<float>[] Dog { get; }

        public int NOctaveLayers { get; }

        public float ContrastThreshold { get; }

        public float EdgeThreshold { get; }

        public float Sigma { get; }

        public int Threshold { get; }
    }

    /// <summary>
    /// OpenCV <c>SIFT_Impl::findScaleSpaceExtrema</c>. Appends the keypoints to
    /// <paramref name="keypoints"/> in the order of a serial scan: octave, layer, row, column,
    /// then orientation peak.
    /// </summary>
    public static void Find(Mat<float>[] gauss, Mat<float>[] dog, int nOctaveLayers, double contrastThreshold, double edgeThreshold, double sigma, List<KeyPoint> keypoints)
    {
        int nOctaves = gauss.Length / (nOctaveLayers + 3);
        int threshold = ScalarConverter.CvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * 1);
        var p = new Params(gauss, dog, nOctaveLayers, (float)contrastThreshold, (float)edgeThreshold, (float)sigma, threshold);

        // Scan unit u is row ImgBorder + u − segmentStart[s] of segment s = o · nOctaveLayers + (i − 1).
        int segments = nOctaves * nOctaveLayers;
        var segmentStart = new int[segments + 1];
        for (int s = 0; s < segments; s++)
        {
            Mat<float> img = dog[((s / nOctaveLayers) * (nOctaveLayers + 2)) + (s % nOctaveLayers) + 1];
            segmentStart[s + 1] = segmentStart[s] + Math.Max(img.Height - (2 * SiftConstants.ImgBorder), 0);
        }

        int units = segmentStart[segments];
        if (units == 0)
        {
            return;
        }

        // The first stripe appends to keypoints. Each other stripe stores its keypoints at the index
        // of its first unit, so the serial order does not depend on the thread count.
        List<KeyPoint>?[] perStripe = ArrayPool<List<KeyPoint>?>.Shared.Rent(units);
        try
        {
            Array.Clear(perStripe, 0, units);
            ParallelRows.For(units, 4096, (start, end) =>
            {
                List<KeyPoint>? result = start == 0 ? keypoints : null;
                ScanRows(p, segmentStart, start, end, ref result);
                perStripe[start] = result;
            });

            int count = keypoints.Count;
            for (int u = 1; u < units; u++)
            {
                count += perStripe[u]?.Count ?? 0;
            }

            keypoints.EnsureCapacity(count);
            for (int u = 1; u < units; u++)
            {
                if (perStripe[u] is { } stripe)
                {
                    keypoints.AddRange(stripe);
                }
            }
        }
        finally
        {
            ArrayPool<List<KeyPoint>?>.Shared.Return(perStripe, clearArray: true);
        }
    }

    private static void ScanRows(Params p, int[] segmentStart, int start, int end, ref List<KeyPoint>? result)
    {
        Span<float> hist = stackalloc float[SiftConstants.OriHistBins];
        using var scratch = new OrientationScratch();
        int s = 0;
        for (int u = start; u < end; u++)
        {
            while (u >= segmentStart[s + 1])
            {
                s++;
            }

            int o = s / p.NOctaveLayers;
            int i = (s % p.NOctaveLayers) + 1;
            ScanRow(p, o, i, SiftConstants.ImgBorder + u - segmentStart[s], hist, scratch, ref result);
        }
    }

    private static void ScanRow(Params p, int o, int layer0, int r, Span<float> hist, OrientationScratch scratch, ref List<KeyPoint>? result)
    {
        int nl = p.NOctaveLayers;
        int idx = (o * (nl + 2)) + layer0;
        Mat<float> img = p.Dog[idx];
        Mat<float> prev = p.Dog[idx - 1];
        Mat<float> next = p.Dog[idx + 1];
        int step = img.Stride;
        int cols = img.Width;
        float[] cur = img.DangerousGetArray();
        float[] prv = prev.DangerousGetArray();
        float[] nxt = next.DangerousGetArray();
        int rowOff = img.DangerousGetOffset() + (r * step);
        int prevOff = prev.DangerousGetOffset() + (r * prev.Stride);
        int nextOff = next.DangerousGetOffset() + (r * next.Stride);
        float thr = p.Threshold;

        int c = SiftConstants.ImgBorder;
        int cEnd = cols - SiftConstants.ImgBorder;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<float>.Count;
            var vthr = new Vector<float>(thr);
            for (; c <= cEnd - vc; c += vc)
            {
                uint mask = CandidateMask(cur, prv, nxt, rowOff + c, prevOff + c, nextOff + c, step, vthr);
                while (mask != 0)
                {
                    int k = BitOperations.TrailingZeroCount(mask);
                    mask &= mask - 1;
                    ProcessCandidate(p, o, layer0, r, c + k, hist, scratch, ref result);
                }
            }
        }

        for (; c < cEnd; c++)
        {
            float val = cur[rowOff + c];
            if (!(MathF.Abs(val) <= thr) && IsExtremum(cur, prv, nxt, rowOff + c, prevOff + c, nextOff + c, step, val))
            {
                ProcessCandidate(p, o, layer0, r, c, hist, scratch, ref result);
            }
        }
    }

    /// <summary>
    /// Lanes that are scale-space extrema, as the OpenCV vector loop of
    /// <c>findScaleSpaceExtremaT::process</c>; bit <c>k</c> is column <c>c + k</c>.
    /// </summary>
    private static uint CandidateMask(float[] cur, float[] prv, float[] nxt, int ci, int pi, int ni, int step, Vector<float> vthr)
    {
        var val = new Vector<float>(cur, ci);
        Vector<int> cond = Vector.GreaterThan(Vector.Abs(val), vthr);
        if (cond == Vector<int>.Zero)
        {
            return 0;
        }

        (Vector<float> vmax, Vector<float> vmin) = MinMax8(cur, ci, step);
        Vector<int> condp = cond & Vector.GreaterThan(val, Vector<float>.Zero) & Vector.GreaterThanOrEqual(val, vmax);
        Vector<int> condm = cond & Vector.LessThan(val, Vector<float>.Zero) & Vector.LessThanOrEqual(val, vmin);
        if ((condp | condm) == Vector<int>.Zero)
        {
            return 0;
        }

        (vmax, vmin) = MinMax8(prv, pi, step);
        condp &= Vector.GreaterThanOrEqual(val, vmax);
        condm &= Vector.LessThanOrEqual(val, vmin);
        if ((condp | condm) == Vector<int>.Zero)
        {
            return 0;
        }

        var p11 = new Vector<float>(prv, pi);
        var n11 = new Vector<float>(nxt, ni);
        (vmax, vmin) = MinMax8(nxt, ni, step);
        condp &= Vector.GreaterThanOrEqual(val, Vector.Max(vmax, Vector.Max(n11, p11)));
        condm &= Vector.LessThanOrEqual(val, Vector.Min(vmin, Vector.Min(n11, p11)));
        return MaskBits(condp | condm);
    }

    /// <summary>Bit <c>k</c> set when lane <c>k</c> of <paramref name="mask"/> is set.</summary>
    internal static uint MaskBits(Vector<int> mask)
    {
        return Vector<int>.Count switch
        {
            4 => mask.AsVector128().ExtractMostSignificantBits(),
            8 => mask.AsVector256().ExtractMostSignificantBits(),
            _ => (uint)mask.AsVector512().ExtractMostSignificantBits(),
        };
    }

    private static (Vector<float> Max, Vector<float> Min) MinMax8(float[] a, int i, int step)
    {
        var v00 = new Vector<float>(a, i - step - 1);
        var v01 = new Vector<float>(a, i - step);
        var v02 = new Vector<float>(a, i - step + 1);
        var v10 = new Vector<float>(a, i - 1);
        var v12 = new Vector<float>(a, i + 1);
        var v20 = new Vector<float>(a, i + step - 1);
        var v21 = new Vector<float>(a, i + step);
        var v22 = new Vector<float>(a, i + step + 1);
        Vector<float> max = Vector.Max(Vector.Max(Vector.Max(v00, v01), Vector.Max(v02, v10)), Vector.Max(Vector.Max(v12, v20), Vector.Max(v21, v22)));
        Vector<float> min = Vector.Min(Vector.Min(Vector.Min(v00, v01), Vector.Min(v02, v10)), Vector.Min(Vector.Min(v12, v20), Vector.Min(v21, v22)));
        return (max, min);
    }

    /// <summary>Refines one extremum and appends one keypoint per orientation peak.</summary>
    private static void ProcessCandidate(Params p, int o, int layer0, int r, int c, Span<float> hist, OrientationScratch scratch, ref List<KeyPoint>? result)
    {
        int nl = p.NOctaveLayers;
        var kpt = default(KeyPoint);
        int r1 = r;
        int c1 = c;
        int layer = layer0;
        if (!AdjustLocalExtrema(p, ref kpt, o, ref layer, ref r1, ref c1))
        {
            return;
        }

        float sclOctv = kpt.Size * 0.5f / (1 << o);
        float omax = CalcOrientationHist(
            p.Gauss[(o * (nl + 3)) + layer],
            c1,
            r1,
            ScalarConverter.CvRound(SiftConstants.OriRadius * sclOctv),
            SiftConstants.OriSigFctr * sclOctv,
            hist,
            scratch);
        float magThr = omax * SiftConstants.OriPeakRatio;
        const int n = SiftConstants.OriHistBins;
        for (int j = 0; j < n; j++)
        {
            int l = j > 0 ? j - 1 : n - 1;
            int r2 = j < n - 1 ? j + 1 : 0;
            if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= magThr)
            {
                float bin = j + (0.5f * (hist[l] - hist[r2]) / (hist[l] - (2 * hist[j]) + hist[r2]));
                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                kpt.Angle = 360f - ((360f / n) * bin);
                if (MathF.Abs(kpt.Angle - 360f) < SiftConstants.FltEpsilon)
                {
                    kpt.Angle = 0f;
                }

                result ??= new List<KeyPoint>();
                result.Add(kpt);
            }
        }
    }

    private static bool IsExtremum(float[] cur, float[] prv, float[] nxt, int ci, int pi, int ni, int step, float val)
    {
        if (val > 0)
        {
            if (val >= Max8(cur, ci, step) && val >= Max8(prv, pi, step) && val >= Max8(nxt, ni, step))
            {
                return val >= StdMax(prv[pi], nxt[ni]);
            }

            return false;
        }

        if (val <= Min8(cur, ci, step) && val <= Min8(prv, pi, step) && val <= Min8(nxt, ni, step))
        {
            return val <= StdMin(prv[pi], nxt[ni]);
        }

        return false;
    }

    private static float Max8(float[] a, int i, int step)
        => StdMax(
            StdMax(StdMax(a[i - step - 1], a[i - step]), StdMax(a[i - step + 1], a[i - 1])),
            StdMax(StdMax(a[i + 1], a[i + step - 1]), StdMax(a[i + step], a[i + step + 1])));

    private static float Min8(float[] a, int i, int step)
        => StdMin(
            StdMin(StdMin(a[i - step - 1], a[i - step]), StdMin(a[i - step + 1], a[i - 1])),
            StdMin(StdMin(a[i + 1], a[i + step - 1]), StdMin(a[i + step], a[i + step + 1])));

    private static float StdMax(float a, float b) => a < b ? b : a;

    private static float StdMin(float a, float b) => b < a ? b : a;

    /// <summary>OpenCV <c>adjustLocalExtrema</c> with float DoG images.</summary>
    private static bool AdjustLocalExtrema(Params p, ref KeyPoint kpt, int octv, ref int layer, ref int r, ref int c)
    {
        const float imgScale = 1f / 255f;
        const float derivScale = imgScale * 0.5f;
        const float secondDerivScale = imgScale;
        const float crossDerivScale = imgScale * 0.25f;
        int nl = p.NOctaveLayers;

        float xi = 0, xr = 0, xc = 0;
        int i = 0;
        for (; i < SiftConstants.MaxInterpSteps; i++)
        {
            int idx = (octv * (nl + 2)) + layer;
            Mat<float> img = p.Dog[idx];
            Mat<float> prev = p.Dog[idx - 1];
            Mat<float> next = p.Dog[idx + 1];

            float dx = (img.At(r, c + 1) - img.At(r, c - 1)) * derivScale;
            float dy = (img.At(r + 1, c) - img.At(r - 1, c)) * derivScale;
            float ds = (next.At(r, c) - prev.At(r, c)) * derivScale;

            float v2 = img.At(r, c) * 2;
            float dxx = (img.At(r, c + 1) + img.At(r, c - 1) - v2) * secondDerivScale;
            float dyy = (img.At(r + 1, c) + img.At(r - 1, c) - v2) * secondDerivScale;
            float dss = (next.At(r, c) + prev.At(r, c) - v2) * secondDerivScale;
            float dxy = (img.At(r + 1, c + 1) - img.At(r + 1, c - 1) - img.At(r - 1, c + 1) + img.At(r - 1, c - 1)) * crossDerivScale;
            float dxs = (next.At(r, c + 1) - next.At(r, c - 1) - prev.At(r, c + 1) + prev.At(r, c - 1)) * crossDerivScale;
            float dys = (next.At(r + 1, c) - next.At(r - 1, c) - prev.At(r + 1, c) + prev.At(r - 1, c)) * crossDerivScale;

            Solve3(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss, dx, dy, ds, out float x0, out float x1, out float x2);

            xi = -x2;
            xr = -x1;
            xc = -x0;

            if (MathF.Abs(xi) < 0.5f && MathF.Abs(xr) < 0.5f && MathF.Abs(xc) < 0.5f)
            {
                break;
            }

            const float limit = int.MaxValue / 3;
            if (MathF.Abs(xi) > limit || MathF.Abs(xr) > limit || MathF.Abs(xc) > limit)
            {
                return false;
            }

            c += ScalarConverter.CvRound(xc);
            r += ScalarConverter.CvRound(xr);
            layer += ScalarConverter.CvRound(xi);

            if (layer < 1 || layer > nl ||
                c < SiftConstants.ImgBorder || c >= img.Width - SiftConstants.ImgBorder ||
                r < SiftConstants.ImgBorder || r >= img.Height - SiftConstants.ImgBorder)
            {
                return false;
            }
        }

        if (i >= SiftConstants.MaxInterpSteps)
        {
            return false;
        }

        float contr;
        {
            int idx = (octv * (nl + 2)) + layer;
            Mat<float> img = p.Dog[idx];
            Mat<float> prev = p.Dog[idx - 1];
            Mat<float> next = p.Dog[idx + 1];
            float dx = (img.At(r, c + 1) - img.At(r, c - 1)) * derivScale;
            float dy = (img.At(r + 1, c) - img.At(r - 1, c)) * derivScale;
            float ds = (next.At(r, c) - prev.At(r, c)) * derivScale;
            float t = 0f + (dx * xc);
            t += dy * xr;
            t += ds * xi;

            contr = (img.At(r, c) * imgScale) + (t * 0.5f);
            if (MathF.Abs(contr) * nl < p.ContrastThreshold)
            {
                return false;
            }

            float v2 = img.At(r, c) * 2f;
            float dxx = (img.At(r, c + 1) + img.At(r, c - 1) - v2) * secondDerivScale;
            float dyy = (img.At(r + 1, c) + img.At(r - 1, c) - v2) * secondDerivScale;
            float dxy = (img.At(r + 1, c + 1) - img.At(r + 1, c - 1) - img.At(r - 1, c + 1) + img.At(r - 1, c - 1)) * crossDerivScale;
            float tr = dxx + dyy;
            float det = (dxx * dyy) - (dxy * dxy);

            float e = p.EdgeThreshold;
            if (det <= 0 || tr * tr * e >= (e + 1) * (e + 1) * det)
            {
                return false;
            }
        }

        kpt.Pt = new Point2f((c + xc) * (1 << octv), (r + xr) * (1 << octv));
        kpt.Octave = octv + (layer << 8) + (ScalarConverter.CvRound((xi + 0.5) * 255) << 16);
        kpt.Size = p.Sigma * MathF.Pow(2f, (layer + xi) / nl) * (1 << octv) * 2;
        kpt.Response = MathF.Abs(contr);
        kpt.ClassId = -1;
        kpt.Angle = 0f;
        return true;
    }

    /// <summary>OpenCV <c>Matx_FastSolveOp&lt;float, 3, 3, 1&gt;</c> (Cramer's rule); zero for a singular matrix.</summary>
    private static void Solve3(
        float a00, float a01, float a02,
        float a10, float a11, float a12,
        float a20, float a21, float a22,
        float b0, float b1, float b2,
        out float x0, out float x1, out float x2)
    {
        float d = (a00 * ((a11 * a22) - (a21 * a12))) -
                  (a01 * ((a10 * a22) - (a20 * a12))) +
                  (a02 * ((a10 * a21) - (a20 * a11)));
        if (d == 0)
        {
            x0 = x1 = x2 = 0;
            return;
        }

        d = 1 / d;
        x0 = d * ((b0 * ((a11 * a22) - (a12 * a21))) -
                  (a01 * ((b1 * a22) - (a12 * b2))) +
                  (a02 * ((b1 * a21) - (a11 * b2))));
        x1 = d * ((a00 * ((b1 * a22) - (a12 * b2))) -
                  (b0 * ((a10 * a22) - (a12 * a20))) +
                  (a02 * ((a10 * b2) - (b1 * a20))));
        x2 = d * ((a00 * ((a11 * b2) - (b1 * a21))) -
                  (a01 * ((a10 * b2) - (b1 * a20))) +
                  (b0 * ((a10 * a21) - (a11 * a20))));
    }

    /// <summary>Per-thread buffers of <see cref="CalcOrientationHist"/>. X, Y, Ori and W are pooled arrays.</summary>
    private sealed class OrientationScratch : IDisposable
    {
        public float[] X = ArrayPool<float>.Shared.Rent(256);
        public float[] Y = ArrayPool<float>.Shared.Rent(256);
        public float[] Ori = ArrayPool<float>.Shared.Rent(256);
        public float[] W = ArrayPool<float>.Shared.Rent(256);
        public float[] TempHist = new float[SiftConstants.OriHistBins + 4];

        public void Ensure(int len)
        {
            if (X.Length < len)
            {
                Dispose();
                X = ArrayPool<float>.Shared.Rent(len);
                Y = ArrayPool<float>.Shared.Rent(len);
                Ori = ArrayPool<float>.Shared.Rent(len);
                W = ArrayPool<float>.Shared.Rent(len);
            }
        }

        public void Dispose()
        {
            ArrayPool<float>.Shared.Return(X);
            ArrayPool<float>.Shared.Return(Y);
            ArrayPool<float>.Shared.Return(Ori);
            ArrayPool<float>.Shared.Return(W);
        }
    }

    /// <summary>
    /// OpenCV <c>calcOrientationHist</c>. The smoothing uses the 128-bit vector form
    /// <c>(t[-2] + t[2])/16 + ((t[-1] + t[1])·4/16 + t[0]·6/16)</c>, which covers all 36 bins.
    /// </summary>
    private static float CalcOrientationHist(Mat<float> img, int ptX, int ptY, int radius, float sigma, Span<float> hist, OrientationScratch s)
    {
        const int n = SiftConstants.OriHistBins;
        int len = ((radius * 2) + 1) * ((radius * 2) + 1);
        float expfScale = -1f / (2f * sigma * sigma);
        s.Ensure(len);
        float[] xs = s.X;
        float[] ys = s.Y;
        float[] w = s.W;
        float[] ori = s.Ori;
        Span<float> temphist = s.TempHist;
        temphist.Clear();

        float[] data = img.DangerousGetArray();
        int off = img.DangerousGetOffset();
        int stride = img.Stride;
        int k = 0;
        for (int i = -radius; i <= radius; i++)
        {
            int y = ptY + i;
            if (y <= 0 || y >= img.Height - 1)
            {
                continue;
            }

            int row = off + (y * stride);
            for (int j = -radius; j <= radius; j++)
            {
                int x = ptX + j;
                if (x <= 0 || x >= img.Width - 1)
                {
                    continue;
                }

                xs[k] = data[row + x + 1] - data[row + x - 1];
                ys[k] = data[row - stride + x] - data[row + stride + x];
                w[k] = ((i * i) + (j * j)) * expfScale;
                k++;
            }
        }

        len = k;
        FastMath.Exp32f(w.AsSpan(0, len), w.AsSpan(0, len));
        FastMath.FastAtan32fDegrees(ys.AsSpan(0, len), xs.AsSpan(0, len), ori.AsSpan(0, len));
        float[] mag = xs;
        FastMath.Magnitude32f(xs.AsSpan(0, len), ys.AsSpan(0, len), mag.AsSpan(0, len));

        const float nd360 = n / 360f;
        for (k = 0; k < len; k++)
        {
            int bin = ScalarConverter.CvRound(nd360 * ori[k]);
            if (bin >= n)
            {
                bin -= n;
            }

            if (bin < 0)
            {
                bin += n;
            }

            temphist[bin + 2] += w[k] * mag[k];
        }

        temphist[1] = temphist[n + 1];
        temphist[0] = temphist[n];
        temphist[n + 2] = temphist[2];
        temphist[n + 3] = temphist[3];

        const float d1 = 1f / 16f;
        const float d4 = 4f / 16f;
        const float d6 = 6f / 16f;
        for (int i = 0; i < n; i++)
        {
            float inner = ((temphist[i + 1] + temphist[i + 3]) * d4) + (temphist[i + 2] * d6);
            hist[i] = ((temphist[i] + temphist[i + 4]) * d1) + inner;
        }

        float maxval = hist[0];
        for (int i = 1; i < n; i++)
        {
            maxval = StdMax(maxval, hist[i]);
        }

        return maxval;
    }
}
