// This implementation is ported from OpenCV's robust line fitting.
//
// Original source:
//   modules/geometry/src/linefit.cpp (fitLine2D_wods, calcDist2D, weightL1, weightL12,
//   weightHuber, weightFair, weightWelsch, fitLine2D, fitLine)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/linefit.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>fitLine</c> for 2-D points, with the iteratively reweighted M-estimators.</summary>
/// <remarks>
/// The port fuses OpenCV passes that are independent: <c>calcDist2D</c>, the weight function and
/// the weight sum run in one loop, and the weight normalization runs in the <c>fitLine2D_wods</c>
/// loop. Each double sum keeps its OpenCV order, so the results are bit-identical.
/// </remarks>
internal static class FitLine2D
{
    private const double Eps = 1e-6;
    private const float FltEpsilon = 1.1920928955078125e-7f;

    private interface IWeight
    {
        static abstract float Compute(float d, float c);
    }

    /// <summary>Fits a line. Writes (vx, vy, x0, y0) to <paramref name="line"/>.</summary>
    public static void Fit(ReadOnlySpan<Point2f> points, DistanceTypes dist, float param, float reps, float aeps, Span<float> line)
    {
        line[..4].Clear();
        switch (dist)
        {
            case DistanceTypes.L2:
                WodsUnit(points, line);
                break;
            case DistanceTypes.L1:
                Fit<L1Weight>(points, 0, reps, aeps, line);
                break;
            case DistanceTypes.L12:
                Fit<L12Weight>(points, 0, reps, aeps, line);
                break;
            case DistanceTypes.Huber:
                Fit<HuberWeight>(points, param <= 0 ? 1.345f : param, reps, aeps, line);
                break;
            case DistanceTypes.Fair:
                Fit<FairWeight>(points, param == 0 ? 1 / 1.3998f : 1 / param, reps, aeps, line);
                break;
            case DistanceTypes.Welsch:
                Fit<WelschWeight>(points, param == 0 ? 1 / 2.9846f : 1 / param, reps, aeps, line);
                break;
            default:
                throw new ArgumentException("Unknown distance type.", nameof(dist));
        }
    }

    private static void Fit<TWeight>(ReadOnlySpan<Point2f> points, float c, float reps, float aeps, Span<float> line)
        where TWeight : IWeight
    {
        int count = points.Length;
        double epsErr = count * FltEpsilon;
        float rdelta = reps != 0 ? reps : 1.0f;
        float adelta = aeps != 0 ? aeps : 0.01f;
        double minErr = double.MaxValue, err = 0;
        var rng = new CvRng(ulong.MaxValue);

        float[] rented = ArrayPool<float>.Shared.Rent(count);
        try
        {
            Span<float> w = rented.AsSpan(0, count);
            Span<float> cur = stackalloc float[4];
            Span<float> prev = stackalloc float[4];

            for (int k = 0; k < 20; k++)
            {
                bool first = true;
                w.Clear();

                for (int i = 0; i < Math.Min(count, 10);)
                {
                    int j = rng.Uniform(0, count);
                    if (w[j] < FltEpsilon)
                    {
                        w[j] = 1f;
                        i++;
                    }
                }

                WodsScaled(points, w, 1.0, false, cur);
                for (int i = 0; i < 30; i++)
                {
                    if (first)
                    {
                        first = false;
                    }
                    else
                    {
                        double t = (cur[0] * prev[0]) + (cur[1] * prev[1]);
                        t = Math.Max(t, -1.0);
                        t = Math.Min(t, 1.0);
                        if (Math.Abs(Math.Acos(t)) < adelta)
                        {
                            float x = MathF.Abs(cur[2] - prev[2]);
                            float y = MathF.Abs(cur[3] - prev[3]);
                            float d = x > y ? x : y;
                            if (d < rdelta)
                            {
                                break;
                            }
                        }
                    }

                    // The weights of a final iteration are unused: the next restart clears them.
                    err = DistAndWeights<TWeight>(points, cur, w, c, out double sumW);
                    if (err < minErr)
                    {
                        minErr = err;
                        cur.CopyTo(line);
                        if (err < epsErr)
                        {
                            break;
                        }
                    }

                    cur.CopyTo(prev);
                    if (Math.Abs(sumW) > FltEpsilon)
                    {
                        WodsScaled(points, w, 1.0 / sumW, true, cur);
                    }
                    else
                    {
                        w.Fill(1f);
                        WodsScaled(points, w, 1.0, false, cur);
                    }
                }

                if (err < minErr)
                {
                    minErr = err;
                    cur.CopyTo(line);
                    if (err < epsErr)
                    {
                        break;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary><c>fitLine2D_wods</c> with unit weights.</summary>
    private static void WodsUnit(ReadOnlySpan<Point2f> points, Span<float> line)
    {
        int count = points.Length;
        double x = 0, y = 0, x2 = 0, y2 = 0, xy = 0;
        ref Point2f p = ref MemoryMarshal.GetReference(points);
        for (int i = 0; i < count; i++)
        {
            Point2f q = Unsafe.Add(ref p, i);
            float px = q.X, py = q.Y;
            x += px;
            y += py;
            x2 += px * px;
            y2 += py * py;
            xy += px * py;
        }

        Finish(x, y, x2, y2, xy, (float)count, line);
    }

    /// <summary>
    /// <c>fitLine2D_wods</c> with weights. With <paramref name="normalize"/>, weight i is
    /// <c>(float)(w[i] * scale)</c>, the OpenCV normalized weight.
    /// </summary>
    private static void WodsScaled(ReadOnlySpan<Point2f> points, ReadOnlySpan<float> weights, double scale, bool normalize, Span<float> line)
    {
        int count = points.Length;
        double x = 0, y = 0, x2 = 0, y2 = 0, xy = 0, w = 0;
        ref Point2f p = ref MemoryMarshal.GetReference(points);
        ref float wr = ref MemoryMarshal.GetReference(weights);
        for (int i = 0; i < count; i++)
        {
            Point2f q = Unsafe.Add(ref p, i);
            float px = q.X, py = q.Y;
            float wi = Unsafe.Add(ref wr, i);
            if (normalize)
            {
                wi = (float)(wi * scale);
            }

            x += wi * px;
            y += wi * py;
            x2 += wi * px * px;
            y2 += wi * py * py;
            xy += wi * px * py;
            w += wi;
        }

        Finish(x, y, x2, y2, xy, w, line);
    }

    private static void Finish(double x, double y, double x2, double y2, double xy, double w, Span<float> line)
    {
        x /= w;
        y /= w;
        x2 /= w;
        y2 /= w;
        xy /= w;

        double dx2 = x2 - (x * x);
        double dy2 = y2 - (y * y);
        double dxy = xy - (x * y);

        float t = (float)Math.Atan2(2 * dxy, dx2 - dy2) / 2;
        line[0] = MathF.Cos(t);
        line[1] = MathF.Sin(t);
        line[2] = (float)x;
        line[3] = (float)y;
    }

    /// <summary>
    /// <c>calcDist2D</c>, the weight function and the weight sum. Returns the distance sum.
    /// </summary>
    private static double DistAndWeights<TWeight>(ReadOnlySpan<Point2f> points, ReadOnlySpan<float> line, Span<float> w, float c, out double sumW)
        where TWeight : IWeight
    {
        float px = line[2], py = line[3];
        float nx = line[1], ny = -line[0];
        double sum = 0, sw = 0;
        ref Point2f p = ref MemoryMarshal.GetReference(points);
        ref float wr = ref MemoryMarshal.GetReference(w);
        for (int j = 0; j < points.Length; j++)
        {
            Point2f q = Unsafe.Add(ref p, j);
            float x = q.X - px;
            float y = q.Y - py;
            float d = MathF.Abs((nx * x) + (ny * y));
            sum += d;
            float wj = TWeight.Compute(d, c);
            Unsafe.Add(ref wr, j) = wj;
            sw += wj;
        }

        sumW = sw;
        return sum;
    }

    private readonly struct L1Weight : IWeight
    {
        public static float Compute(float d, float c) => (float)(1.0 / Math.Max(Math.Abs((double)d), Eps));
    }

    private readonly struct L12Weight : IWeight
    {
        public static float Compute(float d, float c) => 1.0f / (float)Math.Sqrt(1 + (double)(d * d * 0.5));
    }

    private readonly struct HuberWeight : IWeight
    {
        public static float Compute(float d, float c) => d < c ? 1.0f : c / d;
    }

    private readonly struct FairWeight : IWeight
    {
        public static float Compute(float d, float c) => 1 / (1 + (d * c));
    }

    private readonly struct WelschWeight : IWeight
    {
        public static float Compute(float d, float c) => MathF.Exp(-d * d * c * c);
    }
}
