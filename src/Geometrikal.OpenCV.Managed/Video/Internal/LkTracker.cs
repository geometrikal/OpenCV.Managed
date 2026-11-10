// This implementation is ported from the OpenCV sparse optical flow module.
//
// Original sources:
//   modules/video/src/lkpyramid.cpp (LKTrackerInvoker::operator(), CV_SIMD128 SSE path)
//   modules/video/src/lkpyramid.hpp (LKTrackerInvoker)
//   modules/core/include/opencv2/core/hal/intrin_sse.hpp (v_dotprod, v_reduce_sum)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Video.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// One pyramid level of OpenCV <c>LKTrackerInvoker</c>. Integer bilinear sums are exact. Float
/// sums follow the OpenCV SSE path: elements in the 8-wide vector part of a window row go to
/// 4-lane accumulators, the remaining elements go to one scalar sum.
/// </summary>
internal static class LkTracker
{
    private const int WBits = 14;
    private const float FltScale = 1f / (1 << 20);
    private const float FltEpsilon = 1.1920929E-07f;

    /// <summary>Selects the scalar emulation of the vector lanes on every processor.</summary>
    internal static bool ForceScalar { get; set; }

    /// <summary>Per-level parameters.</summary>
    public readonly record struct Level(
        Mat<byte> I,
        Mat<short> DerivI,
        Mat<byte> J,
        Size WinSize,
        int MaxCount,
        double Epsilon,
        int LevelIndex,
        int MaxLevel,
        bool UseInitialFlow,
        bool GetMinEigenvals,
        float MinEigThreshold);

    public static void Run(in Level p, Point2f[] prevPts, Point2f[] nextPts, byte[] status, float[]? err)
    {
        Level lv = p;
        long work = (long)p.WinSize.Width * p.WinSize.Height * p.I.Channels * 16;
        ParallelRows.For(prevPts.Length, work, (s, e) => Range(lv, prevPts, nextPts, status, err, s, e));
    }

    private static int Round(float v) => (int)MathF.Round(v, MidpointRounding.ToEven);

    private static short Sat(int v) => (short)Math.Clamp(v, short.MinValue, short.MaxValue);

    private static void Weights(float a, float b, out int iw00, out int iw01, out int iw10, out int iw11)
    {
        iw00 = Round((1f - a) * (1f - b) * (1 << WBits));
        iw01 = Round(a * (1f - b) * (1 << WBits));
        iw10 = Round((1f - a) * b * (1 << WBits));
        iw11 = (1 << WBits) - iw00 - iw01 - iw10;
    }

    private static void Range(Level p, Point2f[] prevPts, Point2f[] nextPts, byte[] status, float[]? err, int start, int end)
    {
        int n = p.WinSize.Width * p.I.Channels;
        short[] iwin = ArrayPool<short>.Shared.Rent(n * p.WinSize.Height);
        short[] dwin = ArrayPool<short>.Shared.Rent(n * p.WinSize.Height * 2);
        Point2f[] scaled = ArrayPool<Point2f>.Shared.Rent(end - start);
        try
        {
            RangeCore(p, prevPts, nextPts, status, err, start, end, iwin, dwin, scaled);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(scaled);
            ArrayPool<short>.Shared.Return(dwin);
            ArrayPool<short>.Shared.Return(iwin);
        }
    }

    /// <param name="iwin">Scratch of at least <c>winSize.Width · channels · winSize.Height</c> elements.</param>
    /// <param name="dwin">Scratch of twice the length of <paramref name="iwin"/>.</param>
    /// <param name="scaled">Scratch of at least <c>end − start</c> elements.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RangeCore(Level p, Point2f[] prevPts, Point2f[] nextPts, byte[] status, float[]? err, int start, int end, short[] iwin, short[] dwin, Point2f[] scaled)
    {
        Mat<byte> imgI = p.I, imgJ = p.J;
        Mat<short> deriv = p.DerivI;
        int ww = p.WinSize.Width, wh = p.WinSize.Height;
        int level = p.LevelIndex;
        float halfX = (ww - 1) * 0.5f, halfY = (wh - 1) * 0.5f;
        int cn = imgI.Channels, cn2 = cn * 2;
        int n = ww * cn;
        int nv = n >= 8 ? n & ~7 : 0;

        byte[] ib = imgI.DangerousGetArray();
        int iOff = imgI.DangerousGetOffset(), stepI = imgI.Stride;
        short[] db = deriv.DangerousGetArray();
        int dOff = deriv.DangerousGetOffset(), dstep = deriv.Stride;
        byte[] jb = imgJ.DangerousGetArray();
        int jOff = imgJ.DangerousGetOffset(), stepJ = imgJ.Stride;

        Span<float> acc = stackalloc float[12];
        Span<float> bacc = stackalloc float[8];
        Span<int> dv = stackalloc int[8];

        bool sse = LkSse.IsSupported && !ForceScalar;
        float lvlScale = (float)(1.0 / (1 << level));
        for (int i = start; i < end; i++)
        {
            var prevPt = new Point2f(prevPts[i].X * lvlScale, prevPts[i].Y * lvlScale);
            Point2f nextPt;
            if (level == p.MaxLevel)
            {
                nextPt = p.UseInitialFlow ? new Point2f(nextPts[i].X * lvlScale, nextPts[i].Y * lvlScale) : prevPt;
            }
            else
            {
                nextPt = new Point2f(nextPts[i].X * 2f, nextPts[i].Y * 2f);
            }

            nextPts[i] = nextPt;
            scaled[i - start] = prevPt;
        }

        for (int ptidx = start; ptidx < end; ptidx++)
        {
            float px = scaled[ptidx - start].X - halfX, py = scaled[ptidx - start].Y - halfY;
            int ipx = ScalarConverter.CvFloor(px), ipy = ScalarConverter.CvFloor(py);
            if (ipx < -ww || ipx >= deriv.Width || ipy < -wh || ipy >= deriv.Height)
            {
                if (level == 0)
                {
                    status[ptidx] = 0;
                    if (err != null)
                    {
                        err[ptidx] = 0;
                    }
                }

                continue;
            }

            Weights(px - ipx, py - ipy, out int iw00, out int iw01, out int iw10, out int iw11);

            float iA11 = 0, iA12 = 0, iA22 = 0;
            if (sse)
            {
                LkSse.Patch(
                    ib, iOff + (ipy * stepI) + (ipx * cn), stepI, cn, db, dOff + (ipy * dstep) + (ipx * cn2), dstep, cn2, iwin, dwin, n, nv, wh,
                    iw00, iw01, iw10, iw11, out iA11, out iA12, out iA22);
            }
            else
            {
                PatchScalar(ib, iOff + (ipy * stepI) + (ipx * cn), stepI, cn, db, dOff + (ipy * dstep) + (ipx * cn2), dstep, cn2, iwin, dwin, n, nv, wh, iw00, iw01, iw10, iw11, acc, out iA11, out iA12, out iA22);
            }

            float a11 = iA11 * FltScale, a12 = iA12 * FltScale, a22 = iA22 * FltScale;
            float det = (a11 * a22) - (a12 * a12);
            float minEig = (a22 + a11 - MathF.Sqrt(((a11 - a22) * (a11 - a22)) + (4f * a12 * a12))) / (2 * ww * wh);

            if (err != null && p.GetMinEigenvals)
            {
                err[ptidx] = minEig;
            }

            if (minEig < p.MinEigThreshold || det < FltEpsilon)
            {
                if (level == 0)
                {
                    status[ptidx] = 0;
                }

                continue;
            }

            det = 1f / det;
            float nx = nextPts[ptidx].X - halfX, ny = nextPts[ptidx].Y - halfY;
            float pdx = 0, pdy = 0;

            for (int j = 0; j < p.MaxCount; j++)
            {
                int inx = ScalarConverter.CvFloor(nx), iny = ScalarConverter.CvFloor(ny);
                if (inx < -ww || inx >= imgJ.Width || iny < -wh || iny >= imgJ.Height)
                {
                    if (level == 0)
                    {
                        status[ptidx] = 0;
                    }

                    break;
                }

                Weights(nx - inx, ny - iny, out iw00, out iw01, out iw10, out iw11);
                float ib1, ib2;
                int sj = jOff + (iny * stepJ) + (inx * cn);
                if (sse)
                {
                    LkSse.Diff(jb, sj, stepJ, cn, iwin, dwin, n, nv, wh, iw00, iw01, iw10, iw11, out ib1, out ib2);
                }
                else
                {
                    DiffScalar(jb, sj, stepJ, cn, iwin, dwin, n, nv, wh, iw00, iw01, iw10, iw11, bacc, dv, out ib1, out ib2);
                }

                float b1 = ib1 * FltScale, b2 = ib2 * FltScale;

                float dx = ((a12 * b2) - (a22 * b1)) * det;
                float dy = ((a12 * b1) - (a11 * b2)) * det;
                nx += dx;
                ny += dy;
                nextPts[ptidx] = new Point2f(nx + halfX, ny + halfY);

                if (((double)dx * dx) + ((double)dy * dy) <= p.Epsilon)
                {
                    break;
                }

                if (j > 0 && Math.Abs(dx + pdx) < 0.01 && Math.Abs(dy + pdy) < 0.01)
                {
                    nextPts[ptidx] = new Point2f(nextPts[ptidx].X - (dx * 0.5f), nextPts[ptidx].Y - (dy * 0.5f));
                    break;
                }

                pdx = dx;
                pdy = dy;
            }

            if (status[ptidx] != 0 && err != null && level == 0 && !p.GetMinEigenvals)
            {
                float ex = nextPts[ptidx].X - halfX, ey = nextPts[ptidx].Y - halfY;
                int iex = ScalarConverter.CvFloor(ex), iey = ScalarConverter.CvFloor(ey);
                if (iex < -ww || iex >= imgJ.Width || iey < -wh || iey >= imgJ.Height)
                {
                    status[ptidx] = 0;
                    continue;
                }

                Weights(ex - iex, ey - iey, out iw00, out iw01, out iw10, out iw11);
                float errval = ErrorSum(jb, jOff + (iey * stepJ) + (iex * cn), stepJ, cn, iwin, n, wh, iw00, iw01, iw10, iw11);
                err[ptidx] = errval * 1f / (32 * ww * cn * wh);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float ErrorSum(byte[] jb, int s0, int stepJ, int cn, short[] iwin, int n, int wh, int iw00, int iw01, int iw10, int iw11)
    {
        ref byte src = ref MemoryMarshal.GetArrayDataReference(jb);
        ref short iw = ref MemoryMarshal.GetArrayDataReference(iwin);
        float errval = 0f;
        for (int y = 0; y < wh; y++)
        {
            int s = s0 + (y * stepJ);
            int w = y * n;
            for (int x = 0; x < n; x++)
            {
                int q = s + x;
                int diff = (((Unsafe.Add(ref src, q) * iw00) + (Unsafe.Add(ref src, q + cn) * iw01)
                    + (Unsafe.Add(ref src, q + stepJ) * iw10) + (Unsafe.Add(ref src, q + stepJ + cn) * iw11) + 256) >> (WBits - 5))
                    - Unsafe.Add(ref iw, w + x);
                errval += MathF.Abs(diff);
            }
        }

        return errval;
    }

    /// <summary>Scalar emulation of <see cref="LkSse.Patch"/>.</summary>
    private static void PatchScalar(
        byte[] ib, int s0, int stepI, int cn, short[] db, int d0, int dstep, int cn2, short[] iwin, short[] dwin, int n, int nv, int wh,
        int iw00, int iw01, int iw10, int iw11, Span<float> acc, out float iA11, out float iA12, out float iA22)
    {
        iA11 = 0;
        iA12 = 0;
        iA22 = 0;
        acc.Clear();
        for (int y = 0; y < wh; y++)
        {
            int s = s0 + (y * stepI);
            int d = d0 + (y * dstep);
            int w = y * n;
            for (int x = 0; x < n; x++, s++, d += 2)
            {
                int ival = ((ib[s] * iw00) + (ib[s + cn] * iw01) + 256 + (ib[s + stepI] * iw10) + (ib[s + stepI + cn] * iw11)) >> (WBits - 5);
                int ix = ((db[d] * iw00) + (db[d + cn2] * iw01) + (1 << (WBits - 1)) + (db[d + dstep] * iw10) + (db[d + dstep + cn2] * iw11)) >> WBits;
                int iy = ((db[d + 1] * iw00) + (db[d + cn2 + 1] * iw01) + (1 << (WBits - 1)) + (db[d + dstep + 1] * iw10) + (db[d + dstep + cn2 + 1] * iw11)) >> WBits;
                if (x < nv)
                {
                    short sx = Sat(ix), sy = Sat(iy);
                    iwin[w + x] = Sat(ival);
                    dwin[(w + x) * 2] = sx;
                    dwin[((w + x) * 2) + 1] = sy;
                    float fx = sx, fy = sy;
                    int lane = x & 3;
                    acc[lane] += fx * fx;
                    acc[4 + lane] += fx * fy;
                    acc[8 + lane] += fy * fy;
                }
                else
                {
                    iwin[w + x] = (short)ival;
                    dwin[(w + x) * 2] = (short)ix;
                    dwin[((w + x) * 2) + 1] = (short)iy;
                    iA11 += ix * ix;
                    iA12 += ix * iy;
                    iA22 += iy * iy;
                }
            }
        }

        iA11 += (acc[0] + acc[2]) + (acc[1] + acc[3]);
        iA12 += (acc[4] + acc[6]) + (acc[5] + acc[7]);
        iA22 += (acc[8] + acc[10]) + (acc[9] + acc[11]);
    }

    /// <summary>Scalar emulation of <see cref="LkSse.Diff"/>.</summary>
    private static void DiffScalar(
        byte[] jb, int s0, int stepJ, int cn, short[] iwin, short[] dwin, int n, int nv, int wh,
        int iw00, int iw01, int iw10, int iw11, Span<float> bacc, Span<int> dv, out float ib1, out float ib2)
    {
        ib1 = 0;
        ib2 = 0;
        bacc.Clear();
        for (int y = 0; y < wh; y++)
        {
            int s = s0 + (y * stepJ);
            int w = y * n;
            int x = 0;
            for (; x < nv; x += 8)
            {
                for (int k = 0; k < 8; k++)
                {
                    int q = s + x + k;
                    int jv = ((jb[q] * iw00) + (jb[q + cn] * iw01) + 256 + (jb[q + stepJ] * iw10) + (jb[q + stepJ + cn] * iw11)) >> (WBits - 5);
                    dv[k] = (short)(Sat(jv) - iwin[w + x + k]);
                }

                int e = (w + x) * 2;
                for (int k = 0; k < 4; k++)
                {
                    int e0 = e + (k * 2), e1 = e + ((k + 4) * 2);
                    bacc[k] += (dv[k] * dwin[e0]) + (dv[k + 4] * dwin[e1]);
                    bacc[4 + k] += (dv[k] * dwin[e0 + 1]) + (dv[k + 4] * dwin[e1 + 1]);
                }
            }

            for (; x < n; x++)
            {
                int q = s + x;
                int diff = (((jb[q] * iw00) + (jb[q + cn] * iw01) + (jb[q + stepJ] * iw10) + (jb[q + stepJ + cn] * iw11) + 256) >> (WBits - 5)) - iwin[w + x];
                ib1 += diff * dwin[(w + x) * 2];
                ib2 += diff * dwin[((w + x) * 2) + 1];
            }
        }

        ib1 += (bacc[0] + bacc[2]) + (bacc[1] + bacc[3]);
        ib2 += (bacc[4] + bacc[6]) + (bacc[5] + bacc[7]);
    }
}
