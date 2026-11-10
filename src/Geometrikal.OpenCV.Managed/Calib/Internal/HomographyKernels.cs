// This implementation is ported from the OpenCV homography estimator callback.
//
// Original sources:
//   modules/geometry/src/fundam.cpp (HomographyEstimatorCallback)
//   modules/core/include/opencv2/core/matx.inl.hpp (3 × 3 determinant)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>HomographyEstimatorCallback</c>: a normalized DLT homography from 4 or more pairs.
/// </summary>
internal readonly struct HomographyKernel : IEstimatorKernel
{
    private static readonly int[] Triples = [0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3];

    public int ModelPoints => 4;

    public int ModelSize => 9;

    public bool CheckSubset(ReadOnlySpan<Point2f> ms1, ReadOnlySpan<Point2f> ms2, int count)
    {
        if (PointSetRegistrator.HaveCollinearPoints(ms1, count) || PointSetRegistrator.HaveCollinearPoints(ms2, count))
        {
            return false;
        }

        if (count == 4)
        {
            int negative = 0;
            for (int i = 0; i < 4; i++)
            {
                int t0 = Triples[3 * i], t1 = Triples[(3 * i) + 1], t2 = Triples[(3 * i) + 2];
                double da = Det3(ms1[t0], ms1[t1], ms1[t2]);
                double db = Det3(ms2[t0], ms2[t1], ms2[t2]);
                negative += da * db < 0 ? 1 : 0;
            }

            if (negative != 0 && negative != 4)
            {
                return false;
            }
        }

        return true;
    }

    public int RunKernel(ReadOnlySpan<Point2f> big, ReadOnlySpan<Point2f> small, Span<double> model)
    {
        // big is the source (OpenCV M), small is the destination (OpenCV m).
        int count = big.Length;
        double cmx = 0, cmy = 0, cMx = 0, cMy = 0, smx = 0, smy = 0, sMx = 0, sMy = 0;
        for (int i = 0; i < count; i++)
        {
            cmx += small[i].X;
            cmy += small[i].Y;
            cMx += big[i].X;
            cMy += big[i].Y;
        }

        cmx /= count;
        cmy /= count;
        cMx /= count;
        cMy /= count;

        for (int i = 0; i < count; i++)
        {
            smx += Math.Abs(small[i].X - cmx);
            smy += Math.Abs(small[i].Y - cmy);
            sMx += Math.Abs(big[i].X - cMx);
            sMy += Math.Abs(big[i].Y - cMy);
        }

        const double DblEpsilon = 2.220446049250313e-16;
        if (Math.Abs(smx) < DblEpsilon || Math.Abs(smy) < DblEpsilon || Math.Abs(sMx) < DblEpsilon || Math.Abs(sMy) < DblEpsilon)
        {
            return 0;
        }

        smx = count / smx;
        smy = count / smy;
        sMx = count / sMx;
        sMy = count / sMy;

        Span<double> ltl = stackalloc double[81];
        ltl.Clear();
        Span<double> lx = stackalloc double[9];
        Span<double> ly = stackalloc double[9];
        lx[2] = 1;
        lx[3] = 0;
        lx[4] = 0;
        lx[5] = 0;
        ly[0] = 0;
        ly[1] = 0;
        ly[2] = 0;
        ly[5] = 1;
        for (int i = 0; i < count; i++)
        {
            double x = (small[i].X - cmx) * smx, y = (small[i].Y - cmy) * smy;
            double bx = (big[i].X - cMx) * sMx, by = (big[i].Y - cMy) * sMy;
            lx[0] = bx;
            lx[1] = by;
            lx[6] = -x * bx;
            lx[7] = -x * by;
            lx[8] = -x;
            ly[3] = bx;
            ly[4] = by;
            ly[6] = -y * bx;
            ly[7] = -y * by;
            ly[8] = -y;
            for (int j = 0; j < 9; j++)
            {
                for (int k = j; k < 9; k++)
                {
                    ltl[(j * 9) + k] += (lx[j] * lx[k]) + (ly[j] * ly[k]);
                }
            }
        }

        for (int j = 0; j < 9; j++)
        {
            for (int k = 0; k < j; k++)
            {
                ltl[(j * 9) + k] = ltl[(k * 9) + j];
            }
        }

        Span<double> w = stackalloc double[9];
        Span<double> v = stackalloc double[81];
        LapackDecomp.Jacobi<double>(ltl, 9, w, v, 9, 9);

        // H0 is the eigenvector of the smallest eigenvalue (row 8); row 7 is the temporary.
        Span<double> h0 = v.Slice(72, 9);
        Span<double> htemp = v.Slice(63, 9);
        ReadOnlySpan<double> invHnorm = [1.0 / smx, 0, cmx, 0, 1.0 / smy, cmy, 0, 0, 1];
        ReadOnlySpan<double> hnorm2 = [sMx, 0, -cMx * sMx, 0, sMy, -cMy * sMy, 0, 0, 1];
        Gemm3x3(invHnorm, h0, htemp);
        Gemm3x3(htemp, hnorm2, h0);

        h0.CopyTo(model);
        PointSetRegistrator.ScaleInPlace(model[..9], PointSetRegistrator.ScaleFor(h0[8]));
        return 1;
    }

    public void ComputeError(ReadOnlySpan<Point2f> big, ReadOnlySpan<Point2f> small, ReadOnlySpan<double> h, Span<float> err)
    {
        float h0 = (float)h[0], h1 = (float)h[1], h2 = (float)h[2];
        float h3 = (float)h[3], h4 = (float)h[4], h5 = (float)h[5];
        float h6 = (float)h[6], h7 = (float)h[7], h8 = (float)h[8];
        for (int i = 0; i < big.Length; i++)
        {
            float x = big[i].X, y = big[i].Y;
            float ww = 1.0f / ((h6 * x) + (h7 * y) + h8);
            float dx = (((h0 * x) + (h1 * y) + h2) * ww) - small[i].X;
            float dy = (((h3 * x) + (h4 * y) + h5) * ww) - small[i].Y;
            err[i] = (dx * dx) + (dy * dy);
        }
    }

    public int FindInliers(ReadOnlySpan<Point2f> big, ReadOnlySpan<Point2f> small, ReadOnlySpan<double> h, float t, Span<byte> mask)
    {
        float h0 = (float)h[0], h1 = (float)h[1], h2 = (float)h[2];
        float h3 = (float)h[3], h4 = (float)h[4], h5 = (float)h[5];
        float h6 = (float)h[6], h7 = (float)h[7], h8 = (float)h[8];
        int n = big.Length, i = 0, nz = 0;
        if (Avx.IsSupported && n >= 4)
        {
            ref float p1 = ref ErrorLanes.Floats(big);
            ref float p2 = ref ErrorLanes.Floats(small);
            var na = Vector256.Create(h0, h4, h0, h4, h0, h4, h0, h4);
            var nb = Vector256.Create(h1, h3, h1, h3, h1, h3, h1, h3);
            var nc = Vector256.Create(h2, h5, h2, h5, h2, h5, h2, h5);
            var da = Vector256.Create(h6, h7, h6, h7, h6, h7, h6, h7);
            var db = Vector256.Create(h7, h6, h7, h6, h7, h6, h7, h6);
            var dc = Vector256.Create(h8);
            var one = Vector256.Create(1.0f);
            var tt = Vector256.Create(t);
            for (; i <= n - 4; i += 4)
            {
                Vector256<float> v = Vector256.LoadUnsafe(ref p1, (nuint)(2 * i));
                Vector256<float> w = Vector256.LoadUnsafe(ref p2, (nuint)(2 * i));
                Vector256<float> s = ErrorLanes.SwapPairs(v);

                // The y lanes add the two products in swapped order; float addition is commutative.
                Vector256<float> ww = one / (((v * da) + (s * db)) + dc);
                Vector256<float> d = ((((v * na) + (s * nb)) + nc) * ww) - w;
                Vector256<float> sq = d * d;
                nz += ErrorLanes.StoreMask(sq + ErrorLanes.SwapPairs(sq), tt, mask, i);
            }
        }

        for (; i < n; i++)
        {
            float x = big[i].X, y = big[i].Y;
            float ww = 1.0f / ((h6 * x) + (h7 * y) + h8);
            float dx = (((h0 * x) + (h1 * y) + h2) * ww) - small[i].X;
            float dy = (((h3 * x) + (h4 * y) + h5) * ww) - small[i].Y;
            int f = (dx * dx) + (dy * dy) <= t ? 1 : 0;
            mask[i] = (byte)f;
            nz += f;
        }

        return nz;
    }

    /// <summary>
    /// <c>gemm</c> of two 3 × 3 <c>double</c> matrices on the small-matrix path:
    /// <c>((a0·b0 + a1·b1) + a2·b2)·1 + 0</c>.
    /// </summary>
    private static void Gemm3x3(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> d)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double t = (a[3 * i] * b[j]) + (a[(3 * i) + 1] * b[3 + j]) + (a[(3 * i) + 2] * b[6 + j]);
                d[(3 * i) + j] = (t * 1.0) + 0.0;
            }
        }
    }

    /// <summary><c>determinant(Matx33d)</c> of the rows (x, y, 1).</summary>
    private static double Det3(Point2f p0, Point2f p1, Point2f p2)
    {
        double a00 = p0.X, a01 = p0.Y, a02 = 1.0;
        double a10 = p1.X, a11 = p1.Y, a12 = 1.0;
        double a20 = p2.X, a21 = p2.Y, a22 = 1.0;
        return (a00 * ((a11 * a22) - (a21 * a12)))
            - (a01 * ((a10 * a22) - (a20 * a12)))
            + (a02 * ((a10 * a21) - (a20 * a11)));
    }
}
