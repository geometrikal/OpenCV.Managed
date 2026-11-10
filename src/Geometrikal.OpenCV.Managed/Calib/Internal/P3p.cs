// This implementation is ported from the OpenCV P3P solver (Ding, Shachkov; from PoseLib, Viktor Larsson).
//
// Original sources:
//   modules/geometry/src/p3p.cpp (p3p::estimate, p3p::calibrateAndNormalizePointsPnP,
//     yaqding::solve_cubic_single_real, yaqding::root2real, yaqding::compute_pq, yaqding::refine_lambda)
//   modules/geometry/src/p3p.h
//   modules/core/include/opencv2/core/matx.inl.hpp, operations.hpp (Matx dot, cross, product, 3x3 inverse)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).
// The P3P solver code is Copyright (c) 2020, Viktor Larsson, under the BSD 3-clause license in p3p.cpp.

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// P3P pose solver (<c>cv::p3p</c>, <c>SOLVEPNP_P3P</c>).
/// </summary>
internal static class P3p
{
    /// <summary>
    /// Calculates the camera poses that project 3 object points to 3 normalized image points (<c>p3p::estimate</c>).
    /// </summary>
    /// <param name="objectPoints">The object points. The solver uses only the first 3 points.</param>
    /// <param name="normalizedPoints">The normalized (undistorted, no camera matrix) image points. The solver uses only the first 3 points.</param>
    /// <param name="rotations">The output rotation matrices (row-major, 9 elements). The solver appends one matrix for each solution.</param>
    /// <param name="translations">The output translation vectors (3 elements). The solver appends one vector for each solution.</param>
    /// <returns>The number of solutions: 0 to 4.</returns>
    public static int Estimate(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> normalizedPoints,
        List<double[]> rotations,
        List<double[]> translations)
    {
        // calibrateAndNormalizePointsPnP
        Span<double> xc = stackalloc double[9];
        for (int i = 0; i < 3; i++)
        {
            double kInvU = normalizedPoints[i].X;
            double kInvV = normalizedPoints[i].Y;
            double xNorm = 1.0 / Math.Sqrt(kInvU * kInvU + kInvV * kInvV + 1);
            xc[i * 3] = kInvU * xNorm;
            xc[i * 3 + 1] = kInvV * xNorm;
            xc[i * 3 + 2] = xNorm;
        }

        Span<double> xw = stackalloc double[9];
        for (int i = 0; i < 3; i++)
        {
            xw[i * 3] = objectPoints[i].X;
            xw[i * 3 + 1] = objectPoints[i].Y;
            xw[i * 3 + 2] = objectPoints[i].Z;
        }

        // X01, X02, X12 = X_copy[i] - X_copy[j].
        double x01x = xw[0] - xw[3], x01y = xw[1] - xw[4], x01z = xw[2] - xw[5];
        double x02x = xw[0] - xw[6], x02y = xw[1] - xw[7], x02z = xw[2] - xw[8];
        double x12x = xw[3] - xw[6], x12y = xw[4] - xw[7], x12z = xw[5] - xw[8];

        double a01 = NormL2Sqr(x01x, x01y, x01z);
        double a02 = NormL2Sqr(x02x, x02y, x02z);
        double a12 = NormL2Sqr(x12x, x12y, x12z);

        // Index of the point in slot k after the swaps.
        int i0 = 0, i1 = 1, i2 = 2;

        if (a01 > a02)
        {
            if (a01 > a12)
            {
                (i0, i2) = (i2, i0);
                (a01, a12) = (a12, a01);
                x01x = -x12x;
                x01y = -x12y;
                x01z = -x12z;
                x02x = -x02x;
                x02y = -x02y;
                x02z = -x02z;
            }
        }
        else if (a02 > a12)
        {
            (i0, i1) = (i1, i0);
            (a02, a12) = (a12, a02);
            x01x = -x01x;
            x01y = -x01y;
            x01z = -x01z;
            x02x = x12x;
            x02y = x12y;
            x02z = x12z;
        }

        ReadOnlySpan<double> b0 = xc.Slice(i0 * 3, 3);
        ReadOnlySpan<double> b1 = xc.Slice(i1 * 3, 3);
        ReadOnlySpan<double> b2 = xc.Slice(i2 * 3, 3);
        ReadOnlySpan<double> w0p = xw.Slice(i0 * 3, 3);

        double a12d = 1.0 / a12;
        double a = a01 * a12d;
        double b = a02 * a12d;

        double m01 = Dot(b0, b1);
        double m02 = Dot(b0, b2);
        double m12 = Dot(b1, b2);

        double m12sq = -m12 * m12 + 1.0;
        double m02sq = -1.0 + m02 * m02;
        double m01sq = -1.0 + m01 * m01;
        double ab = a * b;
        double bsq = b * b;
        double asq = a * a;
        double m013 = -2.0 + 2.0 * m01 * m02 * m12;
        double bsqm12sq = bsq * m12sq;
        double asqm12sq = asq * m12sq;
        double abm12sq = 2.0 * ab * m12sq;

        double k3Inv = 1.0 / (bsqm12sq + b * m02sq);
        double k2 = k3Inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
        double k1 = k3Inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
        double k0 = k3Inv * (asqm12sq + a * m01sq);

        bool g = SolveCubicSingleReal(k2, k1, k0, out double s);

        Span<double> c = stackalloc double[9];
        c[0] = -a + s * (1 - b);
        c[1] = -m02 * s;
        c[2] = a * m12 + b * m12 * s;
        c[3] = c[1];
        c[4] = s + 1;
        c[5] = -m01;
        c[6] = c[2];
        c[7] = c[5];
        c[8] = -a - b * s + 1;

        Span<double> pq = stackalloc double[6];
        ComputePq(c, pq);

        // XX = [X01, X02, X01 x X02] (columns), then XX = XX.inv().
        double crx = x01y * x02z - x01z * x02y;
        double cry = x01z * x02x - x01x * x02z;
        double crz = x01x * x02y - x01y * x02x;
        Span<double> xx = stackalloc double[9];
        xx[0] = x01x; xx[3] = x01y; xx[6] = x01z;
        xx[1] = x02x; xx[4] = x02y; xx[7] = x02z;
        xx[2] = crx; xx[5] = cry; xx[8] = crz;
        Span<double> xxInv = stackalloc double[9];
        Invert3x3(xx, xxInv);

        Span<double> taus = stackalloc double[2];
        int nSols = 0;
        for (int i = 0; i < 2; ++i)
        {
            double p0 = pq[i * 3];
            double p1 = pq[i * 3 + 1];
            double p2 = pq[i * 3 + 2];
            bool switch12 = Math.Abs(p0) <= Math.Abs(p1);

            if (switch12)
            {
                double w0 = -p0 / p1;
                double w1 = -p2 / p1;
                double ca = 1.0 / (w1 * w1 - b);
                double cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
                double cc = (w0 * w0 - 2 * m02 * w0 - b + 1.0) * ca;

                if (!Root2Real(cb, cc, out taus[0], out taus[1]))
                {
                    continue;
                }

                for (int t = 0; t < 2; t++)
                {
                    double tau = taus[t];
                    if (tau <= 0)
                    {
                        continue;
                    }

                    double d2 = Math.Sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
                    double d1 = tau * d2;
                    double d0 = (w0 * d2 + w1 * d1);
                    if (d0 < 0)
                    {
                        continue;
                    }

                    RefineLambda(ref d0, ref d1, ref d2, a01, a02, a12, m01, m02, m12);
                    AddSolution(d0, d1, d2, b0, b1, b2, w0p, xxInv, rotations, translations);
                    ++nSols;
                }
            }
            else
            {
                double w0 = -p1 / p0;
                double w1 = -p2 / p0;
                double ca = 1.0 / (-a * w1 * w1 + 2 * a * m12 * w1 - a + 1);
                double cb = 2 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
                double cc = (1 - a * w0 * w0) * ca;

                if (!Root2Real(cb, cc, out taus[0], out taus[1]))
                {
                    continue;
                }

                for (int t = 0; t < 2; t++)
                {
                    double tau = taus[t];
                    if (tau <= 0)
                    {
                        continue;
                    }

                    double d0 = Math.Sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
                    double d1 = tau * d0;
                    double d2 = w0 * d0 + w1 * d1;

                    if (d2 < 0)
                    {
                        continue;
                    }

                    RefineLambda(ref d0, ref d1, ref d2, a01, a02, a12, m01, m02, m12);
                    AddSolution(d0, d1, d2, b0, b1, b2, w0p, xxInv, rotations, translations);
                    ++nSols;
                }
            }

            if (nSols > 0 && g)
            {
                break;
            }
        }

        return nSols;
    }

    /// <summary>
    /// Calculates <c>R = YY * XX</c> and <c>t = d0 * x[0] - R * X[0]</c> and appends them to the outputs.
    /// </summary>
    private static void AddSolution(
        double d0,
        double d1,
        double d2,
        ReadOnlySpan<double> x0,
        ReadOnlySpan<double> x1,
        ReadOnlySpan<double> x2,
        ReadOnlySpan<double> w0,
        ReadOnlySpan<double> xx,
        List<double[]> rotations,
        List<double[]> translations)
    {
        double d0x0 = x0[0] * d0, d0x1 = x0[1] * d0, d0x2 = x0[2] * d0;
        double v1x = d0x0 - x1[0] * d1, v1y = d0x1 - x1[1] * d1, v1z = d0x2 - x1[2] * d1;
        double v2x = d0x0 - x2[0] * d2, v2y = d0x1 - x2[1] * d2, v2z = d0x2 - x2[2] * d2;
        double crx = v1y * v2z - v1z * v2y;
        double cry = v1z * v2x - v1x * v2z;
        double crz = v1x * v2y - v1y * v2x;

        // YY rows: (v1x, v2x, crx), (v1y, v2y, cry), (v1z, v2z, crz).
        var r = new double[9];
        for (int j = 0; j < 3; j++)
        {
            r[j] = 0.0 + v1x * xx[j] + v2x * xx[3 + j] + crx * xx[6 + j];
            r[3 + j] = 0.0 + v1y * xx[j] + v2y * xx[3 + j] + cry * xx[6 + j];
            r[6 + j] = 0.0 + v1z * xx[j] + v2z * xx[3 + j] + crz * xx[6 + j];
        }

        var t = new double[3];
        for (int k = 0; k < 3; k++)
        {
            double rx = 0.0 + r[k * 3] * w0[0] + r[k * 3 + 1] * w0[1] + r[k * 3 + 2] * w0[2];
            t[k] = (k == 0 ? d0x0 : k == 1 ? d0x1 : d0x2) - rx;
        }

        rotations.Add(r);
        translations.Add(t);
    }

    private static double NormL2Sqr(double x, double y, double z)
    {
        double s = 0.0;
        s += x * x;
        s += y * y;
        s += z * z;
        return s;
    }

    private static double Dot(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        double s = 0.0;
        s += u[0] * v[0];
        s += u[1] * v[1];
        s += u[2] * v[2];
        return s;
    }

    /// <summary>
    /// <c>Matx33d::inv</c> (<c>Matx_FastInvOp&lt;3, 3&gt;</c>). The result is zero when the determinant is zero.
    /// </summary>
    private static void Invert3x3(ReadOnlySpan<double> a, Span<double> b)
    {
        double d = a[0] * (a[4] * a[8] - a[7] * a[5]) -
            a[1] * (a[3] * a[8] - a[6] * a[5]) +
            a[2] * (a[3] * a[7] - a[6] * a[4]);
        if (d == 0)
        {
            b.Clear();
            return;
        }

        d = 1 / d;
        b[0] = (a[4] * a[8] - a[5] * a[7]) * d;
        b[1] = (a[2] * a[7] - a[1] * a[8]) * d;
        b[2] = (a[1] * a[5] - a[2] * a[4]) * d;

        b[3] = (a[5] * a[6] - a[3] * a[8]) * d;
        b[4] = (a[0] * a[8] - a[2] * a[6]) * d;
        b[5] = (a[2] * a[3] - a[0] * a[5]) * d;

        b[6] = (a[3] * a[7] - a[4] * a[6]) * d;
        b[7] = (a[1] * a[6] - a[0] * a[7]) * d;
        b[8] = (a[0] * a[4] - a[1] * a[3]) * d;
    }

    private static bool SolveCubicSingleReal(double c2, double c1, double c0, out double root)
    {
        double a = c1 - c2 * c2 / 3.0;
        double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
        double c = b * b / 4.0 + a * a * a / 27.0;
        if (c != 0)
        {
            if (c > 0)
            {
                c = Math.Sqrt(c);
                b *= -0.5;
                root = Math.Cbrt(b + c) + Math.Cbrt(b - c) - c2 / 3.0;
                return true;
            }

            c = 3.0 * b / (2.0 * a) * Math.Sqrt(-3.0 / a);
            root = 2.0 * Math.Sqrt(-a / 3.0) * Math.Cos(Math.Acos(c) / 3.0) - c2 / 3.0;
        }
        else
        {
            root = -c2 / 3.0 + (a != 0 ? (3.0 * b / a) : 0);
        }

        return false;
    }

    private static bool Root2Real(double b, double c, out double r1, out double r2)
    {
        const double Threshold = -1.0e-12;
        double v = b * b - 4.0 * c;
        if (v < Threshold)
        {
            r1 = r2 = -0.5 * b;
            return v >= 0;
        }

        if (v > Threshold && v < 0.0)
        {
            r1 = -0.5 * b;
            r2 = -2;
            return true;
        }

        double y = Math.Sqrt(v);
        if (b < 0)
        {
            r1 = 0.5 * (-b + y);
            r2 = 0.5 * (-b - y);
        }
        else
        {
            r1 = 2.0 * c / (-b + y);
            r2 = 2.0 * c / (-b - y);
        }

        return true;
    }

    /// <summary>
    /// <c>yaqding::compute_pq</c>. <paramref name="c"/> is a row-major 3x3 matrix. The method changes <paramref name="c"/>.
    /// </summary>
    private static void ComputePq(Span<double> c, Span<double> pq)
    {
        double adj00 = c[5] * c[7] - c[4] * c[8];
        double adj11 = c[2] * c[6] - c[0] * c[8];
        double adj22 = c[1] * c[3] - c[0] * c[4];
        double adj01 = c[1] * c[8] - c[2] * c[7];
        double adj02 = c[2] * c[4] - c[1] * c[5];
        double adj10 = adj01;
        double adj12 = c[0] * c[5] - c[2] * c[3];
        double adj20 = adj02;
        double adj21 = adj12;

        double v0, v1, v2, scale;
        if (adj00 > adj11)
        {
            if (adj00 > adj22)
            {
                scale = 1.0 / Math.Sqrt(adj00);
                v0 = adj00 * scale; v1 = adj10 * scale; v2 = adj20 * scale;
            }
            else
            {
                scale = 1.0 / Math.Sqrt(adj22);
                v0 = adj02 * scale; v1 = adj12 * scale; v2 = adj22 * scale;
            }
        }
        else if (adj11 > adj22)
        {
            scale = 1.0 / Math.Sqrt(adj11);
            v0 = adj01 * scale; v1 = adj11 * scale; v2 = adj21 * scale;
        }
        else
        {
            scale = 1.0 / Math.Sqrt(adj22);
            v0 = adj02 * scale; v1 = adj12 * scale; v2 = adj22 * scale;
        }

        c[1] -= v2;
        c[2] += v1;
        c[5] -= v0;
        c[3] += v2;
        c[6] -= v1;
        c[7] += v0;

        pq[0] = c[0];
        pq[1] = c[3];
        pq[2] = c[6];
        pq[3] = c[0];
        pq[4] = c[1];
        pq[5] = c[2];
    }

    private static void RefineLambda(
        ref double lambda1,
        ref double lambda2,
        ref double lambda3,
        double a12,
        double a13,
        double a23,
        double b12,
        double b13,
        double b23)
    {
        for (int iter = 0; iter < 5; ++iter)
        {
            double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
            double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
            double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);

            if (Math.Abs(r1) + Math.Abs(r2) + Math.Abs(r3) < 1e-10)
            {
                return;
            }

            double x11 = lambda1 - lambda2 * b12;
            double x12 = lambda2 - lambda1 * b12;
            double x21 = lambda1 - lambda3 * b13;
            double x23 = lambda3 - lambda1 * b13;
            double x32 = lambda2 - lambda3 * b23;
            double x33 = lambda3 - lambda2 * b23;
            double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33);

            lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
            lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
            lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
        }
    }
}
