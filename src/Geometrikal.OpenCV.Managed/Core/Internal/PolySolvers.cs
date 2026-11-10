// This implementation is ported from the OpenCV polynomial solvers.
//
// Original source:
//   modules/core/src/mathfuncs.cpp (solveCubic, solvePoly)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/mathfuncs.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <c>solveCubic</c> and <c>solvePoly</c> (Durand-Kerner) in OpenCV operation order. Complex
/// values are (re, im) pairs with the <c>cv::Complex&lt;double&gt;</c> operators.
/// </summary>
internal static class PolySolvers
{
    /// <summary>
    /// <c>solveCubic</c> on the coefficients a0..a3 (a0 = 1 for three coefficients). Writes three
    /// roots and returns the root count (-1 for infinitely many).
    /// </summary>
    public static int SolveCubic(double a0, double a1, double a2, double a3, out double x0, out double x1, out double x2)
    {
        const double dblEps = 2.220446049250313e-16;
        int n = 0;
        x0 = x1 = x2 = 0;

        double maxCoeff = Math.Max(Math.Max(Math.Abs(a0), Math.Abs(a1)), Math.Max(Math.Abs(a2), Math.Abs(a3)));
        if (maxCoeff < dblEps)
        {
            return -1;
        }

        if (Math.Abs(a0) < dblEps * maxCoeff)
        {
            if (Math.Abs(a1) < dblEps * maxCoeff)
            {
                if (Math.Abs(a2) < dblEps * maxCoeff)
                {
                    n = Math.Abs(a3) < dblEps * maxCoeff ? -1 : 0;
                }
                else
                {
                    x0 = -a3 / a2;
                    n = 1;
                }
            }
            else
            {
                double d = (a2 * a2) - (4 * a1 * a3);
                if (d >= 0)
                {
                    d = Math.Sqrt(d);
                    double q1 = (-a2 + d) * 0.5;
                    double q2 = (a2 + d) * -0.5;
                    if (Math.Abs(q1) > Math.Abs(q2))
                    {
                        x0 = q1 / a1;
                        x1 = a3 / q1;
                    }
                    else
                    {
                        x0 = q2 / a1;
                        x1 = a3 / q2;
                    }

                    n = d > 0 ? 2 : 1;
                }
            }
        }
        else
        {
            double scale = 1.0 / maxCoeff;
            a0 *= scale;
            a1 *= scale;
            a2 *= scale;
            a3 *= scale;

            a0 = 1.0 / a0;
            a1 *= a0;
            a2 *= a0;
            a3 *= a0;

            double q = ((a1 * a1) - (3 * a2)) * (1.0 / 9);
            double r = ((a1 * ((2 * a1 * a1) - (9 * a2))) + (27 * a3)) * (1.0 / 54);
            double qcubed = q * q * q;
            double d = ((a1 * a1 * ((a2 * a2) - (4 * a1 * a3))) + (2 * a2 * ((9 * a1 * a3) - (2 * a2 * a2))) - (27 * a3 * a3)) * (1.0 / 108);

            if (d > 0)
            {
                double theta = Math.Acos(r / Math.Sqrt(qcubed));
                double sqrtQ = Math.Sqrt(q);
                double t0 = -2 * sqrtQ;
                double t1 = theta * (1.0 / 3);
                double t2 = a1 * (1.0 / 3);
                x0 = (t0 * Math.Cos(t1)) - t2;
                x1 = (t0 * Math.Cos(t1 + (2.0 * Math.PI / 3))) - t2;
                x2 = (t0 * Math.Cos(t1 + (4.0 * Math.PI / 3))) - t2;
                n = 3;
            }
            else if (d == 0)
            {
                x0 = (-2 * Math.Cbrt(r)) - (a1 / 3);
                x1 = Math.Cbrt(r) - (a1 / 3);
                x2 = 0;
                n = x0 == x1 ? 1 : 2;
                x1 = x0 == x1 ? 0 : x1;
            }
            else
            {
                d = Math.Sqrt(-d);
                double e = Math.Cbrt(d + Math.Abs(r));
                if (r > 0)
                {
                    e = -e;
                }

                x0 = (e + (q / e)) - (a1 * (1.0 / 3));
                n = 1;
            }
        }

        return n;
    }

    /// <summary>
    /// <c>solvePoly</c>: roots of <c>Σ coeffs[i] xⁱ</c> (complex coefficients as re/im pairs,
    /// index 0 is the constant term). Writes <c>coeffs.Length/2 - 1</c> complex roots and returns
    /// the last maximum root change.
    /// </summary>
    public static double SolvePoly(ReadOnlySpan<double> coeffsReIm, Span<double> rootsReIm, int maxIters, bool realInput)
    {
        const double dblEps = 2.220446049250313e-16;
        int n0 = (coeffsReIm.Length / 2) - 1, n = n0;
        double[] cre = new double[n0 + 1], cim = new double[n0 + 1];
        for (int i = 0; i <= n0; i++)
        {
            cre[i] = coeffsReIm[2 * i];
            cim[i] = coeffsReIm[(2 * i) + 1];
        }

        double[] rre = new double[n0 + 1], rim = new double[n0 + 1];

        for (; n > 1; n--)
        {
            if (Math.Abs(cre[n]) + Math.Abs(cim[n]) > dblEps)
            {
                break;
            }
        }

        double[] absCoeffs = new double[n + 1];
        for (int i = 0; i <= n; i++)
        {
            absCoeffs[i] = Math.Sqrt((cre[i] * cre[i]) + (cim[i] * cim[i]));
        }

        int nonZeroCoeffs = 0;
        double[] u = new double[n], v = new double[n];
        for (int i = 0; i <= n; i++)
        {
            double coeff = absCoeffs[i];
            if (coeff > dblEps)
            {
                if (i != n)
                {
                    u[i] = 2.0 * Math.Pow(coeff / absCoeffs[n], 1.0 / (n - i));
                }

                if (i != 0)
                {
                    v[i - 1] = 0.5 * Math.Pow(absCoeffs[0] / coeff, 1.0 / i);
                }

                nonZeroCoeffs++;
            }
        }

        double scale = 1;
        if (nonZeroCoeffs > 2)
        {
            int maxU = 0, minV = 0;
            for (int i = 1; i < n; i++)
            {
                if (u[i] > u[maxU])
                {
                    maxU = i;
                }

                if (v[i] < v[minV])
                {
                    minV = i;
                }
            }

            u[maxU] = 0;
            v[minV] = 0;
            scale = (CvSum(u) + CvSum(v)) / ((2 * nonZeroCoeffs) - 2);
        }

        double pre = scale, pim = 0;
        double rcre = Math.Cos(2 * Math.PI / n), rcim = Math.Sin(2 * Math.PI / n);
        for (int i = 0; i < n; i++)
        {
            rre[i] = pre;
            rim[i] = pim;
            (pre, pim) = ((pre * rcre) - (pim * rcim), (pre * rcim) + (pim * rcre));
        }

        maxIters = maxIters <= 0 ? 1000 : maxIters;
        double maxDiff = 0;
        for (int iter = 0; iter < maxIters; iter++)
        {
            maxDiff = 0;
            for (int i = 0; i < n; i++)
            {
                pre = rre[i];
                pim = rim[i];
                double nre = cre[n], nim = cim[n], dre = cre[n], dim = cim[n];
                int numSameRoot = 1;
                for (int j = 0; j < n; j++)
                {
                    (nre, nim) = ((nre * pre) - (nim * pim) + cre[n - j - 1], (nre * pim) + (nim * pre) + cim[n - j - 1]);
                    if (j != i)
                    {
                        double ddre = pre - rre[j], ddim = pim - rim[j];
                        if (ddre != 0 || ddim != 0)
                        {
                            (dre, dim) = ((dre * ddre) - (dim * ddim), (dre * ddim) + (dim * ddre));
                        }
                        else
                        {
                            numSameRoot++;
                        }
                    }
                }

                double t = 1.0 / ((dre * dre) + (dim * dim));
                (nre, nim) = (((nre * dre) + (nim * dim)) * t, ((-nre * dim) + (nim * dre)) * t);

                if (numSameRoot > 1)
                {
                    double oldRe = nre, oldIm = nim;
                    int squareRootTimes = numSameRoot % 2 == 0 ? numSameRoot / 2 : (numSameRoot / 2) - 1;
                    for (int j = 0; j < squareRootTimes; j++)
                    {
                        nre = (oldRe * oldRe) + (oldIm * oldIm);
                        nre = Math.Sqrt(nre);
                        nre += oldRe;
                        nim = nre - oldRe;
                        nre /= 2;
                        nre = Math.Sqrt(nre);
                        nim /= 2;
                        nim = Math.Sqrt(nim);
                        if (oldRe < 0)
                        {
                            nim = -nim;
                        }
                    }

                    if (numSameRoot % 2 != 0)
                    {
                        SolveCubic(
                            64,
                            -48 * oldRe,
                            -((15 * Math.Pow(oldRe, 2)) + (27 * Math.Pow(oldIm, 2))),
                            -Math.Pow(oldRe, 3),
                            out double cr0,
                            out _,
                            out _);
                        nre = Math.Cbrt(cr0);
                        nim = Math.Sqrt((Math.Pow(nre, 2) / 3) - (oldRe / (3 * nre)));
                    }
                }

                rre[i] = pre - nre;
                rim[i] = pim - nim;
                maxDiff = Math.Max(maxDiff, Math.Sqrt((nre * nre) + (nim * nim)));
            }

            if (maxDiff <= 0)
            {
                break;
            }
        }

        if (realInput)
        {
            const double verySmallEps = 1e-100;
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(rim[i]) < verySmallEps)
                {
                    rim[i] = 0;
                }
            }
        }

        for (; n < n0; n++)
        {
            rre[n + 1] = rre[n];
            rim[n + 1] = rim[n];
        }

        for (int i = 0; i < n0; i++)
        {
            rootsReIm[2 * i] = rre[i];
            rootsReIm[(2 * i) + 1] = rim[i];
        }

        return maxDiff;
    }

    /// <summary><c>cv::sum</c> of an n × 1 double column.</summary>
    private static double CvSum(double[] values)
        => Reductions.Sum(new Mat<double>(values, 1, values.Length)).V0;
}
