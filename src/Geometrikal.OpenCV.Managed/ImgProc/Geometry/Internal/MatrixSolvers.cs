// This implementation is ported from the OpenCV dense linear algebra.
//
// Original sources:
//   modules/core/src/matrix_decomp.cpp (LUImpl)
//   modules/core/src/lapack.cpp (JacobiSVDImpl_, _SVDcompute)
//   modules/core/src/matmul.simd.hpp (MulTransposedR)
//   modules/core/include/opencv2/core/operations.hpp (RNG::next)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>Small dense solvers for the transform-matrix helpers, in OpenCV operation order.</summary>
internal static class MatrixSolvers
{
    private const double DblEpsilon = 2.220446049250313e-16;
    private const double DblMin = 2.2250738585072014e-308;

    /// <summary>
    /// <c>LUImpl&lt;double&gt;</c> with one right-hand side: Gaussian elimination with partial
    /// pivoting on the row-major <paramref name="a"/> (m × m) and back substitution into
    /// <paramref name="b"/>. Returns false, and leaves <paramref name="b"/> undefined, when a pivot is
    /// below <c>100 * DBL_EPSILON</c>.
    /// </summary>
    public static bool LuSolve(Span<double> a, Span<double> b, int m)
    {
        const double eps = DblEpsilon * 100;
        for (int i = 0; i < m; i++)
        {
            int k = i;
            for (int j = i + 1; j < m; j++)
            {
                if (Math.Abs(a[(j * m) + i]) > Math.Abs(a[(k * m) + i]))
                {
                    k = j;
                }
            }

            if (Math.Abs(a[(k * m) + i]) < eps)
            {
                return false;
            }

            if (k != i)
            {
                for (int j = i; j < m; j++)
                {
                    (a[(i * m) + j], a[(k * m) + j]) = (a[(k * m) + j], a[(i * m) + j]);
                }

                (b[i], b[k]) = (b[k], b[i]);
            }

            double d = -1 / a[(i * m) + i];
            for (int j = i + 1; j < m; j++)
            {
                double alpha = a[(j * m) + i] * d;
                for (int c = i + 1; c < m; c++)
                {
                    a[(j * m) + c] += alpha * a[(i * m) + c];
                }

                b[j] += alpha * b[i];
            }
        }

        for (int i = m - 1; i >= 0; i--)
        {
            double s = b[i];
            for (int c = i + 1; c < m; c++)
            {
                s -= a[(i * m) + c] * b[c];
            }

            b[i] = s / a[(i * m) + i];
        }

        return true;
    }

    /// <summary><c>mulTransposed(a, dst, true)</c>: <c>dst = aᵀ a</c> for a row-major rows × cols matrix.</summary>
    public static void MulTransposed(ReadOnlySpan<double> a, int rows, int cols, Span<double> dst)
    {
        for (int i = 0; i < cols; i++)
        {
            for (int j = i; j < cols; j++)
            {
                double s = 0;
                for (int k = 0; k < rows; k++)
                {
                    s += a[(k * cols) + i] * a[(k * cols) + j];
                }

                dst[(i * cols) + j] = s;
                dst[(j * cols) + i] = s;
            }
        }
    }

    /// <summary>
    /// Left singular vectors of a symmetric n × n matrix, as OpenCV <c>SVDecomp(a, w, u)</c>:
    /// <c>JacobiSVDImpl_&lt;double&gt;</c> on <c>aᵀ</c> (equal to <paramref name="a"/>). On return,
    /// row <c>i</c> of <paramref name="a"/> holds column <c>i</c> of <c>u</c>, sorted by
    /// decreasing singular value.
    /// </summary>
    public static void SymmetricSvdVectors(Span<double> a, int n)
    {
        const double eps = DblEpsilon * 10;
        int m = n;
        Span<double> w = stackalloc double[n];
        int maxIter = Math.Max(m, 30);

        for (int i = 0; i < n; i++)
        {
            double sd = 0;
            for (int k = 0; k < m; k++)
            {
                double t = a[(i * m) + k];
                sd += t * t;
            }

            w[i] = sd;
        }

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;
            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    Span<double> ai = a.Slice(i * m, m);
                    Span<double> aj = a.Slice(j * m, m);
                    double wa = w[i], p = 0, wb = w[j];
                    for (int k = 0; k < m; k++)
                    {
                        p += ai[k] * aj[k];
                    }

                    if (Math.Abs(p) <= eps * Math.Sqrt(wa * wb))
                    {
                        continue;
                    }

                    p *= 2;
                    double beta = wa - wb, gamma = double.Hypot(p, beta);
                    double c, s;
                    if (beta < 0)
                    {
                        double delta = (gamma - beta) * 0.5;
                        s = Math.Sqrt(delta / gamma);
                        c = p / (gamma * s * 2);
                    }
                    else
                    {
                        c = Math.Sqrt((gamma + beta) / (gamma * 2));
                        s = p / (gamma * c * 2);
                    }

                    wa = wb = 0;
                    for (int k = 0; k < m; k++)
                    {
                        double t0 = (c * ai[k]) + (s * aj[k]);
                        double t1 = (-s * ai[k]) + (c * aj[k]);
                        ai[k] = t0;
                        aj[k] = t1;
                        wa += t0 * t0;
                        wb += t1 * t1;
                    }

                    w[i] = wa;
                    w[j] = wb;
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }
        }

        for (int i = 0; i < n; i++)
        {
            double sd = 0;
            for (int k = 0; k < m; k++)
            {
                double t = a[(i * m) + k];
                sd += t * t;
            }

            w[i] = Math.Sqrt(sd);
        }

        for (int i = 0; i < n - 1; i++)
        {
            int j = i;
            for (int k = i + 1; k < n; k++)
            {
                if (w[j] < w[k])
                {
                    j = k;
                }
            }

            if (i != j)
            {
                (w[i], w[j]) = (w[j], w[i]);
                for (int k = 0; k < m; k++)
                {
                    (a[(i * m) + k], a[(j * m) + k]) = (a[(j * m) + k], a[(i * m) + k]);
                }
            }
        }

        ulong rng = 0x12345678;
        for (int i = 0; i < n; i++)
        {
            double sd = w[i];
            for (int ii = 0; ii < 100 && sd <= DblMin; ii++)
            {
                double val0 = 1.0 / m;
                for (int k = 0; k < m; k++)
                {
                    a[(i * m) + k] = (NextRng(ref rng) & 256) != 0 ? val0 : -val0;
                }

                for (int iter = 0; iter < 2; iter++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        sd = 0;
                        for (int k = 0; k < m; k++)
                        {
                            sd += a[(i * m) + k] * a[(j * m) + k];
                        }

                        double asum = 0;
                        for (int k = 0; k < m; k++)
                        {
                            double t = a[(i * m) + k] - (sd * a[(j * m) + k]);
                            a[(i * m) + k] = t;
                            asum += Math.Abs(t);
                        }

                        asum = asum > eps * 100 ? 1 / asum : 0;
                        for (int k = 0; k < m; k++)
                        {
                            a[(i * m) + k] *= asum;
                        }
                    }
                }

                sd = 0;
                for (int k = 0; k < m; k++)
                {
                    double t = a[(i * m) + k];
                    sd += t * t;
                }

                sd = Math.Sqrt(sd);
            }

            double scale = sd > DblMin ? 1 / sd : 0.0;
            for (int k = 0; k < m; k++)
            {
                a[(i * m) + k] *= scale;
            }
        }
    }

    /// <summary>OpenCV <c>RNG::next</c> (multiply-with-carry).</summary>
    private static uint NextRng(ref ulong state)
    {
        state = unchecked(((ulong)(uint)state * 4164903690UL) + (uint)(state >> 32));
        return (uint)state;
    }
}
