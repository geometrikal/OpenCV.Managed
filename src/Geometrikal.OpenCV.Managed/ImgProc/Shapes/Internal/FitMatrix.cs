// This implementation is ported from the OpenCV dense linear algebra.
//
// Original sources:
//   modules/core/src/lapack.cpp (JacobiSVDImpl_, SVBkSbImpl_, _SVDcompute, solve with DECOMP_SVD,
//   determinant)
//   modules/core/src/matrix_decomp.cpp (LUImpl)
//   modules/core/src/lda.cpp (EigenvalueDecomposition, eigenNonSymmetric)
//   modules/core/src/matrix_operations.cpp (sortIdx_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Small dense double-precision solvers for the ellipse fits, in OpenCV operation order. The
/// SVD is the OpenCV Jacobi SVD. OpenCV builds that use LAPACK replace it for matrices with
/// 25 or more rows (see the shape-fitting notes).
/// </summary>
internal static class FitMatrix
{
    private const double DblEpsilon = 2.220446049250313e-16;
    private const double DblMin = 2.2250738585072014e-308;
    private const double FltEpsilon = 1.1920928955078125e-7;

    /// <summary>
    /// <c>JacobiSVDImpl_&lt;double&gt;</c>. <paramref name="at"/> holds <c>Aᵀ</c> (n rows of m).
    /// On return its first n1 rows hold the left singular vectors, <paramref name="w"/> the
    /// singular values in decreasing order and <paramref name="vt"/> (n × n) the right singular
    /// vectors as rows.
    /// </summary>
    public static void JacobiSvd(Span<double> at, Span<double> w, Span<double> vt, int m, int n, int n1)
    {
        const double minval = DblMin;
        const double eps = DblEpsilon * 10;
        Span<double> wd = stackalloc double[n];
        int maxIter = Math.Max(m, 30);

        for (int i = 0; i < n; i++)
        {
            double sd = 0;
            for (int k = 0; k < m; k++)
            {
                double t = at[(i * m) + k];
                sd += t * t;
            }

            wd[i] = sd;
            for (int k = 0; k < n; k++)
            {
                vt[(i * n) + k] = 0;
            }

            vt[(i * n) + i] = 1;
        }

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;
            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    Span<double> ai = at.Slice(i * m, m);
                    Span<double> aj = at.Slice(j * m, m);
                    double a = wd[i], p = 0, b = wd[j];
                    for (int k = 0; k < m; k++)
                    {
                        p += ai[k] * aj[k];
                    }

                    if (Math.Abs(p) <= eps * Math.Sqrt(a * b))
                    {
                        continue;
                    }

                    p *= 2;
                    double beta = a - b, gamma = double.Hypot(p, beta);
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

                    a = b = 0;
                    for (int k = 0; k < m; k++)
                    {
                        double t0 = (c * ai[k]) + (s * aj[k]);
                        double t1 = (-s * ai[k]) + (c * aj[k]);
                        ai[k] = t0;
                        aj[k] = t1;
                        a += t0 * t0;
                        b += t1 * t1;
                    }

                    wd[i] = a;
                    wd[j] = b;
                    changed = true;

                    Span<double> vi = vt.Slice(i * n, n);
                    Span<double> vj = vt.Slice(j * n, n);
                    for (int k = 0; k < n; k++)
                    {
                        double t0 = (c * vi[k]) + (s * vj[k]);
                        double t1 = (-s * vi[k]) + (c * vj[k]);
                        vi[k] = t0;
                        vj[k] = t1;
                    }
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
                double t = at[(i * m) + k];
                sd += t * t;
            }

            wd[i] = Math.Sqrt(sd);
        }

        for (int i = 0; i < n - 1; i++)
        {
            int j = i;
            for (int k = i + 1; k < n; k++)
            {
                if (wd[j] < wd[k])
                {
                    j = k;
                }
            }

            if (i != j)
            {
                (wd[i], wd[j]) = (wd[j], wd[i]);
                for (int k = 0; k < m; k++)
                {
                    (at[(i * m) + k], at[(j * m) + k]) = (at[(j * m) + k], at[(i * m) + k]);
                }

                for (int k = 0; k < n; k++)
                {
                    (vt[(i * n) + k], vt[(j * n) + k]) = (vt[(j * n) + k], vt[(i * n) + k]);
                }
            }
        }

        wd.CopyTo(w);

        var rng = new CvRng(0x12345678);
        for (int i = 0; i < n1; i++)
        {
            double sd = i < n ? wd[i] : 0;
            for (int ii = 0; ii < 100 && sd <= minval; ii++)
            {
                // A zero singular value: build an orthogonal left vector from a random vector.
                double val0 = 1.0 / m;
                for (int k = 0; k < m; k++)
                {
                    at[(i * m) + k] = (rng.Next() & 256) != 0 ? val0 : -val0;
                }

                for (int iter = 0; iter < 2; iter++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        sd = 0;
                        for (int k = 0; k < m; k++)
                        {
                            sd += at[(i * m) + k] * at[(j * m) + k];
                        }

                        double asum = 0;
                        for (int k = 0; k < m; k++)
                        {
                            double t = at[(i * m) + k] - (sd * at[(j * m) + k]);
                            at[(i * m) + k] = t;
                            asum += Math.Abs(t);
                        }

                        asum = asum > eps * 100 ? 1 / asum : 0;
                        for (int k = 0; k < m; k++)
                        {
                            at[(i * m) + k] *= asum;
                        }
                    }
                }

                sd = 0;
                for (int k = 0; k < m; k++)
                {
                    double t = at[(i * m) + k];
                    sd += t * t;
                }

                sd = Math.Sqrt(sd);
            }

            double scale = sd > minval ? 1 / sd : 0.0;
            for (int k = 0; k < m; k++)
            {
                at[(i * m) + k] *= scale;
            }
        }
    }

    /// <summary>
    /// <c>SVBkSbImpl_&lt;double&gt;</c> with one right-hand side: <c>x = V · diag(1/w) · Uᵀ · b</c>.
    /// <paramref name="u"/> holds the left singular vectors as rows of m, <paramref name="v"/> the
    /// right singular vectors as rows of n.
    /// </summary>
    public static void SvBackSubst(int m, int n, ReadOnlySpan<double> w, ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> b, Span<double> x)
    {
        int nm = Math.Min(m, n);
        double threshold = 0;
        for (int j = 0; j < n; j++)
        {
            x[j] = 0;
        }

        for (int i = 0; i < nm; i++)
        {
            threshold += w[i];
        }

        threshold *= DblEpsilon * 2;

        for (int i = 0; i < nm; i++)
        {
            double wi = w[i];
            if (Math.Abs(wi) <= threshold)
            {
                continue;
            }

            wi = 1 / wi;
            double s = 0;
            for (int j = 0; j < m; j++)
            {
                s += u[(i * m) + j] * b[j];
            }

            s *= wi;
            for (int j = 0; j < n; j++)
            {
                x[j] = x[j] + (s * v[(i * n) + j]);
            }
        }
    }

    /// <summary>
    /// <c>SVDecomp(a, w, u, vt)</c> for a row-major m × n matrix with m ≥ n. Returns
    /// <c>uᵀ</c> (n rows of m) in <paramref name="ut"/>.
    /// </summary>
    public static void SvDecomp(ReadOnlySpan<double> a, int m, int n, Span<double> w, Span<double> ut, Span<double> vt)
    {
        for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < n; c++)
            {
                ut[(c * m) + r] = a[(r * n) + c];
            }
        }

        JacobiSvd(ut, w, vt, m, n, n);
    }

    /// <summary><c>solve(a, b, x, DECOMP_SVD)</c> for a row-major m × n matrix (m ≥ n) and one right-hand side.</summary>
    public static void SolveSvd(ReadOnlySpan<double> a, int m, int n, ReadOnlySpan<double> b, Span<double> x)
    {
        double[] rented = ArrayPool<double>.Shared.Rent(m * n);
        Span<double> at = rented.AsSpan(0, m * n);
        Span<double> w = stackalloc double[n];
        Span<double> vt = stackalloc double[n * n];
        SvDecomp(a, m, n, w, at, vt);
        SvBackSubst(m, n, w, at, vt, b, x);
        ArrayPool<double>.Shared.Return(rented);
    }

    /// <summary>
    /// <c>determinant</c> of a row-major n × n matrix (n ≥ 4) through <c>LUImpl</c> with
    /// <c>DBL_EPSILON * 100</c>. The input is not changed.
    /// </summary>
    public static double LuDeterminant(ReadOnlySpan<double> src, int n)
    {
        Span<double> a = stackalloc double[n * n];
        src[..(n * n)].CopyTo(a);
        const double eps = DblEpsilon * 100;
        int p = 1;
        for (int i = 0; i < n; i++)
        {
            int k = i;
            for (int j = i + 1; j < n; j++)
            {
                if (Math.Abs(a[(j * n) + i]) > Math.Abs(a[(k * n) + i]))
                {
                    k = j;
                }
            }

            if (Math.Abs(a[(k * n) + i]) < eps)
            {
                return 0;
            }

            if (k != i)
            {
                for (int j = i; j < n; j++)
                {
                    (a[(i * n) + j], a[(k * n) + j]) = (a[(k * n) + j], a[(i * n) + j]);
                }

                p = -p;
            }

            double d = -1 / a[(i * n) + i];
            for (int j = i + 1; j < n; j++)
            {
                double alpha = a[(j * n) + i] * d;
                for (int c = i + 1; c < n; c++)
                {
                    a[(j * n) + c] += alpha * a[(i * n) + c];
                }
            }
        }

        double result = p;
        for (int i = 0; i < n; i++)
        {
            result *= a[(i * n) + i];
        }

        return result;
    }

    /// <summary>
    /// <c>eigenNonSymmetric</c>: eigenvalues in decreasing order and the matching eigenvectors as
    /// rows (n × n), from the JAMA Hessenberg/QR algorithm. Complex pairs give their real parts.
    /// </summary>
    public static void EigenNonSymmetric(ReadOnlySpan<double> src, int n, Span<double> evals, Span<double> evects)
    {
        var h = new double[n][];
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            h[i] = src.Slice(i * n, n).ToArray();
            v[i] = new double[n];
        }

        var d = new double[n];
        var e = new double[n];
        var ort = new double[n];
        Orthes(n, h, v, ort);
        Hqr2(n, h, v, d, e);

        Span<int> idx = stackalloc int[n];
        for (int i = 0; i < n; i++)
        {
            idx[i] = i;
        }

        // std::sort on at most 32 elements is the stable insertion sort of the MSVC library.
        for (int i = 1; i < n; i++)
        {
            int val = idx[i];
            int hole = i;
            while (hole > 0 && d[val] < d[idx[hole - 1]])
            {
                idx[hole] = idx[hole - 1];
                hole--;
            }

            idx[hole] = val;
        }

        for (int j = 0; j < n / 2; j++)
        {
            (idx[j], idx[n - 1 - j]) = (idx[n - 1 - j], idx[j]);
        }

        for (int i = 0; i < n; i++)
        {
            evals[i] = d[idx[i]];
            for (int j = 0; j < n; j++)
            {
                evects[(i * n) + j] = v[j][idx[i]];
            }
        }
    }

    private static void Orthes(int n, double[][] h, double[][] v, double[] ort)
    {
        int low = 0;
        int high = n - 1;

        for (int m = low + 1; m < high; m++)
        {
            double scale = 0.0;
            for (int i = m; i <= high; i++)
            {
                scale = scale + Math.Abs(h[i][m - 1]);
            }

            if (scale != 0.0)
            {
                double hh = 0.0;
                for (int i = high; i >= m; i--)
                {
                    ort[i] = h[i][m - 1] / scale;
                    hh += ort[i] * ort[i];
                }

                double g = Math.Sqrt(hh);
                if (ort[m] > 0)
                {
                    g = -g;
                }

                hh = hh - (ort[m] * g);
                ort[m] = ort[m] - g;

                for (int j = m; j < n; j++)
                {
                    double f = 0.0;
                    for (int i = high; i >= m; i--)
                    {
                        f += ort[i] * h[i][j];
                    }

                    f = f / hh;
                    for (int i = m; i <= high; i++)
                    {
                        h[i][j] -= f * ort[i];
                    }
                }

                for (int i = 0; i <= high; i++)
                {
                    double f = 0.0;
                    for (int j = high; j >= m; j--)
                    {
                        f += ort[j] * h[i][j];
                    }

                    f = f / hh;
                    for (int j = m; j <= high; j++)
                    {
                        h[i][j] -= f * ort[j];
                    }
                }

                ort[m] = scale * ort[m];
                h[m][m - 1] = scale * g;
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                v[i][j] = i == j ? 1.0 : 0.0;
            }
        }

        for (int m = high - 1; m > low; m--)
        {
            if (h[m][m - 1] != 0.0)
            {
                for (int i = m + 1; i <= high; i++)
                {
                    ort[i] = h[i][m - 1];
                }

                for (int j = m; j <= high; j++)
                {
                    double g = 0.0;
                    for (int i = m; i <= high; i++)
                    {
                        g += ort[i] * v[i][j];
                    }

                    g = (g / ort[m]) / h[m][m - 1];
                    for (int i = m; i <= high; i++)
                    {
                        v[i][j] += g * ort[i];
                    }
                }
            }
        }
    }

    private static void ComplexDiv(double xr, double xi, double yr, double yi, out double cdivr, out double cdivi)
    {
        double r, dv;
        if (Math.Abs(yr) > Math.Abs(yi))
        {
            r = yi / yr;
            dv = yr + (r * yi);
            cdivr = (xr + (r * xi)) / dv;
            cdivi = (xi - (r * xr)) / dv;
        }
        else
        {
            r = yr / yi;
            dv = yi + (r * yr);
            cdivr = ((r * xr) + xi) / dv;
            cdivi = ((r * xi) - xr) / dv;
        }
    }

    private static void Hqr2(int nn, double[][] h, double[][] v, double[] d, double[] e)
    {
        int maxIters = 1000 * nn;
        int n1 = nn - 1;
        const int low = 0;
        int high = nn - 1;
        const double eps = DblEpsilon;
        double exshift = 0.0;

        double norm = 0.0;
        for (int i = 0; i < nn; i++)
        {
            for (int j = Math.Max(i - 1, 0); j < nn; j++)
            {
                norm += Math.Abs(h[i][j]);
            }
        }

        int iter = 0;
        while (n1 >= low)
        {
            int l = n1;
            while (l > low)
            {
                if (norm < FltEpsilon)
                {
                    break;
                }

                double s = Math.Abs(h[l - 1][l - 1]) + Math.Abs(h[l][l]);
                if (s == 0.0)
                {
                    s = norm;
                }

                if (Math.Abs(h[l][l - 1]) < eps * s)
                {
                    break;
                }

                l--;
            }

            if (l == n1)
            {
                h[n1][n1] = h[n1][n1] + exshift;
                d[n1] = h[n1][n1];
                e[n1] = 0.0;
                n1--;
                iter = 0;
            }
            else if (l == n1 - 1)
            {
                double w = h[n1][n1 - 1] * h[n1 - 1][n1];
                double p = (h[n1 - 1][n1 - 1] - h[n1][n1]) * 0.5;
                double q = (p * p) + w;
                double z = Math.Sqrt(Math.Abs(q));
                h[n1][n1] = h[n1][n1] + exshift;
                h[n1 - 1][n1 - 1] = h[n1 - 1][n1 - 1] + exshift;
                double x = h[n1][n1];

                if (q >= 0)
                {
                    z = p >= 0 ? p + z : p - z;
                    d[n1 - 1] = x + z;
                    d[n1] = d[n1 - 1];
                    if (z != 0.0)
                    {
                        d[n1] = x - (w / z);
                    }

                    e[n1 - 1] = 0.0;
                    e[n1] = 0.0;
                    x = h[n1][n1 - 1];
                    double s = Math.Abs(x) + Math.Abs(z);
                    p = x / s;
                    q = z / s;
                    double r = Math.Sqrt((p * p) + (q * q));
                    p = p / r;
                    q = q / r;

                    for (int j = n1 - 1; j < nn; j++)
                    {
                        z = h[n1 - 1][j];
                        h[n1 - 1][j] = (q * z) + (p * h[n1][j]);
                        h[n1][j] = (q * h[n1][j]) - (p * z);
                    }

                    for (int i = 0; i <= n1; i++)
                    {
                        z = h[i][n1 - 1];
                        h[i][n1 - 1] = (q * z) + (p * h[i][n1]);
                        h[i][n1] = (q * h[i][n1]) - (p * z);
                    }

                    for (int i = low; i <= high; i++)
                    {
                        z = v[i][n1 - 1];
                        v[i][n1 - 1] = (q * z) + (p * v[i][n1]);
                        v[i][n1] = (q * v[i][n1]) - (p * z);
                    }
                }
                else
                {
                    d[n1 - 1] = x + p;
                    d[n1] = x + p;
                    e[n1 - 1] = z;
                    e[n1] = -z;
                }

                n1 = n1 - 2;
                iter = 0;
            }
            else
            {
                double x = h[n1][n1];
                double y = 0.0;
                double w = 0.0;
                if (l < n1)
                {
                    y = h[n1 - 1][n1 - 1];
                    w = h[n1][n1 - 1] * h[n1 - 1][n1];
                }

                if (iter == 10)
                {
                    exshift += x;
                    for (int i = low; i <= n1; i++)
                    {
                        h[i][i] -= x;
                    }

                    double s = Math.Abs(h[n1][n1 - 1]) + Math.Abs(h[n1 - 1][n1 - 2]);
                    x = y = 0.75 * s;
                    w = -0.4375 * s * s;
                }

                if (iter == 30)
                {
                    double s = (y - x) * 0.5;
                    s = (s * s) + w;
                    if (s > 0)
                    {
                        s = Math.Sqrt(s);
                        if (y < x)
                        {
                            s = -s;
                        }

                        s = x - (w / (((y - x) * 0.5) + s));
                        for (int i = low; i <= n1; i++)
                        {
                            h[i][i] -= s;
                        }

                        exshift += s;
                        x = y = w = 0.964;
                    }
                }

                iter = iter + 1;
                if (iter > maxIters)
                {
                    throw new InvalidOperationException("The eigenvalue algorithm does not converge.");
                }

                double p = double.NaN;
                double q = double.NaN;
                double r = double.NaN;

                int m = n1 - 2;
                while (m >= l)
                {
                    double z = h[m][m];
                    r = x - z;
                    double s = y - z;
                    p = (((r * s) - w) / h[m + 1][m]) + h[m][m + 1];
                    q = h[m + 1][m + 1] - z - r - s;
                    r = h[m + 2][m + 1];
                    s = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                    p = p / s;
                    q = q / s;
                    r = r / s;
                    if (m == l)
                    {
                        break;
                    }

                    if (Math.Abs(h[m][m - 1]) * (Math.Abs(q) + Math.Abs(r)) <
                        eps * (Math.Abs(p) * (Math.Abs(h[m - 1][m - 1]) + Math.Abs(z) + Math.Abs(h[m + 1][m + 1]))))
                    {
                        break;
                    }

                    m--;
                }

                for (int i = m + 2; i <= n1; i++)
                {
                    h[i][i - 2] = 0.0;
                    if (i > m + 2)
                    {
                        h[i][i - 3] = 0.0;
                    }
                }

                for (int k = m; k < n1; k++)
                {
                    bool notlast = k != n1 - 1;
                    if (k != m)
                    {
                        p = h[k][k - 1];
                        q = h[k + 1][k - 1];
                        r = notlast ? h[k + 2][k - 1] : 0.0;
                        x = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                        if (x != 0.0)
                        {
                            p = p / x;
                            q = q / x;
                            r = r / x;
                        }
                    }

                    if (x == 0.0)
                    {
                        break;
                    }

                    double s = Math.Sqrt((p * p) + (q * q) + (r * r));
                    if (p < 0)
                    {
                        s = -s;
                    }

                    if (s != 0)
                    {
                        if (k != m)
                        {
                            h[k][k - 1] = -s * x;
                        }
                        else if (l != m)
                        {
                            h[k][k - 1] = -h[k][k - 1];
                        }

                        p = p + s;
                        x = p / s;
                        y = q / s;
                        double z = r / s;
                        q = q / p;
                        r = r / p;

                        for (int j = k; j < nn; j++)
                        {
                            p = h[k][j] + (q * h[k + 1][j]);
                            if (notlast)
                            {
                                p = p + (r * h[k + 2][j]);
                                h[k + 2][j] = h[k + 2][j] - (p * z);
                            }

                            h[k][j] -= p * x;
                            h[k + 1][j] -= p * y;
                        }

                        for (int i = 0; i <= Math.Min(n1, k + 3); i++)
                        {
                            p = (x * h[i][k]) + (y * h[i][k + 1]);
                            if (notlast)
                            {
                                p = p + (z * h[i][k + 2]);
                                h[i][k + 2] = h[i][k + 2] - (p * r);
                            }

                            h[i][k] -= p;
                            h[i][k + 1] -= p * q;
                        }

                        for (int i = low; i <= high; i++)
                        {
                            p = (x * v[i][k]) + (y * v[i][k + 1]);
                            if (notlast)
                            {
                                p = p + (z * v[i][k + 2]);
                                v[i][k + 2] = v[i][k + 2] - (p * r);
                            }

                            v[i][k] = v[i][k] - p;
                            v[i][k + 1] = v[i][k + 1] - (p * q);
                        }
                    }
                }
            }
        }

        if (norm < FltEpsilon)
        {
            return;
        }

        for (n1 = nn - 1; n1 >= 0; n1--)
        {
            double p = d[n1];
            double q = e[n1];

            if (q == 0)
            {
                double z = double.NaN;
                double s = double.NaN;
                int l = n1;
                h[n1][n1] = 1.0;
                for (int i = n1 - 1; i >= 0; i--)
                {
                    double w = h[i][i] - p;
                    double r = 0.0;
                    for (int j = l; j <= n1; j++)
                    {
                        r = r + (h[i][j] * h[j][n1]);
                    }

                    if (e[i] < 0.0)
                    {
                        z = w;
                        s = r;
                    }
                    else
                    {
                        l = i;
                        if (e[i] == 0.0)
                        {
                            h[i][n1] = w != 0.0 ? -r / w : -r / (eps * norm);
                        }
                        else
                        {
                            double x = h[i][i + 1];
                            double y = h[i + 1][i];
                            q = ((d[i] - p) * (d[i] - p)) + (e[i] * e[i]);
                            double t = ((x * s) - (z * r)) / q;
                            h[i][n1] = t;
                            if (Math.Abs(x) > Math.Abs(z))
                            {
                                h[i + 1][n1] = (-r - (w * t)) / x;
                            }
                            else
                            {
                                h[i + 1][n1] = (-s - (y * t)) / z;
                            }
                        }

                        double tt = Math.Abs(h[i][n1]);
                        if ((eps * tt) * tt > 1)
                        {
                            double invT = 1.0 / tt;
                            for (int j = i; j <= n1; j++)
                            {
                                h[j][n1] *= invT;
                            }
                        }
                    }
                }
            }
            else if (q < 0)
            {
                double z = double.NaN;
                double r = double.NaN;
                double s = double.NaN;
                int l = n1 - 1;

                if (Math.Abs(h[n1][n1 - 1]) > Math.Abs(h[n1 - 1][n1]))
                {
                    h[n1 - 1][n1 - 1] = q / h[n1][n1 - 1];
                    h[n1 - 1][n1] = -(h[n1][n1] - p) / h[n1][n1 - 1];
                }
                else
                {
                    ComplexDiv(0.0, -h[n1 - 1][n1], h[n1 - 1][n1 - 1] - p, q, out h[n1 - 1][n1 - 1], out h[n1 - 1][n1]);
                }

                h[n1][n1 - 1] = 0.0;
                h[n1][n1] = 1.0;
                for (int i = n1 - 2; i >= 0; i--)
                {
                    double ra = 0.0, sa = 0.0, vr, vi;
                    for (int j = l; j <= n1; j++)
                    {
                        ra = ra + (h[i][j] * h[j][n1 - 1]);
                        sa = sa + (h[i][j] * h[j][n1]);
                    }

                    double w = h[i][i] - p;

                    if (e[i] < 0.0)
                    {
                        z = w;
                        r = ra;
                        s = sa;
                    }
                    else
                    {
                        l = i;
                        if (e[i] == 0)
                        {
                            ComplexDiv(-ra, -sa, w, q, out h[i][n1 - 1], out h[i][n1]);
                        }
                        else
                        {
                            double x = h[i][i + 1];
                            double y = h[i + 1][i];
                            vr = ((d[i] - p) * (d[i] - p)) + (e[i] * e[i]) - (q * q);
                            vi = (d[i] - p) * 2.0 * q;
                            if (vr == 0.0 && vi == 0.0)
                            {
                                vr = eps * norm * (Math.Abs(w) + Math.Abs(q) + Math.Abs(x) + Math.Abs(y) + Math.Abs(z));
                            }

                            ComplexDiv(
                                (x * r) - (z * ra) + (q * sa),
                                (x * s) - (z * sa) - (q * ra),
                                vr,
                                vi,
                                out h[i][n1 - 1],
                                out h[i][n1]);
                            if (Math.Abs(x) > (Math.Abs(z) + Math.Abs(q)))
                            {
                                h[i + 1][n1 - 1] = (-ra - (w * h[i][n1 - 1]) + (q * h[i][n1])) / x;
                                h[i + 1][n1] = (-sa - (w * h[i][n1]) - (q * h[i][n1 - 1])) / x;
                            }
                            else
                            {
                                ComplexDiv(-r - (y * h[i][n1 - 1]), -s - (y * h[i][n1]), z, q, out h[i + 1][n1 - 1], out h[i + 1][n1]);
                            }
                        }

                        double t = Math.Max(Math.Abs(h[i][n1 - 1]), Math.Abs(h[i][n1]));
                        if ((eps * t) * t > 1)
                        {
                            for (int j = i; j <= n1; j++)
                            {
                                h[j][n1 - 1] = h[j][n1 - 1] / t;
                                h[j][n1] = h[j][n1] / t;
                            }
                        }
                    }
                }
            }
        }

        for (int j = nn - 1; j >= low; j--)
        {
            for (int i = low; i <= high; i++)
            {
                double z = 0.0;
                for (int k = low; k <= Math.Min(j, high); k++)
                {
                    z += v[i][k] * h[k][j];
                }

                v[i][j] = z;
            }
        }
    }
}
