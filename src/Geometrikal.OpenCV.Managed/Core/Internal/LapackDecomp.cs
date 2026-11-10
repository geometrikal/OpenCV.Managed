// This implementation is ported from the OpenCV dense linear algebra.
//
// Original sources:
//   modules/core/src/matrix_decomp.cpp (LUImpl, CholImpl, QRImpl)
//   modules/core/src/lapack.cpp (JacobiImpl_, JacobiSVDImpl_, MatrAXPY, SVBkSbImpl_)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using static Geometrikal.OpenCV.Managed.Core.Internal.LapackScalar;

/// <summary>
/// Dense decompositions in OpenCV operation order. Matrices are row-major spans with an element
/// step. <c>T</c> is <c>float</c> or <c>double</c>; mixed-precision expressions follow the C++
/// promotion rules of the original templates.
/// </summary>
internal static class LapackDecomp
{
    /// <summary>
    /// <c>LUImpl</c>: Gaussian elimination with partial pivoting of the m × m matrix
    /// <paramref name="a"/> and forward and back substitution of the m × n right-hand side
    /// <paramref name="b"/> (skipped when <paramref name="b"/> is empty). Returns the permutation
    /// sign, or 0 when a pivot is below <c>FLT_EPSILON*10</c> or <c>DBL_EPSILON*100</c>.
    /// </summary>
    public static int Lu<T>(Span<T> a, int astep, int m, Span<T> b, int bstep, int n)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T eps = F<T>(IsFloat<T>() ? (float)(FltEpsilon * 10) : DblEpsilon * 100);
        bool hasB = !b.IsEmpty;
        int p = 1;

        for (int i = 0; i < m; i++)
        {
            int k = i;
            for (int j = i + 1; j < m; j++)
            {
                if (T.Abs(a[(j * astep) + i]) > T.Abs(a[(k * astep) + i]))
                {
                    k = j;
                }
            }

            if (T.Abs(a[(k * astep) + i]) < eps)
            {
                return 0;
            }

            if (k != i)
            {
                for (int j = i; j < m; j++)
                {
                    (a[(i * astep) + j], a[(k * astep) + j]) = (a[(k * astep) + j], a[(i * astep) + j]);
                }

                if (hasB)
                {
                    for (int j = 0; j < n; j++)
                    {
                        (b[(i * bstep) + j], b[(k * bstep) + j]) = (b[(k * bstep) + j], b[(i * bstep) + j]);
                    }
                }

                p = -p;
            }

            T d = -T.One / a[(i * astep) + i];
            for (int j = i + 1; j < m; j++)
            {
                T alpha = a[(j * astep) + i] * d;
                for (int c = i + 1; c < m; c++)
                {
                    a[(j * astep) + c] += alpha * a[(i * astep) + c];
                }

                if (hasB)
                {
                    for (int c = 0; c < n; c++)
                    {
                        b[(j * bstep) + c] += alpha * b[(i * bstep) + c];
                    }
                }
            }
        }

        if (hasB)
        {
            for (int i = m - 1; i >= 0; i--)
            {
                for (int j = 0; j < n; j++)
                {
                    T s = b[(i * bstep) + j];
                    for (int c = i + 1; c < m; c++)
                    {
                        s -= a[(i * astep) + c] * b[(c * bstep) + j];
                    }

                    b[(i * bstep) + j] = s / a[(i * astep) + i];
                }
            }
        }

        return p;
    }

    /// <summary>
    /// <c>CholImpl</c>: Cholesky factorization of the symmetric m × m matrix <paramref name="a"/>
    /// (lower triangle, inverse diagonal during the solve) and solution of the m × n right-hand
    /// side <paramref name="b"/>. Returns false when the matrix is not positive definite.
    /// </summary>
    public static bool Cholesky<T>(Span<T> a, int astep, int m, Span<T> b, int bstep, int n)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double eps = D(Epsilon<T>());
        double s;
        for (int i = 0; i < m; i++)
        {
            int j;
            for (j = 0; j < i; j++)
            {
                s = D(a[(i * astep) + j]);
                for (int k = 0; k < j; k++)
                {
                    s -= D(a[(i * astep) + k] * a[(j * astep) + k]);
                }

                a[(i * astep) + j] = F<T>(s * D(a[(j * astep) + j]));
            }

            s = D(a[(i * astep) + i]);
            for (int k = 0; k < j; k++)
            {
                double t = D(a[(i * astep) + k]);
                s -= t * t;
            }

            if (s < eps)
            {
                return false;
            }

            a[(i * astep) + i] = F<T>(1.0 / Math.Sqrt(s));
        }

        if (b.IsEmpty)
        {
            for (int i = 0; i < m; i++)
            {
                a[(i * astep) + i] = T.One / a[(i * astep) + i];
            }

            return true;
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                s = D(b[(i * bstep) + j]);
                for (int k = 0; k < i; k++)
                {
                    s -= D(a[(i * astep) + k] * b[(k * bstep) + j]);
                }

                b[(i * bstep) + j] = F<T>(s * D(a[(i * astep) + i]));
            }
        }

        for (int i = m - 1; i >= 0; i--)
        {
            for (int j = 0; j < n; j++)
            {
                s = D(b[(i * bstep) + j]);
                for (int k = m - 1; k > i; k--)
                {
                    s -= D(a[(k * astep) + i] * b[(k * bstep) + j]);
                }

                b[(i * bstep) + j] = F<T>(s * D(a[(i * astep) + i]));
            }
        }

        for (int i = 0; i < m; i++)
        {
            a[(i * astep) + i] = T.One / a[(i * astep) + i];
        }

        return true;
    }

    /// <summary>
    /// <c>QRImpl</c>: Householder QR of the m × n matrix <paramref name="a"/> and least-squares
    /// solution of the m × k right-hand side <paramref name="b"/> (result in the first n rows).
    /// Returns 0 when a diagonal element of R is below <c>FLT_EPSILON*10</c> or <c>DBL_EPSILON*100</c>.
    /// </summary>
    public static int Qr<T>(Span<T> a, int astep, int m, int n, int k, Span<T> b, int bstep)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T eps = F<T>(IsFloat<T>() ? (float)(FltEpsilon * 10) : DblEpsilon * 100);
        T two = T.One + T.One;
        T[] buffer = new T[m + n];
        Span<T> vl = buffer.AsSpan(0, m);
        Span<T> hFactors = buffer.AsSpan(m, n);

        for (int l = 0; l < n; l++)
        {
            int vlSize = m - l;
            T vlNorm = T.Zero;
            for (int i = 0; i < vlSize; i++)
            {
                vl[i] = a[((l + i) * astep) + l];
                vlNorm += vl[i] * vl[i];
            }

            T tmpV = vl[0];
            T sign = vl[0] >= T.Zero ? T.One : -T.One;
            vl[0] = vl[0] + (sign * T.Sqrt(vlNorm));
            vlNorm = T.Sqrt(vlNorm + (vl[0] * vl[0]) - (tmpV * tmpV));
            for (int i = 0; i < vlSize; i++)
            {
                vl[i] /= vlNorm;
            }

            for (int j = l; j < n; j++)
            {
                T vlA = T.Zero;
                for (int i = l; i < m; i++)
                {
                    vlA += vl[i - l] * a[(i * astep) + j];
                }

                for (int i = l; i < m; i++)
                {
                    a[(i * astep) + j] -= two * vl[i - l] * vlA;
                }
            }

            hFactors[l] = vl[0] * vl[0];
            for (int i = 1; i < vlSize; i++)
            {
                a[((l + i) * astep) + l] = vl[i] / vl[0];
            }
        }

        if (!b.IsEmpty)
        {
            for (int l = 0; l < n; l++)
            {
                vl[0] = T.One;
                for (int j = 1; j < m - l; j++)
                {
                    vl[j] = a[((j + l) * astep) + l];
                }

                for (int j = 0; j < k; j++)
                {
                    T vlB = T.Zero;
                    for (int i = l; i < m; i++)
                    {
                        vlB += vl[i - l] * b[(i * bstep) + j];
                    }

                    for (int i = l; i < m; i++)
                    {
                        b[(i * bstep) + j] -= two * vl[i - l] * vlB * hFactors[l];
                    }
                }
            }

            for (int i = n - 1; i >= 0; i--)
            {
                for (int j = n - 1; j > i; j--)
                {
                    for (int p = 0; p < k; p++)
                    {
                        b[(i * bstep) + p] -= b[(j * bstep) + p] * a[(i * astep) + j];
                    }
                }

                if (T.Abs(a[(i * astep) + i]) < eps)
                {
                    return 0;
                }

                for (int p = 0; p < k; p++)
                {
                    b[(i * bstep) + p] /= a[(i * astep) + i];
                }
            }
        }

        return 1;
    }

    /// <summary>
    /// <c>JacobiImpl_</c>: eigenvalues <paramref name="w"/> and eigenvectors (rows of
    /// <paramref name="v"/>, optional) of the symmetric n × n matrix <paramref name="a"/>, sorted
    /// by decreasing eigenvalue. The upper triangle of <paramref name="a"/> is destroyed.
    /// </summary>
    public static void Jacobi<T>(Span<T> a, int astep, Span<T> w, Span<T> v, int vstep, int n)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T eps = Epsilon<T>();
        bool hasV = !v.IsEmpty;
        if (hasV)
        {
            for (int i = 0; i < n; i++)
            {
                v.Slice(i * vstep, n).Clear();
                v[(i * vstep) + i] = T.One;
            }
        }

        int maxIters = n * n * 30;
        Span<int> indR = n <= 256 ? stackalloc int[n] : new int[n];
        Span<int> indC = n <= 256 ? stackalloc int[n] : new int[n];
        T mv;
        int m;

        for (int k = 0; k < n; k++)
        {
            w[k] = a[(astep + 1) * k];
            if (k < n - 1)
            {
                m = k + 1;
                mv = T.Abs(a[(astep * k) + m]);
                for (int i = k + 2; i < n; i++)
                {
                    T val = T.Abs(a[(astep * k) + i]);
                    if (mv < val)
                    {
                        mv = val;
                        m = i;
                    }
                }

                indR[k] = m;
            }

            if (k > 0)
            {
                m = 0;
                mv = T.Abs(a[k]);
                for (int i = 1; i < k; i++)
                {
                    T val = T.Abs(a[(astep * i) + k]);
                    if (mv < val)
                    {
                        mv = val;
                        m = i;
                    }
                }

                indC[k] = m;
            }
        }

        if (n > 1)
        {
            for (int iters = 0; iters < maxIters; iters++)
            {
                int k = 0;
                mv = T.Abs(a[indR[0]]);
                for (int i = 1; i < n - 1; i++)
                {
                    T val = T.Abs(a[(astep * i) + indR[i]]);
                    if (mv < val)
                    {
                        mv = val;
                        k = i;
                    }
                }

                int l = indR[k];
                for (int i = 1; i < n; i++)
                {
                    T val = T.Abs(a[(astep * indC[i]) + i]);
                    if (mv < val)
                    {
                        mv = val;
                        k = indC[i];
                        l = i;
                    }
                }

                T p = a[(astep * k) + l];
                if (T.Abs(p) <= eps)
                {
                    break;
                }

                T y = F<T>(D(w[l] - w[k]) * 0.5);
                T t = T.Abs(y) + Hypot(p, y);
                T s = Hypot(p, t);
                T c = t / s;
                s = p / s;
                t = (p / t) * p;
                if (y < T.Zero)
                {
                    s = -s;
                    t = -t;
                }

                a[(astep * k) + l] = T.Zero;
                w[k] -= t;
                w[l] += t;

                for (int i = 0; i < k; i++)
                {
                    Rotate(ref a[(astep * i) + k], ref a[(astep * i) + l], c, s);
                }

                for (int i = k + 1; i < l; i++)
                {
                    Rotate(ref a[(astep * k) + i], ref a[(astep * i) + l], c, s);
                }

                for (int i = l + 1; i < n; i++)
                {
                    Rotate(ref a[(astep * k) + i], ref a[(astep * l) + i], c, s);
                }

                if (hasV)
                {
                    for (int i = 0; i < n; i++)
                    {
                        Rotate(ref v[(vstep * k) + i], ref v[(vstep * l) + i], c, s);
                    }
                }

                for (int j = 0; j < 2; j++)
                {
                    int idx = j == 0 ? k : l;
                    if (idx < n - 1)
                    {
                        m = idx + 1;
                        mv = T.Abs(a[(astep * idx) + m]);
                        for (int i = idx + 2; i < n; i++)
                        {
                            T val = T.Abs(a[(astep * idx) + i]);
                            if (mv < val)
                            {
                                mv = val;
                                m = i;
                            }
                        }

                        indR[idx] = m;
                    }

                    if (idx > 0)
                    {
                        m = 0;
                        mv = T.Abs(a[idx]);
                        for (int i = 1; i < idx; i++)
                        {
                            T val = T.Abs(a[(astep * i) + idx]);
                            if (mv < val)
                            {
                                mv = val;
                                m = i;
                            }
                        }

                        indC[idx] = m;
                    }
                }
            }
        }

        for (int k = 0; k < n - 1; k++)
        {
            m = k;
            for (int i = k + 1; i < n; i++)
            {
                if (w[m] < w[i])
                {
                    m = i;
                }
            }

            if (k != m)
            {
                (w[m], w[k]) = (w[k], w[m]);
                if (hasV)
                {
                    for (int i = 0; i < n; i++)
                    {
                        (v[(vstep * m) + i], v[(vstep * k) + i]) = (v[(vstep * k) + i], v[(vstep * m) + i]);
                    }
                }
            }
        }
    }

    private static void Rotate<T>(ref T v0, ref T v1, T c, T s)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T a0 = v0, b0 = v1;
        v0 = (a0 * c) - (b0 * s);
        v1 = (a0 * s) + (b0 * c);
    }

    /// <summary>
    /// <c>hal::SVD32f</c>/<c>hal::SVD64f</c> without the LAPACK hook (<c>JacobiSVDImpl_</c>).
    /// <paramref name="at"/> holds the n vectors of length m (rows of <c>Aᵀ</c>); on return its
    /// first <paramref name="n1"/> rows hold the left singular vectors. <paramref name="vt"/>
    /// (n × n, optional) receives <c>Vᵀ</c>. Pass an empty <paramref name="vt"/> for values only.
    /// </summary>
    public static void JacobiSvd<T>(Span<T> at, int astep, Span<T> w, Span<T> vt, int vstep, int m, int n, int n1)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        bool hasVt = !vt.IsEmpty;
        n1 = !hasVt ? 0 : n1 < 0 ? n : n1;
        double minval = IsFloat<T>() ? FltMin : DblMin;
        T eps = F<T>(IsFloat<T>() ? (float)(FltEpsilon * 2) : DblEpsilon * 10);
        double epsD = D(eps);
        Span<double> wbuf = n <= 16 ? stackalloc double[16] : new double[n];
        int maxIter = Math.Max(m, 30);
        double sd;

        for (int i = 0; i < n; i++)
        {
            sd = 0;
            for (int k = 0; k < m; k++)
            {
                double t = D(at[(i * astep) + k]);
                sd += t * t;
            }

            wbuf[i] = sd;
            if (hasVt)
            {
                vt.Slice(i * vstep, n).Clear();
                vt[(i * vstep) + i] = T.One;
            }
        }

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;
            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    Span<T> ai = at.Slice(i * astep, m);
                    Span<T> aj = at.Slice(j * astep, m);
                    double a = wbuf[i], p = 0, b = wbuf[j];
                    for (int k = 0; k < m; k++)
                    {
                        p += D(ai[k]) * D(aj[k]);
                    }

                    if (Math.Abs(p) <= epsD * Math.Sqrt(a * b))
                    {
                        continue;
                    }

                    p *= 2;
                    double beta = a - b, gamma = Hypot(p, beta);
                    T c, s;
                    if (beta < 0)
                    {
                        double delta = (gamma - beta) * 0.5;
                        s = F<T>(Math.Sqrt(delta / gamma));
                        c = F<T>(p / (gamma * D(s) * 2));
                    }
                    else
                    {
                        c = F<T>(Math.Sqrt((gamma + beta) / (gamma * 2)));
                        s = F<T>(p / (gamma * D(c) * 2));
                    }

                    a = b = 0;
                    for (int k = 0; k < m; k++)
                    {
                        T t0 = (c * ai[k]) + (s * aj[k]);
                        T t1 = (-s * ai[k]) + (c * aj[k]);
                        ai[k] = t0;
                        aj[k] = t1;
                        a += D(t0) * D(t0);
                        b += D(t1) * D(t1);
                    }

                    wbuf[i] = a;
                    wbuf[j] = b;
                    changed = true;

                    if (hasVt)
                    {
                        Span<T> vi = vt.Slice(i * vstep, n);
                        Span<T> vj = vt.Slice(j * vstep, n);
                        for (int k = 0; k < n; k++)
                        {
                            T t0 = (c * vi[k]) + (s * vj[k]);
                            T t1 = (-s * vi[k]) + (c * vj[k]);
                            vi[k] = t0;
                            vj[k] = t1;
                        }
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
            sd = 0;
            for (int k = 0; k < m; k++)
            {
                double t = D(at[(i * astep) + k]);
                sd += t * t;
            }

            wbuf[i] = Math.Sqrt(sd);
        }

        for (int i = 0; i < n - 1; i++)
        {
            int j = i;
            for (int k = i + 1; k < n; k++)
            {
                if (wbuf[j] < wbuf[k])
                {
                    j = k;
                }
            }

            if (i != j)
            {
                (wbuf[i], wbuf[j]) = (wbuf[j], wbuf[i]);
                if (hasVt)
                {
                    for (int k = 0; k < m; k++)
                    {
                        (at[(i * astep) + k], at[(j * astep) + k]) = (at[(j * astep) + k], at[(i * astep) + k]);
                    }

                    for (int k = 0; k < n; k++)
                    {
                        (vt[(i * vstep) + k], vt[(j * vstep) + k]) = (vt[(j * vstep) + k], vt[(i * vstep) + k]);
                    }
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            w[i] = F<T>(wbuf[i]);
        }

        if (!hasVt)
        {
            return;
        }

        ulong rng = 0x12345678;
        T epsT100 = eps * F<T>(100);
        for (int i = 0; i < n1; i++)
        {
            sd = i < n ? wbuf[i] : 0;
            for (int ii = 0; ii < 100 && sd <= minval; ii++)
            {
                T val0 = F<T>(1.0 / m);
                for (int k = 0; k < m; k++)
                {
                    at[(i * astep) + k] = (NextRng(ref rng) & 256) != 0 ? val0 : -val0;
                }

                for (int iter = 0; iter < 2; iter++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        sd = 0;
                        for (int k = 0; k < m; k++)
                        {
                            sd += D(at[(i * astep) + k] * at[(j * astep) + k]);
                        }

                        T asum = T.Zero;
                        for (int k = 0; k < m; k++)
                        {
                            T t = F<T>(D(at[(i * astep) + k]) - (sd * D(at[(j * astep) + k])));
                            at[(i * astep) + k] = t;
                            asum += T.Abs(t);
                        }

                        asum = asum > epsT100 ? T.One / asum : T.Zero;
                        for (int k = 0; k < m; k++)
                        {
                            at[(i * astep) + k] *= asum;
                        }
                    }
                }

                sd = 0;
                for (int k = 0; k < m; k++)
                {
                    double t = D(at[(i * astep) + k]);
                    sd += t * t;
                }

                sd = Math.Sqrt(sd);
            }

            T sc = F<T>(sd > minval ? 1 / sd : 0.0);
            for (int k = 0; k < m; k++)
            {
                at[(i * astep) + k] *= sc;
            }
        }
    }

    /// <summary>
    /// <c>SVBkSbImpl_</c>: <c>x = V diag(1/w) Uᵀ b</c> (n × nb) with singular values below
    /// <c>eps * Σw</c> treated as zero. An empty <paramref name="b"/> gives the pseudo-inverse
    /// (nb = m). <paramref name="uT"/> and <paramref name="vT"/> state that the singular vectors
    /// are stored in rows.
    /// </summary>
    public static void SvBkSb<T>(
        int m,
        int n,
        ReadOnlySpan<T> w,
        int incw,
        ReadOnlySpan<T> u,
        int ldu,
        bool uT,
        ReadOnlySpan<T> v,
        int ldv,
        bool vT,
        ReadOnlySpan<T> b,
        int ldb,
        int nb,
        Span<T> x,
        int ldx)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double eps = IsFloat<T>() ? (float)(DblEpsilon * 2) : DblEpsilon * 2;
        double threshold = 0;
        int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
        int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
        int nm = Math.Min(m, n);
        bool hasB = !b.IsEmpty;
        if (!hasB)
        {
            nb = m;
        }

        for (int i = 0; i < n; i++)
        {
            x.Slice(i * ldx, nb).Clear();
        }

        for (int i = 0; i < nm; i++)
        {
            threshold += D(w[i * incw]);
        }

        threshold *= eps;
        double[]? buffer = nb > 1 ? new double[nb] : null;

        for (int i = 0, uo = 0, vo = 0; i < nm; i++, uo += udelta0, vo += vdelta0)
        {
            double wi = D(w[i * incw]);
            if (Math.Abs(wi) <= threshold)
            {
                continue;
            }

            wi = 1 / wi;
            if (nb == 1)
            {
                double s = 0;
                if (hasB)
                {
                    for (int j = 0; j < m; j++)
                    {
                        s += D(u[uo + (j * udelta1)] * b[j * ldb]);
                    }
                }
                else
                {
                    s = D(u[uo]);
                }

                s *= wi;
                for (int j = 0; j < n; j++)
                {
                    x[j * ldx] = F<T>(D(x[j * ldx]) + (s * D(v[vo + (j * vdelta1)])));
                }
            }
            else
            {
                if (hasB)
                {
                    Array.Clear(buffer!);
                    for (int r = 0; r < m; r++)
                    {
                        T s = u[uo + (r * udelta1)];
                        ReadOnlySpan<T> brow = b.Slice(r * ldb, nb);
                        for (int j = 0; j < nb; j++)
                        {
                            buffer![j] = buffer[j] + D(s * brow[j]);
                        }
                    }

                    for (int j = 0; j < nb; j++)
                    {
                        buffer![j] *= wi;
                    }
                }
                else
                {
                    for (int j = 0; j < nb; j++)
                    {
                        buffer![j] = D(u[uo + (j * udelta1)]) * wi;
                    }
                }

                for (int r = 0; r < n; r++)
                {
                    double s = D(v[vo + (r * vdelta1)]);
                    Span<T> xrow = x.Slice(r * ldx, nb);
                    for (int j = 0; j < nb; j++)
                    {
                        xrow[j] = F<T>(D(xrow[j]) + (s * buffer![j]));
                    }
                }
            }
        }
    }
}
