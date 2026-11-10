// This implementation is ported from the OpenCV matrix multiplication and from the reference
// BLAS routine that the OpenCV built-in LAPACK uses.
//
// Original sources:
//   modules/core/src/matmul.dispatch.cpp (gemm)
//   modules/core/src/matmul.simd.hpp (gemmImpl, GEMMSingleMul, GEMMBlockMul, GEMMStore)
//   modules/core/src/hal_internal.cpp (lapack_gemm, HAL_GEMM_SMALL_MATRIX_THRESH)
//   3rdparty/clapack/src/sgemm.c, dgemm.c (reference BLAS xGEMM)
//
// OpenCV source location in this repository:
//   src/OpenCV/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static Geometrikal.OpenCV.Managed.Core.Internal.LapackScalar;

/// <summary>
/// <c>D = alpha·op(A)·op(B) + beta·op(C)</c> in OpenCV operation order. Below 100 rows of
/// <c>A</c> the kernels follow <c>gemmImpl</c> (double accumulation). From 100 rows of <c>A</c>
/// OpenCV with the built-in LAPACK calls reference <c>cblas_?gemm</c>; the kernels follow the
/// reference BLAS element order with <see cref="Vector{T}"/> lanes across output columns.
/// </summary>
internal static class GemmKernels
{
    /// <summary><c>HAL_GEMM_SMALL_MATRIX_THRESH</c> of hal_internal.cpp.</summary>
    public const int BlasRowThreshold = 100;

    private const int RowTile = 4;
    private const int KPanel = 256;
    private const int NPanel = 512;

    public static ReadOnlySpan<T> Whole<T>(Mat<T> m)
        where T : unmanaged
        => new(m.DangerousGetArray(), m.DangerousGetOffset(), ((m.Height - 1) * m.Stride) + m.RowLength);

    public static Span<T> WholeMutable<T>(Mat<T> m)
        where T : unmanaged
        => new(m.DangerousGetArray(), m.DangerousGetOffset(), ((m.Height - 1) * m.Stride) + m.RowLength);

    /// <summary>
    /// Runs the multiplication into the new matrix <paramref name="d"/>. <paramref name="c"/> is
    /// null when OpenCV drops it (<c>beta == 0</c>). For <c>float</c>, <paramref name="alpha"/> and
    /// <paramref name="beta"/> are already rounded to <c>float</c>.
    /// </summary>
    public static void Run<T>(Mat<T> a, Mat<T> b, double alpha, Mat<T>? c, double beta, Mat<T> d, int flags, int len)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (a.Height >= BlasRowThreshold)
        {
            ReferenceBlas(a, b, alpha, c, beta, d, flags, len);
        }
        else
        {
            OpenCvGemm(a, b, alpha, c, beta, d, flags, len);
        }
    }

    private static void OpenCvGemm<T>(Mat<T> am, Mat<T> bm, double alpha, Mat<T>? cm, double beta, Mat<T> dm, int flags, int len)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int dH = dm.Height, dW = dm.Width;
        ReadOnlySpan<T> a = Whole(am);
        ReadOnlySpan<T> b = Whole(bm);
        ReadOnlySpan<T> c = cm is null ? default : Whole(cm);
        Span<T> d = WholeMutable(dm);
        int ds = dm.Stride;
        bool aT = (flags & 1) != 0, bT = (flags & 2) != 0, cT = (flags & 4) != 0;
        int ai = aT ? 1 : am.Stride, ak = aT ? am.Stride : 1;
        int bk = bT ? 1 : bm.Stride, bj = bT ? bm.Stride : 1;
        int ci = 0, cj = 0;
        if (cm is not null)
        {
            ci = cT ? 1 : cm.Stride;
            cj = cT ? cm.Stride : 1;
        }

        bool hasC = cm is not null;

        bool smallPath = flags == 0 && len >= 2 && len <= 4 && (len == dW || len == dH)
            && !(!IsFloat<T>() && len == 4 && len != dW && dW > 16);
        if (smallPath)
        {
            for (int i = 0; i < dH; i++)
            {
                for (int j = 0; j < dW; j++)
                {
                    T t = a[i * ai] * b[j * bj];
                    for (int k = 1; k < len; k++)
                    {
                        t += a[(i * ai) + (k * ak)] * b[(k * bk) + (j * bj)];
                    }

                    double cv = hasC ? D(c[(i * ci) + (j * cj)]) : 0.0;
                    d[(i * ds) + j] = F<T>((D(t) * alpha) + (cv * beta));
                }
            }

            return;
        }

        bool dot = bT || ((dW == 1 || len == 1) && (bm.IsContinuous || bm.Height == 1));

        if (len == 1)
        {
            for (int i = 0; i < dH; i++)
            {
                double al = D(a[i * ai]) * alpha;
                for (int j = 0; j < dW; j++)
                {
                    double s0 = al * D(b[j * bj]);
                    d[(i * ds) + j] = F<T>(hasC ? s0 + (D(c[(i * ci) + (j * cj)]) * beta) : s0);
                }
            }

            return;
        }

        if (dot)
        {
            // gemmImpl: a block product with k-blocks smaller than len sums each k-block in two lanes.
            int dk0 = len;
            bool single = ((dH <= 64 || dW <= 64) && len <= 10000) || len <= 10 || (dW <= 128 && dH <= 128 && len <= 128);
            if (!single)
            {
                int dm0 = Math.Min(128, dH), dn0 = Math.Min(128, dW);
                dk0 = Math.Min(Math.Min(16384 / dm0, 16384 / dn0), len);
            }

            int bjEff = bj, bkEff = bk;
            for (int i = 0; i < dH; i++)
            {
                for (int j = 0; j < dW; j++)
                {
                    int ao = i * ai, bo = j * bjEff;
                    double s;
                    if (dk0 < len)
                    {
                        double acc = 0;
                        for (int k = 0, dk; k < len; k += dk)
                        {
                            dk = dk0;
                            if (k + dk >= len || (8 * (k + dk)) + dk > 8 * len)
                            {
                                dk = len - k;
                            }

                            double s0 = acc, s1 = 0;
                            int kk = 0;
                            for (; kk <= dk - 2; kk += 2)
                            {
                                s0 += D(a[ao + ((k + kk) * ak)]) * D(b[bo + ((k + kk) * bkEff)]);
                                s1 += D(a[ao + ((k + kk + 1) * ak)]) * D(b[bo + ((k + kk + 1) * bkEff)]);
                            }

                            for (; kk < dk; kk++)
                            {
                                s0 += D(a[ao + ((k + kk) * ak)]) * D(b[bo + ((k + kk) * bkEff)]);
                            }

                            acc = s0 + s1;
                        }

                        double t = alpha * acc;
                        d[(i * ds) + j] = F<T>(hasC ? t + (beta * D(c[(i * ci) + (j * cj)])) : t);
                        continue;
                    }

                    double q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                    int kq = 0;
                    for (; kq <= len - 4; kq += 4)
                    {
                        q0 += D(a[ao + (kq * ak)]) * D(b[bo + (kq * bkEff)]);
                        q1 += D(a[ao + ((kq + 1) * ak)]) * D(b[bo + ((kq + 1) * bkEff)]);
                        q2 += D(a[ao + ((kq + 2) * ak)]) * D(b[bo + ((kq + 2) * bkEff)]);
                        q3 += D(a[ao + ((kq + 3) * ak)]) * D(b[bo + ((kq + 3) * bkEff)]);
                    }

                    for (; kq < len; kq++)
                    {
                        q0 += D(a[ao + (kq * ak)]) * D(b[bo + (kq * bkEff)]);
                    }

                    s = (q0 + q1 + q2 + q3) * alpha;
                    d[(i * ds) + j] = F<T>(hasC ? s + (D(c[(i * ci) + (j * cj)]) * beta) : s);
                }
            }

            return;
        }

        // Sequential double sums over k for every element; B rows are contiguous here.
        double[]? rented = null;
        Span<double> accum = dW <= 512 ? stackalloc double[dW] : (rented = ArrayPool<double>.Shared.Rent(dW)).AsSpan(0, dW);
        for (int i = 0; i < dH; i++)
        {
            accum.Clear();
            for (int k = 0; k < len; k++)
            {
                AxpyWiden(accum, D(a[(i * ai) + (k * ak)]), b.Slice(k * bk, dW));
            }

            Span<T> drow = d.Slice(i * ds, dW);
            if (hasC)
            {
                for (int j = 0; j < dW; j++)
                {
                    double s0 = accum[j] * alpha;
                    drow[j] = F<T>(s0 + (D(c[(i * ci) + (j * cj)]) * beta));
                }
            }
            else
            {
                for (int j = 0; j < dW; j++)
                {
                    drow[j] = F<T>(accum[j] * alpha);
                }
            }
        }

        if (rented is not null)
        {
            ArrayPool<double>.Shared.Return(rented);
        }
    }

    /// <summary><c>acc[j] += s * (double)row[j]</c> for every j.</summary>
    private static void AxpyWiden<T>(Span<double> acc, double s, ReadOnlySpan<T> row)
        where T : unmanaged
    {
        int j = 0;
        int n = acc.Length;
        if (typeof(T) == typeof(double))
        {
            ReadOnlySpan<double> r = MemoryMarshal.Cast<T, double>(row);
            if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
            {
                var vs = new Vector<double>(s);
                ref double accRef = ref MemoryMarshal.GetReference(acc);
                ref double rRef = ref MemoryMarshal.GetReference(r);
                for (; j <= n - Vector<double>.Count; j += Vector<double>.Count)
                {
                    Vector<double> v = Vector.LoadUnsafe(ref accRef, (nuint)j) + (vs * Vector.LoadUnsafe(ref rRef, (nuint)j));
                    v.StoreUnsafe(ref accRef, (nuint)j);
                }
            }

            for (; j < n; j++)
            {
                acc[j] += s * r[j];
            }
        }
        else
        {
            ReadOnlySpan<float> r = MemoryMarshal.Cast<T, float>(row);
            if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
            {
                var vs = new Vector<double>(s);
                ref double accRef = ref MemoryMarshal.GetReference(acc);
                ref float rRef = ref MemoryMarshal.GetReference(r);
                int dc = Vector<double>.Count;
                for (; j <= n - Vector<float>.Count; j += Vector<float>.Count)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref rRef, (nuint)j), out Vector<double> lo, out Vector<double> hi);
                    (Vector.LoadUnsafe(ref accRef, (nuint)j) + (vs * lo)).StoreUnsafe(ref accRef, (nuint)j);
                    (Vector.LoadUnsafe(ref accRef, (nuint)(j + dc)) + (vs * hi)).StoreUnsafe(ref accRef, (nuint)(j + dc));
                }
            }

            for (; j < n; j++)
            {
                acc[j] += s * r[j];
            }
        }
    }

    /// <summary>
    /// Reference xGEMM element order through <c>cblas_?gemm(CblasRowMajor, ...)</c>: every output
    /// element is rounded to <c>T</c> after each term, in increasing k order.
    /// </summary>
    private static void ReferenceBlas<T>(Mat<T> am, Mat<T> bm, double alphaD, Mat<T>? cm, double betaD, Mat<T> dm, int flags, int len)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T alpha = F<T>(alphaD), beta = F<T>(betaD);
        int dH = dm.Height, dW = dm.Width, ds = dm.Stride;
        Span<T> d = WholeMutable(dm);
        bool aT = (flags & 1) != 0, bT = (flags & 2) != 0, cT = (flags & 4) != 0;

        // lapack_gemm copies C (or its transpose) into D first.
        if (cm is not null)
        {
            ReadOnlySpan<T> c = Whole(cm);
            int ci = cT ? 1 : cm.Stride, cj = cT ? cm.Stride : 1;
            for (int i = 0; i < dH; i++)
            {
                Span<T> drow = d.Slice(i * ds, dW);
                for (int j = 0; j < dW; j++)
                {
                    drow[j] = c[(i * ci) + (j * cj)];
                }
            }
        }

        if ((alpha == T.Zero || len == 0) && beta == T.One)
        {
            return;
        }

        if (alpha == T.Zero)
        {
            for (int i = 0; i < dH; i++)
            {
                Span<T> drow = d.Slice(i * ds, dW);
                for (int j = 0; j < dW; j++)
                {
                    drow[j] = beta == T.Zero ? T.Zero : beta * drow[j];
                }
            }

            return;
        }

        ReadOnlySpan<T> a = Whole(am);
        ReadOnlySpan<T> b = Whole(bm);
        int ai = aT ? 1 : am.Stride, ak = aT ? am.Stride : 1;
        bool axpy = !bT;

        T[] pBuf = ArrayPool<T>.Shared.Rent(dH * len);
        T[]? rBuf = null;
        T[]? cBuf = null;
        try
        {
            // P[i, k]: the scalar factor of each term; R row k: the vector factor of each term.
            for (int i = 0; i < dH; i++)
            {
                for (int k = 0; k < len; k++)
                {
                    T v = a[(i * ai) + (k * ak)];
                    pBuf[(i * len) + k] = axpy ? alpha * v : v;
                }
            }

            ReadOnlySpan<T> r;
            int rs;
            if (axpy)
            {
                r = b;
                rs = bm.Stride;
            }
            else
            {
                rBuf = ArrayPool<T>.Shared.Rent(len * dW);
                for (int j = 0; j < dW; j++)
                {
                    ReadOnlySpan<T> brow = b.Slice(j * bm.Stride, len);
                    for (int k = 0; k < len; k++)
                    {
                        rBuf[(k * dW) + j] = brow[k];
                    }
                }

                r = rBuf.AsSpan(0, len * dW);
                rs = dW;
            }

            if (axpy)
            {
                for (int i = 0; i < dH; i++)
                {
                    Span<T> drow = d.Slice(i * ds, dW);
                    if (beta == T.Zero)
                    {
                        drow.Clear();
                    }
                    else if (beta != T.One)
                    {
                        for (int j = 0; j < dW; j++)
                        {
                            drow[j] = beta * drow[j];
                        }
                    }
                }
            }
            else if (beta != T.Zero)
            {
                cBuf = ArrayPool<T>.Shared.Rent(dH * dW);
                for (int i = 0; i < dH; i++)
                {
                    d.Slice(i * ds, dW).CopyTo(cBuf.AsSpan(i * dW, dW));
                    d.Slice(i * ds, dW).Clear();
                }
            }
            else
            {
                for (int i = 0; i < dH; i++)
                {
                    d.Slice(i * ds, dW).Clear();
                }
            }

            var job = new BlasJob<T>(pBuf, len, rBuf, r.Length, bm, rs, axpy, dm, dW);
            int groups = (dH + RowTile - 1) / RowTile;
            ParallelRows.For(groups, (long)RowTile * len * dW / Math.Max(1, Vector<T>.Count), (g0, g1) => BlasStripe(job, g0, g1));

            if (!axpy)
            {
                for (int i = 0; i < dH; i++)
                {
                    Span<T> drow = d.Slice(i * ds, dW);
                    if (beta == T.Zero)
                    {
                        for (int j = 0; j < dW; j++)
                        {
                            drow[j] = alpha * drow[j];
                        }
                    }
                    else
                    {
                        Span<T> crow = cBuf.AsSpan(i * dW, dW);
                        for (int j = 0; j < dW; j++)
                        {
                            drow[j] = (alpha * drow[j]) + (beta * crow[j]);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(pBuf);
            if (rBuf is not null)
            {
                ArrayPool<T>.Shared.Return(rBuf);
            }

            if (cBuf is not null)
            {
                ArrayPool<T>.Shared.Return(cBuf);
            }
        }
    }

    private sealed class BlasJob<T>(T[] p, int len, T[]? rPacked, int rLength, Mat<T> b, int rs, bool axpy, Mat<T> d, int dW)
        where T : unmanaged
    {
        public readonly T[] P = p;
        public readonly int Len = len;
        public readonly T[]? RPacked = rPacked;
        public readonly int RLength = rLength;
        public readonly Mat<T> B = b;
        public readonly int Rs = rs;
        public readonly bool Axpy = axpy;
        public readonly Mat<T> D = d;
        public readonly int DW = dW;

        public ReadOnlySpan<T> R => RPacked is not null ? RPacked.AsSpan(0, RLength) : Whole(B);
    }

    private static void BlasStripe<T>(BlasJob<T> job, int g0, int g1)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int dW = job.DW, len = job.Len, rs = job.Rs, ds = job.D.Stride;
        int r0 = g0 * RowTile, r1 = Math.Min(job.D.Height, g1 * RowTile);
        ReadOnlySpan<T> p = job.P;
        ReadOnlySpan<T> r = job.R;
        Span<T> d = WholeMutable(job.D);

        for (int k0 = 0; k0 < len; k0 += KPanel)
        {
            int k1 = Math.Min(len, k0 + KPanel);
            for (int j0 = 0; j0 < dW; j0 += NPanel)
            {
                int j1 = Math.Min(dW, j0 + NPanel);
                int i = r0;
                for (; i + RowTile <= r1; i += RowTile)
                {
                    Tile4(p, len, r, rs, d, ds, i, k0, k1, j0, j1);
                }

                for (; i < r1; i++)
                {
                    Tile1(p, len, r, rs, d, ds, i, k0, k1, j0, j1);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Tile4<T>(ReadOnlySpan<T> p, int len, ReadOnlySpan<T> r, int rs, Span<T> d, int ds, int i, int k0, int k1, int j0, int j1)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int vc = Vector<T>.Count;
        ref T pRef = ref MemoryMarshal.GetReference(p);
        ref T rRef = ref MemoryMarshal.GetReference(r);
        ref T dRef = ref MemoryMarshal.GetReference(d);
        nuint d0 = (nuint)(i * ds), d1 = (nuint)((i + 1) * ds), d2 = (nuint)((i + 2) * ds), d3 = (nuint)((i + 3) * ds);
        int p0 = i * len, p1 = (i + 1) * len, p2 = (i + 2) * len, p3 = (i + 3) * len;
        int j = j0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= j1 - (2 * vc); j += 2 * vc)
            {
                nuint jj = (nuint)j, jv = (nuint)(j + vc);
                Vector<T> a0 = Vector.LoadUnsafe(ref dRef, d0 + jj), b0 = Vector.LoadUnsafe(ref dRef, d0 + jv);
                Vector<T> a1 = Vector.LoadUnsafe(ref dRef, d1 + jj), b1 = Vector.LoadUnsafe(ref dRef, d1 + jv);
                Vector<T> a2 = Vector.LoadUnsafe(ref dRef, d2 + jj), b2 = Vector.LoadUnsafe(ref dRef, d2 + jv);
                Vector<T> a3 = Vector.LoadUnsafe(ref dRef, d3 + jj), b3 = Vector.LoadUnsafe(ref dRef, d3 + jv);
                for (int k = k0; k < k1; k++)
                {
                    nuint ro = (nuint)(k * rs);
                    Vector<T> x0 = Vector.LoadUnsafe(ref rRef, ro + jj);
                    Vector<T> x1 = Vector.LoadUnsafe(ref rRef, ro + jv);
                    var s = new Vector<T>(Unsafe.Add(ref pRef, p0 + k));
                    a0 += s * x0;
                    b0 += s * x1;
                    s = new Vector<T>(Unsafe.Add(ref pRef, p1 + k));
                    a1 += s * x0;
                    b1 += s * x1;
                    s = new Vector<T>(Unsafe.Add(ref pRef, p2 + k));
                    a2 += s * x0;
                    b2 += s * x1;
                    s = new Vector<T>(Unsafe.Add(ref pRef, p3 + k));
                    a3 += s * x0;
                    b3 += s * x1;
                }

                a0.StoreUnsafe(ref dRef, d0 + jj);
                b0.StoreUnsafe(ref dRef, d0 + jv);
                a1.StoreUnsafe(ref dRef, d1 + jj);
                b1.StoreUnsafe(ref dRef, d1 + jv);
                a2.StoreUnsafe(ref dRef, d2 + jj);
                b2.StoreUnsafe(ref dRef, d2 + jv);
                a3.StoreUnsafe(ref dRef, d3 + jj);
                b3.StoreUnsafe(ref dRef, d3 + jv);
            }

            for (; j <= j1 - vc; j += vc)
            {
                nuint jj = (nuint)j;
                Vector<T> a0 = Vector.LoadUnsafe(ref dRef, d0 + jj);
                Vector<T> a1 = Vector.LoadUnsafe(ref dRef, d1 + jj);
                Vector<T> a2 = Vector.LoadUnsafe(ref dRef, d2 + jj);
                Vector<T> a3 = Vector.LoadUnsafe(ref dRef, d3 + jj);
                for (int k = k0; k < k1; k++)
                {
                    Vector<T> x0 = Vector.LoadUnsafe(ref rRef, (nuint)(k * rs) + jj);
                    a0 += new Vector<T>(Unsafe.Add(ref pRef, p0 + k)) * x0;
                    a1 += new Vector<T>(Unsafe.Add(ref pRef, p1 + k)) * x0;
                    a2 += new Vector<T>(Unsafe.Add(ref pRef, p2 + k)) * x0;
                    a3 += new Vector<T>(Unsafe.Add(ref pRef, p3 + k)) * x0;
                }

                a0.StoreUnsafe(ref dRef, d0 + jj);
                a1.StoreUnsafe(ref dRef, d1 + jj);
                a2.StoreUnsafe(ref dRef, d2 + jj);
                a3.StoreUnsafe(ref dRef, d3 + jj);
            }
        }

        for (; j < j1; j++)
        {
            T a0 = d[(i * ds) + j], a1 = d[((i + 1) * ds) + j], a2 = d[((i + 2) * ds) + j], a3 = d[((i + 3) * ds) + j];
            for (int k = k0; k < k1; k++)
            {
                T x = r[(k * rs) + j];
                a0 += p[p0 + k] * x;
                a1 += p[p1 + k] * x;
                a2 += p[p2 + k] * x;
                a3 += p[p3 + k] * x;
            }

            d[(i * ds) + j] = a0;
            d[((i + 1) * ds) + j] = a1;
            d[((i + 2) * ds) + j] = a2;
            d[((i + 3) * ds) + j] = a3;
        }
    }

    private static void Tile1<T>(ReadOnlySpan<T> p, int len, ReadOnlySpan<T> r, int rs, Span<T> d, int ds, int i, int k0, int k1, int j0, int j1)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int vc = Vector<T>.Count;
        ref T rRef = ref MemoryMarshal.GetReference(r);
        ref T dRef = ref MemoryMarshal.GetReference(d);
        nuint d0 = (nuint)(i * ds);
        int p0 = i * len;
        int j = j0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= j1 - vc; j += vc)
            {
                nuint jj = (nuint)j;
                Vector<T> a0 = Vector.LoadUnsafe(ref dRef, d0 + jj);
                for (int k = k0; k < k1; k++)
                {
                    a0 += new Vector<T>(p[p0 + k]) * Vector.LoadUnsafe(ref rRef, (nuint)(k * rs) + jj);
                }

                a0.StoreUnsafe(ref dRef, d0 + jj);
            }
        }

        for (; j < j1; j++)
        {
            T a0 = d[(i * ds) + j];
            for (int k = k0; k < k1; k++)
            {
                a0 += p[p0 + k] * r[(k * rs) + j];
            }

            d[(i * ds) + j] = a0;
        }
    }
}
