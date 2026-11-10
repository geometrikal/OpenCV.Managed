// This implementation is ported from the OpenCV dense linear algebra.
//
// Original sources:
//   modules/core/src/lapack.cpp (determinant, invert, solve, eigen)
//   modules/core/src/matmul.dispatch.cpp (gemm, transform, perspectiveTransform, mulTransposed)
//   modules/core/src/matmul.simd.hpp (MulTransposedR, MulTransposedL)
//   modules/core/src/matrix_operations.cpp (setIdentity, trace, completeSymm)
//   modules/core/src/mathfuncs.cpp (solveCubic, solvePoly)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using System.Numerics;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core.Internal;
using static Geometrikal.OpenCV.Managed.Core.Internal.LapackScalar;

/// <summary>
/// Dense linear algebra on single-channel <c>float</c> and <c>double</c> matrices, mirroring the
/// OpenCV core functions <c>gemm</c>, <c>determinant</c>, <c>invert</c>, <c>solve</c>,
/// <c>eigen</c>, <c>solveCubic</c>, <c>solvePoly</c>, <c>transform</c>,
/// <c>perspectiveTransform</c>, <c>mulTransposed</c>, <c>setIdentity</c>, <c>trace</c> and
/// <c>completeSymm</c>.
/// </summary>
/// <remarks>
/// A matrix with <c>Height</c> rows and <c>Width</c> columns is a <see cref="Mat{T}"/> with one
/// channel. A vector is an n × 1 matrix. The functions do not modify their sources, except the
/// explicit in-place functions <see cref="CompleteSymm{T}"/> and <see cref="SetIdentity{T}"/>.
/// The results are equal to OpenCV 5.0 built with its internal LAPACK, except where the
/// documentation of a function states a tolerance. See <c>Docs/Notes/LinearAlgebraPortNotes.md</c>.
/// </remarks>
public static class LinearAlgebra
{
    /// <summary>
    /// Computes the generalized matrix product <c>alpha·op(a)·op(b) + beta·op(c)</c>
    /// (<c>cv::gemm</c>).
    /// </summary>
    /// <param name="a">First single-channel factor.</param>
    /// <param name="b">Second single-channel factor.</param>
    /// <param name="alpha">Factor of the product.</param>
    /// <param name="c">Optional added matrix. Ignored when <paramref name="beta"/> is 0.</param>
    /// <param name="beta">Factor of <paramref name="c"/>.</param>
    /// <param name="flags">Operand transpositions.</param>
    /// <returns>A new matrix with the rows of <c>op(a)</c> and the columns of <c>op(b)</c>.</returns>
    /// <remarks>
    /// For <c>float</c>, <paramref name="alpha"/> and <paramref name="beta"/> are rounded to
    /// <c>float</c>, as in OpenCV. When <paramref name="a"/> has fewer than 100 rows, the result
    /// is bit-exact with the OpenCV <c>gemmImpl</c> (double accumulation). From 100 rows, the
    /// result is bit-exact with the reference BLAS <c>sgemm</c>/<c>dgemm</c> that OpenCV calls
    /// through its internal LAPACK (accumulation in <c>T</c>). Two-channel (complex) matrices are
    /// not supported.
    /// </remarks>
    public static Mat<T> Gemm<T>(Mat<T> a, Mat<T> b, double alpha, Mat<T>? c, double beta, GemmFlags flags = GemmFlags.None)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        RequireSingleChannel(a, nameof(a));
        RequireSingleChannel(b, nameof(b));
        int f = (int)flags & 7;
        bool aT = (f & 1) != 0, bT = (f & 2) != 0, cT = (f & 4) != 0;
        int dH = aT ? a.Width : a.Height;
        int len = aT ? a.Height : a.Width;
        int dW = bT ? b.Height : b.Width;
        if ((bT ? b.Width : b.Height) != len)
        {
            throw new ArgumentException("The inner dimensions of op(a) and op(b) differ.", nameof(b));
        }

        if (IsFloat<T>())
        {
            alpha = (float)alpha;
        }

        Mat<T>? cEff = null;
        if (beta != 0.0 && c is not null)
        {
            RequireSingleChannel(c, nameof(c));
            if ((cT ? c.Width : c.Height) != dH || (cT ? c.Height : c.Width) != dW)
            {
                throw new ArgumentException("op(c) must have the size of the product.", nameof(c));
            }

            cEff = c;
        }

        if (IsFloat<T>())
        {
            beta = (float)beta;
        }

        if (beta == 0.0)
        {
            cEff = null;
        }

        var d = new Mat<T>(dW, dH);
        GemmKernels.Run(a, b, alpha, cEff, beta, d, f, len);
        return d;
    }

    /// <summary>Returns the determinant of a square matrix (<c>cv::determinant</c>).</summary>
    /// <param name="source">Square single-channel matrix.</param>
    /// <returns>The determinant, computed in <c>double</c>.</returns>
    /// <remarks>Sizes up to 3 × 3 use the explicit formulas of OpenCV; larger sizes use LU decomposition.</remarks>
    public static double Determinant<T>(Mat<T> source)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        RequireSquare(source, nameof(source));
        int n = source.Height;
        if (n == 1)
        {
            return D(source.At(0, 0));
        }

        if (n == 2)
        {
            return Det2(source);
        }

        if (n == 3)
        {
            return Det3(source);
        }

        Span<T> a = n <= 16 ? stackalloc T[n * n] : new T[n * n];
        CopyPacked(source, a);
        double result = LapackDecomp.Lu<T>(a, n, n, default, 0, 0);
        if (result != 0)
        {
            for (int i = 0; i < n; i++)
            {
                result *= D(a[(i * n) + i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the inverse or pseudo-inverse of a matrix (<c>cv::invert</c>).
    /// </summary>
    /// <param name="source">Single-channel matrix. Must be square unless <paramref name="method"/> is <see cref="DecompTypes.Svd"/>.</param>
    /// <param name="inverse">The inverse (columns × rows). Zero when the inversion fails.</param>
    /// <param name="method"><see cref="DecompTypes.Lu"/>, <see cref="DecompTypes.Cholesky"/>, <see cref="DecompTypes.Svd"/> or <see cref="DecompTypes.Eig"/>.</param>
    /// <returns>
    /// For LU and Cholesky: 1 on success, 0 when the matrix is singular (or not positive definite).
    /// For SVD and EIG: the inverse condition number <c>w[n-1]/w[0]</c>, or 0 when <c>w[0]</c> is
    /// below the machine epsilon.
    /// </returns>
    /// <remarks>
    /// For SVD with fewer rows than columns, OpenCV reads <c>w[n-1]</c> past the singular values
    /// (into the first elements of <c>Vᵀ</c>); the port returns the same value.
    /// </remarks>
    public static double Invert<T>(Mat<T> source, out Mat<T> inverse, DecompTypes method = DecompTypes.Lu)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(source);
        RequireSingleChannel(source, nameof(source));
        int m = source.Height, n = source.Width;
        double eps = IsFloat<T>() ? FltEpsilon : DblEpsilon;

        if (method == DecompTypes.Svd)
        {
            var svd = new Svd<T>(source);
            inverse = svd.BackSubst(null);
            int nm = Math.Min(m, n);
            T w0 = svd.W.At(0, 0);
            T wLast = n - 1 < nm ? svd.W.At(n - 1, 0) : svd.Vt.GetReadOnlySpan()[n - 1 - nm];
            return D(w0) >= eps ? D(wLast / w0) : 0;
        }

        if (m != n)
        {
            throw new ArgumentException("The matrix must be square for this method.", nameof(source));
        }

        if (method == DecompTypes.Eig)
        {
            Eigen(source, out Mat<T> w, out Mat<T> vt);
            Mat<T> u = MatrixTransforms.Transpose(vt);
            inverse = Svd<T>.BackSubst(w, u, vt, null);
            T w0 = w.At(0, 0);
            return D(w0) >= eps ? D(w.At(n - 1, 0) / w0) : 0;
        }

        if (method != DecompTypes.Lu && method != DecompTypes.Cholesky)
        {
            throw new ArgumentException("Unsupported decomposition method for Invert.", nameof(method));
        }

        var dst = new Mat<T>(n, n);
        bool result = false;
        if (n <= 3)
        {
            result = InvertSmall(source, dst);
            if (!result)
            {
                dst.Clear();
            }

            inverse = dst;
            return result ? 1 : 0;
        }

        Span<T> a = n <= 16 ? stackalloc T[n * n] : new T[n * n];
        CopyPacked(source, a);
        T[] id = dst.DangerousGetArray();
        for (int i = 0; i < n; i++)
        {
            id[(i * n) + i] = T.One;
        }

        result = method == DecompTypes.Lu
            ? LapackDecomp.Lu<T>(a, n, n, id, n, n) != 0
            : LapackDecomp.Cholesky<T>(a, n, n, id, n, n);
        if (!result)
        {
            dst.Clear();
        }

        inverse = dst;
        return result ? 1 : 0;
    }

    /// <summary>
    /// Solves the linear system or least-squares problem <c>source·x = rhs</c> (<c>cv::solve</c>).
    /// </summary>
    /// <param name="source">Single-channel m × n coefficient matrix (m ≥ n).</param>
    /// <param name="rhs">Single-channel m × nb right-hand side.</param>
    /// <param name="solution">The n × nb solution. Zero when the method fails.</param>
    /// <param name="flags">
    /// Decomposition method, optionally combined with <see cref="DecompTypes.Normal"/>. LU and
    /// Cholesky need a square <paramref name="source"/> unless <see cref="DecompTypes.Normal"/> is set.
    /// </param>
    /// <returns>False when <paramref name="source"/> is singular for LU, not positive definite for Cholesky, or rank deficient for QR.</returns>
    /// <remarks>
    /// SVD and QR give the least-squares solution of overdetermined systems. EIG without
    /// <see cref="DecompTypes.Normal"/> needs a square <paramref name="source"/>.
    /// </remarks>
    public static bool Solve<T>(Mat<T> source, Mat<T> rhs, out Mat<T> solution, DecompTypes flags = DecompTypes.Lu)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(rhs);
        RequireSingleChannel(source, nameof(source));
        RequireSingleChannel(rhs, nameof(rhs));
        bool isNormal = (flags & DecompTypes.Normal) != 0;
        DecompTypes method = flags & ~DecompTypes.Normal;
        if (method is not (DecompTypes.Lu or DecompTypes.Svd or DecompTypes.Eig or DecompTypes.Cholesky or DecompTypes.Qr))
        {
            throw new ArgumentException("Unsupported decomposition method.", nameof(flags));
        }

        if ((method == DecompTypes.Lu || method == DecompTypes.Cholesky) && !isNormal && source.Height != source.Width)
        {
            throw new ArgumentException("LU and Cholesky need a square matrix without DecompTypes.Normal.", nameof(source));
        }

        int m = source.Height, n = source.Width, nb = rhs.Width;
        if ((method == DecompTypes.Lu || method == DecompTypes.Cholesky) && !isNormal && m <= 3 && nb == 1)
        {
            if (rhs.Height != m)
            {
                throw new ArgumentException("The right-hand side must have the rows of the matrix.", nameof(rhs));
            }

            solution = new Mat<T>(1, n);
            bool ok = SolveSmall(source, rhs, solution);
            return ok;
        }

        if (m < n)
        {
            throw new ArgumentException("The function can not solve under-determined linear systems.", nameof(source));
        }

        if (m != rhs.Height)
        {
            throw new ArgumentException("The right-hand side must have the rows of the matrix.", nameof(rhs));
        }

        int mEff = m;
        if (m == n)
        {
            isNormal = false;
        }
        else if (isNormal)
        {
            mEff = n;
            if (method == DecompTypes.Svd)
            {
                method = DecompTypes.Eig;
            }
        }

        if (method == DecompTypes.Eig && !isNormal && m != n)
        {
            throw new NotSupportedException("EIG without DecompTypes.Normal needs a square matrix.");
        }

        bool stackA = !isNormal && method is DecompTypes.Lu or DecompTypes.Cholesky && n <= 16;
        Span<T> aSmall = stackA ? stackalloc T[n * n] : default;
        T[] a;
        if (stackA)
        {
            CopyPacked(source, aSmall);
            a = [];
        }
        else if (isNormal)
        {
            a = MulTransposedPacked(source, true);
        }
        else if (method != DecompTypes.Svd)
        {
            a = Pack(source);
        }
        else
        {
            a = Pack(MatrixTransforms.Transpose(source));
        }

        Span<T> aSpan = stackA ? aSmall : a;

        T[] dst;
        Mat<T> src2 = rhs;
        if (!isNormal)
        {
            dst = method is DecompTypes.Lu or DecompTypes.Cholesky ? Pack(rhs) : new T[n * nb];
        }
        else
        {
            var tmp = new Mat<T>(nb, n);
            GemmKernels.Run(source, rhs, 1.0, null, 0.0, tmp, 1, m);
            if (method is DecompTypes.Lu or DecompTypes.Cholesky)
            {
                dst = tmp.DangerousGetArray();
            }
            else
            {
                dst = new T[n * nb];
                src2 = tmp;
            }
        }

        bool result;
        if (method == DecompTypes.Lu)
        {
            result = LapackDecomp.Lu<T>(aSpan, n, n, dst, nb, nb) != 0;
        }
        else if (method == DecompTypes.Cholesky)
        {
            result = LapackDecomp.Cholesky<T>(aSpan, n, n, dst, nb, nb);
        }
        else if (method == DecompTypes.Qr)
        {
            T[] rhsBuf;
            if (isNormal || m == n)
            {
                Pack(src2).CopyTo(dst, 0);
                rhsBuf = dst;
            }
            else
            {
                rhsBuf = Pack(src2);
            }

            result = LapackDecomp.Qr<T>(a, n, mEff, n, nb, rhsBuf, nb) != 0;
            if (!ReferenceEquals(rhsBuf, dst))
            {
                Array.Copy(rhsBuf, dst, n * nb);
            }
        }
        else
        {
            T[] v = new T[n * n];
            T[] w = new T[n];
            T[] u;
            if (method == DecompTypes.Eig)
            {
                LapackDecomp.Jacobi<T>(a, n, w, v, n, n);
                u = v;
            }
            else
            {
                LapackDecomp.JacobiSvd<T>(a, mEff, w, v, n, mEff, n, -1);
                u = a;
            }

            int ldu = method == DecompTypes.Eig ? n : mEff;
            T[] b2 = Pack(src2);
            LapackDecomp.SvBkSb<T>(mEff, n, w, 1, u, ldu, true, v, n, true, b2, nb, nb, dst, nb);
            result = true;
        }

        if (!result)
        {
            Array.Clear(dst);
        }

        solution = new Mat<T>(dst, nb, n);
        return result;
    }

    /// <summary>
    /// Computes the eigenvalues and eigenvectors of a symmetric matrix (<c>cv::eigen</c>,
    /// Jacobi method).
    /// </summary>
    /// <param name="source">Symmetric square single-channel matrix. Only the upper triangle is read.</param>
    /// <param name="eigenvalues">n × 1 eigenvalues in descending order.</param>
    /// <param name="eigenvectors">n × n matrix; row i is the eigenvector of eigenvalue i.</param>
    /// <returns>True. The Jacobi method always returns a result.</returns>
    public static bool Eigen<T>(Mat<T> source, out Mat<T> eigenvalues, out Mat<T> eigenvectors)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        RequireSquare(source, nameof(source));
        int n = source.Height;
        T[] a = Pack(source);
        T[] w = new T[n];
        T[] v = new T[n * n];
        LapackDecomp.Jacobi<T>(a, n, w, v, n, n);
        eigenvalues = new Mat<T>(w, 1, n);
        eigenvectors = new Mat<T>(v, n, n);
        return true;
    }

    /// <summary>
    /// Computes the eigenvalues of a symmetric matrix (<c>cv::eigen</c> without eigenvectors).
    /// </summary>
    /// <param name="source">Symmetric square single-channel matrix. Only the upper triangle is read.</param>
    /// <param name="eigenvalues">n × 1 eigenvalues in descending order.</param>
    /// <returns>True.</returns>
    public static bool Eigen<T>(Mat<T> source, out Mat<T> eigenvalues)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        RequireSquare(source, nameof(source));
        int n = source.Height;
        T[] a = Pack(source);
        T[] w = new T[n];
        LapackDecomp.Jacobi<T>(a, n, w, default, 0, n);
        eigenvalues = new Mat<T>(w, 1, n);
        return true;
    }

    /// <summary>
    /// Computes the singular value decomposition (<c>cv::SVDecomp</c>). See <see cref="Svd{T}"/>.
    /// </summary>
    public static void SVDecomp<T>(Mat<T> source, out Mat<T> w, out Mat<T> u, out Mat<T> vt, SvdFlags flags = SvdFlags.None)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var svd = new Svd<T>(source, flags);
        w = svd.W;
        u = svd.U;
        vt = svd.Vt;
    }

    /// <summary>Back substitution with a singular value decomposition (<c>cv::SVBackSubst</c>). See <see cref="Svd{T}.BackSubst(Mat{T}, Mat{T}, Mat{T}, Mat{T}?)"/>.</summary>
    public static Mat<T> SVBackSubst<T>(Mat<T> w, Mat<T> u, Mat<T> vt, Mat<T>? rhs)
        where T : unmanaged, IFloatingPointIeee754<T>
        => Svd<T>.BackSubst(w, u, vt, rhs);

    /// <summary>
    /// Finds the real roots of a cubic equation (<c>cv::solveCubic</c>).
    /// </summary>
    /// <param name="coefficients">
    /// Three coefficients (<c>x³ + c0·x² + c1·x + c2</c>) or four coefficients
    /// (<c>c0·x³ + c1·x² + c2·x + c3</c>).
    /// </param>
    /// <param name="roots">Receives three values; only the first (return value) values are roots.</param>
    /// <returns>The number of real roots: 0, 1, 2 or 3, or -1 when every value is a root.</returns>
    public static int SolveCubic(ReadOnlySpan<double> coefficients, Span<double> roots)
    {
        if (coefficients.Length is not (3 or 4))
        {
            throw new ArgumentException("SolveCubic needs 3 or 4 coefficients.", nameof(coefficients));
        }

        if (roots.Length < 3)
        {
            throw new ArgumentException("SolveCubic needs room for 3 roots.", nameof(roots));
        }

        int i = coefficients.Length == 4 ? 0 : -1;
        double a0 = i == 0 ? coefficients[0] : 1.0;
        int n = PolySolvers.SolveCubic(a0, coefficients[i + 1], coefficients[i + 2], coefficients[i + 3], out roots[0], out roots[1], out roots[2]);
        return n;
    }

    /// <summary>
    /// Finds the real roots of a cubic equation with <c>float</c> coefficients and roots
    /// (<c>cv::solveCubic</c> with <c>CV_32F</c>). The computation uses <c>double</c>.
    /// </summary>
    /// <param name="coefficients">Three or four coefficients, as in <see cref="SolveCubic(ReadOnlySpan{double}, Span{double})"/>.</param>
    /// <param name="roots">Receives three values.</param>
    /// <returns>The number of real roots, or -1 when every value is a root.</returns>
    public static int SolveCubic(ReadOnlySpan<float> coefficients, Span<float> roots)
    {
        Span<double> c = stackalloc double[coefficients.Length];
        for (int k = 0; k < c.Length; k++)
        {
            c[k] = coefficients[k];
        }

        Span<double> r = stackalloc double[3];
        if (roots.Length < 3)
        {
            throw new ArgumentException("SolveCubic needs room for 3 roots.", nameof(roots));
        }

        int n = SolveCubic(c, r);
        roots[0] = (float)r[0];
        roots[1] = (float)r[1];
        roots[2] = (float)r[2];
        return n;
    }

    /// <summary>
    /// Finds the real or complex roots of a polynomial (<c>cv::solvePoly</c>, Durand-Kerner method).
    /// </summary>
    /// <param name="coefficients">Real coefficients; index i is the coefficient of <c>xⁱ</c>.</param>
    /// <param name="roots">Receives <c>coefficients.Length - 1</c> roots.</param>
    /// <param name="maxIterations">Maximum number of iterations; values ≤ 0 mean 1000.</param>
    /// <returns>The largest root change of the last iteration.</returns>
    /// <remarks>
    /// When the leading coefficients are zero, OpenCV fills the surplus roots from an
    /// uninitialized buffer slot; the port fills them with zero.
    /// </remarks>
    public static double SolvePoly(ReadOnlySpan<double> coefficients, Span<Complex> roots, int maxIterations = 300)
    {
        if (coefficients.Length < 2)
        {
            throw new ArgumentException("SolvePoly needs at least 2 coefficients.", nameof(coefficients));
        }

        double[] c = new double[coefficients.Length * 2];
        for (int k = 0; k < coefficients.Length; k++)
        {
            c[2 * k] = coefficients[k];
        }

        return SolvePolyCore(c, roots, maxIterations, true);
    }

    /// <summary>
    /// Finds the roots of a polynomial with complex coefficients (<c>cv::solvePoly</c> with a
    /// two-channel input).
    /// </summary>
    /// <param name="coefficients">Complex coefficients; index i is the coefficient of <c>xⁱ</c>.</param>
    /// <param name="roots">Receives <c>coefficients.Length - 1</c> roots.</param>
    /// <param name="maxIterations">Maximum number of iterations; values ≤ 0 mean 1000.</param>
    /// <returns>The largest root change of the last iteration.</returns>
    public static double SolvePoly(ReadOnlySpan<Complex> coefficients, Span<Complex> roots, int maxIterations = 300)
    {
        if (coefficients.Length < 2)
        {
            throw new ArgumentException("SolvePoly needs at least 2 coefficients.", nameof(coefficients));
        }

        double[] c = new double[coefficients.Length * 2];
        for (int k = 0; k < coefficients.Length; k++)
        {
            c[2 * k] = coefficients[k].Real;
            c[(2 * k) + 1] = coefficients[k].Imaginary;
        }

        return SolvePolyCore(c, roots, maxIterations, false);
    }

    /// <summary>
    /// Applies a perspective matrix to 2D points (<c>cv::perspectiveTransform</c>).
    /// </summary>
    /// <param name="source">Source points.</param>
    /// <param name="matrix">3 × 3 single-channel matrix.</param>
    /// <returns>
    /// New points <c>(x', y') = (X/W, Y/W)</c> with <c>(X, Y, W) = matrix·(x, y, 1)</c>. A point
    /// with <c>|W| ≤ FLT_EPSILON</c> becomes (0, 0).
    /// </returns>
    public static Point2f[] PerspectiveTransform(ReadOnlySpan<Point2f> source, Mat<double> matrix)
    {
        var dst = GC.AllocateUninitializedArray<Point2f>(source.Length);
        PerspectiveTransform(source, dst, matrix);
        return dst;
    }

    /// <summary>Applies a 3 × 3 perspective matrix to 2D points into <paramref name="destination"/>. See <see cref="PerspectiveTransform(ReadOnlySpan{Point2f}, Mat{double})"/>.</summary>
    public static void PerspectiveTransform(ReadOnlySpan<Point2f> source, Span<Point2f> destination, Mat<double> matrix)
        => PerspectivePoints(MemoryMarshal.Cast<Point2f, float>(source), MemoryMarshal.Cast<Point2f, float>(destination), source.Length, destination.Length, matrix, 2);

    /// <summary>Applies a 3 × 3 perspective matrix to double-precision 2D points. See <see cref="PerspectiveTransform(ReadOnlySpan{Point2f}, Mat{double})"/>.</summary>
    public static Point2d[] PerspectiveTransform(ReadOnlySpan<Point2d> source, Mat<double> matrix)
    {
        var dst = GC.AllocateUninitializedArray<Point2d>(source.Length);
        PerspectiveTransform(source, dst, matrix);
        return dst;
    }

    /// <summary>Applies a 3 × 3 perspective matrix to double-precision 2D points into <paramref name="destination"/>.</summary>
    public static void PerspectiveTransform(ReadOnlySpan<Point2d> source, Span<Point2d> destination, Mat<double> matrix)
        => PerspectivePoints(MemoryMarshal.Cast<Point2d, double>(source), MemoryMarshal.Cast<Point2d, double>(destination), source.Length, destination.Length, matrix, 2);

    /// <summary>Applies a 4 × 4 perspective matrix to 3D points. A point with <c>|W| ≤ FLT_EPSILON</c> becomes (0, 0, 0).</summary>
    public static Point3f[] PerspectiveTransform(ReadOnlySpan<Point3f> source, Mat<double> matrix)
    {
        var dst = GC.AllocateUninitializedArray<Point3f>(source.Length);
        PerspectiveTransform(source, dst, matrix);
        return dst;
    }

    /// <summary>Applies a 4 × 4 perspective matrix to 3D points into <paramref name="destination"/>.</summary>
    public static void PerspectiveTransform(ReadOnlySpan<Point3f> source, Span<Point3f> destination, Mat<double> matrix)
        => PerspectivePoints(MemoryMarshal.Cast<Point3f, float>(source), MemoryMarshal.Cast<Point3f, float>(destination), source.Length, destination.Length, matrix, 3);

    /// <summary>Applies a 4 × 4 perspective matrix to double-precision 3D points.</summary>
    public static Point3d[] PerspectiveTransform(ReadOnlySpan<Point3d> source, Mat<double> matrix)
    {
        var dst = GC.AllocateUninitializedArray<Point3d>(source.Length);
        PerspectiveTransform(source, dst, matrix);
        return dst;
    }

    /// <summary>Applies a 4 × 4 perspective matrix to double-precision 3D points into <paramref name="destination"/>.</summary>
    public static void PerspectiveTransform(ReadOnlySpan<Point3d> source, Span<Point3d> destination, Mat<double> matrix)
        => PerspectivePoints(MemoryMarshal.Cast<Point3d, double>(source), MemoryMarshal.Cast<Point3d, double>(destination), source.Length, destination.Length, matrix, 3);

    /// <summary>
    /// Applies a perspective matrix to every element of a multi-channel matrix
    /// (<c>cv::perspectiveTransform</c>).
    /// </summary>
    /// <param name="source"><c>float</c> or <c>double</c> matrix with scn channels; each element is a point.</param>
    /// <param name="matrix">(dcn+1) × (scn+1) single-channel matrix.</param>
    /// <returns>A new matrix with the size of <paramref name="source"/> and dcn channels.</returns>
    /// <remarks>
    /// For channel combinations other than 2→2, 3→3 and 3→2, OpenCV multiplies the projected
    /// coordinates by W instead of dividing by W. The port reproduces this behaviour.
    /// </remarks>
    public static Mat<T> PerspectiveTransform<T>(Mat<T> source, Mat<double> matrix)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(matrix);
        RequireSingleChannel(matrix, nameof(matrix));
        int scn = source.Channels, dcn = matrix.Height - 1;
        if (matrix.Width != scn + 1 || dcn < 1)
        {
            throw new ArgumentException("The matrix must be (dcn+1) × (scn+1).", nameof(matrix));
        }

        double[] m = Pack(matrix);
        var dst = new Mat<T>(source.Width, source.Height, dcn);
        if (source.IsContinuous)
        {
            PointTransformKernels.Perspective(source.GetReadOnlySpan(), dst.GetSpan(), m, source.PixelCount, scn, dcn);
        }
        else
        {
            for (int y = 0; y < source.Height; y++)
            {
                PointTransformKernels.Perspective(source.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), m, source.Width, scn, dcn);
            }
        }

        return dst;
    }

    /// <summary>
    /// Applies an affine matrix to every element of a multi-channel matrix (<c>cv::transform</c>).
    /// </summary>
    /// <param name="source"><c>float</c> or <c>double</c> matrix with scn channels.</param>
    /// <param name="matrix">dcn × scn or dcn × (scn+1) single-channel matrix. The last column is the offset.</param>
    /// <returns>A new matrix with the size of <paramref name="source"/> and dcn channels.</returns>
    /// <remarks>
    /// For a <c>float</c> source the matrix is rounded to <c>float</c> and the arithmetic is in
    /// <c>float</c>, as in OpenCV. A one-channel source with a one-channel result (a
    /// <c>convertTo</c> in OpenCV) is not supported.
    /// </remarks>
    public static Mat<T> Transform<T>(Mat<T> source, Mat<double> matrix)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(matrix);
        RequireSingleChannel(matrix, nameof(matrix));
        int scn = source.Channels, dcn = matrix.Height;
        if (matrix.Width != scn && matrix.Width != scn + 1)
        {
            throw new ArgumentException("The matrix must be dcn × scn or dcn × (scn+1).", nameof(matrix));
        }

        if (scn == 1 && dcn == 1)
        {
            throw new NotSupportedException("A one-channel to one-channel transform is a convertTo in OpenCV; use Conversions.ConvertTo.");
        }

        T[] m = PrepareTransformMatrix<T>(matrix, scn, dcn, out bool isDiag);
        var dst = new Mat<T>(source.Width, source.Height, dcn);
        if (source.IsContinuous)
        {
            PointTransformKernels.Transform<T>(source.GetReadOnlySpan(), dst.GetSpan(), m, source.PixelCount, scn, dcn, isDiag);
        }
        else
        {
            for (int y = 0; y < source.Height; y++)
            {
                PointTransformKernels.Transform<T>(source.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), m, source.Width, scn, dcn, isDiag);
            }
        }

        return dst;
    }

    /// <summary>Applies a 2 × 2 or 2 × 3 affine matrix to 2D points (<c>cv::transform</c>; arithmetic in <c>float</c>).</summary>
    public static Point2f[] Transform(ReadOnlySpan<Point2f> source, Mat<double> matrix)
        => TransformPoints<Point2f, float>(source, matrix, 2, 2);

    /// <summary>Applies a 2 × 2 or 2 × 3 affine matrix to double-precision 2D points (<c>cv::transform</c>).</summary>
    public static Point2d[] Transform(ReadOnlySpan<Point2d> source, Mat<double> matrix)
        => TransformPoints<Point2d, double>(source, matrix, 2, 2);

    /// <summary>Applies a 3 × 3 or 3 × 4 affine matrix to 3D points (<c>cv::transform</c>; arithmetic in <c>float</c>).</summary>
    public static Point3f[] Transform(ReadOnlySpan<Point3f> source, Mat<double> matrix)
        => TransformPoints<Point3f, float>(source, matrix, 3, 3);

    /// <summary>Applies a 3 × 3 or 3 × 4 affine matrix to double-precision 3D points (<c>cv::transform</c>).</summary>
    public static Point3d[] Transform(ReadOnlySpan<Point3d> source, Mat<double> matrix)
        => TransformPoints<Point3d, double>(source, matrix, 3, 3);

    /// <summary>
    /// Computes <c>scale·srcᵀ·src</c> (<paramref name="aTa"/> true) or <c>scale·src·srcᵀ</c>
    /// (<c>cv::mulTransposed</c> without delta; destination type equal to the source type).
    /// </summary>
    /// <param name="source">Single-channel matrix.</param>
    /// <param name="aTa">True for <c>srcᵀ·src</c>; false for <c>src·srcᵀ</c>.</param>
    /// <param name="scale">Factor of the product.</param>
    /// <returns>A new symmetric matrix.</returns>
    public static Mat<T> MulTransposed<T>(Mat<T> source, bool aTa, double scale = 1.0)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(source);
        RequireSingleChannel(source, nameof(source));
        int n = aTa ? source.Width : source.Height;
        T[] data = MulTransposedPacked(source, aTa, scale);
        return new Mat<T>(data, n, n);
    }

    /// <summary>
    /// Sets a matrix to a scaled identity in place (<c>cv::setIdentity</c>): <paramref name="value"/>
    /// on the main diagonal and zero elsewhere. Non-square matrices are accepted.
    /// </summary>
    /// <param name="matrix">Single-channel matrix. This matrix is modified.</param>
    /// <param name="value">Diagonal value. For <c>float</c> it is rounded to <c>float</c>.</param>
    public static void SetIdentity<T>(Mat<T> matrix, double value = 1.0)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(matrix);
        RequireSingleChannel(matrix, nameof(matrix));
        T v = F<T>(value);
        for (int i = 0; i < matrix.Height; i++)
        {
            Span<T> row = matrix.GetRowSpan(i);
            row.Clear();
            if (i < matrix.Width)
            {
                row[i] = v;
            }
        }
    }

    /// <summary>Returns the sum of the main diagonal (<c>cv::trace</c>), accumulated in <c>double</c>.</summary>
    /// <param name="matrix">Single-channel matrix. Non-square matrices use min(rows, columns) diagonal elements.</param>
    public static double Trace<T>(Mat<T> matrix)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(matrix);
        RequireSingleChannel(matrix, nameof(matrix));
        int nm = Math.Min(matrix.Height, matrix.Width);
        double s = 0;
        for (int i = 0; i < nm; i++)
        {
            s += D(matrix.At(i, i));
        }

        return s;
    }

    /// <summary>
    /// Copies one triangle of a square matrix to the other triangle in place (<c>cv::completeSymm</c>).
    /// </summary>
    /// <param name="matrix">Square matrix (any element type and channel count). This matrix is modified.</param>
    /// <param name="lowerToUpper">True copies the lower triangle to the upper triangle; false copies the upper triangle to the lower triangle.</param>
    public static void CompleteSymm<T>(Mat<T> matrix, bool lowerToUpper = false)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Width != matrix.Height)
        {
            throw new ArgumentException("The matrix must be square.", nameof(matrix));
        }

        int rows = matrix.Height, cn = matrix.Channels;
        for (int i = 0; i < rows; i++)
        {
            int j0 = lowerToUpper ? i + 1 : 0;
            int j1 = lowerToUpper ? rows : i;
            Span<T> row = matrix.GetRowSpan(i);
            for (int j = j0; j < j1; j++)
            {
                matrix.GetReadOnlyRowSpan(j).Slice(i * cn, cn).CopyTo(row.Slice(j * cn, cn));
            }
        }
    }

    internal static T[] Pack<T>(Mat<T> m)
        where T : unmanaged
        => m.IsContinuous ? m.GetReadOnlySpan().ToArray() : m.ToArray();

    private static void CopyPacked<T>(Mat<T> m, Span<T> dst)
        where T : unmanaged
    {
        int w = m.RowLength;
        for (int i = 0; i < m.Height; i++)
        {
            m.GetReadOnlyRowSpan(i).CopyTo(dst.Slice(i * w, w));
        }
    }

    private static double SolvePolyCore(double[] c, Span<Complex> roots, int maxIterations, bool realInput)
    {
        int n0 = (c.Length / 2) - 1;
        if (roots.Length < n0)
        {
            throw new ArgumentException($"SolvePoly needs room for {n0} roots.", nameof(roots));
        }

        double[] r = new double[2 * n0];
        double maxDiff = PolySolvers.SolvePoly(c, r, maxIterations, realInput);
        for (int i = 0; i < n0; i++)
        {
            roots[i] = new Complex(r[2 * i], r[(2 * i) + 1]);
        }

        return maxDiff;
    }

    private static void PerspectivePoints<T>(ReadOnlySpan<T> src, Span<T> dst, int count, int dstCount, Mat<double> matrix, int cn)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        RequireSingleChannel(matrix, nameof(matrix));
        if (matrix.Width != cn + 1 || matrix.Height != cn + 1)
        {
            throw new ArgumentException($"The matrix must be {cn + 1} × {cn + 1}.", nameof(matrix));
        }

        if (dstCount < count)
        {
            throw new ArgumentException("The destination is shorter than the source.", nameof(dst));
        }

        PointTransformKernels.Perspective(src, dst, Pack(matrix), count, cn, cn);
    }

    private static TPoint[] TransformPoints<TPoint, T>(ReadOnlySpan<TPoint> source, Mat<double> matrix, int scn, int dcn)
        where TPoint : unmanaged
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(matrix);
        RequireSingleChannel(matrix, nameof(matrix));
        if (matrix.Height != dcn || (matrix.Width != scn && matrix.Width != scn + 1))
        {
            throw new ArgumentException($"The matrix must be {dcn} × {scn} or {dcn} × {scn + 1}.", nameof(matrix));
        }

        T[] m = PrepareTransformMatrix<T>(matrix, scn, dcn, out bool isDiag);
        var dst = new TPoint[source.Length];
        PointTransformKernels.Transform<T>(MemoryMarshal.Cast<TPoint, T>(source), MemoryMarshal.Cast<TPoint, T>(dst.AsSpan()), m, source.Length, scn, dcn, isDiag);
        return dst;
    }

    private static T[] PrepareTransformMatrix<T>(Mat<double> matrix, int scn, int dcn, out bool isDiag)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T[] m = new T[dcn * (scn + 1)];
        for (int i = 0; i < dcn; i++)
        {
            ReadOnlySpan<double> row = matrix.GetReadOnlyRowSpan(i);
            for (int j = 0; j < matrix.Width; j++)
            {
                m[(i * (scn + 1)) + j] = F<T>(row[j]);
            }
        }

        isDiag = false;
        if (scn == dcn)
        {
            double eps = IsFloat<T>() ? FltEpsilon : DblEpsilon;
            isDiag = true;
            for (int i = 0; isDiag && i < scn; i++)
            {
                for (int j = 0; isDiag && j < scn; j++)
                {
                    if (i != j && Math.Abs(D(m[(i * (scn + 1)) + j])) > eps)
                    {
                        isDiag = false;
                    }
                }
            }
        }

        return m;
    }

    /// <summary><c>mulTransposed</c> with the destination type equal to the source type, packed.</summary>
    private static T[] MulTransposedPacked<T>(Mat<T> src, bool aTa, double scale = 1.0)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = src.Height, cols = src.Width;
        int n = aTa ? cols : rows;
        if (n >= 100 && rows >= 100 && cols >= 100)
        {
            var d = new Mat<T>(n, n);
            GemmKernels.Run(src, src, IsFloat<T>() ? (float)scale : scale, null, 0.0, d, aTa ? 1 : 2, aTa ? rows : cols);
            return d.DangerousGetArray();
        }

        T[] dst = new T[n * n];
        if (aTa)
        {
            // MulTransposedR: column i against columns j ≥ i, sequential double sums over rows.
            double[] colBuf = new double[rows];
            for (int i = 0; i < cols; i++)
            {
                for (int k = 0; k < rows; k++)
                {
                    colBuf[k] = D(src.At(k, i));
                }

                for (int j = i; j < cols; j++)
                {
                    double s = 0;
                    for (int k = 0; k < rows; k++)
                    {
                        s += colBuf[k] * D(src.At(k, j));
                    }

                    dst[(i * n) + j] = F<T>(s * scale);
                }
            }
        }
        else
        {
            for (int i = 0; i < rows; i++)
            {
                ReadOnlySpan<T> r1 = src.GetReadOnlyRowSpan(i);
                for (int j = i; j < rows; j++)
                {
                    ReadOnlySpan<T> r2 = src.GetReadOnlyRowSpan(j);
                    double s = 0;
                    int k = 0;
                    if (IsFloat<T>())
                    {
                        for (; k <= cols - 4; k += 4)
                        {
                            s += (D(r1[k]) * D(r2[k])) + (D(r1[k + 1]) * D(r2[k + 1])) + (D(r1[k + 2]) * D(r2[k + 2])) + (D(r1[k + 3]) * D(r2[k + 3]));
                        }
                    }
                    else
                    {
                        // MulTransposedL double: two 128-bit lanes, each adding a pair of products per step.
                        double l0 = 0, l1 = 0;
                        for (; k <= cols - 4; k += 4)
                        {
                            l0 += (D(r1[k]) * D(r2[k])) + (D(r1[k + 2]) * D(r2[k + 2]));
                            l1 += (D(r1[k + 1]) * D(r2[k + 1])) + (D(r1[k + 3]) * D(r2[k + 3]));
                        }

                        s += l0 + l1;
                    }

                    for (; k < cols; k++)
                    {
                        s += D(r1[k]) * D(r2[k]);
                    }

                    dst[(i * n) + j] = F<T>(s * scale);
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                dst[(i * n) + j] = dst[(j * n) + i];
            }
        }

        return dst;
    }

    /// <summary>Reads an n × n matrix (n ≤ 3) into row-major doubles.</summary>
    private static void LoadSmall<T>(Mat<T> m, int n, Span<double> s)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        for (int i = 0; i < n; i++)
        {
            ReadOnlySpan<T> row = m.GetReadOnlyRowSpan(i);
            for (int j = 0; j < n; j++)
            {
                s[(i * n) + j] = D(row[j]);
            }
        }
    }

    private static double Det2<T>(Mat<T> m)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Span<double> s = stackalloc double[4];
        LoadSmall(m, 2, s);
        return Det2(s);
    }

    private static double Det3<T>(Mat<T> m)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Span<double> s = stackalloc double[9];
        LoadSmall(m, 3, s);
        return Det3(s);
    }

    private static double Det2(ReadOnlySpan<double> s)
        => (s[0] * s[3]) - (s[1] * s[2]);

    private static double Det3(ReadOnlySpan<double> s)
        => (s[0] * ((s[4] * s[8]) - (s[5] * s[7]))) - (s[1] * ((s[3] * s[8]) - (s[5] * s[6]))) + (s[2] * ((s[3] * s[7]) - (s[4] * s[6])));

    private static bool InvertSmall<T>(Mat<T> src, Mat<T> dst)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int n = src.Height;
        Span<double> s = stackalloc double[9];
        LoadSmall(src, n, s);
        Span<T> o = dst.GetSpan();
        if (n == 1)
        {
            if (s[0] == 0)
            {
                return false;
            }

            o[0] = F<T>(1.0 / s[0]);
            return true;
        }

        if (n == 2)
        {
            double d = Det2(s);
            if (d == 0)
            {
                return false;
            }

            d = 1.0 / d;
            if (IsFloat<T>())
            {
                // CV_SIMD128 path: float products with (float)d.
                ReadOnlySpan<T> r0 = src.GetReadOnlyRowSpan(0), r1 = src.GetReadOnlyRowSpan(1);
                T df = F<T>(d);
                o[0] = r1[1] * df;
                o[1] = r0[1] * -df;
                o[2] = r1[0] * -df;
                o[3] = r0[0] * df;
            }
            else
            {
                // CV_SIMD128_64F path: off-diagonal elements are 0 - s*d.
                o[0] = F<T>(s[3] * d);
                o[1] = F<T>(0.0 - (s[1] * d));
                o[2] = F<T>(0.0 - (s[2] * d));
                o[3] = F<T>(s[0] * d);
            }

            return true;
        }

        Span<double> r = stackalloc double[9];
        if (!Invert3x3(s, r))
        {
            return false;
        }

        for (int i = 0; i < 9; i++)
        {
            o[i] = F<T>(r[i]);
        }

        return true;
    }

    /// <summary>
    /// The explicit 3 × 3 formula of <c>invert(DECOMP_LU)</c> for a packed row-major matrix.
    /// Returns false and leaves <paramref name="dst"/> unchanged for a zero determinant.
    /// <paramref name="dst"/> may be <paramref name="src"/>.
    /// </summary>
    internal static bool Invert3x3(ReadOnlySpan<double> src, Span<double> dst)
    {
        double det = Det3(src);
        if (det == 0)
        {
            return false;
        }

        det = 1.0 / det;
        double s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3], s4 = src[4], s5 = src[5], s6 = src[6], s7 = src[7], s8 = src[8];
        dst[0] = ((s4 * s8) - (s5 * s7)) * det;
        dst[1] = ((s2 * s7) - (s1 * s8)) * det;
        dst[2] = ((s1 * s5) - (s2 * s4)) * det;
        dst[3] = ((s5 * s6) - (s3 * s8)) * det;
        dst[4] = ((s0 * s8) - (s2 * s6)) * det;
        dst[5] = ((s2 * s3) - (s0 * s5)) * det;
        dst[6] = ((s3 * s7) - (s4 * s6)) * det;
        dst[7] = ((s1 * s6) - (s0 * s7)) * det;
        dst[8] = ((s0 * s4) - (s1 * s3)) * det;
        return true;
    }

    private static bool SolveSmall<T>(Mat<T> src, Mat<T> b, Mat<T> dst)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int n = src.Height;
        Span<double> s = stackalloc double[9];
        LoadSmall(src, n, s);
        Span<T> o = dst.GetSpan();
        if (n == 1)
        {
            if (s[0] == 0)
            {
                return false;
            }

            o[0] = F<T>(D(b.GetReadOnlyRowSpan(0)[0]) / s[0]);
            return true;
        }

        if (n == 2)
        {
            double d = Det2(s);
            if (d == 0)
            {
                return false;
            }

            d = 1.0 / d;
            double b0 = D(b.GetReadOnlyRowSpan(0)[0]), b1 = D(b.GetReadOnlyRowSpan(1)[0]);
            double t = ((b0 * s[3]) - (b1 * s[1])) * d;
            o[1] = F<T>(((b1 * s[0]) - (b0 * s[2])) * d);
            o[0] = F<T>(t);
            return true;
        }

        double det = Det3(s);
        if (det == 0)
        {
            return false;
        }

        det = 1.0 / det;
        T fb1 = b.GetReadOnlyRowSpan(1)[0];
        double s00 = s[0], s01 = s[1], s02 = s[2], s10 = s[3], s11 = s[4], s12 = s[5], s20 = s[6], s21 = s[7], s22 = s[8];
        double bb0 = D(b.GetReadOnlyRowSpan(0)[0]), bb1 = D(fb1), bb2 = D(b.GetReadOnlyRowSpan(2)[0]);
        if (IsFloat<T>())
        {
            // bf(1)*Sf(2,2) is a float product in the OpenCV expression for t[1].
            double b1s22 = D(fb1 * src.GetReadOnlyRowSpan(2)[2]);
            o[0] = F<T>(det * ((bb0 * ((s11 * s22) - (s12 * s21))) - (s01 * ((bb1 * s22) - (s12 * bb2))) + (s02 * ((bb1 * s21) - (s11 * bb2)))));
            o[1] = F<T>(det * ((s00 * (b1s22 - (s12 * bb2))) - (bb0 * ((s10 * s22) - (s12 * s20))) + (s02 * ((s10 * bb2) - (bb1 * s20)))));
            o[2] = F<T>(det * ((s00 * ((s11 * bb2) - (bb1 * s21))) - (s01 * ((s10 * bb2) - (bb1 * s20))) + (bb0 * ((s10 * s21) - (s11 * s20)))));
        }
        else
        {
            o[0] = F<T>(((((s11 * s22) - (s12 * s21)) * bb0) + (((s02 * s21) - (s01 * s22)) * bb1) + (((s01 * s12) - (s02 * s11)) * bb2)) * det);
            o[1] = F<T>(((((s12 * s20) - (s10 * s22)) * bb0) + (((s00 * s22) - (s02 * s20)) * bb1) + (((s02 * s10) - (s00 * s12)) * bb2)) * det);
            o[2] = F<T>(((((s10 * s21) - (s11 * s20)) * bb0) + (((s01 * s20) - (s00 * s21)) * bb1) + (((s00 * s11) - (s01 * s10)) * bb2)) * det);
        }

        return true;
    }

    private static void RequireSingleChannel<T>(Mat<T> m, string name)
        where T : unmanaged
    {
        if (m.Channels != 1)
        {
            throw new ArgumentException("The matrix must have one channel.", name);
        }
    }

    private static void RequireSquare<T>(Mat<T> m, string name)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(m, name);
        RequireSingleChannel(m, name);
        if (m.Width != m.Height)
        {
            throw new ArgumentException("The matrix must be square.", name);
        }
    }
}
