// This implementation is ported from the OpenCV singular value decomposition.
//
// Original sources:
//   modules/core/src/lapack.cpp (_SVDcompute, SVD::compute, SVD::backSubst)
//   modules/core/include/opencv2/core/operations.hpp (SVD::solveZ)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Core.Internal;
using static Geometrikal.OpenCV.Managed.Core.Internal.LapackScalar;

/// <summary>
/// Singular value decomposition <c>A = U·diag(W)·Vᵀ</c> of a single-channel <c>float</c> or
/// <c>double</c> matrix (<c>cv::SVD</c>, one-sided Jacobi method).
/// </summary>
/// <typeparam name="T"><c>float</c> or <c>double</c>.</typeparam>
/// <remarks>
/// For an m × n source with nm = min(m, n): <see cref="W"/> is nm × 1 in descending order,
/// <see cref="U"/> is m × nm (m × m with <see cref="SvdFlags.FullUV"/>) and <see cref="Vt"/> is
/// nm × n (n × n with <see cref="SvdFlags.FullUV"/>). The source is not modified.
/// When the larger dimension is 25 or more, OpenCV with its internal LAPACK calls
/// <c>gesdd</c> instead of the Jacobi method; the port always uses the Jacobi method, and the
/// results then agree only within a tolerance (singular vectors can differ in sign).
/// </remarks>
public sealed class Svd<T>
    where T : unmanaged, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Decomposes <paramref name="source"/> (<c>SVD::operator()</c>).
    /// </summary>
    /// <param name="source">Single-channel matrix.</param>
    /// <param name="flags">Decomposition flags. With <see cref="SvdFlags.NoUV"/>, <see cref="U"/> and <see cref="Vt"/> are empty (1 × 1 zero) matrices and <see cref="HasVectors"/> is false.</param>
    public Svd(Mat<T> source, SvdFlags flags = SvdFlags.None)
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels != 1)
        {
            throw new ArgumentException("The matrix must have one channel.", nameof(source));
        }

        Compute(source, flags, out Mat<T> w, out Mat<T>? u, out Mat<T>? vt);
        W = w;
        HasVectors = u is not null;
        U = u ?? new Mat<T>(1, 1);
        Vt = vt ?? new Mat<T>(1, 1);
    }

    /// <summary>Singular values, nm × 1, in descending order.</summary>
    public Mat<T> W { get; }

    /// <summary>Left singular vectors in columns.</summary>
    public Mat<T> U { get; }

    /// <summary>Transposed right singular vectors (right singular vectors in rows).</summary>
    public Mat<T> Vt { get; }

    /// <summary>False when the decomposition was computed with <see cref="SvdFlags.NoUV"/>.</summary>
    public bool HasVectors { get; }

    /// <summary>
    /// Computes the singular values only (<c>SVD::compute(a, w)</c>).
    /// </summary>
    /// <param name="source">Single-channel matrix.</param>
    /// <returns>nm × 1 singular values in descending order.</returns>
    public static Mat<T> ComputeValues(Mat<T> source)
        => new Svd<T>(source, SvdFlags.NoUV).W;

    /// <summary>
    /// Solves <c>A·x = rhs</c> in the least-squares sense with this decomposition
    /// (<c>SVD::backSubst</c>).
    /// </summary>
    /// <param name="rhs">m × nb right-hand side, or null for the pseudo-inverse (n × m).</param>
    /// <returns>The n × nb solution.</returns>
    public Mat<T> BackSubst(Mat<T>? rhs)
    {
        if (!HasVectors)
        {
            throw new InvalidOperationException("The decomposition was computed with SvdFlags.NoUV.");
        }

        return BackSubst(W, U, Vt, rhs);
    }

    /// <summary>
    /// Computes <c>x = V·diag(W)⁻¹·Uᵀ·rhs</c> (<c>SVD::backSubst</c>). Singular values below
    /// <c>2·DBL_EPSILON·ΣW</c> (<c>float</c>: the same value rounded to <c>float</c>) count as zero.
    /// </summary>
    /// <param name="w">Singular values: nm × 1, 1 × nm, or a Vt.Height × U.Width matrix with the values on the diagonal.</param>
    /// <param name="u">m × (≥ nm) left singular vectors in columns.</param>
    /// <param name="vt">(≥ nm) × n right singular vectors in rows.</param>
    /// <param name="rhs">m × nb right-hand side, or null for the pseudo-inverse.</param>
    /// <returns>The n × nb solution.</returns>
    public static Mat<T> BackSubst(Mat<T> w, Mat<T> u, Mat<T> vt, Mat<T>? rhs)
    {
        ThrowIfUnsupported<T>();
        ArgumentNullException.ThrowIfNull(w);
        ArgumentNullException.ThrowIfNull(u);
        ArgumentNullException.ThrowIfNull(vt);
        int m = u.Height, n = vt.Width, nb = rhs is not null ? rhs.Width : m, nm = Math.Min(m, n);
        bool wOk = (w.Width == nm && w.Height == 1) || (w.Width == 1 && w.Height == nm) || (w.Width == u.Width && w.Height == vt.Height);
        if (u.Width < nm || vt.Height < nm || !wOk)
        {
            throw new ArgumentException("The sizes of w, u and vt do not form a decomposition.");
        }

        if (rhs is not null && rhs.Height != m)
        {
            throw new ArgumentException("The right-hand side must have the rows of u.", nameof(rhs));
        }

        int incw = w.Height == 1 ? 1 : w.Width == 1 ? w.Stride : w.Stride + 1;
        var x = new Mat<T>(nb, n);
        LapackDecomp.SvBkSb<T>(
            m,
            n,
            GemmKernels.Whole(w),
            incw,
            GemmKernels.Whole(u),
            u.Stride,
            false,
            GemmKernels.Whole(vt),
            vt.Stride,
            true,
            rhs is null ? default : GemmKernels.Whole(rhs),
            rhs?.Stride ?? 0,
            nb,
            x.GetSpan(),
            nb);
        return x;
    }

    /// <summary>
    /// Solves the homogeneous system <c>A·x = 0</c> for <c>‖x‖ = 1</c> (<c>SVD::solveZ</c>): the
    /// right singular vector of the smallest singular value.
    /// </summary>
    /// <param name="source">Single-channel m × n matrix.</param>
    /// <returns>The n × 1 solution.</returns>
    public static Mat<T> SolveZ(Mat<T> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        var svd = new Svd<T>(source, source.Height >= source.Width ? SvdFlags.None : SvdFlags.FullUV);
        int cols = svd.Vt.Width;
        var dst = new Mat<T>(1, cols);
        svd.Vt.GetReadOnlyRowSpan(svd.Vt.Height - 1).CopyTo(dst.GetSpan());
        return dst;
    }

    private static void Compute(Mat<T> src, SvdFlags flags, out Mat<T> w, out Mat<T>? u, out Mat<T>? vt)
    {
        int m = src.Height, n = src.Width;
        bool computeUv = (flags & SvdFlags.NoUV) == 0;
        bool fullUv = computeUv && (flags & SvdFlags.FullUV) != 0;
        bool at = false;
        if (m < n)
        {
            (m, n) = (n, m);
            at = true;
        }

        int urows = fullUv ? m : n;
        T[] tempU = new T[urows * m];
        T[] tempW = new T[n];
        T[]? tempV = computeUv ? new T[n * n] : null;

        // temp_a is the first n rows of temp_u: the n vectors of length m.
        T[] a = !at ? LinearAlgebra.Pack(MatrixTransforms.Transpose(src)) : LinearAlgebra.Pack(src);
        Array.Copy(a, tempU, n * m);

        LapackDecomp.JacobiSvd<T>(tempU, m, tempW, tempV, n, m, n, computeUv ? urows : 0);
        w = new Mat<T>(tempW, 1, n);
        if (!computeUv)
        {
            u = vt = null;
            return;
        }

        var tu = new Mat<T>(tempU, m, urows);
        var tv = new Mat<T>(tempV!, n, n);
        if (!at)
        {
            u = MatrixTransforms.Transpose(tu);
            vt = tv;
        }
        else
        {
            u = MatrixTransforms.Transpose(tv);
            vt = tu;
        }
    }
}
