namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Matrix decomposition methods for <see cref="LinearAlgebra.Invert{T}"/> and
/// <see cref="LinearAlgebra.Solve{T}"/>, mirroring OpenCV <c>cv::DecompTypes</c>.
/// </summary>
[Flags]
public enum DecompTypes
{
    /// <summary>Gaussian elimination with partial pivoting (<c>DECOMP_LU</c>).</summary>
    Lu = 0,

    /// <summary>Singular value decomposition (<c>DECOMP_SVD</c>). Accepts singular and non-square matrices.</summary>
    Svd = 1,

    /// <summary>Eigenvalue decomposition (<c>DECOMP_EIG</c>). The matrix must be symmetric.</summary>
    Eig = 2,

    /// <summary>Cholesky factorization (<c>DECOMP_CHOLESKY</c>). The matrix must be symmetric and positive definite.</summary>
    Cholesky = 3,

    /// <summary>QR factorization (<c>DECOMP_QR</c>). Accepts non-square matrices.</summary>
    Qr = 4,

    /// <summary>
    /// Flag for <see cref="LinearAlgebra.Solve{T}"/> (<c>DECOMP_NORMAL</c>). The solver uses the normal
    /// equations <c>srcᵀ src x = srcᵀ b</c> instead of <c>src x = b</c>.
    /// </summary>
    Normal = 16,
}
