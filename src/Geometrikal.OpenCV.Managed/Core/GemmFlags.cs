namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Transposition flags for <see cref="LinearAlgebra.Gemm{T}"/>, mirroring OpenCV <c>cv::GemmFlags</c>.
/// </summary>
[Flags]
public enum GemmFlags
{
    /// <summary>No operand is transposed.</summary>
    None = 0,

    /// <summary>Transposes the first operand (<c>GEMM_1_T</c>).</summary>
    ATranspose = 1,

    /// <summary>Transposes the second operand (<c>GEMM_2_T</c>).</summary>
    BTranspose = 2,

    /// <summary>Transposes the third operand (<c>GEMM_3_T</c>).</summary>
    CTranspose = 4,
}
