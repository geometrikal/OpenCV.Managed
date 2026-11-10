namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Flags for <see cref="Svd{T}"/>, mirroring OpenCV <c>cv::SVD::Flags</c>.
/// </summary>
[Flags]
public enum SvdFlags
{
    /// <summary>Computes the singular values and the compact <c>U</c> and <c>Vᵀ</c>.</summary>
    None = 0,

    /// <summary>
    /// Allows the decomposition to modify the source (<c>MODIFY_A</c>). The managed port never
    /// modifies the source and accepts the flag for compatibility.
    /// </summary>
    ModifyA = 1,

    /// <summary>Computes only the singular values (<c>NO_UV</c>).</summary>
    NoUV = 2,

    /// <summary>Computes a full-size <c>U</c> (rows × rows) when the source is not square (<c>FULL_UV</c>).</summary>
    FullUV = 4,
}
