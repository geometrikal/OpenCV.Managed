namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Contour approximation method, mirroring OpenCV's
/// <c>cv::ContourApproximationModes</c>.
///
/// The integer values match the OpenCV constants exactly. (Chain-code output,
/// which OpenCV represents with the value 0, is not supported by this port.)
/// </summary>
public enum ContourApproximationMode
{
    /// <summary>
    /// Stores absolutely all the contour points (OpenCV
    /// <c>CHAIN_APPROX_NONE</c>). Consecutive points differ by at most one in
    /// each coordinate.
    /// </summary>
    None = 1,

    /// <summary>
    /// Compresses horizontal, vertical, and diagonal segments, leaving only
    /// their end points (OpenCV <c>CHAIN_APPROX_SIMPLE</c>).
    /// </summary>
    Simple = 2,

    /// <summary>
    /// Applies the Teh-Chin L1 chain approximation algorithm (OpenCV
    /// <c>CHAIN_APPROX_TC89_L1</c>).
    /// </summary>
    Tc89L1 = 3,

    /// <summary>
    /// Applies the Teh-Chin k-cosine chain approximation algorithm (OpenCV
    /// <c>CHAIN_APPROX_TC89_KCOS</c>).
    /// </summary>
    Tc89KCos = 4,
}
