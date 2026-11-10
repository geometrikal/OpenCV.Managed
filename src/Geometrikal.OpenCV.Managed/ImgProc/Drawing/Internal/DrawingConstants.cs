// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Shared fixed-point constants used throughout the drawing port, mirroring the
/// anonymous enum and constants at the top of OpenCV's <c>drawing.cpp</c>.
/// </summary>
internal static class DrawingConstants
{
    /// <summary>
    /// Number of fractional bits in the sub-pixel fixed-point representation
    /// (OpenCV <c>XY_SHIFT</c>). A coordinate is stored as <c>value &lt;&lt; XY_SHIFT</c>.
    /// </summary>
    public const int XY_SHIFT = 16;

    /// <summary>
    /// The fixed-point representation of <c>1.0</c> (OpenCV <c>XY_ONE</c>),
    /// equal to <c>1 &lt;&lt; XY_SHIFT</c>.
    /// </summary>
    public const int XY_ONE = 1 << XY_SHIFT;

    /// <summary>
    /// Maximum supported line/outline thickness (OpenCV <c>MAX_THICKNESS</c>).
    /// </summary>
    public const int MAX_THICKNESS = 32767;
}
