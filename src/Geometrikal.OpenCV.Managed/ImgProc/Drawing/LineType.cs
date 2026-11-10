namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Line type for the drawing routines, mirroring OpenCV's <c>cv::LineTypes</c>.
/// </summary>
public enum LineType
{
    /// <summary>
    /// 4-connected Bresenham line (OpenCV <c>LINE_4</c>). The iterator may move
    /// only up, down, left and right.
    /// </summary>
    FourConnected = 4,

    /// <summary>
    /// 8-connected Bresenham line (OpenCV <c>LINE_8</c>). The iterator may also
    /// move diagonally.
    /// </summary>
    EightConnected = 8,

    /// <summary>
    /// Anti-aliased line (OpenCV <c>LINE_AA</c>). The anti-aliased path applies
    /// to 8-bit images with 1, 3 or 4 channels. Images of other element types
    /// use <see cref="EightConnected"/>, as OpenCV does. 8-bit images with 2
    /// channels draw 8-connected outlines with the anti-aliased fill rounding.
    /// </summary>
    AntiAliased = 16,
}
