namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Marker shapes for <see cref="Drawing.DrawMarker{T}"/>, mirroring OpenCV's
/// <c>cv::MarkerTypes</c>.
/// </summary>
public enum MarkerType
{
    /// <summary>Horizontal and vertical lines (OpenCV <c>MARKER_CROSS</c>).</summary>
    Cross = 0,

    /// <summary>Two diagonal lines (OpenCV <c>MARKER_TILTED_CROSS</c>).</summary>
    TiltedCross = 1,

    /// <summary>Cross and tilted cross (OpenCV <c>MARKER_STAR</c>).</summary>
    Star = 2,

    /// <summary>Diamond outline (OpenCV <c>MARKER_DIAMOND</c>).</summary>
    Diamond = 3,

    /// <summary>Square outline (OpenCV <c>MARKER_SQUARE</c>).</summary>
    Square = 4,

    /// <summary>Upward triangle outline (OpenCV <c>MARKER_TRIANGLE_UP</c>).</summary>
    TriangleUp = 5,

    /// <summary>Downward triangle outline (OpenCV <c>MARKER_TRIANGLE_DOWN</c>).</summary>
    TriangleDown = 6,
}
