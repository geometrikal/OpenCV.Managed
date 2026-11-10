namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// An axis-aligned integer rectangle, mirroring OpenCV's <c>cv::Rect</c>.
///
/// <see cref="X"/> and <see cref="Y"/> give the top-left corner; the rectangle
/// covers the half-open range [X, X + Width) x [Y, Y + Height).
/// </summary>
/// <param name="X">Left edge of the rectangle.</param>
/// <param name="Y">Top edge of the rectangle.</param>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
public readonly record struct Rect(int X, int Y, int Width, int Height);
