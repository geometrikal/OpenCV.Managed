// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A 64-bit integer 2D point, mirroring OpenCV's <c>cv::Point2l</c>.
///
/// <para>The drawing routines carry sub-pixel coordinates in a fixed-point
/// representation (see <see cref="DrawingConstants.XY_SHIFT"/>) whose intermediate
/// values can exceed the 32-bit range, so coordinates are held as
/// <see cref="long"/>.</para>
/// </summary>
/// <param name="X">Horizontal coordinate.</param>
/// <param name="Y">Vertical coordinate.</param>
internal readonly record struct PointL(long X, long Y)
{
    /// <summary>Returns the component-wise sum of two points.</summary>
    public static PointL operator +(PointL a, PointL b) => new(a.X + b.X, a.Y + b.Y);

    /// <summary>Returns the component-wise difference of two points.</summary>
    public static PointL operator -(PointL a, PointL b) => new(a.X - b.X, a.Y - b.Y);

    /// <summary>Widens an integer <see cref="Point"/> to a <see cref="PointL"/>.</summary>
    public static explicit operator PointL(Point p) => new(p.X, p.Y);
}
