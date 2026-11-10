// This type mirrors OpenCV's RotatedRect.
//
// Original sources:
//   modules/core/include/opencv2/core/types.hpp (RotatedRect)
//   modules/core/src/types.cpp (RotatedRect::points)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A rotated rectangle in image coordinates (X to the right, Y downward), mirroring OpenCV's
/// <c>cv::RotatedRect</c>.
/// </summary>
/// <param name="Center">Center of the rectangle.</param>
/// <param name="Size">Width and height of the rectangle.</param>
/// <param name="Angle">
/// Rotation angle in degrees. A positive angle turns the width side clockwise on screen
/// (Y downward).
/// </param>
public readonly record struct RotatedRect(Point2f Center, Size2f Size, float Angle)
{
    /// <summary>
    /// Returns the four corners, as OpenCV <c>RotatedRect::points</c>. For angle 0 the order is
    /// bottom-left, top-left, top-right, bottom-right (Y downward).
    /// </summary>
    /// <returns>A new array of four corners.</returns>
    public Point2f[] Points()
    {
        var pts = new Point2f[4];
        Points(pts);
        return pts;
    }

    /// <summary>Writes the four corners in the order of <see cref="Points()"/>.</summary>
    /// <param name="destination">Destination with at least four elements.</param>
    /// <exception cref="ArgumentException">The destination has fewer than four elements.</exception>
    public void Points(Span<Point2f> destination)
    {
        if (destination.Length < 4)
        {
            throw new ArgumentException("The destination must have at least four elements.", nameof(destination));
        }

        double angle = Angle * Math.PI / 180.0;
        float b = (float)Math.Cos(angle) * 0.5f;
        float a = (float)Math.Sin(angle) * 0.5f;
        float ah = a * Size.Height;
        float aw = a * Size.Width;
        float bh = b * Size.Height;
        float bw = b * Size.Width;
        float cx = Center.X;
        float cy = Center.Y;
        destination[0] = new Point2f(cx - ah - bw, cy + bh - aw);
        destination[1] = new Point2f(cx + ah - bw, cy - bh - aw);
        destination[2] = new Point2f(cx + ah + bw, cy - bh + aw);
        destination[3] = new Point2f(cx - ah + bw, cy + bh + aw);
    }
}
