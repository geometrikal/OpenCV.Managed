namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// An integer 2D point in image coordinates (X to the right, Y downward),
/// mirroring OpenCV's <c>cv::Point</c>.
/// </summary>
/// <param name="X">Horizontal coordinate, increasing to the right.</param>
/// <param name="Y">Vertical coordinate, increasing downward.</param>
public readonly record struct Point(int X, int Y)
{
    /// <summary>
    /// Returns the component-wise sum of two points.
    /// </summary>
    public static Point operator +(Point a, Point b) => new(a.X + b.X, a.Y + b.Y);

    /// <summary>
    /// Returns the component-wise difference of two points.
    /// </summary>
    public static Point operator -(Point a, Point b) => new(a.X - b.X, a.Y - b.Y);
}
