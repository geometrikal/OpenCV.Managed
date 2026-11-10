namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A double-precision 2D point in image coordinates (X to the right, Y downward), mirroring
/// OpenCV's <c>cv::Point2d</c>.
/// </summary>
/// <param name="X">Horizontal coordinate, increasing to the right.</param>
/// <param name="Y">Vertical coordinate, increasing downward.</param>
public readonly record struct Point2d(double X, double Y);
