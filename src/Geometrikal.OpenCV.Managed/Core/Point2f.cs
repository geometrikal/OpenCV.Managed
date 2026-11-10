namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A single-precision 2D point in image coordinates (X to the right, Y downward), mirroring
/// OpenCV's <c>cv::Point2f</c>.
/// </summary>
/// <param name="X">Horizontal coordinate, increasing to the right.</param>
/// <param name="Y">Vertical coordinate, increasing downward.</param>
public readonly record struct Point2f(float X, float Y);
