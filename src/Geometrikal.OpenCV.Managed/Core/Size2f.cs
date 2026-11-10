namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Single-precision 2D dimensions (width and height), mirroring OpenCV's <c>cv::Size2f</c>.
/// </summary>
/// <param name="Width">Width.</param>
/// <param name="Height">Height.</param>
public readonly record struct Size2f(float Width, float Height);
