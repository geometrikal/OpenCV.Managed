namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Integer 2D dimensions (width and height), mirroring OpenCV's <c>cv::Size</c>.
/// </summary>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
public readonly record struct Size(int Width, int Height);
