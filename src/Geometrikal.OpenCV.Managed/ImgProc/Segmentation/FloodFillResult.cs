namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Result of <see cref="FloodFilling"/>.</summary>
/// <param name="Area">Number of filled pixels. 0 when the seed pixel is set in the mask.</param>
/// <param name="Rect">
/// Bounding rectangle of the filled pixels in image coordinates. (0, 0, 0, 0) when
/// <paramref name="Area"/> is 0.
/// </param>
public readonly record struct FloodFillResult(int Area, Rect Rect);
