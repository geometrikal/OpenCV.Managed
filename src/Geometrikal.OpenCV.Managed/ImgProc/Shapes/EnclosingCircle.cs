namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>A circle that encloses a point set, as returned by OpenCV <c>minEnclosingCircle</c>.</summary>
/// <param name="Center">Center of the circle.</param>
/// <param name="Radius">Radius of the circle.</param>
public readonly record struct EnclosingCircle(Point2f Center, float Radius);
