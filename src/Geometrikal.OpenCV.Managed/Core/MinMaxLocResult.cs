namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>Result of <see cref="Reductions.MinMaxLoc{T}(Mat{T}, Mat{byte})"/>.</summary>
/// <param name="MinVal">Minimum value.</param>
/// <param name="MaxVal">Maximum value.</param>
/// <param name="MinLoc">Position (x, y) of the first minimum in row-major order.</param>
/// <param name="MaxLoc">Position (x, y) of the first maximum in row-major order.</param>
public readonly record struct MinMaxLocResult(double MinVal, double MaxVal, Point MinLoc, Point MaxLoc);
