namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// A 2-D line as returned by OpenCV <c>fitLine</c>: the unit direction (<see cref="Vx"/>,
/// <see cref="Vy"/>) and one point (<see cref="X0"/>, <see cref="Y0"/>) on the line.
/// </summary>
/// <param name="Vx">X component of the unit direction vector.</param>
/// <param name="Vy">Y component of the unit direction vector.</param>
/// <param name="X0">X coordinate of a point on the line.</param>
/// <param name="Y0">Y coordinate of a point on the line.</param>
public readonly record struct FittedLine(float Vx, float Vy, float X0, float Y0);
