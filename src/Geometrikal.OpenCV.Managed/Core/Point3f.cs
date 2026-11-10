namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A single-precision 3D point, mirroring OpenCV's <c>cv::Point3f</c>.
/// </summary>
/// <param name="X">First coordinate.</param>
/// <param name="Y">Second coordinate.</param>
/// <param name="Z">Third coordinate.</param>
public readonly record struct Point3f(float X, float Y, float Z);
