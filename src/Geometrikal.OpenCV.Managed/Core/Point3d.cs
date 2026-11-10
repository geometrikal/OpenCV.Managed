namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A double-precision 3D point, mirroring OpenCV's <c>cv::Point3d</c>.
/// </summary>
/// <param name="X">First coordinate.</param>
/// <param name="Y">Second coordinate.</param>
/// <param name="Z">Third coordinate.</param>
public readonly record struct Point3d(double X, double Y, double Z);
