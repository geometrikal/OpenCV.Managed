// This type is based on the OpenCV stitching camera parameters.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/camera.hpp
//   modules/stitching/src/camera.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// The intrinsic and extrinsic parameters of one camera, as OpenCV <c>detail::CameraParams</c>.
/// </summary>
/// <remarks>
/// <see cref="R"/> is always a <c>double</c> matrix. OpenCV stores it as <c>CV_64F</c> after an
/// estimator and as <c>CV_32F</c> after <see cref="Stitcher"/> converts it and after a bundle
/// adjuster. The port keeps the same values: the bundle adjusters, <see cref="WaveCorrection"/>
/// and the stitcher round <see cref="R"/> to <c>float</c> where OpenCV uses <c>CV_32F</c>.
/// </remarks>
public sealed class CameraParams
{
    /// <summary>Gets or sets the focal length. The default is 1.</summary>
    public double Focal { get; set; } = 1;

    /// <summary>Gets or sets the aspect ratio (focal_y / focal_x). The default is 1.</summary>
    public double Aspect { get; set; } = 1;

    /// <summary>Gets or sets the principal point x coordinate.</summary>
    public double Ppx { get; set; }

    /// <summary>Gets or sets the principal point y coordinate.</summary>
    public double Ppy { get; set; }

    /// <summary>
    /// Gets or sets the 3 × 3 rotation (or, for the affine estimators, the affine transform with
    /// the last row 0 0 1). The default is the identity.
    /// </summary>
    public Mat<double> R { get; set; } = Identity();

    /// <summary>Gets or sets the 3 × 1 translation. The default is zero.</summary>
    public Mat<double> T { get; set; } = new(1, 3);

    /// <summary>
    /// Returns the intrinsic matrix <c>[focal 0 ppx; 0 focal·aspect ppy; 0 0 1]</c>, as OpenCV
    /// <c>CameraParams::K</c>.
    /// </summary>
    /// <returns>A new 3 × 3 matrix.</returns>
    public Mat<double> K()
    {
        Mat<double> k = Identity();
        k.At(0, 0) = Focal;
        k.At(0, 2) = Ppx;
        k.At(1, 1) = Focal * Aspect;
        k.At(1, 2) = Ppy;
        return k;
    }

    /// <summary>Returns a deep copy (<see cref="R"/> and <see cref="T"/> are cloned).</summary>
    public CameraParams Clone() => new()
    {
        Focal = Focal,
        Aspect = Aspect,
        Ppx = Ppx,
        Ppy = Ppy,
        R = R.Clone(),
        T = T.Clone(),
    };

    internal static Mat<double> Identity()
    {
        var m = new Mat<double>(3, 3);
        m.At(0, 0) = m.At(1, 1) = m.At(2, 2) = 1;
        return m;
    }
}
