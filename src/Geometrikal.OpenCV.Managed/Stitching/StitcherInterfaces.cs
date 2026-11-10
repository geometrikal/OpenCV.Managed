// These types are based on the OpenCV stitching pipeline extension points.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/warpers.hpp (WarperCreator, PlaneWarper,
//   AffineWarper, CylindricalWarper, SphericalWarper)
//   modules/stitching/src/matchers.cpp (computeImageFeatures for image arrays)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Computes the features of the registration images for <see cref="Stitcher"/>, as the OpenCV
/// <c>Feature2D</c> passed to <c>Stitcher::setFeaturesFinder</c>.
/// </summary>
public interface IFeaturesFinder
{
    /// <summary>
    /// Computes the features of each image (OpenCV <c>computeImageFeatures</c>: <c>detect</c>,
    /// then <c>compute</c>). <see cref="ImageFeatures.ImgIdx"/> is set to the image index.
    /// </summary>
    /// <param name="images">8-bit images. They are not modified.</param>
    /// <param name="masks">Optional masks, one per image, or null.</param>
    /// <returns>One feature set per image.</returns>
    ImageFeatures[] Compute(IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>?>? masks);
}

/// <summary>ORB features for <see cref="Stitcher"/> (the OpenCV default <c>ORB::create()</c>).</summary>
/// <param name="orb">The ORB detector and extractor.</param>
public sealed class OrbFeaturesFinder(Orb orb) : IFeaturesFinder
{
    /// <summary>Initializes a new instance with the default ORB settings.</summary>
    public OrbFeaturesFinder()
        : this(new Orb())
    {
    }

    /// <summary>Gets the ORB detector and extractor.</summary>
    public Orb Orb { get; } = orb ?? throw new ArgumentNullException(nameof(orb));

    /// <inheritdoc/>
    public ImageFeatures[] Compute(IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>?>? masks)
        => ImageFeatures.Compute(Orb, images, masks);
}

/// <summary>SIFT features for <see cref="Stitcher"/>.</summary>
/// <param name="sift">The SIFT detector and extractor.</param>
public sealed class SiftFeaturesFinder(Sift sift) : IFeaturesFinder
{
    /// <summary>Gets the SIFT detector and extractor.</summary>
    public Sift Sift { get; } = sift ?? throw new ArgumentNullException(nameof(sift));

    /// <inheritdoc/>
    public ImageFeatures[] Compute(IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>?>? masks)
    {
        ArgumentNullException.ThrowIfNull(images);
        var result = new ImageFeatures[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            Mat<byte> image = images[i];
            Mat<byte>? mask = masks is null || masks.Count == 0 ? null : masks[i];
            List<KeyPoint> keypoints = Sift.Detect(image, mask);
            var size = new Size(image.Width, image.Height);
            if (Sift.DescriptorType == SiftDescriptorType.Float32)
            {
                Sift.Compute(image, keypoints, out Mat<float>? fd);
                result[i] = new ImageFeatures(i, size, keypoints.ToArray(), fd);
            }
            else
            {
                Sift.Compute(image, keypoints, out Mat<byte>? bd);
                result[i] = new ImageFeatures(i, size, keypoints.ToArray(), bd);
            }
        }

        return result;
    }
}

/// <summary>
/// Creates a <see cref="RotationWarper"/> for a given scale, as OpenCV <c>WarperCreator</c>.
/// <see cref="Stitcher"/> creates one warper for the seam scale and one for the compositing scale.
/// </summary>
public abstract class WarperCreator
{
    /// <summary>Creates a warper.</summary>
    /// <param name="scale">Projection scale (the focal length at the working resolution).</param>
    /// <returns>A new warper.</returns>
    public abstract RotationWarper Create(float scale);
}

/// <summary>Creates <see cref="PlaneWarper"/> instances (OpenCV <c>cv::PlaneWarper</c>).</summary>
public sealed class PlaneWarperCreator : WarperCreator
{
    /// <inheritdoc/>
    public override RotationWarper Create(float scale) => new PlaneWarper(scale);
}

/// <summary>Creates <see cref="AffineWarper"/> instances (OpenCV <c>cv::AffineWarper</c>).</summary>
public sealed class AffineWarperCreator : WarperCreator
{
    /// <inheritdoc/>
    public override RotationWarper Create(float scale) => new AffineWarper(scale);
}

/// <summary>Creates <see cref="CylindricalWarper"/> instances (OpenCV <c>cv::CylindricalWarper</c>).</summary>
public sealed class CylindricalWarperCreator : WarperCreator
{
    /// <inheritdoc/>
    public override RotationWarper Create(float scale) => new CylindricalWarper(scale);
}

/// <summary>Creates <see cref="SphericalWarper"/> instances (OpenCV <c>cv::SphericalWarper</c>).</summary>
public sealed class SphericalWarperCreator : WarperCreator
{
    /// <inheritdoc/>
    public override RotationWarper Create(float scale) => new SphericalWarper(scale);
}
