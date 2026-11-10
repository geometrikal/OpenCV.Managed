// This type is based on the OpenCV stitching feature container and helper.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/matchers.hpp (ImageFeatures)
//   modules/stitching/src/matchers.cpp (computeImageFeatures)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// The keypoints and descriptors of one image, as OpenCV <c>detail::ImageFeatures</c>.
/// </summary>
/// <remarks>
/// Exactly one of <see cref="ByteDescriptors"/> and <see cref="FloatDescriptors"/> is set when
/// there are keypoints. One descriptor row belongs to each keypoint. The matchers select the
/// distance from the descriptor type: Hamming for <c>byte</c> descriptors and L2 for
/// <c>float</c> descriptors.
/// </remarks>
public sealed class ImageFeatures
{
    /// <summary>Initializes a new instance with no keypoints.</summary>
    public ImageFeatures()
    {
    }

    /// <summary>Initializes a new instance with binary descriptors (for example ORB).</summary>
    /// <param name="imgIdx">Image index.</param>
    /// <param name="imgSize">Size of the image the keypoints come from.</param>
    /// <param name="keypoints">Keypoints. The array is used without a copy.</param>
    /// <param name="descriptors">One row per keypoint, or null when there are no keypoints.</param>
    public ImageFeatures(int imgIdx, Size imgSize, KeyPoint[] keypoints, Mat<byte>? descriptors)
    {
        ImgIdx = imgIdx;
        ImgSize = imgSize;
        Keypoints = keypoints ?? throw new ArgumentNullException(nameof(keypoints));
        ByteDescriptors = descriptors;
    }

    /// <summary>Initializes a new instance with <c>float</c> descriptors (for example SIFT).</summary>
    /// <param name="imgIdx">Image index.</param>
    /// <param name="imgSize">Size of the image the keypoints come from.</param>
    /// <param name="keypoints">Keypoints. The array is used without a copy.</param>
    /// <param name="descriptors">One row per keypoint, or null when there are no keypoints.</param>
    public ImageFeatures(int imgIdx, Size imgSize, KeyPoint[] keypoints, Mat<float>? descriptors)
    {
        ImgIdx = imgIdx;
        ImgSize = imgSize;
        Keypoints = keypoints ?? throw new ArgumentNullException(nameof(keypoints));
        FloatDescriptors = descriptors;
    }

    /// <summary>Gets or sets the image index (OpenCV <c>img_idx</c>).</summary>
    public int ImgIdx { get; set; }

    /// <summary>Gets or sets the image size (OpenCV <c>img_size</c>).</summary>
    public Size ImgSize { get; set; }

    /// <summary>Gets or sets the keypoints.</summary>
    public KeyPoint[] Keypoints { get; set; } = [];

    /// <summary>Gets or sets the binary descriptors (OpenCV <c>CV_8U</c>), or null.</summary>
    public Mat<byte>? ByteDescriptors { get; set; }

    /// <summary>Gets or sets the <c>float</c> descriptors (OpenCV <c>CV_32F</c>), or null.</summary>
    public Mat<float>? FloatDescriptors { get; set; }

    /// <summary>Gets a value indicating whether the descriptors are <c>float</c>.</summary>
    public bool HasFloatDescriptors => FloatDescriptors is not null;

    /// <summary>Returns a shallow copy: the keypoint array and descriptor matrices are shared.</summary>
    public ImageFeatures ShallowCopy() => (ImageFeatures)MemberwiseClone();

    /// <summary>
    /// Detects ORB keypoints and computes their descriptors, as OpenCV
    /// <c>computeImageFeatures(orb, image, features, mask)</c>.
    /// </summary>
    /// <param name="orb">ORB detector and extractor.</param>
    /// <param name="image">8-bit image with 1, 3 or 4 channels. It is not modified.</param>
    /// <param name="mask">Optional 8-bit mask of the image size.</param>
    /// <param name="imgIdx">Value for <see cref="ImgIdx"/>.</param>
    /// <returns>The features of the image.</returns>
    public static ImageFeatures Compute(Orb orb, Mat<byte> image, Mat<byte>? mask = null, int imgIdx = 0)
    {
        ArgumentNullException.ThrowIfNull(orb);
        ArgumentNullException.ThrowIfNull(image);
        (KeyPoint[] keypoints, Mat<byte>? descriptors) = orb.DetectAndCompute(image, mask);
        return new ImageFeatures(imgIdx, new Size(image.Width, image.Height), keypoints, descriptors);
    }

    /// <summary>
    /// Detects SIFT keypoints and computes their descriptors, as OpenCV
    /// <c>computeImageFeatures(sift, image, features, mask)</c>. The descriptor type follows
    /// <see cref="Sift.DescriptorType"/>.
    /// </summary>
    /// <param name="sift">SIFT detector and extractor.</param>
    /// <param name="image">8-bit image with 1, 3 or 4 channels. It is not modified.</param>
    /// <param name="mask">Optional 8-bit mask of the image size.</param>
    /// <param name="imgIdx">Value for <see cref="ImgIdx"/>.</param>
    /// <returns>The features of the image.</returns>
    public static ImageFeatures Compute(Sift sift, Mat<byte> image, Mat<byte>? mask = null, int imgIdx = 0)
    {
        ArgumentNullException.ThrowIfNull(sift);
        ArgumentNullException.ThrowIfNull(image);
        var size = new Size(image.Width, image.Height);
        if (sift.DescriptorType == SiftDescriptorType.Float32)
        {
            List<KeyPoint> kp = sift.DetectAndCompute(image, mask, out Mat<float>? fd);
            return new ImageFeatures(imgIdx, size, kp.ToArray(), fd);
        }

        List<KeyPoint> kb = sift.DetectAndCompute(image, mask, out Mat<byte>? bd);
        return new ImageFeatures(imgIdx, size, kb.ToArray(), bd);
    }

    /// <summary>
    /// Computes the features of several images, as the OpenCV <c>computeImageFeatures</c>
    /// overload for image arrays: <c>detect</c>, then <c>compute</c>. <see cref="ImgIdx"/> is
    /// set to the image index.
    /// </summary>
    /// <param name="orb">ORB detector and extractor.</param>
    /// <param name="images">Images.</param>
    /// <param name="masks">Optional masks, one per image.</param>
    /// <returns>One feature set per image.</returns>
    public static ImageFeatures[] Compute(Orb orb, IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>?>? masks = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        var result = new ImageFeatures[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            Mat<byte>? mask = masks is null || masks.Count == 0 ? null : masks[i];
            Mat<byte> image = images[i];
            KeyPoint[] detected = orb.Detect(image, mask);
            (KeyPoint[] keypoints, Mat<byte>? descriptors) = orb.Compute(image, detected);
            result[i] = new ImageFeatures(i, new Size(image.Width, image.Height), keypoints, descriptors);
        }

        return result;
    }
}
