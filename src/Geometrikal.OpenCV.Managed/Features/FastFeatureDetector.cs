// This implementation is ported from the OpenCV FAST detector.
//
// Original sources:
//   modules/features/src/fast.cpp (FastFeatureDetector_Impl)
//   modules/features/include/opencv2/features.hpp (FastFeatureDetector)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features.Internal;

/// <summary>
/// FAST keypoint detector with settings, as OpenCV <c>FastFeatureDetector</c>. See
/// <see cref="Fast"/> for the detection rules.
/// </summary>
public sealed class FastFeatureDetector
{
    /// <summary>
    /// Creates a detector, as OpenCV <c>FastFeatureDetector::create</c>.
    /// </summary>
    /// <param name="threshold">Intensity difference threshold.</param>
    /// <param name="nonmaxSuppression">Enable non-maximum suppression.</param>
    /// <param name="type">Circle pattern.</param>
    public FastFeatureDetector(int threshold = 10, bool nonmaxSuppression = true, FastDetectorType type = FastDetectorType.Type9_16)
    {
        Threshold = threshold;
        NonmaxSuppression = nonmaxSuppression;
        Type = type;
    }

    /// <summary>Intensity difference threshold; clamped to [0, 255] at detection.</summary>
    public int Threshold { get; set; }

    /// <summary>Non-maximum suppression flag.</summary>
    public bool NonmaxSuppression { get; set; }

    /// <summary>Circle pattern.</summary>
    public FastDetectorType Type { get; set; }

    /// <summary>
    /// Detects keypoints, as OpenCV <c>FastFeatureDetector::detect</c>: FAST on the gray image,
    /// then <see cref="KeyPointsFilter.RunByPixelsMask"/>.
    /// </summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels. 3- and 4-channel images are converted with <c>COLOR_BGR2GRAY</c>. It is not modified.</param>
    /// <param name="mask">Optional 8-bit single-channel mask of the image size; keypoints on zero mask pixels are removed.</param>
    /// <returns>A new array of keypoints in row-major order.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentException">The image channel count is not 1, 3 or 4, or the mask does not match the image size.</exception>
    public KeyPoint[] Detect(Mat<byte> image, Mat<byte>? mask = null)
    {
        Mat<byte> gray = Fast.ToGray(image, nameof(image));
        if (mask is not null && (mask.Width != image.Width || mask.Height != image.Height || mask.Channels != 1))
        {
            throw new ArgumentException("The mask must be single-channel with the image size.", nameof(mask));
        }

        if (mask is null)
        {
            return FastEngine.Detect(gray, Threshold, NonmaxSuppression, Type);
        }

        var keypoints = new List<KeyPoint>();
        FastEngine.Detect(gray, keypoints, Threshold, NonmaxSuppression, Type);
        KeyPointsFilter.RunByPixelsMask(keypoints, mask);
        return keypoints.ToArray();
    }
}
