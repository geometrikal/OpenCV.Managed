// This implementation is ported from the OpenCV FAST detector.
//
// Original sources:
//   modules/features/src/fast.cpp (FAST, FAST_t, FastFeatureDetector_Impl::detect)
//   modules/features/src/fast_score.cpp (makeOffsets, cornerScore)
//   modules/features/include/opencv2/features.hpp (declarations)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// FAST corner detection, as OpenCV <c>cv::FAST</c>.
/// </summary>
/// <remarks>
/// <para><b>Output.</b> Keypoints are in row-major order (y, then x, ascending). Each keypoint has
/// size 7, angle −1, octave 0 and class −1. With non-maximum suppression, the response is the
/// corner score: the largest threshold for which the pixel is still a corner. Without
/// non-maximum suppression, the response is 0.</para>
/// <para><b>Pixels.</b> Corners are searched in rows 3 to <c>height − 4</c> and columns 3 to
/// <c>width − 4</c>. Pixels outside the image are never read. A region of interest is an
/// isolated image.</para>
/// <para><b>OpenCV quirks reproduced.</b> The candidate test of OpenCV compares 8 pairs of circle
/// pixels. For <see cref="FastDetectorType.Type9_16"/> this test is a necessary condition of the
/// arc test. For <see cref="FastDetectorType.Type5_8"/> and
/// <see cref="FastDetectorType.Type7_12"/> the pairs repeat circle pixels, so the test is
/// stricter than the arc test: a <see cref="FastDetectorType.Type5_8"/> corner needs all 8 circle
/// pixels on the same side of the threshold band. The port applies the same test.</para>
/// <para><b>Threshold.</b> The threshold is clamped to [0, 255]. OpenCV clamps it for the scalar
/// code only; its <see cref="FastDetectorType.Type9_16"/> vector code wraps a value outside
/// [0, 255] to 8 bits, so OpenCV results for such values depend on the SIMD width.</para>
/// <para><b>Score.</b> The score is the OpenCV SIMD <c>cornerScore</c>, which does not depend on
/// the threshold. OpenCV builds without SIMD use a scalar score that starts from the
/// threshold.</para>
/// </remarks>
public static class Fast
{
    /// <summary>
    /// Detects FAST corners, as OpenCV <c>cv::FAST(image, keypoints, threshold, nonmaxSuppression, type)</c>.
    /// </summary>
    /// <param name="image">8-bit single-channel image. It is not modified.</param>
    /// <param name="threshold">Intensity difference between the centre and the circle pixels; clamped to [0, 255].</param>
    /// <param name="nonmaxSuppression">Keep only corners whose score is larger than the scores of the 8 neighbours.</param>
    /// <param name="type">Circle pattern.</param>
    /// <returns>A new array of keypoints in row-major order.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="image"/> has more than one channel.</exception>
    public static KeyPoint[] Detect(Mat<byte> image, int threshold, bool nonmaxSuppression = true, FastDetectorType type = FastDetectorType.Type9_16)
    {
        ValidateGray(image);
        return FastEngine.Detect(image, threshold, nonmaxSuppression, type);
    }

    internal static void ValidateGray(Mat<byte> image)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels != 1)
        {
            throw new ArgumentException("The image must have one channel.", nameof(image));
        }
    }

    /// <summary>OpenCV <c>COLOR_BGR2GRAY</c> for 3- and 4-channel images; a single-channel image is returned unchanged.</summary>
    internal static Mat<byte> ToGray(Mat<byte> image, string paramName)
    {
        ArgumentNullException.ThrowIfNull(image, paramName);
        return image.Channels switch
        {
            1 => image,
            3 => ColorConversions.CvtColor(image, ColorConversionCode.Bgr2Gray),
            4 => ColorConversions.CvtColor(image, ColorConversionCode.Bgra2Gray),
            _ => throw new ArgumentException("The image must have 1, 3 or 4 channels.", paramName),
        };
    }
}
