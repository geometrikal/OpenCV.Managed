// This implementation is ported from the OpenCV ORB detector and descriptor.
//
// Original sources:
//   modules/features/src/orb.cpp (ORB::create, ORB_Impl)
//   modules/features/src/feature2d.cpp (Feature2D::detect, Feature2D::compute)
//   modules/features/include/opencv2/features.hpp (declarations)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features.Internal;

/// <summary>
/// ORB (oriented FAST and rotated BRIEF) keypoint detector and binary descriptor extractor, as
/// OpenCV <c>cv::ORB</c> on the CPU path.
/// </summary>
/// <remarks>
/// <para><b>Pyramid.</b> Level <c>l</c> has the scale <c>scaleFactor^(l − firstLevel)</c> in single
/// precision. Levels are resized with <see cref="ImgProc.InterpolationFlags.LinearExact"/> from
/// the previous level (or from the input for levels below <see cref="FirstLevel"/>) and stored
/// with a <see cref="BorderType.Reflect101"/> border in one buffer, as OpenCV.</para>
/// <para><b>Keypoints.</b> Positions are in input image coordinates. <see cref="KeyPoint.Octave"/>
/// is the pyramid level, <see cref="KeyPoint.Size"/> is <c>patchSize · scale</c>,
/// <see cref="KeyPoint.Angle"/> is the intensity-centroid angle in degrees and
/// <see cref="KeyPoint.Response"/> is the Harris or FAST score. Keypoints are grouped by level.
/// Inside a level, OpenCV orders them with <c>std::nth_element</c> (implementation-defined). The
/// port uses <see cref="KeyPointsFilter.RetainBest"/>: a level that is reduced is sorted by
/// decreasing score, with ties in row-major detection order; a level that is not reduced keeps
/// the order of the previous step. The set of keypoints and the value of every member equal
/// OpenCV.</para>
/// <para><b>Descriptors.</b> One row of 32 bytes per keypoint, sampled on the pyramid level after
/// a 7x7 Gaussian blur with sigma 2. With <see cref="WtaK"/> 2 each bit compares a point pair;
/// with 3 or 4 each 2-bit field is the index of the brightest point of a tuple.</para>
/// <para><b>Input.</b> 8-bit images with 1, 3 or 4 channels; 3- and 4-channel images are converted
/// with <c>COLOR_BGR2GRAY</c>. A region of interest is an isolated image. OpenCV builds the border
/// of level <see cref="FirstLevel"/> from the parent matrix of a region of interest.</para>
/// <para><b>Thread safety.</b> An instance can run on several threads at the same time when its
/// properties do not change.</para>
/// </remarks>
public sealed class Orb
{
    private int _firstLevel;

    /// <summary>
    /// Creates an ORB instance, as OpenCV <c>ORB::create</c>.
    /// </summary>
    /// <param name="nfeatures">Maximum number of keypoints to keep.</param>
    /// <param name="scaleFactor">Pyramid decimation ratio; larger than 1. OpenCV <c>ORB::create</c> takes a <c>float</c>, so the default is <c>(double)1.2f</c>.</param>
    /// <param name="nlevels">Number of pyramid levels.</param>
    /// <param name="edgeThreshold">Border, in pixels of each level, in which no keypoint is detected.</param>
    /// <param name="firstLevel">Pyramid level of the input image; non-negative.</param>
    /// <param name="wtaK">Points per descriptor comparison: 2, 3 or 4.</param>
    /// <param name="scoreType">Keypoint ranking score.</param>
    /// <param name="patchSize">Size of the oriented BRIEF patch; at least 2.</param>
    /// <param name="fastThreshold">FAST threshold.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="firstLevel"/> is negative.</exception>
    public Orb(
        int nfeatures = 500,
        float scaleFactor = 1.2f,
        int nlevels = 8,
        int edgeThreshold = 31,
        int firstLevel = 0,
        int wtaK = 2,
        OrbScoreType scoreType = OrbScoreType.Harris,
        int patchSize = 31,
        int fastThreshold = 20)
    {
        MaxFeatures = nfeatures;
        ScaleFactor = scaleFactor;
        NLevels = nlevels;
        EdgeThreshold = edgeThreshold;
        FirstLevel = firstLevel;
        WtaK = wtaK;
        ScoreType = scoreType;
        PatchSize = patchSize;
        FastThreshold = fastThreshold;
    }

    /// <summary>Maximum number of keypoints to keep (OpenCV <c>nfeatures</c>).</summary>
    public int MaxFeatures { get; set; }

    /// <summary>Pyramid decimation ratio; must be larger than 1 at detection.</summary>
    public double ScaleFactor { get; set; }

    /// <summary>Number of pyramid levels.</summary>
    public int NLevels { get; set; }

    /// <summary>Border in which no keypoint is detected.</summary>
    public int EdgeThreshold { get; set; }

    /// <summary>Pyramid level of the input image; non-negative.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is negative.</exception>
    public int FirstLevel
    {
        get => _firstLevel;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _firstLevel = value;
        }
    }

    /// <summary>Points per descriptor comparison: 2, 3 or 4 (checked when descriptors are computed).</summary>
    public int WtaK { get; set; }

    /// <summary>Keypoint ranking score.</summary>
    public OrbScoreType ScoreType { get; set; }

    /// <summary>Size of the oriented BRIEF patch; at least 2.</summary>
    public int PatchSize { get; set; }

    /// <summary>FAST threshold.</summary>
    public int FastThreshold { get; set; }

    /// <summary>Descriptor length in bytes (32).</summary>
    public int DescriptorSize => OrbEngine.DescriptorSize;

    /// <summary>
    /// Detects keypoints, as OpenCV <c>Feature2D::detect</c>.
    /// </summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels. It is not modified.</param>
    /// <param name="mask">Optional 8-bit single-channel mask of the image size; keypoints are detected only where the mask is non-zero.</param>
    /// <returns>A new array of keypoints, grouped by level.</returns>
    /// <exception cref="ArgumentException">The image, mask or settings are invalid.</exception>
    public KeyPoint[] Detect(Mat<byte> image, Mat<byte>? mask = null)
    {
        var keypoints = new List<KeyPoint>();
        Run(image, mask, keypoints, useProvidedKeypoints: false, doDescriptors: false);
        return keypoints.ToArray();
    }

    /// <summary>
    /// Computes descriptors for given keypoints, as OpenCV <c>Feature2D::compute</c>. Keypoints
    /// within <see cref="EdgeThreshold"/> of the image border are removed. When the octaves are not
    /// in ascending order, the keypoints are grouped by octave (stable).
    /// </summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels. It is not modified.</param>
    /// <param name="keypoints">Keypoints in image coordinates; <see cref="KeyPoint.Octave"/> selects the pyramid level and <see cref="KeyPoint.Angle"/> the orientation.</param>
    /// <returns>The kept keypoints and their descriptors (null when no keypoint is kept).</returns>
    /// <exception cref="ArgumentException">The image or settings are invalid, or an octave is negative.</exception>
    public (KeyPoint[] Keypoints, Mat<byte>? Descriptors) Compute(Mat<byte> image, IReadOnlyList<KeyPoint> keypoints)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        var list = new List<KeyPoint>(keypoints);
        Mat<byte>? descriptors = Run(image, null, list, useProvidedKeypoints: true, doDescriptors: true);
        return (list.ToArray(), descriptors);
    }

    /// <summary>
    /// Detects keypoints and computes their descriptors, as OpenCV
    /// <c>ORB::detectAndCompute(image, mask, keypoints, descriptors, false)</c>.
    /// </summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels. It is not modified.</param>
    /// <param name="mask">Optional 8-bit single-channel mask of the image size.</param>
    /// <returns>The keypoints and their descriptors (null when no keypoint is found).</returns>
    /// <exception cref="ArgumentException">The image, mask or settings are invalid.</exception>
    public (KeyPoint[] Keypoints, Mat<byte>? Descriptors) DetectAndCompute(Mat<byte> image, Mat<byte>? mask = null)
    {
        var keypoints = new List<KeyPoint>();
        Mat<byte>? descriptors = Run(image, mask, keypoints, useProvidedKeypoints: false, doDescriptors: true);
        return (keypoints.ToArray(), descriptors);
    }

    private Mat<byte>? Run(Mat<byte> image, Mat<byte>? mask, List<KeyPoint> keypoints, bool useProvidedKeypoints, bool doDescriptors)
    {
        Mat<byte> gray = Fast.ToGray(image, nameof(image));
        if (mask is not null && (mask.Width != image.Width || mask.Height != image.Height || mask.Channels != 1))
        {
            throw new ArgumentException("The mask must be single-channel with the image size.", nameof(mask));
        }

        if (PatchSize < 2)
        {
            throw new ArgumentException("PatchSize must be at least 2.");
        }

        if (!(ScaleFactor > 1))
        {
            throw new ArgumentException("ScaleFactor must be larger than 1.");
        }

        if (!useProvidedKeypoints && NLevels < 1)
        {
            throw new ArgumentException("NLevels must be at least 1.");
        }

        var settings = new OrbSettings(MaxFeatures, ScaleFactor, NLevels, EdgeThreshold, FirstLevel, WtaK, ScoreType, PatchSize, FastThreshold);
        return OrbEngine.Run(settings, gray, mask, keypoints, useProvidedKeypoints, doDescriptors);
    }
}
