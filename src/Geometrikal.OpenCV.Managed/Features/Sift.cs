// This implementation is ported from the OpenCV SIFT detector and descriptor.
//
// Original sources:
//   modules/features/src/sift.dispatch.cpp (SIFT_Impl, SIFT::create, detectAndCompute)
//   modules/features/src/sift.simd.hpp (findScaleSpaceExtrema, calcSIFTDescriptor)
//   modules/features/src/feature2d.cpp (Feature2D::detect, Feature2D::compute)
//   modules/features/include/opencv2/features.hpp (declaration)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.Features.Internal;

/// <summary>Element type of SIFT descriptors, as the OpenCV <c>descriptorType</c> argument.</summary>
public enum SiftDescriptorType
{
    /// <summary><c>CV_32F</c>: float descriptors (the OpenCV default).</summary>
    Float32 = 5,

    /// <summary><c>CV_8U</c>: byte descriptors.</summary>
    Byte = 0,
}

/// <summary>
/// Scale-invariant feature transform (SIFT) keypoint detector and descriptor extractor, as
/// OpenCV <c>cv::SIFT</c> (D. Lowe, 2004).
/// </summary>
/// <remarks>
/// <para><b>Input.</b> 8-bit images with 1, 3 or 4 channels. Images with 3 or 4 channels are
/// converted to gray with the BGR coefficients (<c>COLOR_BGR2GRAY</c>), as OpenCV. The mask is
/// an optional single-channel 8-bit image of the image size; keypoints at a zero mask pixel are
/// removed after detection.</para>
/// <para><b>Results.</b> The keypoints are sorted by OpenCV <c>KeyPointsFilter::removeDuplicatedSorted</c>
/// (x and y ascending, size descending, angle ascending). When <see cref="NFeatures"/> is
/// positive, <c>KeyPointsFilter::retainBest</c> keeps the strongest keypoints and the port then
/// orders them by decreasing response (see <see cref="KeyPointsFilter.RetainBest"/>; the OpenCV
/// order is implementation-defined). The output does not depend on the thread count.</para>
/// <para><b>Descriptors.</b> One row of 128 values per keypoint. Each value is a rounded integer in
/// <c>[0, 255]</c>, stored as <c>float</c> or <c>byte</c> by <see cref="DescriptorType"/>. A result
/// without keypoints has no descriptor matrix (<see langword="null"/>), because a
/// <see cref="Mat{T}"/> cannot have zero rows.</para>
/// <para><b>Exactness.</b> The arithmetic follows the OpenCV 5.0 128-bit baseline code (no AVX,
/// no FMA). <c>expf</c>, <c>powf</c>, <c>cosf</c> and <c>sinf</c> are the C runtime functions
/// through <see cref="MathF"/>; results equal the Windows build of OpenCV when both use the same
/// C runtime.</para>
/// <para><b>Mutating.</b> <see cref="Compute(Mat{byte}, IReadOnlyList{KeyPoint}, out Mat{float}?)"/>
/// does not change the keypoint list. The other methods return new lists.</para>
/// </remarks>
public sealed class Sift
{
    /// <summary>Descriptor length in elements (<c>descriptorSize()</c>).</summary>
    public const int DescriptorSize = SiftDescriptor.Length;

    /// <summary>
    /// Creates a SIFT detector, as OpenCV <c>SIFT::create</c>.
    /// </summary>
    /// <param name="nFeatures">Number of best keypoints to keep; 0 keeps all.</param>
    /// <param name="nOctaveLayers">Layers per octave; Lowe uses 3.</param>
    /// <param name="contrastThreshold">Contrast threshold that rejects weak extrema; the effective threshold is <c>contrastThreshold / nOctaveLayers</c>.</param>
    /// <param name="edgeThreshold">Edge threshold that rejects edge-like extrema; a larger value keeps more keypoints.</param>
    /// <param name="sigma">Gaussian sigma of the octave-0 image.</param>
    /// <param name="descriptorType">Descriptor element type.</param>
    /// <param name="enablePreciseUpscale">True: the octave −1 image comes from <c>warpAffine</c> with a 0.5 scale (OpenCV default). False: from <c>resize</c> with bilinear interpolation.</param>
    /// <exception cref="ArgumentOutOfRangeException">A parameter is out of range.</exception>
    public Sift(
        int nFeatures = 0,
        int nOctaveLayers = 3,
        double contrastThreshold = 0.04,
        double edgeThreshold = 10,
        double sigma = 1.6,
        SiftDescriptorType descriptorType = SiftDescriptorType.Float32,
        bool enablePreciseUpscale = true)
    {
        NFeatures = nFeatures;
        NOctaveLayers = nOctaveLayers;
        ContrastThreshold = contrastThreshold;
        EdgeThreshold = edgeThreshold;
        Sigma = sigma;
        if (descriptorType is not (SiftDescriptorType.Float32 or SiftDescriptorType.Byte))
        {
            throw new ArgumentOutOfRangeException(nameof(descriptorType));
        }

        DescriptorType = descriptorType;
        EnablePreciseUpscale = enablePreciseUpscale;
    }

    /// <summary>Number of best keypoints to keep; 0 or less keeps all.</summary>
    public int NFeatures { get; set; }

    /// <summary>Layers per octave.</summary>
    public int NOctaveLayers
    {
        get => _nOctaveLayers;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            _nOctaveLayers = value;
        }
    }

    /// <summary>Contrast threshold.</summary>
    public double ContrastThreshold { get; set; }

    /// <summary>Edge threshold.</summary>
    public double EdgeThreshold { get; set; }

    /// <summary>Gaussian sigma of the octave-0 image.</summary>
    public double Sigma { get; set; }

    /// <summary>Descriptor element type.</summary>
    public SiftDescriptorType DescriptorType { get; }

    /// <summary>True when the octave −1 image uses the precise <c>warpAffine</c> upscale.</summary>
    public bool EnablePreciseUpscale { get; set; }

    private int _nOctaveLayers;

    /// <summary>Detects keypoints, as OpenCV <c>Feature2D::detect</c>.</summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels.</param>
    /// <param name="mask">Optional single-channel mask of the image size.</param>
    /// <returns>A new list of keypoints.</returns>
    public List<KeyPoint> Detect(Mat<byte> image, Mat<byte>? mask = null)
    {
        var keypoints = new List<KeyPoint>();
        Run(image, mask, keypoints, false, 0, out _, out _);
        return keypoints;
    }

    /// <summary>
    /// Detects keypoints and computes their float descriptors, as OpenCV
    /// <c>detectAndCompute</c> with <c>useProvidedKeypoints = false</c>.
    /// </summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels.</param>
    /// <param name="mask">Optional single-channel mask of the image size.</param>
    /// <param name="descriptors">Receives one row per keypoint, or <see langword="null"/> when no keypoint is found.</param>
    /// <returns>A new list of keypoints.</returns>
    /// <exception cref="InvalidOperationException"><see cref="DescriptorType"/> is not <see cref="SiftDescriptorType.Float32"/>.</exception>
    public List<KeyPoint> DetectAndCompute(Mat<byte> image, Mat<byte>? mask, out Mat<float>? descriptors)
    {
        RequireType(SiftDescriptorType.Float32);
        var keypoints = new List<KeyPoint>();
        Run(image, mask, keypoints, false, 1, out descriptors, out _);
        return keypoints;
    }

    /// <summary>
    /// Detects keypoints and computes their byte descriptors, as OpenCV
    /// <c>detectAndCompute</c> with <c>descriptorType = CV_8U</c>.
    /// </summary>
    /// <inheritdoc cref="DetectAndCompute(Mat{byte}, Mat{byte}?, out Mat{float}?)"/>
    /// <exception cref="InvalidOperationException"><see cref="DescriptorType"/> is not <see cref="SiftDescriptorType.Byte"/>.</exception>
    public List<KeyPoint> DetectAndCompute(Mat<byte> image, Mat<byte>? mask, out Mat<byte>? descriptors)
    {
        RequireType(SiftDescriptorType.Byte);
        var keypoints = new List<KeyPoint>();
        Run(image, mask, keypoints, false, 2, out _, out descriptors);
        return keypoints;
    }

    /// <summary>
    /// Computes float descriptors for given keypoints, as OpenCV <c>Feature2D::compute</c>
    /// (<c>detectAndCompute</c> with <c>useProvidedKeypoints = true</c>). The keypoint
    /// octaves select the pyramid levels; a keypoint with octave −1 builds the upscaled level.
    /// </summary>
    /// <param name="image">8-bit image with 1, 3 or 4 channels.</param>
    /// <param name="keypoints">Keypoints with SIFT octave encoding (for example from <see cref="Detect"/>). Not changed.</param>
    /// <param name="descriptors">Receives one row per keypoint, or <see langword="null"/> when the list is empty.</param>
    /// <exception cref="ArgumentException">A keypoint octave is below −1 or a layer is above <see cref="NOctaveLayers"/> + 2.</exception>
    /// <exception cref="InvalidOperationException"><see cref="DescriptorType"/> is not <see cref="SiftDescriptorType.Float32"/>.</exception>
    public void Compute(Mat<byte> image, IReadOnlyList<KeyPoint> keypoints, out Mat<float>? descriptors)
    {
        RequireType(SiftDescriptorType.Float32);
        ArgumentNullException.ThrowIfNull(keypoints);
        descriptors = null;
        if (keypoints.Count == 0)
        {
            return;
        }

        var list = new List<KeyPoint>(keypoints);
        Run(image, null, list, true, 1, out descriptors, out _);
    }

    /// <summary>Computes byte descriptors for given keypoints.</summary>
    /// <inheritdoc cref="Compute(Mat{byte}, IReadOnlyList{KeyPoint}, out Mat{float}?)"/>
    /// <exception cref="InvalidOperationException"><see cref="DescriptorType"/> is not <see cref="SiftDescriptorType.Byte"/>.</exception>
    public void Compute(Mat<byte> image, IReadOnlyList<KeyPoint> keypoints, out Mat<byte>? descriptors)
    {
        RequireType(SiftDescriptorType.Byte);
        ArgumentNullException.ThrowIfNull(keypoints);
        descriptors = null;
        if (keypoints.Count == 0)
        {
            return;
        }

        var list = new List<KeyPoint>(keypoints);
        Run(image, null, list, true, 2, out _, out descriptors);
    }

    private void RequireType(SiftDescriptorType type)
    {
        if (DescriptorType != type)
        {
            throw new InvalidOperationException($"The detector creates {DescriptorType} descriptors.");
        }
    }

    /// <summary>OpenCV <c>SIFT_Impl::detectAndCompute</c>. <paramref name="descriptorMode"/>: 0 none, 1 float, 2 byte.</summary>
    private void Run(
        Mat<byte> image,
        Mat<byte>? mask,
        List<KeyPoint> keypoints,
        bool useProvidedKeypoints,
        int descriptorMode,
        out Mat<float>? floatOut,
        out Mat<byte>? byteOut)
    {
        floatOut = null;
        byteOut = null;
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels is not (1 or 3 or 4))
        {
            throw new ArgumentException("The image must have 1, 3 or 4 channels.", nameof(image));
        }

        if (mask is not null && mask.Channels != 1)
        {
            throw new ArgumentException("The mask must have one channel.", nameof(mask));
        }

        int nl = NOctaveLayers;
        int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
        if (useProvidedKeypoints)
        {
            firstOctave = 0;
            int maxOctave = int.MinValue;
            foreach (KeyPoint kp in keypoints)
            {
                SiftDescriptor.UnpackOctave(kp, out int octave, out int layer, out _);
                firstOctave = Math.Min(firstOctave, octave);
                maxOctave = Math.Max(maxOctave, octave);
                actualNLayers = Math.Max(actualNLayers, layer - 2);
            }

            firstOctave = Math.Min(firstOctave, 0);
            if (firstOctave < -1 || actualNLayers > nl)
            {
                throw new ArgumentException("A keypoint octave or layer is outside the SIFT pyramid.", nameof(keypoints));
            }

            actualNOctaves = maxOctave - firstOctave + 1;
        }

        var rented = new List<PooledMat<float>>();
        try
        {
            Mat<float> baseImage = SiftPyramid.CreateInitialImage(image, firstOctave < 0, (float)Sigma, EnablePreciseUpscale, rented);
            int nOctaves = actualNOctaves > 0
                ? actualNOctaves
                : ScalarConverter.CvRound((Math.Log(Math.Min(baseImage.Width, baseImage.Height)) / Math.Log(2.0)) - 2) - firstOctave;

            Mat<float>[] gpyr = SiftPyramid.BuildGaussianPyramid(baseImage, nOctaves, nl, Sigma, rented);

            if (!useProvidedKeypoints)
            {
                Mat<float>[] dogpyr = SiftPyramid.BuildDoGPyramid(gpyr, nl, rented);
                keypoints.Clear();
                SiftExtrema.Find(gpyr, dogpyr, nl, ContrastThreshold, EdgeThreshold, Sigma, keypoints);
                KeyPointsFilter.RemoveDuplicatedSorted(keypoints);
                if (NFeatures > 0)
                {
                    KeyPointsFilter.RetainBest(keypoints, NFeatures);
                }

                if (firstOctave < 0)
                {
                    float scale = 1f / (1 << -firstOctave);
                    for (int i = 0; i < keypoints.Count; i++)
                    {
                        KeyPoint kpt = keypoints[i];
                        kpt.Octave = (kpt.Octave & ~255) | ((kpt.Octave + firstOctave) & 255);
                        kpt.Pt = new Point2f(kpt.Pt.X * scale, kpt.Pt.Y * scale);
                        kpt.Size *= scale;
                        keypoints[i] = kpt;
                    }
                }

                if (mask is not null)
                {
                    KeyPointsFilter.RunByPixelsMask(keypoints, mask);
                }
            }

            if (descriptorMode == 0 || keypoints.Count == 0)
            {
                return;
            }

            if (descriptorMode == 1)
            {
                floatOut = new Mat<float>(DescriptorSize, keypoints.Count, 1, clear: false);
                SiftDescriptor.Compute(gpyr, keypoints, nl, firstOctave, floatOut, null);
            }
            else
            {
                byteOut = new Mat<byte>(DescriptorSize, keypoints.Count, 1, clear: false);
                SiftDescriptor.Compute(gpyr, keypoints, nl, firstOctave, null, byteOut);
            }
        }
        finally
        {
            foreach (PooledMat<float> m in rented)
            {
                m.Dispose();
            }
        }
    }
}
