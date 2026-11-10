// This implementation is ported from the OpenCV distance transforms.
//
// Original sources:
//   modules/imgproc/src/distransform.cpp (cv::distanceTransform, both overloads)
//   modules/imgproc/include/opencv2/imgproc.hpp
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/distransform.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>distanceTransform</c> overloads, without IPP.
/// </summary>
/// <remarks>
/// <para><b>Input.</b> A single-channel <c>byte</c> image. Every pixel gets the distance to the
/// nearest zero pixel. Nonzero values are all equal foreground. The source is not modified.
/// Pixels outside the image are not zero pixels.</para>
/// <para><b>Routing.</b> The OpenCV 5.0 routing applies:</para>
/// <list type="bullet">
/// <item><description><see cref="DistanceTypes.L1"/> and <see cref="DistanceTypes.C"/> without
/// labels always use the 3x3 mask; <see cref="DistanceTransformMasks.Mask5"/> and
/// <see cref="DistanceTransformMasks.Precise"/> give the 3x3 result.</description></item>
/// <item><description><see cref="DistanceTransformMasks.Precise"/> computes the exact Euclidean
/// distance and does not check the distance type: every <see cref="DistanceTypes"/> value except
/// L1 and C gives the same result.</description></item>
/// <item><description>The labelled transform always uses the 5x5 mask. The mask argument is not
/// read, and is not validated.</description></item>
/// <item><description>The <c>byte</c> destination overload is the OpenCV <c>dstType = CV_8U</c>
/// path: L1 with a 3x3 mask, saturated at 255. The mask argument is not read.</description></item>
/// </list>
/// <para><b>Values (OpenCV 5.0).</b> Chamfer distances are fixed-point sums with 16 fraction bits.
/// A pixel that no zero pixel reaches (an image without zero pixels) gets
/// <c>(uint.MaxValue - w) / 65536</c>, where <c>w</c> is the largest mask weight; the precise
/// transform gives 65536. Precise distances of 65536 pixels or more are 65536. cv2 4.x before 4.9
/// clamps chamfer distances at 8192 instead.</para>
/// <para><b>Threads.</b> The precise transform and the 8-bit L1 transform run their row and column
/// passes in parallel. The float chamfer transforms are sequential, as OpenCV. The output does not depend on
/// <see cref="CvParallel.MaxDegreeOfParallelism"/>.</para>
/// </remarks>
public static class DistanceTransforms
{
    /// <summary>
    /// Computes the distance from every pixel to the nearest zero pixel, as OpenCV
    /// <c>distanceTransform(src, dst, distanceType, maskSize, CV_32F)</c>.
    /// </summary>
    /// <param name="source">Single-channel source image.</param>
    /// <param name="distanceType">L1, L2 or C; any value with <see cref="DistanceTransformMasks.Precise"/>.</param>
    /// <param name="maskSize">Mask size; see the routing rules on <see cref="DistanceTransforms"/>.</param>
    /// <returns>A new distance image of the source size.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="source"/> has more than one channel.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="maskSize"/> is not 0, 3 or 5, or a chamfer mask is used with a distance type
    /// other than L1, L2 or C.
    /// </exception>
    public static Mat<float> DistanceTransform(Mat<byte> source, DistanceTypes distanceType, DistanceTransformMasks maskSize)
    {
        ArgumentNullException.ThrowIfNull(source);
        var destination = new Mat<float>(source.Width, source.Height, 1, clear: false);
        DistanceTransform(source, destination, distanceType, maskSize);
        return destination;
    }

    /// <summary>
    /// Computes the distance from every pixel to the nearest zero pixel into
    /// <paramref name="destination"/>, as OpenCV <c>distanceTransform(src, dst, distanceType, maskSize, CV_32F)</c>.
    /// </summary>
    /// <param name="source">Single-channel source image.</param>
    /// <param name="destination">Single-channel destination of the source size; every element is written. It may be an ROI.</param>
    /// <param name="distanceType">L1, L2 or C; any value with <see cref="DistanceTransformMasks.Precise"/>.</param>
    /// <param name="maskSize">Mask size; see the routing rules on <see cref="DistanceTransforms"/>.</param>
    /// <exception cref="ArgumentNullException">An argument is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">A matrix has more than one channel, or the sizes differ.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="maskSize"/> is not 0, 3 or 5, or a chamfer mask is used with a distance type
    /// other than L1, L2 or C.
    /// </exception>
    public static void DistanceTransform(Mat<byte> source, Mat<float> destination, DistanceTypes distanceType, DistanceTransformMasks maskSize)
    {
        ValidateSource(source);
        ValidateDestination(source, destination, nameof(destination));
        if (maskSize is not (DistanceTransformMasks.Mask3 or DistanceTransformMasks.Mask5 or DistanceTransformMasks.Precise))
        {
            throw new ArgumentOutOfRangeException(nameof(maskSize), maskSize, "The mask size must be 3, 5 or 0 (precise).");
        }

        if (distanceType is DistanceTypes.C or DistanceTypes.L1)
        {
            maskSize = DistanceTransformMasks.Mask3;
        }

        if (maskSize == DistanceTransformMasks.Precise)
        {
            PreciseDistance.Transform(source, destination);
            return;
        }

        ValidateChamferType(distanceType);
        if (maskSize == DistanceTransformMasks.Mask3)
        {
            ChamferDistance.Transform3x3(source, destination, ChamferDistance.Weights(distanceType, false));
        }
        else
        {
            ChamferDistance.Transform5x5(source, destination, ChamferDistance.Weights(distanceType, true));
        }
    }

    /// <summary>
    /// Computes the L1 distance from every pixel to the nearest zero pixel into a <c>byte</c>
    /// destination, as OpenCV <c>distanceTransform(src, dst, DIST_L1, maskSize, CV_8U)</c>.
    /// Distances above 255 saturate to 255.
    /// </summary>
    /// <param name="source">Single-channel source image.</param>
    /// <param name="destination">
    /// Single-channel destination of the source size; every element is written. It may be an ROI,
    /// and it may be <paramref name="source"/> itself.
    /// </param>
    /// <param name="distanceType">Must be <see cref="DistanceTypes.L1"/>.</param>
    /// <param name="maskSize">Not read; the 3x3 L1 mask is used, as OpenCV.</param>
    /// <exception cref="ArgumentNullException">An argument is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// A matrix has more than one channel, the sizes differ, or <paramref name="distanceType"/> is
    /// not L1. OpenCV ignores <c>dstType</c> for other types and returns a float image; a
    /// <c>byte</c> destination cannot hold that result.
    /// </exception>
    public static void DistanceTransform(Mat<byte> source, Mat<byte> destination, DistanceTypes distanceType, DistanceTransformMasks maskSize)
    {
        _ = maskSize;
        ValidateSource(source);
        ValidateDestination(source, destination, nameof(destination));
        if (distanceType != DistanceTypes.L1)
        {
            throw new ArgumentException("A byte destination requires the L1 distance.", nameof(distanceType));
        }

        DistanceL1Byte.Transform(source, destination);
    }

    /// <summary>
    /// Computes the distance to the nearest zero pixel and the label of that zero pixel or of its
    /// component, as OpenCV <c>distanceTransform(src, dst, labels, distanceType, maskSize, labelType)</c>.
    /// </summary>
    /// <param name="source">Single-channel source image.</param>
    /// <param name="distanceType">L1, L2 or C.</param>
    /// <param name="maskSize">Not read; the labelled transform always uses the 5x5 mask, as OpenCV.</param>
    /// <param name="labelType">Label numbering of the zero pixels.</param>
    /// <returns>New distance and label images of the source size.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="source"/> has more than one channel.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Invalid <paramref name="labelType"/> or <paramref name="distanceType"/>.</exception>
    public static DistanceTransformResult DistanceTransformWithLabels(
        Mat<byte> source,
        DistanceTypes distanceType,
        DistanceTransformMasks maskSize,
        DistanceTransformLabelTypes labelType = DistanceTransformLabelTypes.ConnectedComponent)
    {
        ArgumentNullException.ThrowIfNull(source);
        var distances = new Mat<float>(source.Width, source.Height, 1, clear: false);
        var labels = new Mat<int>(source.Width, source.Height, 1, clear: false);
        DistanceTransformWithLabels(source, distances, labels, distanceType, maskSize, labelType);
        return new DistanceTransformResult(distances, labels);
    }

    /// <summary>
    /// Computes the distance to the nearest zero pixel and the label of that zero pixel or of its
    /// component into <paramref name="destination"/> and <paramref name="labels"/>, as OpenCV
    /// <c>distanceTransform(src, dst, labels, distanceType, maskSize, labelType)</c>.
    /// </summary>
    /// <remarks>
    /// Zero pixels keep their initial label. Every other pixel gets the label of the neighbour that
    /// gives its distance; the 5x5 mask neighbours are tested in the OpenCV order and the first
    /// minimum wins. A pixel that no zero pixel reaches gets label 0.
    /// </remarks>
    /// <param name="source">Single-channel source image.</param>
    /// <param name="destination">Single-channel distance destination of the source size. It may be an ROI.</param>
    /// <param name="labels">Single-channel label destination of the source size. It may be an ROI.</param>
    /// <param name="distanceType">L1, L2 or C.</param>
    /// <param name="maskSize">Not read; the labelled transform always uses the 5x5 mask, as OpenCV.</param>
    /// <param name="labelType">Label numbering of the zero pixels.</param>
    /// <exception cref="ArgumentNullException">An argument is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">A matrix has more than one channel, or the sizes differ.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Invalid <paramref name="labelType"/> or <paramref name="distanceType"/>.</exception>
    public static void DistanceTransformWithLabels(
        Mat<byte> source,
        Mat<float> destination,
        Mat<int> labels,
        DistanceTypes distanceType,
        DistanceTransformMasks maskSize,
        DistanceTransformLabelTypes labelType = DistanceTransformLabelTypes.ConnectedComponent)
    {
        _ = maskSize;
        ValidateSource(source);
        ValidateDestination(source, destination, nameof(destination));
        ValidateDestination(source, labels, nameof(labels));
        if (labelType is not (DistanceTransformLabelTypes.ConnectedComponent or DistanceTransformLabelTypes.Pixel))
        {
            throw new ArgumentOutOfRangeException(nameof(labelType), labelType, "Unknown label type.");
        }

        ValidateChamferType(distanceType);
        ChamferDistance.Transform5x5Labels(source, destination, labels, ChamferDistance.Weights(distanceType, true), labelType);
    }

    private static void ValidateSource(Mat<byte> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels != 1)
        {
            throw new ArgumentException("The source must be single-channel.", nameof(source));
        }
    }

    private static void ValidateDestination<T>(Mat<byte> source, Mat<T> destination, string name)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(destination, name);
        if (destination.Channels != 1)
        {
            throw new ArgumentException("The destination must be single-channel.", name);
        }

        if (destination.Width != source.Width || destination.Height != source.Height)
        {
            throw new ArgumentException("The destination must have the size of the source.", name);
        }
    }

    private static void ValidateChamferType(DistanceTypes distanceType)
    {
        if (distanceType is not (DistanceTypes.C or DistanceTypes.L1 or DistanceTypes.L2))
        {
            throw new ArgumentOutOfRangeException(nameof(distanceType), distanceType, "The chamfer masks support L1, L2 and C only.");
        }
    }
}
