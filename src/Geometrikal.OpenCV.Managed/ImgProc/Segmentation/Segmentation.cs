// This implementation is ported from the OpenCV marker-based watershed.
//
// Original sources:
//   modules/imgproc/src/segmentation.cpp (cv::watershed)
//   modules/imgproc/include/opencv2/imgproc.hpp
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/segmentation.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>Managed ports of OpenCV image segmentation functions.</summary>
public static class Segmentation
{
    /// <summary>
    /// Performs marker-based image segmentation with the watershed algorithm, as OpenCV
    /// <c>watershed</c>. <paramref name="markers"/> is modified in place.
    /// </summary>
    /// <remarks>
    /// <para><b>Input markers.</b> Positive values are seeds; pixels with the same value belong to
    /// the same region. 0 marks an unknown pixel.</para>
    /// <para><b>Output markers.</b> The outer one-pixel frame is set to -1 before the seeds are
    /// read, so seeds on the frame have no effect. Negative interior values are set to 0. Every
    /// unknown pixel that a seed reaches gets the seed value, or -1 where two regions meet. Unknown
    /// pixels that no seed reaches stay 0.</para>
    /// <para><b>Order.</b> Pixels are flooded in order of the maximum absolute channel difference to
    /// the neighbour that queued them (0 first), first in first out within one difference. The
    /// channel order of <paramref name="image"/> does not change the result. The function is
    /// single-threaded.</para>
    /// </remarks>
    /// <param name="image">Three-channel source image. It is not modified. It may be an ROI.</param>
    /// <param name="markers">Single-channel marker image of the image size, read and written. It may be an ROI.</param>
    /// <exception cref="ArgumentNullException">An argument is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="image"/> does not have three channels, <paramref name="markers"/> is not
    /// single-channel, or the sizes differ.
    /// </exception>
    public static void Watershed(Mat<byte> image, Mat<int> markers)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(markers);
        if (image.Channels != 3)
        {
            throw new ArgumentException("The image must have three channels.", nameof(image));
        }

        if (markers.Channels != 1)
        {
            throw new ArgumentException("The markers must be single-channel.", nameof(markers));
        }

        if (markers.Width != image.Width || markers.Height != image.Height)
        {
            throw new ArgumentException("The markers must have the size of the image.", nameof(markers));
        }

        WatershedFlood.Run(image, markers);
    }
}
