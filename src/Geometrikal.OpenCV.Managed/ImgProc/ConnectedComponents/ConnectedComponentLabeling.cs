// This implementation is ported from the OpenCV connected components labeling.
//
// Original sources:
//   modules/imgproc/src/connectedcomponents.cpp (connectedComponents, connectedComponentsWithStats)
//   modules/imgproc/include/opencv2/imgproc.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>connectedComponents</c> and <c>connectedComponentsWithStats</c>.
/// </summary>
/// <remarks>
/// <para><b>Input.</b> A single-channel <c>byte</c> or <c>sbyte</c> image. Every nonzero element
/// is foreground, including negative <c>sbyte</c> values. Pixels outside the image are background.
/// The source is not modified.</para>
/// <para><b>Labels.</b> <c>TLabel</c> is <c>int</c> (OpenCV <c>CV_32S</c>) or <c>ushort</c>
/// (<c>CV_16U</c>). Label 0 is the background; the components have the labels
/// <c>1 .. LabelCount - 1</c>. <see cref="ConnectedComponentsAlgorithm"/> selects the numbering
/// order. The labels equal the OpenCV 5.0 labels for every algorithm value.</para>
/// <para><b>ushort overflow.</b> As OpenCV 5.0, <c>ushort</c> labels throw
/// <see cref="OverflowException"/> when the scan of the selected algorithm creates provisional
/// label 65535. The provisional count can exceed the final component count.</para>
/// <para><b>Threads.</b> Large images are labeled in row stripes, as the OpenCV parallel
/// variants. The output does not depend on <see cref="CvParallel.MaxDegreeOfParallelism"/>.</para>
/// </remarks>
public static class ConnectedComponentLabeling
{
    /// <summary>Labels the connected components of a binary image, as OpenCV <c>connectedComponents</c>.</summary>
    /// <typeparam name="TLabel"><c>int</c> or <c>ushort</c>.</typeparam>
    /// <param name="image">Single-channel source image.</param>
    /// <param name="connectivity">4 or 8.</param>
    /// <param name="algorithm">Labeling algorithm; it selects the numbering order.</param>
    /// <returns>The label count and a new label image.</returns>
    /// <exception cref="ArgumentException"><paramref name="image"/> has more than one channel.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Invalid <paramref name="connectivity"/> or <paramref name="algorithm"/>.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="TLabel"/> is not <c>int</c> or <c>ushort</c>.</exception>
    /// <exception cref="OverflowException">The <c>ushort</c> label range overflowed.</exception>
    public static ConnectedComponentsResult<TLabel> ConnectedComponents<TLabel>(
        Mat<byte> image, int connectivity = 8, ConnectedComponentsAlgorithm algorithm = ConnectedComponentsAlgorithm.Default)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        var labels = new Mat<TLabel>(image.Width, image.Height, clear: false);
        int count = Label(Input(image), labels, connectivity, algorithm, false, out _, out _);
        return new ConnectedComponentsResult<TLabel>(count, labels);
    }

    /// <inheritdoc cref="ConnectedComponents{TLabel}(Mat{byte}, int, ConnectedComponentsAlgorithm)"/>
    public static ConnectedComponentsResult<TLabel> ConnectedComponents<TLabel>(
        Mat<sbyte> image, int connectivity = 8, ConnectedComponentsAlgorithm algorithm = ConnectedComponentsAlgorithm.Default)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        var labels = new Mat<TLabel>(image.Width, image.Height, clear: false);
        int count = Label(Input(image), labels, connectivity, algorithm, false, out _, out _);
        return new ConnectedComponentsResult<TLabel>(count, labels);
    }

    /// <summary>
    /// Labels the connected components of a binary image into <paramref name="labels"/>, as OpenCV
    /// <c>connectedComponents</c>.
    /// </summary>
    /// <typeparam name="TLabel"><c>int</c> or <c>ushort</c>.</typeparam>
    /// <param name="image">Single-channel source image.</param>
    /// <param name="labels">
    /// Single-channel destination of the source size; every element is written. It may be an ROI.
    /// </param>
    /// <param name="connectivity">4 or 8.</param>
    /// <param name="algorithm">Labeling algorithm; it selects the numbering order.</param>
    /// <returns>The number of labels, including the background label 0.</returns>
    /// <exception cref="ArgumentException">
    /// <paramref name="image"/> or <paramref name="labels"/> has more than one channel, or the sizes differ.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">Invalid <paramref name="connectivity"/> or <paramref name="algorithm"/>.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="TLabel"/> is not <c>int</c> or <c>ushort</c>.</exception>
    /// <exception cref="OverflowException">The <c>ushort</c> label range overflowed.</exception>
    public static int ConnectedComponents<TLabel>(
        Mat<byte> image,
        Mat<TLabel> labels,
        int connectivity = 8,
        ConnectedComponentsAlgorithm algorithm = ConnectedComponentsAlgorithm.Default)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ValidateDestination(image.Width, image.Height, labels);
        return Label(Input(image), labels, connectivity, algorithm, false, out _, out _);
    }

    /// <inheritdoc cref="ConnectedComponents{TLabel}(Mat{byte}, Mat{TLabel}, int, ConnectedComponentsAlgorithm)"/>
    public static int ConnectedComponents<TLabel>(
        Mat<sbyte> image,
        Mat<TLabel> labels,
        int connectivity = 8,
        ConnectedComponentsAlgorithm algorithm = ConnectedComponentsAlgorithm.Default)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ValidateDestination(image.Width, image.Height, labels);
        return Label(Input(image), labels, connectivity, algorithm, false, out _, out _);
    }

    /// <summary>
    /// Labels the connected components of a binary image and computes the bounding box, area and
    /// centroid of every label, as OpenCV <c>connectedComponentsWithStats</c>.
    /// </summary>
    /// <typeparam name="TLabel"><c>int</c> or <c>ushort</c>.</typeparam>
    /// <param name="image">Single-channel source image.</param>
    /// <param name="connectivity">4 or 8.</param>
    /// <param name="algorithm">Labeling algorithm; it selects the numbering order.</param>
    /// <returns>
    /// The label count, a new label image, the statistics and the centroids. Label 0 has a
    /// statistics row and a centroid like every other label. Centroids are the exact pixel
    /// coordinate sums divided by the area.
    /// </returns>
    /// <exception cref="ArgumentException"><paramref name="image"/> has more than one channel.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Invalid <paramref name="connectivity"/> or <paramref name="algorithm"/>.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="TLabel"/> is not <c>int</c> or <c>ushort</c>.</exception>
    /// <exception cref="OverflowException">The <c>ushort</c> label range overflowed.</exception>
    public static ConnectedComponentsStatsResult<TLabel> ConnectedComponentsWithStats<TLabel>(
        Mat<byte> image, int connectivity = 8, ConnectedComponentsAlgorithm algorithm = ConnectedComponentsAlgorithm.Default)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        var labels = new Mat<TLabel>(image.Width, image.Height, clear: false);
        int count = Label(Input(image), labels, connectivity, algorithm, true, out Mat<int>? stats, out Mat<double>? centroids);
        return new ConnectedComponentsStatsResult<TLabel>(count, labels, stats!, centroids!);
    }

    /// <inheritdoc cref="ConnectedComponentsWithStats{TLabel}(Mat{byte}, int, ConnectedComponentsAlgorithm)"/>
    public static ConnectedComponentsStatsResult<TLabel> ConnectedComponentsWithStats<TLabel>(
        Mat<sbyte> image, int connectivity = 8, ConnectedComponentsAlgorithm algorithm = ConnectedComponentsAlgorithm.Default)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        var labels = new Mat<TLabel>(image.Width, image.Height, clear: false);
        int count = Label(Input(image), labels, connectivity, algorithm, true, out Mat<int>? stats, out Mat<double>? centroids);
        return new ConnectedComponentsStatsResult<TLabel>(count, labels, stats!, centroids!);
    }

    private static CclImage Input(Mat<byte> image)
    {
        EnsureSingleChannel(image.Channels, nameof(image));
        return new CclImage(image.DangerousGetArray(), image.DangerousGetOffset(), image.Stride, image.Width, image.Height);
    }

    private static CclImage Input(Mat<sbyte> image)
    {
        EnsureSingleChannel(image.Channels, nameof(image));

        // The runtime allows the same-size reinterpretation sbyte[] -> byte[]; foreground is a nonzero byte.
        return new CclImage((byte[])(object)image.DangerousGetArray(), image.DangerousGetOffset(), image.Stride, image.Width, image.Height);
    }

    private static int Label<TLabel>(
        CclImage image,
        Mat<TLabel> labels,
        int connectivity,
        ConnectedComponentsAlgorithm algorithm,
        bool withStats,
        out Mat<int>? stats,
        out Mat<double>? centroids)
        where TLabel : unmanaged
    {
        if (typeof(TLabel) != typeof(int) && typeof(TLabel) != typeof(ushort))
        {
            throw new NotSupportedException("The label type must be int or ushort.");
        }

        if (connectivity is not (4 or 8))
        {
            throw new ArgumentOutOfRangeException(nameof(connectivity), connectivity, "Connectivity must be 4 or 8.");
        }

        if (algorithm is < ConnectedComponentsAlgorithm.Default or > ConnectedComponentsAlgorithm.Spaghetti)
        {
            throw new ArgumentOutOfRangeException(nameof(algorithm), algorithm, "Unknown connected components algorithm.");
        }

        return CclEngine.Run(image, labels, connectivity, algorithm, withStats, out stats, out centroids);
    }

    private static void ValidateDestination<TLabel>(int width, int height, Mat<TLabel> labels)
        where TLabel : unmanaged
    {
        ArgumentNullException.ThrowIfNull(labels);
        EnsureSingleChannel(labels.Channels, nameof(labels));
        if (labels.Width != width || labels.Height != height)
        {
            throw new ArgumentException("The label matrix must have the size of the image.", nameof(labels));
        }
    }

    private static void EnsureSingleChannel(int channels, string name)
    {
        if (channels != 1)
        {
            throw new ArgumentException("The matrix must be single-channel.", name);
        }
    }
}
