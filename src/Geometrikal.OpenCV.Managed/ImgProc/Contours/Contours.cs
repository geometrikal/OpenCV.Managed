// This implementation is ported from the OpenCV contour implementation.
//
// Original sources:
//   modules/imgproc/src/contours_new.cpp      (in this repo: src/OpenCV/modules/imgproc/src/contours_new.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Contour extraction, a managed port of OpenCV's <c>cv::findContours</c>
/// (Suzuki-Abe border following, <c>modules/imgproc/src/contours_new.cpp</c>).
/// </summary>
public static class Contours
{
    /// <summary>
    /// Finds contours (connected-component boundaries) in a binary image, using
    /// the Suzuki-Abe algorithm, and returns them together with their topological
    /// hierarchy.
    /// </summary>
    /// <param name="image">
    /// The single-channel (<see cref="Mat{T}.Channels"/> == 1) source image. The
    /// image is treated as binary: a pixel is <em>foreground</em> when its value
    /// is non-zero and <em>background</em> when it is zero, matching OpenCV's
    /// internal <c>threshold(img, 0, 1, THRESH_BINARY)</c> on an 8-bit unsigned
    /// image. Regions of interest and buffers with a stride larger than their row
    /// length are fully supported; only per-row spans are read.
    /// </param>
    /// <param name="mode">
    /// The retrieval mode, controlling which contours are returned and how the
    /// hierarchy is organised:
    /// <list type="bullet">
    /// <item><description><see cref="RetrievalMode.External"/> — only the
    /// outermost contours; every hierarchy parent is <c>-1</c>.</description></item>
    /// <item><description><see cref="RetrievalMode.List"/> — all contours as a
    /// flat sibling list with no parent/child links.</description></item>
    /// <item><description><see cref="RetrievalMode.ConnectedComponents"/> — a
    /// two-level hierarchy: component outer boundaries at the top level, their
    /// hole boundaries at the second level.</description></item>
    /// <item><description><see cref="RetrievalMode.Tree"/> — the full nesting
    /// hierarchy.</description></item>
    /// </list>
    /// </param>
    /// <param name="method">
    /// The approximation method applied to each border:
    /// <list type="bullet">
    /// <item><description><see cref="ContourApproximationMode.None"/> — every
    /// border pixel is stored.</description></item>
    /// <item><description><see cref="ContourApproximationMode.Simple"/> —
    /// horizontal, vertical, and diagonal runs are collapsed to their end
    /// points.</description></item>
    /// <item><description><see cref="ContourApproximationMode.Tc89L1"/> /
    /// <see cref="ContourApproximationMode.Tc89KCos"/> — the Teh-Chin dominant
    /// point approximations.</description></item>
    /// </list>
    /// </param>
    /// <param name="offset">
    /// A coordinate offset added to every returned point. Useful when the image
    /// is a region of interest extracted from a larger image and the contours are
    /// to be expressed in the coordinate space of that larger image. Defaults to
    /// <c>(0, 0)</c>.
    /// </param>
    /// <returns>
    /// A <see cref="ContourResult"/> whose <see cref="ContourResult.Contours"/>
    /// and <see cref="ContourResult.Hierarchy"/> lists are always of equal length.
    /// Contours are returned in OpenCV's discovery order (a depth-first walk of
    /// the tree, newest sibling first). Each contour is a sequence of the border
    /// pixels themselves, in image coordinates (X right, Y down): outer contours
    /// wind counter-clockwise and holes clockwise. Each hierarchy entry gives the
    /// indices <c>(Next, Previous, FirstChild, Parent)</c>, with <c>-1</c> where
    /// there is no such neighbour. When no contours are found, both lists are
    /// empty.
    /// </returns>
    /// <remarks>
    /// The source image is never modified: all border marking happens on a
    /// private, internally allocated, zero-bordered working copy.
    ///
    /// <para><b>Deliberate differences from OpenCV.</b> The following OpenCV
    /// behaviours are intentionally not reproduced:
    /// <list type="bullet">
    /// <item><description><c>RETR_FLOODFILL</c> (which requires 32-bit integer
    /// images), <c>LINK_RUNS</c>, and raw chain-code output (OpenCV's
    /// <c>method == 0</c>) are unsupported.</description></item>
    /// <item><description><see cref="RetrievalMode.List"/> always runs through the
    /// hierarchical scanner. OpenCV 5 uses a parallel <c>findTRUContours</c> fast
    /// path when no hierarchy is requested; that path yields equivalent contours,
    /// and this port matches the ordering of OpenCV's hierarchical path.</description></item>
    /// <item><description>The horizontal-run scan is scalar (no SIMD), and border
    /// point/label storage uses managed lists rather than OpenCV's block
    /// storage.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="image"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="image"/> is not single-channel.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="mode"/> or <paramref name="method"/> is not one of the
    /// supported enumeration values.
    /// </exception>
    public static ContourResult FindContours(
        Mat<byte> image,
        RetrievalMode mode,
        ContourApproximationMode method,
        Point offset = default)
    {
        // 1. Validation (spec 5.2).
        ArgumentNullException.ThrowIfNull(image);

        if (image.Channels != 1)
        {
            throw new ArgumentException(
                "The input image must be single-channel.",
                nameof(image));
        }

        if (mode is not (RetrievalMode.External
            or RetrievalMode.List
            or RetrievalMode.ConnectedComponents
            or RetrievalMode.Tree))
        {
            throw new ArgumentOutOfRangeException(
                nameof(mode),
                mode,
                "Unsupported contour retrieval mode.");
        }

        if (method is not (ContourApproximationMode.None
            or ContourApproximationMode.Simple
            or ContourApproximationMode.Tc89L1
            or ContourApproximationMode.Tc89KCos))
        {
            throw new ArgumentOutOfRangeException(
                nameof(method),
                method,
                "Unsupported contour approximation method.");
        }

        // 2. Build the fused border + binarize working image (spec 5.1).
        //    Equivalent to copyMakeBorder(src, 1, 1, 1, 1, BORDER_CONSTANT, 0)
        //    followed by threshold(0, 1, THRESH_BINARY).
        int paddedWidth = checked(image.Width + 2);
        int paddedHeight = checked(image.Height + 2);
        var working = new sbyte[checked(paddedWidth * paddedHeight)];

        for (int y = 0; y < image.Height; y++)
        {
            ReadOnlySpan<byte> srcRow = image.GetReadOnlyRowSpan(y);
            int rowBase = ((y + 1) * paddedWidth) + 1;
            for (int x = 0; x < image.Width; x++)
            {
                working[rowBase + x] = srcRow[x] != 0 ? (sbyte)1 : (sbyte)0;
            }
        }

        // 3-4. Scan every border into the tree (spec 5.7-5.8).
        var scanner = new ContourScanner(
            working,
            paddedWidth,
            paddedHeight,
            mode,
            method,
            offset + new Point(-1, -1));

        while (scanner.FindNext())
        {
        }

        // 5. Flatten the tree into the public result (spec 7).
        return ContourResultBuilder.Build(scanner.Tree);
    }
}
