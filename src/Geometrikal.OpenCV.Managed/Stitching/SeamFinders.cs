// This implementation is ported from the OpenCV stitching seam finders.
//
// Original sources:
//   modules/stitching/src/seam_finders.cpp (SeamFinder, NoSeamFinder, PairwiseSeamFinder,
//     VoronoiSeamFinder, DpSeamFinder, GraphCutSeamFinder)
//   modules/stitching/include/opencv2/stitching/detail/seam_finders.hpp (declarations)
//   modules/stitching/src/util.cpp (overlapRoi)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>Seam finder kinds of <see cref="SeamFinder.CreateDefault"/> (OpenCV <c>SeamFinder::NO</c>, <c>VORONOI_SEAM</c>, <c>DP_SEAM</c>).</summary>
public enum SeamFinderType
{
    /// <summary><see cref="NoSeamFinder"/>.</summary>
    No = 0,

    /// <summary><see cref="VoronoiSeamFinder"/>.</summary>
    VoronoiSeam = 1,

    /// <summary><see cref="DpSeamFinder"/> with <see cref="DpSeamCostFunction.Color"/>.</summary>
    DpSeam = 2,
}

/// <summary>Cost functions of <see cref="DpSeamFinder"/> (OpenCV <c>DpSeamFinder::CostFunction</c>).</summary>
public enum DpSeamCostFunction
{
    /// <summary>Squared colour difference of the two images across the seam (<c>COLOR</c>).</summary>
    Color = 0,

    /// <summary>Squared colour difference divided by the gray-level Sobel gradient magnitudes plus 1 (<c>COLOR_GRAD</c>).</summary>
    ColorGrad = 1,
}

/// <summary>Edge cost types of <see cref="GraphCutSeamFinder"/> (OpenCV <c>GraphCutSeamFinderBase::CostType</c>).</summary>
public enum GraphCutCostType
{
    /// <summary>Squared colour difference (<c>COST_COLOR</c>).</summary>
    Color = 0,

    /// <summary>Squared colour difference divided by the squared Sobel gradient norms (<c>COST_COLOR_GRAD</c>).</summary>
    ColorGrad = 1,
}

/// <summary>
/// Base class of the seam estimators, ported from OpenCV <c>cv::detail::SeamFinder</c>.
/// </summary>
/// <remarks>
/// <para><b>Inputs.</b> <c>src[i]</c> is image <c>i</c>, placed with its top-left corner at
/// <c>corners[i]</c> in the common (panorama) coordinates. <c>masks[i]</c> is a one-channel
/// <c>byte</c> matrix with the size of <c>src[i]</c>; a non-zero value marks a valid pixel.</para>
/// <para><b>Mutation.</b> <c>Find</c> changes the masks in place: in each overlap, it sets to 0
/// the pixels that the seam assigns to the other image. The images are not changed. A mask can
/// be a region of interest of a larger matrix.</para>
/// <para><b>Errors.</b> The input checks run before any mask changes.</para>
/// </remarks>
public abstract class SeamFinder
{
    private protected SeamFinder()
    {
    }

    /// <summary>Creates a seam finder as OpenCV <c>SeamFinder::createDefault</c>.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not defined.</exception>
    public static SeamFinder CreateDefault(SeamFinderType type) => type switch
    {
        SeamFinderType.No => new NoSeamFinder(),
        SeamFinderType.VoronoiSeam => new VoronoiSeamFinder(),
        SeamFinderType.DpSeam => new DpSeamFinder(),
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "Unsupported seam finder method."),
    };

    /// <summary>Estimates the seams and updates <paramref name="masks"/> in place (OpenCV <c>find</c>).</summary>
    /// <typeparam name="T">Image element type. See the derived class for the supported types.</typeparam>
    /// <param name="src">Source images.</param>
    /// <param name="corners">Top-left corner of each image.</param>
    /// <param name="masks">One-channel mask of each image; updated in place.</param>
    /// <exception cref="ArgumentException">The list counts differ, or a mask is not a one-channel matrix with the image size.</exception>
    /// <exception cref="NotSupportedException">The element type or channel count is not supported.</exception>
    public abstract void Find<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
        where T : unmanaged;

    private protected static void ValidateLists<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(src);
        ArgumentNullException.ThrowIfNull(corners);
        ArgumentNullException.ThrowIfNull(masks);
        if (corners.Count != src.Count || masks.Count != src.Count)
        {
            throw new ArgumentException("src, corners and masks must have the same count.");
        }

        for (int i = 0; i < src.Count; i++)
        {
            ArgumentNullException.ThrowIfNull(src[i]);
            ValidateMask(masks[i], src[i].Width, src[i].Height, i);
        }
    }

    private protected static void ValidateMask(Mat<byte> mask, int width, int height, int i)
    {
        if (mask is null)
        {
            throw new ArgumentNullException(nameof(mask), $"Mask {i} is null.");
        }

        if (mask.Channels != 1 || mask.Width != width || mask.Height != height)
        {
            throw new ArgumentException($"Mask {i} must be a one-channel matrix of {width}x{height}.");
        }
    }

    private protected static Mat<T>[] ToArray<T>(IReadOnlyList<Mat<T>> src)
        where T : unmanaged
    {
        var images = new Mat<T>[src.Count];
        for (int i = 0; i < images.Length; i++)
        {
            images[i] = src[i];
        }

        return images;
    }
}

/// <summary>
/// Stub seam estimator that does not change the masks, ported from OpenCV
/// <c>cv::detail::NoSeamFinder</c>.
/// </summary>
public sealed class NoSeamFinder : SeamFinder
{
    /// <summary>Does nothing, as OpenCV. The arguments are not validated.</summary>
    /// <typeparam name="T">Any unmanaged element type.</typeparam>
    public override void Find<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
    }
}

/// <summary>
/// Base class of the pairwise seam estimators, ported from OpenCV
/// <c>cv::detail::PairwiseSeamFinder</c>.
/// </summary>
/// <remarks>
/// The estimator visits the image pairs (i, j) with i &lt; j in index order. For each pair with
/// a non-empty overlap of the image rectangles, it resolves the overlap of the two masks. The
/// masks that a pair changes are the inputs of the next pairs.
/// </remarks>
public abstract class PairwiseSeamFinder : SeamFinder
{
    private protected PairwiseSeamFinder()
    {
    }

    private protected static void Run(IReadOnlyList<Size> sizes, IReadOnlyList<Point> corners, Action<int, int, Rect> findInPair)
    {
        for (int i = 0; i < sizes.Count - 1; i++)
        {
            for (int j = i + 1; j < sizes.Count; j++)
            {
                if (SeamRoi.OverlapRoi(corners[i], corners[j], sizes[i], sizes[j], out Rect roi))
                {
                    findInPair(i, j, roi);
                }
            }
        }
    }
}

/// <summary>
/// Voronoi-diagram seam estimator, ported from OpenCV <c>cv::detail::VoronoiSeamFinder</c>.
/// </summary>
/// <remarks>
/// For each overlapping pair, the estimator takes the overlap rectangle with a 10-pixel gap on
/// each side. It computes the L1 3x3 distance transform to the pixels that only the first mask
/// covers and to the pixels that only the second mask covers. A pixel of the overlap rectangle
/// stays in the second mask if it is strictly nearer to the second set; otherwise it stays in
/// the first mask. The other mask gets 0 at that pixel, also where that mask was already 0. The
/// image pixels are not read, so any element type and channel count is accepted.
/// </remarks>
public sealed class VoronoiSeamFinder : PairwiseSeamFinder
{
    /// <summary>Estimates the seams from the image sizes (OpenCV <c>find(src, corners, masks)</c>).</summary>
    /// <typeparam name="T">Any unmanaged element type; only the image sizes are used.</typeparam>
    public override void Find<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
        ValidateLists(src, corners, masks);
        var sizes = new Size[src.Count];
        for (int i = 0; i < sizes.Length; i++)
        {
            sizes[i] = new Size(src[i].Width, src[i].Height);
        }

        FindCore(sizes, corners, masks);
    }

    /// <summary>Estimates the seams from image sizes (OpenCV <c>find(sizes, corners, masks)</c>).</summary>
    /// <param name="sizes">Size of each image.</param>
    /// <param name="corners">Top-left corner of each image.</param>
    /// <param name="masks">One-channel mask of each image with the size in <paramref name="sizes"/>; updated in place.</param>
    /// <exception cref="ArgumentException">The list counts differ, or a mask does not have its size.</exception>
    public void Find(IReadOnlyList<Size> sizes, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
        ArgumentNullException.ThrowIfNull(sizes);
        ArgumentNullException.ThrowIfNull(corners);
        ArgumentNullException.ThrowIfNull(masks);
        if (corners.Count != sizes.Count || masks.Count != sizes.Count)
        {
            throw new ArgumentException("sizes, corners and masks must have the same count.");
        }

        for (int i = 0; i < sizes.Count; i++)
        {
            ValidateMask(masks[i], sizes[i].Width, sizes[i].Height, i);
        }

        FindCore(sizes, corners, masks);
    }

    private static void FindCore(IReadOnlyList<Size> sizes, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
        if (sizes.Count == 0)
        {
            return;
        }

        Run(sizes, corners, (i, j, roi) => SeamVoronoi.FindInPair(masks[i], masks[j], corners[i], corners[j], roi));
    }
}

/// <summary>
/// Dynamic-programming seam estimator, ported from OpenCV <c>cv::detail::DpSeamFinder</c>.
/// </summary>
/// <remarks>
/// <para><b>Algorithm.</b> The estimator processes the image pairs in decreasing order of the
/// squared distance between the image centres. For each pair, it labels the 4-connected
/// components of the union of the masks (first only, second only, both), finds the adjacent
/// components, and resolves each conflict between an intersection component and a neighbour:
/// it merges a component with one neighbour, or it cuts the component along a minimum-cost
/// path between two seam tips. Then it removes the pixels of each mask that the labels assign
/// to the other image.</para>
/// <para><b>Images.</b> <c>byte</c> or <c>float</c> with 3 or 4 channels (OpenCV <c>CV_8UC3</c>,
/// <c>CV_8UC4</c>, <c>CV_32FC3</c>, <c>CV_32FC4</c>); all images have the same type. The fourth
/// channel is not read by the colour cost. <see cref="DpSeamCostFunction.ColorGrad"/> converts
/// each image to gray (BGR or BGRA) and takes 3x3 Sobel derivatives.</para>
/// <para><b>Pair order.</b> Pairs with equal centre distances keep the index order before the
/// reversal, as the OpenCV <c>std::sort</c> insertion sort for up to 32 pairs (8 images).</para>
/// </remarks>
public sealed class DpSeamFinder : SeamFinder
{
    /// <summary>Creates the estimator with a cost function.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="costFunction"/> is not defined.</exception>
    public DpSeamFinder(DpSeamCostFunction costFunction = DpSeamCostFunction.Color)
    {
        CostFunction = costFunction;
    }

    /// <summary>Gets or sets the cost function.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is not defined.</exception>
    public DpSeamCostFunction CostFunction
    {
        get;
        set => field = value is DpSeamCostFunction.Color or DpSeamCostFunction.ColorGrad
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown cost function.");
    }

    /// <inheritdoc cref="SeamFinder.Find{T}"/>
    /// <typeparam name="T"><c>byte</c> or <c>float</c>.</typeparam>
    public override void Find<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
        ValidateLists(src, corners, masks);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"DpSeamFinder does not support element type '{typeof(T).Name}'.");
        }

        if (src.Count == 0)
        {
            return;
        }

        int cn = src[0].Channels;
        for (int i = 0; i < src.Count; i++)
        {
            if (src[i].Channels != cn || (cn != 3 && cn != 4))
            {
                throw new NotSupportedException("DpSeamFinder images must all have 3 or all have 4 channels.");
            }
        }

        using var dp = new SeamDp<T>(CostFunction, ToArray(src));
        dp.Find(corners, masks);
    }
}

/// <summary>
/// Minimum graph-cut seam estimator, ported from OpenCV <c>cv::detail::GraphCutSeamFinder</c>
/// (Kwatra et al. 2003).
/// </summary>
/// <remarks>
/// <para><b>Algorithm.</b> For each overlapping pair (index order), the estimator builds a grid
/// graph over the overlap rectangle with a 10-pixel gap on each side. A pixel in the first mask
/// gets a source weight of <see cref="TerminalCost"/>; a pixel in the second mask gets a sink
/// weight of <see cref="TerminalCost"/>. An edge between two adjacent pixels costs the colour
/// cost plus 1, plus <see cref="BadRegionPenalty"/> when one of the four mask values is 0. After
/// the max-flow, a source-side pixel is removed from the second mask when the first mask
/// covers it; a sink-side pixel is removed from the first mask when the second mask covers it.</para>
/// <para><b>Images.</b> <c>float</c> with 3 channels (OpenCV <c>CV_32FC3</c>), as the OpenCV
/// stitching pipeline (<c>images_warped_f</c>). Other types throw
/// <see cref="NotSupportedException"/>.</para>
/// </remarks>
public sealed class GraphCutSeamFinder : SeamFinder
{
    /// <summary>Creates the estimator.</summary>
    /// <param name="costType">Edge cost type.</param>
    /// <param name="terminalCost">Terminal weight of the masked pixels.</param>
    /// <param name="badRegionPenalty">Edge penalty where a mask is 0.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="costType"/> is not defined.</exception>
    public GraphCutSeamFinder(GraphCutCostType costType = GraphCutCostType.ColorGrad, float terminalCost = 10000f, float badRegionPenalty = 1000f)
    {
        if (costType is not (GraphCutCostType.Color or GraphCutCostType.ColorGrad))
        {
            throw new ArgumentOutOfRangeException(nameof(costType), costType, "Unsupported pixel similarity measure.");
        }

        CostType = costType;
        TerminalCost = terminalCost;
        BadRegionPenalty = badRegionPenalty;
    }

    /// <summary>Gets the edge cost type.</summary>
    public GraphCutCostType CostType { get; }

    /// <summary>Gets the terminal weight of the masked pixels.</summary>
    public float TerminalCost { get; }

    /// <summary>Gets the edge penalty where a mask is 0.</summary>
    public float BadRegionPenalty { get; }

    /// <inheritdoc cref="SeamFinder.Find{T}"/>
    /// <typeparam name="T"><c>float</c>; the images have 3 channels.</typeparam>
    public override void Find<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
        ValidateLists(src, corners, masks);
        if (typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"GraphCutSeamFinder supports float images only, not '{typeof(T).Name}'.");
        }

        for (int i = 0; i < src.Count; i++)
        {
            if (src[i].Channels != 3)
            {
                throw new NotSupportedException("GraphCutSeamFinder images must have 3 channels.");
            }
        }

        if (src.Count == 0)
        {
            return;
        }

        var images = (Mat<float>[])(object)ToArray(src);
        var dx = new float[images.Length][];
        var dy = new float[images.Length][];
        var sizes = new Size[images.Length];
        for (int i = 0; i < images.Length; i++)
        {
            // The colour cost does not read the gradients.
            (dx[i], dy[i]) = CostType == GraphCutCostType.ColorGrad ? SeamGraphCut.Gradients(images[i]) : ([], []);
            sizes[i] = new Size(images[i].Width, images[i].Height);
        }

        GraphCutCostType costType = CostType;
        float terminalCost = TerminalCost, penalty = BadRegionPenalty;
        for (int i = 0; i < sizes.Length - 1; i++)
        {
            for (int j = i + 1; j < sizes.Length; j++)
            {
                if (SeamRoi.OverlapRoi(corners[i], corners[j], sizes[i], sizes[j], out Rect roi))
                {
                    SeamGraphCut.FindInPair(
                        images[i], images[j], dx[i], dy[i], dx[j], dy[j], masks[i], masks[j], corners[i], corners[j], roi, costType, terminalCost, penalty);
                }
            }
        }
    }
}
