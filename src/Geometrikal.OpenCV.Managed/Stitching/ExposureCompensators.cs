// This implementation is ported from the OpenCV stitching exposure compensators.
//
// Original sources:
//   modules/stitching/src/exposure_compensate.cpp
//   modules/stitching/include/opencv2/stitching/detail/exposure_compensate.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>Exposure compensator kind, as OpenCV <c>ExposureCompensator</c> constants.</summary>
public enum ExposureCompensatorType
{
    /// <summary><see cref="NoExposureCompensator"/>.</summary>
    No = 0,

    /// <summary><see cref="GainCompensator"/>.</summary>
    Gain = 1,

    /// <summary><see cref="BlocksGainCompensator"/>.</summary>
    GainBlocks = 2,

    /// <summary><see cref="ChannelsCompensator"/>.</summary>
    Channels = 3,

    /// <summary><see cref="BlocksChannelsCompensator"/>.</summary>
    ChannelsBlocks = 4,
}

/// <summary>
/// Base class of the exposure compensators, as OpenCV <c>detail::ExposureCompensator</c>.
/// </summary>
/// <remarks>
/// <para><b>Sequence.</b> <c>Feed</c> estimates the gains from images placed at their corners.
/// <c>Apply</c> multiplies one image by its gains, in place.</para>
/// <para><b>Inputs.</b> Images are <c>byte</c> images (1 or 3 channels for the gain compensators,
/// 3 channels for the channel and block compensators). A mask pixel takes part when it equals the
/// mask level (255 for the overload without levels). <c>Feed</c> does not modify the images.
/// OpenCV multiplies the fed images by the intermediate gains when the feed count is above 1;
/// the port works on copies.</para>
/// </remarks>
public abstract class ExposureCompensator
{
    /// <summary>Gets or sets a value that tells <c>Feed</c> to compute new gains. When false, existing gains of the same image count are kept.</summary>
    public bool UpdateGain { get; set; } = true;

    /// <summary>Creates a compensator of the given kind with default parameters, as OpenCV <c>createDefault</c>.</summary>
    /// <param name="type">Compensator kind.</param>
    /// <returns>The compensator.</returns>
    public static ExposureCompensator CreateDefault(ExposureCompensatorType type) => type switch
    {
        ExposureCompensatorType.No => new NoExposureCompensator(),
        ExposureCompensatorType.Gain => new GainCompensator(),
        ExposureCompensatorType.GainBlocks => new BlocksGainCompensator(),
        ExposureCompensatorType.Channels => new ChannelsCompensator(),
        ExposureCompensatorType.ChannelsBlocks => new BlocksChannelsCompensator(),
        _ => throw new ArgumentOutOfRangeException(nameof(type)),
    };

    /// <summary>Estimates the gains, as OpenCV <c>feed(corners, images, masks)</c> with mask level 255.</summary>
    /// <param name="corners">Top-left corners of the images in panorama coordinates.</param>
    /// <param name="images">Images. They are not modified.</param>
    /// <param name="masks">1-channel masks of the image sizes.</param>
    public void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>> masks)
    {
        ArgumentNullException.ThrowIfNull(masks);
        Feed(corners, images, masks.Select(m => (m, (byte)255)).ToArray());
    }

    /// <summary>Estimates the gains with mask levels, as OpenCV <c>feed(corners, images, level_masks)</c>.</summary>
    /// <param name="corners">Top-left corners of the images in panorama coordinates.</param>
    /// <param name="images">Images. They are not modified.</param>
    /// <param name="masks">Masks and the mask value that marks the valid pixels.</param>
    public abstract void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks);

    /// <summary>Compensates one image in place, as OpenCV <c>apply</c>.</summary>
    /// <param name="index">Image index in the fed list.</param>
    /// <param name="corner">Image corner (not used).</param>
    /// <param name="image">Image. It is modified.</param>
    /// <param name="mask">Image mask (not used).</param>
    public abstract void Apply(int index, Point corner, Mat<byte> image, Mat<byte>? mask);

    private protected static void ValidateFeed<TMask>(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<TMask> masks)
    {
        ArgumentNullException.ThrowIfNull(corners);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(masks);
        if (corners.Count != images.Count || images.Count != masks.Count)
        {
            throw new ArgumentException("The corner, image and mask lists must have the same length.");
        }
    }
}

/// <summary>Compensator that does nothing, as OpenCV <c>detail::NoExposureCompensator</c>.</summary>
public sealed class NoExposureCompensator : ExposureCompensator
{
    /// <inheritdoc/>
    public override void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks)
    {
    }

    /// <inheritdoc/>
    public override void Apply(int index, Point corner, Mat<byte> image, Mat<byte>? mask)
    {
    }
}

/// <summary>
/// Compensator with one gain per image, as OpenCV <c>detail::GainCompensator</c>.
/// </summary>
/// <remarks>
/// <para>For each overlapping pair the compensator sums the pixel intensities (1 channel) or the
/// pixel L2 norms (3 channels) in <c>double</c>, and solves the linear system of OpenCV
/// <c>singleFeed</c> (alpha 0.01, beta 100) with LU in <c>double</c>. Images without overlap keep
/// the gain 1. With <see cref="NrFeeds"/> above 1, each further feed runs on the images multiplied
/// by the previous gains, and the gains multiply.</para>
/// <para>With <see cref="SimilarityThreshold"/> below 1, only overlap pixels whose colour distance
/// (L2 of the differences / 255) is at most the threshold count; the similarity mask is eroded and
/// dilated with a 3×3 rectangle. The masks are computed on the first feed and kept for later feeds
/// of the same instance, as OpenCV.</para>
/// <para>The gains are bit-exact with OpenCV built without Eigen (cv2 wheels).</para>
/// </remarks>
public class GainCompensator : ExposureCompensator
{
    private readonly List<Mat<byte>> _similarities = [];
    private double[] _gains = [];

    /// <summary>Creates a gain compensator.</summary>
    /// <param name="nrFeeds">Number of feed iterations.</param>
    public GainCompensator(int nrFeeds = 1)
    {
        NrFeeds = nrFeeds;
    }

    /// <summary>Gets or sets the number of feed iterations.</summary>
    public int NrFeeds { get; set; }

    /// <summary>Gets or sets the similarity threshold. A value of 1 or more disables the similarity mask.</summary>
    public double SimilarityThreshold { get; set; } = 1;

    /// <summary>Gets or sets the gains, one per image, as OpenCV <c>gains</c>/<c>setMatGains</c>.</summary>
    public double[] Gains
    {
        get => (double[])_gains.Clone();
        set => _gains = (double[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <inheritdoc/>
    public override void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks)
    {
        ValidateFeed(corners, images, masks);
        int n = images.Count;
        PrepareSimilarityMask(corners, images);
        IReadOnlyList<Mat<byte>> work = images;
        double[]? accumulated = null;
        for (int f = 0; f < NrFeeds; f++)
        {
            if (f > 0)
            {
                if (ReferenceEquals(work, images))
                {
                    work = images.Select(m => m.Clone()).ToArray();
                }

                for (int i = 0; i < n; i++)
                {
                    Apply(i, corners[i], work[i], masks[i].Mask);
                }
            }

            SingleFeed(corners, work, masks);
            if (f == 0)
            {
                accumulated = (double[])_gains.Clone();
            }
            else
            {
                for (int i = 0; i < accumulated!.Length; i++)
                {
                    accumulated[i] *= _gains[i];
                }
            }
        }

        if (accumulated is not null)
        {
            _gains = accumulated;
        }
    }

    /// <inheritdoc/>
    public override void Apply(int index, Point corner, Mat<byte> image, Mat<byte>? mask)
    {
        ArgumentNullException.ThrowIfNull(image);
        Arithmetic.Multiply(image, Scalar.All(_gains[index]), image);
    }

    /// <summary>OpenCV <c>prepareSimilarityMask</c>.</summary>
    internal void PrepareSimilarityMask(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images)
    {
        if (SimilarityThreshold >= 1 || _similarities.Count > 0)
        {
            return;
        }

        for (int i = 0; i < images.Count; i++)
        {
            for (int j = i; j < images.Count; j++)
            {
                if (OverlapViews(corners, images, i, j, out Mat<byte> s1, out Mat<byte> s2, out _))
                {
                    _similarities.Add(ExposureKernels.SimilarityMask(s1, s2, SimilarityThreshold));
                }
            }
        }
    }

    private void SingleFeed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks)
    {
        int n = images.Count;
        if (n == 0)
        {
            return;
        }

        int channels = images[0].Channels;
        if (images.Any(m => m.Channels != channels) || channels is not (1 or 3))
        {
            throw new ArgumentException("All images must have the same channel count, 1 or 3.", nameof(images));
        }

        var nMat = new int[n, n];
        var iMat = new double[n, n];
        var skip = new bool[n];
        Array.Fill(skip, true);
        int simIndex = 0;
        var pairs = new List<(int I, int J, Mat<byte> Sub1, Mat<byte> Sub2, Mat<byte> M1, Mat<byte> M2, Mat<byte>? Sim)>();
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                if (!OverlapViews(corners, images, i, j, out Mat<byte> sub1, out Mat<byte> sub2, out Rect roi))
                {
                    continue;
                }

                Mat<byte> m1 = masks[i].Mask.Roi(roi.X - corners[i].X, roi.Y - corners[i].Y, roi.Width, roi.Height);
                Mat<byte> m2 = masks[j].Mask.Roi(roi.X - corners[j].X, roi.Y - corners[j].Y, roi.Width, roi.Height);
                Mat<byte>? sim = _similarities.Count > 0 ? _similarities[simIndex++] : null;
                pairs.Add((i, j, sub1, sub2, m1, m2, sim));
            }
        }

        var counts = new int[pairs.Count];
        var sums1 = new double[pairs.Count];
        var sums2 = new double[pairs.Count];
        ParallelRows.ForEachSerial(pairs.Count, p =>
        {
            var (i, j, sub1, sub2, m1, m2, sim) = pairs[p];
            counts[p] = ExposureKernels.OverlapSums(sub1, sub2, m1, masks[i].Level, m2, masks[j].Level, sim, out sums1[p], out sums2[p]);
        });

        for (int p = 0; p < pairs.Count; p++)
        {
            int i = pairs[p].I, j = pairs[p].J;
            int count = counts[p];
            nMat[i, j] = nMat[j, i] = Math.Max(1, count);
            if (count == 0)
            {
                continue;
            }

            if (i != j)
            {
                skip[i] = false;
                skip[j] = false;
            }

            iMat[i, j] = sums1[p] / nMat[i, j];
            iMat[j, i] = sums2[p] / nMat[i, j];
        }

        if (!UpdateGain && _gains.Length == n)
        {
            return;
        }

        const double alpha = 0.01;
        const double beta = 100;
        int numEq = skip.Count(s => !s);
        _gains = Enumerable.Repeat(1.0, n).ToArray();
        if (numEq == 0)
        {
            return;
        }

        var a = new Mat<double>(numEq, numEq);
        var b = new Mat<double>(1, numEq);
        for (int i = 0, ki = 0; i < n; i++)
        {
            if (skip[i])
            {
                continue;
            }

            for (int j = 0, kj = 0; j < n; j++)
            {
                if (skip[j])
                {
                    continue;
                }

                b.At(ki, 0) += beta * nMat[i, j];
                a.At(ki, ki) += beta * nMat[i, j];
                if (j != i)
                {
                    a.At(ki, ki) += 2 * alpha * iMat[i, j] * iMat[i, j] * nMat[i, j];
                    a.At(ki, kj) -= 2 * alpha * iMat[i, j] * iMat[j, i] * nMat[i, j];
                }

                ++kj;
            }

            ++ki;
        }

        LinearAlgebra.Solve(a, b, out Mat<double> x, DecompTypes.Lu);
        for (int i = 0, j = 0; i < n; i++)
        {
            if (!skip[i])
            {
                _gains[i] = x.At(j++, 0);
            }
        }
    }

    private static bool OverlapViews(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, int i, int j, out Mat<byte> sub1, out Mat<byte> sub2, out Rect roi)
    {
        if (!StitchingUtils.OverlapRoi(corners[i], corners[j], new Size(images[i].Width, images[i].Height), new Size(images[j].Width, images[j].Height), out roi))
        {
            sub1 = sub2 = null!;
            return false;
        }

        sub1 = images[i].Roi(roi.X - corners[i].X, roi.Y - corners[i].Y, roi.Width, roi.Height);
        sub2 = images[j].Roi(roi.X - corners[j].X, roi.Y - corners[j].Y, roi.Width, roi.Height);
        return true;
    }
}

/// <summary>
/// Compensator with one gain per image and channel, as OpenCV <c>detail::ChannelsCompensator</c>.
/// </summary>
/// <remarks>
/// Runs <see cref="GainCompensator"/> on each channel of 3-channel images. The similarity mask is
/// computed once from the 3-channel images and used for every channel.
/// </remarks>
public class ChannelsCompensator : ExposureCompensator
{
    private Scalar[] _gains = [];

    /// <summary>Creates a channels compensator.</summary>
    /// <param name="nrFeeds">Number of feed iterations.</param>
    public ChannelsCompensator(int nrFeeds = 1)
    {
        NrFeeds = nrFeeds;
    }

    /// <summary>Gets or sets the number of feed iterations.</summary>
    public int NrFeeds { get; set; }

    /// <summary>Gets or sets the similarity threshold. A value of 1 or more disables the similarity mask.</summary>
    public double SimilarityThreshold { get; set; } = 1;

    /// <summary>Gets or sets the gains, one <see cref="Scalar"/> (channels 0 to 2) per image.</summary>
    public Scalar[] Gains
    {
        get => (Scalar[])_gains.Clone();
        set => _gains = (Scalar[])(value ?? throw new ArgumentNullException(nameof(value))).Clone();
    }

    /// <inheritdoc/>
    public override void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks)
    {
        ValidateFeed(corners, images, masks);
        if (images.Any(m => m.Channels != 3))
        {
            throw new ArgumentException("The images must have 3 channels.", nameof(images));
        }

        var planes = new Mat<byte>[3][];
        for (int c = 0; c < 3; c++)
        {
            planes[c] = new Mat<byte>[images.Count];
        }

        for (int i = 0; i < images.Count; i++)
        {
            Mat<byte>[] split = Conversions.Split(images[i]);
            for (int c = 0; c < 3; c++)
            {
                planes[c][i] = split[c];
            }
        }

        var gains = new double[images.Count, 3];
        var compensator = new GainCompensator(NrFeeds) { SimilarityThreshold = SimilarityThreshold };
        compensator.PrepareSimilarityMask(corners, images);
        for (int c = 0; c < 3; c++)
        {
            compensator.Feed(corners, planes[c], masks);
            double[] g = compensator.Gains;
            for (int i = 0; i < g.Length; i++)
            {
                gains[i, c] = g[i];
            }
        }

        _gains = new Scalar[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            _gains[i] = new Scalar(gains[i, 0], gains[i, 1], gains[i, 2]);
        }
    }

    /// <inheritdoc/>
    public override void Apply(int index, Point corner, Mat<byte> image, Mat<byte>? mask)
    {
        ArgumentNullException.ThrowIfNull(image);
        Arithmetic.Multiply(image, _gains[index], image);
    }
}

/// <summary>
/// Base class of the block compensators, as OpenCV <c>detail::BlocksCompensator</c>.
/// </summary>
/// <remarks>
/// <para>Each image is divided into a grid of blocks of about <see cref="BlockSize"/>. A gain
/// (or channels) compensator runs on all blocks. The gains of each image form a gain map, which is
/// smoothed <see cref="NrGainsFilteringIterations"/> times with the separable kernel
/// [0.25 0.5 0.25] (reflect-101 border).</para>
/// <para><c>Apply</c> resizes the gain map to the image size (bilinear) and computes
/// <c>saturate_cast&lt;byte&gt;((float)v · gain)</c>. It accepts 3-channel images only.</para>
/// </remarks>
public abstract class BlocksCompensator : ExposureCompensator
{
    private Mat<float>[] _gainMaps = [];

    private protected BlocksCompensator(int blockWidth, int blockHeight, int nrFeeds)
    {
        BlockSize = new Size(blockWidth, blockHeight);
        NrFeeds = nrFeeds;
    }

    /// <summary>Gets or sets the requested block size.</summary>
    public Size BlockSize { get; set; }

    /// <summary>Gets or sets the number of feed iterations.</summary>
    public int NrFeeds { get; set; }

    /// <summary>Gets or sets the similarity threshold. A value of 1 or more disables the similarity mask.</summary>
    public double SimilarityThreshold { get; set; } = 1;

    /// <summary>Gets or sets the number of gain-map smoothing passes.</summary>
    public int NrGainsFilteringIterations { get; set; } = 2;

    /// <summary>Gets or sets the gain maps, one per image (1 channel for gains, 3 channels for channel gains).</summary>
    public Mat<float>[] GainMaps
    {
        get => _gainMaps.Select(m => m.Clone()).ToArray();
        set => _gainMaps = (value ?? throw new ArgumentNullException(nameof(value))).Select(m => m.Clone()).ToArray();
    }

    /// <inheritdoc/>
    public override void Apply(int index, Point corner, Mat<byte> image, Mat<byte>? mask)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels != 3)
        {
            throw new ArgumentException("The image must have 3 channels.", nameof(image));
        }

        Mat<float> map = _gainMaps[index];
        if (map.Width != image.Width || map.Height != image.Height)
        {
            map = Geometry.Resize(map, new Size(image.Width, image.Height), 0, 0, InterpolationFlags.Linear);
        }

        if (map.Channels != 3)
        {
            map = Conversions.Merge(map, map, map);
        }

        ExposureKernels.MultiplyByFloatMap(image, map);
    }

    private protected void FeedWithStrategy(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks, bool channels)
    {
        ValidateFeed(corners, images, masks);
        int n = images.Count;
        var perImage = new Size[n];
        var blockCorners = new List<Point>();
        var blockImages = new List<Mat<byte>>();
        var blockMasks = new List<(Mat<byte>, byte)>();
        for (int i = 0; i < n; i++)
        {
            Mat<byte> img = images[i];
            var bl = new Size((img.Width + BlockSize.Width - 1) / BlockSize.Width, (img.Height + BlockSize.Height - 1) / BlockSize.Height);
            int bw = (img.Width + bl.Width - 1) / bl.Width;
            int bh = (img.Height + bl.Height - 1) / bl.Height;
            perImage[i] = bl;
            for (int by = 0; by < bl.Height; by++)
            {
                for (int bx = 0; bx < bl.Width; bx++)
                {
                    int x0 = bx * bw, y0 = by * bh;
                    int x1 = Math.Min(x0 + bw, img.Width), y1 = Math.Min(y0 + bh, img.Height);
                    blockCorners.Add(new Point(corners[i].X + x0, corners[i].Y + y0));
                    blockImages.Add(img.Roi(x0, y0, x1 - x0, y1 - y0));
                    blockMasks.Add((masks[i].Mask.Roi(x0, y0, x1 - x0, y1 - y0), masks[i].Level));
                }
            }
        }

        if (!UpdateGain && _gainMaps.Length == n)
        {
            return;
        }

        int cn;
        Func<int, int, double> gainOf;
        if (channels)
        {
            var c = new ChannelsCompensator(NrFeeds) { SimilarityThreshold = SimilarityThreshold };
            c.Feed(blockCorners, blockImages, blockMasks);
            Scalar[] g = c.Gains;
            cn = 3;
            gainOf = (b, ch) => ch == 0 ? g[b].V0 : ch == 1 ? g[b].V1 : g[b].V2;
        }
        else
        {
            var c = new GainCompensator(NrFeeds) { SimilarityThreshold = SimilarityThreshold };
            c.Feed(blockCorners, blockImages, blockMasks);
            double[] g = c.Gains;
            cn = 1;
            gainOf = (b, _) => g[b];
        }

        float[] ker = [0.25f, 0.5f, 0.25f];
        _gainMaps = new Mat<float>[n];
        int blIdx = 0;
        for (int i = 0; i < n; i++)
        {
            Size bl = perImage[i];
            var map = new Mat<float>(bl.Width, bl.Height, cn);
            for (int by = 0; by < bl.Height; by++)
            {
                for (int bx = 0; bx < bl.Width; bx++, blIdx++)
                {
                    for (int ch = 0; ch < cn; ch++)
                    {
                        map.At(by, bx, ch) = (float)gainOf(blIdx, ch);
                    }
                }
            }

            for (int it = 0; it < NrGainsFilteringIterations; it++)
            {
                map = Filters.SepFilter2D<float, float>(map, ker, ker);
            }

            _gainMaps[i] = map;
        }
    }
}

/// <summary>Block-wise gain compensator, as OpenCV <c>detail::BlocksGainCompensator</c>.</summary>
public sealed class BlocksGainCompensator : BlocksCompensator
{
    /// <summary>Creates a block-wise gain compensator.</summary>
    /// <param name="blockWidth">Requested block width.</param>
    /// <param name="blockHeight">Requested block height.</param>
    /// <param name="nrFeeds">Number of feed iterations.</param>
    public BlocksGainCompensator(int blockWidth = 32, int blockHeight = 32, int nrFeeds = 1)
        : base(blockWidth, blockHeight, nrFeeds)
    {
    }

    /// <inheritdoc/>
    public override void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks)
        => FeedWithStrategy(corners, images, masks, channels: false);
}

/// <summary>Block-wise channels compensator, as OpenCV <c>detail::BlocksChannelsCompensator</c>.</summary>
public sealed class BlocksChannelsCompensator : BlocksCompensator
{
    /// <summary>Creates a block-wise channels compensator.</summary>
    /// <param name="blockWidth">Requested block width.</param>
    /// <param name="blockHeight">Requested block height.</param>
    /// <param name="nrFeeds">Number of feed iterations.</param>
    public BlocksChannelsCompensator(int blockWidth = 32, int blockHeight = 32, int nrFeeds = 1)
        : base(blockWidth, blockHeight, nrFeeds)
    {
    }

    /// <inheritdoc/>
    public override void Feed(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> images, IReadOnlyList<(Mat<byte> Mask, byte Level)> masks)
        => FeedWithStrategy(corners, images, masks, channels: true);
}
