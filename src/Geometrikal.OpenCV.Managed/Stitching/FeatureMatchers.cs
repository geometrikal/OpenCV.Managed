// This implementation is ported from the OpenCV 5.0 stitching feature matchers.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/matchers.hpp
//   modules/stitching/src/matchers.cpp (MatchPairsBody, CpuMatcher, FeaturesMatcher::match,
//   BestOf2NearestMatcher, BestOf2NearestRangeMatcher, AffineBestOf2NearestMatcher)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Base class of the pairwise feature matchers, as OpenCV <c>detail::FeaturesMatcher</c>.
/// </summary>
public abstract class FeaturesMatcher
{
    /// <summary>Initializes a new instance.</summary>
    /// <param name="isThreadSafe">True when <see cref="Match(ImageFeatures, ImageFeatures, MatchesInfo)"/> can run for several pairs at the same time.</param>
    protected FeaturesMatcher(bool isThreadSafe)
    {
        IsThreadSafe = isThreadSafe;
    }

    /// <summary>Gets a value indicating whether the pairs are matched in parallel.</summary>
    public bool IsThreadSafe { get; }

    /// <summary>
    /// Matches the features of two images, as OpenCV <c>FeaturesMatcher::operator()(features1,
    /// features2, matches_info)</c>.
    /// </summary>
    /// <param name="features1">Features of the source image.</param>
    /// <param name="features2">Features of the destination image.</param>
    /// <param name="matchesInfo">Result. The matcher overwrites the matches and sets the other fields it computes.</param>
    public void Match(ImageFeatures features1, ImageFeatures features2, MatchesInfo matchesInfo)
    {
        ArgumentNullException.ThrowIfNull(features1);
        ArgumentNullException.ThrowIfNull(features2);
        ArgumentNullException.ThrowIfNull(matchesInfo);
        MatchPair(features1, features2, matchesInfo);
    }

    /// <summary>
    /// Matches all image pairs (i, j) with i &lt; j, as OpenCV <c>FeaturesMatcher::operator()(features,
    /// pairwise_matches, mask)</c>.
    /// </summary>
    /// <param name="features">Features of all images.</param>
    /// <param name="mask">
    /// Optional n × n mask. A pair (i, j) with i &lt; j is matched only when <c>mask(i, j)</c> is not 0.
    /// </param>
    /// <returns>
    /// The n × n pairwise matches. Element <c>i·n + j</c> holds the pair (i, j). Element
    /// <c>j·n + i</c> holds the same matches with the query and train indices swapped and the
    /// inverse transform. Pairs that are not matched (including i = j) keep the default values.
    /// </returns>
    public MatchesInfo[] Match(IReadOnlyList<ImageFeatures> features, Mat<byte>? mask = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        int n = features.Count;
        return MatchPairs(features, NearPairs(features, mask, n), n);
    }

    /// <summary>Releases internal buffers, as OpenCV <c>collectGarbage</c>. The CPU matchers keep no buffers.</summary>
    public virtual void CollectGarbage()
    {
    }

    /// <summary>Matches one pair (OpenCV <c>match</c>).</summary>
    /// <param name="features1">Features of the source image.</param>
    /// <param name="features2">Features of the destination image.</param>
    /// <param name="matchesInfo">Result.</param>
    protected abstract void MatchPair(ImageFeatures features1, ImageFeatures features2, MatchesInfo matchesInfo);

    /// <summary>Lists the pairs (i, j), i &lt; j, to match.</summary>
    /// <param name="features">Features of all images.</param>
    /// <param name="mask">Optional n × n mask.</param>
    /// <param name="n">Number of images.</param>
    /// <returns>The pairs in OpenCV order.</returns>
    private protected virtual List<(int From, int To)> NearPairs(IReadOnlyList<ImageFeatures> features, Mat<byte>? mask, int n)
        => NearPairs(features, mask, n, int.MaxValue);

    private protected static List<(int From, int To)> NearPairs(IReadOnlyList<ImageFeatures> features, Mat<byte>? mask, int n, int rangeWidth)
    {
        if (mask is not null && (mask.Width != n || mask.Height < 1 || mask.Channels != 1))
        {
            throw new ArgumentException("The mask must be n × n with one channel.", nameof(mask));
        }

        var pairs = new List<(int, int)>();
        for (int i = 0; i < n - 1; i++)
        {
            int end = (int)Math.Min(n, (long)i + 1 + rangeWidth);
            for (int j = i + 1; j < end; j++)
            {
                if (features[i].Keypoints.Length > 0 && features[j].Keypoints.Length > 0
                    && (mask is null || mask.At(i, j) != 0))
                {
                    pairs.Add((i, j));
                }
            }
        }

        return pairs;
    }

    private MatchesInfo[] MatchPairs(IReadOnlyList<ImageFeatures> features, List<(int From, int To)> pairs, int n)
    {
        var result = new MatchesInfo[n * n];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = new MatchesInfo();
        }

        void Body(int p)
        {
            (int from, int to) = pairs[p];
            MatchesInfo info = result[(from * n) + to];
            MatchPair(features[from], features[to], info);
            info.SrcImgIdx = from;
            info.DstImgIdx = to;

            MatchesInfo dual = info.Clone();
            dual.SrcImgIdx = to;
            dual.DstImgIdx = from;
            if (info.H is not null)
            {
                LinearAlgebra.Invert(info.H, out Mat<double> inv);
                dual.H = inv;
            }

            for (int k = 0; k < dual.Matches.Count; k++)
            {
                DMatch m = dual.Matches[k];
                dual.Matches[k] = new DMatch(m.TrainIdx, m.QueryIdx, m.ImgIdx, m.Distance);
            }

            result[(to * n) + from] = dual;
        }

        if (IsThreadSafe && pairs.Count > 1)
        {
            ParallelRows.ForEachSerial(pairs.Count, Body);
        }
        else
        {
            for (int p = 0; p < pairs.Count; p++)
            {
                Body(p);
            }
        }

        return result;
    }
}

/// <summary>
/// Matcher that keeps the best of the two nearest neighbours of each feature and estimates a
/// homography for each pair, as OpenCV <c>detail::BestOf2NearestMatcher</c>.
/// </summary>
/// <remarks>
/// <para>
/// The nearest-neighbour search is exhaustive: <see cref="BFMatcher"/> with
/// <see cref="NormType.Hamming"/> for <c>byte</c> descriptors and <see cref="NormType.L2"/> for
/// <c>float</c> descriptors. OpenCV uses FLANN (LSH for <c>byte</c>, randomized k-d trees for
/// <c>float</c>), which is approximate. Both give the same matches when FLANN finds the exact
/// two nearest neighbours; the port can keep matches that FLANN misses.
/// </para>
/// <para>
/// A match is kept when <c>d₁ &lt; (1 - matchConf)·d₂</c> in the 1→2 search, or in the 2→1
/// search when the 1→2 search did not give the same pair. The homography uses keypoint
/// coordinates relative to the image centres and RANSAC (threshold 3). The confidence is
/// <c>numInliers / (8 + 0.3·numMatches)</c>, set to 0 when it is above
/// <c>matchesConfidenceThresh</c>. With at least <c>numMatchesThresh2</c> inliers the homography
/// is estimated again from the inliers only.
/// </para>
/// </remarks>
public class BestOf2NearestMatcher : FeaturesMatcher
{
    /// <summary>Initializes a new instance with the OpenCV defaults.</summary>
    /// <param name="matchConf">Match distance ratio margin: a match is kept when d₁ &lt; (1 - matchConf)·d₂.</param>
    /// <param name="numMatchesThresh1">Minimum number of matches to estimate the homography.</param>
    /// <param name="numMatchesThresh2">Minimum number of inliers to estimate the homography again from the inliers.</param>
    /// <param name="matchesConfidenceThresh">Confidence above which the pair is treated as too close (confidence 0).</param>
    public BestOf2NearestMatcher(float matchConf = 0.3f, int numMatchesThresh1 = 6, int numMatchesThresh2 = 6, double matchesConfidenceThresh = 3.0)
        : base(true)
    {
        MatchConf = matchConf;
        NumMatchesThresh1 = numMatchesThresh1;
        NumMatchesThresh2 = numMatchesThresh2;
        MatchesConfidenceThresh = matchesConfidenceThresh;
    }

    /// <summary>Gets the match distance ratio margin.</summary>
    public float MatchConf { get; }

    /// <summary>Gets the minimum number of matches to estimate a transform.</summary>
    public int NumMatchesThresh1 { get; }

    /// <summary>Gets the minimum number of inliers to estimate the transform again.</summary>
    public int NumMatchesThresh2 { get; }

    /// <summary>Gets the confidence above which a pair is treated as too close.</summary>
    public double MatchesConfidenceThresh { get; }

    /// <inheritdoc/>
    protected override void MatchPair(ImageFeatures features1, ImageFeatures features2, MatchesInfo matchesInfo)
    {
        CpuMatch(features1, features2, MatchConf, matchesInfo);
        List<DMatch> matches = matchesInfo.Matches;
        if (matches.Count < NumMatchesThresh1)
        {
            return;
        }

        int count = matches.Count;
        Point2f[] src = ArrayPool<Point2f>.Shared.Rent(count);
        Point2f[] dst = ArrayPool<Point2f>.Shared.Rent(count);
        try
        {
            for (int i = 0; i < count; i++)
            {
                src[i] = Centered(features1, matches[i].QueryIdx);
                dst[i] = Centered(features2, matches[i].TrainIdx);
            }

            EstimateHomography(src.AsSpan(0, count), dst.AsSpan(0, count), matchesInfo);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(src);
            ArrayPool<Point2f>.Shared.Return(dst);
        }
    }

    private void EstimateHomography(ReadOnlySpan<Point2f> src, ReadOnlySpan<Point2f> dst, MatchesInfo matchesInfo)
    {
        GeometricEstimationResult first = Homography(src, dst);
        matchesInfo.H = first.Model;
        matchesInfo.InliersMask = first.InlierMask;
        if (first.Model is null || Math.Abs(LinearAlgebra.Determinant(first.Model)) < DblEpsilon)
        {
            return;
        }

        int numInliers = 0;
        foreach (byte b in first.InlierMask)
        {
            if (b != 0)
            {
                numInliers++;
            }
        }

        matchesInfo.NumInliers = numInliers;
        double confidence = numInliers / (8 + (0.3 * src.Length));
        matchesInfo.Confidence = confidence > MatchesConfidenceThresh ? 0.0 : confidence;
        if (numInliers < NumMatchesThresh2)
        {
            return;
        }

        Point2f[] srcIn = ArrayPool<Point2f>.Shared.Rent(numInliers);
        Point2f[] dstIn = ArrayPool<Point2f>.Shared.Rent(numInliers);
        try
        {
            int idx = 0;
            for (int i = 0; i < src.Length; i++)
            {
                if (first.InlierMask[i] == 0)
                {
                    continue;
                }

                srcIn[idx] = src[i];
                dstIn[idx] = dst[i];
                idx++;
            }

            matchesInfo.H = Homography(srcIn.AsSpan(0, numInliers), dstIn.AsSpan(0, numInliers)).Model;
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(srcIn);
            ArrayPool<Point2f>.Shared.Return(dstIn);
        }
    }

    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>OpenCV <c>CpuMatcher::match</c> with exhaustive search.</summary>
    internal static void CpuMatch(ImageFeatures features1, ImageFeatures features2, float matchConf, MatchesInfo matchesInfo)
    {
        if (features1.HasFloatDescriptors != features2.HasFloatDescriptors)
        {
            throw new ArgumentException("The descriptor types of the two images differ.");
        }

        matchesInfo.Matches = [];
        DMatch[][] forward, backward;
        if (features1.HasFloatDescriptors)
        {
            Mat<float>? d1 = features1.FloatDescriptors, d2 = features2.FloatDescriptors;
            if (d1 is null || d2 is null)
            {
                return;
            }

            var bf = new BFMatcher(NormType.L2);
            forward = bf.KnnMatch(d1, d2, 2);
            backward = bf.KnnMatch(d2, d1, 2);
        }
        else
        {
            Mat<byte>? d1 = features1.ByteDescriptors, d2 = features2.ByteDescriptors;
            if (d1 is null || d2 is null)
            {
                return;
            }

            var bf = new BFMatcher(NormType.Hamming);
            forward = bf.KnnMatch(d1, d2, 2);
            backward = bf.KnnMatch(d2, d1, 2);
        }

        float ratio = 1f - matchConf;
        var set = new HashSet<long>();
        List<DMatch> result = matchesInfo.Matches;
        foreach (DMatch[] pair in forward)
        {
            if (pair.Length < 2)
            {
                continue;
            }

            DMatch m0 = pair[0], m1 = pair[1];
            if (m0.Distance < ratio * m1.Distance)
            {
                result.Add(m0);
                set.Add(((long)m0.QueryIdx << 32) | (uint)m0.TrainIdx);
            }
        }

        foreach (DMatch[] pair in backward)
        {
            if (pair.Length < 2)
            {
                continue;
            }

            DMatch m0 = pair[0], m1 = pair[1];
            if (m0.Distance < ratio * m1.Distance && !set.Contains(((long)m0.TrainIdx << 32) | (uint)m0.QueryIdx))
            {
                result.Add(new DMatch(m0.TrainIdx, m0.QueryIdx, m0.Distance));
            }
        }
    }

    private static Point2f Centered(ImageFeatures features, int index)
    {
        Point2f p = features.Keypoints[index].Pt;
        return new Point2f(p.X - (features.ImgSize.Width * 0.5f), p.Y - (features.ImgSize.Height * 0.5f));
    }

    private static GeometricEstimationResult Homography(ReadOnlySpan<Point2f> src, ReadOnlySpan<Point2f> dst)
    {
        if (src.Length < 4)
        {
            return new GeometricEstimationResult(null, new byte[src.Length]);
        }

        return GeometricEstimation.FindHomography(src, dst, RobustEstimationMethod.Ransac);
    }
}

/// <summary>
/// <see cref="BestOf2NearestMatcher"/> that matches each image only with the next
/// <see cref="RangeWidth"/> - 1 images, as OpenCV <c>detail::BestOf2NearestRangeMatcher</c>.
/// Use it for ordered sequences, for example consecutive drone frames.
/// </summary>
public sealed class BestOf2NearestRangeMatcher : BestOf2NearestMatcher
{
    /// <summary>Initializes a new instance with the OpenCV defaults.</summary>
    /// <param name="rangeWidth">Image i is matched with the images i + 1 to i + rangeWidth - 1.</param>
    /// <param name="matchConf">Match distance ratio margin.</param>
    /// <param name="numMatchesThresh1">Minimum number of matches to estimate the homography.</param>
    /// <param name="numMatchesThresh2">Minimum number of inliers to estimate the homography again.</param>
    public BestOf2NearestRangeMatcher(int rangeWidth = 5, float matchConf = 0.3f, int numMatchesThresh1 = 6, int numMatchesThresh2 = 6)
        : base(matchConf, numMatchesThresh1, numMatchesThresh2)
    {
        RangeWidth = rangeWidth;
    }

    /// <summary>Gets the range width.</summary>
    public int RangeWidth { get; }

    private protected override List<(int From, int To)> NearPairs(IReadOnlyList<ImageFeatures> features, Mat<byte>? mask, int n)
        => NearPairs(features, mask, n, RangeWidth);
}

/// <summary>
/// Matcher that estimates an affine transform for each pair, as OpenCV
/// <c>detail::AffineBestOf2NearestMatcher</c>. The <see cref="Stitcher"/> SCANS mode uses it.
/// </summary>
/// <remarks>
/// The matches are found as in <see cref="BestOf2NearestMatcher"/>. The transform uses image
/// coordinates and <see cref="GeometricEstimation.EstimateAffine2D(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double, int)"/>
/// (full affine) or <see cref="GeometricEstimation.EstimateAffinePartial2D(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double, int)"/>
/// with the default settings. <see cref="MatchesInfo.H"/> is the 2 × 3 model extended with the
/// row 0 0 1. The confidence is not limited by a threshold.
/// </remarks>
public sealed class AffineBestOf2NearestMatcher : BestOf2NearestMatcher
{
    /// <summary>Initializes a new instance with the OpenCV defaults.</summary>
    /// <param name="fullAffine">True for 6 degrees of freedom; false for rotation, uniform scale and translation.</param>
    /// <param name="matchConf">Match distance ratio margin.</param>
    /// <param name="numMatchesThresh1">Minimum number of matches to estimate the transform.</param>
    public AffineBestOf2NearestMatcher(bool fullAffine = false, float matchConf = 0.3f, int numMatchesThresh1 = 6)
        : base(matchConf, numMatchesThresh1, numMatchesThresh1)
    {
        FullAffine = fullAffine;
    }

    /// <summary>Gets a value indicating whether the full affine model is estimated.</summary>
    public bool FullAffine { get; }

    /// <inheritdoc/>
    protected override void MatchPair(ImageFeatures features1, ImageFeatures features2, MatchesInfo matchesInfo)
    {
        CpuMatch(features1, features2, MatchConf, matchesInfo);
        List<DMatch> matches = matchesInfo.Matches;
        if (matches.Count < NumMatchesThresh1)
        {
            return;
        }

        int count = matches.Count;
        Point2f[] src = ArrayPool<Point2f>.Shared.Rent(count);
        Point2f[] dst = ArrayPool<Point2f>.Shared.Rent(count);
        GeometricEstimationResult r;
        try
        {
            for (int i = 0; i < count; i++)
            {
                src[i] = features1.Keypoints[matches[i].QueryIdx].Pt;
                dst[i] = features2.Keypoints[matches[i].TrainIdx].Pt;
            }

            r = FullAffine
                ? GeometricEstimation.EstimateAffine2D(src.AsSpan(0, count), dst.AsSpan(0, count))
                : GeometricEstimation.EstimateAffinePartial2D(src.AsSpan(0, count), dst.AsSpan(0, count));
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(src);
            ArrayPool<Point2f>.Shared.Return(dst);
        }

        matchesInfo.InliersMask = r.InlierMask;
        if (r.Model is null)
        {
            matchesInfo.H = null;
            matchesInfo.Confidence = 0;
            matchesInfo.NumInliers = 0;
            return;
        }

        int numInliers = 0;
        foreach (byte b in r.InlierMask)
        {
            if (b != 0)
            {
                numInliers++;
            }
        }

        matchesInfo.NumInliers = numInliers;
        matchesInfo.Confidence = numInliers / (8 + (0.3 * matches.Count));

        var h = new Mat<double>(3, 3);
        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                h.At(y, x) = r.Model.At(y, x);
            }
        }

        h.At(2, 2) = 1;
        matchesInfo.H = h;
    }
}
