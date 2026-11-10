// This type is based on the OpenCV stitching pairwise match container.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/matchers.hpp (MatchesInfo)
//   modules/stitching/src/matchers.cpp (MatchesInfo copy semantics)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// The matches and the pairwise transform of an image pair, as OpenCV <c>detail::MatchesInfo</c>.
/// </summary>
/// <remarks>
/// A pairwise match array has <c>n × n</c> elements for <c>n</c> images. Element
/// <c>i·n + j</c> describes the pair (i, j). <see cref="Matches"/> uses
/// <see cref="DMatch.QueryIdx"/> for the keypoints of the source image and
/// <see cref="DMatch.TrainIdx"/> for the keypoints of the destination image.
/// </remarks>
public sealed class MatchesInfo
{
    /// <summary>Gets or sets the source image index (OpenCV <c>src_img_idx</c>). The default is -1.</summary>
    public int SrcImgIdx { get; set; } = -1;

    /// <summary>Gets or sets the destination image index (OpenCV <c>dst_img_idx</c>). The default is -1.</summary>
    public int DstImgIdx { get; set; } = -1;

    /// <summary>Gets or sets the matches.</summary>
    public List<DMatch> Matches { get; set; } = [];

    /// <summary>Gets or sets the geometrically consistent matches mask: one byte per match, 1 for an inlier.</summary>
    public byte[] InliersMask { get; set; } = [];

    /// <summary>Gets or sets the number of geometrically consistent matches.</summary>
    public int NumInliers { get; set; }

    /// <summary>
    /// Gets or sets the 3 × 3 transform from the source image to the destination image, or null
    /// (OpenCV empty <c>H</c>). The homography matchers express it in coordinates relative to the
    /// image centres; the affine matcher uses image coordinates.
    /// </summary>
    public Mat<double>? H { get; set; }

    /// <summary>Gets or sets the confidence that the two images are from the same panorama.</summary>
    public double Confidence { get; set; }

    /// <summary>Returns a deep copy, as the OpenCV copy assignment (<c>H</c> is cloned).</summary>
    public MatchesInfo Clone() => new()
    {
        SrcImgIdx = SrcImgIdx,
        DstImgIdx = DstImgIdx,
        Matches = new List<DMatch>(Matches),
        InliersMask = (byte[])InliersMask.Clone(),
        NumInliers = NumInliers,
        H = H?.Clone(),
        Confidence = Confidence,
    };
}
