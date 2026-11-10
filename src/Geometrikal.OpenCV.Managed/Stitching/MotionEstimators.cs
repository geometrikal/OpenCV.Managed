// This implementation is ported from the OpenCV 5.0 stitching motion estimators.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp
//   modules/stitching/src/motion_estimators.cpp (HomographyBasedEstimator,
//   AffineBasedEstimator, CalcRotation, CalcAffineTransform)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>
/// Base class of the camera parameter estimators, as OpenCV <c>detail::Estimator</c>.
/// </summary>
public abstract class Estimator
{
    /// <summary>
    /// Estimates the camera parameters from the pairwise matches, as OpenCV
    /// <c>Estimator::operator()</c>.
    /// </summary>
    /// <param name="features">Features of the n images.</param>
    /// <param name="pairwiseMatches">The n × n pairwise matches.</param>
    /// <param name="cameras">
    /// Camera parameters. An estimator replaces the content; a bundle adjuster reads the initial
    /// parameters and writes the refined parameters.
    /// </param>
    /// <returns>True on success.</returns>
    public bool Estimate(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches, List<CameraParams> cameras)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(pairwiseMatches);
        ArgumentNullException.ThrowIfNull(cameras);
        if (pairwiseMatches.Count != features.Count * features.Count)
        {
            throw new ArgumentException("There must be n × n pairwise matches.", nameof(pairwiseMatches));
        }

        return EstimateCore(features, pairwiseMatches, cameras);
    }

    /// <summary>Runs the estimation (OpenCV <c>estimate</c>).</summary>
    /// <param name="features">Features of the n images.</param>
    /// <param name="pairwiseMatches">The n × n pairwise matches.</param>
    /// <param name="cameras">Camera parameters.</param>
    /// <returns>True on success.</returns>
    protected abstract bool EstimateCore(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches, List<CameraParams> cameras);
}

/// <summary>
/// Rotation estimator from pairwise homographies, as OpenCV <c>detail::HomographyBasedEstimator</c>.
/// </summary>
/// <remarks>
/// When the focal lengths are not given, <see cref="Autocalibration.EstimateFocal"/> sets one
/// focal length for all cameras. The rotations are chained along the maximum spanning tree of
/// the match graph (edge weight: number of inliers) from the tree centre:
/// <c>R_to = R_from · K_from⁻¹ · H⁻¹ · K_to</c>. The rotation of the centre camera is the
/// identity. The principal points are the image centres.
/// </remarks>
public sealed class HomographyBasedEstimator : Estimator
{
    /// <summary>Initializes a new instance.</summary>
    /// <param name="isFocalsEstimated">
    /// True when <c>cameras</c> already holds the focal lengths, aspect ratios and principal
    /// points (relative to the image origin); false to estimate the focal lengths.
    /// </param>
    public HomographyBasedEstimator(bool isFocalsEstimated = false)
    {
        IsFocalsEstimated = isFocalsEstimated;
    }

    /// <summary>Gets a value indicating whether the focal lengths are given.</summary>
    public bool IsFocalsEstimated { get; }

    /// <inheritdoc/>
    protected override bool EstimateCore(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches, List<CameraParams> cameras)
    {
        int n = features.Count;
        if (!IsFocalsEstimated)
        {
            double[] focals = Autocalibration.EstimateFocal(features, pairwiseMatches);
            cameras.Clear();
            for (int i = 0; i < n; i++)
            {
                cameras.Add(new CameraParams { Focal = focals[i] });
            }
        }
        else
        {
            if (cameras.Count != n)
            {
                throw new ArgumentException("There must be one camera per image.", nameof(cameras));
            }

            for (int i = 0; i < n; i++)
            {
                cameras[i].Ppx -= 0.5 * features[i].ImgSize.Width;
                cameras[i].Ppy -= 0.5 * features[i].ImgSize.Height;
            }
        }

        Graph spanTree = MatchesGraph.FindMaxSpanningTree(n, pairwiseMatches, out List<int> centers);
        spanTree.WalkBreadthFirst(centers[0], edge =>
        {
            CameraParams from = cameras[edge.From], to = cameras[edge.To];
            Mat<double> kFrom = SmallMat.K(from.Focal, from.Focal * from.Aspect, from.Ppx, from.Ppy);
            Mat<double> kTo = SmallMat.K(to.Focal, to.Focal * to.Aspect, to.Ppx, to.Ppy);
            Mat<double> h = pairwiseMatches[(edge.From * n) + edge.To].H!;
            Mat<double> r = SmallMat.Mul(SmallMat.Mul(SmallMat.Inv(kFrom), SmallMat.Inv(h)), kTo);
            to.R = SmallMat.Mul(from.R, r);
        });

        for (int i = 0; i < n; i++)
        {
            cameras[i].Ppx += 0.5 * features[i].ImgSize.Width;
            cameras[i].Ppy += 0.5 * features[i].ImgSize.Height;
        }

        return true;
    }
}

/// <summary>
/// Affine transform estimator from pairwise affine transforms, as OpenCV
/// <c>detail::AffineBasedEstimator</c>. The transforms are chained along the maximum spanning
/// tree from the tree centre: <c>R_to = R_from · H</c>. The estimator always succeeds.
/// </summary>
public sealed class AffineBasedEstimator : Estimator
{
    /// <inheritdoc/>
    protected override bool EstimateCore(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches, List<CameraParams> cameras)
    {
        int n = features.Count;
        cameras.Clear();
        for (int i = 0; i < n; i++)
        {
            cameras.Add(new CameraParams());
        }

        Graph spanTree = MatchesGraph.FindMaxSpanningTree(n, pairwiseMatches, out List<int> centers);
        spanTree.WalkBreadthFirst(centers[0], edge =>
        {
            cameras[edge.To].R = SmallMat.Mul(cameras[edge.From].R, pairwiseMatches[(edge.From * n) + edge.To].H!);
        });
        return true;
    }
}
