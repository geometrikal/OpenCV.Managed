namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Result of <see cref="DistanceTransforms.DistanceTransformWithLabels(Mat{byte}, DistanceTypes, DistanceTransformMasks, DistanceTransformLabelTypes)"/>.
/// </summary>
/// <param name="Distances">Distance to the nearest zero pixel, of the source size.</param>
/// <param name="Labels">Label of the nearest zero pixel or zero-pixel component, of the source size.</param>
public sealed record DistanceTransformResult(Mat<float> Distances, Mat<int> Labels);
