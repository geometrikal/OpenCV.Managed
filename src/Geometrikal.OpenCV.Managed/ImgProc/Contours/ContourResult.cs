namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// The result of a contour-detection run: the detected contours together with
/// their hierarchy, mirroring the pair of outputs returned by OpenCV's
/// <c>cv::findContours</c>.
///
/// <see cref="Contours"/> and <see cref="Hierarchy"/> are parallel lists:
/// <c>Contours.Count == Hierarchy.Count</c> always holds, and the hierarchy
/// entry at index <c>i</c> describes the contour at index <c>i</c>.
/// </summary>
/// <param name="Contours">
/// The detected contours in depth-first tree order (newest sibling first),
/// each an array of pixel coordinates of its border points.
/// </param>
/// <param name="Hierarchy">
/// Per-contour hierarchy relationships, one entry per contour, using <c>-1</c>
/// for absent relationships.
/// </param>
public sealed record ContourResult(
    IReadOnlyList<Point[]> Contours,
    IReadOnlyList<ContourHierarchy> Hierarchy);
