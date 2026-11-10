namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Result of <see cref="ConnectedComponentLabeling.ConnectedComponents{TLabel}(Mat{byte}, int, ConnectedComponentsAlgorithm)"/>.
/// </summary>
/// <param name="LabelCount">Number of labels, including the background label 0.</param>
/// <param name="Labels">Label image of the source size. Values are in <c>[0, LabelCount - 1]</c>.</param>
public sealed record ConnectedComponentsResult<TLabel>(int LabelCount, Mat<TLabel> Labels)
    where TLabel : unmanaged;

/// <summary>
/// Result of <see cref="ConnectedComponentLabeling.ConnectedComponentsWithStats{TLabel}(Mat{byte}, int, ConnectedComponentsAlgorithm)"/>.
/// </summary>
/// <param name="LabelCount">Number of labels, including the background label 0.</param>
/// <param name="Labels">Label image of the source size. Values are in <c>[0, LabelCount - 1]</c>.</param>
/// <param name="Stats">
/// Statistics with 5 columns and <paramref name="LabelCount"/> rows, as OpenCV <c>stats</c>. Row
/// <c>l</c> describes label <c>l</c>; <see cref="ConnectedComponentsStat"/> indexes the columns.
/// A label without pixels (label 0 when the image has no background pixel) has the row
/// <c>[-1, int.MaxValue, 0, 0, 0]</c>.
/// </param>
/// <param name="Centroids">
/// Centroids with 2 columns (x, y) and <paramref name="LabelCount"/> rows, as OpenCV
/// <c>centroids</c>. A label without pixels has the quiet NaN <c>0x7FF8000000000000</c> in both
/// columns.
/// </param>
public sealed record ConnectedComponentsStatsResult<TLabel>(
    int LabelCount, Mat<TLabel> Labels, Mat<int> Stats, Mat<double> Centroids)
    where TLabel : unmanaged
{
    /// <summary>Returns one statistic of one label.</summary>
    public int GetStat(int label, ConnectedComponentsStat stat) => Stats.At(label, (int)stat);

    /// <summary>Returns the bounding box of one label.</summary>
    public Rect GetBoundingBox(int label)
        => new(Stats.At(label, 0), Stats.At(label, 1), Stats.At(label, 2), Stats.At(label, 3));
}
