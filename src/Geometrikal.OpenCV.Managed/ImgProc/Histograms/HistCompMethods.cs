namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Histogram comparison method for <see cref="Histograms.CompareHist(DenseHistogram, DenseHistogram, HistCompMethods)"/>,
/// mirroring OpenCV <c>HistCompMethods</c> (same numeric values).
/// </summary>
public enum HistCompMethods
{
    /// <summary>Correlation.</summary>
    Correl = 0,

    /// <summary>Chi-square: <c>sum((h1 - h2)² / h1)</c>.</summary>
    Chisqr = 1,

    /// <summary>Intersection: <c>sum(min(h1, h2))</c>.</summary>
    Intersect = 2,

    /// <summary>Bhattacharyya distance (normalised, also known as Hellinger distance).</summary>
    Bhattacharyya = 3,

    /// <summary>Synonym for <see cref="Bhattacharyya"/>.</summary>
    Hellinger = 3,

    /// <summary>Alternative chi-square: <c>2 * sum((h1 - h2)² / (h1 + h2))</c>.</summary>
    ChisqrAlt = 4,

    /// <summary>Kullback-Leibler divergence: <c>sum(h1 * log(h1 / h2))</c>.</summary>
    KLDiv = 5,
}
