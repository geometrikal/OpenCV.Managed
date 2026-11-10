namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Runs alone: it clears the process-wide switch that selects the vectorized triangular row sums
/// of the byte stackBlur row pass.
/// </summary>
[Collection(nameof(Geometrikal.OpenCV.Managed.Tests.Core.CvParallelCollection))]
public sealed class StackBlurRowEquivalenceTests
{
    /// <summary>The triangular row sums equal OpenCV's running-sum row pass for every width and radius tried.</summary>
    [Fact]
    public void StackBlur_TriangularRows_MatchRunningSums()
    {
        var random = new Random(3);
        foreach (int cn in new[] { 1, 3, 4 })
        {
            foreach (int radius in new[] { 5, 6, 7, 12, 20, 32 })
            {
                foreach (int w in new[] { (2 * radius) + 2, (2 * radius) + 3, (2 * radius) + 4, (3 * radius) + 3, 37, 64, 129, 333 })
                {
                    if (w < (2 * radius) + 2)
                    {
                        continue;
                    }

                    var src = new Mat<byte>(w, 5, cn);
                    random.NextBytes(src.GetSpan());
                    var k = new Size((2 * radius) + 1, 1);
                    byte[] fast = Filters.StackBlur(src, k).ToArray();
                    try
                    {
                        StackBlurEngine.TriangularRowsEnabled = false;
                        Assert.Equal(Filters.StackBlur(src, k).ToArray(), fast);
                    }
                    finally
                    {
                        StackBlurEngine.TriangularRowsEnabled = true;
                    }
                }
            }
        }
    }
}
