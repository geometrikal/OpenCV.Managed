namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System.Collections.Generic;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Adjudicated coverage for the two golden cases the data-driven
/// <see cref="ContoursGoldenTests"/> runner excludes as "suspect", so the
/// OpenCV 5-vs-legacy TC89_L1 difference stays fully tested rather than dropped.
///
/// The golden data was captured with cv2 4.5.5 (the legacy Teh-Chin
/// implementation), whereas this library ports OpenCV 5's rewritten
/// <c>contours_approx.cpp</c>. For exactly one contour on this 64x64 image the
/// two disagree: OpenCV 5's <c>calc_new_start</c> "two points left" branch
/// (<c>contours_approx.cpp</c> lines 186-223) inspects only the immediate
/// neighbours of the chain wrap point and appends a copy of <c>ares[0]</c>
/// without removing the original, so <c>pass_cleanup</c> starts past index 0 and
/// the origin survives duplicated at both ends. Legacy cv2 emits the 5-point
/// sequence; OpenCV 5 emits the 6-point sequence, and this port matches OpenCV 5
/// by design (see <c>ImgProc/Contours/Internal/ChainApproximatorTc89.cs</c>).
/// </summary>
public sealed class ContoursOpenCv5DifferenceTests
{
    // The one differing contour, as legacy cv2 4.5.5 emits it (5 points).
    private static readonly Point[] LegacySequence =
    {
        new(39, 56), new(39, 59), new(37, 62), new(36, 60), new(36, 56),
    };

    // The same contour as OpenCV 5 (and this port) emits it: the origin
    // (36,56) is duplicated at both ends -> 6 points.
    private static readonly Point[] OpenCv5Sequence =
    {
        new(36, 56), new(39, 56), new(39, 59), new(37, 62), new(36, 60), new(36, 56),
    };

    [Theory]
    [InlineData("rand_64_p50_list_tc89l1")]
    [InlineData("rand_64_p50_tree_tc89l1")]
    public void SuspectCase_DiffersFromLegacyGolden_AtExactlyOneContour(string caseName)
    {
        ContoursGoldenTests.GoldenCase golden = ContoursGoldenTests.AllCases[caseName];
        Assert.True(
            golden.Suspect,
            $"Case '{caseName}' must be flagged suspect in the golden data.");

        Mat<byte> image = ContoursGoldenTests.BuildImage(golden);
        var mode = (RetrievalMode)golden.Mode;
        var method = (ContourApproximationMode)golden.Method;
        var offset = new Point(golden.OffsetX, golden.OffsetY);

        ContourResult result = Contours.FindContours(image, mode, method, offset);

        // Count and hierarchy are unaffected by the point-level difference.
        Assert.Equal(golden.Contours.Count, result.Contours.Count);
        Assert.Equal(golden.Hierarchy.Count, result.Hierarchy.Count);
        for (int i = 0; i < golden.Hierarchy.Count; i++)
        {
            List<int> eh = golden.Hierarchy[i];
            Assert.Equal(
                new ContourHierarchy(eh[0], eh[1], eh[2], eh[3]),
                result.Hierarchy[i]);
        }

        // Locate every contour index whose points differ from the legacy golden.
        var differing = new List<int>();
        for (int i = 0; i < golden.Contours.Count; i++)
        {
            if (!PointsEqual(ToPoints(golden.Contours[i]), result.Contours[i]))
            {
                differing.Add(i);
            }
        }

        // Exactly ONE contour may differ; more than one is unadjudicated and
        // must fail loudly rather than be tolerated.
        int idx = Assert.Single(differing);

        // At that index golden holds the legacy 5-point sequence and the port
        // reproduces OpenCV 5's 6-point (origin-duplicated) sequence.
        Assert.Equal(LegacySequence, ToPoints(golden.Contours[idx]));
        Assert.Equal(OpenCv5Sequence, result.Contours[idx]);

        // Every other contour is point-for-point identical to golden.
        for (int i = 0; i < golden.Contours.Count; i++)
        {
            if (i == idx)
            {
                continue;
            }

            Assert.Equal(ToPoints(golden.Contours[i]), result.Contours[i]);
        }
    }

    private static Point[] ToPoints(List<List<int>> pts)
    {
        var arr = new Point[pts.Count];
        for (int i = 0; i < pts.Count; i++)
        {
            arr[i] = new Point(pts[i][0], pts[i][1]);
        }

        return arr;
    }

    private static bool PointsEqual(Point[] a, Point[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}
