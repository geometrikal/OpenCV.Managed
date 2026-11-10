namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System.Collections.Generic;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Analytic behavioural tests for <see cref="Contours.FindContours"/> that do
/// not depend on the golden JSON: empty/degenerate inputs, source immutability,
/// ROI/stride isolation, the offset shift property, determinism, and argument
/// validation.
/// </summary>
public sealed class ContoursBasicTests
{
    private static readonly RetrievalMode[] AllModes =
    {
        RetrievalMode.External,
        RetrievalMode.List,
        RetrievalMode.ConnectedComponents,
        RetrievalMode.Tree,
    };

    private static readonly ContourApproximationMode[] AllMethods =
    {
        ContourApproximationMode.None,
        ContourApproximationMode.Simple,
        ContourApproximationMode.Tc89L1,
        ContourApproximationMode.Tc89KCos,
    };

    // -- Empty input --------------------------------------------------------

    [Theory]
    [InlineData(RetrievalMode.External)]
    [InlineData(RetrievalMode.List)]
    [InlineData(RetrievalMode.ConnectedComponents)]
    [InlineData(RetrievalMode.Tree)]
    public void FindContours_AllZero_ReturnsEmptyResult(RetrievalMode mode)
    {
        var image = new Mat<byte>(8, 8);

        ContourResult result = Contours.FindContours(image, mode, ContourApproximationMode.Simple);

        Assert.Empty(result.Contours);
        Assert.Empty(result.Hierarchy);
        Assert.Equal(result.Contours.Count, result.Hierarchy.Count);
    }

    // -- Single pixel -------------------------------------------------------

    [Theory]
    [InlineData(RetrievalMode.External, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.External, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.List, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.List, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.ConnectedComponents, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.ConnectedComponents, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.Tree, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.Tree, ContourApproximationMode.Simple)]
    public void FindContours_SingleCenterPixel_ReturnsOnePointRootContour(
        RetrievalMode mode,
        ContourApproximationMode method)
    {
        var image = new Mat<byte>(5, 5);
        image.At(2, 2) = 255;

        ContourResult result = Contours.FindContours(image, mode, method);

        Point[] contour = Assert.Single(result.Contours);
        Point single = Assert.Single(contour);
        Assert.Equal(new Point(2, 2), single);

        ContourHierarchy h = Assert.Single(result.Hierarchy);
        Assert.Equal(new ContourHierarchy(-1, -1, -1, -1), h);
    }

    // -- Solid 3x3 external border sequence (analytic; cv2-verified) --------

    [Fact]
    public void FindContours_Solid3x3_External_None_TracesBorderSequence()
    {
        // Solid 3x3 block with top-left at (1,1) inside a 5x5 image.
        var image = new Mat<byte>(5, 5);
        for (int y = 1; y <= 3; y++)
        {
            for (int x = 1; x <= 3; x++)
            {
                image.At(y, x) = 255;
            }
        }

        ContourResult result = Contours.FindContours(
            image, RetrievalMode.External, ContourApproximationMode.None);

        Point[] contour = Assert.Single(result.Contours);

        // Suzuki-Abe outer border, counter-clockwise in image (y-down) coords.
        // This exact ordering is reproduced by the cv2 4.5.5 golden generator
        // (see tools/generate_golden.py::sanity_checks solid_3x3).
        Point[] expected =
        {
            new(1, 1), new(1, 2), new(1, 3), new(2, 3),
            new(3, 3), new(3, 2), new(3, 1), new(2, 1),
        };

        Assert.Equal(expected, contour);
    }

    // -- Source immutability ------------------------------------------------

    [Fact]
    public void FindContours_DoesNotModifySource()
    {
        Mat<byte> image = BuildPatternImage();
        byte[] before = image.ToArray();

        foreach (RetrievalMode mode in AllModes)
        {
            foreach (ContourApproximationMode method in AllMethods)
            {
                _ = Contours.FindContours(image, mode, method);
                byte[] after = image.ToArray();
                Assert.True(
                    AreEqual(before, after),
                    $"Source buffer was modified by mode={mode}, method={method}.");
            }
        }
    }

    // -- ROI / stride isolation (MANDATORY) ---------------------------------

    [Fact]
    public void FindContours_RoiWithStride_MatchesPackedImage()
    {
        // 20x20 parent, ROI(3,4,10,8) carrying a hollow rectangle.
        var parent = new Mat<byte>(20, 20);
        Mat<byte> roi = parent.Roi(3, 4, 10, 8);
        FillHollowRectOutline(roi);

        // Identical packed 10x8 image (continuous, stride == width).
        var packed = new Mat<byte>(10, 8);
        FillHollowRectOutline(packed);

        Assert.False(roi.IsContinuous, "ROI into a wider parent must be non-continuous.");
        Assert.True(packed.IsContinuous);

        foreach (RetrievalMode mode in AllModes)
        {
            foreach (ContourApproximationMode method in AllMethods)
            {
                ContourResult fromRoi = Contours.FindContours(roi, mode, method);
                ContourResult fromPacked = Contours.FindContours(packed, mode, method);
                AssertResultsEqual(fromPacked, fromRoi);
            }
        }
    }

    [Fact]
    public void FindContours_RoiSurroundedByNoise_IsIsolatedFromParent()
    {
        // Parent is entirely foreground (worst-case neighbour "noise"); the ROI
        // window is cleared and given a clean hollow rectangle. The result must
        // match the standalone packed image, proving row-span isolation.
        var parent = new Mat<byte>(20, 20);
        parent.Fill(255);

        Mat<byte> roi = parent.Roi(3, 4, 10, 8);
        roi.Clear();
        FillHollowRectOutline(roi);

        var packed = new Mat<byte>(10, 8);
        FillHollowRectOutline(packed);

        foreach (RetrievalMode mode in AllModes)
        {
            foreach (ContourApproximationMode method in AllMethods)
            {
                ContourResult fromRoi = Contours.FindContours(roi, mode, method);
                ContourResult fromPacked = Contours.FindContours(packed, mode, method);
                AssertResultsEqual(fromPacked, fromRoi);
            }
        }
    }

    // -- Offset shift property ----------------------------------------------

    [Fact]
    public void FindContours_OffsetShiftsPoints_HierarchyUnchanged()
    {
        Mat<byte> image = BuildNestedRingsImage();
        var offset = new Point(5, -3);

        ContourResult baseline = Contours.FindContours(
            image, RetrievalMode.Tree, ContourApproximationMode.Simple);
        ContourResult shifted = Contours.FindContours(
            image, RetrievalMode.Tree, ContourApproximationMode.Simple, offset);

        Assert.Equal(baseline.Contours.Count, shifted.Contours.Count);
        Assert.True(baseline.Contours.Count > 1, "Nested rings should produce several contours.");

        for (int i = 0; i < baseline.Contours.Count; i++)
        {
            Point[] b = baseline.Contours[i];
            Point[] s = shifted.Contours[i];
            Assert.Equal(b.Length, s.Length);
            for (int j = 0; j < b.Length; j++)
            {
                Assert.Equal(b[j] + offset, s[j]);
            }

            // Hierarchy is a topology property; it must not change with offset.
            Assert.Equal(baseline.Hierarchy[i], shifted.Hierarchy[i]);
        }
    }

    // -- Determinism --------------------------------------------------------

    [Fact]
    public void FindContours_RepeatedRuns_AreDeterministic()
    {
        Mat<byte> image = BuildPatternImage();

        foreach (RetrievalMode mode in AllModes)
        {
            foreach (ContourApproximationMode method in AllMethods)
            {
                ContourResult first = Contours.FindContours(image, mode, method);
                ContourResult second = Contours.FindContours(image, mode, method);
                AssertResultsEqual(first, second);
            }
        }
    }

    // -- Validation ---------------------------------------------------------

    [Fact]
    public void FindContours_NullImage_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            Contours.FindContours(
                null!, RetrievalMode.Tree, ContourApproximationMode.Simple));
    }

    [Fact]
    public void FindContours_MultiChannelImage_Throws()
    {
        var image = new Mat<byte>(4, 4, channels: 3);

        Assert.Throws<ArgumentException>(() =>
            Contours.FindContours(image, RetrievalMode.Tree, ContourApproximationMode.Simple));
    }

    [Fact]
    public void FindContours_InvalidMode_Throws()
    {
        var image = new Mat<byte>(4, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Contours.FindContours(image, (RetrievalMode)99, ContourApproximationMode.Simple));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    public void FindContours_InvalidMethod_Throws(int method)
    {
        var image = new Mat<byte>(4, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Contours.FindContours(image, RetrievalMode.Tree, (ContourApproximationMode)method));
    }

    // -- Helpers ------------------------------------------------------------

    private static void AssertResultsEqual(ContourResult expected, ContourResult actual)
    {
        Assert.Equal(expected.Contours.Count, actual.Contours.Count);
        Assert.Equal(expected.Hierarchy.Count, actual.Hierarchy.Count);

        for (int i = 0; i < expected.Contours.Count; i++)
        {
            Assert.Equal(expected.Contours[i], actual.Contours[i]);
        }

        for (int i = 0; i < expected.Hierarchy.Count; i++)
        {
            Assert.Equal(expected.Hierarchy[i], actual.Hierarchy[i]);
        }
    }

    private static bool AreEqual(byte[] a, byte[] b)
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

    private static void FillHollowRectOutline(Mat<byte> m)
    {
        int w = m.Width;
        int h = m.Height;

        for (int x = 0; x < w; x++)
        {
            m.At(0, x) = 255;
            m.At(h - 1, x) = 255;
        }

        for (int y = 0; y < h; y++)
        {
            m.At(y, 0) = 255;
            m.At(y, w - 1) = 255;
        }
    }

    private static Mat<byte> BuildPatternImage()
    {
        // A hollow rectangle, an interior solid blob (creating a hole), and a
        // couple of stray pixels: exercises external + hole + isolated contours.
        var m = new Mat<byte>(14, 12);

        for (int x = 1; x <= 10; x++)
        {
            m.At(1, x) = 255;
            m.At(9, x) = 255;
        }

        for (int y = 1; y <= 9; y++)
        {
            m.At(y, 1) = 255;
            m.At(y, 10) = 255;
        }

        for (int y = 4; y <= 6; y++)
        {
            for (int x = 4; x <= 7; x++)
            {
                m.At(y, x) = 255;
            }
        }

        m.At(11, 12) = 255;
        m.At(11, 2) = 255;

        return m;
    }

    private static Mat<byte> BuildNestedRingsImage()
    {
        // Three nested 1px hollow squares in a 16x16 image (a parent-child chain
        // under RETR_TREE), matching the "triple_hollow" golden shape.
        var m = new Mat<byte>(16, 16);
        DrawSquareOutline(m, 1, 1, 14, 14);
        DrawSquareOutline(m, 4, 4, 11, 11);
        DrawSquareOutline(m, 6, 6, 9, 9);
        return m;
    }

    private static void DrawSquareOutline(Mat<byte> m, int x0, int y0, int x1, int y1)
    {
        for (int x = x0; x <= x1; x++)
        {
            m.At(y0, x) = 255;
            m.At(y1, x) = 255;
        }

        for (int y = y0; y <= y1; y++)
        {
            m.At(y, x0) = 255;
            m.At(y, x1) = 255;
        }
    }
}
