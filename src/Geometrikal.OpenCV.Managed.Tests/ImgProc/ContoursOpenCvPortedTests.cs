namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System.Collections.Generic;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Managed ports of the analytic assertions from the OpenCV upstream test
/// <c>modules/imgproc/test/test_contours_new.cpp</c> (the CV_8U paths of the
/// <c>rectangle</c> and <c>small</c> parameterised tests).
///
/// Only the analytic branches are ported: the <c>rectangle</c> test asserts
/// contour counts + <c>CHAIN_APPROX_NONE</c> perimeter lengths and exact
/// <c>CHAIN_APPROX_SIMPLE</c> corner sequences (upstream leaves TC89 unchecked
/// here); the <c>small</c> test asserts one contour per isolated blob for every
/// mode and method. The <c>RETR_FLOODFILL</c> / CV_32S branches, the chain-code
/// (<c>method == 0</c>) branches, and the <c>drawContours</c>-based
/// <c>new_accuracy</c> / <c>approx</c> tests are out of scope (§1, §8.2.3), and
/// the <c>deep</c> test is intentionally skipped (see remarks below).
/// </summary>
public sealed class ContoursOpenCvPortedTests
{
    // ---------------------------------------------------------------------
    // Port of TEST_P(Imgproc_FindContours_Modes1, rectangle)
    //
    // Upstream draws a 1px rectangle outline and checks, per retrieval mode:
    //   RETR_EXTERNAL       -> 1 contour; NONE size == int_perimeter;
    //                          SIMPLE == ext_ref (4 corners).
    //   RETR_LIST           -> 2 contours; NONE sizes {int_perimeter-4, int_perimeter};
    //                          SIMPLE {int_ref, ext_ref}.
    //   RETR_CCOMP/TREE     -> 2 contours; NONE sizes {int_perimeter, int_perimeter-4};
    //                          SIMPLE {ext_ref, int_ref}.
    // where ext_perimeter = 2w + 2h and int_perimeter = ext_perimeter - 4.
    //
    // Rather than upstream's RNG-driven sizes we use fixed rectangles; the
    // analytic relationships are size-independent. The reference point layouts
    // below were cross-checked against cv2 4.5.5.
    // ---------------------------------------------------------------------

    // imgW, imgH, x0, y0, w, h  (outline pixels span x in [x0, x0+w-1], y in [y0, y0+h-1])
    private static readonly int[][] Rectangles =
    {
        new[] { 20, 20, 2, 3, 12, 9 },
        new[] { 12, 10, 1, 1, 8, 6 },
    };

    public static IEnumerable<object[]> ModeMethodData()
    {
        int[] modes = { 0, 1, 2, 3 }; // External, List, ConnectedComponents, Tree
        int[] methods = { 1, 2 };     // None, Simple
        foreach (int mode in modes)
        {
            foreach (int method in methods)
            {
                yield return new object[] { mode, method };
            }
        }
    }

    [Theory]
    [MemberData(nameof(ModeMethodData))]
    public void FindContours_Rectangle_MatchesAnalyticReference(int modeInt, int methodInt)
    {
        var mode = (RetrievalMode)modeInt;
        var method = (ContourApproximationMode)methodInt;

        foreach (int[] r in Rectangles)
        {
            RunRectangleCase(mode, method, r[0], r[1], r[2], r[3], r[4], r[5]);
        }
    }

    private static void RunRectangleCase(
        RetrievalMode mode,
        ContourApproximationMode method,
        int imgW,
        int imgH,
        int x0,
        int y0,
        int w,
        int h)
    {
        var image = new Mat<byte>(imgW, imgH);
        DrawRectOutline(image, x0, y0, w, h);

        Point[] extRef = ExternalReference(x0, y0, w, h);
        Point[] intRef = InternalReference(x0, y0, w, h);
        int extPerimeter = (w * 2) + (h * 2);
        int intPerimeter = extPerimeter - 4;

        ContourResult result = Contours.FindContours(image, mode, method);
        IReadOnlyList<Point[]> c = result.Contours;

        string tag = $"mode={mode}, method={method}, rect=({x0},{y0},{w},{h})";

        if (mode == RetrievalMode.External)
        {
            Assert.Single(c);
            if (method == ContourApproximationMode.None)
            {
                Assert.Equal(intPerimeter, c[0].Length);
            }
            else
            {
                Assert.Equal(extRef, c[0]);
            }
        }
        else
        {
            Assert.Equal(2, c.Count);
            if (mode == RetrievalMode.List)
            {
                if (method == ContourApproximationMode.None)
                {
                    Assert.Equal(intPerimeter - 4, c[0].Length);
                    Assert.Equal(intPerimeter, c[1].Length);
                }
                else
                {
                    Assert.Equal(intRef, c[0]);
                    Assert.Equal(extRef, c[1]);
                }
            }
            else
            {
                // RETR_CCOMP and RETR_TREE: outer contour first.
                if (method == ContourApproximationMode.None)
                {
                    Assert.Equal(intPerimeter, c[0].Length);
                    Assert.Equal(intPerimeter - 4, c[1].Length);
                }
                else
                {
                    Assert.Equal(extRef, c[0]);
                    Assert.Equal(intRef, c[1]);
                }
            }
        }

        Assert.Equal(c.Count, result.Hierarchy.Count);
        _ = tag;
    }

    private static Point[] ExternalReference(int x0, int y0, int w, int h)
    {
        // Mirrors upstream ext_ref: {tl, tl+(0,h-1), br+(-1,-1), tl+(w-1,0)}.
        return new[]
        {
            new Point(x0, y0),
            new Point(x0, y0 + h - 1),
            new Point(x0 + w - 1, y0 + h - 1),
            new Point(x0 + w - 1, y0),
        };
    }

    private static Point[] InternalReference(int x0, int y0, int w, int h)
    {
        // Mirrors upstream int_ref (the 8 inner-hole dominant points).
        return new[]
        {
            new Point(x0, y0 + 1),
            new Point(x0 + 1, y0),
            new Point(x0 + w - 2, y0),
            new Point(x0 + w - 1, y0 + 1),
            new Point(x0 + w - 1, y0 + h - 2),
            new Point(x0 + w - 2, y0 + h - 1),
            new Point(x0 + 1, y0 + h - 1),
            new Point(x0, y0 + h - 2),
        };
    }

    private static void DrawRectOutline(Mat<byte> m, int x0, int y0, int w, int h)
    {
        int x1 = x0 + w - 1;
        int y1 = y0 + h - 1;

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

    // ---------------------------------------------------------------------
    // Port of TEST_P(Imgproc_FindContours_Modes1, small)
    //
    // Many isolated 1-2px blobs; for the CV_8U path upstream asserts one
    // contour per blob for every retrieval mode and method (the diagonal
    // 2px blobs only split under RETR_FLOODFILL, which is out of scope).
    // Blob layout verified against cv2 4.5.5.
    // ---------------------------------------------------------------------

    private const int GridN = 8;
    private const int Cell = 5;
    private const int Margin = 2;

    [Theory]
    [InlineData(RetrievalMode.External, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.External, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.External, ContourApproximationMode.Tc89L1)]
    [InlineData(RetrievalMode.External, ContourApproximationMode.Tc89KCos)]
    [InlineData(RetrievalMode.List, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.List, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.List, ContourApproximationMode.Tc89L1)]
    [InlineData(RetrievalMode.List, ContourApproximationMode.Tc89KCos)]
    [InlineData(RetrievalMode.ConnectedComponents, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.ConnectedComponents, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.ConnectedComponents, ContourApproximationMode.Tc89L1)]
    [InlineData(RetrievalMode.ConnectedComponents, ContourApproximationMode.Tc89KCos)]
    [InlineData(RetrievalMode.Tree, ContourApproximationMode.None)]
    [InlineData(RetrievalMode.Tree, ContourApproximationMode.Simple)]
    [InlineData(RetrievalMode.Tree, ContourApproximationMode.Tc89L1)]
    [InlineData(RetrievalMode.Tree, ContourApproximationMode.Tc89KCos)]
    public void FindContours_SmallBlobs_OneContourPerBlob(
        RetrievalMode mode,
        ContourApproximationMode method)
    {
        int size = (GridN * Cell) + (2 * Margin);
        var image = new Mat<byte>(size, size);
        int blobCount = 0;

        for (int row = 0; row < GridN; row++)
        {
            for (int col = 0; col < GridN; col++)
            {
                int index = (row * GridN) + col;
                int kind = index % 9;
                int cx = (col * Cell) + Margin + 2;
                int cy = (row * Cell) + Margin + 2;

                image.At(cy, cx) = 255;
                if (kind != 0)
                {
                    (int dx, int dy) = SmallOffset(kind);
                    image.At(cy + dy, cx + dx) = 255;
                }

                blobCount++;
            }
        }

        ContourResult result = Contours.FindContours(image, mode, method);

        Assert.Equal(blobCount, result.Contours.Count);
        Assert.Equal(result.Contours.Count, result.Hierarchy.Count);
    }

    private static (int Dx, int Dy) SmallOffset(int kind) => kind switch
    {
        1 => (1, 0),
        2 => (1, -1),
        3 => (0, -1),
        4 => (-1, -1),
        5 => (-1, 0),
        6 => (-1, 1),
        7 => (0, 1),
        8 => (1, 1),
        _ => (0, 0),
    };

    // ---------------------------------------------------------------------
    // TEST_P(Imgproc_FindContours_Modes1, deep) is intentionally NOT ported.
    //
    // Upstream builds 249 concentric 1px rectangles in a 1000x1000 image and
    // asserts 2*NUM contours. The count relationship is already exercised at
    // small scale by the nested-rings golden case (triple_hollow_16x16) and the
    // offset-shift basic test; reproducing the full 1000x1000 / 249-level image
    // here adds runtime and memory cost without additional analytic coverage,
    // and the equivalent nesting depth is covered by the golden battery.
    // ---------------------------------------------------------------------
}
