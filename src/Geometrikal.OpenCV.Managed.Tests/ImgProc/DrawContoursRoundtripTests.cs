namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// cv2-independent roundtrip tests coupling the managed
/// <see cref="Contours.FindContours"/> and
/// <see cref="Drawing.DrawContours"/> ports.
///
/// For a battery of binary masks: extract contours, redraw them FILLED onto a
/// zeroed matrix, and extract contours again. The re-extracted contours and
/// hierarchy must be identical to the first pass (FILLED redraw is idempotent
/// for the contour set). For solid shapes (no holes) the redrawn matrix must
/// additionally equal the original mask byte-for-byte, since filling a region's
/// border contour reproduces the region.
/// </summary>
public sealed class DrawContoursRoundtripTests
{
    private static readonly Scalar White = new(255);

    public static IEnumerable<object[]> Cases()
    {
        string[] masks = { "solid_rect", "solid_lshape", "solid_plus", "hollow_rect" };
        RetrievalMode[] modes = { RetrievalMode.External, RetrievalMode.Tree };
        foreach (string mask in masks)
        {
            foreach (RetrievalMode mode in modes)
            {
                yield return new object[] { mask, mode };
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FillContours_Roundtrip_IsIdempotent(string maskName, RetrievalMode mode)
    {
        (Mat<byte> mask, bool isSolid) = BuildMask(maskName);

        ContourResult first = Contours.FindContours(mask, mode, ContourApproximationMode.None);
        Assert.NotEmpty(first.Contours);

        var redrawn = new Mat<byte>(mask.Width, mask.Height);
        Drawing.DrawContours(
            redrawn,
            first.Contours,
            contourIndex: -1,
            White,
            thickness: Drawing.Filled,
            lineType: LineType.EightConnected,
            hierarchy: first.Hierarchy,
            maxLevel: int.MaxValue);

        ContourResult second = Contours.FindContours(redrawn, mode, ContourApproximationMode.None);

        AssertResultsEqual(first, second);

        // Solid shapes: the FILLED redraw must equal the original mask exactly.
        if (isSolid)
        {
            Assert.Equal(mask.ToArray(), redrawn.ToArray());
        }
    }

    /// <summary>
    /// A solid shape whose contour is drawn FILLED onto a zero mat reproduces
    /// the shape byte-for-byte, independent of retrieval mode.
    /// </summary>
    [Fact]
    public void FilledSolidRect_RedrawEqualsOriginal()
    {
        (Mat<byte> mask, _) = BuildMask("solid_rect");

        ContourResult r = Contours.FindContours(
            mask, RetrievalMode.External, ContourApproximationMode.None);

        var redrawn = new Mat<byte>(mask.Width, mask.Height);
        Drawing.DrawContours(redrawn, r.Contours, -1, White, thickness: Drawing.Filled);

        Assert.Equal(mask.ToArray(), redrawn.ToArray());
    }

    // -- mask constructors --------------------------------------------------

    private static (Mat<byte> Mask, bool IsSolid) BuildMask(string name)
    {
        var m = new Mat<byte>(24, 24);
        bool solid = true;

        switch (name)
        {
            case "solid_rect":
                FillRect(m, 4, 3, 15, 14);
                break;

            case "solid_lshape":
                FillRect(m, 3, 3, 7, 18);   // vertical bar (5px wide)
                FillRect(m, 3, 14, 18, 18); // horizontal foot (5px tall)
                break;

            case "solid_plus":
                FillRect(m, 3, 9, 20, 14);  // horizontal bar
                FillRect(m, 9, 3, 14, 20);  // vertical bar
                break;

            case "hollow_rect":
                // 2px-thick square wall around a hole in [7,16] x [7,16].
                FillRect(m, 4, 4, 19, 19);
                ClearRect(m, 7, 7, 16, 16);
                solid = false;
                break;

            default:
                throw new ArgumentException($"Unknown mask '{name}'.", nameof(name));
        }

        return (m, solid);
    }

    private static void FillRect(Mat<byte> m, int x0, int y0, int x1, int y1)
    {
        for (int y = y0; y <= y1; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = x0; x <= x1; x++)
            {
                row[x] = 255;
            }
        }
    }

    private static void ClearRect(Mat<byte> m, int x0, int y0, int x1, int y1)
    {
        for (int y = y0; y <= y1; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = x0; x <= x1; x++)
            {
                row[x] = 0;
            }
        }
    }

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
}
