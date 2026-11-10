namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Analytic (cv2-independent) tests for the <c>maxLevel</c> / hierarchy walk of
/// <see cref="Drawing.DrawContours"/>, pinning the OpenCV 5 selection semantics
/// (spec §5.13), which deliberately differ from OpenCV ≤ 4.6.
///
/// The fixture is three concentric 1-pixel square rings (16×16), matching the
/// contour battery's <c>triple_hollow</c> shape:
/// <list type="bullet">
///   <item><description>ring 1 outline (1,1)–(14,14)</description></item>
///   <item><description>ring 2 outline (4,4)–(11,11)</description></item>
///   <item><description>ring 3 outline (6,6)–(9,9)</description></item>
/// </list>
/// Under <see cref="RetrievalMode.Tree"/> this yields a single top-level outer
/// boundary and a strictly nested chain of six contours (each ring contributes
/// an outer and a hole boundary):
/// <c>O1(L0) → I1(L1) → O2(L2) → I2(L3) → O3(L4) → I3(L5)</c>.
///
/// With <c>contourIndex = -1</c> the OpenCV 5 selection is:
/// <list type="bullet">
///   <item><description><c>maxLevel = 0</c> → the whole index range (ALL six
///   contours), because the <c>maxLevel == 0</c> shortcut fills <c>i..end</c>
///   directly rather than walking the tree — this is the ≤ 4.6 divergence.</description></item>
///   <item><description><c>maxLevel = 1</c> → levels 0–1 only: {O1, I1}.</description></item>
///   <item><description><c>maxLevel = 2</c> → levels 0–2: {O1, I1, O2}.</description></item>
///   <item><description><c>maxLevel = int.MaxValue</c> → ALL six contours.</description></item>
/// </list>
/// Expected pixels are derived analytically (axis-aligned even-odd fills) and
/// verified by sampling known coordinates plus the two rock-solid full-image
/// identities (draw-all reproduces the original mask).
/// </summary>
public sealed class DrawContoursMaxLevelTests
{
    private const int Size = 16;
    private static readonly Scalar White = new(255);

    // -- FILLED: draw-all identities (maxLevel 0 and int.MaxValue) ----------

    [Fact]
    public void Filled_MaxLevel0_FillsAllContours_ReproducingMask()
    {
        // maxLevel == 0 with contourIndex = -1 fills the whole range (OpenCV 5),
        // so the even-odd fill of every boundary reproduces the original rings.
        Mat<byte> mask = BuildTripleRings();
        Mat<byte> drawn = DrawAll(thickness: Drawing.Filled, maxLevel: 0);
        Assert.Equal(mask.ToArray(), drawn.ToArray());
    }

    [Fact]
    public void Filled_MaxLevelMax_FillsAllContours_ReproducingMask()
    {
        Mat<byte> mask = BuildTripleRings();
        Mat<byte> drawn = DrawAll(thickness: Drawing.Filled, maxLevel: int.MaxValue);
        Assert.Equal(mask.ToArray(), drawn.ToArray());
    }

    // -- FILLED: maxLevel 1 selects {O1, I1} -> only ring 1 ------------------

    [Fact]
    public void Filled_MaxLevel1_FillsOnlyRing1()
    {
        Mat<byte> drawn = DrawAll(thickness: Drawing.Filled, maxLevel: 1);

        // ring-1 edge pixel: filled (member of O1 only -> odd).
        AssertPixel(drawn, 1, 7, 255);
        AssertPixel(drawn, 7, 1, 255);
        // ring 2 / ring 3 and every interior region: empty.
        AssertPixel(drawn, 7, 4, 0);   // ring 2 edge
        AssertPixel(drawn, 7, 6, 0);   // ring 3 edge
        AssertPixel(drawn, 7, 5, 0);   // gap ring2..ring3
        AssertPixel(drawn, 7, 7, 0);   // centre hole
        AssertPixel(drawn, 7, 3, 0);   // gap ring1..ring2
    }

    // -- FILLED: maxLevel 2 selects {O1, I1, O2} -> ring1 + solid O2 square --

    [Fact]
    public void Filled_MaxLevel2_FillsRing1AndSolidRing2Square()
    {
        Mat<byte> drawn = DrawAll(thickness: Drawing.Filled, maxLevel: 2);

        // ring-1 edge still present.
        AssertPixel(drawn, 1, 7, 255);
        // O2's whole square interior [4,11]^2 becomes solid: ring2 edge, the
        // gap that used to be background, ring3 edge, and the centre hole are
        // ALL filled at maxLevel 2.
        AssertPixel(drawn, 7, 4, 255);   // ring 2 edge (in O2)
        AssertPixel(drawn, 7, 5, 255);   // gap ring2..ring3 (inside O2)
        AssertPixel(drawn, 7, 6, 255);   // ring 3 edge (inside O2)
        AssertPixel(drawn, 7, 7, 255);   // centre hole (inside O2)
        AssertPixel(drawn, 5, 7, 255);   // inside O2
        // the [2,3] gap ring1..ring2 lies OUTSIDE O2 -> still empty.
        AssertPixel(drawn, 7, 3, 0);
    }

    // -- thickness 1 (outline): which ring outlines are drawn ----------------

    [Fact]
    public void Outline_MaxLevel0_And_Max_DrawAllRings()
    {
        Mat<byte> mask = BuildTripleRings();

        // maxLevel 0 (range shortcut) and int.MaxValue both draw every contour;
        // the union of all border pixels of 1px rings is the whole mask.
        Assert.Equal(mask.ToArray(), DrawAll(thickness: 1, maxLevel: 0).ToArray());
        Assert.Equal(mask.ToArray(), DrawAll(thickness: 1, maxLevel: int.MaxValue).ToArray());
    }

    [Fact]
    public void Outline_MaxLevel1_DrawsOnlyRing1()
    {
        Mat<byte> drawn = DrawAll(thickness: 1, maxLevel: 1);

        AssertPixel(drawn, 7, 1, 255);   // ring 1 outline
        AssertPixel(drawn, 1, 7, 255);   // ring 1 outline
        AssertPixel(drawn, 7, 4, 0);     // ring 2 outline not drawn
        AssertPixel(drawn, 7, 6, 0);     // ring 3 outline not drawn
        // interior regions never carry outline pixels.
        AssertPixel(drawn, 7, 5, 0);
        AssertPixel(drawn, 7, 7, 0);
    }

    [Fact]
    public void Outline_MaxLevel2_DrawsRing1AndRing2()
    {
        Mat<byte> drawn = DrawAll(thickness: 1, maxLevel: 2);

        AssertPixel(drawn, 7, 1, 255);   // ring 1 outline
        AssertPixel(drawn, 7, 4, 255);   // ring 2 outline (O2 selected)
        AssertPixel(drawn, 7, 6, 0);     // ring 3 outline not drawn
        // outline mode does NOT fill the O2 interior, unlike FILLED maxLevel 2.
        AssertPixel(drawn, 7, 5, 0);
        AssertPixel(drawn, 7, 7, 0);
    }

    // -- fixture helpers ----------------------------------------------------

    private static Mat<byte> DrawAll(int thickness, int maxLevel)
    {
        Mat<byte> mask = BuildTripleRings();
        ContourResult result = Contours.FindContours(
            mask, RetrievalMode.Tree, ContourApproximationMode.None);

        // Sanity: the fixture must produce the six-contour nested chain the
        // analytic expectations assume (one top-level root).
        Assert.Equal(6, result.Contours.Count);
        int roots = 0;
        foreach (ContourHierarchy h in result.Hierarchy)
        {
            if (h.Parent < 0)
            {
                roots++;
            }
        }

        Assert.Equal(1, roots);

        var image = new Mat<byte>(Size, Size);
        Drawing.DrawContours(
            image,
            result.Contours,
            contourIndex: -1,
            White,
            thickness,
            LineType.EightConnected,
            result.Hierarchy,
            maxLevel);

        return image;
    }

    private static Mat<byte> BuildTripleRings()
    {
        var m = new Mat<byte>(Size, Size);
        DrawRingOutline(m, 1, 1, 14, 14);
        DrawRingOutline(m, 4, 4, 11, 11);
        DrawRingOutline(m, 6, 6, 9, 9);
        return m;
    }

    private static void DrawRingOutline(Mat<byte> m, int x0, int y0, int x1, int y1)
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

    private static void AssertPixel(Mat<byte> m, int x, int y, byte expected)
    {
        byte actual = m.At(y, x);
        Assert.True(
            actual == expected,
            $"pixel (x={x}, y={y}) expected {expected} but was {actual}.");
    }
}
