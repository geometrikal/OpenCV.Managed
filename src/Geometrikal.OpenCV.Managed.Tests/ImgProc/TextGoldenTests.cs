namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Linq;

using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Compares <see cref="TextMetrics.GetFontScaleFromHeight"/> with the cv2 5.0.0
/// values in the <c>fontScale</c> section of <c>TestData/drawing_aa_golden.json</c>.
/// </summary>
public sealed class TextGoldenTests
{
    public static IEnumerable<object[]> FontScaleCases() =>
        DrawingAntiAliasedGoldenTests.Root.GetProperty("fontScale").EnumerateArray().Select(e => new object[]
        {
            e.GetProperty("fontFace").GetInt32(),
            e.GetProperty("pixelHeight").GetInt32(),
            e.GetProperty("thickness").GetInt32(),
            e.GetProperty("expected").GetDouble(),
        });

    [Theory]
    [MemberData(nameof(FontScaleCases))]
    public void GetFontScaleFromHeight_MatchesOpenCv(int fontFace, int pixelHeight, int thickness, double expected)
    {
        double actual = TextMetrics.GetFontScaleFromHeight((HersheyFonts)fontFace, pixelHeight, thickness);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void GetFontScaleFromHeight_UnknownFont_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TextMetrics.GetFontScaleFromHeight((HersheyFonts)8, 10));
    }
}
