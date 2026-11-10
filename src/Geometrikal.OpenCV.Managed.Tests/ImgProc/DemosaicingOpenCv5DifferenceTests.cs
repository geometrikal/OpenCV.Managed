namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins the OpenCV 5.0 behaviour of the demosaicing golden cases where cv2 4.5.5 differs:
/// <list type="bullet">
/// <item>PR #25968 (issue #25823): the 4.5.5 u8 SIMD Bayer-to-gray kernel descaled wrongly. OpenCV 5.0
/// equals the scalar formula, which is also the 16-bit result.</item>
/// <item>PR #27226 (issues #5089, #27225): 4.5.5 VNG wrote each interior pixel two rows above its
/// source neighbourhood and copied border pixels. OpenCV 5.0 pads the source by reflection.</item>
/// <item>PR #27226: VNG on images smaller than 8 pixels falls back to bilinear interpolation; 4.5.5
/// used the RG pattern for every VNG code.</item>
/// </list>
/// </summary>
public sealed class DemosaicingOpenCv5DifferenceTests
{
    internal static readonly string[] GrayCases =
        [.. from p in new[] { "BG", "GB", "RG", "GR" } from s in new[] { "37x29", "64x48", "40x9" } select $"gray_{p}_u8_{s}"];

    internal static readonly string[] VngShiftCases =
        [.. from p in new[] { "BG", "GB", "RG", "GR" } from s in new[] { "37x29", "64x48", "40x9" } select $"vng_{p}_u8_{s}"];

    internal static readonly string[] VngSmallCases =
        [.. from p in new[] { "BG", "GB", "GR" } from s in new[] { "7x7", "3x3" } select $"vng_{p}_u8_{s}"];

    public static TheoryData<string> Gray() => new(GrayCases);

    public static TheoryData<string> VngShift() => new(VngShiftCases);

    public static TheoryData<string> VngSmall() => new(VngSmallCases);

    [Theory]
    [MemberData(nameof(Gray))]
    public void BayerToGray_U8_EqualsTheScalarFormula(string name)
    {
        var (img, r) = DemosaicingGoldenTests.Find(name);
        var src = Source(img);
        var pattern = Enum.Parse<BayerPattern>(r.GetProperty("pattern").GetString()!);
        var wide = new Mat<ushort>(src.ToArray().Select(v => (ushort)v).ToArray(), src.Width, src.Height);

        byte[] actual = Demosaicing.BayerToGray(src, pattern).ToArray();

        Assert.Equal(Demosaicing.BayerToGray(wide, pattern).ToArray().Select(v => (byte)v).ToArray(), actual);
        Assert.NotEqual(DemosaicingGoldenTests.Expected<byte>(r), actual);
    }

    [Theory]
    [MemberData(nameof(VngShift))]
    public void Vng_Interior_IsTwoRowsBelowCv2(string name)
    {
        var (img, r) = DemosaicingGoldenTests.Find(name);
        int w = img.GetProperty("width").GetInt32(), h = img.GetProperty("height").GetInt32();
        byte[] actual = DemosaicingGoldenTests.Run<byte>(img, r);
        byte[] expected = DemosaicingGoldenTests.Expected<byte>(r);

        int compared = 0, differ = 0;
        for (int y = 4; y < h - 4; y++)
        {
            for (int x = 4; x < w - 4; x++)
            {
                for (int k = 0; k < 3; k++)
                {
                    int a = actual[((y * w) + x) * 3 + k], e = expected[(((y - 2) * w) + x) * 3 + k];
                    compared++;
                    differ += a != e ? 1 : 0;
                }
            }
        }

        // The ng = 7 scale differs between the SSE blocks and the scalar code; block positions moved with the padding.
        Assert.True(differ <= compared / 1000, $"{name}: {differ} of {compared} interior values differ");
    }

    [Theory]
    [MemberData(nameof(VngSmall))]
    public void Vng_SmallImage_IsBilinearWithItsOwnPattern(string name)
    {
        var (img, r) = DemosaicingGoldenTests.Find(name);
        var src = Source(img);
        var pattern = Enum.Parse<BayerPattern>(r.GetProperty("pattern").GetString()!);

        byte[] actual = Demosaicing.Demosaic(src, pattern, DemosaicMethod.Vng).ToArray();

        Assert.Equal(Demosaicing.Demosaic(src, pattern).ToArray(), actual);
        Assert.Equal(DemosaicingGoldenTests.Expected<byte>(r), Demosaicing.Demosaic(src, BayerPattern.RG).ToArray());
    }

    private static Mat<byte> Source(System.Text.Json.JsonElement img) =>
        new(Convert.FromBase64String(img.GetProperty("data").GetString()!), img.GetProperty("width").GetInt32(), img.GetProperty("height").GetInt32());
}
