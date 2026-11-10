namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/demosaic_golden.json</c> through
/// <see cref="Demosaicing"/>. Outputs must match exactly.
/// Regenerate with <c>python tools/generate_demosaic_golden.py</c>.
/// </summary>
public sealed class DemosaicingGoldenTests
{
    private static readonly List<(JsonElement Image, JsonElement Result)> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.Result.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.Result.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    /// <summary>The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences, pinned in <see cref="DemosaicingOpenCv5DifferenceTests"/>.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Cases.Where(c => c.Result.GetProperty("suspect").GetBoolean()).Select(c => c.Result.GetProperty("name").GetString()!)
            .Order(StringComparer.Ordinal).ToArray();
        string[] expected = DemosaicingOpenCv5DifferenceTests.GrayCases.Concat(DemosaicingOpenCv5DifferenceTests.VngShiftCases)
            .Concat(DemosaicingOpenCv5DifferenceTests.VngSmallCases).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(expected, actual);
    }

    internal static (JsonElement Image, JsonElement Result) Find(string name) =>
        Cases.Single(x => x.Result.GetProperty("name").GetString() == name);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Demosaic_MatchesOpenCvGolden(string name)
    {
        var (img, r) = Find(name);
        if (img.GetProperty("depth").GetString() == "u8")
        {
            Assert.Equal(Expected<byte>(r), Run<byte>(img, r));
        }
        else
        {
            Assert.Equal(Expected<ushort>(r), Run<ushort>(img, r));
        }
    }

    internal static T[] Run<T>(JsonElement img, JsonElement r)
        where T : unmanaged
    {
        var src = new Mat<T>(MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(img.GetProperty("data").GetString()!)).ToArray(),
            img.GetProperty("width").GetInt32(), img.GetProperty("height").GetInt32());
        var pattern = Enum.Parse<BayerPattern>(r.GetProperty("pattern").GetString()!);
        var result = r.GetProperty("method").GetString() switch
        {
            "gray" => Demosaicing.BayerToGray(src, pattern),
            "bgr" => Demosaicing.Demosaic(src, pattern),
            "bgra" => Demosaicing.Demosaic(src, pattern, dstChannels: 4),
            "vng" => Demosaicing.Demosaic(src, pattern, DemosaicMethod.Vng),
            _ => Demosaicing.Demosaic(src, pattern, DemosaicMethod.EdgeAware),
        };
        return result.ToArray();
    }

    internal static T[] Expected<T>(JsonElement r)
        where T : unmanaged =>
        MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(r.GetProperty("result").GetString()!)).ToArray();

    private static List<(JsonElement, JsonElement)> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "demosaic_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("images").EnumerateArray()
            .SelectMany(i => i.GetProperty("results").EnumerateArray().Select(r => (i.Clone(), r.Clone())))
            .ToList();
    }
}
