namespace Geometrikal.OpenCV.Managed.Tests.Photo;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Photo;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/inpaint_golden.json</c> through
/// <see cref="Inpainting.Inpaint{T}"/>. Outputs must match exactly.
/// Regenerate with <c>python tools/generate_inpaint_golden.py</c>.
/// </summary>
public sealed class InpaintGoldenTests
{
    private static readonly List<(JsonElement Image, JsonElement Result)> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.Result.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.Result.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    /// <summary>The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences, pinned in <see cref="InpaintOpenCv5DifferenceTests"/>.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_HaveReasonsAndLeaveDirectCases()
    {
        Assert.All(Cases.Where(c => c.Result.GetProperty("suspect").GetBoolean()),
            c => Assert.False(string.IsNullOrEmpty(c.Result.GetProperty("reason").GetString())));
        Assert.True(CaseNames().Count() >= 20);
    }

    internal static string[] SuspectNames() =>
        Cases.Where(c => c.Result.GetProperty("suspect").GetBoolean()).Select(c => c.Result.GetProperty("name").GetString()!).ToArray();

    internal static (JsonElement Image, JsonElement Result) Find(string name) =>
        Cases.Single(x => x.Result.GetProperty("name").GetString() == name);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Inpaint_MatchesOpenCvGolden(string name)
    {
        var (img, r) = Find(name);
        switch (img.GetProperty("name").GetString())
        {
            case "u16": Assert.Equal(Expected<ushort>(r), Run<ushort>(img, r)); break;
            case "f32": Assert.Equal(Expected<float>(r), Run<float>(img, r)); break;
            default: Assert.Equal(Expected<byte>(r), Run<byte>(img, r)); break;
        }
    }

    internal static T[] Run<T>(JsonElement img, JsonElement r)
        where T : unmanaged
    {
        int w = img.GetProperty("width").GetInt32(), h = img.GetProperty("height").GetInt32();
        var src = new Mat<T>(MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(img.GetProperty("data").GetString()!)).ToArray(),
            w, h, img.GetProperty("channels").GetInt32());
        var mask = new Mat<byte>(Convert.FromBase64String(r.GetProperty("mask").GetString()!), w, h);
        var method = r.GetProperty("method").GetString() == "telea" ? InpaintMethod.Telea : InpaintMethod.NavierStokes;
        return Inpainting.Inpaint(src, mask, r.GetProperty("radius").GetDouble(), method).ToArray();
    }

    internal static T[] Expected<T>(JsonElement r)
        where T : unmanaged =>
        MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(r.GetProperty("result").GetString()!)).ToArray();

    private static List<(JsonElement, JsonElement)> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "inpaint_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("images").EnumerateArray()
            .SelectMany(i => i.GetProperty("results").EnumerateArray().Select(r => (i.Clone(), r.Clone())))
            .ToList();
    }
}
