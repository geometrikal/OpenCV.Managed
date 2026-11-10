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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/precorner_golden.json</c> through
/// <see cref="Corners.PreCornerDetect{T}"/>. Outputs must match bit for bit.
/// Regenerate with <c>python tools/generate_precorner_golden.py</c>.
/// </summary>
public sealed class PreCornerDetectGoldenTests
{
    private static readonly List<JsonElement> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_HasCasesAndNoSuspects()
    {
        Assert.NotEmpty(Cases);
        Assert.DoesNotContain(Cases, c => c.GetProperty("suspect").GetBoolean());
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void PreCornerDetect_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32(), k = c.GetProperty("ksize").GetInt32();
        var border = (BorderType)c.GetProperty("border").GetInt32();
        byte[] src = Convert.FromBase64String(c.GetProperty("src").GetString()!);
        var actual = c.GetProperty("depth").GetString() == "u8"
            ? Corners.PreCornerDetect(new Mat<byte>(src, w, h), k, border)
            : Corners.PreCornerDetect(new Mat<float>(MemoryMarshal.Cast<byte, float>(src).ToArray(), w, h), k, border);

        Assert.Equal(Convert.FromBase64String(c.GetProperty("result").GetString()!), MemoryMarshal.AsBytes(actual.ToArray().AsSpan()).ToArray());
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "precorner_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
