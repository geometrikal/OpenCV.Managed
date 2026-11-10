namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/divspectrums_golden.json</c> through
/// <see cref="SpectralTransforms.DivSpectrums{T}"/>. Outputs must match bit for bit.
/// Regenerate with <c>python tools/generate_divspectrums_golden.py</c>.
/// </summary>
public sealed class DivSpectrumsGoldenTests
{
    private static readonly List<JsonElement> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    public static IEnumerable<object[]> SuspectNames() =>
        Cases.Where(c => c.GetProperty("suspect").GetBoolean()).Select(c => new object[] { c.GetProperty("name").GetString()! });

    /// <summary>Float cases are suspects: commit e1dba2c6d2 (2024) moved the float row pairs to double.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreTheFloatCases() =>
        Assert.All(Cases, c => Assert.Equal(c.GetProperty("depth").GetString() == "f32", c.GetProperty("suspect").GetBoolean()));

    /// <summary>The internal 4.5.5 mode changes only the float row pairs and reproduces every float case.</summary>
    [Theory]
    [MemberData(nameof(SuspectNames))]
    public void DivSpectrums_Legacy455Mode_ReproducesCv2(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        Assert.Equal(Convert.FromBase64String(c.GetProperty("result").GetString()!), Run<float>(c, legacy: true));
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void DivSpectrums_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        byte[] expected = Convert.FromBase64String(c.GetProperty("result").GetString()!);
        byte[] actual = c.GetProperty("depth").GetString() == "f32" ? Run<float>(c) : Run<double>(c);
        Assert.Equal(expected, actual);
    }

    private static byte[] Run<T>(JsonElement c, bool legacy = false)
        where T : unmanaged
    {
        int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32(), cn = c.GetProperty("channels").GetInt32();
        Mat<T> M(string key) => new(MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(c.GetProperty(key).GetString()!)).ToArray(), w, h, cn);
        var flags = c.GetProperty("rows").GetBoolean() ? DftFlags.Rows : DftFlags.None;
        var result = SpectralTransforms.DivSpectrums(M("a"), M("b"), flags, c.GetProperty("conj").GetBoolean(), legacy);
        return MemoryMarshal.AsBytes(result.ToArray().AsSpan()).ToArray();
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "divspectrums_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
