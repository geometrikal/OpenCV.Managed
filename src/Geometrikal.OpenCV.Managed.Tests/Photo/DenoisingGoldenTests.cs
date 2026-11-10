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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/nlmeans_golden.json</c> through
/// <see cref="Denoising"/>. Outputs must match exactly.
/// Regenerate with <c>python tools/generate_nlmeans_golden.py</c>.
/// </summary>
public sealed class DenoisingGoldenTests
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
    public void Denoising_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32(), cn = c.GetProperty("channels").GetInt32();
        int t = c.GetProperty("t").GetInt32(), s = c.GetProperty("s").GetInt32();
        float[] hs = c.GetProperty("h").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        byte[] src = Convert.FromBase64String(c.GetProperty("src").GetString()!);
        byte[] expected = Convert.FromBase64String(c.GetProperty("result").GetString()!);
        byte[] actual;
        if (c.GetProperty("op").GetString() == "colored")
        {
            actual = Denoising.FastNlMeansDenoisingColored(new Mat<byte>(src, w, h, cn), hs[0], (float)c.GetProperty("hColor").GetDouble(), t, s).ToArray();
        }
        else
        {
            var norm = c.GetProperty("norm").GetString() == "L1" ? NormType.L1 : NormType.L2;
            actual = c.GetProperty("depth").GetString() == "u8"
                ? Denoising.FastNlMeansDenoising(new Mat<byte>(src, w, h, cn), hs, t, s, norm).ToArray()
                : MemoryMarshal.AsBytes(Denoising.FastNlMeansDenoising(new Mat<ushort>(MemoryMarshal.Cast<byte, ushort>(src).ToArray(), w, h, cn), hs, t, s, norm).ToArray().AsSpan()).ToArray();
        }

        Assert.Equal(expected, actual);
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "nlmeans_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
