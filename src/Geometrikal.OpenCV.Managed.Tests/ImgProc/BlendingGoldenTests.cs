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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/blend_golden.json</c> through
/// <see cref="Blending.BlendLinear{T}"/>. Outputs must match bit for bit.
/// Regenerate with <c>python tools/generate_blend_golden.py</c>.
/// </summary>
public sealed class BlendingGoldenTests
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
    public void BlendLinear_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        byte[] expected = Convert.FromBase64String(c.GetProperty("result").GetString()!);
        byte[] actual = c.GetProperty("depth").GetString() == "u8" ? Run<byte>(c) : Run<float>(c);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void BlendLinear_InvalidInputs_Throw()
    {
        var w = new Mat<float>(4, 4);
        Assert.Throws<NotSupportedException>(() => Blending.BlendLinear(new Mat<ushort>(4, 4), new Mat<ushort>(4, 4), w, w));
        Assert.Throws<ArgumentException>(() => Blending.BlendLinear(new Mat<byte>(4, 4), new Mat<byte>(4, 4, 3), w, w));
        Assert.Throws<ArgumentException>(() => Blending.BlendLinear(new Mat<byte>(4, 4), new Mat<byte>(4, 4), new Mat<float>(4, 3), w));
    }

    private static byte[] Run<T>(JsonElement c)
        where T : unmanaged
    {
        int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32(), cn = c.GetProperty("channels").GetInt32();
        Mat<TM> M<TM>(string key, int ch)
            where TM : unmanaged =>
            new(MemoryMarshal.Cast<byte, TM>(Convert.FromBase64String(c.GetProperty(key).GetString()!)).ToArray(), w, h, ch);
        var result = Blending.BlendLinear(M<T>("a", cn), M<T>("b", cn), M<float>("w1", 1), M<float>("w2", 1));
        return MemoryMarshal.AsBytes(result.ToArray().AsSpan()).ToArray();
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "blend_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
