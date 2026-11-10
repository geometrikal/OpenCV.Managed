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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/polar_golden.json</c> through
/// <see cref="Geometry.WarpPolar{T}"/>. Outputs must match exactly.
/// Regenerate with <c>python tools/generate_polar_golden.py</c>.
/// </summary>
public sealed class WarpPolarGoldenTests
{
    private static readonly List<JsonElement> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    [Fact]
    public void GoldenFile_HasNoSuspectCases() =>
        Assert.DoesNotContain(Cases, c => c.GetProperty("suspect").GetBoolean());

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void WarpPolar_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        if (c.GetProperty("src").GetProperty("depth").GetString() == "u8")
        {
            Run<byte>(c, name);
        }
        else
        {
            Run<float>(c, name);
        }
    }

    private static void Run<T>(JsonElement c, string name)
        where T : unmanaged, IEquatable<T>
    {
        var src = ToMat<T>(c.GetProperty("src"));
        var expected = ToMat<T>(c.GetProperty("result"));
        var dsize = c.GetProperty("dsize");
        var center = c.GetProperty("center");
        int flags = c.GetProperty("flags").GetInt32();

        var actual = Geometry.WarpPolar(src, new Size(dsize[0].GetInt32(), dsize[1].GetInt32()),
            new Point2f(center[0].GetSingle(), center[1].GetSingle()), c.GetProperty("maxRadius").GetDouble(),
            (InterpolationFlags)(flags & 7), (WarpPolarMode)(flags & 256), (WarpFlags)(flags & (8 | 16)));

        Assert.Equal((expected.Width, expected.Height, expected.Channels), (actual.Width, actual.Height, actual.Channels));
        T[] e = expected.ToArray(), a = actual.ToArray();
        int bad = Enumerable.Range(0, e.Length).Count(i => !e[i].Equals(a[i]));
        int first = Enumerable.Range(0, e.Length).FirstOrDefault(i => !e[i].Equals(a[i]));
        Assert.True(bad == 0, $"{name}: {bad} of {e.Length} differ; first at {first}: expected {e[first]}, actual {a[first]}");
    }

    private static Mat<T> ToMat<T>(JsonElement m)
        where T : unmanaged
    {
        T[] data = MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(m.GetProperty("data").GetString()!)).ToArray();
        return new Mat<T>(data, m.GetProperty("width").GetInt32(), m.GetProperty("height").GetInt32(), m.GetProperty("channels").GetInt32());
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "polar_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
