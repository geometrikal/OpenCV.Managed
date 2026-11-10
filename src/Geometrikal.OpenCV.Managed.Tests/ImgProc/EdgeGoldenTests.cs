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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/edge_golden.json</c> (Canny image and
/// dx/dy overloads, spatialGradient) through <see cref="EdgeDetection"/>. Outputs must match
/// byte for byte. Regenerate with <c>python tools/generate_edge_golden.py</c>.
/// </summary>
public sealed class EdgeGoldenTests
{
    internal static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    /// <summary>The edge battery has no cv2 4.5.5 versus OpenCV 5.0 difference.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
        => Assert.Empty(Data.Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Edge_MatchesOpenCvGolden(string name)
    {
        EdgeCase c = Data.Cases.Single(x => x.Name == name);
        byte[][] actual = Run(c);
        Assert.Equal(c.Expected.Length, actual.Length);
        for (int k = 0; k < actual.Length; k++)
        {
            ArrayData e = Data.Arrays[c.Expected[k]];
            byte[] expected = Convert.FromBase64String(e.Data);
            if (!expected.AsSpan().SequenceEqual(actual[k]))
            {
                int diffs = 0;
                int first = -1;
                for (int i = 0; i < Math.Min(expected.Length, actual[k].Length); i++)
                {
                    if (expected[i] != actual[k][i])
                    {
                        diffs++;
                        first = first < 0 ? i : first;
                    }
                }

                Assert.Fail($"Golden mismatch for '{name}' output {k} ({e.Width}x{e.Height}x{e.Channels}): " +
                    $"{diffs} differing bytes, first at {first}; lengths {expected.Length}/{actual[k].Length}.");
            }
        }
    }

    internal static byte[][] Run(EdgeCase c)
    {
        JsonElement p = c.Params;
        switch (c.Op)
        {
            case "canny":
            {
                Mat<byte> img = Input<byte>(c, 0);
                Mat<byte> edges = EdgeDetection.Canny(
                    img, p.GetProperty("threshold1").GetDouble(), p.GetProperty("threshold2").GetDouble(),
                    p.GetProperty("apertureSize").GetInt32(), p.GetProperty("l2").GetBoolean());
                return [Raw(edges)];
            }

            case "cannyDeriv":
            {
                Mat<byte> edges = EdgeDetection.Canny(
                    Input<short>(c, 0), Input<short>(c, 1), p.GetProperty("threshold1").GetDouble(),
                    p.GetProperty("threshold2").GetDouble(), p.GetProperty("l2").GetBoolean());
                return [Raw(edges)];
            }

            case "spatialGradient":
            {
                var border = (BorderType)p.GetProperty("borderType").GetInt32();
                EdgeDetection.SpatialGradient(Input<byte>(c, 0), out Mat<short> dx, out Mat<short> dy, 3, border);
                return [Raw(dx), Raw(dy)];
            }

            default:
                throw new InvalidOperationException(c.Op);
        }
    }

    /// <summary>Builds input <paramref name="index"/>; a case with a ROI returns that view of the parent.</summary>
    internal static Mat<T> Input<T>(EdgeCase c, int index)
        where T : unmanaged
    {
        ArrayData a = Data.Arrays[c.Inputs[index]];
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[raw.Length / Marshal.SizeOf<T>()];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        var mat = new Mat<T>(buffer, a.Width, a.Height, a.Channels);
        return c.Roi is { Length: 4 } r ? mat.Roi(r[0], r[1], r[2], r[3]) : mat;
    }

    private static byte[] Raw<T>(Mat<T> m)
        where T : unmanaged
        => MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray();

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "edge_golden.json");
        using FileStream stream = File.OpenRead(path);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        return JsonSerializer.Deserialize<GoldenFile>(stream, options) ?? throw new InvalidOperationException("empty golden file");
    }

    internal sealed class GoldenFile
    {
        public List<ArrayData> Arrays { get; set; } = [];

        public List<EdgeCase> Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class EdgeCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int[] Inputs { get; set; } = [];

        public JsonElement Params { get; set; }

        public int[] Expected { get; set; } = [];

        public int[]? Roi { get; set; }

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }
    }
}
