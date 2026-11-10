namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/seam_golden.json</c> (Voronoi, DP and
/// graph-cut seam finders). The output masks must match byte for byte. Regenerate with
/// <c>tools/generate_seam_golden.py</c>.
/// </summary>
public sealed class SeamFinderGoldenTests
{
    private static readonly JsonDocument Doc = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "seam_golden.json")));

    private static JsonElement Root => Doc.RootElement;

    public static IEnumerable<object[]> CaseNames()
        => Root.GetProperty("cases").EnumerateArray()
            .Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.Equal("5.0.0", Root.GetProperty("meta").GetProperty("cv2").GetString());
        Assert.True(Root.GetProperty("cases").GetArrayLength() > 80);
    }

    /// <summary>No golden case differs between cv2 5.0.0 and the port.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.DoesNotContain(Root.GetProperty("cases").EnumerateArray(), c => c.GetProperty("suspect").GetBoolean());

    [Fact]
    public void GoldenFile_CoversEveryFinderAndType()
    {
        var keys = Root.GetProperty("cases").EnumerateArray()
            .Select(c => $"{c.GetProperty("params").GetProperty("finder").GetString()}/{c.GetProperty("dtype").GetString()}c{c.GetProperty("cn").GetInt32()}")
            .Distinct().OrderBy(s => s, StringComparer.Ordinal).ToArray();
        Assert.Equal(new[] { "dp/f32c3", "dp/f32c4", "dp/u8c3", "dp/u8c4", "graphcut/f32c3", "voronoi/f32c3" }, keys);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void SeamFinder_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Root.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("name").GetString() == name);
        JsonElement scene = Root.GetProperty("scenes")[c.GetProperty("scene").GetInt32()];
        string dtype = c.GetProperty("dtype").GetString()!;
        int cn = c.GetProperty("cn").GetInt32();

        Point[] corners = scene.GetProperty("corners").EnumerateArray()
            .Select(p => new Point(p[0].GetInt32(), p[1].GetInt32())).ToArray();
        Mat<byte>[] masks = scene.GetProperty("masks").EnumerateArray().Select(LoadMat<byte>).ToArray();
        Mat<float>[] images = scene.GetProperty("images").EnumerateArray().Select(LoadMat<float>).ToArray();
        SeamFinder finder = CreateFinder(c.GetProperty("params"));

        if (dtype == "u8")
        {
            finder.Find(images.Select(im => ToByte(WithChannels(im, cn))).ToArray(), corners, masks);
        }
        else
        {
            finder.Find(images.Select(im => WithChannels(im, cn)).ToArray(), corners, masks);
        }

        JsonElement[] expected = c.GetProperty("expected").EnumerateArray().ToArray();
        Assert.Equal(expected.Length, masks.Length);
        for (int i = 0; i < masks.Length; i++)
        {
            byte[] exp = Convert.FromBase64String(expected[i].GetProperty("data").GetString()!);
            byte[] act = masks[i].ToArray();
            if (!exp.AsSpan().SequenceEqual(act))
            {
                int k = 0;
                while (exp[k] == act[k])
                {
                    k++;
                }

                int w = masks[i].Width;
                int diffs = exp.Zip(act).Count(p => p.First != p.Second);
                Assert.Fail($"Golden mismatch for '{name}' mask {i} at ({k % w}, {k / w}); {diffs} of {exp.Length} bytes differ.");
            }
        }
    }

    internal static SeamFinder CreateFinder(JsonElement p)
    {
        string cost = p.TryGetProperty("cost", out JsonElement ce) ? ce.GetString()! : string.Empty;
        return p.GetProperty("finder").GetString() switch
        {
            "voronoi" => new VoronoiSeamFinder(),
            "dp" => new DpSeamFinder(Enum.Parse<DpSeamCostFunction>(cost)),
            _ => new GraphCutSeamFinder(
                Enum.Parse<GraphCutCostType>(cost),
                p.GetProperty("terminal").GetSingle(),
                p.GetProperty("penalty").GetSingle()),
        };
    }

    internal static Mat<T> LoadMat<T>(JsonElement e)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(e.GetProperty("data").GetString()!);
        T[] data = MemoryMarshal.Cast<byte, T>(raw).ToArray();
        return new Mat<T>(data, e.GetProperty("w").GetInt32(), e.GetProperty("h").GetInt32(), e.GetProperty("cn").GetInt32());
    }

    /// <summary>The generator's 4-channel copy: alpha 255 after the three colour channels.</summary>
    private static Mat<float> WithChannels(Mat<float> im, int cn)
    {
        if (cn == 3)
        {
            return im;
        }

        var o = new Mat<float>(im.Width, im.Height, 4);
        for (int y = 0; y < im.Height; y++)
        {
            for (int x = 0; x < im.Width; x++)
            {
                for (int k = 0; k < 3; k++)
                {
                    o.At(y, x, k) = im.At(y, x, k);
                }

                o.At(y, x, 3) = 255f;
            }
        }

        return o;
    }

    /// <summary>The generator's <c>np.clip(np.rint(im), 0, 255)</c> to uint8.</summary>
    private static Mat<byte> ToByte(Mat<float> im)
    {
        float[] src = im.ToArray();
        var dst = new byte[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = (byte)Math.Clamp(MathF.Round(src[i], MidpointRounding.ToEven), 0f, 255f);
        }

        return new Mat<byte>(dst, im.Width, im.Height, im.Channels);
    }
}
