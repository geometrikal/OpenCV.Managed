namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Replays the cv2 5.0.0 ORB reference outputs in <c>TestData/orb_golden.json</c>. Keypoint sets
/// are compared sorted by (octave, y, x) because OpenCV orders each level with
/// <c>std::nth_element</c>; every member must match bit for bit, and each descriptor must match the
/// descriptor of its keypoint. <c>compute</c> cases compare the output order exactly. Regenerate
/// with <c>tools/generate_orb_golden.py</c>.
/// </summary>
public sealed class OrbGoldenTests
{
    public static IEnumerable<object[]> OrbCases()
        => OrbGolden.Cases.Where(c => !c.Suspect && c.Op == "orb").Select(c => new object[] { c.Name });

    public static IEnumerable<object[]> ComputeCases()
        => OrbGolden.Cases.Where(c => !c.Suspect && c.Op == "orbCompute").Select(c => new object[] { c.Name });

    [Fact]
    public void GoldenFile_LoadsAndCoversEveryOperation()
    {
        Assert.Equal(new[] { "fast", "orb", "orbCompute" }, OrbGolden.Cases.Select(c => c.Op).Distinct().OrderBy(o => o, StringComparer.Ordinal));
        Assert.True(OrbGolden.Cases.Count(c => c.Op == "orb") > 40);
        Assert.Contains(OrbGolden.Cases, c => c.Op == "orb" && c.Params.GetProperty("wtaK").GetInt32() == 3);
        Assert.Contains(OrbGolden.Cases, c => c.Op == "orb" && c.Params.GetProperty("wtaK").GetInt32() == 4);
        Assert.Contains(OrbGolden.Cases, c => c.Op == "orb" && c.Params.GetProperty("scoreType").GetString() == "Fast");
        Assert.Contains(OrbGolden.Cases, c => c.Op == "orb" && c.Mask is not null);
    }

    /// <summary>No golden case differs between cv2 5.0.0 and the port.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(OrbGolden.Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Theory]
    [MemberData(nameof(OrbCases))]
    public void DetectAndCompute_MatchesOpenCvGolden(string name)
    {
        OrbGolden.Case c = OrbGolden.Cases.Single(x => x.Name == name);
        Mat<byte> image = OrbGolden.BuildImage(c.Image);
        Mat<byte>? mask = OrbGolden.BuildMask(c.Mask, image.Width, image.Height);
        Orb orb = OrbGolden.CreateOrb(c.Params);

        (KeyPoint[] kps, Mat<byte>? desc) = orb.DetectAndCompute(image, mask);
        float[] expRows = OrbGolden.Floats(c.Keypoints!);
        byte[] expDesc = Convert.FromBase64String(c.Descriptors!);
        Assert.Equal(c.Count, kps.Length);
        Assert.Equal(c.Count == 0, desc is null);

        int[] expOrder = SortedOrder(expRows, c.Count);
        int[] actOrder = SortedOrder(OrbGolden.Rows(kps), kps.Length);
        for (int i = 0; i < c.Count; i++)
        {
            int e = expOrder[i];
            int a = actOrder[i];
            AssertKeyPoint(name, expRows, e, kps[a]);
            Assert.True(
                expDesc.AsSpan(e * 32, 32).SequenceEqual(desc!.GetReadOnlyRowSpan(a)),
                $"{name}: descriptor mismatch for keypoint ({kps[a].Pt.X}, {kps[a].Pt.Y}) octave {kps[a].Octave}.");
        }

        KeyPoint[] detected = orb.Detect(image, mask);
        Assert.Equal(kps, detected);
    }

    [Theory]
    [MemberData(nameof(ComputeCases))]
    public void Compute_MatchesOpenCvGolden(string name)
    {
        OrbGolden.Case c = OrbGolden.Cases.Single(x => x.Name == name);
        Mat<byte> image = OrbGolden.BuildImage(c.Image);
        Orb orb = OrbGolden.CreateOrb(c.Params);
        float[] inRows = OrbGolden.Floats(c.InputKeypoints!);
        var input = new KeyPoint[c.InputCount];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = new KeyPoint(inRows[6 * i], inRows[(6 * i) + 1], inRows[(6 * i) + 2], inRows[(6 * i) + 3], inRows[(6 * i) + 4], (int)inRows[(6 * i) + 5]);
        }

        (KeyPoint[] kps, Mat<byte>? desc) = orb.Compute(image, input);
        float[] expRows = OrbGolden.Floats(c.Keypoints!);
        byte[] expDesc = Convert.FromBase64String(c.Descriptors!);
        Assert.Equal(c.Count, kps.Length);
        for (int i = 0; i < c.Count; i++)
        {
            AssertKeyPoint(name, expRows, i, kps[i]);
            Assert.True(expDesc.AsSpan(i * 32, 32).SequenceEqual(desc!.GetReadOnlyRowSpan(i)), $"{name}: descriptor {i} mismatch.");
        }
    }

    private static void AssertKeyPoint(string name, float[] rows, int e, KeyPoint k)
    {
        float[] act = { k.Pt.X, k.Pt.Y, k.Size, k.Angle, k.Response, k.Octave };
        for (int f = 0; f < 6; f++)
        {
            float exp = rows[(6 * e) + f];
            Assert.True(
                BitConverter.SingleToInt32Bits(exp) == BitConverter.SingleToInt32Bits(act[f]),
                $"{name}: keypoint field {f} expected {exp:R} actual {act[f]:R} at ({k.Pt.X}, {k.Pt.Y}) octave {k.Octave}.");
        }
    }

    private static int[] SortedOrder(float[] rows, int count)
    {
        int[] order = Enumerable.Range(0, count).ToArray();
        Array.Sort(order, (i, j) =>
        {
            int c = rows[(6 * i) + 5].CompareTo(rows[(6 * j) + 5]);
            if (c == 0)
            {
                c = rows[(6 * i) + 1].CompareTo(rows[(6 * j) + 1]);
            }

            return c != 0 ? c : rows[6 * i].CompareTo(rows[6 * j]);
        });
        return order;
    }
}

/// <summary>Loader and image recipes of <c>TestData/orb_golden.json</c>.</summary>
internal static class OrbGolden
{
    internal static readonly List<Case> Cases = Load();

    internal sealed record Case(
        string Name,
        string Op,
        bool Suspect,
        JsonElement Image,
        JsonElement? Mask,
        JsonElement Params,
        int Count,
        string? Keypoints,
        string? KeypointsSha,
        string? Descriptors,
        int InputCount,
        string? InputKeypoints);

    internal static float[] Floats(string b64) => MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(b64)).ToArray();

    internal static float[] Rows(KeyPoint[] kps)
    {
        float[] rows = new float[kps.Length * 6];
        for (int i = 0; i < kps.Length; i++)
        {
            rows[6 * i] = kps[i].Pt.X;
            rows[(6 * i) + 1] = kps[i].Pt.Y;
            rows[(6 * i) + 2] = kps[i].Size;
            rows[(6 * i) + 3] = kps[i].Angle;
            rows[(6 * i) + 4] = kps[i].Response;
            rows[(6 * i) + 5] = kps[i].Octave;
        }

        return rows;
    }

    internal static Orb CreateOrb(JsonElement p) => new(
        p.GetProperty("nfeatures").GetInt32(),
        (float)p.GetProperty("scaleFactor").GetDouble(),
        p.GetProperty("nlevels").GetInt32(),
        p.GetProperty("edgeThreshold").GetInt32(),
        p.GetProperty("firstLevel").GetInt32(),
        p.GetProperty("wtaK").GetInt32(),
        Enum.Parse<OrbScoreType>(p.GetProperty("scoreType").GetString()!),
        p.GetProperty("patchSize").GetInt32(),
        p.GetProperty("fastThreshold").GetInt32());

    internal static uint Hash(uint x, uint y)
    {
        uint h = unchecked((x * 73856093u) ^ (y * 19349663u));
        h ^= h >> 13;
        h = unchecked(h * 0x5BD1E995u);
        h ^= h >> 15;
        return h;
    }

    /// <summary>Rebuilds an image recipe of the generator (<c>build_image</c>).</summary>
    internal static Mat<byte> BuildImage(JsonElement spec)
    {
        int w = spec.GetProperty("width").GetInt32();
        int h = spec.GetProperty("height").GetInt32();
        int seed = spec.GetProperty("seed").GetInt32();
        string kind = spec.GetProperty("kind").GetString()!;
        long[] img = new long[w * h];
        if (kind == "noise")
        {
            int r = spec.GetProperty("radius").GetInt32();
            long gain = spec.GetProperty("gain").GetInt64();
            int k = (2 * r) + 1;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    long s = 0;
                    for (int dy = 0; dy < k; dy++)
                    {
                        for (int dx = 0; dx < k; dx++)
                        {
                            s += Hash(unchecked((uint)(x + dx) + (uint)(131 * seed)), (uint)(y + dy)) & 255u;
                        }
                    }

                    long avg = s / (k * k);
                    img[(y * w) + x] = Math.Clamp(128 + ((avg - 128) * gain), 0, 255);
                }
            }
        }
        else if (kind == "checker")
        {
            int sq = spec.GetProperty("square").GetInt32();
            long lo = spec.GetProperty("low").GetInt64();
            long hi = spec.GetProperty("high").GetInt64();
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    img[(y * w) + x] = (((x / sq) + (y / sq)) % 2) == 0 ? lo : hi;
                }
            }
        }
        else
        {
            Array.Fill(img, spec.GetProperty("background").GetInt64());
        }

        if (spec.TryGetProperty("shapes", out JsonElement shapes))
        {
            Paint(img, w, h, shapes);
        }

        int jitter = spec.TryGetProperty("jitter", out JsonElement j) ? j.GetInt32() : 0;
        if (jitter > 0)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    long hv = Hash((uint)x + 977u, unchecked((uint)y + (uint)(7 * seed))) % (uint)((2 * jitter) + 1);
                    img[(y * w) + x] = Math.Clamp(img[(y * w) + x] + hv - jitter, 0, 255);
                }
            }
        }

        var mat = new Mat<byte>(w, h);
        for (int i = 0; i < img.Length; i++)
        {
            mat.GetSpan()[i] = (byte)img[i];
        }

        return mat;
    }

    /// <summary>Rebuilds a mask recipe of the generator (<c>build_mask</c>).</summary>
    internal static Mat<byte>? BuildMask(JsonElement? spec, int w, int h)
    {
        if (spec is null)
        {
            return null;
        }

        long[] m = new long[w * h];
        Paint(m, w, h, spec.Value);
        var mat = new Mat<byte>(w, h);
        for (int i = 0; i < m.Length; i++)
        {
            mat.GetSpan()[i] = (byte)m[i];
        }

        return mat;
    }

    private static void Paint(long[] img, int w, int h, JsonElement shapes)
    {
        foreach (JsonElement s in shapes.EnumerateArray())
        {
            int[] v = s.EnumerateArray().Select(e => e.GetInt32()).ToArray();
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    bool inside = v[0] == 0
                        ? x >= v[1] && x < v[3] && y >= v[2] && y < v[4]
                        : ((long)(x - v[1]) * (x - v[1])) + ((long)(y - v[2]) * (y - v[2])) <= (long)v[3] * v[3];
                    if (inside)
                    {
                        img[(y * w) + x] = v[0] == 0 ? v[5] : v[4];
                    }
                }
            }
        }
    }

    private static List<Case> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "orb_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var cases = new List<Case>();
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            JsonElement mask = c.GetProperty("mask");
            cases.Add(new Case(
                c.GetProperty("name").GetString()!,
                c.GetProperty("op").GetString()!,
                c.GetProperty("suspect").GetBoolean(),
                c.GetProperty("image").Clone(),
                mask.ValueKind == JsonValueKind.Null ? null : mask.Clone(),
                c.GetProperty("params").Clone(),
                c.GetProperty("count").GetInt32(),
                c.TryGetProperty("keypoints", out JsonElement k) ? k.GetString() : null,
                c.TryGetProperty("keypointsSha", out JsonElement ks) ? ks.GetString() : null,
                c.TryGetProperty("descriptors", out JsonElement d) ? d.GetString() : null,
                c.TryGetProperty("inputCount", out JsonElement ic) ? ic.GetInt32() : 0,
                c.TryGetProperty("inputKeypoints", out JsonElement ik) ? ik.GetString() : null));
        }

        return cases;
    }
}
