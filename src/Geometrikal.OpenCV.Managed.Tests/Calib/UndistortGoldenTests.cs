namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/undistort_golden.json</c> for the
/// undistortion functions. Every comparison is byte-exact; large outputs are compared by SHA-256.
/// Regenerate with <c>python tools/generate_undistort_golden.py</c>.
/// </summary>
public sealed class UndistortGoldenTests
{
    private static readonly string[] Ops = ["map", "undistort", "upts", "uimgpts", "defk", "optk"];

    public static IEnumerable<object[]> CaseNames()
        => UndistortGolden.Cases.Where(c => !c.Suspect && Ops.Contains(c.Op)).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(UndistortGolden.Cases.Count(c => Ops.Contains(c.Op)) > 400);

    [Fact]
    public void GoldenFile_HasNoSuspectCases() => Assert.DoesNotContain(UndistortGolden.Cases, c => c.Suspect);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Undistortion_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = UndistortGolden.Get(name);
        Mat<double> a = c.InD("a");
        double[] dist = c.Has("dist") ? c.InD("dist").ToArray() : [];
        switch (c.Op)
        {
            case "map":
            {
                var size = new Size(c.Int("width"), c.Int("height"));
                Mat<double>? r = c.Has("r") ? c.InD("r") : null;
                Mat<double>? newK = c.Has("newk") ? c.InD("newk") : null;
                switch (c.Str("mtype"))
                {
                    case "f1":
                    {
                        (Mat<float> mx, Mat<float> my) = Undistortion.InitUndistortRectifyMap(a, dist, r, newK, size);
                        c.Compare("map1", mx);
                        c.Compare("map2", my);
                        break;
                    }

                    case "f2":
                        c.Compare("map1", Undistortion.InitUndistortRectifyMapInterleaved(a, dist, r, newK, size));
                        break;
                    default:
                    {
                        (Mat<short> xy, Mat<ushort> f) = Undistortion.InitUndistortRectifyMapFixedPoint(a, dist, r, newK, size);
                        c.Compare("map1", xy);
                        c.Compare("map2", f);
                        break;
                    }
                }

                break;
            }

            case "undistort":
            {
                Mat<double>? newK = c.Has("newk") ? c.InD("newk") : null;
                if (c.InputDtype("src") == "u8")
                {
                    c.Compare("dst", Undistortion.Undistort(c.In<byte>("src"), a, dist, newK));
                }
                else
                {
                    c.Compare("dst", Undistortion.Undistort(c.In<float>("src"), a, dist, newK));
                }

                break;
            }

            case "upts":
            case "uimgpts":
            {
                Mat<double>? r = c.Has("r") ? c.InD("r") : null;
                Mat<double>? p = c.Has("p") ? c.InD("p") : null;
                TermCriteria? crit = null;
                if (c.Op == "upts" && c.Params.GetProperty("crit").GetArrayLength() == 3)
                {
                    JsonElement e = c.Params.GetProperty("crit");
                    crit = new TermCriteria((TermCriteriaType)e[0].GetInt32(), e[1].GetInt32(), e[2].GetDouble());
                }

                if (c.InputDtype("src") == "f32")
                {
                    Point2f[] src = MemoryMarshal.Cast<float, Point2f>(c.In<float>("src").ToArray()).ToArray();
                    Point2f[] dst = c.Op == "upts"
                        ? Undistortion.UndistortPoints(src, a, dist, r, p, crit)
                        : Undistortion.UndistortImagePoints(src, a, dist);
                    c.Compare("dst", new Mat<float>(MemoryMarshal.Cast<Point2f, float>(dst).ToArray(), 2, dst.Length));
                }
                else
                {
                    Point2d[] src = MemoryMarshal.Cast<double, Point2d>(c.InD("src").ToArray()).ToArray();
                    Point2d[] dst = c.Op == "upts"
                        ? Undistortion.UndistortPoints(src, a, dist, r, p, crit)
                        : Undistortion.UndistortImagePoints(src, a, dist);
                    c.Compare("dst", new Mat<double>(MemoryMarshal.Cast<Point2d, double>(dst).ToArray(), 2, dst.Length));
                }

                break;
            }

            case "defk":
                c.Compare("k", Undistortion.GetDefaultNewCameraMatrix(a, new Size(c.Int("width"), c.Int("height")), c.Params.GetProperty("center").GetBoolean()));
                break;

            case "optk":
            {
                Mat<double> k = Undistortion.GetOptimalNewCameraMatrix(
                    a,
                    dist,
                    new Size(c.Int("width"), c.Int("height")),
                    c.Params.GetProperty("alpha").GetDouble(),
                    new Size(c.Int("nw"), c.Int("nh")),
                    out Rect roi,
                    c.Params.GetProperty("center").GetBoolean());
                c.Compare("k", k);
                int[] expected = c.Params.GetProperty("roi").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                Assert.Equal(expected, new[] { roi.X, roi.Y, roi.Width, roi.Height });
                break;
            }

            default:
                throw new InvalidOperationException(c.Op);
        }
    }
}

/// <summary>Loader for <c>TestData/undistort_golden.json</c>.</summary>
internal static class UndistortGolden
{
    private static readonly Lazy<(Dictionary<string, JsonElement> Shared, List<GoldenCase> Cases)> Data = new(Load);

    public static IReadOnlyList<GoldenCase> Cases => Data.Value.Cases;

    public static GoldenCase Get(string name) => Data.Value.Cases.Single(c => c.Name == name);

    internal static JsonElement Resolve(JsonElement e)
        => e.TryGetProperty("ref", out JsonElement r) ? Data.Value.Shared[r.GetString()!] : e;

    private static (Dictionary<string, JsonElement>, List<GoldenCase>) Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "undistort_golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonElement root = doc.RootElement.Clone();
        var shared = root.GetProperty("shared").EnumerateObject().ToDictionary(p => p.Name, p => p.Value);
        var cases = root.GetProperty("cases").EnumerateArray().Select(e => new GoldenCase(e)).ToList();
        return (shared, cases);
    }
}

/// <summary>One golden case.</summary>
internal sealed class GoldenCase(JsonElement e)
{
    public string Name { get; } = e.GetProperty("name").GetString()!;

    public string Op { get; } = e.GetProperty("op").GetString()!;

    public bool Suspect { get; } = e.GetProperty("suspect").GetBoolean();

    public JsonElement Params { get; } = e.GetProperty("params");

    private JsonElement Inputs { get; } = e.GetProperty("inputs");

    private JsonElement Outputs { get; } = e.GetProperty("outputs");

    public bool Has(string input) => Inputs.TryGetProperty(input, out _);

    public int Int(string p) => Params.GetProperty(p).GetInt32();

    public string Str(string p) => Params.GetProperty(p).GetString()!;

    public string InputDtype(string input) => UndistortGolden.Resolve(Inputs.GetProperty(input)).GetProperty("dtype").GetString()!;

    public Mat<double> InD(string input) => In<double>(input);

    public Mat<T> In<T>(string input)
        where T : unmanaged
    {
        JsonElement m = UndistortGolden.Resolve(Inputs.GetProperty(input));
        byte[] raw = Convert.FromBase64String(m.GetProperty("data").GetString()!);
        T[] data = MemoryMarshal.Cast<byte, T>(raw).ToArray();
        return new Mat<T>(data, m.GetProperty("width").GetInt32(), m.GetProperty("height").GetInt32(), m.GetProperty("channels").GetInt32());
    }

    public void Compare<T>(string output, Mat<T> actual)
        where T : unmanaged
    {
        JsonElement m = Outputs.GetProperty(output);
        int w = m.GetProperty("width").GetInt32(), h = m.GetProperty("height").GetInt32(), ch = m.GetProperty("channels").GetInt32();
        Assert.True(actual.Width * actual.Channels == w * ch && actual.Height == h, $"{Name}/{output}: shape {actual.Width}x{actual.Height}x{actual.Channels}, expected {w}x{h}x{ch}");
        byte[] bytes = MemoryMarshal.AsBytes(actual.ToArray().AsSpan()).ToArray();
        if (m.TryGetProperty("sha256", out JsonElement hash))
        {
            Assert.True(hash.GetString() == Convert.ToHexStringLower(SHA256.HashData(bytes)), $"{Name}/{output}: SHA-256 differs");
            return;
        }

        byte[] expected = Convert.FromBase64String(m.GetProperty("data").GetString()!);
        if (expected.AsSpan().SequenceEqual(bytes))
        {
            return;
        }

        T[] e = MemoryMarshal.Cast<byte, T>(expected).ToArray();
        T[] a = actual.ToArray();
        int diffs = 0, first = -1;
        for (int i = 0; i < e.Length; i++)
        {
            if (!MemoryMarshal.AsBytes(e.AsSpan(i, 1)).SequenceEqual(MemoryMarshal.AsBytes(a.AsSpan(i, 1))))
            {
                diffs++;
                first = first < 0 ? i : first;
            }
        }

        Assert.Fail($"{Name}/{output}: {diffs} of {e.Length} elements differ; first at {first}: expected {e[first]}, actual {a[first]}");
    }
}
