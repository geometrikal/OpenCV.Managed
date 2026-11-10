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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/hough_golden.json</c> through
/// <see cref="HoughTransforms"/>. Lines, segments and circles must match exactly and in order.
/// Regenerate with <c>python tools/generate_hough_golden.py</c>.
/// </summary>
public sealed class HoughGoldenTests
{
    private static readonly List<JsonElement> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    /// <summary>The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences, pinned in <see cref="HoughOpenCv5DifferenceTests"/>.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Cases.Where(c => c.GetProperty("suspect").GetBoolean()).Select(c => c.GetProperty("name").GetString()!)
            .OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.Equal(HoughOpenCv5DifferenceTests.RhoBiasCases.Append("lines_c_range").Order(StringComparer.Ordinal).ToArray(), actual);
    }

    internal static JsonElement Find(string name) => Cases.Single(x => x.GetProperty("name").GetString() == name);

    internal static Mat<byte> ImageOf(JsonElement c) => Image(c);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Hough_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        var p = c.GetProperty("params");
        double P(string key, double fallback = 0) => p.TryGetProperty(key, out var v) ? v.GetDouble() : fallback;
        int count = c.GetProperty("count").GetInt32();
        byte[] expected = Convert.FromBase64String(c.GetProperty("result").GetString()!);
        switch (c.GetProperty("op").GetString())
        {
            case "lines":
            {
                var lines = HoughTransforms.HoughLines(Image(c), P("rho"), P("theta"), (int)P("threshold"),
                    P("srn"), P("stn"), P("minTheta"), P("maxTheta", Math.PI));
                var e = MemoryMarshal.Cast<byte, float>(expected);
                Assert.Equal(count, lines.Length);
                for (int i = 0; i < count; i++)
                {
                    Assert.True(BitEqual(e[2 * i], lines[i].Rho) && BitEqual(e[(2 * i) + 1], lines[i].Theta),
                        $"{name} line {i}: expected ({e[2 * i]:R}, {e[(2 * i) + 1]:R}), actual ({lines[i].Rho:R}, {lines[i].Theta:R})");
                }

                break;
            }

            case "linesP":
            {
                var segments = HoughTransforms.HoughLinesP(Image(c), P("rho"), P("theta"), (int)P("threshold"), P("minLineLength"), P("maxLineGap"));
                var e = MemoryMarshal.Cast<byte, int>(expected);
                Assert.Equal(count, segments.Length);
                for (int i = 0; i < count; i++)
                {
                    Assert.Equal(new LineSegment(e[4 * i], e[(4 * i) + 1], e[(4 * i) + 2], e[(4 * i) + 3]), segments[i]);
                }

                break;
            }

            case "circles":
            {
                var circles = HoughTransforms.HoughCircles(Image(c), P("dp"), P("minDist"), P("param1"), P("param2"),
                    (int)P("minRadius"), (int)P("maxRadius"));
                var e = MemoryMarshal.Cast<byte, float>(expected);
                Assert.Equal(count, circles.Length);
                for (int i = 0; i < count; i++)
                {
                    Assert.True(BitEqual(e[3 * i], circles[i].X) && BitEqual(e[(3 * i) + 1], circles[i].Y) && BitEqual(e[(3 * i) + 2], circles[i].Radius),
                        $"{name} circle {i}: expected ({e[3 * i]:R}, {e[(3 * i) + 1]:R}, {e[(3 * i) + 2]:R}), actual ({circles[i].X:R}, {circles[i].Y:R}, {circles[i].Radius:R})");
                }

                break;
            }

            case "pointSet":
            {
                var pts = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(c.GetProperty("points").GetString()!));
                var points = new List<Point2f>();
                for (int i = 0; i < pts.Length; i += 2)
                {
                    points.Add(new Point2f(pts[i], pts[i + 1]));
                }

                var lines = HoughTransforms.HoughLinesPointSet(points, 20, 1, 0, 360, 1, 0, Math.PI / 2, Math.PI / 180);
                var e = MemoryMarshal.Cast<byte, double>(expected);
                Assert.Equal(count, lines.Length);
                for (int i = 0; i < count; i++)
                {
                    Assert.Equal(new HoughPointSetLine(e[3 * i], e[(3 * i) + 1], e[(3 * i) + 2]), lines[i]);
                }

                break;
            }

            default:
                throw new InvalidOperationException(c.GetProperty("op").GetString());
        }
    }

    private static bool BitEqual(float a, float b) => BitConverter.SingleToInt32Bits(a) == BitConverter.SingleToInt32Bits(b);

    private static Mat<byte> Image(JsonElement c)
    {
        var img = c.GetProperty("image");
        return new Mat<byte>(Convert.FromBase64String(img.GetProperty("data").GetString()!), img.GetProperty("width").GetInt32(), img.GetProperty("height").GetInt32());
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "hough_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
