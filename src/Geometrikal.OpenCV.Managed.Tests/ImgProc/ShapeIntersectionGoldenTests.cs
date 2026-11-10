namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/intersection_golden.json</c> through
/// <see cref="ShapeIntersection"/>. Areas, types and vertices must match bit for bit.
/// Regenerate with <c>python tools/generate_intersection_golden.py</c>.
/// </summary>
public sealed class ShapeIntersectionGoldenTests
{
    private static readonly List<JsonElement> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    /// <summary>The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences, pinned in <see cref="ShapeIntersectionOpenCv5DifferenceTests"/>.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Cases.Where(c => c.GetProperty("suspect").GetBoolean()).Select(c => c.GetProperty("name").GetString()!)
            .Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(ShapeIntersectionOpenCv5DifferenceTests.BufferCases.Concat(ShapeIntersectionOpenCv5DifferenceTests.PrecisionCases)
            .Order(StringComparer.Ordinal).ToArray(), actual);
    }

    internal static JsonElement Find(string name) => Cases.Single(x => x.GetProperty("name").GetString() == name);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Intersection_MatchesOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.GetProperty("name").GetString() == name);
        var expected = Points(c.GetProperty("result"));
        if (c.GetProperty("op").GetString() == "convex")
        {
            var p1 = Points(c.GetProperty("p1"));
            var p2 = Points(c.GetProperty("p2"));
            bool nested = c.GetProperty("nested").GetBoolean();
            var (area, region) = c.GetProperty("int").GetBoolean()
                ? ShapeIntersection.IntersectConvexConvex(ToInt(p1), ToInt(p2), nested)
                : ShapeIntersection.IntersectConvexConvex(p1, p2, nested);
            Assert.Equal((float)c.GetProperty("area").GetDouble(), area);
            Assert.Equal(expected, region);
        }
        else
        {
            var (type, region) = ShapeIntersection.RotatedRectangleIntersection(Rect(c.GetProperty("r1")), Rect(c.GetProperty("r2")));
            Assert.Equal((RectanglesIntersectType)c.GetProperty("type").GetInt32(), type);
            Assert.Equal(expected, region);
        }
    }

    internal static Point2f[] Points(JsonElement e) =>
        e.EnumerateArray().Select(p => new Point2f((float)p[0].GetDouble(), (float)p[1].GetDouble())).ToArray();

    internal static Point[] ToInt(Point2f[] p) => p.Select(v => new Point((int)v.X, (int)v.Y)).ToArray();

    internal static RotatedRect Rect(JsonElement e)
    {
        float F(int i) => (float)e[i].GetDouble();
        return new RotatedRect(new Point2f(F(0), F(1)), new Size2f(F(2), F(3)), F(4));
    }

    private static List<JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "intersection_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(e => e.Clone()).ToList();
    }
}
