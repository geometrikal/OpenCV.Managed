namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins the intersection golden cases where cv2 4.5.5 and OpenCV 5.0 differ:
/// <list type="bullet">
/// <item>The cv2 4.5.5 wheel returns the unprocessed internal buffer with the FLT_MAX sentinel for
/// touching and disjoint squares. The OpenCV 5.0 source removes the sentinel and the closing
/// duplicate; a single touching vertex gives no intersection.</item>
/// <item>PR #26974 (2025): segment intersections subtract in double (1-ulp differences).</item>
/// <item>PR #23690 (2023): rectangles are shifted by their mean centre first (1-ulp differences).</item>
/// </list>
/// </summary>
public sealed class ShapeIntersectionOpenCv5DifferenceTests
{
    internal static readonly string[] BufferCases =
        [.. from k in new[] { "disjoint", "touch_edge", "touch_vertex" } from h in new[] { "n", "x" } select $"cc_{k}_{h}"];

    internal static readonly string[] PrecisionCases =
        ["cc_14_n", "cc_14_x", .. new[] { 11, 13, 16, 25, 27, 29, 43, 45, 47, 49, 50, 59 }.Select(i => $"rr_{i}")];

    public static TheoryData<string> Buffer() => new(BufferCases);

    public static TheoryData<string> Precision() => new(PrecisionCases);

    [Theory]
    [MemberData(nameof(Buffer))]
    public void IntersectConvexConvex_DegenerateCases_ReturnTheSourceResult(string name)
    {
        var c = ShapeIntersectionGoldenTests.Find(name);
        var (area, region) = ShapeIntersection.IntersectConvexConvex(
            ShapeIntersectionGoldenTests.Points(c.GetProperty("p1")), ShapeIntersectionGoldenTests.Points(c.GetProperty("p2")), c.GetProperty("nested").GetBoolean());
        var cv2 = ShapeIntersectionGoldenTests.Points(c.GetProperty("result"));

        Assert.Equal(0f, area);
        Assert.Equal(float.MaxValue, cv2[0].X);
        Point2f[] expected = name.StartsWith("cc_touch_edge") ? [new(10, 0), new(10, 10)] : [];
        Assert.Equal(expected, region);
    }

    [Theory]
    [MemberData(nameof(Precision))]
    public void Intersection_DiffersFromCv2ByRoundingOnly(string name)
    {
        var c = ShapeIntersectionGoldenTests.Find(name);
        var expected = ShapeIntersectionGoldenTests.Points(c.GetProperty("result"));
        Point2f[] region;
        if (c.GetProperty("op").GetString() == "convex")
        {
            var p1 = ShapeIntersectionGoldenTests.Points(c.GetProperty("p1"));
            var p2 = ShapeIntersectionGoldenTests.Points(c.GetProperty("p2"));
            bool nested = c.GetProperty("nested").GetBoolean();
            float area;
            (area, region) = c.GetProperty("int").GetBoolean()
                ? ShapeIntersection.IntersectConvexConvex(ShapeIntersectionGoldenTests.ToInt(p1), ShapeIntersectionGoldenTests.ToInt(p2), nested)
                : ShapeIntersection.IntersectConvexConvex(p1, p2, nested);
            Assert.Equal(c.GetProperty("area").GetDouble(), area, 1e-5 * Math.Max(1, area));
        }
        else
        {
            RectanglesIntersectType type;
            (type, region) = ShapeIntersection.RotatedRectangleIntersection(
                ShapeIntersectionGoldenTests.Rect(c.GetProperty("r1")), ShapeIntersectionGoldenTests.Rect(c.GetProperty("r2")));
            Assert.Equal((RectanglesIntersectType)c.GetProperty("type").GetInt32(), type);
        }

        Assert.Equal(expected.Length, region.Length);
        for (int i = 0; i < region.Length; i++)
        {
            Assert.Equal(expected[i].X, region[i].X, 1e-4f);
            Assert.Equal(expected[i].Y, region[i].Y, 1e-4f);
        }

        Assert.NotEqual(expected, region);
    }
}
