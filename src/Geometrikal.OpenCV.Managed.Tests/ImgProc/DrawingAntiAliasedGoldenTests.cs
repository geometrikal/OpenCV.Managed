namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 cases in <c>TestData/drawing_aa_golden.json</c> through
/// <see cref="Drawing"/> and compares the raw pixel buffers byte for byte.
/// Regenerate the data with <c>tools/generate_drawing_aa_golden.py</c>.
/// </summary>
public sealed class DrawingAntiAliasedGoldenTests
{
    private static readonly Lazy<JsonDocument> Golden = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "drawing_aa_golden.json"))));

    internal static JsonElement Root => Golden.Value.RootElement;

    public static IEnumerable<object[]> CaseNames() =>
        Root.GetProperty("cases").EnumerateArray().Select(c => new object[] { c.GetProperty("name").GetString()! });

    [Fact]
    public void GoldenFile_IsOpenCv500AndNonEmpty()
    {
        Assert.Equal("5.0.0", Root.GetProperty("opencv").GetString());
        Assert.True(Root.GetProperty("cases").GetArrayLength() > 200);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Draw_MatchesOpenCvGolden(string caseName)
    {
        JsonElement c = Root.GetProperty("cases").EnumerateArray()
            .First(e => e.GetProperty("name").GetString() == caseName);

        byte[] actual = c.GetProperty("dtype").GetString() switch
        {
            "u8" => Run<byte>(c),
            "u16" => Run<ushort>(c),
            "f32" => Run<float>(c),
            _ => throw new InvalidOperationException("unknown dtype"),
        };

        if (c.TryGetProperty("expected", out JsonElement exp))
        {
            byte[] expected = Convert.FromBase64String(exp.GetString()!);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                if (expected[i] != actual[i])
                {
                    Assert.Fail($"{caseName}: first mismatch at byte {i}: expected {expected[i]}, actual {actual[i]}");
                }
            }
        }
        else
        {
            string hash = Convert.ToHexString(SHA256.HashData(actual)).ToLowerInvariant();
            Assert.Equal(c.GetProperty("sha256").GetString(), hash);
        }
    }

    internal static byte[] Run<T>(JsonElement c)
        where T : unmanaged
    {
        int w = c.GetProperty("width").GetInt32();
        int h = c.GetProperty("height").GetInt32();
        int ch = c.GetProperty("channels").GetInt32();
        var img = new Mat<T>(w, h, ch);
        if (c.GetProperty("background").GetString() == "pattern")
        {
            for (int y = 0; y < h; y++)
            {
                Span<T> row = img.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    for (int k = 0; k < ch; k++)
                    {
                        row[(x * ch) + k] = FromInt<T>(((x * 7) + (y * 13) + (k * 29) + 11) & 255);
                    }
                }
            }
        }

        Execute(img, c.GetProperty("op").GetString()!, c.GetProperty("args"));
        return MemoryMarshal.AsBytes(img.ToArray().AsSpan()).ToArray();
    }

    private static T FromInt<T>(int v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return (T)(object)(byte)v;
        }

        if (typeof(T) == typeof(ushort))
        {
            return (T)(object)(ushort)v;
        }

        return (T)(object)(float)v;
    }

    private static void Execute<T>(Mat<T> img, string op, JsonElement a)
        where T : unmanaged
    {
        Scalar color = ColorOf(a);
        LineType lt = a.TryGetProperty("lineType", out JsonElement l) ? (LineType)l.GetInt32() : LineType.EightConnected;
        int shift = a.TryGetProperty("shift", out JsonElement s) ? s.GetInt32() : 0;
        int thickness = a.TryGetProperty("thickness", out JsonElement t) ? t.GetInt32() : 1;

        switch (op)
        {
            case "line":
                Drawing.Line(img, Pt(a, "p1"), Pt(a, "p2"), color, thickness, lt, shift);
                break;
            case "rectangle":
                Drawing.Rectangle(img, Pt(a, "p1"), Pt(a, "p2"), color, thickness, lt, shift);
                break;
            case "rectangle_rect":
                int[] r = a.GetProperty("rect").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                Drawing.Rectangle(img, new Rect(r[0], r[1], r[2], r[3]), color, thickness, lt, shift);
                break;
            case "circle":
                Drawing.Circle(img, Pt(a, "center"), a.GetProperty("radius").GetInt32(), color, thickness, lt, shift);
                break;
            case "ellipse":
                Point ax = Pt(a, "axes");
                Drawing.Ellipse(img, Pt(a, "center"), new Size(ax.X, ax.Y), a.GetProperty("angle").GetDouble(),
                    a.GetProperty("start").GetDouble(), a.GetProperty("end").GetDouble(), color, thickness, lt, shift);
                break;
            case "fillConvexPoly":
                Drawing.FillConvexPoly<T>(img, Pts(a.GetProperty("points")), color, lt, shift);
                break;
            case "fillPoly":
                Drawing.FillPoly(img, Polys(a), color, lt, shift, Pt(a, "offset"));
                break;
            case "polylines":
                Drawing.Polylines(img, Polys(a), a.GetProperty("closed").GetBoolean(), color, thickness, lt, shift);
                break;
            case "drawContours":
                Drawing.DrawContours(img, Polys(a), a.GetProperty("index").GetInt32(), color, thickness, lt,
                    offset: Pt(a, "offset"));
                break;
            case "arrowedLine":
                Drawing.ArrowedLine(img, Pt(a, "p1"), Pt(a, "p2"), color, thickness, lt, shift, a.GetProperty("tip").GetDouble());
                break;
            case "drawMarker":
                Drawing.DrawMarker(img, Pt(a, "position"), color, (MarkerType)a.GetProperty("markerType").GetInt32(),
                    a.GetProperty("markerSize").GetInt32(), thickness, lt);
                break;
            default:
                throw new InvalidOperationException(op);
        }
    }

    private static Scalar ColorOf(JsonElement a)
    {
        double[] v = a.GetProperty("color").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        return new Scalar(v[0], v[1], v[2], v[3]);
    }

    private static Point Pt(JsonElement a, string name)
    {
        JsonElement p = a.GetProperty(name);
        return new Point(p[0].GetInt32(), p[1].GetInt32());
    }

    private static Point[] Pts(JsonElement arr) =>
        arr.EnumerateArray().Select(p => new Point(p[0].GetInt32(), p[1].GetInt32())).ToArray();

    private static Point[][] Polys(JsonElement a) =>
        a.GetProperty("polys").EnumerateArray().Select(Pts).ToArray();
}
