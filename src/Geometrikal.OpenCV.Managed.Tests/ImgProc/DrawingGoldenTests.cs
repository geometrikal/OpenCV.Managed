namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Data-driven runner that replays the OpenCV (cv2 4.5.5) reference pixel
/// buffers captured in <c>TestData/drawing_golden.json</c> and asserts the
/// managed <see cref="Drawing"/> port reproduces them byte-for-byte.
///
/// Each case names an operation (<c>line</c>, <c>rectangle</c>, <c>circle</c>,
/// …), its arguments, and the base64-encoded raw row-major pixel buffer OpenCV
/// produced from a zero background. The runner allocates a matrix of the case's
/// element type (u8 / u16 / f32), replays the operation, and compares the raw
/// bytes. <c>drawContours</c> cases store the input mask instead of a contour
/// list: the runner re-runs the managed, OpenCV-5-exact
/// <see cref="Contours.FindContours"/> to obtain the contours, then draws.
///
/// Regenerate the golden data with <c>python tools/generate_drawing_golden.py</c>.
/// </summary>
public sealed class DrawingGoldenTests
{
    internal static readonly IReadOnlyDictionary<string, DrawingCase> AllCases = LoadCases();

    /// <summary>
    /// xUnit theory source: one row per golden case, keyed by name.
    ///
    /// Cases flagged "suspect" (cv2 4.5.5 output known to conflict with the
    /// OpenCV 5 analytic ports) are excluded here for manual adjudication
    /// rather than skipped dynamically, because xUnit 2.9.3 lacks Assert.Skip.
    /// Version-risk cases carry <c>_oob_</c> / <c>_ellipsex_</c> in their names
    /// so they are trivial to mark suspect if a conflict surfaces.
    /// </summary>
    public static IEnumerable<object[]> CaseNames()
    {
        foreach (string name in AllCases.Keys.OrderBy(n => n, StringComparer.Ordinal))
        {
            if (AllCases[name].Suspect)
            {
                continue;
            }

            yield return new object[] { name };
        }
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Draw_MatchesOpenCvGolden(string caseName)
    {
        DrawingCase c = AllCases[caseName];

        byte[] expected = Convert.FromBase64String(c.Expected);
        byte[] actual = RunCase(c);

        if (!SpansEqual(expected, actual))
        {
            Assert.Fail(BuildDiff(c, expected, actual));
        }
    }

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.NotEmpty(AllCases);
    }

    /// <summary>
    /// Visibility guard: the set of golden cases flagged "suspect" must be
    /// EXACTLY the five adjudicated cv2 4.5.5-vs-OpenCV 5 conflicts (two
    /// upstream ThickLine changes: the pre-clip margin block and the
    /// LINE_4/LINE_8 thin-line condition swap), each carrying a non-empty
    /// adjudication reason. Any new suspect case must be adjudicated (see
    /// <see cref="DrawingOpenCv5DifferenceTests"/>) and added here
    /// deliberately; this fact fails loudly otherwise.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreKnownAndAdjudicated()
    {
        string[] expected =
        {
            "circle_ellipsex_line4_r7",
            "circle_ellipsex_shift2_r5",
            "ellipse_ellipsex_a0_t1",
            "ellipse_ellipsex_a30_t1",
            "line_clip_thick_l8",
        };

        string[] actual = AllCases.Values
            .Where(c => c.Suspect)
            .Select(c => c.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(expected, actual);

        foreach (string name in expected)
        {
            Assert.False(
                string.IsNullOrWhiteSpace(AllCases[name].SuspectReason),
                $"Suspect case '{name}' must carry a non-empty SuspectReason.");
        }
    }

    // -- case execution -----------------------------------------------------

    internal static byte[] RenderActualBytes(string caseName) => RunCase(AllCases[caseName]);

    private static byte[] RunCase(DrawingCase c)
    {
        // drawContours is always u8 and needs a single-channel mask fed to the
        // managed FindContours, so it takes a dedicated non-generic path.
        if (c.Op == "drawContours")
        {
            return DrawContoursCase(c);
        }

        return c.Dtype switch
        {
            "u8" => Draw<byte>(c),
            "u16" => Draw<ushort>(c),
            "f32" => Draw<float>(c),
            _ => throw new InvalidDataException($"Unknown dtype '{c.Dtype}' in case '{c.Name}'."),
        };
    }

    private static byte[] Draw<T>(DrawingCase c)
        where T : unmanaged
    {
        var mat = new Mat<T>(c.Width, c.Height, c.Channels);
        DrawingArgs a = c.Args;

        Scalar color = ToScalar(a.Color);
        var lineType = (LineType)(a.LineType ?? 8);
        int shift = a.Shift ?? 0;
        int thickness = a.Thickness ?? 1;

        switch (c.Op)
        {
            case "line":
                Drawing.Line(mat, Pt(a.Pt1), Pt(a.Pt2), color, thickness, lineType, shift);
                break;
            case "rectangle":
                Drawing.Rectangle(mat, Pt(a.Pt1), Pt(a.Pt2), color, thickness, lineType, shift);
                break;
            case "rect2":
                Drawing.Rectangle(mat, ToRect(a.Rect!), color, thickness, lineType, shift);
                break;
            case "circle":
                Drawing.Circle(mat, Pt(a.Center), a.Radius ?? 0, color, thickness, lineType, shift);
                break;
            case "ellipse":
                Drawing.Ellipse(mat, Pt(a.Center), ToSize(a.Axes!), a.Angle ?? 0,
                    a.StartAngle ?? 0, a.EndAngle ?? 0, color, thickness, lineType, shift);
                break;
            case "fillConvexPoly":
                Drawing.FillConvexPoly<T>(mat, ToPoints(a.Points!), color, lineType, shift);
                break;
            case "fillPoly":
                Drawing.FillPoly(mat, ToPolys(a.Polygons!), color, lineType, shift, Pt(a.Offset));
                break;
            case "polylines":
                Drawing.Polylines(mat, ToPolys(a.Polylines!), a.IsClosed ?? false, color,
                    thickness, lineType, shift);
                break;
            default:
                throw new InvalidDataException($"Unknown op '{c.Op}' in case '{c.Name}'.");
        }

        return RawBytes(mat);
    }

    private static byte[] DrawContoursCase(DrawingCase c)
    {
        DrawingArgs a = c.Args;

        var mask = new Mat<byte>(c.Width, c.Height);
        for (int y = 0; y < c.Height; y++)
        {
            Span<byte> row = mask.GetRowSpan(y);
            string source = a.MaskRows![y];
            for (int x = 0; x < c.Width; x++)
            {
                row[x] = source[x] == '1' ? (byte)255 : (byte)0;
            }
        }

        ContourResult result = Contours.FindContours(
            mask, (RetrievalMode)(a.Mode ?? 0), (ContourApproximationMode)(a.Method ?? 1));

        var target = new Mat<byte>(c.Width, c.Height, c.Channels);

        // hierarchy = null + default maxLevel mirrors the generator's
        // cv2.drawContours(..., hierarchy=None) version-stable selection.
        Drawing.DrawContours(
            target,
            result.Contours,
            a.ContourIndex ?? -1,
            ToScalar(a.Color),
            a.Thickness ?? 1,
            (LineType)(a.LineType ?? 8),
            hierarchy: null,
            maxLevel: int.MaxValue,
            offset: Pt(a.Offset));

        return RawBytes(target);
    }

    // -- conversion helpers -------------------------------------------------

    private static byte[] RawBytes<T>(Mat<T> mat)
        where T : unmanaged
    {
        T[] packed = mat.ToArray();
        return MemoryMarshal.AsBytes<T>(packed.AsSpan()).ToArray();
    }

    private static Scalar ToScalar(double[]? color)
    {
        if (color is null || color.Length == 0)
        {
            return default;
        }

        double v0 = color[0];
        double v1 = color.Length > 1 ? color[1] : 0;
        double v2 = color.Length > 2 ? color[2] : 0;
        double v3 = color.Length > 3 ? color[3] : 0;
        return new Scalar(v0, v1, v2, v3);
    }

    private static Point Pt(int[]? xy) =>
        xy is null ? default : new Point(xy[0], xy[1]);

    private static Rect ToRect(int[] r) => new(r[0], r[1], r[2], r[3]);

    private static Size ToSize(int[] wh) => new(wh[0], wh[1]);

    private static Point[] ToPoints(int[][] pts)
    {
        var result = new Point[pts.Length];
        for (int i = 0; i < pts.Length; i++)
        {
            result[i] = new Point(pts[i][0], pts[i][1]);
        }

        return result;
    }

    private static Point[][] ToPolys(int[][][] polys)
    {
        var result = new Point[polys.Length][];
        for (int i = 0; i < polys.Length; i++)
        {
            result[i] = ToPoints(polys[i]);
        }

        return result;
    }

    private static bool SpansEqual(byte[] a, byte[] b)
    {
        return a.AsSpan().SequenceEqual(b);
    }

    private static string BuildDiff(DrawingCase c, byte[] expected, byte[] actual)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Drawing golden mismatch for case '{c.Name}' "
            + $"(op={c.Op}, {c.Width}x{c.Height}x{c.Channels}, dtype={c.Dtype}).");

        if (expected.Length != actual.Length)
        {
            sb.AppendLine($"Length differs: expected={expected.Length} actual={actual.Length}");
            return sb.ToString();
        }

        int diff = 0;
        int firstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
            {
                diff++;
                if (firstIdx < 0)
                {
                    firstIdx = i;
                }
            }
        }

        sb.AppendLine($"Differing bytes: {diff} of {expected.Length}");
        if (firstIdx >= 0)
        {
            int bytesPerPixel = c.Channels * (c.Dtype == "u16" ? 2 : c.Dtype == "f32" ? 4 : 1);
            int pixel = firstIdx / bytesPerPixel;
            int px = pixel % c.Width;
            int py = pixel / c.Width;
            sb.AppendLine($"First diff at byte {firstIdx} (pixel x={px}, y={py}): "
                + $"expected={expected[firstIdx]} actual={actual[firstIdx]}");
        }

        return sb.ToString();
    }

    // -- loading ------------------------------------------------------------

    private static Dictionary<string, DrawingCase> LoadCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "drawing_golden.json");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"Golden data not found at '{path}'. Run 'python tools/generate_drawing_golden.py'.",
                path);
        }

        using FileStream stream = File.OpenRead(path);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        GoldenFile file = JsonSerializer.Deserialize<GoldenFile>(stream, options)
            ?? throw new InvalidDataException("Golden data deserialized to null.");

        var dict = new Dictionary<string, DrawingCase>(file.Cases.Count);
        foreach (DrawingCase c in file.Cases)
        {
            dict[c.Name] = c;
        }

        return dict;
    }

    private sealed class GoldenFile
    {
        public List<DrawingCase> Cases { get; set; } = new();
    }

    internal sealed class DrawingCase
    {
        public string Name { get; set; } = string.Empty;

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Dtype { get; set; } = "u8";

        public string Op { get; set; } = string.Empty;

        public DrawingArgs Args { get; set; } = new();

        public string Expected { get; set; } = string.Empty;

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }
    }

    internal sealed class DrawingArgs
    {
        public int[]? Pt1 { get; set; }

        public int[]? Pt2 { get; set; }

        public int[]? Center { get; set; }

        public int[]? Axes { get; set; }

        public int[]? Rect { get; set; }

        public double[]? Color { get; set; }

        public int? Thickness { get; set; }

        public int? LineType { get; set; }

        public int? Shift { get; set; }

        public int? Radius { get; set; }

        public double? Angle { get; set; }

        public double? StartAngle { get; set; }

        public double? EndAngle { get; set; }

        public int[][]? Points { get; set; }

        public int[][][]? Polygons { get; set; }

        public int[][][]? Polylines { get; set; }

        public bool? IsClosed { get; set; }

        public int[]? Offset { get; set; }

        public string[]? MaskRows { get; set; }

        public int? Mode { get; set; }

        public int? Method { get; set; }

        public int? ContourIndex { get; set; }
    }
}
