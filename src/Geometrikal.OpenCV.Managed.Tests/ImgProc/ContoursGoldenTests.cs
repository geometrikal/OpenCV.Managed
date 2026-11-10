namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Data-driven runner that replays the OpenCV (cv2 4.5.5) reference outputs
/// captured in <c>TestData/contours_golden.json</c> and asserts the managed
/// <see cref="Contours.FindContours"/> port reproduces them exactly: same
/// contour count, same point sequences (order and value), same hierarchy.
///
/// Regenerate the golden data with <c>python tools/generate_golden.py</c>.
/// </summary>
public sealed class ContoursGoldenTests
{
    internal static readonly IReadOnlyDictionary<string, GoldenCase> AllCases = LoadCases();

    /// <summary>
    /// xUnit theory source: one row per golden case, keyed by name.
    ///
    /// Cases flagged "suspect" (cv2 4.5.5 output known to conflict with the
    /// OpenCV 5 analytic ports) are excluded here for manual adjudication
    /// rather than skipped dynamically, because xUnit 2.9.3 lacks Assert.Skip.
    /// </summary>
    public static IEnumerable<object[]> CaseNames()
    {
        foreach (string name in AllCases.Keys.OrderBy(n => n, System.StringComparer.Ordinal))
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
    public void FindContours_MatchesOpenCvGolden(string caseName)
    {
        GoldenCase c = AllCases[caseName];

        Mat<byte> image = BuildImage(c);

        var mode = (RetrievalMode)c.Mode;
        var method = (ContourApproximationMode)c.Method;
        var offset = new Point(c.OffsetX, c.OffsetY);

        ContourResult result = Contours.FindContours(image, mode, method, offset);

        AssertMatches(caseName, c, result);
    }

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.NotEmpty(AllCases);
    }

    /// <summary>
    /// Visibility guard so suspect cases can never silently rot: the set of
    /// golden cases flagged "suspect" must be EXACTLY the two adjudicated
    /// OpenCV 5-vs-legacy TC89_L1 conflicts, each carrying a non-empty
    /// adjudication reason. Any new suspect case must be adjudicated (see
    /// <see cref="ContoursOpenCv5DifferenceTests"/>) and added here
    /// deliberately; this fact fails loudly otherwise.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreKnownAndAdjudicated()
    {
        string[] expected =
        {
            "rand_64_p50_list_tc89l1",
            "rand_64_p50_tree_tc89l1",
        };

        string[] actual = AllCases.Values
            .Where(c => c.Suspect)
            .Select(c => c.Name)
            .OrderBy(n => n, System.StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(expected, actual);

        foreach (string name in expected)
        {
            Assert.False(
                string.IsNullOrWhiteSpace(AllCases[name].SuspectReason),
                $"Suspect case '{name}' must carry a non-empty SuspectReason.");
        }
    }

    private static void AssertMatches(string caseName, GoldenCase expected, ContourResult actual)
    {
        bool ok = expected.Contours.Count == actual.Contours.Count
            && expected.Hierarchy.Count == actual.Hierarchy.Count;

        if (ok)
        {
            for (int i = 0; i < expected.Contours.Count && ok; i++)
            {
                IReadOnlyList<int[]> ec = ConvertPoints(expected.Contours[i]);
                Point[] ac = actual.Contours[i];
                if (ec.Count != ac.Length)
                {
                    ok = false;
                    break;
                }

                for (int j = 0; j < ec.Count; j++)
                {
                    if (ec[j][0] != ac[j].X || ec[j][1] != ac[j].Y)
                    {
                        ok = false;
                        break;
                    }
                }
            }
        }

        if (ok)
        {
            for (int i = 0; i < expected.Hierarchy.Count; i++)
            {
                List<int> eh = expected.Hierarchy[i];
                ContourHierarchy ah = actual.Hierarchy[i];
                if (eh[0] != ah.Next || eh[1] != ah.Previous
                    || eh[2] != ah.FirstChild || eh[3] != ah.Parent)
                {
                    ok = false;
                    break;
                }
            }
        }

        if (!ok)
        {
            Assert.Fail(BuildDiff(caseName, expected, actual));
        }
    }

    private static IReadOnlyList<int[]> ConvertPoints(List<List<int>> pts)
    {
        var arr = new int[pts.Count][];
        for (int i = 0; i < pts.Count; i++)
        {
            arr[i] = new[] { pts[i][0], pts[i][1] };
        }

        return arr;
    }

    private static string BuildDiff(string caseName, GoldenCase expected, ContourResult actual)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Golden mismatch for case '{caseName}' "
            + $"(mode={expected.Mode}, method={expected.Method}, "
            + $"offset=({expected.OffsetX},{expected.OffsetY}), "
            + $"{expected.Width}x{expected.Height}).");
        sb.AppendLine($"Contour count  expected={expected.Contours.Count} actual={actual.Contours.Count}");
        sb.AppendLine($"Hierarchy count expected={expected.Hierarchy.Count} actual={actual.Hierarchy.Count}");

        sb.AppendLine("-- image rows --");
        foreach (string row in expected.Rows)
        {
            sb.AppendLine("  " + row);
        }

        int n = System.Math.Max(expected.Contours.Count, actual.Contours.Count);
        for (int i = 0; i < n; i++)
        {
            sb.AppendLine($"[contour {i}]");
            sb.AppendLine("  expected: " + (i < expected.Contours.Count
                ? DumpExpectedPoints(expected.Contours[i])
                : "<missing>"));
            sb.AppendLine("  actual:   " + (i < actual.Contours.Count
                ? DumpActualPoints(actual.Contours[i])
                : "<missing>"));
        }

        int hn = System.Math.Max(expected.Hierarchy.Count, actual.Hierarchy.Count);
        sb.AppendLine("-- hierarchy [next,prev,child,parent] --");
        for (int i = 0; i < hn; i++)
        {
            string e = i < expected.Hierarchy.Count
                ? $"[{string.Join(",", expected.Hierarchy[i])}]"
                : "<missing>";
            ContourHierarchy a = i < actual.Hierarchy.Count ? actual.Hierarchy[i] : default;
            string astr = i < actual.Hierarchy.Count
                ? $"[{a.Next},{a.Previous},{a.FirstChild},{a.Parent}]"
                : "<missing>";
            sb.AppendLine($"  {i}: expected={e} actual={astr}");
        }

        return sb.ToString();
    }

    private static string DumpExpectedPoints(List<List<int>> pts)
    {
        var sb = new StringBuilder();
        sb.Append('(').Append(pts.Count).Append(") ");
        foreach (List<int> p in pts)
        {
            sb.Append('[').Append(p[0]).Append(',').Append(p[1]).Append("] ");
        }

        return sb.ToString();
    }

    private static string DumpActualPoints(Point[] pts)
    {
        var sb = new StringBuilder();
        sb.Append('(').Append(pts.Length).Append(") ");
        foreach (Point p in pts)
        {
            sb.Append('[').Append(p.X).Append(',').Append(p.Y).Append("] ");
        }

        return sb.ToString();
    }

    internal static Mat<byte> BuildImage(GoldenCase c)
    {
        var mat = new Mat<byte>(c.Width, c.Height);
        for (int y = 0; y < c.Height; y++)
        {
            Span<byte> row = mat.GetRowSpan(y);
            string source = c.Rows[y];
            for (int x = 0; x < c.Width; x++)
            {
                row[x] = source[x] == '1' ? (byte)255 : (byte)0;
            }
        }

        return mat;
    }

    private static Dictionary<string, GoldenCase> LoadCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "contours_golden.json");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"Golden data not found at '{path}'. Run 'python tools/generate_golden.py'.",
                path);
        }

        using FileStream stream = File.OpenRead(path);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        GoldenFile file = JsonSerializer.Deserialize<GoldenFile>(stream, options)
            ?? throw new InvalidDataException("Golden data deserialized to null.");

        var dict = new Dictionary<string, GoldenCase>(file.Cases.Count);
        foreach (GoldenCase c in file.Cases)
        {
            dict[c.Name] = c;
        }

        return dict;
    }

    private sealed class GoldenFile
    {
        public List<GoldenCase> Cases { get; set; } = new();
    }

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = string.Empty;

        public int Width { get; set; }

        public int Height { get; set; }

        public List<string> Rows { get; set; } = new();

        public int Mode { get; set; }

        public int Method { get; set; }

        public int OffsetX { get; set; }

        public int OffsetY { get; set; }

        public List<List<List<int>>> Contours { get; set; } = new();

        public List<List<int>> Hierarchy { get; set; } = new();

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }
    }
}
