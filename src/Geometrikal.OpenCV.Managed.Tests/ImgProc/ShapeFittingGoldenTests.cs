namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/shapefit_golden.json</c> through
/// <see cref="ShapeFitting"/>. Every value must be equal, except the <c>fitEllipse</c> cases with
/// 25 or more points (<c>lapack</c>): the cv2 5.0.0 wheel computes their SVD with LAPACK and the
/// port uses the OpenCV Jacobi SVD, so those cases use a relative tolerance of 1e-5.
/// Regenerate with <c>python tools/generate_shapefit_golden.py</c> and cv2 5.0.0.
/// </summary>
public sealed class ShapeFittingGoldenTests
{
    private const double LapackTolerance = 1e-5;

    internal static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.Equal("5.0.0", Data.Cv2);
        Assert.True(Data.Cases.Length > 1000);
    }

    /// <summary>The battery has no cv2 5.0.0 versus port differences.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.DoesNotContain(Data.Cases, c => c.Suspect);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void ShapeFitting_MatchesOpenCvGolden(string name)
    {
        ShapeCase c = Data.Cases.Single(x => x.Name == name);
        InputSet input = Data.Inputs[c.Input];
        bool isInt = input.Kind == "i32";
        Point[] ip = isInt ? ToPoints(input.Points) : [];
        Point2f[] fp = isInt ? [] : ToPoints2f(input.Points);
        JsonElement exp = c.Expected;
        bool expectError = exp.TryGetProperty("error", out _);

        switch (c.Op)
        {
            case "convexHull":
            {
                bool cw = c.Params.GetProperty("clockwise").GetBoolean();
                double[] actual = isInt
                    ? ShapeFitting.ConvexHull(ip, cw).SelectMany(p => new double[] { p.X, p.Y }).ToArray()
                    : ShapeFitting.ConvexHull(fp, cw).SelectMany(p => new double[] { p.X, p.Y }).ToArray();
                AssertExact(name, Doubles(exp.GetProperty("points")), actual);
                break;
            }

            case "convexHullIndices":
            {
                bool cw = c.Params.GetProperty("clockwise").GetBoolean();
                int[] actual = isInt ? ShapeFitting.ConvexHullIndices(ip, cw) : ShapeFitting.ConvexHullIndices(fp, cw);
                Assert.Equal(Ints(exp.GetProperty("indices")), actual);
                break;
            }

            case "isContourConvex":
                Assert.Equal(exp.GetProperty("value").GetBoolean(), isInt ? ShapeFitting.IsContourConvex(ip) : ShapeFitting.IsContourConvex(fp));
                break;

            case "convexityDefects":
            {
                int[] hull = Ints(c.Params.GetProperty("hull"));
                if (expectError)
                {
                    Assert.ThrowsAny<ArgumentException>(() => ShapeFitting.ConvexityDefects(ip, hull));
                    break;
                }

                int[] actual = ShapeFitting.ConvexityDefects(ip, hull)
                    .SelectMany(d => new[] { d.StartIndex, d.EndIndex, d.FarthestIndex, d.FixedPointDepth }).ToArray();
                Assert.Equal(Ints(exp.GetProperty("defects")), actual);
                break;
            }

            case "minAreaRect":
                AssertExact(name, Doubles(exp.GetProperty("rect")), Rect5(isInt ? ShapeFitting.MinAreaRect(ip) : ShapeFitting.MinAreaRect(fp)));
                break;

            case "boxPoints":
            {
                double[] r = Doubles(c.Params.GetProperty("rect"));
                var box = new RotatedRect(new Point2f((float)r[0], (float)r[1]), new Size2f((float)r[2], (float)r[3]), (float)r[4]);
                double[] actual = ShapeFitting.BoxPoints(box).SelectMany(p => new double[] { p.X, p.Y }).ToArray();
                AssertExact(name, Doubles(exp.GetProperty("points")), actual);
                break;
            }

            case "minEnclosingCircle":
            {
                EnclosingCircle circle = isInt ? ShapeFitting.MinEnclosingCircle(ip) : ShapeFitting.MinEnclosingCircle(fp);
                AssertExact(name, Doubles(exp.GetProperty("circle")), [circle.Center.X, circle.Center.Y, circle.Radius]);
                break;
            }

            case "fitEllipse":
            case "fitEllipseAMS":
            case "fitEllipseDirect":
            {
                CvRng.SetTheRngSeed(c.Params.GetProperty("seed").GetInt32());
                Func<RotatedRect> fit = (c.Op, isInt) switch
                {
                    ("fitEllipse", true) => () => ShapeFitting.FitEllipse(ip),
                    ("fitEllipse", false) => () => ShapeFitting.FitEllipse(fp),
                    ("fitEllipseAMS", true) => () => ShapeFitting.FitEllipseAms(ip),
                    ("fitEllipseAMS", false) => () => ShapeFitting.FitEllipseAms(fp),
                    ("fitEllipseDirect", true) => () => ShapeFitting.FitEllipseDirect(ip),
                    _ => () => ShapeFitting.FitEllipseDirect(fp),
                };
                if (expectError)
                {
                    Assert.ThrowsAny<Exception>(() => fit());
                    break;
                }

                double[] expected = Doubles(exp.GetProperty("rect"));
                double[] actualRect = Rect5(fit());
                if (c.Lapack)
                {
                    AssertClose(name, expected, actualRect, LapackTolerance);
                }
                else
                {
                    AssertExact(name, expected, actualRect);
                }

                break;
            }

            case "fitLine":
            {
                var dist = Enum.Parse<DistanceTypes>(c.Params.GetProperty("dist").GetString()!);
                double param = c.Params.GetProperty("param").GetDouble();
                double reps = c.Params.GetProperty("reps").GetDouble();
                double aeps = c.Params.GetProperty("aeps").GetDouble();
                FittedLine line = isInt ? ShapeFitting.FitLine(ip, dist, param, reps, aeps) : ShapeFitting.FitLine(fp, dist, param, reps, aeps);
                AssertExact(name, Doubles(exp.GetProperty("line")), [line.Vx, line.Vy, line.X0, line.Y0]);
                break;
            }

            case "pointPolygonTest":
            {
                bool md = c.Params.GetProperty("measureDist").GetBoolean();
                double[] tp = Doubles(c.Params.GetProperty("points"));
                var actual = new double[tp.Length / 2];
                for (int i = 0; i < actual.Length; i++)
                {
                    var pt = new Point2f((float)tp[2 * i], (float)tp[(2 * i) + 1]);
                    actual[i] = isInt ? ShapeFitting.PointPolygonTest(ip, pt, md) : ShapeFitting.PointPolygonTest(fp, pt, md);
                }

                AssertExact(name, Doubles(exp.GetProperty("values")), actual);
                break;
            }

            default:
                throw new InvalidOperationException(c.Op);
        }
    }

    private static double[] Rect5(RotatedRect r) => [r.Center.X, r.Center.Y, r.Size.Width, r.Size.Height, r.Angle];

    private static void AssertExact(string name, double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(expected[i].Equals(actual[i]), $"{name}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}");
        }
    }

    private static void AssertClose(string name, double[] expected, double[] actual, double rel)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tol = rel * Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"{name}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}");
        }
    }

    private static Point[] ToPoints(double[] v)
        => Enumerable.Range(0, v.Length / 2).Select(i => new Point((int)v[2 * i], (int)v[(2 * i) + 1])).ToArray();

    private static Point2f[] ToPoints2f(double[] v)
        => Enumerable.Range(0, v.Length / 2).Select(i => new Point2f((float)v[2 * i], (float)v[(2 * i) + 1])).ToArray();

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(Num).ToArray();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

    private static double Num(JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.String)
        {
            return e.GetString() switch
            {
                "NaN" => BitConverter.Int64BitsToDouble(0x7FF8000000000000),
                "Infinity" => double.PositiveInfinity,
                _ => double.NegativeInfinity,
            };
        }

        return e.GetDouble();
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "shapefit_golden.json");
        if (!File.Exists(path))
        {
            // The JSON is not in the csproj copy list yet: read it from the source tree.
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "TestData", "shapefit_golden.json")))
            {
                dir = dir.Parent;
            }

            path = dir is null ? path : Path.Combine(dir.FullName, "TestData", "shapefit_golden.json");
        }

        using FileStream stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<GoldenFile>(stream, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
    }

    internal sealed class GoldenFile
    {
        public string Cv2 { get; set; } = string.Empty;

        public InputSet[] Inputs { get; set; } = [];

        public ShapeCase[] Cases { get; set; } = [];
    }

    internal sealed class InputSet
    {
        public string Name { get; set; } = string.Empty;

        public string Kind { get; set; } = string.Empty;

        public double[] Points { get; set; } = [];
    }

    internal sealed class ShapeCase
    {
        public string Name { get; set; } = string.Empty;

        public string Op { get; set; } = string.Empty;

        public int Input { get; set; }

        public JsonElement Params { get; set; }

        public JsonElement Expected { get; set; }

        public bool Lapack { get; set; }

        public bool Suspect { get; set; }
    }
}
