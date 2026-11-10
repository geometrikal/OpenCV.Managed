namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/epipolar_golden.json</c> through
/// <see cref="EpipolarGeometry"/>. Masks must be identical; matrices and points must be bit-exact.
/// Regenerate with <c>python tools/generate_epipolar_golden.py</c>.
/// </summary>
public sealed class EpipolarGoldenTests
{
    private static readonly Dictionary<string, JsonElement> Cases = Load();

    public static IEnumerable<object[]> CaseNames()
        => Cases.Where(c => !c.Value.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.Key }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Cases.Count > 200);

    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Cases.Where(c => c.Value.GetProperty("suspect").GetBoolean()).Select(c => c.Key));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void EpipolarGeometry_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Cases[name];
        switch (c.GetProperty("kind").GetString())
        {
            case "essential":
                RunEssential(name, c);
                break;
            case "recover":
                RunRecover(name, c);
                break;
            case "decompose":
                RunDecompose(name, c);
                break;
            case "triangulate":
                RunTriangulate(name, c);
                break;
            case "fundamental":
                RunFundamental(name, c);
                break;
            case "epilines":
                RunEpilines(name, c);
                break;
            case "homogeneous":
                RunHomogeneous(name, c);
                break;
            default:
                Assert.Fail($"Unknown kind in {name}.");
                break;
        }
    }

    private static readonly Mat<double> K = new(new double[] { 1000, 0, 960, 0, 1000, 540, 0, 0, 1 }, 3, 3);

    private static void RunEssential(string name, JsonElement c)
    {
        var method = (RobustEstimationMethod)c.GetProperty("method").GetInt32();
        double prob = c.GetProperty("prob").GetDouble(), thr = c.GetProperty("thr").GetDouble();
        int maxIters = c.GetProperty("maxIters").GetInt32();
        bool focalPp = c.GetProperty("focalPp").GetBoolean();
        EpipolarEstimationResult r;
        if (c.GetProperty("dtype").GetString() == "f32")
        {
            Point2f[] p1 = Points<Point2f>(c, "p1"), p2 = Points<Point2f>(c, "p2");
            r = focalPp
                ? EpipolarGeometry.FindEssentialMat(p1, p2, 1000.0, new Point2d(960, 540), method, prob, thr, maxIters)
                : EpipolarGeometry.FindEssentialMat(p1, p2, K, method, prob, thr, maxIters);
        }
        else
        {
            Point2d[] p1 = Points<Point2d>(c, "p1"), p2 = Points<Point2d>(c, "p2");
            r = focalPp
                ? EpipolarGeometry.FindEssentialMat(p1, p2, 1000.0, new Point2d(960, 540), method, prob, thr, maxIters)
                : EpipolarGeometry.FindEssentialMat(p1, p2, K, method, prob, thr, maxIters);
        }

        CheckModel(name, c, r, EssentialTolerance);
    }

    private static void RunFundamental(string name, JsonElement c)
    {
        var method = (FundamentalMatMethod)c.GetProperty("method").GetInt32();
        double conf = c.GetProperty("conf").GetDouble(), thr = c.GetProperty("thr").GetDouble();
        int maxIters = c.GetProperty("maxIters").GetInt32();
        EpipolarEstimationResult r = c.GetProperty("dtype").GetString() == "f32"
            ? EpipolarGeometry.FindFundamentalMat(Points<Point2f>(c, "p1"), Points<Point2f>(c, "p2"), method, thr, conf, maxIters)
            : EpipolarGeometry.FindFundamentalMat(Points<Point2d>(c, "p1"), Points<Point2d>(c, "p2"), method, thr, conf, maxIters);
        CheckModel(name, c, r);
    }

    /// <summary>
    /// Absolute tolerance on the unit-norm essential matrices. <c>solvePoly</c> uses .NET
    /// <c>Math.Pow</c>, <c>Cos</c>, <c>Sin</c> and <c>Cbrt</c>, so the polynomial roots are not
    /// bit-exact (see <c>Docs/Notes/LinearAlgebraPortNotes.md</c>).
    /// </summary>
    private const double EssentialTolerance = 1e-10;

    private static void CheckModel(string name, JsonElement c, EpipolarEstimationResult r, double tolerance = 0)
    {
        if (!c.TryGetProperty("model", out _))
        {
            Assert.Null(r.Model);
            return;
        }

        Assert.NotNull(r.Model);
        Assert.Equal(c.GetProperty("rows").GetInt32(), r.Model!.Height);
        Assert.Equal(Bytes(c, "mask"), r.InlierMask);
        if (tolerance == 0)
        {
            AssertBits(name, Doubles(c, "model"), r.Model.ToArray());
            return;
        }

        double[] expected = Doubles(c, "model"), actual = r.Model.ToArray();
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"{name}: [{i}] = {actual[i]:R}, expected {expected[i]:R}.");
        }
    }

    private static void RunRecover(string name, JsonElement c)
    {
        var e = new Mat<double>(Doubles(c, "E"), 3, 3);
        byte[] inMask = c.TryGetProperty("inMask", out _) ? Bytes(c, "inMask") : [];
        bool hasDist = c.GetProperty("dist").ValueKind == JsonValueKind.Number;
        double dist = hasDist ? c.GetProperty("dist").GetDouble() : EpipolarGeometry.DefaultDistanceThreshold;
        RecoverPoseResult r = c.GetProperty("dtype").GetString() == "f32"
            ? EpipolarGeometry.RecoverPose(e, Points<Point2f>(c, "p1"), Points<Point2f>(c, "p2"), K, dist, inMask, hasDist)
            : EpipolarGeometry.RecoverPose(e, Points<Point2d>(c, "p1"), Points<Point2d>(c, "p2"), K, dist, inMask, hasDist);
        Assert.Equal(c.GetProperty("ret").GetInt32(), r.InlierCount);
        Assert.Equal(Bytes(c, "outMask"), r.Mask);
        AssertBits(name + " R", Doubles(c, "R"), r.R.ToArray());
        AssertBits(name + " t", Doubles(c, "t"), r.T.ToArray());
        if (hasDist)
        {
            Assert.NotNull(r.TriangulatedPoints);
            AssertBits(name + " tri", Doubles(c, "tri"), r.TriangulatedPoints!.ToArray());
        }
    }

    private static void RunDecompose(string name, JsonElement c)
    {
        (Mat<double> r1, Mat<double> r2, Mat<double> t) = EpipolarGeometry.DecomposeEssentialMat(new Mat<double>(Doubles(c, "E"), 3, 3));
        AssertBits(name + " R1", Doubles(c, "R1"), r1.ToArray());
        AssertBits(name + " R2", Doubles(c, "R2"), r2.ToArray());
        AssertBits(name + " t", Doubles(c, "t"), t.ToArray());
    }

    private static void RunTriangulate(string name, JsonElement c)
    {
        var p1m = new Mat<double>(Doubles(c, "P1"), 4, 3);
        var p2m = new Mat<double>(Doubles(c, "P2"), 4, 3);
        double[] actual;
        if (c.GetProperty("dtype").GetString() == "f32")
        {
            actual = EpipolarGeometry.TriangulatePoints(p1m, p2m, Points<Point2f>(c, "p1"), Points<Point2f>(c, "p2"))
                .ToArray().Select(v => (double)v).ToArray();
        }
        else
        {
            actual = EpipolarGeometry.TriangulatePoints(p1m, p2m, Points<Point2d>(c, "p1"), Points<Point2d>(c, "p2")).ToArray();
        }

        AssertBits(name, Doubles(c, "X"), actual);
    }

    private static void RunEpilines(string name, JsonElement c)
    {
        var f = new Mat<double>(Doubles(c, "F"), 3, 3);
        int which = c.GetProperty("which").GetInt32();
        if (c.GetProperty("dtype").GetString() == "f32")
        {
            Point3f[] lines = EpipolarGeometry.ComputeCorrespondEpilines(Points<Point2f>(c, "pts"), which, f);
            Assert.Equal(Points<Point3f>(c, "lines"), lines);
        }
        else
        {
            Point3d[] lines = EpipolarGeometry.ComputeCorrespondEpilines(Points<Point2d>(c, "pts"), which, f);
            Assert.Equal(Points<Point3d>(c, "lines"), lines);
        }
    }

    private static void RunHomogeneous(string name, JsonElement c)
    {
        if (c.GetProperty("dtype").GetString() == "f32")
        {
            Assert.Equal(Points<Point2f>(c, "from"), EpipolarGeometry.ConvertPointsFromHomogeneous(Points<Point3f>(c, "pts3")));
            Assert.Equal(Points<Point3f>(c, "to"), EpipolarGeometry.ConvertPointsToHomogeneous(Points<Point2f>(c, "pts2")));
        }
        else
        {
            Assert.Equal(Points<Point2d>(c, "from"), EpipolarGeometry.ConvertPointsFromHomogeneous(Points<Point3d>(c, "pts3")));
            Assert.Equal(Points<Point3d>(c, "to"), EpipolarGeometry.ConvertPointsToHomogeneous(Points<Point2d>(c, "pts2")));
        }
    }

    private static void AssertBits(string name, double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(expected[i]) != BitConverter.DoubleToInt64Bits(actual[i])
                && !(double.IsNaN(expected[i]) && double.IsNaN(actual[i])))
            {
                Assert.Fail($"{name}: [{i}] = {actual[i]:R}, expected {expected[i]:R} (difference {actual[i] - expected[i]:E3}).");
            }
        }
    }

    private static byte[] Bytes(JsonElement c, string key) => Convert.FromBase64String(c.GetProperty(key).GetString()!);

    private static double[] Doubles(JsonElement c, string key) => MemoryMarshal.Cast<byte, double>(Bytes(c, key)).ToArray();

    private static T[] Points<T>(JsonElement c, string key)
        where T : unmanaged
        => MemoryMarshal.Cast<byte, T>(Bytes(c, key)).ToArray();

    private static Dictionary<string, JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "epipolar_golden.json");
        using FileStream stream = File.OpenRead(path);
        JsonDocument doc = JsonDocument.Parse(stream);
        return doc.RootElement.GetProperty("cases").EnumerateArray()
            .ToDictionary(e => e.GetProperty("name").GetString()!, e => e.Clone(), StringComparer.Ordinal);
    }
}
