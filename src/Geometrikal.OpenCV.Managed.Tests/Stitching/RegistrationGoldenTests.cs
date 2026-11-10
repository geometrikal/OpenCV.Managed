namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/stitch_reg_golden.json</c> through the
/// stitching registration port. Regenerate with <c>python tools/generate_stitch_reg_golden.py</c>.
/// </summary>
/// <remarks>
/// The matcher cases compare with a Python reference of <c>matchers.cpp</c> that uses
/// <c>cv2.BFMatcher</c> instead of FLANN; pairs where the cv2 FLANN matches are equal to the
/// reference are also compared with cv2. The estimator, bundle adjuster and wave correction cases
/// start from the stored cv2 inputs of each stage.
/// </remarks>
public sealed class RegistrationGoldenTests
{
    private static readonly Golden Data = Load();

    public static IEnumerable<object[]> MatcherCaseNames() => Data.MatcherCases.Select(c => new object[] { c.GetProperty("name").GetString()! });

    public static IEnumerable<object[]> EstimatorCaseNames() => Data.EstimatorCases.Select(c => new object[] { c.GetProperty("name").GetString()! });

    public static IEnumerable<object[]> FocalCaseNames() => Data.FocalCases.Select(c => new object[] { c.GetProperty("name").GetString()! });

    public static IEnumerable<object[]> ComponentCaseNames() => Data.ComponentCases.Select(c => new object[] { c.GetProperty("name").GetString()! });

    public static IEnumerable<object[]> CalibCaseNames() => Data.CalibCases.Select(c => new object[] { c.GetProperty("name").GetString()! });

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.True(Data.MatcherCases.Count >= 10);
        Assert.True(Data.EstimatorCases.Count >= 5);
        Assert.True(Data.FocalCases.Count >= 10);
    }

    [Theory]
    [MemberData(nameof(MatcherCaseNames))]
    public void Matcher_MatchesBfReference(string name)
    {
        JsonElement c = Find(Data.MatcherCases, name);
        MatchesInfo[] actual = RunMatcher(c);
        JsonElement[] reference = c.GetProperty("reference").EnumerateArray().ToArray();
        Assert.Equal(reference.Length, actual.Length);
        for (int k = 0; k < actual.Length; k++)
        {
            AssertMatchesInfo(reference[k], actual[k], $"{name}[{k}]");
        }
    }

    [Theory]
    [MemberData(nameof(MatcherCaseNames))]
    public void Matcher_MatchesCv2WherePairMatchesAgree(string name)
    {
        JsonElement c = Find(Data.MatcherCases, name);
        MatchesInfo[] actual = RunMatcher(c);
        JsonElement[] cv2 = c.GetProperty("cv2").EnumerateArray().ToArray();
        for (int k = 0; k < actual.Length; k++)
        {
            int[] q = Ints(cv2[k], "query"), t = Ints(cv2[k], "train");
            if (q.SequenceEqual(actual[k].Matches.Select(m => m.QueryIdx)) && t.SequenceEqual(actual[k].Matches.Select(m => m.TrainIdx)))
            {
                AssertMatchesInfo(cv2[k], actual[k], $"{name}[{k}] (cv2)");
            }
        }
    }

    [Theory]
    [MemberData(nameof(EstimatorCaseNames))]
    public void Estimator_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Find(Data.EstimatorCases, name);
        (ImageFeatures[] features, MatchesInfo[] pairwise) = EstimatorInput(c);
        Estimator est = c.GetProperty("estimator").GetString() == "homography"
            ? new HomographyBasedEstimator()
            : new AffineBasedEstimator();
        var cameras = new List<CameraParams>();
        bool ok = est.Estimate(features, pairwise, cameras);
        Assert.Equal(c.GetProperty("ok").GetBoolean(), ok);
        AssertCameras(c.GetProperty("cameras"), cameras, 0, $"{name} estimator");
    }

    [Theory]
    [MemberData(nameof(EstimatorCaseNames))]
    public void BundleAdjuster_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Find(Data.EstimatorCases, name);
        (ImageFeatures[] features, MatchesInfo[] pairwise) = EstimatorInput(c);
        foreach (JsonElement adj in c.GetProperty("adjusters").EnumerateArray())
        {
            string kind = adj.GetProperty("kind").GetString()!;
            BundleAdjusterBase ba = kind switch
            {
                "ray" => new BundleAdjusterRay(),
                "reproj" => new BundleAdjusterReproj(),
                "affine" => new BundleAdjusterAffine(),
                "partial" => new BundleAdjusterAffinePartial(),
                _ => new NoBundleAdjuster(),
            };
            ba.ConfThresh = adj.GetProperty("confThresh").GetDouble();
            if (adj.TryGetProperty("refinementMask", out JsonElement rm))
            {
                int[] v = rm.EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var mask = new byte[3, 3];
                for (int i = 0; i < 9; i++)
                {
                    mask[i / 3, i % 3] = (byte)v[i];
                }

                ba.RefinementMask = mask;
            }

            List<CameraParams> cameras = Cameras(c.GetProperty("cameras"));
            foreach (CameraParams cam in cameras)
            {
                cam.R = RoundToFloat(cam.R);
            }

            bool ok = ba.Estimate(features, pairwise, cameras);
            Assert.Equal(adj.GetProperty("ok").GetBoolean(), ok);
            if (ok)
            {
                double tol = adj.GetProperty("lapackSolve").GetBoolean() ? LapackSolveTolerance : 0;
                AssertCameras(adj.GetProperty("cameras"), cameras, tol, $"{name} {kind} conf={ba.ConfThresh}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(EstimatorCaseNames))]
    public void WaveCorrect_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Find(Data.EstimatorCases, name);
        foreach (JsonElement adj in c.GetProperty("adjusters").EnumerateArray())
        {
            foreach (JsonElement wave in adj.GetProperty("waves").EnumerateArray())
            {
                List<Mat<double>> rmats = Cameras(adj.GetProperty("cameras")).Select(cam => cam.R).ToList();
                var kind = (WaveCorrectKind)wave.GetProperty("kind").GetInt32();
                WaveCorrection.WaveCorrect(rmats, kind);
                JsonElement[] expected = wave.GetProperty("R").EnumerateArray().ToArray();
                for (int i = 0; i < rmats.Count; i++)
                {
                    AssertClose(Doubles(expected[i]), rmats[i].ToArray(), WaveTolerance, $"{name} {adj.GetProperty("kind").GetString()} wave {kind} R[{i}]");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(FocalCaseNames))]
    public void FocalsFromHomography_MatchesReference(string name)
    {
        JsonElement c = Find(Data.FocalCases, name);
        var h = new Mat<double>(Doubles(c.GetProperty("H")), 3, 3);
        Autocalibration.FocalsFromHomography(h, out double f0, out double f1, out bool ok0, out bool ok1);
        Assert.Equal(c.GetProperty("f0Ok").GetBoolean(), ok0);
        Assert.Equal(c.GetProperty("f1Ok").GetBoolean(), ok1);
        double[] f = Doubles(c.GetProperty("f"));
        if (ok0)
        {
            AssertClose([f[0]], [f0], 0, $"{name} f0");
        }

        if (ok1)
        {
            AssertClose([f[1]], [f1], 0, $"{name} f1");
        }
    }

    [Theory]
    [MemberData(nameof(ComponentCaseNames))]
    public void LeaveBiggestComponent_AndGraphString_MatchOpenCvGolden(string name)
    {
        JsonElement c = Find(Data.ComponentCases, name);
        JsonElement matcher = Find(Data.MatcherCases, c.GetProperty("matcherCase").GetString()!);
        ImageFeatures[] features = Features(matcher.GetProperty("set").GetString()!);
        MatchesInfo[] pairwise = matcher.GetProperty("cv2").EnumerateArray().Select(ToMatchesInfo).ToArray();
        float conf = c.GetProperty("conf").GetSingle();
        string[] paths = c.GetProperty("paths").EnumerateArray().Select(e => e.GetString()!).ToArray();

        Assert.Equal(c.GetProperty("graph").GetString(), MatchesGraph.MatchesGraphAsString(paths, pairwise, conf));

        int n = features.Length;
        int[] indices = MatchesGraph.LeaveBiggestComponent(ref features, ref pairwise, conf);
        int[] expected = c.GetProperty("indices").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Assert.Equal(expected, indices);
        Assert.Equal(indices.Length, features.Length);
        Assert.Equal(indices.Length * indices.Length, pairwise.Length);
        if (indices.Length < n)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                for (int j = 0; j < indices.Length; j++)
                {
                    Assert.Equal(i, pairwise[(i * indices.Length) + j].SrcImgIdx);
                    Assert.Equal(j, pairwise[(i * indices.Length) + j].DstImgIdx);
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(CalibCaseNames))]
    public void CalibrateRotatingCamera_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Find(Data.CalibCases, name);
        var hs = c.GetProperty("Hs").EnumerateArray().Select(e => new Mat<double>(Doubles(e), 3, 3)).ToList();
        bool ok = Autocalibration.CalibrateRotatingCamera(hs, out Mat<double>? k);
        Assert.Equal(c.GetProperty("ok").GetBoolean(), ok);
        if (ok)
        {
            AssertClose(Doubles(c.GetProperty("K")), k!.ToArray(), 1e-12, name);
        }
    }

    [Fact]
    public void MatchesGraph_FormatDouble_FollowsOstreamDefault()
    {
        Assert.Equal("2.45125", MatchesGraph.FormatDouble(2.4512534818941507));
        Assert.Equal("1e-05", MatchesGraph.FormatDouble(1e-5));
        Assert.Equal("1.23457e+06", MatchesGraph.FormatDouble(1234567));
        Assert.Equal("0", MatchesGraph.FormatDouble(0));
    }

    private const double WaveTolerance = 0;

    /// <summary>
    /// Relative tolerance for adjustments with 25 or more parameters: cv2 solves the LevMarq step
    /// with the LAPACK SVD, the port with the OpenCV Jacobi SVD.
    /// </summary>
    private const double LapackSolveTolerance = 1e-5;

    private static MatchesInfo[] RunMatcher(JsonElement c)
    {
        JsonElement spec = c.GetProperty("spec");
        string kind = spec.GetProperty("kind").GetString()!;
        float conf = spec.GetProperty("matchConf").GetSingle();
        int t1 = spec.GetProperty("thresh1").GetInt32();
        int t2 = spec.GetProperty("thresh2").GetInt32();
        FeaturesMatcher matcher = kind switch
        {
            "best" => new BestOf2NearestMatcher(conf, t1, t2, spec.GetProperty("confThresh").GetDouble()),
            "range" => new BestOf2NearestRangeMatcher(spec.GetProperty("rangeWidth").GetInt32(), conf, t1, t2),
            _ => new AffineBestOf2NearestMatcher(spec.GetProperty("fullAffine").GetBoolean(), conf, t1),
        };
        ImageFeatures[] features = Features(c.GetProperty("set").GetString()!);
        Mat<byte>? mask = null;
        if (c.TryGetProperty("mask", out JsonElement m))
        {
            mask = new Mat<byte>(Convert.FromBase64String(m.GetString()!), features.Length, features.Length);
        }

        return matcher.Match(features, mask);
    }

    private static (ImageFeatures[] Features, MatchesInfo[] Pairwise) EstimatorInput(JsonElement c)
    {
        JsonElement matcher = Find(Data.MatcherCases, c.GetProperty("matcherCase").GetString()!);
        ImageFeatures[] features = Features(matcher.GetProperty("set").GetString()!);
        MatchesInfo[] pairwise = matcher.GetProperty("cv2").EnumerateArray().Select(ToMatchesInfo).ToArray();
        return (features, pairwise);
    }

    private static ImageFeatures[] Features(string set)
    {
        JsonElement[] tiles = Data.TileSets.GetProperty(set).EnumerateArray().ToArray();
        var result = new ImageFeatures[tiles.Length];
        for (int i = 0; i < tiles.Length; i++)
        {
            JsonElement t = tiles[i];
            int count = t.GetProperty("count").GetInt32();
            float[] pts = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(t.GetProperty("pts").GetString()!)).ToArray();
            var kps = new KeyPoint[count];
            for (int k = 0; k < count; k++)
            {
                kps[k] = new KeyPoint(pts[2 * k], pts[(2 * k) + 1], 31f);
            }

            int cols = t.GetProperty("descCols").GetInt32();
            Mat<byte>? desc = count == 0 ? null : new Mat<byte>(Convert.FromBase64String(t.GetProperty("desc").GetString()!), cols, count);
            result[i] = new ImageFeatures(i, new Size(t.GetProperty("width").GetInt32(), t.GetProperty("height").GetInt32()), kps, desc);
        }

        return result;
    }

    private static MatchesInfo ToMatchesInfo(JsonElement e)
    {
        int[] q = Ints(e, "query"), t = Ints(e, "train");
        float[] d = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(e.GetProperty("dist").GetString()!)).ToArray();
        var info = new MatchesInfo
        {
            SrcImgIdx = e.GetProperty("src").GetInt32(),
            DstImgIdx = e.GetProperty("dst").GetInt32(),
            Matches = Enumerable.Range(0, q.Length).Select(i => new DMatch(q[i], t[i], d[i])).ToList(),
            InliersMask = Convert.FromBase64String(e.GetProperty("mask").GetString()!),
            NumInliers = e.GetProperty("numInliers").GetInt32(),
            Confidence = e.GetProperty("confidence").GetDouble(),
        };
        if (e.TryGetProperty("H", out JsonElement h))
        {
            info.H = new Mat<double>(Doubles(h), 3, 3);
        }

        return info;
    }

    private static void AssertMatchesInfo(JsonElement expected, MatchesInfo actual, string where)
    {
        Assert.True(expected.GetProperty("src").GetInt32() == actual.SrcImgIdx, $"{where}: src");
        Assert.True(expected.GetProperty("dst").GetInt32() == actual.DstImgIdx, $"{where}: dst");
        int[] q = Ints(expected, "query"), t = Ints(expected, "train");
        float[] d = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(expected.GetProperty("dist").GetString()!)).ToArray();
        Assert.True(q.Length == actual.Matches.Count, $"{where}: match count {actual.Matches.Count}, expected {q.Length}");
        for (int i = 0; i < q.Length; i++)
        {
            DMatch m = actual.Matches[i];
            Assert.True(q[i] == m.QueryIdx && t[i] == m.TrainIdx && d[i] == m.Distance, $"{where}: match {i}");
        }

        byte[] mask = Convert.FromBase64String(expected.GetProperty("mask").GetString()!);
        Assert.True(mask.AsSpan().SequenceEqual(actual.InliersMask), $"{where}: inlier mask");
        Assert.True(expected.GetProperty("numInliers").GetInt32() == actual.NumInliers, $"{where}: num inliers");
        double conf = expected.GetProperty("confidence").GetDouble();
        Assert.True(BitConverter.DoubleToInt64Bits(conf) == BitConverter.DoubleToInt64Bits(actual.Confidence), $"{where}: confidence {actual.Confidence:R}, expected {conf:R}");
        if (expected.TryGetProperty("H", out JsonElement h))
        {
            Assert.True(actual.H is not null, $"{where}: H missing");
            AssertClose(Doubles(h), actual.H!.ToArray(), 0, $"{where}: H");
        }
        else
        {
            Assert.True(actual.H is null, $"{where}: H must be empty");
        }
    }

    private static List<CameraParams> Cameras(JsonElement arr)
        => arr.EnumerateArray().Select(e => new CameraParams
        {
            Focal = e.GetProperty("focal").GetDouble(),
            Aspect = e.GetProperty("aspect").GetDouble(),
            Ppx = e.GetProperty("ppx").GetDouble(),
            Ppy = e.GetProperty("ppy").GetDouble(),
            R = new Mat<double>(Doubles(e.GetProperty("R")), 3, 3),
        }).ToList();

    private static void AssertCameras(JsonElement expected, List<CameraParams> actual, double tol, string where)
    {
        List<CameraParams> exp = Cameras(expected);
        Assert.Equal(exp.Count, actual.Count);
        for (int i = 0; i < exp.Count; i++)
        {
            AssertClose([exp[i].Focal, exp[i].Aspect, exp[i].Ppx, exp[i].Ppy], [actual[i].Focal, actual[i].Aspect, actual[i].Ppx, actual[i].Ppy], tol, $"{where} camera {i} intrinsics");
            AssertClose(exp[i].R.ToArray(), actual[i].R.ToArray(), tol, $"{where} camera {i} R");
        }
    }

    /// <summary>
    /// Bitwise equality for tol = 0; otherwise |a - e| ≤ tol·max(1, |e|). The failure message
    /// reports the largest difference.
    /// </summary>
    private static void AssertClose(double[] expected, double[] actual, double tol, string where)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = -1;
        double worstDiff = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual[i];
            bool same = BitConverter.DoubleToInt64Bits(e) == BitConverter.DoubleToInt64Bits(a) || (e == 0 && a == 0);
            double diff = same ? 0 : Math.Abs(a - e) / Math.Max(1, Math.Abs(e));
            if (!same && (double.IsNaN(diff) || diff > worstDiff))
            {
                worst = i;
                worstDiff = double.IsNaN(diff) ? double.PositiveInfinity : diff;
            }
        }

        if (worst >= 0 && (tol == 0 || worstDiff > tol))
        {
            Assert.Fail($"{where}[{worst}] = {actual[worst]:R}, expected {expected[worst]:R} (relative difference {worstDiff:E3}, tolerance {tol:E1}).");
        }
    }

    private static Mat<double> RoundToFloat(Mat<double> m)
    {
        double[] a = m.ToArray();
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)a[i];
        }

        return new Mat<double>(a, m.Width, m.Height);
    }

    private static int[] Ints(JsonElement e, string prop)
        => MemoryMarshal.Cast<byte, int>(Convert.FromBase64String(e.GetProperty(prop).GetString()!)).ToArray();

    private static double[] Doubles(JsonElement e)
        => MemoryMarshal.Cast<byte, double>(Convert.FromBase64String(e.GetString()!)).ToArray();

    private static JsonElement Find(List<JsonElement> cases, string name) => cases.Single(c => c.GetProperty("name").GetString() == name);

    private static Golden Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "stitch_reg_golden.json");
        JsonElement root = JsonDocument.Parse(File.ReadAllText(path)).RootElement;
        return new Golden(
            root.GetProperty("tileSets"),
            root.GetProperty("matcherCases").EnumerateArray().ToList(),
            root.GetProperty("estimatorCases").EnumerateArray().ToList(),
            root.GetProperty("focalCases").EnumerateArray().ToList(),
            root.GetProperty("componentCases").EnumerateArray().ToList(),
            root.GetProperty("calibCases").EnumerateArray().ToList());
    }

    private sealed record Golden(
        JsonElement TileSets,
        List<JsonElement> MatcherCases,
        List<JsonElement> EstimatorCases,
        List<JsonElement> FocalCases,
        List<JsonElement> ComponentCases,
        List<JsonElement> CalibCases);
}
