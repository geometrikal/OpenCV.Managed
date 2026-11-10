namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 5.0.0 <c>solvePnP</c>, <c>solvePnPGeneric</c>, <c>solveP3P</c>,
/// <c>solvePnPRansac</c>, <c>solvePnPRefineLM</c> and <c>solvePnPRefineVVS</c> reference outputs in
/// <c>TestData/pnp_golden.json</c> through <see cref="PnP"/>. Every comparison is bit-exact unless
/// the case is in <see cref="Tolerances"/>.
/// </summary>
public sealed class PnpGoldenTests
{
    private static readonly Lazy<Dictionary<string, JsonElement>> Data = new(Load);

    /// <summary>Methods whose results differ from cv2 in the last bits, with the relative tolerance used.</summary>
    private static readonly Dictionary<string, double> Tolerances = new(StringComparer.Ordinal);

    /// <summary>Relative tolerance of <c>solvePnPRefineVVS</c> when cv2 routes the SVD to LAPACK.</summary>
    private const double VvsLapackTolerance = 1e-12;

    public static IEnumerable<object[]> CaseNames()
        => Data.Value.Keys.OrderBy(k => k, StringComparer.Ordinal).Select(k => new object[] { k });

    [Fact]
    public void GoldenFile_HasCases() => Assert.True(Data.Value.Count > 1000);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void PnP_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Data.Value[name];
        string op = c.GetProperty("op").GetString()!;
        JsonElement p = c.GetProperty("params");
        JsonElement inp = c.GetProperty("inputs");
        JsonElement outp = c.GetProperty("outputs");
        double[] objV = Doubles(inp.GetProperty("obj"));
        double[] imgV = Doubles(inp.GetProperty("img"));
        var obj = new Point3d[objV.Length / 3];
        for (int i = 0; i < obj.Length; i++)
        {
            obj[i] = new Point3d(objV[3 * i], objV[3 * i + 1], objV[3 * i + 2]);
        }

        var img = new Point2d[imgV.Length / 2];
        for (int i = 0; i < img.Length; i++)
        {
            img[i] = new Point2d(imgV[2 * i], imgV[2 * i + 1]);
        }

        var a = new Mat<double>(Doubles(inp.GetProperty("a")), 3, 3);
        double[] dist = Doubles(inp.GetProperty("dist"));
        string method = p.TryGetProperty("method", out JsonElement m) ? m.GetString()! : "iterative";
        double tol = Tolerances.TryGetValue(method, out double t) ? t : 0;

        switch (op)
        {
            case "pnp":
            {
                bool ok = PnP.SolvePnP(obj, img, a, dist, Method(method), out Point3d r, out Point3d tv);
                Assert.Equal(outp.GetProperty("ok").GetBoolean(), ok);
                if (ok)
                {
                    Equal(Doubles(outp.GetProperty("rvec")), r, tol);
                    Equal(Doubles(outp.GetProperty("tvec")), tv, tol);
                }

                break;
            }

            case "generic":
            {
                PnPSolution[] sols = PnP.SolvePnPGeneric(obj, img, a, dist, Method(method));
                Assert.Equal(outp.GetProperty("n").GetInt32(), sols.Length);
                JsonElement[] rv = outp.GetProperty("rvecs").EnumerateArray().ToArray();
                JsonElement[] tvv = outp.GetProperty("tvecs").EnumerateArray().ToArray();
                double[] errs = Doubles(outp.GetProperty("errs"));
                for (int i = 0; i < sols.Length; i++)
                {
                    Equal(Doubles(rv[i]), sols[i].RotationVector, tol);
                    Equal(Doubles(tvv[i]), sols[i].TranslationVector, tol);
                    Close(errs[i], sols[i].ReprojectionError, tol);
                }

                break;
            }

            case "p3p":
            {
                PnPPose[] sols = PnP.SolveP3P(obj, img, a, dist, Method(method));
                Assert.Equal(outp.GetProperty("n").GetInt32(), sols.Length);
                JsonElement[] rv = outp.GetProperty("rvecs").EnumerateArray().ToArray();
                JsonElement[] tvv = outp.GetProperty("tvecs").EnumerateArray().ToArray();
                for (int i = 0; i < sols.Length; i++)
                {
                    Equal(Doubles(rv[i]), sols[i].RotationVector, tol);
                    Equal(Doubles(tvv[i]), sols[i].TranslationVector, tol);
                }

                break;
            }

            case "ransac":
            {
                int iters = p.GetProperty("iters").GetInt32();
                float thr = (float)p.GetProperty("thr").GetDouble();
                double conf = p.GetProperty("conf").GetDouble();
                PnPPose? guess = null;
                if (inp.TryGetProperty("rvec0", out JsonElement gr))
                {
                    double[] g0 = Doubles(gr);
                    double[] g1 = Doubles(inp.GetProperty("tvec0"));
                    guess = new PnPPose(new Point3d(g0[0], g0[1], g0[2]), new Point3d(g1[0], g1[1], g1[2]));
                }

                PnPRansacResult res = PnP.SolvePnPRansac(obj, img, a, dist, Method(method), iters, thr, conf, guess);
                Assert.Equal(outp.GetProperty("ok").GetBoolean(), res.Success);
                if (res.Success)
                {
                    int[] inl = outp.GetProperty("inliers").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    Assert.Equal(inl, res.Inliers);
                    Equal(Doubles(outp.GetProperty("rvec")), res.RotationVector, tol);
                    Equal(Doubles(outp.GetProperty("tvec")), res.TranslationVector, tol);
                }

                break;
            }

            case "lm":
            case "vvs":
            {
                double[] r0 = Doubles(inp.GetProperty("rvec0"));
                double[] t0 = Doubles(inp.GetProperty("tvec0"));
                var r = new Point3d(r0[0], r0[1], r0[2]);
                var tv = new Point3d(t0[0], t0[1], t0[2]);
                var crit = new TermCriteria((TermCriteriaType)p.GetProperty("type").GetInt32(), p.GetProperty("max").GetInt32(), p.GetProperty("eps").GetDouble());
                if (op == "lm")
                {
                    PnP.SolvePnPRefineLM(obj, img, a, dist, ref r, ref tv, crit);
                }
                else
                {
                    PnP.SolvePnPRefineVVS(obj, img, a, dist, ref r, ref tv, crit, p.GetProperty("lambda").GetDouble());
                }

                // cv2 computes the pseudo-inverse of the 2N × 6 interaction matrix with LAPACK when 2N ≥ 25.
                double rt = 2 * obj.Length >= 25 ? VvsLapackTolerance : 0;
                Equal(Doubles(outp.GetProperty("rvec")), r, rt);
                Equal(Doubles(outp.GetProperty("tvec")), tv, rt);
                break;
            }

            default:
                throw new InvalidOperationException(op);
        }
    }

    internal static SolvePnPMethod Method(string name) => name switch
    {
        "iterative" => SolvePnPMethod.Iterative,
        "epnp" => SolvePnPMethod.Epnp,
        "p3p" => SolvePnPMethod.P3p,
        "ap3p" => SolvePnPMethod.Ap3p,
        "ippe" => SolvePnPMethod.Ippe,
        "ippe_square" => SolvePnPMethod.IppeSquare,
        "sqpnp" => SolvePnPMethod.Sqpnp,
        _ => throw new ArgumentException(name),
    };

    private static void Equal(double[] expected, Point3d actual, double tol)
    {
        Close(expected[0], actual.X, tol);
        Close(expected[1], actual.Y, tol);
        Close(expected[2], actual.Z, tol);
    }

    private static void Close(double expected, double actual, double tol)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"expected NaN, actual {actual:R}");
        }
        else if (tol == 0)
        {
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual), $"expected {expected:R}, actual {actual:R}");
        }
        else
        {
            Assert.True(Math.Abs(expected - actual) <= tol * Math.Max(1, Math.Abs(expected)), $"expected {expected:R}, actual {actual:R}");
        }
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(Number).ToArray();

    private static double Number(JsonElement x) => x.ValueKind == JsonValueKind.String
        ? x.GetString() switch { "nan" => double.NaN, "inf" => double.PositiveInfinity, _ => double.NegativeInfinity }
        : x.GetDouble();

    private static Dictionary<string, JsonElement> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "pnp_golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
        return doc.RootElement.Clone().GetProperty("cases").EnumerateArray().ToDictionary(e => e.GetProperty("name").GetString()!, e => e);
    }
}
