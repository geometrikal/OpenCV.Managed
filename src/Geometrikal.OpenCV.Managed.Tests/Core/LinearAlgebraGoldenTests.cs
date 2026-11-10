namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/linalg_golden.json</c> through
/// <see cref="LinearAlgebra"/> and <see cref="Svd{T}"/>. A case with tolerance 0 must match byte
/// for byte; a positive tolerance is relative to the largest expected magnitude.
/// Regenerate with <c>python tools/generate_linalg_golden.py</c>.
/// </summary>
public sealed class LinearAlgebraGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Data.Cases.Count > 500);

    /// <summary>
    /// The adjudicated cv2 differences: cv2 solves square DECOMP_QR systems of 30 or more rows
    /// through its LAPACK HAL with a wrong result. Pinned in
    /// <see cref="LinearAlgebraBasicTests.Solve_Qr_LargeSquare_MatchesLu"/>.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Equal(
            new[] { "solve_f32_qr_50_nb1", "solve_f64_qr_50_nb1" },
            Data.Cases.Where(c => c.Suspect).Select(c => c.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray());

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void LinearAlgebra_MatchesOpenCvGolden(string name)
    {
        LinalgCase c = Data.Cases.Single(x => x.Name == name);
        if (Data.Arrays[c.Inputs[0]].Dtype == "f32")
        {
            Run<float>(c);
        }
        else
        {
            Run<double>(c);
        }
    }

    private static void Run<T>(LinalgCase c)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Mat<T> In(int i) => ToMat<T>(Data.Arrays[c.Inputs[i]]);
        Mat<double> InD(int i) => ToMat<double>(Data.Arrays[c.Inputs[i]]);

        switch (c.Op)
        {
            case "gemm":
            {
                Mat<T>? cm = c.Inputs.Length > 2 ? In(2) : null;
                Mat<T> d = LinearAlgebra.Gemm(In(0), In(1), c.P("alpha"), cm, c.P("beta"), (GemmFlags)(int)c.P("flags"));
                Compare(c, "d", d);
                break;
            }

            case "det":
                CompareValue(c, 0, LinearAlgebra.Determinant(In(0)));
                break;

            case "invert":
            {
                double r = LinearAlgebra.Invert(In(0), out Mat<T> inv, (DecompTypes)(int)c.P("method"));
                CompareValue(c, 0, r);
                Compare(c, "inv", inv);
                break;
            }

            case "solve":
            {
                Mat<T> a = In(0);
                bool ok = LinearAlgebra.Solve(a, In(1), out Mat<T> x, (DecompTypes)(int)c.P("flags"));
                Assert.Equal(c.Values[0] != 0, ok);
                bool smallPath = a.Height <= 3 && a.Height == a.Width && x.Width == 1;
                if (ok || !smallPath)
                {
                    Compare(c, "x", x);
                }

                break;
            }

            case "eigen":
            {
                Assert.True(LinearAlgebra.Eigen(In(0), out Mat<T> w, out Mat<T> v));
                Compare(c, "w", w);
                Compare(c, "v", v);
                Assert.True(LinearAlgebra.Eigen(In(0), out Mat<T> w2));
                Compare(c, "w", w2);
                break;
            }

            case "svd":
            {
                Mat<T> a = In(0);
                var flags = (SvdFlags)(int)c.P("flags");
                var svd = new Svd<T>(a, flags);
                Compare(c, "w", svd.W);
                if (c.Params.ContainsKey("lapackSvd"))
                {
                    CheckReconstruction(a, svd, c.Tolerance * 10);
                }
                else if (c.Outputs.ContainsKey("u"))
                {
                    Compare(c, "u", svd.U);
                    Compare(c, "vt", svd.Vt);
                }

                break;
            }

            case "svbksb":
                Compare(c, "x", Svd<T>.BackSubst(In(0), In(1), In(2), In(3)));
                break;

            case "solvez":
                Compare(c, "x", Svd<T>.SolveZ(In(0)));
                break;

            case "cubic":
            {
                Mat<T> co = In(0);
                int n;
                Mat<T> roots = new(1, 3);
                if (typeof(T) == typeof(float))
                {
                    float[] r = new float[3];
                    n = LinearAlgebra.SolveCubic(MemoryMarshal.Cast<T, float>(co.GetReadOnlySpan()), r);
                    MemoryMarshal.Cast<float, T>(r.AsSpan()).CopyTo(roots.GetSpan());
                }
                else
                {
                    double[] r = new double[3];
                    n = LinearAlgebra.SolveCubic(MemoryMarshal.Cast<T, double>(co.GetReadOnlySpan()), r);
                    MemoryMarshal.Cast<double, T>(r.AsSpan()).CopyTo(roots.GetSpan());
                }

                Assert.Equal((int)c.Values[0], n);
                Compare(c, "roots", roots);
                break;
            }

            case "poly":
            {
                double[] co = InD(0).ToArray();
                var roots = new Complex[co.Length - 1];
                LinearAlgebra.SolvePoly(co, roots, (int)c.P("maxIters"));
                var m = new Mat<double>(2, roots.Length);
                for (int i = 0; i < roots.Length; i++)
                {
                    m.At(i, 0) = roots[i].Real;
                    m.At(i, 1) = roots[i].Imaginary;
                }

                Compare(c, "roots", m);
                break;
            }

            case "persp":
            {
                Mat<T> src = In(0);
                Mat<double> h = InD(1);
                Mat<T> d = LinearAlgebra.PerspectiveTransform(src, h);
                Compare(c, "d", d);
                if (src.Width == 1 && src.Channels == h.Width - 1 && h.Width == h.Height)
                {
                    Compare(c, "d", PointsPerspective(src, h));
                }

                break;
            }

            case "transform":
            {
                Mat<T> src = In(0);
                Mat<double> m = InD(1);
                Compare(c, "d", LinearAlgebra.Transform(src, m));
                if (src.Width == 1 && src.Channels == m.Height && src.Channels is 2 or 3)
                {
                    Compare(c, "d", PointsTransform(src, m));
                }

                break;
            }

            case "multrans":
                Compare(c, "d", LinearAlgebra.MulTransposed(In(0), c.P("aTa") != 0, c.P("scale")));
                break;

            case "trace":
                CompareValue(c, 0, LinearAlgebra.Trace(In(0)));
                break;

            default:
                throw new InvalidOperationException(c.Op);
        }
    }

    private static Mat<T> PointsPerspective<T>(Mat<T> src, Mat<double> h)
        where T : unmanaged
    {
        var d = new Mat<T>(1, src.Height, src.Channels);
        if (typeof(T) == typeof(float) && src.Channels == 2)
        {
            LinearAlgebra.PerspectiveTransform(MemoryMarshal.Cast<T, Point2f>(src.GetReadOnlySpan()), MemoryMarshal.Cast<T, Point2f>(d.GetSpan()), h);
        }
        else if (typeof(T) == typeof(double) && src.Channels == 2)
        {
            LinearAlgebra.PerspectiveTransform(MemoryMarshal.Cast<T, Point2d>(src.GetReadOnlySpan()), MemoryMarshal.Cast<T, Point2d>(d.GetSpan()), h);
        }
        else if (typeof(T) == typeof(float))
        {
            LinearAlgebra.PerspectiveTransform(MemoryMarshal.Cast<T, Point3f>(src.GetReadOnlySpan()), MemoryMarshal.Cast<T, Point3f>(d.GetSpan()), h);
        }
        else
        {
            LinearAlgebra.PerspectiveTransform(MemoryMarshal.Cast<T, Point3d>(src.GetReadOnlySpan()), MemoryMarshal.Cast<T, Point3d>(d.GetSpan()), h);
        }

        return d;
    }

    private static Mat<T> PointsTransform<T>(Mat<T> src, Mat<double> m)
        where T : unmanaged
    {
        T[] data = (typeof(T) == typeof(float), src.Channels) switch
        {
            (true, 2) => MemoryMarshal.Cast<Point2f, T>(LinearAlgebra.Transform(MemoryMarshal.Cast<T, Point2f>(src.GetReadOnlySpan()), m).AsSpan()).ToArray(),
            (false, 2) => MemoryMarshal.Cast<Point2d, T>(LinearAlgebra.Transform(MemoryMarshal.Cast<T, Point2d>(src.GetReadOnlySpan()), m).AsSpan()).ToArray(),
            (true, _) => MemoryMarshal.Cast<Point3f, T>(LinearAlgebra.Transform(MemoryMarshal.Cast<T, Point3f>(src.GetReadOnlySpan()), m).AsSpan()).ToArray(),
            (false, _) => MemoryMarshal.Cast<Point3d, T>(LinearAlgebra.Transform(MemoryMarshal.Cast<T, Point3d>(src.GetReadOnlySpan()), m).AsSpan()).ToArray(),
        };
        return new Mat<T>(data, 1, src.Height, src.Channels);
    }

    private static void CheckReconstruction<T>(Mat<T> a, Svd<T> svd, double tol)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int m = a.Height, n = a.Width, nm = Math.Min(m, n);
        double scale = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int k = 0; k < nm; k++)
                {
                    s += double.CreateTruncating(svd.U.At(i, k)) * double.CreateTruncating(svd.W.At(k, 0)) * double.CreateTruncating(svd.Vt.At(k, j));
                }

                double e = double.CreateTruncating(a.At(i, j));
                scale = Math.Max(scale, Math.Abs(e));
                Assert.True(Math.Abs(s - e) <= tol * Math.Max(1, scale) + 1e-5, $"reconstruction ({i},{j}): {s} vs {e}");
            }
        }
    }

    private static void CompareValue(LinalgCase c, int index, double actual)
    {
        double expected = c.Values[index];
        if (c.Tolerance == 0)
        {
            Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual) || expected == actual, $"{c.Name}: value {actual:R} != {expected:R}");
        }
        else
        {
            Assert.True(Math.Abs(actual - expected) <= c.Tolerance * Math.Max(1, Math.Abs(expected)), $"{c.Name}: value {actual:R} != {expected:R}");
        }
    }

    private static void Compare<T>(LinalgCase c, string output, Mat<T> actual)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        GoldenArray e = Data.Arrays[c.Outputs[output]];
        Assert.True(e.Height * e.Width * e.Channels == actual.ElementCount, $"{c.Name}/{output}: {e.Height}x{e.Width}x{e.Channels} expected, {actual.Height}x{actual.Width}x{actual.Channels} actual");
        T[] exp = MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(e.Data)).ToArray();
        T[] act = actual.ToArray();
        double maxAbs = exp.Length == 0 ? 0 : exp.Max(v => Math.Abs(double.CreateTruncating(v)));
        for (int i = 0; i < exp.Length; i++)
        {
            double ev = double.CreateTruncating(exp[i]), av = double.CreateTruncating(act[i]);
            bool ok = c.Tolerance == 0
                ? exp[i].Equals(act[i]) && (ev != 0 || double.IsNegative(ev) == double.IsNegative(av))
                : Math.Abs(ev - av) <= c.Tolerance * Math.Max(1, maxAbs);
            Assert.True(ok, $"{c.Name}/{output}[{i}]: actual {av:R}, expected {ev:R}");
        }
    }

    private static Mat<T> ToMat<T>(GoldenArray a)
        where T : unmanaged
    {
        T[] data = MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(a.Data)).ToArray();
        return new Mat<T>(data, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "linalg_golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        var arrays = doc.RootElement.GetProperty("arrays").EnumerateArray().Select(a => new GoldenArray(
            a.GetProperty("dtype").GetString()!,
            a.GetProperty("width").GetInt32(),
            a.GetProperty("height").GetInt32(),
            a.GetProperty("channels").GetInt32(),
            a.GetProperty("data").GetString()!)).ToList();
        var cases = doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new LinalgCase(
            c.GetProperty("name").GetString()!,
            c.GetProperty("op").GetString()!,
            c.GetProperty("inputs").EnumerateArray().Select(i => i.GetInt32()).ToArray(),
            c.GetProperty("params").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.ValueKind switch
            {
                JsonValueKind.True => 1.0,
                JsonValueKind.False => 0.0,
                _ => p.Value.GetDouble(),
            }),
            c.GetProperty("outputs").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32()),
            c.GetProperty("values").EnumerateArray().Select(v => v.GetDouble()).ToArray(),
            c.GetProperty("tolerance").GetDouble(),
            c.GetProperty("suspect").GetBoolean())).ToList();
        return new GoldenFile(arrays, cases);
    }

    private sealed record GoldenArray(string Dtype, int Width, int Height, int Channels, string Data);

    private sealed record LinalgCase(
        string Name,
        string Op,
        int[] Inputs,
        Dictionary<string, double> Params,
        Dictionary<string, int> Outputs,
        double[] Values,
        double Tolerance,
        bool Suspect)
    {
        public double P(string key) => Params[key];
    }

    private sealed record GoldenFile(List<GoldenArray> Arrays, List<LinalgCase> Cases);
}
