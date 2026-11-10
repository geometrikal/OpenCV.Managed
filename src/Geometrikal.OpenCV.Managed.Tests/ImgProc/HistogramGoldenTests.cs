namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/histogram_golden.json</c> through
/// calcHist, calcBackProject, compareHist, equalizeHist, CLAHE, integral and normalize. Outputs
/// must match byte for byte; compareHist values must be equal. Cases flagged <c>model</c> come
/// from scalar models of the OpenCV 5.0 source (non-uniform bins, CLAHE bit shift).
/// Regenerate with <c>python tools/generate_histogram_golden.py</c>.
/// </summary>
public sealed class HistogramGoldenTests
{
    internal static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Data.Cases.Length > 100);

    /// <summary>
    /// The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences: compareHist CHISQR and CHISQR_ALT
    /// with a vector body. Each one is pinned in <see cref="HistogramOpenCv5DifferenceTests"/>.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Data.Cases.Where(c => c.Suspect).Select(c => c.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        string[] expected = new[] { "n1000", "n255", "n256", "n259", "16x16", "4x5x6" }
            .SelectMany(n => new[] { $"compareHist_{n}_m1", $"compareHist_{n}_m4" })
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();
        Assert.Equal(expected, actual);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Histogram_MatchesOpenCvGolden(string name)
    {
        HistCase c = Data.Cases.Single(x => x.Name == name);
        Result r = Run(c);
        if (c.Values is not null)
        {
            Assert.Equal(c.Values.Length, r.Values!.Length);
            for (int i = 0; i < c.Values.Length; i++)
            {
                Assert.True(c.Values[i].Equals(r.Values[i]), $"{name}[{i}]: expected {c.Values[i]:R}, actual {r.Values[i]:R}");
            }

            return;
        }

        Assert.Equal(c.Expected.Length, r.Outputs.Count);
        for (int k = 0; k < c.Expected.Length; k++)
        {
            ArrayData e = Data.Arrays[c.Expected[k]];
            (string dtype, byte[] bytes) = r.Outputs[k];
            Assert.Equal(e.Dtype, dtype);
            byte[] exp = Convert.FromBase64String(e.Data);
            Assert.Equal(exp.Length, bytes.Length);
            if (!exp.AsSpan().SequenceEqual(bytes))
            {
                int i = 0;
                while (exp[i] == bytes[i])
                {
                    i++;
                }

                Assert.Fail($"Golden mismatch for '{name}' output {k} at byte {i} of {exp.Length}.");
            }
        }
    }

    internal static Result Run(HistCase c)
    {
        string dtype = Data.Arrays[c.Inputs[0]].Dtype;
        return dtype switch
        {
            "u8" => Run<byte>(c),
            "u16" => Run<ushort>(c),
            "i16" => Run<short>(c),
            "f32" => Run<float>(c),
            "f64" => Run<double>(c),
            _ => throw new InvalidOperationException(dtype),
        };
    }

    private static Result Run<T>(HistCase c)
        where T : unmanaged
    {
        int[]? roiInputs = c.Params.TryGetValue("roiInputs", out JsonElement ri) ? ri.EnumerateArray().Select(e => e.GetInt32()).ToArray() : null;
        int[]? roi = c.Params.TryGetValue("roi", out JsonElement re) ? re.EnumerateArray().Select(e => e.GetInt32()).ToArray() : null;
        Mat<TIn> In<TIn>(int i)
            where TIn : unmanaged
        {
            Mat<TIn> m = Load<TIn>(Data.Arrays[c.Inputs[i]]);
            return roi is not null && roiInputs!.Contains(i) ? m.Roi(roi[0], roi[1], roi[2], roi[3]) : m;
        }

        int I(string key) => c.Params[key].GetInt32();
        double P(string key) => c.Params[key].GetDouble();
        bool B(string key) => c.Params.TryGetValue(key, out JsonElement v) && v.ValueKind == JsonValueKind.True;
        int[] Ints(string key) => c.Params[key].EnumerateArray().Select(e => e.GetInt32()).ToArray();

        switch (c.Op)
        {
            case "calcHist":
            {
                int n = I("images");
                Mat<T>[] images = Enumerable.Range(0, n).Select(In<T>).ToArray();
                Mat<byte>? mask = B("mask") ? In<byte>(n) : null;
                int[] sizes = Ints("histSize");
                float[][]? ranges = Ranges(c, sizes);
                var hist = new DenseHistogram(sizes);
                bool accumulate = B("accumulate");
                if (accumulate)
                {
                    Load<float>(Data.Arrays[c.Inputs[n + 1]]).GetReadOnlySpan().CopyTo(hist.Values);
                }

                Histograms.CalcHist<T>(images, Ints("channels"), mask, hist, ranges, B("uniform"), accumulate);
                return Result.Of(hist.Values.ToArray());
            }

            case "calcBackProject":
            {
                int n = I("images");
                Mat<T>[] images = Enumerable.Range(0, n).Select(In<T>).ToArray();
                int[] sizes = Ints("histSize");
                var hist = new DenseHistogram(sizes);
                Load<float>(Data.Arrays[c.Inputs[n]]).GetReadOnlySpan().CopyTo(hist.Values);
                Mat<T> r = Histograms.CalcBackProject<T>(images, Ints("channels"), hist, Ranges(c, sizes), P("scale"), B("uniform"));
                return Result.Of(r);
            }

            case "compareHist":
            {
                int[] sizes = Ints("sizes");
                var h1 = new DenseHistogram(sizes);
                var h2 = new DenseHistogram(sizes);
                In<float>(0).GetReadOnlySpan().CopyTo(h1.Values);
                In<float>(1).GetReadOnlySpan().CopyTo(h2.Values);
                double v = Histograms.CompareHist(h1, h2, (HistCompMethods)I("method"));
                double vm = Histograms.CompareHist(In<float>(0), In<float>(1), (HistCompMethods)I("method"));
                Assert.Equal(v, vm);
                return new Result { Values = [v] };
            }

            case "equalizeHist":
                return Result.Of(Histograms.EqualizeHist(In<byte>(0)));

            case "clahe":
            {
                int[] grid = Ints("grid");
                var clahe = new Clahe(P("clipLimit"), new Size(grid[0], grid[1])) { BitShift = I("bitShift") };
                return typeof(T) == typeof(byte)
                    ? Result.Of(clahe.Apply(In<byte>(0)))
                    : Result.Of(clahe.Apply(In<ushort>(0)));
            }

            case "integral":
                return Integral(In<T>(0), c.Params["sdepth"].GetString()!,
                    c.Params["sqdepth"].ValueKind == JsonValueKind.Null ? null : c.Params["sqdepth"].GetString(), B("tilted"));

            case "normalize":
            {
                Mat<T> src = In<T>(0);
                Mat<byte>? mask = B("mask") ? In<byte>(1) : null;
                var nt = (NormType)I("normType");
                double a = P("alpha");
                double b = P("beta");
                return c.Params["dtype"].GetString() switch
                {
                    "u8" => Result.Of(Conversions.Normalize<T, byte>(src, a, b, nt, mask)),
                    "f32" => Result.Of(Conversions.Normalize<T, float>(src, a, b, nt, mask)),
                    _ => Result.Of(Conversions.Normalize<T, double>(src, a, b, nt, mask)),
                };
            }

            default:
                throw new InvalidOperationException($"unknown op '{c.Op}'");
        }
    }

    private static Result Integral<T>(Mat<T> src, string sd, string? sq, bool tilted)
        where T : unmanaged
        => sd switch
        {
            "i32" => Integral<T, int>(src, sq, tilted),
            "f32" => Integral<T, float>(src, sq, tilted),
            _ => Integral<T, double>(src, sq, tilted),
        };

    private static Result Integral<T, TSum>(Mat<T> src, string? sq, bool tilted)
        where T : unmanaged
        where TSum : unmanaged
        => sq switch
        {
            null => Result.Of(IntegralImages.Integral<T, TSum>(src)),
            "i32" => Integral<T, TSum, int>(src, tilted),
            "f32" => Integral<T, TSum, float>(src, tilted),
            _ => Integral<T, TSum, double>(src, tilted),
        };

    private static Result Integral<T, TSum, TSq>(Mat<T> src, bool tilted)
        where T : unmanaged
        where TSum : unmanaged
        where TSq : unmanaged
    {
        if (tilted)
        {
            Mat<TSum> s = IntegralImages.Integral<T, TSum, TSq>(src, out Mat<TSq> q, out Mat<TSum> t);
            return Result.Of(s).And(q).And(t);
        }

        Mat<TSum> sum = IntegralImages.Integral<T, TSum, TSq>(src, out Mat<TSq> sqsum);
        return Result.Of(sum).And(sqsum);
    }

    internal static float[][]? Ranges(HistCase c, int[] sizes)
    {
        JsonElement r = c.Params["ranges"];
        if (r.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        float[] flat = r.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        bool uniform = c.Params["uniform"].GetBoolean();
        var result = new float[sizes.Length][];
        int pos = 0;
        for (int i = 0; i < sizes.Length; i++)
        {
            int n = uniform ? 2 : sizes[i] + 1;
            result[i] = flat[pos..(pos + n)];
            pos += n;
        }

        return result;
    }

    internal static string DtypeName<T>()
        where T : unmanaged
        => typeof(T) == typeof(byte) ? "u8"
            : typeof(T) == typeof(ushort) ? "u16"
            : typeof(T) == typeof(short) ? "i16"
            : typeof(T) == typeof(int) ? "i32"
            : typeof(T) == typeof(float) ? "f32"
            : "f64";

    internal static Mat<T> Load<T>(ArrayData a)
        where T : unmanaged
    {
        Assert.Equal(a.Dtype, DtypeName<T>());
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "histogram_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        })!;
    }

    internal sealed class Result
    {
        public List<(string Dtype, byte[] Bytes)> Outputs { get; } = [];

        public double[]? Values { get; init; }

        public static Result Of<T>(Mat<T> m)
            where T : unmanaged
            => new Result().And(m);

        public static Result Of(float[] values)
        {
            var r = new Result();
            r.Outputs.Add(("f32", MemoryMarshal.AsBytes(values.AsSpan()).ToArray()));
            return r;
        }

        public Result And<T>(Mat<T> m)
            where T : unmanaged
        {
            Outputs.Add((DtypeName<T>(), MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray()));
            return this;
        }
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public HistCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class HistCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int[] Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public int[] Expected { get; set; } = [];

        public bool Suspect { get; set; }

        public bool Model { get; set; }

        public double[]? Values { get; set; }

        public double[]? Values5 { get; set; }
    }
}
