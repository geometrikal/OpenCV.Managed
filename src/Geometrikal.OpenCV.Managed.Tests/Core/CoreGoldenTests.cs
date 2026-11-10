namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/core_golden.json</c> through the
/// managed core operations. Array outputs must match byte for byte unless the case has a
/// tolerance; reductions compare with the case tolerance (relative).
/// Regenerate with <c>python tools/generate_core_golden.py</c>.
/// </summary>
public sealed class CoreGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    /// <summary>
    /// The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences. Each one is pinned in
    /// <see cref="CoreOpenCv5DifferenceTests"/>.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Data.Cases.Where(c => c.Suspect).Select(c => c.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.Equal(
            new[]
            {
                "multiply_f32_c1_s0.1", "multiply_f32_c1_s0.3333333333333333",
                "multiply_f32_c3_s0.1", "multiply_f32_c3_s0.3333333333333333",
            },
            actual);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Core_MatchesOpenCvGolden(string name)
    {
        CoreCase c = Data.Cases.Single(x => x.Name == name);
        string dtype = Data.Arrays[c.Inputs[0]].Dtype;
        Result r = dtype switch
        {
            "u8" => Run<byte>(c),
            "i8" => Run<sbyte>(c),
            "u16" => Run<ushort>(c),
            "i16" => Run<short>(c),
            "i32" => Run<int>(c),
            "f32" => Run<float>(c),
            "f64" => Run<double>(c),
            _ => throw new InvalidOperationException(dtype),
        };

        if (c.Values is not null)
        {
            Assert.NotNull(r.Values);
            Assert.Equal(c.Values.Length, r.Values!.Length);
            for (int i = 0; i < c.Values.Length; i++)
            {
                AssertClose(c.Values[i], r.Values[i], c.Tolerance, $"{name}[{i}]");
            }

            return;
        }

        ArrayData expected = Data.Arrays[c.Expected!.Value];
        byte[] exp = Convert.FromBase64String(expected.Data);
        Assert.Equal(expected.Dtype, r.Dtype);
        Assert.Equal((expected.Width, expected.Height, expected.Channels), (r.Width, r.Height, r.Channels));
        if (c.Tolerance > 0)
        {
            double[] e = ToDoubles(exp, expected.Dtype);
            double[] a = ToDoubles(r.Bytes!, r.Dtype);
            for (int i = 0; i < e.Length; i++)
            {
                AssertClose(e[i], a[i], c.Tolerance, $"{name}[{i}]");
            }

            return;
        }

        if (!exp.AsSpan().SequenceEqual(r.Bytes))
        {
            int i = 0;
            while (exp[i] == r.Bytes![i])
            {
                i++;
            }

            Assert.Fail($"Golden mismatch for '{name}' at byte {i} of {exp.Length}.");
        }
    }

    private static Result Run<T>(CoreCase c)
        where T : unmanaged
    {
        Mat<T> In(int i) => Load<T>(Data.Arrays[c.Inputs[i]]);
        Mat<byte> Mask(int i) => Load<byte>(Data.Arrays[c.Inputs[i]]);
        Scalar S(string key = "scalar")
        {
            double[] v = c.Params[key].EnumerateArray().Select(e => e.GetDouble()).ToArray();
            return new Scalar(v[0], v[1], v[2], v[3]);
        }

        double P(string key) => c.Params[key].GetDouble();
        int I(string key) => c.Params[key].GetInt32();

        switch (c.Op)
        {
            case "add": return Of(Arithmetic.Add(In(0), In(1)));
            case "subtract": return Of(Arithmetic.Subtract(In(0), In(1)));
            case "absdiff": return Of(Arithmetic.AbsDiff(In(0), In(1)));
            case "min": return Of(Arithmetic.Min(In(0), In(1)));
            case "max": return Of(Arithmetic.Max(In(0), In(1)));
            case "addS": return Of(Arithmetic.Add(In(0), S()));
            case "subtractS": return Of(Arithmetic.Subtract(In(0), S()));
            case "subtractSR": return Of(Arithmetic.Subtract(S(), In(0)));
            case "absdiffS": return Of(Arithmetic.AbsDiff(In(0), S()));
            case "minS": return Of(Arithmetic.Min(In(0), S()));
            case "maxS": return Of(Arithmetic.Max(In(0), S()));
            case "multiplyS": return Of(Arithmetic.Multiply(In(0), S(), P("scale")));
            case "divideS": return Of(Arithmetic.Divide(In(0), S(), P("scale")));
            case "divideSR": return Of(Arithmetic.Divide(S(), In(0), P("scale")));
            case "multiply": return Of(Arithmetic.Multiply(In(0), In(1), P("scale")));
            case "divide": return Of(Arithmetic.Divide(In(0), In(1), P("scale")));
            case "reciprocal": return Of(Arithmetic.Divide(P("scale"), In(0)));
            case "addWeighted": return Of(Arithmetic.AddWeighted(In(0), P("alpha"), In(1), P("beta"), P("gamma")));
            case "scaleAdd": return Of(Arithmetic.ScaleAdd(In(0), P("alpha"), In(1)));
            case "addMask":
            {
                Mat<T> d = In(2);
                Arithmetic.Add(In(0), In(1), d, Mask(3));
                return Of(d);
            }

            case "subtractMask":
            {
                Mat<T> d = In(2);
                Arithmetic.Subtract(In(0), In(1), d, Mask(3));
                return Of(d);
            }

            case "addSMask":
            {
                Mat<T> d = In(1);
                Arithmetic.Add(In(0), S(), d, Mask(2));
                return Of(d);
            }

            case "bitwise_and": return Of(Bitwise.And(In(0), In(1)));
            case "bitwise_or": return Of(Bitwise.Or(In(0), In(1)));
            case "bitwise_xor": return Of(Bitwise.Xor(In(0), In(1)));
            case "bitwise_andS": return Of(Bitwise.And(In(0), S()));
            case "bitwise_orS": return Of(Bitwise.Or(In(0), S()));
            case "bitwise_xorS": return Of(Bitwise.Xor(In(0), S()));
            case "bitwise_not": return Of(Bitwise.Not(In(0)));
            case "bitwise_andMask":
            {
                Mat<T> d = In(2);
                Bitwise.And(In(0), In(1), d, Mask(3));
                return Of(d);
            }

            case "compare": return Of(Comparison.Compare(In(0), In(1), (CmpType)I("cmp")));
            case "compareS": return Of(Comparison.Compare(In(0), P("value"), (CmpType)I("cmp")));
            case "inRange": return Of(Comparison.InRange(In(0), In(1), In(2)));
            case "inRangeS": return Of(Comparison.InRange(In(0), S("lower"), S("upper")));
            case "copyToMask":
            {
                Mat<T> d = In(1);
                In(0).CopyTo(d, Mask(2));
                return Of(d);
            }

            case "convertTo":
                return c.Params["to"].GetString() switch
                {
                    "u8" => Of(Conversions.ConvertTo<T, byte>(In(0), P("alpha"), P("beta"))),
                    "i8" => Of(Conversions.ConvertTo<T, sbyte>(In(0), P("alpha"), P("beta"))),
                    "u16" => Of(Conversions.ConvertTo<T, ushort>(In(0), P("alpha"), P("beta"))),
                    "i16" => Of(Conversions.ConvertTo<T, short>(In(0), P("alpha"), P("beta"))),
                    "i32" => Of(Conversions.ConvertTo<T, int>(In(0), P("alpha"), P("beta"))),
                    "f32" => Of(Conversions.ConvertTo<T, float>(In(0), P("alpha"), P("beta"))),
                    _ => Of(Conversions.ConvertTo<T, double>(In(0), P("alpha"), P("beta"))),
                };
            case "convertScaleAbs": return Of(Conversions.ConvertScaleAbs(In(0), P("alpha"), P("beta")));
            case "lut":
            {
                ArrayData table = Data.Arrays[c.Inputs[1]];
                Mat<byte> src = Mask(0);
                return table.Dtype switch
                {
                    "u8" => Of(Conversions.Lut(src, Load<byte>(table))),
                    "i16" => Of(Conversions.Lut(src, Load<short>(table))),
                    "f32" => Of(Conversions.Lut(src, Load<float>(table))),
                    _ => Of(Conversions.Lut(src, Load<double>(table))),
                };
            }

            case "merge": return Of(Conversions.Merge(c.Inputs.Select(i => Load<T>(Data.Arrays[i])).ToArray()));
            case "sum": return Values(Reductions.Sum(In(0)));
            case "mean": return Values(Reductions.Mean(In(0)));
            case "meanMask": return Values(Reductions.Mean(In(0), Mask(1)));
            case "meanStdDev":
            case "meanStdDevMask":
            {
                Mat<T> src = In(0);
                Reductions.MeanStdDev(src, out Scalar mean, out Scalar sd, c.Op == "meanStdDevMask" ? Mask(1) : null);
                int cn = src.Channels;
                return new Result { Values = [.. Enumerable.Range(0, cn).Select(k => mean[k]), .. Enumerable.Range(0, cn).Select(k => sd[k])] };
            }

            case "norm": return new Result { Values = [Reductions.Norm(In(0), (NormType)I("normType"))] };
            case "normMask": return new Result { Values = [Reductions.Norm(In(0), (NormType)I("normType"), Mask(1))] };
            case "normDiff": return new Result { Values = [Reductions.Norm(In(0), In(1), (NormType)I("normType"))] };
            case "minMaxLoc":
            case "minMaxLocMask":
            {
                MinMaxLocResult m = Reductions.MinMaxLoc(In(0), c.Op == "minMaxLocMask" ? Mask(1) : null);
                return new Result { Values = [m.MinVal, m.MaxVal, m.MinLoc.X, m.MinLoc.Y, m.MaxLoc.X, m.MaxLoc.Y] };
            }

            case "countNonZero": return new Result { Values = [Reductions.CountNonZero(In(0))] };
            case "reduce":
            {
                var rt = (ReduceType)I("rtype");
                int dim = I("dim");
                return c.Params["to"].GetString() switch
                {
                    "u8" => Of(Reductions.Reduce<T, byte>(In(0), dim, rt)),
                    "u16" => Of(Reductions.Reduce<T, ushort>(In(0), dim, rt)),
                    "i16" => Of(Reductions.Reduce<T, short>(In(0), dim, rt)),
                    "i32" => Of(Reductions.Reduce<T, int>(In(0), dim, rt)),
                    "f32" => Of(Reductions.Reduce<T, float>(In(0), dim, rt)),
                    _ => Of(Reductions.Reduce<T, double>(In(0), dim, rt)),
                };
            }

            default: throw new InvalidOperationException($"unknown op '{c.Op}'");
        }
    }

    private static void AssertClose(double expected, double actual, double tol, string what)
    {
        if (expected.Equals(actual))
        {
            return;
        }

        double scale = Math.Max(1, Math.Abs(expected));
        Assert.True(tol > 0 && Math.Abs(expected - actual) <= tol * scale, $"{what}: expected {expected:R}, actual {actual:R}");
    }

    private static Result Values(Scalar s) => new() { Values = [s.V0, s.V1, s.V2, s.V3] };

    private static Result Of<T>(Mat<T> m)
        where T : unmanaged
        => new()
        {
            Dtype = DtypeName<T>(),
            Width = m.Width,
            Height = m.Height,
            Channels = m.Channels,
            Bytes = MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray(),
        };

    internal static string DtypeName<T>()
        where T : unmanaged
        => typeof(T) == typeof(byte) ? "u8"
            : typeof(T) == typeof(sbyte) ? "i8"
            : typeof(T) == typeof(ushort) ? "u16"
            : typeof(T) == typeof(short) ? "i16"
            : typeof(T) == typeof(int) ? "i32"
            : typeof(T) == typeof(float) ? "f32"
            : "f64";

    private static double[] ToDoubles(byte[] raw, string dtype) => dtype switch
    {
        "u8" => raw.Select(b => (double)b).ToArray(),
        "u16" => MemoryMarshal.Cast<byte, ushort>(raw).ToArray().Select(v => (double)v).ToArray(),
        "i16" => MemoryMarshal.Cast<byte, short>(raw).ToArray().Select(v => (double)v).ToArray(),
        "i32" => MemoryMarshal.Cast<byte, int>(raw).ToArray().Select(v => (double)v).ToArray(),
        "f32" => MemoryMarshal.Cast<byte, float>(raw).ToArray().Select(v => (double)v).ToArray(),
        _ => MemoryMarshal.Cast<byte, double>(raw).ToArray(),
    };

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
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "core_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        })!;
    }

    private sealed class Result
    {
        public string Dtype { get; init; } = "";

        public int Width { get; init; }

        public int Height { get; init; }

        public int Channels { get; init; }

        public byte[]? Bytes { get; init; }

        public double[]? Values { get; init; }
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public CoreCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class CoreCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int[] Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public bool Suspect { get; set; }

        public double Tolerance { get; set; }

        public int? Expected { get; set; }

        public double[]? Values { get; set; }
    }
}
