namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/mathfuncs_golden.json</c> through
/// <see cref="MathFunctions"/>. Outputs must match byte for byte.
/// Regenerate with <c>python tools/generate_mathfuncs_golden.py</c>.
/// </summary>
public sealed class MathFunctionsGoldenTests
{
    private static readonly List<GoldenCase> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    /// <summary>The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences, pinned in <see cref="MathFunctionsOpenCv5DifferenceTests"/>.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Cases.Where(c => c.Suspect).Select(c => c.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.Equal(MathFunctionsOpenCv5DifferenceTests.SuspectCases.Order(StringComparer.Ordinal).ToArray(), actual);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void MathFunctions_MatchOpenCvGolden(string name)
    {
        var c = Cases.Single(x => x.Name == name);
        string dtype = c.Inputs.First(i => i is not null)!.Dtype;
        byte[][] actual = dtype switch
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

        Assert.Equal(c.Outputs.Count, actual.Length);
        for (int k = 0; k < actual.Length; k++)
        {
            byte[] expected = Convert.FromBase64String(c.Outputs[k].Data);
            if (!expected.AsSpan().SequenceEqual(actual[k]))
            {
                if (UsesRsqrtEstimate(c) && !RsqrtMatchesGolden && WithinUlps(expected, actual[k], MaxRsqrtUlps))
                {
                    continue;
                }

                int i = 0;
                while (expected[i] == actual[k][i])
                {
                    i++;
                }

                string dt = c.Outputs[k].Dtype;
                int size = dt switch { "f64" => 8, "f32" or "i32" => 4, "u16" or "i16" => 2, _ => 1 };
                int e = i / size;
                string Value(byte[] bytes) => dt switch
                {
                    "f64" => BitConverter.ToDouble(bytes, e * 8).ToString("R"),
                    "f32" => BitConverter.ToSingle(bytes, e * 4).ToString("R"),
                    "i32" => BitConverter.ToInt32(bytes, e * 4).ToString(),
                    "u16" => BitConverter.ToUInt16(bytes, e * 2).ToString(),
                    "i16" => BitConverter.ToInt16(bytes, e * 2).ToString(),
                    "i8" => ((sbyte)bytes[e]).ToString(),
                    _ => bytes[e].ToString(),
                };
                int differing = 0;
                for (int j = 0; j < expected.Length; j += size)
                {
                    if (!expected.AsSpan(j, size).SequenceEqual(actual[k].AsSpan(j, size)))
                    {
                        differing++;
                    }
                }

                Assert.Fail($"Golden mismatch for '{name}' output {k} at element {e} of {expected.Length / size}: expected {Value(expected)}, actual {Value(actual[k])}; {differing} elements differ.");
            }
        }
    }

    internal static byte[][] Run<T>(GoldenCase c)
        where T : unmanaged
    {
        Mat<T>? In(int i) => c.Inputs[i] is { } a ? ToMat<T>(a) : null;
        bool degrees = c.Params.TryGetValue("degrees", out var d) && d.GetBoolean();
        return c.Op switch
        {
            "magnitude" => [Bytes(MathFunctions.Magnitude(In(0)!, In(1)!))],
            "phase" => [Bytes(MathFunctions.Phase(In(0)!, In(1)!, degrees))],
            "cartToPolar" => ToArray(MathFunctions.CartToPolar(In(0)!, In(1)!, degrees)),
            "polarToCart" => ToArray(MathFunctions.PolarToCart(In(0), In(1)!, degrees)),
            "exp" => [Bytes(MathFunctions.Exp(In(0)!))],
            "log" => [Bytes(MathFunctions.Log(In(0)!))],
            "sqrt" => [Bytes(MathFunctions.Sqrt(In(0)!))],
            "pow" => [Bytes(MathFunctions.Pow(In(0)!, c.Params["power"].GetDouble()))],
            _ => throw new InvalidOperationException(c.Op),
        };
    }

    // The goldens come from an Intel CPU, and AMD CPUs implement the RSQRTSS estimate table differently.
    private static readonly bool RsqrtMatchesGolden = !X86Base.IsSupported || IsIntel();

    // One Newton step after a different RSQRTSS estimate moves the float32 result by at most a few units in the last place.
    private const int MaxRsqrtUlps = 2;

    private static bool UsesRsqrtEstimate(GoldenCase c) =>
        c.Op == "pow" && c.Inputs[0]?.Dtype == "f32" && c.Params["power"].GetDouble() == -0.5;

    private static bool IsIntel()
    {
        (_, int ebx, int ecx, int edx) = X86Base.CpuId(0, 0);
        Span<int> vendor = [ebx, edx, ecx];
        return System.Text.Encoding.ASCII.GetString(MemoryMarshal.AsBytes(vendor)) == "GenuineIntel";
    }

    private static bool WithinUlps(byte[] expected, byte[] actual, int maxUlps)
    {
        ReadOnlySpan<int> e = MemoryMarshal.Cast<byte, int>(expected);
        ReadOnlySpan<int> a = MemoryMarshal.Cast<byte, int>(actual);
        for (int i = 0; i < e.Length; i++)
        {
            if (Math.Abs((long)e[i] - a[i]) > maxUlps)
            {
                return false;
            }
        }

        return true;
    }

    internal static GoldenCase Find(string name) => Cases.Single(c => c.Name == name);

    internal static Mat<T> ToMat<T>(GoldenArray a)
        where T : unmanaged
    {
        byte[] bytes = Convert.FromBase64String(a.Data);
        return new Mat<T>(MemoryMarshal.Cast<byte, T>(bytes).ToArray(), a.Width, a.Height, a.Channels);
    }

    private static byte[][] ToArray<T>((Mat<T> A, Mat<T> B) pair)
        where T : unmanaged => [Bytes(pair.A), Bytes(pair.B)];

    private static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged => MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray();

    private static List<GoldenCase> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "mathfuncs_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var cases = new List<GoldenCase>();
        foreach (var e in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            cases.Add(new GoldenCase(
                e.GetProperty("name").GetString()!,
                e.GetProperty("op").GetString()!,
                e.GetProperty("params").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.Clone()),
                e.GetProperty("inputs").EnumerateArray().Select(a => a.ValueKind == JsonValueKind.Null ? null : Array(a)).ToList(),
                e.GetProperty("outputs").EnumerateArray().Select(a => Array(a)!).ToList(),
                e.GetProperty("suspect").GetBoolean()));
        }

        return cases;

        static GoldenArray Array(JsonElement a) => new(
            a.GetProperty("dtype").GetString()!, a.GetProperty("width").GetInt32(), a.GetProperty("height").GetInt32(),
            a.GetProperty("channels").GetInt32(), a.GetProperty("data").GetString()!);
    }

    internal sealed record GoldenArray(string Dtype, int Width, int Height, int Channels, string Data);

    internal sealed record GoldenCase(
        string Name, string Op, Dictionary<string, JsonElement> Params, List<GoldenArray?> Inputs, List<GoldenArray> Outputs, bool Suspect);
}
