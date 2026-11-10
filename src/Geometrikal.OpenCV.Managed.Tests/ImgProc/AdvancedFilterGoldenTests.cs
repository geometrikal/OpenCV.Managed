namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/advfilter_golden.json</c> (filter2D DFT
/// path, boxFilter, sqrBoxFilter, bilateralFilter, guidedFilter, stackBlur). Outputs must match
/// byte for byte. Cases that depend on the processor's RCPPS table compare exactly only when
/// this processor reproduces the stored RCPPS probe. Regenerate with
/// <c>tools/generate_advfilter_golden.py</c> (cv2 5.0.0 with contrib).
/// </summary>
public sealed class AdvancedFilterGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    /// <summary>True when <c>RCPPS</c> on this processor gives the probe results of the golden processor.</summary>
    internal static bool RcpMatchesGolden { get; } = ProbeRcp();

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Data.Cases.Count > 500);

    /// <summary>
    /// The adjudicated differences: float bilateral images with scattered NaN. OpenCV 5.0
    /// <c>minMaxLoc</c> resets a vector lane at a NaN, so its colour-table scale depends on the NaN
    /// positions; the port skips NaN (a documented <c>MinMaxLoc</c> difference). The NaN arithmetic
    /// itself is covered by the exact <c>bilat_nan_*</c> cases and pinned in
    /// <see cref="AdvancedFilterBasicTests"/>.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Data.Cases.Where(c => c.Suspect).Select(c => c.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.Equal(
            new[] { "bilat_nanrandom_c1_w37", "bilat_nanrandom_c1_w40", "bilat_nanrandom_c3_w37", "bilat_nanrandom_c3_w40" },
            actual);
    }

    [Fact]
    public void GoldenFile_CoversEveryOperation()
    {
        string[] ops = Data.Cases.Select(c => c.Op).Distinct().OrderBy(o => o, StringComparer.Ordinal).ToArray();
        Assert.Equal(new[] { "bilateralFilter", "boxFilter", "filter2D", "guidedFilter", "sqrBoxFilter", "stackBlur" }, ops);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void AdvancedFilter_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Data.Cases.Single(x => x.Name == name);
        (byte[] actual, string dtype) = Run(c);
        if (c.ExpectedSha is not null)
        {
            Assert.Equal(c.ExpectedDtype, dtype);
            Assert.Equal(c.ExpectedSha, Convert.ToHexString(SHA256.HashData(actual)).ToLowerInvariant());
            return;
        }

        ArrayData expected = Data.Arrays[c.Expected!.Value];
        Assert.Equal(expected.Dtype, dtype);
        byte[] exp = Convert.FromBase64String(expected.Data);
        Assert.Equal(exp.Length, actual.Length);
        if (c.Rcp && !RcpMatchesGolden)
        {
            AssertWithinRcpTolerance(exp, actual, dtype, name);
            return;
        }

        if (!exp.AsSpan().SequenceEqual(actual))
        {
            int i = 0;
            while (exp[i] == actual[i])
            {
                i++;
            }

            Assert.Fail($"Golden mismatch for '{name}' at byte {i} of {exp.Length}.");
        }
    }

    /// <summary>
    /// A different <c>RCPPS</c> table changes the inverse variance by about 1e-3 relative, which
    /// moves a float result by a fraction of a grey level and a byte result by at most 1.
    /// </summary>
    private static void AssertWithinRcpTolerance(byte[] expected, byte[] actual, string dtype, string name)
    {
        if (dtype == "u8")
        {
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1, $"{name}[{i}]");
            }

            return;
        }

        ReadOnlySpan<float> e = MemoryMarshal.Cast<byte, float>(expected);
        ReadOnlySpan<float> a = MemoryMarshal.Cast<byte, float>(actual);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.True(Math.Abs(e[i] - a[i]) <= 0.5 + (1e-3 * Math.Abs(e[i])), $"{name}[{i}]: {e[i]} vs {a[i]}");
        }
    }

    private static bool ProbeRcp()
    {
        if (!Sse.IsSupported || Data.RcpProbe.Output is null)
        {
            return false;
        }

        float[] input = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(Data.RcpProbe.Input)).ToArray();
        float[] output = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(Data.RcpProbe.Output)).ToArray();
        for (int i = 0; i + 4 <= input.Length; i += 4)
        {
            Vector128<float> r = Sse.Reciprocal(Vector128.Create(input, i));
            for (int l = 0; l < 4; l++)
            {
                if (BitConverter.SingleToInt32Bits(r[l]) != BitConverter.SingleToInt32Bits(output[i + l]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static (byte[] Bytes, string Dtype) Run(GoldenCase c)
    {
        string srcType = c.Gen is not null ? c.Gen.Dtype : Data.Arrays[c.Inputs[c.Op == "guidedFilter" ? 1 : 0]].Dtype;
        if (c.Op == "guidedFilter")
        {
            string guideType = c.Gen is not null ? c.Gen.Dtype : Data.Arrays[c.Inputs[0]].Dtype;
            return guideType switch
            {
                "u8" => RunGuided<byte>(c, c.Gen is not null ? "u8" : srcType),
                "u16" => RunGuided<ushort>(c, srcType),
                _ => RunGuided<float>(c, srcType),
            };
        }

        return srcType switch
        {
            "u8" => RunSource<byte>(c),
            "u16" => RunSource<ushort>(c),
            "i16" => RunSource<short>(c),
            "f64" => RunSource<double>(c),
            _ => RunSource<float>(c),
        };
    }

    private static (byte[] Bytes, string Dtype) RunSource<TSrc>(GoldenCase c)
        where TSrc : unmanaged
    {
        Mat<TSrc> src = c.Gen is not null ? HashInput<TSrc>(c.Gen.Width, c.Gen.Height, c.Gen.Channels, c.Gen.Seed) : Load<TSrc>(Data.Arrays[c.Inputs[0]]);
        Dictionary<string, JsonElement> p = c.Params;
        BorderType Border() => Enum.Parse<BorderType>(p["border"].GetString()!);
        int I(string key) => p[key].GetInt32();
        double D(string key) => p[key].GetDouble();
        string to = p.TryGetValue("to", out JsonElement t) ? t.GetString()! : "";
        switch (c.Op)
        {
            case "filter2D":
            {
                var anchor = new Point(I("anchorX"), I("anchorY"));
                if (p.TryGetValue("kernelDouble", out JsonElement kd) && kd.GetBoolean())
                {
                    // A float64 kernel reaches the engine only through the internal double overload.
                    Mat<double> k64 = Load<double>(Data.Arrays[I("kernel")]);
                    return Out(Filter2DEngine.Apply<TSrc, double>(src, k64.ToArray(), k64.Width, k64.Height, anchor, D("delta"), Border()));
                }

                Mat<float> kernel = Load<float>(Data.Arrays[I("kernel")]);
                return to switch
                {
                    "u8" => Out(Filters.Filter2D<TSrc, byte>(src, kernel, anchor, D("delta"), Border())),
                    "i16" => Out(Filters.Filter2D<TSrc, short>(src, kernel, anchor, D("delta"), Border())),
                    "u16" => Out(Filters.Filter2D<TSrc, ushort>(src, kernel, anchor, D("delta"), Border())),
                    "f64" => Out(Filters.Filter2D<TSrc, double>(src, kernel, anchor, D("delta"), Border())),
                    _ => Out(Filters.Filter2D<TSrc, float>(src, kernel, anchor, D("delta"), Border())),
                };
            }

            case "boxFilter":
            {
                var k = new Size(I("kw"), I("kh"));
                bool normalize = p["normalize"].GetBoolean();
                return to switch
                {
                    "u8" => Out(Filters.BoxFilter<TSrc, byte>(src, k, null, normalize, Border())),
                    "i16" => Out(Filters.BoxFilter<TSrc, short>(src, k, null, normalize, Border())),
                    _ => Out(Filters.BoxFilter<TSrc, float>(src, k, null, normalize, Border())),
                };
            }

            case "sqrBoxFilter":
            {
                var k = new Size(I("kw"), I("kh"));
                bool normalize = p["normalize"].GetBoolean();
                return to switch
                {
                    "i32" => Out(Filters.SqrBoxFilter<TSrc, int>(src, k, null, normalize, Border())),
                    "f64" => Out(Filters.SqrBoxFilter<TSrc, double>(src, k, null, normalize, Border())),
                    _ => Out(Filters.SqrBoxFilter<TSrc, float>(src, k, null, normalize, Border())),
                };
            }

            case "bilateralFilter":
                return Out(Filters.BilateralFilter(src, I("d"), D("sigmaColor"), D("sigmaSpace"), Border()));

            case "stackBlur":
                return Out(Filters.StackBlur(src, new Size(I("kw"), I("kh"))));

            default:
                throw new InvalidOperationException(c.Op);
        }
    }

    private static (byte[] Bytes, string Dtype) RunGuided<TGuide>(GoldenCase c, string srcType)
        where TGuide : unmanaged
        => srcType == "u8" ? RunGuided<TGuide, byte>(c) : RunGuided<TGuide, float>(c);

    private static (byte[] Bytes, string Dtype) RunGuided<TGuide, TSrc>(GoldenCase c)
        where TGuide : unmanaged
        where TSrc : unmanaged
    {
        Mat<TGuide> guide;
        Mat<TSrc> src;
        if (c.Gen is not null)
        {
            guide = HashInput<TGuide>(c.Gen.Width, c.Gen.Height, c.Gen.Channels, c.Gen.Seed);
            src = HashInput<TSrc>(c.Gen.Width, c.Gen.Height, 1, c.Params["srcSeed"].GetInt32());
        }
        else
        {
            guide = Load<TGuide>(Data.Arrays[c.Inputs[0]]);
            src = Load<TSrc>(Data.Arrays[c.Inputs[1]]);
        }

        int radius = c.Params["radius"].GetInt32();
        double eps = c.Params["eps"].GetDouble();
        double scale = c.Params["scale"].GetDouble();
        return c.Params["to"].GetString() switch
        {
            "u8" => Out(EdgePreservingFilters.GuidedFilter<TGuide, TSrc, byte>(guide, src, radius, eps, scale)),
            "f32" => Out(EdgePreservingFilters.GuidedFilter<TGuide, TSrc, float>(guide, src, radius, eps, scale)),
            _ => Out(EdgePreservingFilters.GuidedFilter(guide, src, radius, eps, scale)),
        };
    }

    /// <summary>The generator's <c>hash_input</c>: element (x, y, c) from <c>Hash(x + 131·seed, y·cn + c)</c>.</summary>
    internal static Mat<T> HashInput<T>(int w, int h, int cn, int seed)
        where T : unmanaged
    {
        var m = new Mat<T>(w, h, cn);
        for (int y = 0; y < h; y++)
        {
            Span<T> row = m.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < cn; c++)
                {
                    uint hv = Hash((uint)(x + (131 * seed)), (uint)((y * cn) + c));
                    if (typeof(T) == typeof(byte))
                    {
                        row[(x * cn) + c] = (T)(object)(byte)(hv & 255);
                    }
                    else
                    {
                        row[(x * cn) + c] = (T)(object)(float)((((hv & 0xFFFF) / 65536.0) - 0.5) * 200.0);
                    }
                }
            }
        }

        return m;
    }

    private static uint Hash(uint x, uint y)
    {
        uint h = unchecked((x * 73856093u) ^ (y * 19349663u));
        h ^= h >> 13;
        h = unchecked(h * 0x5bd1e995u);
        h ^= h >> 15;
        return h;
    }

    private static (byte[] Bytes, string Dtype) Out<T>(Mat<T> m)
        where T : unmanaged
    {
        var bytes = new List<byte>();
        for (int y = 0; y < m.Height; y++)
        {
            bytes.AddRange(MemoryMarshal.AsBytes(m.GetReadOnlyRowSpan(y)).ToArray());
        }

        string dtype = typeof(T) == typeof(byte) ? "u8"
            : typeof(T) == typeof(short) ? "i16"
            : typeof(T) == typeof(ushort) ? "u16"
            : typeof(T) == typeof(int) ? "i32"
            : typeof(T) == typeof(float) ? "f32"
            : "f64";
        return (bytes.ToArray(), dtype);
    }

    internal static Mat<T> Load<T>(ArrayData a)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        Assert.Equal(raw.Length, buffer.Length * Marshal.SizeOf<T>());
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "advfilter_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public ProbeData RcpProbe { get; set; } = new();

        public List<ArrayData> Arrays { get; set; } = [];

        public List<GoldenCase> Cases { get; set; } = [];
    }

    internal sealed class ProbeData
    {
        public string Input { get; set; } = "";

        public string? Output { get; set; }
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class GenData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public int Seed { get; set; }
    }

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public List<int> Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public int? Expected { get; set; }

        public string? ExpectedSha { get; set; }

        public string? ExpectedDtype { get; set; }

        public GenData? Gen { get; set; }

        public bool Rcp { get; set; }

        public bool Suspect { get; set; }
    }
}
