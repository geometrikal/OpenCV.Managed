namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/pyramid_golden.json</c> (pyrDown,
/// pyrUp, buildPyramid). Outputs must match byte for byte. Regenerate with
/// <c>tools/generate_pyramid_golden.py</c>.
/// </summary>
public sealed class PyramidGoldenTests
{
    internal static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Data.Cases.Count > 1000);

    /// <summary>No golden case differs between cv2 5.0.0 and the port.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Data.Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Fact]
    public void GoldenFile_CoversEveryOperationTypeAndBorder()
    {
        Assert.Equal(new[] { "buildPyramid", "pyrDown", "pyrUp" }, Data.Cases.Select(c => c.Op).Distinct().OrderBy(o => o, StringComparer.Ordinal));
        Assert.Equal(new[] { "f32", "f64", "i16", "u16", "u8" }, Data.Arrays.Select(a => a.Dtype).Distinct().OrderBy(o => o, StringComparer.Ordinal));
        Assert.Equal(
            new[] { "Reflect", "Reflect101", "Replicate", "Wrap" },
            Data.Cases.Select(c => c.Params["border"].GetString()!).Distinct().OrderBy(o => o, StringComparer.Ordinal));
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Pyramid_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Data.Cases.Single(x => x.Name == name);
        string dtype = c.Gen?.Dtype ?? Data.Arrays[c.Input!.Value].Dtype;
        List<byte[]> actual = dtype switch
        {
            "u8" => Run<byte>(c),
            "u16" => Run<ushort>(c),
            "i16" => Run<short>(c),
            "f32" => Run<float>(c),
            _ => Run<double>(c),
        };

        Assert.Equal(c.Outputs.Count, actual.Count);
        for (int i = 0; i < actual.Count; i++)
        {
            OutputData o = c.Outputs[i];
            if (o.Sha is not null)
            {
                Assert.Equal(o.Sha, Convert.ToHexString(SHA256.HashData(actual[i])).ToLowerInvariant());
                continue;
            }

            byte[] exp = Convert.FromBase64String(Data.Arrays[o.Array!.Value].Data);
            Assert.Equal(exp.Length, actual[i].Length);
            if (!exp.AsSpan().SequenceEqual(actual[i]))
            {
                int k = 0;
                while (exp[k] == actual[i][k])
                {
                    k++;
                }

                Assert.Fail($"Golden mismatch for '{name}' output {i} at byte {k} of {exp.Length}.");
            }
        }
    }

    private static List<byte[]> Run<T>(GoldenCase c)
        where T : unmanaged
    {
        Mat<T> src = c.Gen is not null
            ? HashInput<T>(c.Gen.Width, c.Gen.Height, c.Gen.Channels, c.Gen.Seed)
            : LoadMat<T>(Data.Arrays[c.Input!.Value]);
        if (c.Params.TryGetValue("roi", out JsonElement roi))
        {
            int[] r = roi.EnumerateArray().Select(e => e.GetInt32()).ToArray();
            src = src.Roi(r[0], r[1], r[2], r[3]);
        }

        BorderType border = Enum.Parse<BorderType>(c.Params["border"].GetString()!);
        switch (c.Op)
        {
            case "pyrDown":
                return [Bytes(Pyramids.PyrDown(src, new Size(c.Params["dw"].GetInt32(), c.Params["dh"].GetInt32()), border))];
            case "pyrUp":
                return [Bytes(Pyramids.PyrUp(src, new Size(c.Params["dw"].GetInt32(), c.Params["dh"].GetInt32()), border))];
            default:
                Mat<T>[] levels = Pyramids.BuildPyramid(src, c.Params["levels"].GetInt32(), border);
                Assert.Same(src, levels[0]);
                return levels.Skip(1).Select(Bytes).ToList();
        }
    }

    /// <summary>The generator's <c>hash_input</c>.</summary>
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
                    float f = (float)((((hv & 0xFFFF) / 65536.0) - 0.5) * 200.0);
                    row[(x * cn) + c] = typeof(T) == typeof(byte) ? (T)(object)(byte)(hv & 255)
                        : typeof(T) == typeof(ushort) ? (T)(object)(ushort)(hv & 0xFFFF)
                        : typeof(T) == typeof(short) ? (T)(object)unchecked((short)(hv & 0xFFFF))
                        : typeof(T) == typeof(float) ? (T)(object)f
                        : (T)(object)(double)f;
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

    internal static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
    {
        var bytes = new List<byte>();
        for (int y = 0; y < m.Height; y++)
        {
            bytes.AddRange(MemoryMarshal.AsBytes(m.GetReadOnlyRowSpan(y)).ToArray());
        }

        return bytes.ToArray();
    }

    internal static Mat<T> LoadMat<T>(ArrayData a)
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
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "pyramid_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public List<ArrayData> Arrays { get; set; } = [];

        public List<GoldenCase> Cases { get; set; } = [];

        public KeyPointData Keypoints { get; set; } = new();
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

    internal sealed class OutputData
    {
        public int? Array { get; set; }

        public string? Sha { get; set; }
    }

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int? Input { get; set; }

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public List<OutputData> Outputs { get; set; } = [];

        public GenData? Gen { get; set; }

        public bool Suspect { get; set; }
    }

    internal sealed class KeyPointData
    {
        /// <summary>Rows of seven uint32 float bit patterns: x1, y1, size1, x2, y2, size2, overlap.</summary>
        public List<List<uint>> Overlap { get; set; } = [];

        public ConvertData Convert { get; set; } = new();
    }

    internal sealed class ConvertData
    {
        public List<List<float>> Indexed { get; set; } = [];

        public List<List<double>> Reverse { get; set; } = [];
    }
}
