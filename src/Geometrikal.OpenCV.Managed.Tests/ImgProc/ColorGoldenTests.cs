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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/color_golden.json</c> through
/// <see cref="ColorConversions.CvtColor{T}"/>; outputs must match byte for byte. "cube" cases
/// convert all 2^24 8-bit triples and compare the SHA-256 of the output.
/// Regenerate with <c>python tools/generate_color_golden.py</c>.
/// </summary>
public sealed class ColorGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => c.Name).Order(StringComparer.Ordinal).Select(n => new object[] { n });

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
        => Assert.Empty(Data.Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void CvtColor_MatchesOpenCvGolden(string name)
    {
        ColorCase c = Data.Cases.Single(x => x.Name == name);
        var code = (ColorConversionCode)c.Value;
        if (c.Cube)
        {
            Mat<byte> output = ColorConversions.CvtColor(Cube(), code);
            Assert.Equal(c.OutChannels, output.Channels);
            Assert.Equal(c.Sha256, Convert.ToHexStringLower(SHA256.HashData(output.GetReadOnlySpan())));
            return;
        }

        ArrayData input = Data.Arrays[c.Input];
        byte[] actual = input.Dtype switch
        {
            "u8" => Run<byte>(input, c, code),
            "u16" => Run<ushort>(input, c, code),
            _ => Run<float>(input, c, code),
        };
        ArrayData expected = Data.Arrays[c.Expected];
        byte[] exp = Convert.FromBase64String(expected.Data);
        if (!exp.AsSpan().SequenceEqual(actual))
        {
            int i = 0;
            while (i < Math.Min(exp.Length, actual.Length) && exp[i] == actual[i])
            {
                i++;
            }

            Assert.Fail($"Golden mismatch for '{name}' at byte {i} of {exp.Length} (actual length {actual.Length}).");
        }
    }

    private static byte[] Run<T>(ArrayData a, ColorCase c, ColorConversionCode code)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        var src = new Mat<T>(buffer, a.Width, a.Height, a.Channels);
        if (c.Roi is { } r)
        {
            src = src.Roi(r[0], r[1], r[2], r[3]);
        }

        Mat<T> dst = ColorConversions.CvtColor(src, code, c.Dcn);
        return MemoryMarshal.AsBytes(dst.ToArray().AsSpan()).ToArray();
    }

    private static Mat<byte> Cube()
    {
        var m = new Mat<byte>(4096, 4096, 3, clear: false);
        Span<byte> s = m.GetSpan();
        for (int i = 0; i < 1 << 24; i++)
        {
            s[3 * i] = (byte)(i >> 16);
            s[(3 * i) + 1] = (byte)(i >> 8);
            s[(3 * i) + 2] = (byte)i;
        }

        return m;
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "color_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
    }

    private sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public ColorCase[] Cases { get; set; } = [];
    }

    private sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    private sealed class ColorCase
    {
        public string Name { get; set; } = "";

        public int Value { get; set; }

        public int Dcn { get; set; }

        public int Input { get; set; }

        public int Expected { get; set; }

        public int[]? Roi { get; set; }

        public bool Cube { get; set; }

        public string Sha256 { get; set; } = "";

        public int OutChannels { get; set; }

        public bool Suspect { get; set; }
    }
}
