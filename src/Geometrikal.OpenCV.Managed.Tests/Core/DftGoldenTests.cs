namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 4.5.5 (IPP disabled) reference outputs in <c>TestData/dft_golden.json</c>
/// through <see cref="SpectralTransforms"/>. Output buffers must match byte for byte.
/// Regenerate with <c>python tools/generate_dft_golden.py</c>.
/// </summary>
public sealed class DftGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    /// <summary>
    /// dxt.cpp is identical in 4.5.5 and 5.0 for the non-IPP CPU path. A future conflict must be
    /// adjudicated and listed here.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
    {
        string[] actual = Data.Cases.Where(c => c.Suspect).Select(c => c.Name).ToArray();
        Assert.Equal(Array.Empty<string>(), actual);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Dft_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Data.Cases.Single(x => x.Name == name);
        if (c.Op == "optimalSize")
        {
            int[] sizes = c.Params["sizes"].EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] actualSizes = sizes.Select(SpectralTransforms.GetOptimalDftSize).ToArray();
            Assert.Equal(c.Values!.Select(v => (int)v).ToArray(), actualSizes);
            return;
        }

        byte[] actual = Data.Arrays[c.Inputs[0]].Dtype == "f32" ? Run<float>(c) : Run<double>(c);
        ArrayData e = Data.Arrays[c.Expected!.Value];
        byte[] expected = Convert.FromBase64String(e.Data);
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            int i = 0;
            while (i < expected.Length && i < actual.Length && expected[i] == actual[i])
            {
                i++;
            }

            int elem = e.Dtype == "f32" ? 4 : 8;
            Assert.Fail($"Golden mismatch for '{name}' at element {i / elem} of {expected.Length / elem} (actual length {actual.Length / elem}).");
        }
    }

    private static byte[] Run<T>(GoldenCase c)
        where T : unmanaged
    {
        Mat<T> src = Load<T>(Data.Arrays[c.Inputs[0]]);
        if (c.Params.TryGetValue("roi", out JsonElement roi))
        {
            int[] r = roi.EnumerateArray().Select(x => x.GetInt32()).ToArray();
            src = src.Roi(r[0], r[1], r[2], r[3]);
        }

        int flags = c.Params["flags"].GetInt32();
        Mat<T> dst = c.Op switch
        {
            "dft" => SpectralTransforms.Dft(src, (DftFlags)flags, c.Params["nonzeroRows"].GetInt32()),
            "dct" => SpectralTransforms.Dct(src, (DctFlags)flags),
            "mulSpectrums" => SpectralTransforms.MulSpectrums(src, Load<T>(Data.Arrays[c.Inputs[1]]), (DftFlags)flags, c.Params["conjB"].GetBoolean()),
            _ => throw new InvalidOperationException(c.Op),
        };

        return MemoryMarshal.AsBytes(dst.ToArray().AsSpan()).ToArray();
    }

    private static Mat<T> Load<T>(ArrayData a)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "dft_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public GoldenCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int[] Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }

        public int? Expected { get; set; }

        public double[]? Values { get; set; }
    }
}
