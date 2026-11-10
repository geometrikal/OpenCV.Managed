namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/threshold_golden.json</c> through
/// <see cref="Thresholding"/>. Output buffers must match byte for byte and returned thresholds
/// must match exactly. Regenerate with <c>python tools/generate_threshold_golden.py</c>.
/// </summary>
public sealed class ThresholdGoldenTests
{
    private static readonly GoldenFile Data = Load();

    private static readonly double PositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    /// <summary>
    /// No cv2 4.5.5 versus OpenCV 5.0 difference is known for the features in the battery. A
    /// future conflict must be adjudicated and listed here.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
    {
        string[] actual = Data.Cases.Where(c => c.Suspect).Select(c => c.Name).ToArray();
        Assert.Equal(Array.Empty<string>(), actual);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Threshold_MatchesOpenCvGolden(string name)
    {
        ThresholdCase c = Data.Cases.Single(x => x.Name == name);
        if (c.Op == "otsuSynthetic")
        {
            Mat<byte> image = SyntheticOtsuImage((int)P(c, "width"), (int)P(c, "height"));
            double used = Thresholding.Threshold(image, null, 0, 255, ThresholdType.Binary | ThresholdType.Otsu | ThresholdType.DryRun);
            Assert.Equal(c.Values![0], used);
            return;
        }

        (byte[] actual, double? returned) = Data.Arrays[c.Inputs[0]].Dtype switch
        {
            "u8" => Run<byte>(c),
            "i16" => Run<short>(c),
            "u16" => Run<ushort>(c),
            "f32" => Run<float>(c),
            "f64" => Run<double>(c),
            string d => throw new InvalidOperationException(d),
        };

        if (c.Values is not null)
        {
            Assert.True(
                BitConverter.DoubleToInt64Bits(c.Values[0]) == BitConverter.DoubleToInt64Bits(returned!.Value) ||
                (double.IsNaN(c.Values[0]) && double.IsNaN(returned.Value)),
                $"Returned threshold {returned} differs from golden {c.Values[0]} for '{name}'.");
        }

        byte[] expected = Convert.FromBase64String(Data.Arrays[c.Expected!.Value].Data);
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            int i = 0;
            while (i < expected.Length && i < actual.Length && expected[i] == actual[i])
            {
                i++;
            }

            Assert.Fail($"Golden mismatch for '{name}' at byte {i} of {expected.Length} (actual length {actual.Length}).");
        }
    }

    /// <summary>The image formula of <c>synthetic_otsu</c> in the generator.</summary>
    internal static Mat<byte> SyntheticOtsuImage(int width, int height)
    {
        var m = new Mat<byte>(width, height);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                row[x] = (byte)(50 + ((x * 7) + (y * 13)) % 200);
            }
        }

        for (int i = 0; i < 10; i++)
        {
            m.At((i * 991) % height, (i * 997) % width) = (byte)i;
        }

        return m;
    }

    private static (byte[] Bytes, double? Returned) Run<T>(ThresholdCase c)
        where T : unmanaged
    {
        Mat<T> src = Load<T>(Data.Arrays[c.Inputs[0]]);
        if (c.Op == "threshold")
        {
            Mat<T> dst = Thresholding.Threshold(src, P(c, "thresh"), P(c, "maxval"), (ThresholdType)(int)P(c, "type"), out double used);
            return (MemoryMarshal.AsBytes(dst.ToArray().AsSpan()).ToArray(), used);
        }

        if (c.Op == "adaptive")
        {
            Mat<byte> dst = Thresholding.AdaptiveThreshold(
                (Mat<byte>)(object)src,
                P(c, "maxval"),
                (AdaptiveThresholdType)(int)P(c, "method"),
                (ThresholdType)(int)P(c, "type"),
                (int)P(c, "blockSize"),
                P(c, "c"));
            return (dst.ToArray(), null);
        }

        throw new InvalidOperationException(c.Op);
    }

    private static double P(ThresholdCase c, string key)
    {
        JsonElement e = c.Params[key];
        if (e.ValueKind != JsonValueKind.String)
        {
            return e.GetDouble();
        }

        // The kernels keep NaN bits, so the input must be Python's positive quiet NaN, not double.NaN (sign bit set).
        string s = e.GetString()!;
        return s == "NaN" ? PositiveNaN : double.Parse(s, CultureInfo.InvariantCulture);
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
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "threshold_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        })!;
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public ThresholdCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class ThresholdCase
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
