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
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/floodfill_golden.json</c> through
/// <see cref="FloodFilling"/>. Area, rectangle, output image and output mask must match byte for
/// byte. Each case also runs on an ROI image and an ROI mask inside padded parents.
/// Regenerate with <c>python tools/generate_floodfill_golden.py</c>.
/// </summary>
public sealed class FloodFillGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Data.Cases.Length > 300);

    /// <summary>The golden data comes from cv2 5.0.0, the version of the ported source; no case is suspect.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
        => Assert.Equal(Array.Empty<string>(), Data.Cases.Where(c => c.Suspect).Select(c => c.Name).ToArray());

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void FloodFill_MatchesOpenCvGolden(string name)
    {
        FloodCase c = Data.Cases.Single(x => x.Name == name);
        switch (Data.Arrays[c.Input].Dtype)
        {
            case "u8":
                Run<byte>(c);
                break;
            case "s32":
                Run<int>(c);
                break;
            default:
                Run<float>(c);
                break;
        }
    }

    private static void Run<T>(FloodCase c)
        where T : unmanaged
    {
        var seed = new Point(c.Seed[0], c.Seed[1]);
        var newVal = ToScalar(c.NewVal);
        var lo = ToScalar(c.LoDiff);
        var up = ToScalar(c.UpDiff);
        var flags = (FloodFillFlags)c.Flags;
        var expectedRect = new Rect(c.Rect[0], c.Rect[1], c.Rect[2], c.Rect[3]);

        Mat<T> image = LoadMat<T>(Data.Arrays[c.Input]);
        Mat<byte>? mask = c.UseMask ? LoadMat<byte>(Data.Arrays[c.MaskInput!.Value]) : null;
        FloodFillResult r = mask is null
            ? FloodFilling.FloodFill(image, seed, newVal, lo, up, flags)
            : FloodFilling.FloodFill(image, mask, seed, newVal, lo, up, flags);
        Assert.Equal(c.Area, r.Area);
        Assert.Equal(expectedRect, r.Rect);
        Compare(c, "image", c.Output, Bytes(image));
        if (mask is not null)
        {
            Compare(c, "mask", c.MaskOutput!.Value, Bytes(mask));
        }

        // ROI image and ROI mask; the parent padding repeats the seed pixel, so a fill that leaves the ROI changes it.
        Mat<T> packed = LoadMat<T>(Data.Arrays[c.Input]);
        int cn = packed.Channels;
        var parent = new Mat<T>(packed.Width + 5, packed.Height + 4, cn);
        for (int y = 0; y < parent.Height; y++)
        {
            for (int x = 0; x < parent.Width; x++)
            {
                for (int ch = 0; ch < cn; ch++)
                {
                    parent.At(y, x, ch) = packed.At(seed.Y, seed.X, ch);
                }
            }
        }

        Mat<T> parentBefore = parent.Clone();
        Mat<T> roi = parent.Roi(3, 2, packed.Width, packed.Height);
        packed.CopyTo(roi);
        Mat<byte>? maskRoi = null;
        Mat<byte>? maskParent = null;
        if (c.UseMask)
        {
            Mat<byte> m = LoadMat<byte>(Data.Arrays[c.MaskInput!.Value]);
            maskParent = new Mat<byte>(m.Width + 3, m.Height + 2);
            maskRoi = maskParent.Roi(2, 1, m.Width, m.Height);
            m.CopyTo(maskRoi);
        }

        FloodFillResult rr = maskRoi is null
            ? FloodFilling.FloodFill(roi, seed, newVal, lo, up, flags)
            : FloodFilling.FloodFill(roi, maskRoi, seed, newVal, lo, up, flags);
        Assert.Equal(c.Area, rr.Area);
        Assert.Equal(expectedRect, rr.Rect);
        Compare(c, "ROI image", c.Output, Bytes(roi));
        if (maskRoi is not null)
        {
            Compare(c, "ROI mask", c.MaskOutput!.Value, Bytes(maskRoi));
            Assert.Equal(0, maskParent!.GetRowSpan(0).ToArray().Sum(v => v));
        }

        packed.CopyTo(parentBefore.Roi(3, 2, packed.Width, packed.Height));
        Mat<T> expectedOut = LoadMat<T>(Data.Arrays[c.Output]);
        expectedOut.CopyTo(parentBefore.Roi(3, 2, packed.Width, packed.Height));
        Assert.True(Bytes(parentBefore).AsSpan().SequenceEqual(Bytes(parent)), $"ROI fill of '{c.Name}' changed pixels outside the ROI.");
    }

    private static Scalar ToScalar(double[] v) => new(v[0], v[1], v[2], v[3]);

    private static void Compare(FloodCase c, string what, int index, byte[] actual)
    {
        byte[] expected = Convert.FromBase64String(Data.Arrays[index].Data);
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            int i = 0;
            while (i < expected.Length && i < actual.Length && expected[i] == actual[i])
            {
                i++;
            }

            Assert.Fail($"Golden {what} mismatch for '{c.Name}' at byte {i} of {expected.Length} (actual length {actual.Length}).");
        }
    }

    private static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
        => MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray();

    private static Mat<T> LoadMat<T>(ArrayData a)
        where T : unmanaged
    {
        byte[] bytes = Convert.FromBase64String(a.Data);
        T[] values = MemoryMarshal.Cast<byte, T>(bytes).ToArray();
        return new Mat<T>(values, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        const string file = "floodfill_golden.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", file);
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
        {
            // The project file may not copy the data yet; read it from the source tree.
            path = Path.Combine(dir.FullName, "TestData", file);
        }

        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public FloodCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class FloodCase
    {
        public string Name { get; set; } = "";

        public int Input { get; set; }

        public int[] Seed { get; set; } = [];

        public double[] NewVal { get; set; } = [];

        public double[] LoDiff { get; set; } = [];

        public double[] UpDiff { get; set; } = [];

        public int Flags { get; set; }

        public bool UseMask { get; set; }

        public int Area { get; set; }

        public int[] Rect { get; set; } = [];

        public int Output { get; set; }

        public int? MaskInput { get; set; }

        public int? MaskOutput { get; set; }

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }
    }
}
