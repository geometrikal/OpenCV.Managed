namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Loader and comparison helpers for <c>TestData/stitch_warp_golden.json</c>.</summary>
internal static class StitchGolden
{
    public static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> Names(params string[] ops)
        => Data.Cases.Where(c => !c.Suspect && ops.Contains(c.Op)).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    public static GoldenCase Case(string name) => Data.Cases.Single(c => c.Name == name);

    public static Mat<T> Input<T>(GoldenCase c, int i)
        where T : unmanaged
        => LoadMat<T>(Data.Arrays[c.Inputs[i]]);

    public static float Bits(JsonElement e) => BitConverter.UInt32BitsToSingle(e.GetUInt32());

    public static double DBits(JsonElement e) => BitConverter.UInt64BitsToDouble(e.GetUInt64());

    public static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

    public static Point[] Corners(GoldenCase c)
        => c.Params["corners"].EnumerateArray().Select(p => new Point(p[0].GetInt32(), p[1].GetInt32())).ToArray();

    public static void AssertOutput<T>(GoldenCase c, int index, Mat<T> actual)
        where T : unmanaged
    {
        OutputData o = c.Outputs[index];
        byte[] bytes = Bytes(actual);
        if (o.Sha is not null)
        {
            Assert.Equal(o.Height, actual.Height);
            Assert.Equal(o.Width, actual.Width);
            Assert.Equal(o.Sha, Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());
            return;
        }

        ArrayData a = Data.Arrays[o.Array!.Value];
        Assert.Equal(a.Height, actual.Height);
        Assert.Equal(a.Width, actual.Width);
        Assert.Equal(a.Channels, actual.Channels);
        byte[] exp = Convert.FromBase64String(a.Data);
        Assert.Equal(exp.Length, bytes.Length);
        if (!exp.AsSpan().SequenceEqual(bytes))
        {
            int k = 0;
            while (exp[k] == bytes[k])
            {
                k++;
            }

            int esz = Marshal.SizeOf<T>();
            int elem = k / esz;
            int rowLen = a.Width * a.Channels;
            Assert.Fail($"Golden mismatch for '{c.Name}' output {index} at element {elem} (row {elem / rowLen}, col {elem % rowLen / a.Channels}, ch {elem % a.Channels}).");
        }
    }

    /// <summary>Maximum absolute element difference to an output stored as raw data.</summary>
    public static double MaxAbsDiff(GoldenCase c, int index, Mat<float> actual)
    {
        ArrayData a = Data.Arrays[c.Outputs[index].Array!.Value];
        Mat<float> exp = LoadMat<float>(a);
        double max = 0;
        for (int y = 0; y < a.Height; y++)
        {
            for (int x = 0; x < a.Width * a.Channels; x++)
            {
                max = Math.Max(max, Math.Abs(exp.GetReadOnlyRowSpan(y)[x] - actual.GetReadOnlyRowSpan(y)[x]));
            }
        }

        return max;
    }

    /// <summary>Maximum absolute difference and count of differing elements to a raw 8-bit output.</summary>
    public static (int Max, int Count) ByteDiff(GoldenCase c, int index, Mat<byte> actual)
    {
        ArrayData a = Data.Arrays[c.Outputs[index].Array!.Value];
        Assert.Equal(a.Width, actual.Width);
        Assert.Equal(a.Height, actual.Height);
        byte[] exp = Convert.FromBase64String(a.Data);
        byte[] act = Bytes(actual);
        int max = 0, count = 0;
        for (int i = 0; i < exp.Length; i++)
        {
            int d = Math.Abs(exp[i] - act[i]);
            max = Math.Max(max, d);
            count += d != 0 ? 1 : 0;
        }

        return (max, count);
    }

    public static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
    {
        var bytes = new byte[m.RowBytes * m.Height];
        for (int y = 0; y < m.Height; y++)
        {
            MemoryMarshal.AsBytes(m.GetReadOnlyRowSpan(y)).CopyTo(bytes.AsSpan(y * m.RowBytes));
        }

        return bytes;
    }

    public static Mat<T> LoadMat<T>(ArrayData a)
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
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "stitch_warp_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public List<ArrayData> Arrays { get; set; } = [];

        public List<GoldenCase> Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class OutputData
    {
        public int? Array { get; set; }

        public string? Sha { get; set; }

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }
    }

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public List<int> Inputs { get; set; } = [];

        public List<OutputData> Outputs { get; set; } = [];

        public Dictionary<string, JsonElement> Scalars { get; set; } = [];

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }
    }
}
