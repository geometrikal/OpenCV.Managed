namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Loader for <c>TestData/alignment_golden.json</c> (written by
/// <c>tools/generate_alignment_golden.py</c>).
/// </summary>
internal static class AlignmentGolden
{
    public static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> Names(string op)
        => Data.Cases.Where(c => c.Op == op && !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    public static IEnumerable<object[]> Names(string op, string prefix)
        => Names(op).Where(o => ((string)o[0]).StartsWith(prefix, StringComparison.Ordinal));

    public static GoldenCase Case(string name) => Data.Cases.Single(c => c.Name == name);

    public static Mat<T> Input<T>(GoldenCase c, string key)
        where T : unmanaged
        => LoadMat<T>(Data.Arrays[c.Inputs[key]]);

    public static string Dtype(GoldenCase c, string key) => Data.Arrays[c.Inputs[key]].Dtype;

    public static double Bits(JsonElement e) => BitConverter.Int64BitsToDouble(e.GetInt64());

    public static Mat<T> LoadMat<T>(ArrayData a)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        Assert.Equal(raw.Length, buffer.Length * Marshal.SizeOf<T>());
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, a.Width, a.Height, a.Channels);
    }

    public static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
    {
        var bytes = new List<byte>();
        for (int y = 0; y < m.Height; y++)
        {
            bytes.AddRange(MemoryMarshal.AsBytes(m.GetReadOnlyRowSpan(y)).ToArray());
        }

        return bytes.ToArray();
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "alignment_golden.json");
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

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public Dictionary<string, int> Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public Dictionary<string, JsonElement> Scalars { get; set; } = [];

        public int? Expected { get; set; }

        public string? ExpectedSha { get; set; }

        public bool Suspect { get; set; }
    }
}
