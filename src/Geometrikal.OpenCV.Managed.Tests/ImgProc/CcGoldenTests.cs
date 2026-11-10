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
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/cc_golden.json</c> through
/// <see cref="ConnectedComponentLabeling"/>. Labels, statistics and centroids must match byte for
/// byte. Cases with stored outputs also run on an ROI source with an ROI label destination.
/// Regenerate with <c>python tools/generate_cc_golden.py</c>.
/// </summary>
public sealed class CcGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    /// <summary>
    /// The generator takes every 4-connectivity output from CCL_WU, which avoids the 4.5.5
    /// Spaghetti4C defect. No other cv2 4.5.5 versus OpenCV 5.0 difference is known for the battery.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
        => Assert.Equal(Array.Empty<string>(), Data.Cases.Where(c => c.Suspect).Select(c => c.Name).ToArray());

    [Fact]
    public void GoldenFile_FourConnectivityCases_UseWuSource()
        => Assert.All(Data.Cases.Where(c => c.Connectivity == 4), c => Assert.Equal(0, c.Source));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void ConnectedComponents_MatchesOpenCvGolden(string name)
    {
        CcCase c = Data.Cases.Single(x => x.Name == name);
        if (c.Ltype == "s32")
        {
            Run<int>(c);
        }
        else
        {
            Run<ushort>(c);
        }
    }

    /// <summary>The image formula of <c>synthetic</c> in the generator.</summary>
    internal static Mat<byte> Synthetic(int width, int height, int seed, int permille)
    {
        var m = new Mat<byte>(width, height);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                row[x] = Hash((uint)(x + (7919 * seed)), (uint)y) % 1000 < (uint)permille ? (byte)255 : (byte)0;
            }
        }

        return m;
    }

    private static uint Hash(uint x, uint y)
    {
        uint h = (x * 73856093u) ^ (y * 19349663u);
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        return h;
    }

    private static void Run<TLabel>(CcCase c)
        where TLabel : unmanaged
    {
        var alg = (ConnectedComponentsAlgorithm)c.Algorithm;
        bool sbyteInput = c.Input is int i && Data.Arrays[i].Dtype == "i8";
        Mat<byte> bytes = c.Synthetic is { } s ? Synthetic(s.Width, s.Height, s.Seed, s.Permille) : LoadBytes(Data.Arrays[c.Input!.Value]);

        if (c.Stats)
        {
            ConnectedComponentsStatsResult<TLabel> r = sbyteInput
                ? ConnectedComponentLabeling.ConnectedComponentsWithStats<TLabel>(AsSByte(bytes), c.Connectivity, alg)
                : ConnectedComponentLabeling.ConnectedComponentsWithStats<TLabel>(bytes, c.Connectivity, alg);
            Assert.Equal(c.N, r.LabelCount);
            Compare(c, "labels", c.Labels, c.LabelsSha, Bytes(r.Labels));
            Compare(c, "stats", c.StatsData, c.StatsDataSha, Bytes(r.Stats));
            Compare(c, "centroids", c.Centroids, c.CentroidsSha, Bytes(r.Centroids));
        }
        else
        {
            ConnectedComponentsResult<TLabel> r = sbyteInput
                ? ConnectedComponentLabeling.ConnectedComponents<TLabel>(AsSByte(bytes), c.Connectivity, alg)
                : ConnectedComponentLabeling.ConnectedComponents<TLabel>(bytes, c.Connectivity, alg);
            Assert.Equal(c.N, r.LabelCount);
            Compare(c, "labels", c.Labels, c.LabelsSha, Bytes(r.Labels));
        }

        if (c.Labels is null)
        {
            return;
        }

        // ROI source and ROI destination with odd offsets and padded rows.
        var parent = new Mat<byte>(bytes.Width + 5, bytes.Height + 3);
        parent.Fill(1);
        Mat<byte> roi = parent.Roi(3, 2, bytes.Width, bytes.Height);
        bytes.CopyTo(roi);
        var labelParent = new Mat<TLabel>(bytes.Width + 4, bytes.Height + 2);
        Mat<TLabel> labelRoi = labelParent.Roi(1, 1, bytes.Width, bytes.Height);
        int n = sbyteInput
            ? ConnectedComponentLabeling.ConnectedComponents(AsSByte(roi), labelRoi, c.Connectivity, alg)
            : ConnectedComponentLabeling.ConnectedComponents(roi, labelRoi, c.Connectivity, alg);
        Assert.Equal(c.N, n);
        Compare(c, "ROI labels", c.Labels, null, Bytes(labelRoi));
    }

    private static Mat<sbyte> AsSByte(Mat<byte> m)
        => new((sbyte[])(object)m.DangerousGetArray(), m.Width, m.Height, 1, m.Stride, m.DangerousGetOffset());

    private static void Compare(CcCase c, string what, int? index, string? sha, byte[] actual)
    {
        if (sha is not null)
        {
            Assert.True(sha == Convert.ToHexStringLower(SHA256.HashData(actual)), $"{what} hash mismatch for '{c.Name}'.");
            return;
        }

        byte[] expected = Convert.FromBase64String(Data.Arrays[index!.Value].Data);
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

    private static Mat<byte> LoadBytes(ArrayData a)
        => new(Convert.FromBase64String(a.Data), a.Width, a.Height);

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "cc_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public CcCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class SyntheticData
    {
        public int Width { get; set; }

        public int Height { get; set; }

        public int Seed { get; set; }

        public int Permille { get; set; }
    }

    internal sealed class CcCase
    {
        public string Name { get; set; } = "";

        public int Connectivity { get; set; }

        public int Algorithm { get; set; }

        public int Source { get; set; }

        public string Ltype { get; set; } = "";

        public bool Stats { get; set; }

        public int N { get; set; }

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }

        public int? Input { get; set; }

        public SyntheticData? Synthetic { get; set; }

        public int? Labels { get; set; }

        public string? LabelsSha { get; set; }

        public int? StatsData { get; set; }

        public string? StatsDataSha { get; set; }

        public int? Centroids { get; set; }

        public string? CentroidsSha { get; set; }
    }
}
