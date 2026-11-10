namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/matcher_golden.json</c> (BFMatcher
/// match, knnMatch, radiusMatch). Every match must be identical, including the distance bits and
/// the order of equal distances. Regenerate with <c>tools/generate_matcher_golden.py</c>.
/// </summary>
public sealed class BFMatcherGoldenTests
{
    private static readonly MatcherGoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Data.Cases.Count > 1000);

    /// <summary>No golden case differs between cv2 5.0.0 and the port.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Data.Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Fact]
    public void GoldenFile_CoversEveryOperationAndNorm()
    {
        Assert.Equal(new[] { "knn", "match", "radius" }, Data.Cases.Select(c => c.Op).Distinct().OrderBy(o => o, StringComparer.Ordinal));
        Assert.Equal(
            new[] { "Hamming", "Hamming2", "L1", "L2", "L2Sqr" },
            Data.Cases.Select(c => c.Norm).Distinct().OrderBy(o => o, StringComparer.Ordinal));
        Assert.Contains(Data.Cases, c => c.Op == "radius" && c.Expected.Any(r => r.Count > 32));
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void BFMatcher_MatchesOpenCvGolden(string name)
    {
        MatcherCase c = Data.Cases.Single(x => x.Name == name);
        DMatch[][] actual = Data.Arrays[c.Query].Dtype == "u8" ? Run<byte>(c) : Run<float>(c);
        Assert.Equal(c.Expected.Count, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            List<long[]> exp = c.Expected[i];
            Assert.True(exp.Count == actual[i].Length, $"'{name}' list {i}: expected {exp.Count} matches, got {actual[i].Length}.");
            for (int j = 0; j < exp.Count; j++)
            {
                DMatch m = actual[i][j];
                var got = new long[] { m.QueryIdx, m.TrainIdx, m.ImgIdx, BitConverter.SingleToUInt32Bits(m.Distance) };
                Assert.True(exp[j].SequenceEqual(got), $"'{name}' list {i} match {j}: expected [{string.Join(",", exp[j])}], got [{string.Join(",", got)}].");
            }
        }
    }

    private static DMatch[][] Run<T>(MatcherCase c)
        where T : unmanaged
    {
        var matcher = new BFMatcher(Enum.Parse<NormType>(c.Norm), c.CrossCheck);
        Mat<T> query = ToMat<T>(c.Query);
        List<Mat<T>> trains = c.Trains.Select(ToMat<T>).ToList();
        List<Mat<byte>?>? masks = c.Masks?.Select(m => (Mat<byte>?)ToMat<byte>(m)).ToList();
        if (c.Single)
        {
            Mat<byte>? mask = masks?[0];
            return c.Op switch
            {
                "match" => [matcher.Match(query, trains[0], mask)],
                "knn" => matcher.KnnMatch(query, trains[0], c.K!.Value, mask, c.Compact),
                _ => matcher.RadiusMatch(query, trains[0], (float)c.Radius!.Value, mask, c.Compact),
            };
        }

        foreach (Mat<T> t in trains)
        {
            matcher.Add(t);
        }

        return c.Op switch
        {
            "match" => [matcher.Match(query, masks)],
            "knn" => matcher.KnnMatch(query, c.K!.Value, masks, c.Compact),
            _ => matcher.RadiusMatch(query, (float)c.Radius!.Value, masks, c.Compact),
        };
    }

    private static Mat<T> ToMat<T>(int index)
        where T : unmanaged
    {
        MatrixData a = Data.Arrays[index];
        byte[] bytes = Convert.FromBase64String(a.Data);
        T[] values = MemoryMarshal.Cast<byte, T>(bytes).ToArray();
        return new Mat<T>(values, a.Cols, a.Rows);
    }

    private static MatcherGoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "matcher_golden.json");
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        return JsonSerializer.Deserialize<MatcherGoldenFile>(File.ReadAllText(path), options)!;
    }

    private sealed class MatcherGoldenFile
    {
        public List<MatrixData> Arrays { get; set; } = [];

        public List<MatcherCase> Cases { get; set; } = [];
    }

    private sealed class MatrixData
    {
        public string Dtype { get; set; } = "";

        public int Rows { get; set; }

        public int Cols { get; set; }

        public string Data { get; set; } = "";
    }

    private sealed class MatcherCase
    {
        public string Name { get; set; } = "";

        public bool Suspect { get; set; }

        public string Op { get; set; } = "";

        public string Norm { get; set; } = "";

        public bool CrossCheck { get; set; }

        public bool Single { get; set; }

        public int Query { get; set; }

        public List<int> Trains { get; set; } = [];

        public List<int>? Masks { get; set; }

        public int? K { get; set; }

        public double? Radius { get; set; }

        public bool Compact { get; set; }

        [JsonPropertyName("expected")]
        public List<List<long[]>> Expected { get; set; } = [];
    }
}
