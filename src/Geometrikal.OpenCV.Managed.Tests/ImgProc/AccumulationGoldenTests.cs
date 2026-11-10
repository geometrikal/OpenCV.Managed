namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/accumulate_golden.json</c> through
/// <see cref="Accumulation"/>. Accumulators must match bit for bit.
/// Regenerate with <c>python tools/generate_accumulate_golden.py</c>.
/// </summary>
public sealed class AccumulationGoldenTests
{
    private static readonly List<(JsonElement Group, JsonElement Result)> Cases = Load();

    public static IEnumerable<object[]> CaseNames() =>
        Cases.Where(c => !c.Result.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.Result.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Cases);

    [Fact]
    public void GoldenFile_HasNoSuspectCases() =>
        Assert.DoesNotContain(Cases, c => c.Result.GetProperty("suspect").GetBoolean());

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Accumulate_MatchesOpenCvGolden(string name)
    {
        var (g, r) = Cases.Single(x => x.Result.GetProperty("name").GetString() == name);
        string sdepth = g.GetProperty("src1").GetProperty("depth").GetString()!;
        bool f64 = g.GetProperty("acc").GetProperty("depth").GetString() == "f64";
        switch (sdepth)
        {
            case "u8" when f64: Run<byte, double>(g, r); break;
            case "u8": Run<byte, float>(g, r); break;
            case "u16" when f64: Run<ushort, double>(g, r); break;
            case "u16": Run<ushort, float>(g, r); break;
            case "f32" when f64: Run<float, double>(g, r); break;
            case "f32": Run<float, float>(g, r); break;
            default: Run<double, double>(g, r); break;
        }
    }

    private static void Run<TSrc, TDst>(JsonElement g, JsonElement r)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        var src1 = ToMat<TSrc>(g.GetProperty("src1"));
        var src2 = ToMat<TSrc>(g.GetProperty("src2"));
        var acc = ToMat<TDst>(g.GetProperty("acc"));
        var mask = g.TryGetProperty("mask", out var m) ? ToMat<byte>(m) : null;
        switch (r.GetProperty("op").GetString())
        {
            case "acc": Accumulation.Accumulate(src1, acc, mask); break;
            case "sqr": Accumulation.AccumulateSquare(src1, acc, mask); break;
            case "prod": Accumulation.AccumulateProduct(src1, src2, acc, mask); break;
            default: Accumulation.AccumulateWeighted(src1, acc, r.GetProperty("alpha").GetDouble(), mask); break;
        }

        byte[] expected = Convert.FromBase64String(r.GetProperty("result").GetString()!);
        byte[] actual = MemoryMarshal.AsBytes(acc.ToArray().AsSpan()).ToArray();
        Assert.Equal(expected, actual);
    }

    private static Mat<T> ToMat<T>(JsonElement m)
        where T : unmanaged
    {
        T[] data = MemoryMarshal.Cast<byte, T>(Convert.FromBase64String(m.GetProperty("data").GetString()!)).ToArray();
        return new Mat<T>(data, m.GetProperty("width").GetInt32(), m.GetProperty("height").GetInt32(), m.GetProperty("channels").GetInt32());
    }

    private static List<(JsonElement, JsonElement)> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "accumulate_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("groups").EnumerateArray()
            .SelectMany(g => g.GetProperty("results").EnumerateArray().Select(r => (g.Clone(), r.Clone())))
            .ToList();
    }
}
