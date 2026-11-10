namespace Geometrikal.OpenCV.Managed.Tests.Video;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Video;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/lk_golden.json</c>
/// (buildOpticalFlowPyramid, calcOpticalFlowPyrLK). Outputs must match bit for bit.
/// Regenerate with <c>tools/generate_lk_golden.py</c>.
/// </summary>
public sealed class OpticalFlowGoldenTests
{
    private static readonly JsonElement Root = Load();
    private static readonly JsonElement[] Arrays = Root.GetProperty("arrays").EnumerateArray().ToArray();
    private static readonly Dictionary<string, JsonElement> Cases =
        Root.GetProperty("cases").EnumerateArray().ToDictionary(c => c.GetProperty("name").GetString()!, c => c);

    public static IEnumerable<object[]> CaseNames()
        => Cases.Values.Where(c => !c.GetProperty("suspect").GetBoolean())
            .Select(c => new object[] { c.GetProperty("name").GetString()! })
            .OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Cases.Count > 500);

    /// <summary>No golden case differs between cv2 5.0.0 and the port.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Cases.Values.Where(c => c.GetProperty("suspect").GetBoolean()).Select(c => c.GetProperty("name").GetString()));

    [Fact]
    public void GoldenFile_CoversEveryOperation()
        => Assert.Equal(
            new[] { "lk", "pyramid" },
            Cases.Values.Select(c => c.GetProperty("op").GetString()!).Distinct().OrderBy(o => o, StringComparer.Ordinal));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void OpticalFlow_MatchesOpenCvGolden(string name)
    {
        JsonElement c = Cases[name];
        if (c.GetProperty("op").GetString() == "pyramid")
        {
            RunPyramid(name, c);
        }
        else
        {
            RunLk(name, c);
        }
    }

    /// <summary>The scalar emulation of the SSE lanes gives the same results as the SSE path.</summary>
    [Fact]
    public void OpticalFlow_ScalarLaneEmulation_MatchesOpenCvGolden()
    {
        bool previous = Geometrikal.OpenCV.Managed.Video.Internal.LkTracker.ForceScalar;
        try
        {
            Geometrikal.OpenCV.Managed.Video.Internal.LkTracker.ForceScalar = true;
            foreach (JsonElement c in Cases.Values.Where(c => c.GetProperty("op").GetString() == "lk"))
            {
                RunLk(c.GetProperty("name").GetString()!, c);
            }
        }
        finally
        {
            Geometrikal.OpenCV.Managed.Video.Internal.LkTracker.ForceScalar = previous;
        }
    }

    private static void RunPyramid(string name, JsonElement c)
    {
        JsonElement p = c.GetProperty("params");
        Mat<byte> img = ToMat<byte>(c.GetProperty("input").GetInt32());
        bool withDeriv = p.GetProperty("withDerivatives").GetBoolean();
        OpticalFlowPyramid pyr = OpticalFlow.BuildOpticalFlowPyramid(
            img,
            Win(p),
            p.GetProperty("maxLevel").GetInt32(),
            withDeriv,
            Enum.Parse<BorderType>(p.GetProperty("pyrBorder").GetString()!),
            Enum.Parse<BorderType>(p.GetProperty("derivBorder").GetString()!),
            p.GetProperty("reuse").GetBoolean());
        Assert.Equal(c.GetProperty("returned").GetInt32(), pyr.MaxLevel);
        JsonElement[] outputs = c.GetProperty("outputs").EnumerateArray().ToArray();
        int step = withDeriv ? 2 : 1;
        Assert.Equal(outputs.Length, pyr.Images.Length * step);
        for (int level = 0; level < pyr.Images.Length; level++)
        {
            CompareBytes($"{name} image {level}", MemoryMarshal.AsBytes(pyr.Images[level].ToArray().AsSpan()).ToArray(), outputs[level * step]);
            if (withDeriv)
            {
                CompareBytes($"{name} deriv {level}", MemoryMarshal.AsBytes(pyr.Derivatives![level].ToArray().AsSpan()).ToArray(), outputs[(level * step) + 1]);
            }
        }
    }

    private static void RunLk(string name, JsonElement c)
    {
        JsonElement p = c.GetProperty("params");
        Mat<byte> prev = ToMat<byte>(c.GetProperty("prev").GetInt32());
        Mat<byte> next = ToMat<byte>(c.GetProperty("next").GetInt32());
        Point2f[] prevPts = Points(c.GetProperty("prevPts"));
        var flags = (OpticalFlowFlags)p.GetProperty("flags").GetInt32();
        Point2f[] nextPts = c.GetProperty("initPts").ValueKind == JsonValueKind.Null ? new Point2f[prevPts.Length] : Points(c.GetProperty("initPts"));
        var status = new byte[prevPts.Length];
        var err = new float[prevPts.Length];
        var criteria = new TermCriteria((TermCriteriaType)p.GetProperty("type").GetInt32(), p.GetProperty("maxCount").GetInt32(), p.GetProperty("epsilon").GetDouble());
        OpticalFlow.CalcOpticalFlowPyrLK(prev, next, prevPts, nextPts, status, err, Win(p), p.GetProperty("maxLevel").GetInt32(), criteria, flags, p.GetProperty("minEig").GetDouble());

        Point2f[] expPts = Points(c.GetProperty("nextPts"));
        byte[] expStatus = c.GetProperty("status").EnumerateArray().Select(v => (byte)v.GetInt32()).ToArray();
        int[] errMask = c.GetProperty("errMask").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        uint[] expErr = c.GetProperty("err").EnumerateArray().Select(v => v.GetUInt32()).ToArray();
        for (int i = 0; i < prevPts.Length; i++)
        {
            Assert.True(expStatus[i] == status[i], $"'{name}' point {i}: status expected {expStatus[i]}, got {status[i]}.");
            Assert.True(
                Bits(expPts[i].X) == Bits(nextPts[i].X) && Bits(expPts[i].Y) == Bits(nextPts[i].Y),
                $"'{name}' point {i}: expected ({expPts[i].X:R}, {expPts[i].Y:R}), got ({nextPts[i].X:R}, {nextPts[i].Y:R}).");
            if (errMask[i] != 0)
            {
                Assert.True(expErr[i] == Bits(err[i]), $"'{name}' point {i}: err expected {BitConverter.UInt32BitsToSingle(expErr[i]):R}, got {err[i]:R}.");
            }
        }
    }

    private static Size Win(JsonElement p) => new(p.GetProperty("win")[0].GetInt32(), p.GetProperty("win")[1].GetInt32());

    private static Point2f[] Points(JsonElement bits)
    {
        uint[] v = bits.EnumerateArray().Select(e => e.GetUInt32()).ToArray();
        var pts = new Point2f[v.Length / 2];
        for (int i = 0; i < pts.Length; i++)
        {
            pts[i] = new Point2f(BitConverter.UInt32BitsToSingle(v[2 * i]), BitConverter.UInt32BitsToSingle(v[(2 * i) + 1]));
        }

        return pts;
    }

    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    private static void CompareBytes(string what, byte[] bytes, JsonElement expected)
    {
        if (expected.TryGetProperty("sha", out JsonElement sha))
        {
            Assert.True(sha.GetString() == Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), $"{what}: hash mismatch.");
            return;
        }

        byte[] exp = Convert.FromBase64String(Arrays[expected.GetProperty("array").GetInt32()].GetProperty("data").GetString()!);
        Assert.Equal(exp.Length, bytes.Length);
        int k = exp.AsSpan().CommonPrefixLength(bytes);
        Assert.True(k == exp.Length, $"{what}: first difference at byte {k}.");
    }

    private static Mat<T> ToMat<T>(int index)
        where T : unmanaged
    {
        JsonElement a = Arrays[index];
        byte[] bytes = Convert.FromBase64String(a.GetProperty("data").GetString()!);
        T[] values = MemoryMarshal.Cast<byte, T>(bytes).ToArray();
        return new Mat<T>(values, a.GetProperty("width").GetInt32(), a.GetProperty("height").GetInt32(), a.GetProperty("channels").GetInt32());
    }

    private static JsonElement Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "lk_golden.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }
}
