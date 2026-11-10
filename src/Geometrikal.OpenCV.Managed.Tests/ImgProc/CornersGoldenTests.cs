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
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/corners_golden.json</c>
/// (cornerMinEigenVal, cornerHarris, cornerEigenValsAndVecs, goodFeaturesToTrack, cornerSubPix).
/// Outputs must match bit for bit. Regenerate with <c>tools/generate_corners_golden.py</c>.
/// </summary>
public sealed class CornersGoldenTests
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
            new[] { "eigenValsVecs", "gftt", "harris", "minEig", "subpix" },
            Cases.Values.Select(c => c.GetProperty("op").GetString()!).Distinct().OrderBy(o => o, StringComparer.Ordinal));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Corners_MatchOpenCvGolden(string name)
    {
        JsonElement c = Cases[name];
        bool isByte = Arrays[c.GetProperty("input").GetInt32()].GetProperty("dtype").GetString() == "u8";
        if (isByte)
        {
            Run<byte>(name, c);
        }
        else
        {
            Run<float>(name, c);
        }
    }

    private static void Run<T>(string name, JsonElement c)
        where T : unmanaged
    {
        Mat<T> src = ToMat<T>(c.GetProperty("input").GetInt32());
        JsonElement p = c.GetProperty("params");
        string op = c.GetProperty("op").GetString()!;
        switch (op)
        {
            case "minEig":
            case "harris":
            case "eigenValsVecs":
            {
                int block = p.GetProperty("blockSize").GetInt32();
                int ksize = p.GetProperty("ksize").GetInt32();
                BorderType border = Enum.Parse<BorderType>(p.GetProperty("border").GetString()!);
                Mat<float> result = op switch
                {
                    "minEig" => Corners.CornerMinEigenVal(src, block, ksize, border),
                    "harris" => Corners.CornerHarris(src, block, ksize, p.GetProperty("k").GetDouble(), border),
                    _ => Corners.CornerEigenValsAndVecs(src, block, ksize, border),
                };
                CompareArray(name, result, c.GetProperty("outputs")[0]);
                break;
            }

            case "gftt":
            {
                Mat<byte>? mask = p.GetProperty("mask").ValueKind == JsonValueKind.Null ? null : ToMat<byte>(p.GetProperty("mask").GetInt32());
                (Point2f[] corners, float[] quality) = Corners.GoodFeaturesToTrackWithQuality(
                    src,
                    p.GetProperty("maxCorners").GetInt32(),
                    p.GetProperty("qualityLevel").GetDouble(),
                    p.GetProperty("minDistance").GetDouble(),
                    mask,
                    p.GetProperty("blockSize").GetInt32(),
                    p.GetProperty("gradientSize").GetInt32(),
                    p.GetProperty("useHarrisDetector").GetBoolean(),
                    p.GetProperty("k").GetDouble());
                uint[][] exp = c.GetProperty("corners").EnumerateArray().Select(e => e.EnumerateArray().Select(v => v.GetUInt32()).ToArray()).ToArray();
                Assert.True(exp.Length == corners.Length, $"'{name}': expected {exp.Length} corners, got {corners.Length}.");
                for (int i = 0; i < exp.Length; i++)
                {
                    uint[] got = [Bits(corners[i].X), Bits(corners[i].Y), Bits(quality[i])];
                    Assert.True(exp[i].SequenceEqual(got), $"'{name}' corner {i}: expected [{string.Join(",", exp[i])}], got [{string.Join(",", got)}].");
                }

                break;
            }

            default:
            {
                Point2f[] pts = c.GetProperty("start").EnumerateArray()
                    .Select(e => new Point2f(BitConverter.UInt32BitsToSingle(e[0].GetUInt32()), BitConverter.UInt32BitsToSingle(e[1].GetUInt32())))
                    .ToArray();
                JsonElement win = p.GetProperty("win"), zero = p.GetProperty("zero");
                var criteria = new TermCriteria((TermCriteriaType)p.GetProperty("type").GetInt32(), p.GetProperty("maxCount").GetInt32(), p.GetProperty("epsilon").GetDouble());
                Corners.CornerSubPix(src, pts, new Size(win[0].GetInt32(), win[1].GetInt32()), new Size(zero[0].GetInt32(), zero[1].GetInt32()), criteria);
                JsonElement[] exp = c.GetProperty("refined").EnumerateArray().ToArray();
                Assert.Equal(exp.Length, pts.Length);
                for (int i = 0; i < pts.Length; i++)
                {
                    uint[] e = [exp[i][0].GetUInt32(), exp[i][1].GetUInt32()];
                    uint[] got = [Bits(pts[i].X), Bits(pts[i].Y)];
                    Assert.True(e.SequenceEqual(got), $"'{name}' point {i}: expected ({BitConverter.UInt32BitsToSingle(e[0])}, {BitConverter.UInt32BitsToSingle(e[1])}), got ({pts[i].X}, {pts[i].Y}).");
                }

                break;
            }
        }
    }

    private static uint Bits(float v) => BitConverter.SingleToUInt32Bits(v);

    private static void CompareArray(string name, Mat<float> actual, JsonElement expected)
    {
        byte[] bytes = MemoryMarshal.AsBytes(actual.ToArray().AsSpan()).ToArray();
        if (expected.TryGetProperty("sha", out JsonElement sha))
        {
            Assert.Equal(sha.GetString(), Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());
            return;
        }

        byte[] exp = Convert.FromBase64String(Arrays[expected.GetProperty("array").GetInt32()].GetProperty("data").GetString()!);
        Assert.Equal(exp.Length, bytes.Length);
        if (!exp.AsSpan().SequenceEqual(bytes))
        {
            int k = 0;
            while (exp[k] == bytes[k])
            {
                k++;
            }

            int e = k / 4;
            Assert.Fail($"Golden mismatch for '{name}' at element {e}: expected {BitConverter.ToSingle(exp, e * 4)}, got {BitConverter.ToSingle(bytes, e * 4)}.");
        }
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
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "corners_golden.json");
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }
}
