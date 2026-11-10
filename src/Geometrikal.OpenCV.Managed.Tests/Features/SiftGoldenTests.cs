namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Replays the cv2 5.0.0 SIFT reference outputs in <c>TestData/sift_golden.json</c>. Keypoints
/// (position, size, angle, response, octave, class) and descriptors must match exactly.
/// Regenerate with <c>tools/generate_sift_golden.py</c>.
/// </summary>
public sealed class SiftGoldenTests
{
    internal static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndCoversModes()
    {
        Assert.Equal("5.0.0", Data.Opencv);
        Assert.True(Data.Cases.Count >= 22);
        Assert.Equal(
            new[] { "compute", "computeOctave0", "detect", "detectAndCompute" },
            Data.Cases.Select(c => c.Mode).Distinct().OrderBy(m => m, StringComparer.Ordinal));
        Assert.Contains(Data.Cases, c => c.DescriptorType == "u8");
        Assert.Contains(Data.Cases, c => c.Mask);
        Assert.Contains(Data.Cases, c => c.NFeatures > 0);
        Assert.Contains(Data.Cases, c => !c.PreciseUpscale);
        Assert.Contains(Data.Cases, c => c.Channels == 3);
        Assert.Contains(Data.Cases, c => c.Channels == 4);
        Assert.Contains(Data.Cases, c => c.DescriptorsSha is not null);
    }

    /// <summary>
    /// <c>FastMath</c> equals cv2 <c>exp</c> (<c>hal::exp32f</c>), <c>phase</c> in degrees
    /// (<c>hal::fastAtan32f</c>) and <c>magnitude</c> (<c>hal::magnitude32f</c>) bit for bit.
    /// </summary>
    [Fact]
    public void FastMath_MatchesOpenCvGolden()
    {
        float[] x = Floats(Data.Math.X);
        float[] y = Floats(Data.Math.Y);
        float[] w = Floats(Data.Math.W);
        var exp = new float[w.Length];
        var phase = new float[x.Length];
        var mag = new float[x.Length];
        FastMath.Exp32f(w, exp);
        FastMath.FastAtan32fDegrees(y, x, phase);
        FastMath.Magnitude32f(x, y, mag);
        Assert.Equal(Floats(Data.Math.Exp).Select(BitConverter.SingleToInt32Bits), exp.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(Floats(Data.Math.PhaseDeg).Select(BitConverter.SingleToInt32Bits), phase.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(Floats(Data.Math.Magnitude).Select(BitConverter.SingleToInt32Bits), mag.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(phase[5], FastMath.FastAtan2(y[5], x[5]));
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Sift_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Data.Cases.Single(x => x.Name == name);
        Mat<byte> image = SiftTestImages.Make(c.Width, c.Height, c.Channels, c.Seed);
        Assert.Equal(c.ImageSha, Convert.ToHexString(SHA256.HashData(image.GetReadOnlySpan())).ToLowerInvariant());
        Mat<byte>? mask = c.Mask ? SiftTestImages.MakeMask(c.Width, c.Height) : null;

        var type = c.DescriptorType == "u8" ? SiftDescriptorType.Byte : SiftDescriptorType.Float32;
        var sift = new Sift(c.NFeatures, c.NOctaveLayers, c.ContrastThreshold, c.EdgeThreshold, c.Sigma, type, c.PreciseUpscale);

        List<KeyPoint> keypoints;
        byte[] descriptors = [];
        if (c.Mode == "detect")
        {
            keypoints = sift.Detect(image, mask);
        }
        else if (c.Mode is "compute" or "computeOctave0")
        {
            keypoints = c.InputKeypoints!.Select(ToKeyPoint).ToList();
            sift.Compute(image, keypoints, out Mat<float>? d);
            descriptors = Bytes(d);
        }
        else if (type == SiftDescriptorType.Byte)
        {
            keypoints = sift.DetectAndCompute(image, mask, out Mat<byte>? d);
            descriptors = d is null ? [] : d.ToArray();
        }
        else
        {
            keypoints = sift.DetectAndCompute(image, mask, out Mat<float>? d);
            descriptors = Bytes(d);
        }

        List<KeyPoint> expected = c.Keypoints.Select(ToKeyPoint).ToList();
        if (c.DescriptorsSha is not null)
        {
            Assert.Equal(expected, keypoints);
            Assert.Equal(c.DescriptorsSha, Convert.ToHexString(SHA256.HashData(descriptors)).ToLowerInvariant());
            return;
        }

        byte[] expectedDesc = Convert.FromBase64String(c.Descriptors);
        int rowBytes = expectedDesc.Length / Math.Max(1, expected.Count);
        List<(KeyPoint Kp, byte[] Row)> exp = Pair(expected, expectedDesc, rowBytes);
        List<(KeyPoint Kp, byte[] Row)> act = Pair(keypoints, descriptors, rowBytes);
        if (c.NFeatures > 0)
        {
            // retainBest order is implementation-defined in OpenCV.
            exp = exp.OrderBy(p => p.Kp, KeyPointOrder.Instance).ToList();
            act = act.OrderBy(p => p.Kp, KeyPointOrder.Instance).ToList();
        }

        Assert.Equal(exp.Count, act.Count);
        for (int i = 0; i < exp.Count; i++)
        {
            Assert.True(exp[i].Kp == act[i].Kp, $"keypoint {i}: expected {exp[i].Kp}, actual {act[i].Kp}");
            Assert.True(exp[i].Row.AsSpan().SequenceEqual(act[i].Row), $"descriptor {i} differs");
        }
    }

    private static List<(KeyPoint Kp, byte[] Row)> Pair(List<KeyPoint> kps, byte[] desc, int rowBytes)
    {
        var list = new List<(KeyPoint, byte[])>(kps.Count);
        for (int i = 0; i < kps.Count; i++)
        {
            byte[] row = desc.Length == 0 ? [] : desc.AsSpan(i * rowBytes, rowBytes).ToArray();
            list.Add((kps[i], row));
        }

        return list;
    }

    private static float[] Floats(string b64)
        => MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(b64)).ToArray();

    private static byte[] Bytes(Mat<float>? d)
        => d is null ? [] : MemoryMarshal.AsBytes(d.ToArray().AsSpan()).ToArray();

    internal static KeyPoint ToKeyPoint(double[] v)
        => new((float)v[0], (float)v[1], (float)v[2], (float)v[3], (float)v[4], (int)v[5], (int)v[6]);

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "sift_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    private sealed class KeyPointOrder : IComparer<KeyPoint>
    {
        public static readonly KeyPointOrder Instance = new();

        public int Compare(KeyPoint a, KeyPoint b)
        {
            int r = a.Pt.X.CompareTo(b.Pt.X);
            if (r == 0)
            {
                r = a.Pt.Y.CompareTo(b.Pt.Y);
            }

            if (r == 0)
            {
                r = b.Size.CompareTo(a.Size);
            }

            if (r == 0)
            {
                r = a.Angle.CompareTo(b.Angle);
            }

            if (r == 0)
            {
                r = b.Response.CompareTo(a.Response);
            }

            return r != 0 ? r : b.Octave.CompareTo(a.Octave);
        }
    }

    internal sealed class GoldenFile
    {
        public string Opencv { get; set; } = "";

        public MathData Math { get; set; } = new();

        public List<GoldenCase> Cases { get; set; } = [];
    }

    internal sealed class MathData
    {
        public string X { get; set; } = "";

        public string Y { get; set; } = "";

        public string W { get; set; } = "";

        public string Exp { get; set; } = "";

        public string PhaseDeg { get; set; } = "";

        public string Magnitude { get; set; } = "";
    }

    internal sealed class GoldenCase
    {
        public string Name { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public int Seed { get; set; }

        public bool Mask { get; set; }

        public string Mode { get; set; } = "";

        public int NFeatures { get; set; }

        public int NOctaveLayers { get; set; }

        public double ContrastThreshold { get; set; }

        public double EdgeThreshold { get; set; }

        public double Sigma { get; set; }

        public string DescriptorType { get; set; } = "";

        public bool PreciseUpscale { get; set; }

        public string ImageSha { get; set; } = "";

        public List<double[]>? InputKeypoints { get; set; }

        public List<double[]> Keypoints { get; set; } = [];

        public string Descriptors { get; set; } = "";

        public string? DescriptorsSha { get; set; }
    }
}

/// <summary>Synthetic textured images of <c>tools/generate_sift_golden.py</c> (integer formula).</summary>
internal static class SiftTestImages
{
    public static uint Hash(long x, long y, long s)
    {
        uint h = unchecked((uint)(x * 73856093) ^ (uint)(y * 19349663) ^ (uint)(s * 83492791));
        h ^= h >> 13;
        h = unchecked(h * 0x5BD1E995u);
        h ^= h >> 15;
        return h;
    }

    public static Mat<byte> Make(int w, int h, int channels, int seed)
    {
        var img = new Mat<byte>(w, h, channels);
        for (int ch = 0; ch < channels; ch++)
        {
            int[] plane = Plane(w, h, seed + (10 * ch));
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    img.At(y, x, ch) = (byte)plane[(y * w) + x];
                }
            }
        }

        return img;
    }

    public static Mat<byte> MakeMask(int w, int h)
    {
        var m = new Mat<byte>(w, h);
        for (int y = 0; y < (3 * h) / 4; y++)
        {
            for (int x = w / 4; x < w; x++)
            {
                m.At(y, x) = 255;
            }
        }

        return m;
    }

    private static int[] Plane(int w, int h, int seed)
    {
        var v = new int[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int b8 = Bilinear(x, y, 8, seed, 6);
                int b32 = Bilinear(x, y, 32, seed + 1, 10);
                v[(y * w) + x] = (((2 * b8) + b32) / 3) + (int)(Hash(x, y, seed + 2) & 15) - 8;
            }
        }

        for (int k = 0; k < 6; k++)
        {
            uint r = Hash(k, 0, seed + 3);
            int x0 = (int)(r % (uint)w);
            int y0 = (int)((r >> 8) % (uint)h);
            int rw = 4 + (int)((r >> 16) % 24);
            int rh = 4 + (int)((r >> 24) % 24);
            int val = (int)(Hash(k, 1, seed + 3) & 255);
            for (int y = y0; y < Math.Min(h, y0 + rh); y++)
            {
                for (int x = x0; x < Math.Min(w, x0 + rw); x++)
                {
                    v[(y * w) + x] = val;
                }
            }
        }

        for (int i = 0; i < v.Length; i++)
        {
            v[i] = Math.Clamp(v[i], 0, 255);
        }

        return v;
    }

    private static int Bilinear(int x, int y, int cell, int s, int shift)
    {
        int gx = x / cell, gy = y / cell, fx = x % cell, fy = y % cell;
        int a = (int)(Hash(gx, gy, s) & 255);
        int b = (int)(Hash(gx + 1, gy, s) & 255);
        int c = (int)(Hash(gx, gy + 1, s) & 255);
        int d = (int)(Hash(gx + 1, gy + 1, s) & 255);
        int v = (a * (cell - fx) * (cell - fy)) + (b * fx * (cell - fy)) + (c * (cell - fx) * fy) + (d * fx * fy);
        return v >> shift;
    }
}
