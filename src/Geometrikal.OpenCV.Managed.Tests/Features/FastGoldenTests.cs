namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Replays the cv2 5.0.0 FAST reference outputs in <c>TestData/orb_golden.json</c>. Keypoints
/// (x, y, response) must match in order. Regenerate with <c>tools/generate_orb_golden.py</c>.
/// </summary>
public sealed class FastGoldenTests
{
    public static IEnumerable<object[]> FastCases()
        => OrbGolden.Cases.Where(c => !c.Suspect && c.Op == "fast").Select(c => new object[] { c.Name });

    [Fact]
    public void GoldenFile_CoversEveryTypeAndMode()
    {
        var fast = OrbGolden.Cases.Where(c => c.Op == "fast").ToList();
        Assert.True(fast.Count > 100);
        Assert.Equal(
            new[] { "Type5_8", "Type7_12", "Type9_16" },
            fast.Select(c => c.Params.GetProperty("type").GetString()!).Distinct().OrderBy(o => o, StringComparer.Ordinal));
        Assert.Contains(fast, c => !c.Params.GetProperty("nonmax").GetBoolean());
        Assert.Contains(fast, c => c.Mask is not null);
    }

    [Theory]
    [MemberData(nameof(FastCases))]
    public void Detect_MatchesOpenCvGolden(string name)
    {
        OrbGolden.Case c = OrbGolden.Cases.Single(x => x.Name == name);
        Mat<byte> image = OrbGolden.BuildImage(c.Image);
        Mat<byte>? mask = OrbGolden.BuildMask(c.Mask, image.Width, image.Height);
        var detector = new FastFeatureDetector(
            c.Params.GetProperty("threshold").GetInt32(),
            c.Params.GetProperty("nonmax").GetBoolean(),
            Enum.Parse<FastDetectorType>(c.Params.GetProperty("type").GetString()!));

        KeyPoint[] kps = detector.Detect(image, mask);
        Assert.Equal(c.Count, kps.Length);
        int[] act = new int[kps.Length * 3];
        for (int i = 0; i < kps.Length; i++)
        {
            Assert.Equal(7f, kps[i].Size);
            Assert.Equal(-1f, kps[i].Angle);
            Assert.Equal(0, kps[i].Octave);
            act[3 * i] = (int)kps[i].Pt.X;
            act[(3 * i) + 1] = (int)kps[i].Pt.Y;
            act[(3 * i) + 2] = (int)kps[i].Response;
            Assert.Equal(kps[i].Pt.X, act[3 * i]);
            Assert.Equal(kps[i].Pt.Y, act[(3 * i) + 1]);
            Assert.Equal(kps[i].Response, act[(3 * i) + 2]);
        }

        byte[] bytes = MemoryMarshal.AsBytes(act.AsSpan()).ToArray();
        if (c.KeypointsSha is not null)
        {
            Assert.Equal(c.KeypointsSha, Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());
            return;
        }

        int[] exp = MemoryMarshal.Cast<byte, int>(Convert.FromBase64String(c.Keypoints!)).ToArray();
        for (int i = 0; i < c.Count; i++)
        {
            Assert.True(
                exp[3 * i] == act[3 * i] && exp[(3 * i) + 1] == act[(3 * i) + 1] && exp[(3 * i) + 2] == act[(3 * i) + 2],
                $"{name}: keypoint {i} expected ({exp[3 * i]}, {exp[(3 * i) + 1]}, {exp[(3 * i) + 2]}) actual ({act[3 * i]}, {act[(3 * i) + 1]}, {act[(3 * i) + 2]}).");
        }
    }
}
