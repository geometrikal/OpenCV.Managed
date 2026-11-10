namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Linq;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;
using Geometrikal.OpenCV.Managed.Features.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>Analytic and invariance tests of <see cref="Fast"/>, <see cref="FastFeatureDetector"/> and <see cref="Orb"/>.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class OrbBasicTests
{
    private static Mat<byte> Noise(int w, int h, int seed)
        => OrbGolden.BuildImage(JsonDocument.Parse(
            $"{{\"kind\":\"noise\",\"width\":{w},\"height\":{h},\"seed\":{seed},\"radius\":2,\"gain\":4,\"jitter\":0}}").RootElement);

    [Fact]
    public void Fast_TinyImage_ReturnsNoKeypoints()
    {
        Assert.Empty(Fast.Detect(new Mat<byte>(6, 50), 10));
        Assert.Empty(Fast.Detect(new Mat<byte>(50, 6), 10));
    }

    [Fact]
    public void Fast_SingleBrightPixel_IsCornerWithExpectedScore()
    {
        var img = new Mat<byte>(15, 15);
        img.Fill(50);
        img.At(7, 7) = 200;
        KeyPoint[] kps = Fast.Detect(img, 20);
        KeyPoint k = Assert.Single(kps);
        Assert.Equal(new Point2f(7, 7), k.Pt);
        Assert.Equal(149f, k.Response);
        Assert.Equal(7f, k.Size);
        Assert.Equal(-1f, k.Angle);

        KeyPoint all = Assert.Single(Fast.Detect(img, 20, nonmaxSuppression: false));
        Assert.Equal(0f, all.Response);
        Assert.Empty(Fast.Detect(img, 150));
    }

    [Fact]
    public void Fast_Type5_8_RequiresAllEightNeighbours()
    {
        var img = new Mat<byte>(15, 15);
        img.Fill(50);
        img.At(7, 7) = 200;
        img.At(6, 7) = 200;
        Assert.DoesNotContain(Fast.Detect(img, 20, true, FastDetectorType.Type5_8), k => k.Pt == new Point2f(7, 7));
    }

    [Fact]
    public void Fast_ThresholdIsClamped()
    {
        Mat<byte> img = Noise(120, 90, 3);
        Assert.Equal(Fast.Detect(img, 255), Fast.Detect(img, 1000));
        Assert.Equal(Fast.Detect(img, 0), Fast.Detect(img, -5));
    }

    [Fact]
    public void Fast_Roi_EqualsCopy()
    {
        Mat<byte> img = Noise(200, 150, 4);
        Mat<byte> roi = img.Roi(17, 9, 150, 120);
        Assert.Equal(Fast.Detect(roi.Clone(), 15, true, FastDetectorType.Type7_12), Fast.Detect(roi, 15, true, FastDetectorType.Type7_12));
    }

    [Fact]
    public void FastFeatureDetector_ColorInput_EqualsGray()
    {
        Mat<byte> gray = Noise(100, 80, 5);
        Mat<byte> bgr = ColorConversions.CvtColor(gray, ColorConversionCode.Gray2Bgr);
        var det = new FastFeatureDetector(12);
        Assert.Equal(det.Detect(gray), det.Detect(bgr));
    }

    [Fact]
    public void FastFeatureDetector_MaskSizeMismatch_Throws()
        => Assert.Throws<ArgumentException>(() => new FastFeatureDetector().Detect(new Mat<byte>(20, 20), new Mat<byte>(10, 20)));

    [Fact]
    public void Fast_Output_DoesNotDependOnThreadCount()
    {
        Mat<byte> img = Noise(1031, 517, 6);
        foreach (FastDetectorType t in Enum.GetValues<FastDetectorType>())
        {
            KeyPoint[] one = WithThreads(1, () => Fast.Detect(img, 10, true, t));
            KeyPoint[] many = WithThreads(8, () => Fast.Detect(img, 10, true, t));
            Assert.NotEmpty(one);
            Assert.Equal(one, many);
        }
    }

    [Fact]
    public void Orb_Output_DoesNotDependOnThreadCount()
    {
        Mat<byte> img = Noise(800, 600, 7);
        var orb = new Orb(1500);
        (KeyPoint[] k1, Mat<byte>? d1) = WithThreads(1, () => orb.DetectAndCompute(img));
        (KeyPoint[] k8, Mat<byte>? d8) = WithThreads(8, () => orb.DetectAndCompute(img));
        Assert.Equal(k1, k8);
        Assert.Equal(d1!.ToArray(), d8!.ToArray());
    }

    [Fact]
    public void Orb_KeypointsAreGroupedByLevel()
    {
        (KeyPoint[] kps, Mat<byte>? desc) = new Orb().DetectAndCompute(Noise(320, 240, 8));
        Assert.Equal(kps.Length, desc!.Height);
        Assert.Equal(32, desc.Width);
        for (int i = 1; i < kps.Length; i++)
        {
            Assert.True(kps[i].Octave >= kps[i - 1].Octave);
        }

        Assert.True(kps.Length <= 500);
    }

    [Fact]
    public void Orb_ComputeOnDetectedKeypoints_EqualsDetectAndCompute()
    {
        Mat<byte> img = Noise(300, 200, 9);
        var orb = new Orb(300);
        (KeyPoint[] kps, Mat<byte>? desc) = orb.DetectAndCompute(img);
        (KeyPoint[] kps2, Mat<byte>? desc2) = orb.Compute(img, kps);
        Assert.Equal(kps, kps2);
        Assert.Equal(desc!.ToArray(), desc2!.ToArray());
    }

    [Fact]
    public void Orb_EmptyResults_GiveNullDescriptors()
    {
        var flat = new Mat<byte>(100, 100);
        flat.Fill(128);
        (KeyPoint[] kps, Mat<byte>? desc) = new Orb().DetectAndCompute(flat);
        Assert.Empty(kps);
        Assert.Null(desc);
        (kps, desc) = new Orb().Compute(flat, Array.Empty<KeyPoint>());
        Assert.Empty(kps);
        Assert.Null(desc);
    }

    [Fact]
    public void Orb_InvalidSettings_Throw()
    {
        Mat<byte> img = Noise(100, 100, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => new Orb(firstLevel: -1));
        Assert.Throws<ArgumentException>(() => new Orb(scaleFactor: 1f).Detect(img));
        Assert.Throws<ArgumentException>(() => new Orb(patchSize: 1).Detect(img));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Orb(wtaK: 5).DetectAndCompute(img));
        Assert.Throws<ArgumentException>(() => new Orb().Detect(img, new Mat<byte>(50, 100)));
        Assert.Throws<ArgumentException>(() => new Orb().Compute(img, new[] { new KeyPoint(50, 50, 31, 0, 0, -1) }));
    }

    [Fact]
    public void Orb_FeaturesPerLevel_IsGeometricAndSumsToTotal()
    {
        int[] n = OrbEngine.FeaturesPerLevel(500, 1.2f, 8);
        Assert.Equal(500, n.Sum());
        Assert.Equal(new[] { 109, 90, 75, 63, 52, 44, 36, 31 }, n);
    }

    [Fact]
    public void Orb_Umax_MatchesCircularPatch()
        => Assert.Equal(new[] { 15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3, 0 }, OrbEngine.Umax(15));

    [Fact]
    public void Orb_FastAtan2_MatchesQuadrants()
    {
        Assert.Equal(0f, OrbEngine.FastAtan2(0, 1));
        Assert.InRange(OrbEngine.FastAtan2(1, 0), 89.99f, 90.01f);
        Assert.InRange(OrbEngine.FastAtan2(0, -1), 179.99f, 180.01f);
        Assert.InRange(OrbEngine.FastAtan2(-1, 0), 269.99f, 270.01f);
        Assert.InRange(OrbEngine.FastAtan2(1, 1), 44.99f, 45.01f);
    }

    [Fact]
    public void OrbRng_MatchesOpenCvSequence()
    {
        // RNG(0xffffffff).next(): state = 0xffffffff * 4164903690 + 0.
        var rng = new OrbRng(0);
        Assert.Equal(unchecked((uint)(0xffffffffUL * 4164903690UL)), rng.Next());
        var r2 = new OrbRng(1);
        Assert.Equal(4164903690u, r2.Next());
        Assert.Equal(-3, new OrbRng(1).Uniform(-3, -3));
    }

    private static T WithThreads<T>(int n, Func<T> f)
    {
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = n;
            return f();
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }
}
