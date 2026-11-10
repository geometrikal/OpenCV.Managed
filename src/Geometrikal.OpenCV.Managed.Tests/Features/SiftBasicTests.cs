namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>Analytic and consistency tests of <see cref="Sift"/>.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class SiftBasicTests
{
    [Fact]
    public void Constructor_Defaults_MatchOpenCv()
    {
        var sift = new Sift();
        Assert.Equal(0, sift.NFeatures);
        Assert.Equal(3, sift.NOctaveLayers);
        Assert.Equal(0.04, sift.ContrastThreshold);
        Assert.Equal(10, sift.EdgeThreshold);
        Assert.Equal(1.6, sift.Sigma);
        Assert.Equal(SiftDescriptorType.Float32, sift.DescriptorType);
        Assert.True(sift.EnablePreciseUpscale);
        Assert.Equal(128, Sift.DescriptorSize);
    }

    [Fact]
    public void DetectAndCompute_UniformImage_ReturnsNoKeypointsAndNullDescriptors()
    {
        var image = new Mat<byte>(64, 64);
        image.Fill(128);
        List<KeyPoint> kps = new Sift().DetectAndCompute(image, null, out Mat<float>? d);
        Assert.Empty(kps);
        Assert.Null(d);
    }

    [Fact]
    public void DetectAndCompute_ZeroMask_RemovesAllKeypoints()
    {
        Mat<byte> image = SiftTestImages.Make(128, 128, 1, 3);
        var mask = new Mat<byte>(128, 128);
        List<KeyPoint> kps = new Sift().DetectAndCompute(image, mask, out Mat<float>? d);
        Assert.Empty(kps);
        Assert.Null(d);
    }

    [Fact]
    public void DetectAndCompute_WrongDescriptorType_Throws()
    {
        Mat<byte> image = SiftTestImages.Make(64, 64, 1, 1);
        Assert.Throws<InvalidOperationException>(() => new Sift().DetectAndCompute(image, null, out Mat<byte>? _));
        Assert.Throws<InvalidOperationException>(
            () => new Sift(descriptorType: SiftDescriptorType.Byte).DetectAndCompute(image, null, out Mat<float>? _));
    }

    [Fact]
    public void DetectAndCompute_TwoChannelImage_Throws()
    {
        var image = new Mat<byte>(32, 32, 2);
        Assert.Throws<ArgumentException>(() => new Sift().Detect(image));
    }

    [Fact]
    public void Detect_EqualsKeypointsOfDetectAndCompute()
    {
        Mat<byte> image = SiftTestImages.Make(160, 120, 1, 21);
        var sift = new Sift();
        List<KeyPoint> a = sift.Detect(image);
        List<KeyPoint> b = sift.DetectAndCompute(image, null, out Mat<float>? _);
        Assert.Equal(a, b);
    }

    [Fact]
    public void ByteDescriptors_EqualRoundedFloatDescriptors()
    {
        Mat<byte> image = SiftTestImages.Make(160, 120, 1, 22);
        List<KeyPoint> kf = new Sift().DetectAndCompute(image, null, out Mat<float>? df);
        List<KeyPoint> kb = new Sift(descriptorType: SiftDescriptorType.Byte).DetectAndCompute(image, null, out Mat<byte>? db);
        Assert.Equal(kf, kb);
        Assert.Equal(df!.ToArray().Select(v => (byte)v), db!.ToArray());
        Assert.All(df.ToArray(), v => Assert.Equal(MathF.Round(v), v));
    }

    [Fact]
    public void Compute_DetectedKeypoints_EqualsDetectAndComputeDescriptors()
    {
        Mat<byte> image = SiftTestImages.Make(160, 120, 1, 23);
        var sift = new Sift();
        List<KeyPoint> kps = sift.DetectAndCompute(image, null, out Mat<float>? expected);
        var copy = kps.ToList();
        sift.Compute(image, kps, out Mat<float>? actual);
        Assert.Equal(copy, kps);
        Assert.Equal(expected!.ToArray(), actual!.ToArray());
    }

    [Fact]
    public void Compute_EmptyKeypoints_ReturnsNull()
    {
        Mat<byte> image = SiftTestImages.Make(64, 64, 1, 1);
        new Sift().Compute(image, new List<KeyPoint>(), out Mat<float>? d);
        Assert.Null(d);
    }

    [Fact]
    public void Compute_OctaveBelowMinusOne_Throws()
    {
        Mat<byte> image = SiftTestImages.Make(64, 64, 1, 1);
        var kp = new KeyPoint(10, 10, 3, 0, 0, 254 | (1 << 8));
        Assert.Throws<ArgumentException>(() => new Sift().Compute(image, new[] { kp }, out Mat<float>? _));
    }

    [Fact]
    public void Bgra_EqualsGrayConvertedImage()
    {
        Mat<byte> bgr = SiftTestImages.Make(128, 96, 3, 30);
        Mat<byte> bgra = ColorConversions.CvtColor(bgr, ColorConversionCode.Bgr2Bgra);
        Mat<byte> gray = ColorConversions.CvtColor(bgr, ColorConversionCode.Bgr2Gray);
        var sift = new Sift();
        List<KeyPoint> a = sift.DetectAndCompute(bgra, null, out Mat<float>? da);
        List<KeyPoint> b = sift.DetectAndCompute(gray, null, out Mat<float>? db);
        Assert.Equal(b, a);
        Assert.Equal(db!.ToArray(), da!.ToArray());
    }

    [Fact]
    public void RoiInput_EqualsPackedImage()
    {
        Mat<byte> big = SiftTestImages.Make(200, 150, 1, 31);
        Mat<byte> roi = big.Roi(13, 7, 150, 120);
        Mat<byte> packed = roi.Clone();
        var sift = new Sift();
        List<KeyPoint> a = sift.DetectAndCompute(roi, null, out Mat<float>? da);
        List<KeyPoint> b = sift.DetectAndCompute(packed, null, out Mat<float>? db);
        Assert.Equal(b, a);
        Assert.Equal(db!.ToArray(), da!.ToArray());
    }

    [Fact]
    public void NFeatures_KeepsStrongestSortedByResponse()
    {
        Mat<byte> image = SiftTestImages.Make(200, 150, 1, 32);
        List<KeyPoint> all = new Sift().Detect(image);
        List<KeyPoint> best = new Sift(nFeatures: 20).Detect(image);
        Assert.True(best.Count >= 20);
        float boundary = all.Select(k => k.Response).OrderByDescending(r => r).ElementAt(19);
        Assert.All(best, k => Assert.True(k.Response >= boundary));
        Assert.Equal(best.OrderByDescending(k => k.Response).Select(k => k.Response), best.Select(k => k.Response));
    }

    [Fact]
    public void Output_DoesNotDependOnThreadCount()
    {
        Mat<byte> image = SiftTestImages.Make(512, 384, 1, 33);
        int saved = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            List<KeyPoint> k1 = new Sift().DetectAndCompute(image, null, out Mat<float>? d1);
            CvParallel.MaxDegreeOfParallelism = 8;
            List<KeyPoint> k8 = new Sift().DetectAndCompute(image, null, out Mat<float>? d8);
            Assert.Equal(k1, k8);
            Assert.Equal(d1!.ToArray(), d8!.ToArray());
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void Keypoints_HaveSiftOctaveEncodingAndValidAngles()
    {
        Mat<byte> image = SiftTestImages.Make(128, 128, 1, 34);
        List<KeyPoint> kps = new Sift().Detect(image);
        Assert.NotEmpty(kps);
        Assert.All(kps, k =>
        {
            int octave = (sbyte)(k.Octave & 255);
            int layer = (k.Octave >> 8) & 255;
            Assert.InRange(octave, -1, 8);
            Assert.InRange(layer, 1, 3);
            Assert.InRange(k.Angle, 0f, 360f);
            Assert.True(k.Angle < 360f);
            Assert.True(k.Response > 0);
            Assert.Equal(-1, k.ClassId);
        });
    }
}
