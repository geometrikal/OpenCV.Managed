namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching;
using Geometrikal.OpenCV.Managed.Tests.Core;

[Collection(nameof(CvParallelCollection))]
public sealed class WarperBasicTests
{
    private static Mat<float> K(float f, float cx, float cy)
    {
        var k = new Mat<float>(3, 3);
        k.At(0, 0) = f; k.At(0, 2) = cx;
        k.At(1, 1) = f; k.At(1, 2) = cy;
        k.At(2, 2) = 1;
        return k;
    }

    private static Mat<float> RotY(double angle)
    {
        var r = new Mat<float>(3, 3);
        r.At(0, 0) = (float)Math.Cos(angle); r.At(0, 2) = (float)Math.Sin(angle);
        r.At(1, 1) = 1;
        r.At(2, 0) = -(float)Math.Sin(angle); r.At(2, 2) = (float)Math.Cos(angle);
        return r;
    }

    public static TheoryData<string> Kinds => new() { "plane", "cylindrical", "spherical" };

    private static RotationWarper Create(string kind, float scale) => kind switch
    {
        "plane" => new PlaneWarper(scale),
        "cylindrical" => new CylindricalWarper(scale),
        _ => new SphericalWarper(scale),
    };

    [Theory]
    [MemberData(nameof(Kinds))]
    public void BuildMaps_EqualsPointwiseBackwardMapping(string kind)
    {
        RotationWarper w = Create(kind, 300);
        Mat<float> k = K(300, 101.5f, 60.5f);
        Mat<float> r = RotY(0.1);
        Rect rect = w.BuildMaps(new Size(203, 121), k, r, out Mat<float> mx, out Mat<float> my);
        Assert.Equal(rect.Width + 1, mx.Width);
        Assert.Equal(rect.Height + 1, mx.Height);
        for (int y = 0; y < mx.Height; y += 7)
        {
            for (int x = 0; x < mx.Width; x++)
            {
                Point2f p = w.WarpPointBackward(new Point2f(rect.X + x, rect.Y + y), k, r);
                Assert.Equal(BitConverter.SingleToUInt32Bits(p.X), BitConverter.SingleToUInt32Bits(mx.At(y, x)));
                Assert.Equal(BitConverter.SingleToUInt32Bits(p.Y), BitConverter.SingleToUInt32Bits(my.At(y, x)));
            }
        }
    }

    [Theory]
    [MemberData(nameof(Kinds))]
    public void Warp_SizeEqualsWarpRoi(string kind)
    {
        RotationWarper w = Create(kind, 200);
        Mat<float> k = K(200, 40, 30);
        Mat<float> r = RotY(-0.2);
        var src = new Mat<byte>(80, 60, 3);
        src.SetTo(Scalar.All(77));
        Rect roi = w.WarpRoi(new Size(80, 60), k, r);
        Point tl = w.Warp(src, k, r, InterpolationFlags.Linear, BorderType.Constant, out Mat<byte> dst);
        Assert.Equal(new Point(roi.X, roi.Y), tl);
        Assert.Equal(roi.Width, dst.Width);
        Assert.Equal(roi.Height, dst.Height);
        Assert.Equal(77, src.At(10, 10, 1));
    }

    [Theory]
    [MemberData(nameof(Kinds))]
    public void BuildMaps_OutputDoesNotDependOnThreadCount(string kind)
    {
        byte[] Run(int threads)
        {
            int saved = CvParallel.MaxDegreeOfParallelism;
            CvParallel.MaxDegreeOfParallelism = threads;
            try
            {
                Create(kind, 900).BuildMaps(new Size(1024, 768), K(900, 512, 384), RotY(0.3), out Mat<float> mx, out Mat<float> my);
                return [.. StitchGolden.Bytes(mx), .. StitchGolden.Bytes(my)];
            }
            finally
            {
                CvParallel.MaxDegreeOfParallelism = saved;
            }
        }

        Assert.Equal(Run(1), Run(8));
    }

    [Fact]
    public void Warpers_InvalidCameraMatrix_Throws()
    {
        var w = new PlaneWarper();
        Assert.Throws<ArgumentException>(() => w.WarpRoi(new Size(4, 4), new Mat<float>(2, 3), RotY(0)));
        Assert.Throws<ArgumentException>(() => new AffineWarper().WarpRoi(new Size(4, 4), K(1, 0, 0), new Mat<float>(3, 2)));
    }

    [Fact]
    public void AffineWarper_PureTranslation_ShiftsRoiByNegatedTranslation()
    {
        var h = new Mat<float>(3, 3);
        h.At(0, 0) = 1; h.At(1, 1) = 1; h.At(2, 2) = 1;
        h.At(0, 2) = 10; h.At(1, 2) = -5;
        Rect roi = new AffineWarper().WarpRoi(new Size(20, 10), K(1, 0, 0), h);
        Assert.Equal(new Rect(-10, 5, 20, 10), roi);
    }
}
