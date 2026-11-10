namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>Analytic and API tests for template matching, phase correlation and ECC alignment.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class AlignmentBasicTests
{
    private static Mat<float> Texture(int w, int h, int seed)
    {
        var m = new Mat<float>(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                m.At(y, x) = (float)(128 + (60 * Math.Sin((0.21 * x) + seed)) + (50 * Math.Cos((0.17 * y) - (0.05 * x * seed))) + (((x * 7) + (y * 13) + seed) % 11));
            }
        }

        return m;
    }

    private static Mat<byte> ToByte(Mat<float> m) => Conversions.ConvertTo<float, byte>(m);

    [Theory]
    [InlineData(TemplateMatchModes.SqDiff, false)]
    [InlineData(TemplateMatchModes.SqDiffNormed, false)]
    [InlineData(TemplateMatchModes.CCorrNormed, true)]
    [InlineData(TemplateMatchModes.CCoeffNormed, true)]
    public void MatchTemplate_TemplateCutFromImage_FindsItsLocation(TemplateMatchModes method, bool maximum)
    {
        Mat<byte> image = ToByte(Texture(90, 70, 3));
        Mat<byte> templ = image.Roi(31, 17, 20, 15).Clone();
        Mat<float> r = TemplateMatching.MatchTemplate(image, templ, method);
        Assert.Equal(71, r.Width);
        Assert.Equal(56, r.Height);
        MinMaxLocResult mm = Reductions.MinMaxLoc(r);
        Assert.Equal(new Point(31, 17), maximum ? mm.MaxLoc : mm.MinLoc);
    }

    [Fact]
    public void MatchTemplate_RoiWithStride_MatchesPackedImage()
    {
        Mat<float> parent = Texture(120, 90, 5);
        Mat<float> view = parent.Roi(7, 4, 100, 80);
        Mat<float> templ = parent.Roi(30, 30, 9, 11).Clone();
        Mat<float> a = TemplateMatching.MatchTemplate(view, templ, TemplateMatchModes.CCoeffNormed);
        Mat<float> b = TemplateMatching.MatchTemplate(view.Clone(), templ, TemplateMatchModes.CCoeffNormed);
        Assert.Equal(b.ToArray(), a.ToArray());
    }

    [Fact]
    public void MatchTemplate_DoesNotModifySource()
    {
        Mat<byte> image = ToByte(Texture(40, 30, 1));
        Mat<byte> templ = image.Roi(5, 5, 6, 6).Clone();
        byte[] before = image.ToArray();
        var mask = new Mat<byte>(6, 6);
        mask.Fill(255);
        _ = TemplateMatching.MatchTemplate(image, templ, TemplateMatchModes.CCoeffNormed, mask);
        Assert.Equal(before, image.ToArray());
    }

    [Fact]
    public void MatchTemplate_FullMask_EqualsUnmaskedWithinRounding()
    {
        Mat<float> image = Texture(50, 40, 2);
        Mat<float> templ = image.Roi(10, 12, 8, 7).Clone();
        var mask = new Mat<float>(8, 7);
        mask.Fill(1f);
        Mat<float> a = TemplateMatching.MatchTemplate(image, templ, TemplateMatchModes.CCorrNormed, mask);
        Mat<float> b = TemplateMatching.MatchTemplate(image, templ, TemplateMatchModes.CCorrNormed);
        for (int y = 0; y < a.Height; y++)
        {
            for (int x = 0; x < a.Width; x++)
            {
                Assert.True(Math.Abs(a.At(y, x) - b.At(y, x)) <= 1e-4);
            }
        }
    }

    [Fact]
    public void MatchTemplate_InvalidArguments_Throw()
    {
        var image = new Mat<byte>(10, 10);
        Assert.Throws<ArgumentException>(() => TemplateMatching.MatchTemplate(image, new Mat<byte>(4, 4, 3), TemplateMatchModes.CCorr));
        Assert.Throws<ArgumentException>(() => TemplateMatching.MatchTemplate(image, new Mat<byte>(12, 4), TemplateMatchModes.CCorr));
        Assert.Throws<ArgumentException>(() => TemplateMatching.MatchTemplate(image, new Mat<byte>(4, 4), TemplateMatchModes.CCorr, new Mat<byte>(3, 4)));
        Assert.Throws<NotSupportedException>(() => TemplateMatching.MatchTemplate(new Mat<short>(10, 10), new Mat<short>(4, 4), TemplateMatchModes.CCorr));
        Assert.Throws<ArgumentOutOfRangeException>(() => TemplateMatching.MatchTemplate(image, new Mat<byte>(4, 4), (TemplateMatchModes)9));
    }

    [Fact]
    public void MatchTemplate_LargeImage_DoesNotDependOnThreadCount()
    {
        Mat<byte> image = ToByte(Texture(700, 600, 4));
        Mat<byte> templ = image.Roi(200, 100, 24, 24).Clone();
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            float[] one = TemplateMatching.MatchTemplate(image, templ, TemplateMatchModes.CCoeffNormed).ToArray();
            CvParallel.MaxDegreeOfParallelism = 8;
            float[] eight = TemplateMatching.MatchTemplate(image, templ, TemplateMatchModes.CCoeffNormed).ToArray();
            Assert.Equal(one, eight);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void PhaseCorrelate_IntegerShift_IsRecovered()
    {
        Mat<double> a = Conversions.ConvertTo<float, double>(Texture(64, 48, 6));
        var b = new Mat<double>(64, 48);
        for (int y = 0; y < 48; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                b.At(y, x) = a.At(((y - 3) + 48) % 48, ((x - 5) + 64) % 64);
            }
        }

        Point2d shift = PhaseCorrelation.PhaseCorrelate(a, b, null, out double response);
        Assert.Equal(5, shift.X, 1);
        Assert.Equal(3, shift.Y, 1);
        Assert.True(response > 0.5);
    }

    [Fact]
    public void PhaseCorrelate_WithWindow_DoesNotModifySources()
    {
        Mat<float> a = Texture(32, 32, 1), b = Texture(32, 32, 2);
        float[] before = a.ToArray();
        _ = PhaseCorrelation.PhaseCorrelate(a, b, PhaseCorrelation.CreateHanningWindow<float>(new Size(32, 32)));
        Assert.Equal(before, a.ToArray());
    }

    [Fact]
    public void PhaseCorrelation_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentException>(() => PhaseCorrelation.PhaseCorrelate(new Mat<float>(8, 8), new Mat<float>(8, 9)));
        Assert.Throws<ArgumentOutOfRangeException>(() => PhaseCorrelation.CreateHanningWindow<float>(new Size(1, 5)));
        Assert.Throws<NotSupportedException>(() => PhaseCorrelation.PhaseCorrelate(new Mat<float>(1, 8), new Mat<float>(1, 8)));
    }

    [Fact]
    public void CreateHanningWindow_IsZeroAtEdgesAndOneNearCentre()
    {
        Mat<double> w = PhaseCorrelation.CreateHanningWindow<double>(new Size(9, 7));
        Assert.Equal(0, w.At(0, 4));
        Assert.Equal(0, w.At(3, 0));
        Assert.Equal(1, w.At(3, 4), 12);
    }

    [Fact]
    public void FindTransformEcc_Translation_IsRecovered()
    {
        Mat<float> templ = Texture(80, 60, 7);
        var shift = new Mat<double>(new[] { 1, 0, 2.0, 0, 1, -1.5 }, 3, 2);
        Mat<float> input = Geometry.WarpAffine(templ, shift, default, InterpolationFlags.Linear, WarpFlags.None, BorderType.Reflect);
        var warp = new Mat<float>(3, 2);
        LinearAlgebra.SetIdentity(warp);
        double rho = EccAlignment.FindTransformEcc(templ, input, warp, MotionTypes.Translation, new TermCriteria(TermCriteriaType.MaxIter | TermCriteriaType.Eps, 50, 1e-6));
        Assert.True(rho > 0.99);
        Assert.Equal(2.0, warp.At(0, 2), 1);
        Assert.Equal(-1.5, warp.At(1, 2), 1);
    }

    [Fact]
    public void ComputeEcc_IdenticalImages_IsOne()
    {
        Mat<float> a = Texture(30, 20, 3);
        Assert.Equal(1.0, EccAlignment.ComputeEcc(a, a.Clone()), 6);
    }

    [Fact]
    public void FindTransformEcc_InvalidArguments_Throw()
    {
        var a = new Mat<float>(10, 10);
        Assert.Throws<ArgumentException>(() => EccAlignment.FindTransformEcc(a, a, new Mat<float>(3, 3), MotionTypes.Affine));
        Assert.Throws<ArgumentException>(() => EccAlignment.FindTransformEcc(a, a, new Mat<float>(3, 2), MotionTypes.Homography));
        Assert.Throws<ArgumentOutOfRangeException>(() => EccAlignment.FindTransformEcc(a, a, new Mat<float>(3, 2), MotionTypes.Affine, null, null, 4));
        Assert.Throws<NotSupportedException>(() => EccAlignment.ComputeEcc(new Mat<short>(4, 4), new Mat<short>(4, 4)));
    }

    [Fact]
    public void FindTransformEcc_DoesNotDependOnThreadCount()
    {
        Mat<float> templ = Texture(300, 200, 8);
        var rot = new Mat<double>(new[] { 0.999, -0.02, 1.0, 0.02, 0.999, -0.5 }, 3, 2);
        Mat<float> input = Geometry.WarpAffine(templ, rot, default, InterpolationFlags.Linear, WarpFlags.None, BorderType.Reflect);
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            var w1 = new Mat<float>(3, 2);
            LinearAlgebra.SetIdentity(w1);
            double r1 = EccAlignment.FindTransformEcc(templ, input, w1, MotionTypes.Euclidean);
            CvParallel.MaxDegreeOfParallelism = 8;
            var w8 = new Mat<float>(3, 2);
            LinearAlgebra.SetIdentity(w8);
            double r8 = EccAlignment.FindTransformEcc(templ, input, w8, MotionTypes.Euclidean);
            Assert.Equal(r1, r8);
            Assert.Equal(w1.ToArray(), w8.ToArray());
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }
}
