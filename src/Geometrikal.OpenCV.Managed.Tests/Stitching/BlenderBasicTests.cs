namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;
using Geometrikal.OpenCV.Managed.Tests.Core;

[Collection(nameof(CvParallelCollection))]
public sealed class BlenderBasicTests
{
    [Fact]
    public void CreateLaplacePyr_Short_LevelsRestoreSource()
    {
        Mat<short> src = Synthetic16(40, 32, 3);
        Mat<short> copy = src.Clone();
        Mat<short>[] pyr = Blender.CreateLaplacePyr(src, 3);
        Assert.Equal(StitchGolden.Bytes(copy), StitchGolden.Bytes(src));
        Assert.Equal(4, pyr.Length);
        Assert.Equal(new[] { 40, 20, 10, 5 }, pyr.Select(p => p.Width));
        Blender.RestoreImageFromLaplacePyr(pyr);
        Assert.Equal(StitchGolden.Bytes(src), StitchGolden.Bytes(pyr[0]));
    }

    [Fact]
    public void Blend_WithoutPrepare_Throws()
    {
        Assert.Throws<InvalidOperationException>(() => new FeatherBlender().Blend(out _, out _));
        Assert.Throws<InvalidOperationException>(() => new MultiBandBlender().Blend(out _, out _));
    }

    [Fact]
    public void Feed_OutsidePanorama_Throws()
    {
        var b = new FeatherBlender();
        b.Prepare(new Rect(0, 0, 10, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => b.Feed(new Mat<short>(5, 5, 3), new Mat<byte>(5, 5), new Point(8, 0)));
    }

    [Fact]
    public void NoBlender_LaterImageOverwritesMaskedPixels()
    {
        var b = new Blender();
        b.Prepare(new[] { new Point(0, 0), new Point(2, 0) }, new[] { new Size(4, 2), new Size(4, 2) });
        var a = new Mat<short>(4, 2, 3);
        a.SetTo(Scalar.All(10));
        var c = new Mat<short>(4, 2, 3);
        c.SetTo(Scalar.All(20));
        var m = new Mat<byte>(4, 2);
        m.SetTo(Scalar.All(255));
        b.Feed(a, m, new Point(0, 0));
        b.Feed(c, m, new Point(2, 0));
        b.Blend(out Mat<short> dst, out Mat<byte> mask);
        Assert.Equal(6, dst.Width);
        Assert.Equal(new short[] { 10, 10, 20, 20, 20, 20 }, Enumerable.Range(0, 6).Select(x => dst.At(0, x, 0)));
        Assert.All(Enumerable.Range(0, 6), x => Assert.Equal(255, mask.At(1, x)));
    }

    [Fact]
    public void MultiBandBlender_Prepare_CropsBandsToPanoramaSize()
    {
        var b = new MultiBandBlender(10);
        b.Prepare(new Rect(0, 0, 16, 9));
        var img = new Mat<byte>(16, 9, 3);
        img.SetTo(Scalar.All(100));
        var m = new Mat<byte>(16, 9);
        m.SetTo(Scalar.All(255));
        b.Feed(img, m, new Point(0, 0));
        b.Blend(out Mat<short> dst, out _);
        Assert.Equal(16, dst.Width);
        Assert.Equal(9, dst.Height);
        Assert.All(StitchGolden.Bytes(dst).Chunk(2).Select(p => BitConverter.ToInt16(p)), v => Assert.InRange(v, 95, 100));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Blenders_OutputDoesNotDependOnThreadCount(bool multiBand)
    {
        byte[] Run(int threads)
        {
            int saved = CvParallel.MaxDegreeOfParallelism;
            CvParallel.MaxDegreeOfParallelism = threads;
            try
            {
                Blender b = multiBand ? new MultiBandBlender(5) : new FeatherBlender(0.05f);
                var corners = new[] { new Point(0, 0), new Point(300, 20), new Point(10, 250) };
                b.Prepare(corners, Enumerable.Repeat(new Size(400, 300), 3).ToArray());
                for (int i = 0; i < 3; i++)
                {
                    Mat<short> img = Synthetic16(400, 300, i + 1);
                    var m = new Mat<byte>(400, 300);
                    m.SetTo(Scalar.All(255));
                    b.Feed(img, m, corners[i]);
                }

                b.Blend(out Mat<short> dst, out _);
                return StitchGolden.Bytes(dst);
            }
            finally
            {
                CvParallel.MaxDegreeOfParallelism = saved;
            }
        }

        Assert.Equal(Run(1), Run(8));
    }

    internal static Mat<short> Synthetic16(int w, int h, int seed)
    {
        var m = new Mat<short>(w, h, 3);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w * 3; x++)
            {
                m.GetRowSpan(y)[x] = (short)(((x * 37) + (y * 91) + (seed * 13)) % 256);
            }
        }

        return m;
    }
}
