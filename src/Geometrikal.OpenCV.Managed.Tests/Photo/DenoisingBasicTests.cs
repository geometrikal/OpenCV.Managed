namespace Geometrikal.OpenCV.Managed.Tests.Photo;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Photo;
using Geometrikal.OpenCV.Managed.Tests.Core;

[Collection(nameof(CvParallelCollection))]
public sealed class DenoisingBasicTests
{
    private static Mat<byte> Noisy(int w, int h, int cn, int seed)
    {
        var m = new Mat<byte>(w, h, cn);
        var rng = new Random(seed);
        for (int y = 0; y < h; y++)
        {
            var row = m.GetRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                row[x] = (byte)Math.Clamp(((x / cn) + y) * 2 + rng.Next(-20, 21), 0, 255);
            }
        }

        return m;
    }

    [Fact]
    public void FastNlMeans_ReducesNoise()
    {
        var src = Noisy(60, 50, 1, 1);
        var dst = Denoising.FastNlMeansDenoising(src, 15f);

        double Rough(Mat<byte> m)
        {
            double s = 0;
            for (int y = 0; y < m.Height; y++)
            {
                var r = m.GetReadOnlyRowSpan(y);
                for (int x = 1; x < r.Length; x++)
                {
                    s += Math.Abs(r[x] - r[x - 1]);
                }
            }

            return s;
        }

        Assert.True(Rough(dst) < Rough(src) * 0.6);
    }

    [Fact]
    public void FastNlMeans_Output_DoesNotDependOnThreadCount()
    {
        var src = Noisy(300, 260, 3, 2);
        int saved = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            var one = Denoising.FastNlMeansDenoising(src, 10f, 5, 11).ToArray();
            CvParallel.MaxDegreeOfParallelism = 8;
            var eight = Denoising.FastNlMeansDenoising(src, 10f, 5, 11).ToArray();
            Assert.Equal(one, eight);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = saved;
        }
    }

    [Fact]
    public void FastNlMeans_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<byte>(70, 60, 2);
        var roi = parent.Roi(5, 7, 41, 33);
        Noisy(41, 33, 2, 3).CopyTo(roi);
        var packed = roi.Clone();

        Assert.Equal(Denoising.FastNlMeansDenoising(packed, 8f, 5, 9).ToArray(), Denoising.FastNlMeansDenoising(roi, 8f, 5, 9).ToArray());
    }

    [Fact]
    public void FastNlMeans_InvalidInputs_Throw()
    {
        Assert.Throws<NotSupportedException>(() => Denoising.FastNlMeansDenoising(new Mat<ushort>(8, 8), 3f));
        Assert.Throws<NotSupportedException>(() => Denoising.FastNlMeansDenoising(new Mat<float>(8, 8), 3f, normType: NormType.L1));
        Assert.Throws<NotSupportedException>(() => Denoising.FastNlMeansDenoising(new Mat<byte>(8, 8), 3f, normType: NormType.Inf));
        Assert.Throws<ArgumentException>(() => Denoising.FastNlMeansDenoising(new Mat<byte>(8, 8, 3), [1f, 2f]));
        Assert.Throws<NotSupportedException>(() => Denoising.FastNlMeansDenoisingColored(new Mat<byte>(8, 8), 3f));
    }
}
