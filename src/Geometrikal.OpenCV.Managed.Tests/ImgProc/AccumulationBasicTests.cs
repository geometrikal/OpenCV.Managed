namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class AccumulationBasicTests
{
    [Fact]
    public void Accumulate_SumsFrames()
    {
        var acc = new Mat<float>(3, 2);
        var frame = new Mat<byte>([1, 2, 3, 4, 5, 6], 3, 2);

        Accumulation.Accumulate(frame, acc);
        Accumulation.Accumulate(frame, acc);
        Accumulation.AccumulateSquare(frame, acc);

        Assert.Equal(new float[] { 3, 8, 15, 24, 35, 48 }, acc.ToArray());
    }

    [Fact]
    public void AccumulateWeighted_MaskedPixels_AreUnchanged()
    {
        var acc = new Mat<double>([10, 10, 10, 10], 4, 1);
        var frame = new Mat<ushort>([20, 20, 20, 20], 4, 1);
        var mask = new Mat<byte>([1, 0, 255, 0], 4, 1);

        Accumulation.AccumulateWeighted(frame, acc, 0.25, mask);

        Assert.Equal(new double[] { 12.5, 10, 12.5, 10 }, acc.ToArray());
    }

    [Fact]
    public void AccumulateProduct_RoiWithStride_MatchesRowByRowReference()
    {
        var parent = new Mat<float>(40, 9, 3);
        var roi = parent.Roi(3, 2, 21, 5);
        var a = new Mat<byte>(21, 5, 3);
        var b = new Mat<byte>(21, 5, 3);
        var rng = new Random(5);
        rng.NextBytes(a.GetSpan());
        rng.NextBytes(b.GetSpan());

        Accumulation.AccumulateProduct(a, b, roi);

        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 63; x++)
            {
                Assert.Equal((float)a.GetReadOnlyRowSpan(y)[x] * b.GetReadOnlyRowSpan(y)[x], roi.GetReadOnlyRowSpan(y)[x]);
            }
        }

        Assert.All(parent.GetReadOnlyRowSpan(1).ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Accumulate_InvalidInputs_Throw()
    {
        Assert.Throws<NotSupportedException>(() => Accumulation.Accumulate(new Mat<double>(2, 2), new Mat<float>(2, 2)));
        Assert.Throws<NotSupportedException>(() => Accumulation.Accumulate(new Mat<short>(2, 2), new Mat<float>(2, 2)));
        Assert.Throws<ArgumentException>(() => Accumulation.Accumulate(new Mat<byte>(2, 2), new Mat<float>(3, 2)));
        Assert.Throws<ArgumentException>(() => Accumulation.Accumulate(new Mat<byte>(2, 2), new Mat<float>(2, 2), new Mat<byte>(2, 2, 2)));
        Assert.Throws<ArgumentException>(() => Accumulation.AccumulateProduct(new Mat<byte>(2, 2), new Mat<byte>(2, 3), new Mat<float>(2, 2)));
    }
}
