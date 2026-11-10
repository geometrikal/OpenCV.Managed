namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

public sealed class MathFunctionsBasicTests
{
    private static Mat<float> Random(int width, int height, int seed, float lo = -50, float hi = 50)
    {
        var random = new Random(seed);
        var m = new Mat<float>(width, height);
        var span = m.GetSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = lo + (float)random.NextDouble() * (hi - lo);
        }

        return m;
    }

    [Fact]
    public void Magnitude_SinglePixel_IsHypotenuse()
    {
        var mag = MathFunctions.Magnitude(new Mat<float>([3f], 1, 1), new Mat<float>([4f], 1, 1));
        Assert.Equal(5f, mag.At(0, 0));
    }

    [Fact]
    public void Phase_Quadrants_AreInRange()
    {
        var x = new Mat<float>([1f, 0f, -1f, 0f, 0f], 5, 1);
        var y = new Mat<float>([0f, 1f, 0f, -1f, 0f], 5, 1);

        var degrees = MathFunctions.Phase(x, y, angleInDegrees: true);

        Assert.Equal(0f, degrees.At(0, 0));
        Assert.Equal(90f, degrees.At(0, 1), 0.05f);
        Assert.Equal(180f, degrees.At(0, 2), 0.05f);
        Assert.Equal(270f, degrees.At(0, 3), 0.05f);
        Assert.Equal(0f, degrees.At(0, 4));
    }

    [Fact]
    public void CartToPolar_ThenPolarToCart_RoundTrips()
    {
        var x = Random(37, 5, 1);
        var y = Random(37, 5, 2);

        var (mag, angle) = MathFunctions.CartToPolar(x, y);
        var (x2, y2) = MathFunctions.PolarToCart(mag, angle);

        for (int i = 0; i < 37 * 5; i++)
        {
            // fastAtan2 is accurate to about 0.3 degrees in the worst case.
            Assert.Equal(x.GetSpan()[i], x2.GetSpan()[i], 0.01f * Math.Max(1, mag.GetSpan()[i]));
            Assert.Equal(y.GetSpan()[i], y2.GetSpan()[i], 0.01f * Math.Max(1, mag.GetSpan()[i]));
        }
    }

    [Fact]
    public void PolarToCart_VectorAndScalarPaths_AreAccurate()
    {
        // 7 elements use the table code only; 9 elements use the vector code.
        foreach (int n in new[] { 7, 9 })
        {
            var angle = Random(n, 1, 3, -720, 720);
            var (x, y) = MathFunctions.PolarToCart(null, angle, angleInDegrees: true);
            for (int i = 0; i < n; i++)
            {
                double radians = angle.At(0, i) * Math.PI / 180;
                Assert.Equal(Math.Cos(radians), x.At(0, i), 2e-6);
                Assert.Equal(Math.Sin(radians), y.At(0, i), 2e-6);
            }
        }
    }

    [Fact]
    public void SinCosCephes_SpecialValues()
    {
        MathKernels.SinCosCephes(float.PositiveInfinity, out float s, out float c);
        Assert.True(float.IsNaN(s) && float.IsNaN(c));
        MathKernels.SinCosCephes(0f, out s, out c);
        Assert.Equal(0f, s);
        Assert.Equal(1f, c);
    }

    [Fact]
    public void Pow_IntegerElements_SaturateAndUseTheNegativeTable()
    {
        var bytes = new Mat<byte>([0, 1, 2, 3, 200], 5, 1);
        Assert.Equal(new byte[] { 0, 1, 8, 27, 255 }, MathFunctions.Pow(bytes, 3).ToArray());
        Assert.Equal(new byte[] { 0, 1, 255, 255, 255 }, MathFunctions.Pow(bytes, 10).ToArray());
        var shorts = new Mat<short>([-2, -1, 0, 1, 2, 7], 6, 1);

        // OpenCV table for power -1 on [-2, -1, 0, 1, 2]: {-1, -1, max, 1, 1}; |x| > 2 gives 0.
        Assert.Equal(new short[] { -1, -1, short.MaxValue, 1, 1, 0 }, MathFunctions.Pow(shorts, -1).ToArray());
    }

    [Fact]
    public void Pow_RealPower_ZeroAndNegativeElements()
    {
        var m = new Mat<float>([0f, -1f, 4f], 3, 1);

        var p = MathFunctions.Pow(m, 1.5);
        var n = MathFunctions.Pow(m, -1.5);

        Assert.Equal(0f, p.At(0, 0));
        Assert.Equal(0x7fffffff, BitConverter.SingleToInt32Bits(p.At(0, 1)));
        Assert.Equal(8f, p.At(0, 2), 1e-5f);
        Assert.Equal(float.PositiveInfinity, n.At(0, 0));
    }

    [Fact]
    public void Pow_InverseSqrt_VectorBlocksAreAccurate()
    {
        var m = Random(19, 1, 4, 0.01f, 100f);

        var r = MathFunctions.Pow(m, -0.5);

        for (int i = 0; i < 19; i++)
        {
            Assert.Equal(1 / Math.Sqrt(m.At(0, i)), r.At(0, i), 1e-6 / Math.Sqrt(m.At(0, i)) * 10);
        }
    }

    [Fact]
    public void Functions_RoiWithStride_MatchPackedImage()
    {
        var parentX = Random(40, 12, 5);
        var parentY = Random(40, 12, 6);
        var roiX = parentX.Roi(3, 2, 21, 7);
        var roiY = parentY.Roi(3, 2, 21, 7);
        var packedX = roiX.Clone();
        var packedY = roiY.Clone();

        Assert.Equal(MathFunctions.Magnitude(packedX, packedY).ToArray(), MathFunctions.Magnitude(roiX, roiY).ToArray());
        Assert.Equal(MathFunctions.Phase(packedX, packedY).ToArray(), MathFunctions.Phase(roiX, roiY).ToArray());
        Assert.Equal(MathFunctions.Exp(packedX).ToArray(), MathFunctions.Exp(roiX).ToArray());
        Assert.Equal(MathFunctions.Pow(packedX, 3).ToArray(), MathFunctions.Pow(roiX, 3).ToArray());
    }

    [Fact]
    public void Functions_DoNotModifySources()
    {
        var x = Random(9, 3, 7);
        var y = Random(9, 3, 8);
        float[] before = x.ToArray();

        MathFunctions.CartToPolar(x, y);
        MathFunctions.PolarToCart(x, y);
        MathFunctions.Pow(x, 2.5);

        Assert.Equal(before, x.ToArray());
    }

    [Fact]
    public void Functions_InvalidInputs_Throw()
    {
        var f = new Mat<float>(4, 4);
        Assert.Throws<NotSupportedException>(() => MathFunctions.Exp(new Mat<byte>(4, 4)));
        Assert.Throws<ArgumentException>(() => MathFunctions.Magnitude(f, new Mat<float>(5, 4)));
        Assert.Throws<NotSupportedException>(() => MathFunctions.Pow(new Mat<byte>(4, 4), 1.5));
        Assert.Throws<ArgumentNullException>(() => MathFunctions.Log<float>(null!));
    }
}
