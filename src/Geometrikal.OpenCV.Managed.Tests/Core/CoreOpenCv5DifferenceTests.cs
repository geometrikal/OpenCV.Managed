namespace Geometrikal.OpenCV.Managed.Tests.Core;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Pins OpenCV 5.0 behaviour where cv2 4.5.5 (the golden reference) differs. Expected values
/// follow the OpenCV 5.0 kernels in <c>modules/core/src/arithm.simd.hpp</c> and
/// <c>norm.simd.hpp</c>.
/// </summary>
public sealed class CoreOpenCv5DifferenceTests
{
    /// <summary>
    /// Golden cases <c>multiply_f32_*_s0.1</c> and <c>*_s0.333…</c>: 5.0 computes
    /// <c>(a * (float)scale) * b</c> in float; 4.5.5 computes in double.
    /// </summary>
    [Fact]
    public void Multiply_Float32WithScale_ComputesInFloat()
    {
        float[] a = [1.1f, -3.7f, 1234.567f, 0.3f];
        float[] b = [7.9f, 2.2f, -0.001f, 3f];
        Mat<float> r = Arithmetic.Multiply(new Mat<float>(a, 4, 1), new Mat<float>(b, 4, 1), 0.1);
        float fs = (float)0.1;
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(a[i] * fs * b[i], r.At(0, i));
        }
    }

    [Fact]
    public void Multiply_Int32Overflow_GivesIntMin()
    {
        Mat<int> r = Arithmetic.Multiply(new Mat<int>([int.MaxValue], 1, 1), new Mat<int>([2], 1, 1));
        Assert.Equal(int.MinValue, r.At(0, 0));
    }

    [Fact]
    public void Multiply_UInt16OverflowOfInt32_GivesZero()
    {
        Mat<ushort> r = Arithmetic.Multiply(new Mat<ushort>([65535], 1, 1), new Mat<ushort>([65535], 1, 1));
        Assert.Equal(0, r.At(0, 0));
    }

    [Fact]
    public void Divide_Int32_ComputesInDouble()
    {
        Mat<int> r = Arithmetic.Divide(new Mat<int>([16777217, int.MaxValue], 2, 1), new Mat<int>([1, 1], 2, 1));
        Assert.Equal(16777217, r.At(0, 0));
        Assert.Equal(int.MaxValue, r.At(0, 1));

        Mat<int> recip = Arithmetic.Divide(1e9, new Mat<int>([3, 7, 0, -3], 4, 1));
        Assert.Equal(new[] { 333333333, 142857143, 0, -333333333 }, recip.ToArray());
    }

    [Fact]
    public void AbsDiff_Int32BeyondIntMax_IsAbsOfWrappedDifference()
    {
        Mat<int> r = Arithmetic.AbsDiff(new Mat<int>([int.MinValue, int.MaxValue], 2, 1), new Mat<int>([1, -1], 2, 1));
        Assert.Equal(new[] { int.MaxValue, int.MinValue }, r.ToArray());
    }

    [Fact]
    public void NormInf_Int32MinValue_IsTwoToThe31()
    {
        Assert.Equal(2147483648.0, Reductions.Norm(new Mat<int>([int.MinValue, 5], 2, 1), NormType.Inf));
    }

    [Fact]
    public void Reduce_UInt16ToFloat_SumsExactlyBeforeConversion()
    {
        var src = new Mat<ushort>(1, 300);
        src.Fill(65535);
        Mat<float> r = Reductions.Reduce<ushort, float>(src, 0, ReduceType.Sum);
        Assert.Equal((float)(300L * 65535), r.At(0, 0));
    }

    [Fact]
    public void Reduce_Sum2_IsNewInOpenCv5()
    {
        Mat<double> r = Reductions.Reduce<byte, double>(new Mat<byte>([1, 2, 3, 4, 5, 6], 3, 2), 0, ReduceType.Sum2);
        Assert.Equal(new[] { 17.0, 29.0, 45.0 }, r.ToArray());
    }
}
