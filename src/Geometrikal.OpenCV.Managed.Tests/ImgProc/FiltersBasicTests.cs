namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Runtime.InteropServices;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Analytic and structural tests for the filtering port: ROI/stride
/// correctness, argument validation, non-mutation, determinism, and pinned
/// behaviour for corners the cv2 golden battery deliberately avoids (see
/// Docs/Notes/FilteringPortNotes.md).
/// </summary>
public sealed class FiltersBasicTests
{
    // ------------------------------------------------------------ helpers

    private static Mat<byte> PatternU8(int width, int height, int channels)
    {
        var m = new Mat<byte>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = (byte)(((i * 31) + (y * 17) + 7) % 256);
            }
        }

        return m;
    }

    private static Mat<float> PatternF32(int width, int height, int channels)
    {
        var m = new Mat<float>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<float> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = (((i * 37) + (y * 11)) % 199) - 99.5f;
            }
        }

        return m;
    }

    private static (Mat<T> Roi, Mat<T> Packed) RoiPair<T>(Mat<T> parent)
        where T : unmanaged
    {
        Mat<T> roi = parent.Roi(3, 4, 10, 8);
        Assert.False(roi.IsContinuous);
        return (roi, roi.Clone());
    }

    private static void AssertMatsEqual<T>(Mat<T> expected, Mat<T> actual)
        where T : unmanaged
    {
        Assert.Equal(expected.Width, actual.Width);
        Assert.Equal(expected.Height, actual.Height);
        Assert.Equal(expected.Channels, actual.Channels);
        Assert.True(
            MemoryMarshal.AsBytes(expected.ToArray().AsSpan())
                .SequenceEqual(MemoryMarshal.AsBytes(actual.ToArray().AsSpan())),
            "matrices differ");
    }

    // --------------------------------------------------------- ROI/stride

    [Fact]
    public void GaussianBlur_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 1));
        AssertMatsEqual(
            Filters.GaussianBlur(packed, new Size(5, 5), 0),
            Filters.GaussianBlur(roi, new Size(5, 5), 0));
    }

    [Fact]
    public void GaussianBlurFloat_RoiWithStride_MatchesPackedImage()
    {
        (Mat<float> roi, Mat<float> packed) = RoiPair(PatternF32(20, 20, 1));
        AssertMatsEqual(
            Filters.GaussianBlur(packed, new Size(3, 3), 0.9),
            Filters.GaussianBlur(roi, new Size(3, 3), 0.9));
    }

    [Theory]
    [InlineData(0, 0, 1.6)]
    [InlineData(5, 3, 0.9)]
    [InlineData(1, 1, 0)]
    public void GaussianBlurFloatInto_MatchesGaussianBlur(int kw, int kh, double sigma)
    {
        Mat<float> src = PatternF32(23, 17, 3);
        var dst = new Mat<float>(src.Width, src.Height, src.Channels);
        Filters.GaussianBlurInto(src, dst, new Size(kw, kh), sigma);
        AssertMatsEqual(Filters.GaussianBlur(src, new Size(kw, kh), sigma), dst);
    }

    [Fact]
    public void Blur_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 2));
        AssertMatsEqual(
            Filters.Blur(packed, new Size(3, 5)),
            Filters.Blur(roi, new Size(3, 5)));
    }

    [Fact]
    public void MedianBlur_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 3));
        AssertMatsEqual(Filters.MedianBlur(packed, 5), Filters.MedianBlur(roi, 5));
        AssertMatsEqual(Filters.MedianBlur(packed, 7), Filters.MedianBlur(roi, 7));
    }

    /// <summary>
    /// The histogram path splits wide images into column stripes. The result must equal a
    /// brute-force median with replicated borders across stripe boundaries.
    /// </summary>
    [Theory]
    [InlineData(7, 1)]
    [InlineData(7, 3)]
    [InlineData(15, 1)]
    [InlineData(15, 4)]
    public void MedianBlur_WideImage_MatchesBruteForce(int ksize, int channels)
    {
        const int width = 600;
        const int height = 23;
        var src = new Mat<byte>(width, height, channels);
        var random = new Random(ksize * 10 + channels);
        for (int y = 0; y < height; y++)
        {
            random.NextBytes(src.GetRowSpan(y));
        }

        Mat<byte> actual = Filters.MedianBlur(src, ksize);

        int r = ksize / 2;
        var window = new byte[ksize * ksize];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int n = 0;
                    for (int dy = -r; dy <= r; dy++)
                    {
                        for (int dx = -r; dx <= r; dx++)
                        {
                            int sy = Math.Clamp(y + dy, 0, height - 1);
                            int sx = Math.Clamp(x + dx, 0, width - 1);
                            window[n++] = src.At(sy, sx, c);
                        }
                    }

                    Array.Sort(window);
                    Assert.Equal(window[window.Length / 2], actual.At(y, x, c));
                }
            }
        }
    }

    [Fact]
    public void Filter2D_RoiWithStride_MatchesPackedImage()
    {
        var kernel = new Mat<float>(3, 3, 1);
        for (int i = 0; i < 3; i++)
        {
            Span<float> row = kernel.GetRowSpan(i);
            for (int j = 0; j < 3; j++)
            {
                row[j] = 0.1f * ((i * 3) + j - 4);
            }
        }

        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 1));
        AssertMatsEqual(
            Filters.Filter2D<byte, byte>(packed, kernel, delta: 40),
            Filters.Filter2D<byte, byte>(roi, kernel, delta: 40));
    }

    [Fact]
    public void SepFilter2D_RoiWithStride_MatchesPackedImage()
    {
        float[] k = [0.25f, 0.5f, 0.25f];
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 1));
        AssertMatsEqual(
            Filters.SepFilter2D<byte, byte>(packed, k, k),
            Filters.SepFilter2D<byte, byte>(roi, k, k));
    }

    [Fact]
    public void Sobel_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 1));
        AssertMatsEqual(
            Derivatives.Sobel<byte, short>(packed, 1, 0),
            Derivatives.Sobel<byte, short>(roi, 1, 0));
    }

    [Fact]
    public void Laplacian_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(20, 20, 1));
        AssertMatsEqual(
            Derivatives.Laplacian<byte, short>(packed, 5),
            Derivatives.Laplacian<byte, short>(roi, 5));
    }

    // ------------------------------------------------- source is untouched

    [Fact]
    public void Filters_DoNotModifySource()
    {
        Mat<byte> src = PatternU8(16, 12, 1);
        byte[] before = src.ToArray();

        _ = Filters.GaussianBlur(src, new Size(5, 5), 0);
        _ = Filters.Blur(src, new Size(3, 3));
        _ = Filters.MedianBlur(src, 5);
        _ = Filters.MedianBlur(src, 7);
        _ = Filters.SepFilter2D<byte, short>(src, [1f, 2f, 1f], [-1f, 0f, 1f]);
        _ = Derivatives.Sobel<byte, short>(src, 1, 1);
        _ = Derivatives.Laplacian<byte, float>(src, 5);

        Assert.Equal(before, src.ToArray());
    }

    // ----------------------------------------------------------- determinism

    [Fact]
    public void Filters_RepeatedCalls_AreDeterministic()
    {
        Mat<float> src = PatternF32(15, 11, 3);
        AssertMatsEqual(
            Filters.GaussianBlur(src, new Size(5, 5), 1.3),
            Filters.GaussianBlur(src, new Size(5, 5), 1.3));
        AssertMatsEqual(
            Derivatives.Sobel<float, float>(src, 1, 0),
            Derivatives.Sobel<float, float>(src, 1, 0));
    }

    // ------------------------------------------------------- zero / uniform

    [Fact]
    public void Filters_AllZeroImage_ProducesZeros()
    {
        var zero = new Mat<byte>(9, 7, 1);
        Assert.All(Filters.GaussianBlur(zero, new Size(5, 5), 0).ToArray(), v => Assert.Equal(0, v));
        Assert.All(Filters.Blur(zero, new Size(3, 3)).ToArray(), v => Assert.Equal(0, v));
        Assert.All(Filters.MedianBlur(zero, 3).ToArray(), v => Assert.Equal(0, v));
        Assert.All(Derivatives.Sobel<byte, short>(zero, 1, 0).ToArray(), v => Assert.Equal(0, v));
        Assert.All(Derivatives.Laplacian<byte, short>(zero, 5).ToArray(), v => Assert.Equal(0, v));
    }

    [Fact]
    public void SmoothingFilters_UniformImage_IsInvariant()
    {
        var uniform = new Mat<byte>(10, 10, 1);
        uniform.Fill(200);
        Assert.All(Filters.GaussianBlur(uniform, new Size(7, 7), 0).ToArray(), v => Assert.Equal(200, v));
        Assert.All(Filters.Blur(uniform, new Size(5, 5)).ToArray(), v => Assert.Equal(200, v));
        Assert.All(Filters.MedianBlur(uniform, 5).ToArray(), v => Assert.Equal(200, v));
    }

    [Fact]
    public void GaussianBlur_SinglePixel_ConstantBorder_Blends()
    {
        // A 1x1 image with a constant border must blend toward zero (the
        // kernel-collapse quirk applies only to non-constant borders).
        var one = new Mat<byte>(1, 1, 1);
        one.Fill(240);
        Mat<byte> outConst = Filters.GaussianBlur(one, new Size(3, 3), 0, 0, BorderType.Constant);
        Assert.True(outConst.At(0, 0) < 240);

        // With a replicating border the kernel collapses to 1x1 → identity.
        Mat<byte> outRepl = Filters.GaussianBlur(one, new Size(3, 3), 0, 0, BorderType.Replicate);
        Assert.Equal(240, outRepl.At(0, 0));
    }

    /// <summary>
    /// U8→U8 blur with kernel area ≤ 256 divides with OpenCV's fixed-point rule, which
    /// rounds exact halves up. Expected values are from cv2 with IPP off and
    /// <c>BORDER_CONSTANT | BORDER_ISOLATED</c>.
    /// </summary>
    [Fact]
    public void Blur_EvenKernel_ConstantBorder_RoundsHalfUp()
    {
        var src = new Mat<byte>(3, 3, 1);
        byte[] values = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                src.At(y, x) = values[(y * 3) + x];
            }
        }

        Mat<byte> result = Filters.Blur(src, new Size(2, 2), null, BorderType.Constant);

        byte[] expected = [3, 8, 13, 13, 30, 40, 28, 60, 70];
        Assert.Equal(expected, result.ToArray());
    }

    /// <summary>
    /// The fixed-point rule maps a zero sum to 1 for a 2×1 kernel (divDelta = 2,
    /// divScale = 2^22), as cv2 does.
    /// </summary>
    [Fact]
    public void Blur_TwoByOneKernel_ZeroImage_MatchesOpenCvFixedPoint()
    {
        var zero = new Mat<byte>(64, 8, 1);
        Assert.All(Filters.Blur(zero, new Size(2, 1)).ToArray(), v => Assert.Equal(1, v));
    }

    // --------------------------------------------------------- kernel values

    [Fact]
    public void GetGaussianKernel_SizeOne_IsIdentity()
    {
        Assert.Equal([1.0], Filters.GetGaussianKernel(1, 0));
    }

    [Fact]
    public void GetDerivKernels_SobelAndScharr_ProduceExpectedTaps()
    {
        (float[] kx, float[] ky) = Derivatives.GetDerivKernels(1, 0, 3);
        Assert.Equal([-1f, 0f, 1f], kx);
        Assert.Equal([1f, 2f, 1f], ky);

        (kx, ky) = Derivatives.GetDerivKernels(2, 0, 5);
        Assert.Equal([1f, 0f, -2f, 0f, 1f], kx);
        Assert.Equal([1f, 4f, 6f, 4f, 1f], ky);

        (kx, ky) = Derivatives.GetDerivKernels(0, 1, ksize: -1);
        Assert.Equal([3f, 10f, 3f], kx);
        Assert.Equal([-1f, 0f, 1f], ky);

        (kx, _) = Derivatives.GetDerivKernels(0, 1, 3, normalize: true);
        Assert.Equal([0.25f, 0.5f, 0.25f], kx);

        // ksize 1 with a first derivative expands that axis to 3 taps.
        (kx, ky) = Derivatives.GetDerivKernels(1, 0, 1);
        Assert.Equal([-1f, 0f, 1f], kx);
        Assert.Equal([1f], ky);
    }

    // ------------------------------------------------------------ validation

    [Fact]
    public void Filters_WrapBorder_Throws()
    {
        Mat<byte> src = PatternU8(8, 8, 1);
        Assert.Throws<NotSupportedException>(() => Filters.GaussianBlur(src, new Size(3, 3), 0, 0, BorderType.Wrap));
        Assert.Throws<NotSupportedException>(() => Filters.Blur(src, new Size(3, 3), null, BorderType.Wrap));
        Assert.Throws<NotSupportedException>(
            () => Filters.SepFilter2D<byte, byte>(src, [1f], [1f], borderType: BorderType.Wrap));
        Assert.Throws<NotSupportedException>(
            () => Derivatives.Sobel<byte, short>(src, 1, 0, borderType: BorderType.Wrap));
        Assert.Throws<NotSupportedException>(
            () => Derivatives.Laplacian<byte, byte>(src, 3, borderType: BorderType.Wrap));
    }

    [Fact]
    public void Filters_InvalidArguments_Throw()
    {
        Mat<byte> src = PatternU8(8, 8, 1);
        Mat<float> srcF = PatternF32(8, 8, 1);

        Assert.Throws<ArgumentNullException>(() => Filters.GaussianBlur((Mat<byte>)null!, new Size(3, 3), 0));
        Assert.Throws<ArgumentException>(() => Filters.GaussianBlur(src, new Size(4, 4), 0));
        Assert.Throws<ArgumentException>(() => Filters.GaussianBlur(src, new Size(0, 0), 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.Blur(src, new Size(0, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.Blur(src, new Size(3, 3), new Point(3, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.MedianBlur(src, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.MedianBlur(src, -3));
        Assert.Throws<NotSupportedException>(() => Filters.MedianBlur(srcF, 7));
        Assert.Throws<NotSupportedException>(() => Filters.MedianBlur(PatternU8(8, 8, 2), 7));
        Assert.Throws<ArgumentException>(() => Filters.SepFilter2D<byte, byte>(src, [], [1f]));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Filters.SepFilter2D<byte, byte>(src, [1f, 1f], [1f], new Point(2, 0)));
        Assert.Throws<ArgumentException>(
            () => Filters.Filter2D<byte, byte>(src, new Mat<float>(2, 2, 2)));
        Assert.Throws<ArgumentException>(() => Derivatives.Sobel<byte, short>(src, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Derivatives.Sobel<byte, short>(src, 1, 0, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Derivatives.Sobel<byte, short>(src, 3, 0, 3));
        Assert.Throws<ArgumentException>(() => Derivatives.Scharr<byte, short>(src, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Derivatives.Laplacian<byte, byte>(src, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Derivatives.Laplacian<byte, byte>(src, -1));
    }

    [Fact]
    public void Filters_UnsupportedTypeCombinations_Throw()
    {
        Mat<float> srcF = PatternF32(8, 8, 1);
        var srcD = new Mat<double>(8, 8, 1);
        Assert.Throws<NotSupportedException>(() => Filters.SepFilter2D<int, int>(new Mat<int>(8, 8, 1), [1f], [1f]));
        Assert.Throws<NotSupportedException>(() => Filters.SepFilter2D<double, float>(srcD, [1f], [1f]));
        Assert.Throws<NotSupportedException>(
            () => Filters.Filter2D<float, byte>(srcF, new Mat<float>(3, 3, 1)));
        Assert.Throws<NotSupportedException>(
            () => Filters.Filter2D<float, double>(srcF, new Mat<float>(3, 3, 1)));
        Assert.Throws<NotSupportedException>(
            () => Filters.Filter2D<ushort, short>(new Mat<ushort>(8, 8, 1), new Mat<float>(3, 3, 1)));
        Assert.Throws<NotSupportedException>(
            () => Filters.BoxFilter<float, short>(srcF, new Size(3, 3)));
        Assert.Throws<NotSupportedException>(() => Derivatives.Sobel<double, float>(srcD, 1, 0));
        Assert.Throws<NotSupportedException>(() => Derivatives.Laplacian<double, float>(srcD, 3));
        Assert.Throws<NotSupportedException>(() => Derivatives.Laplacian<short, byte>(new Mat<short>(8, 8, 1), 1));
        Assert.Throws<NotSupportedException>(() => Derivatives.Laplacian<byte, int>(new Mat<byte>(8, 8, 1), 5));
        Assert.Throws<NotSupportedException>(() => Filters.MedianBlur(new Mat<int>(4, 4, 1), 3));
    }

    // --------------------------------------------------------- 1-D medians

    [Fact]
    public void MedianBlur_SingleRow_UsesClampedNeighbours()
    {
        var src = new Mat<byte>(5, 1, 1);
        byte[] values = [9, 1, 7, 3, 5];
        for (int x = 0; x < 5; x++)
        {
            src.At(0, x) = values[x];
        }

        Mat<byte> result = Filters.MedianBlur(src, 3);

        // Medians of [9,9,1], [9,1,7], [1,7,3], [7,3,5], [3,5,5].
        Assert.Equal([9, 7, 3, 5, 5], result.ToArray());
    }
}
