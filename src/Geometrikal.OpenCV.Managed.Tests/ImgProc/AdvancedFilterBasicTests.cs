namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Structural and analytic tests for the advanced filtering port (filter2D DFT path,
/// sqrBoxFilter, boxFilter column scaling, bilateralFilter, guidedFilter, stackBlur): ROI
/// stride, validation, non-mutation, determinism, edge sizes and the deliberate differences
/// from OpenCV.
/// </summary>
public sealed class AdvancedFilterBasicTests
{
    internal static Mat<byte> PatternU8(int width, int height, int channels, int seed = 0)
    {
        var m = new Mat<byte>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = (byte)(((i * 31) + (y * 17) + (seed * 13) + ((i * y) % 7)) % 256);
            }
        }

        return m;
    }

    internal static Mat<float> PatternF32(int width, int height, int channels, int seed = 0)
    {
        var m = new Mat<float>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<float> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = ((((i * 37) + (y * 11) + seed) % 199) - 99.5f) * 0.73f;
            }
        }

        return m;
    }

    private static Mat<float> Kernel(int kw, int kh)
    {
        var k = new Mat<float>(kw, kh);
        Span<float> v = k.GetSpan();
        for (int i = 0; i < v.Length; i++)
        {
            v[i] = (((i * 37) % 23) - 11) / (float)(kw * kh * 4);
        }

        return k;
    }

    /// <summary>A 60×50 parent with a 37×29 ROI at (7, 9); the ROI is not continuous.</summary>
    private static (Mat<T> Roi, Mat<T> Packed) RoiPair<T>(Mat<T> parent)
        where T : unmanaged
    {
        Mat<T> roi = parent.Roi(7, 9, 37, 29);
        Assert.False(roi.IsContinuous);
        return (roi, roi.Clone());
    }

    // ------------------------------------------------------------------ filter2D

    [Fact]
    public void Filter2D_DftPath_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(60, 50, 3));
        Mat<float> k = Kernel(13, 11);
        Assert.Equal(
            Filters.Filter2D<byte, byte>(packed, k, null, 2.5).ToArray(),
            Filters.Filter2D<byte, byte>(roi, k, null, 2.5).ToArray());
    }

    [Fact]
    public void Filter2D_DftPath_FloatMatchesDoubleCorrelation()
    {
        Mat<float> src = PatternF32(300, 280, 1);
        Mat<float> k = Kernel(13, 11);
        Mat<float> dst = Filters.Filter2D<float, float>(src, k, new Point(2, 3), 0, BorderType.Reflect);
        for (int y = 0; y < src.Height; y += 7)
        {
            for (int x = 0; x < src.Width; x += 5)
            {
                double acc = 0;
                for (int i = 0; i < 11; i++)
                {
                    for (int j = 0; j < 13; j++)
                    {
                        int sy = Borders.BorderInterpolate(y + i - 3, src.Height, BorderType.Reflect);
                        int sx = Borders.BorderInterpolate(x + j - 2, src.Width, BorderType.Reflect);
                        acc += k.At(i, j) * (double)src.At(sy, sx);
                    }
                }

                Assert.True(Math.Abs(acc - dst.At(y, x)) < 1e-3, $"({x},{y}): {acc} vs {dst.At(y, x)}");
            }
        }
    }

    [Fact]
    public void Filter2D_WrapBorder_AcceptedOnlyOnDftPath()
    {
        Mat<byte> src = PatternU8(40, 30, 1);
        Assert.Throws<NotSupportedException>(() => Filters.Filter2D<byte, byte>(src, Kernel(11, 11), null, 0, BorderType.Wrap));
        Mat<byte> dft = Filters.Filter2D<byte, byte>(src, Kernel(13, 11), null, 0, BorderType.Wrap);
        Assert.Equal(src.Width, dft.Width);

        // byte→float switches to the DFT path at an area of 50.
        Assert.Throws<NotSupportedException>(() => Filters.Filter2D<byte, float>(src, Kernel(7, 7), null, 0, BorderType.Wrap));
        _ = Filters.Filter2D<byte, float>(src, Kernel(10, 5), null, 0, BorderType.Wrap);
    }

    [Fact]
    public void Filter2D_DftPath_TinyImage_AndSourceUnchanged()
    {
        Mat<float> src = PatternF32(1, 1, 3);
        float[] before = src.ToArray();
        Mat<float> dst = Filters.Filter2D<float, float>(src, Kernel(31, 31), null, 0, BorderType.Replicate);
        Assert.Equal(before, src.ToArray());
        float sum = 0;
        foreach (float v in Kernel(31, 31).ToArray())
        {
            sum += v;
        }

        Assert.Equal(before[0] * sum, dst.ToArray()[0], 3);
    }

    [Fact]
    public void Filter2D_DftPath_RepeatedCalls_AreDeterministic()
    {
        Mat<byte> src = PatternU8(523, 271, 1);
        Mat<float> k = Kernel(25, 25);
        Assert.Equal(Filters.Filter2D<byte, short>(src, k).ToArray(), Filters.Filter2D<byte, short>(src, k).ToArray());
    }

    // -------------------------------------------------------- box / sqrBoxFilter

    /// <summary>
    /// OpenCV's int32 column sum scales the first 4·⌊W·cn/4⌋ elements with a float factor and the
    /// rest with a double factor. A constant image with sum 15·3 = 45 and scale 1/15 shows
    /// both: 45f·(float)(1/15) = 3.0000002 and 45·(1/15) = 3.
    /// </summary>
    [Fact]
    public void BoxFilter_ByteToFloatNormalized_UsesFloatScaleOnVectorColumns()
    {
        var src = new Mat<byte>(7, 6);
        src.Fill(3);
        Mat<float> dst = Filters.BoxFilter<byte, float>(src, new Size(5, 3));
        float vector = 45f * (float)(1.0 / 15);
        float scalar = (float)(45 * (1.0 / 15));
        Assert.NotEqual(vector, scalar);
        for (int y = 0; y < 6; y++)
        {
            for (int x = 0; x < 7; x++)
            {
                Assert.Equal(x < 4 ? vector : scalar, dst.At(y, x));
            }
        }
    }

    [Fact]
    public void SqrBoxFilter_DefaultOverloads_ReturnOpenCvDepths()
    {
        Mat<float> f = Filters.SqrBoxFilter(PatternU8(9, 7, 1), new Size(3, 3));
        Mat<double> d = Filters.SqrBoxFilter(PatternF32(9, 7, 1), new Size(3, 3));
        Assert.Equal((9, 7), (f.Width, f.Height));
        Assert.Equal((9, 7), (d.Width, d.Height));
    }

    [Fact]
    public void SqrBoxFilter_Unnormalized_IsSumOfSquares()
    {
        Mat<byte> src = PatternU8(11, 9, 1);
        Mat<int> dst = Filters.SqrBoxFilter<byte, int>(src, new Size(3, 3), null, false, BorderType.Replicate);
        for (int y = 0; y < 9; y++)
        {
            for (int x = 0; x < 11; x++)
            {
                int expected = 0;
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        int v = src.At(Math.Clamp(y + i, 0, 8), Math.Clamp(x + j, 0, 10));
                        expected += v * v;
                    }
                }

                Assert.Equal(expected, dst.At(y, x));
            }
        }
    }

    /// <summary>sqrBoxFilter shrinks the kernel of a single-row image only for non-constant borders.</summary>
    [Fact]
    public void SqrBoxFilter_SingleRow_ShrinksOnlyForNonConstantBorder()
    {
        Mat<float> src = PatternF32(17, 1, 1);
        Mat<double> constant = Filters.SqrBoxFilter<float, double>(src, new Size(3, 3), null, true, BorderType.Constant);
        Mat<double> reflect = Filters.SqrBoxFilter<float, double>(src, new Size(3, 3), null, true, BorderType.Reflect);
        Mat<double> reflectRow = Filters.SqrBoxFilter<float, double>(src, new Size(3, 1), null, true, BorderType.Reflect);
        Mat<double> constantRow = Filters.SqrBoxFilter<float, double>(src, new Size(3, 1), null, true, BorderType.Constant);
        Assert.Equal(reflectRow.ToArray(), reflect.ToArray());
        Assert.NotEqual(constantRow.ToArray(), constant.ToArray());
    }

    [Fact]
    public void SqrBoxFilter_RoiWithStride_MatchesPackedImage()
    {
        (Mat<float> roi, Mat<float> packed) = RoiPair(PatternF32(60, 50, 3));
        Assert.Equal(
            Filters.SqrBoxFilter(packed, new Size(9, 5)).ToArray(),
            Filters.SqrBoxFilter(roi, new Size(9, 5)).ToArray());
    }

    [Fact]
    public void SqrBoxFilter_InvalidArguments_Throw()
    {
        Mat<byte> src = PatternU8(8, 8, 1);
        Assert.Throws<ArgumentNullException>(() => Filters.SqrBoxFilter((Mat<byte>)null!, new Size(3, 3)));
        Assert.Throws<NotSupportedException>(() => Filters.SqrBoxFilter<byte, short>(src, new Size(3, 3)));
        Assert.Throws<NotSupportedException>(() => Filters.SqrBoxFilter<float, int>(PatternF32(8, 8, 1), new Size(3, 3)));
        Assert.Throws<NotSupportedException>(() => Filters.SqrBoxFilter(src, new Size(3, 3), null, true, BorderType.Wrap));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.SqrBoxFilter(src, new Size(0, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.SqrBoxFilter(src, new Size(3, 3), new Point(3, 0)));
    }

    // ------------------------------------------------------------ bilateralFilter

    [Fact]
    public void BilateralFilter_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(60, 50, 3));
        Assert.Equal(Filters.BilateralFilter(packed, 7, 40, 3).ToArray(), Filters.BilateralFilter(roi, 7, 40, 3).ToArray());
        (Mat<float> roiF, Mat<float> packedF) = RoiPair(PatternF32(60, 50, 1));
        Assert.Equal(Filters.BilateralFilter(packedF, 5, 30, 2, BorderType.Wrap).ToArray(), Filters.BilateralFilter(roiF, 5, 30, 2, BorderType.Wrap).ToArray());
    }

    /// <summary>OpenCV 5.0 returns a copy for a sigma ≤ 1e-6; OpenCV 4.x replaced a non-positive sigma with 1.</summary>
    [Theory]
    [InlineData(0.0, 3.0)]
    [InlineData(1e-6, 3.0)]
    [InlineData(25.0, 0.0)]
    public void BilateralFilter_SigmaAtMostEpsilon_ReturnsCopy(double sigmaColor, double sigmaSpace)
    {
        Mat<byte> src = PatternU8(21, 13, 3);
        Assert.Equal(src.ToArray(), Filters.BilateralFilter(src, 5, sigmaColor, sigmaSpace).ToArray());
        Mat<float> srcF = PatternF32(21, 13, 1);
        Assert.Equal(srcF.ToArray(), Filters.BilateralFilter(srcF, 5, sigmaColor, sigmaSpace).ToArray());
    }

    [Fact]
    public void BilateralFilter_ConstantFloatImage_ReturnsCopy()
    {
        var src = new Mat<float>(9, 7);
        src.Fill(-2.25f);
        Assert.Equal(src.ToArray(), Filters.BilateralFilter(src, 9, 10, 10).ToArray());
    }

    /// <summary>
    /// Deliberate difference: the port takes the colour range with NaN-skipping min/max. OpenCV
    /// 5.0 starts <c>minMaxLoc</c> from the first element, so a NaN there makes its whole table
    /// NaN. In the port the finite pixels stay finite.
    /// </summary>
    [Fact]
    public void BilateralFilter_LeadingNaN_KeepsFinitePixelsFinite()
    {
        Mat<float> src = PatternF32(19, 11, 1);
        src.At(0, 0) = BitConverter.Int32BitsToSingle(0x7fc00000);
        Mat<float> dst = Filters.BilateralFilter(src, 5, 20, 3);
        float[] values = dst.ToArray();
        for (int i = 1; i < values.Length; i++)
        {
            Assert.True(float.IsFinite(values[i]), $"element {i}");
        }
    }

    /// <summary>The last four 8-bit colour weights use <c>expf</c>; the others the vector exponential (cv2 5.0.0 values).</summary>
    [Theory]
    [InlineData(30.0, 1, 250, new[] { 0x266fec1c, 0x2635a1a6, 0x260959a7, 0x25cf7fae, 0x259c902c, 0x256bfff1 })]
    [InlineData(75.0, 1, 250, new[] { 0x3b7d5b5c, 0x3b725255, 0x3b67b9cd, 0x3b5d8dca, 0x3b53ca77, 0x3b4a6c09 })]
    [InlineData(75.0, 3, 762, new[] { 0x1a39e939, 0x1a2257df, 0x1a0dbcdc, 0x19f773a5, 0x19d7f796, 0x19bc7495 })]
    public void BilateralFilter_ColorWeightTable_MatchesOpenCv(double sigmaColor, int cn, int first, int[] bits)
    {
        float[] table = BilateralEngine.ColorWeights8u((float)(-0.5 / (sigmaColor * sigmaColor)), cn);
        for (int i = 0; i < bits.Length; i++)
        {
            Assert.Equal(bits[i], BitConverter.SingleToInt32Bits(table[first + i]));
        }
    }

    [Theory]
    [InlineData(-0.0f, 0x3f800000)]
    [InlineData(-0.5f, 0x3f1b4598)]
    [InlineData(-1.0f, 0x3ebc5ab2)]
    [InlineData(-3.7f, 0x3cca88fe)]
    [InlineData(-10.25f, 0x38144cd4)]
    [InlineData(-50.0f, 0x1b692beb)]
    [InlineData(-87.9f, 0)]
    [InlineData(-100.0f, 0)]
    public void BilateralFilter_VectorExp_MatchesOpenCvLanes(float x, int bits)
        => Assert.Equal(bits, BitConverter.SingleToInt32Bits(BilateralEngine.VExp(x)));

    [Theory]
    [InlineData(1, 0)]
    [InlineData(2, 0)]
    [InlineData(4, 0)]
    [InlineData(1, -1)]
    [InlineData(3, 9)]
    public void BilateralFilter_TinyImages_DoNotThrow(int size, int d)
    {
        Assert.Equal(size, Filters.BilateralFilter(PatternU8(size, size, 1), d, 30, 2).Width);
        Assert.Equal(size, Filters.BilateralFilter(PatternF32(size, size, 3), d, 30, 2).Width);
    }

    [Fact]
    public void BilateralFilter_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentNullException>(() => Filters.BilateralFilter((Mat<byte>)null!, 5, 1, 1));
        Assert.Throws<NotSupportedException>(() => Filters.BilateralFilter(new Mat<byte>(4, 4, 2), 5, 1, 1));
        Assert.Throws<NotSupportedException>(() => Filters.BilateralFilter(new Mat<byte>(4, 4, 4), 5, 1, 1));
        Assert.Throws<NotSupportedException>(() => Filters.BilateralFilter(new Mat<ushort>(4, 4, 1), 5, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.BilateralFilter(new Mat<byte>(4, 4, 1), 5, 1, 1, BorderType.Transparent));
    }

    [Fact]
    public void BilateralFilter_SourceUnchanged_AndDeterministic()
    {
        Mat<float> src = PatternF32(77, 23, 3);
        float[] before = src.ToArray();
        float[] a = Filters.BilateralFilter(src, 9, 25, 4).ToArray();
        float[] b = Filters.BilateralFilter(src, 9, 25, 4).ToArray();
        Assert.Equal(before, src.ToArray());
        Assert.Equal(a, b);
    }

    // -------------------------------------------------------------- guidedFilter

    [Fact]
    public void GuidedFilter_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roiGuide, Mat<byte> packedGuide) = RoiPair(PatternU8(60, 50, 3, 1));
        (Mat<float> roiSrc, Mat<float> packedSrc) = RoiPair(PatternF32(60, 50, 1, 2));
        Assert.Equal(
            EdgePreservingFilters.GuidedFilter(packedGuide, packedSrc, 3, 50).ToArray(),
            EdgePreservingFilters.GuidedFilter(roiGuide, roiSrc, 3, 50).ToArray());
    }

    [Fact]
    public void GuidedFilter_ReusedObject_MatchesOneShotCall()
    {
        Mat<ushort> guide = new(41, 29, 2);
        Span<ushort> g = guide.GetSpan();
        for (int i = 0; i < g.Length; i++)
        {
            g[i] = (ushort)((i * 977) % 60000);
        }

        GuidedFilter filter = GuidedFilter.Create(guide, 2, 1e4);
        Mat<byte> a = PatternU8(41, 29, 3, 4);
        Mat<float> b = PatternF32(41, 29, 1, 5);
        Assert.Equal(EdgePreservingFilters.GuidedFilter(guide, a, 2, 1e4).ToArray(), filter.Filter(a).ToArray());
        Assert.Equal(EdgePreservingFilters.GuidedFilter<ushort, float, byte>(guide, b, 2, 1e4).ToArray(), filter.Filter<float, byte>(b).ToArray());
        Assert.Equal(filter.Filter(a).ToArray(), filter.Filter(a).ToArray());
    }

    /// <summary>A constant source stays constant; the guide does not leak into the result.</summary>
    [Fact]
    public void GuidedFilter_ConstantSource_StaysConstant()
    {
        var src = new Mat<float>(23, 17);
        src.Fill(42f);
        Mat<float> dst = EdgePreservingFilters.GuidedFilter(PatternU8(23, 17, 3), src, 4, 100);
        foreach (float v in dst.ToArray())
        {
            Assert.Equal(42f, v, 3);
        }
    }

    /// <summary>Byte and float guides with equal values give equal results (the guide is converted exactly).</summary>
    [Fact]
    public void GuidedFilter_ByteAndFloatGuides_GiveEqualResults()
    {
        Mat<byte> guide = PatternU8(3, 40, 1, 7);
        Mat<float> src = PatternF32(3, 40, 1, 8);
        Mat<float> a = EdgePreservingFilters.GuidedFilter(guide, src, 1, 10);
        Mat<float> b = EdgePreservingFilters.GuidedFilter(Conversions.ConvertTo<byte, float>(guide), src, 1, 10);
        Assert.Equal(a.ToArray(), b.ToArray());
    }

    [Fact]
    public void GuidedFilter_SourceUnchanged()
    {
        Mat<byte> guide = PatternU8(31, 22, 3);
        Mat<byte> src = PatternU8(31, 22, 3, 9);
        byte[] g = guide.ToArray(), s = src.ToArray();
        _ = EdgePreservingFilters.GuidedFilter(guide, src, 2, 30, 0.5);
        Assert.Equal(g, guide.ToArray());
        Assert.Equal(s, src.ToArray());
    }

    [Fact]
    public void GuidedFilter_InvalidArguments_Throw()
    {
        Mat<byte> guide = PatternU8(8, 8, 1);
        Assert.Throws<ArgumentNullException>(() => GuidedFilter.Create<byte>(null!, 1, 1));
        Assert.Throws<NotSupportedException>(() => GuidedFilter.Create(new Mat<byte>(8, 8, 4), 1, 1));
        Assert.Throws<NotSupportedException>(() => GuidedFilter.Create(new Mat<short>(8, 8, 1), 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => GuidedFilter.Create(guide, -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => GuidedFilter.Create(guide, 1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => GuidedFilter.Create(guide, 1, 1, 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => GuidedFilter.Create(guide, 1, 1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => GuidedFilter.Create(guide, 1, 1, 0.01));
        GuidedFilter f = GuidedFilter.Create(guide, 1, 1);
        Assert.Throws<ArgumentException>(() => f.Filter(new Mat<byte>(9, 8)));
        Assert.Throws<NotSupportedException>(() => f.Filter(new Mat<ushort>(8, 8)));
        Assert.Throws<NotSupportedException>(() => f.Filter<byte, int>(new Mat<byte>(8, 8)));
    }

    // ----------------------------------------------------------------- stackBlur

    [Fact]
    public void StackBlur_RoiWithStride_MatchesPackedImage()
    {
        (Mat<byte> roi, Mat<byte> packed) = RoiPair(PatternU8(60, 50, 3));
        Assert.Equal(Filters.StackBlur(packed, new Size(7, 5)).ToArray(), Filters.StackBlur(roi, new Size(7, 5)).ToArray());
        (Mat<float> roiF, Mat<float> packedF) = RoiPair(PatternF32(60, 50, 1));
        Assert.Equal(Filters.StackBlur(packedF, new Size(15, 1)).ToArray(), Filters.StackBlur(roiF, new Size(15, 1)).ToArray());
    }

    [Fact]
    public void StackBlur_UnitKernel_ReturnsCopy()
    {
        Mat<float> src = PatternF32(13, 9, 2);
        Assert.Equal(src.ToArray(), Filters.StackBlur(src, new Size(1, 1)).ToArray());
    }

    /// <summary>
    /// The vector division <c>(s·mul) &gt;&gt; shr</c> wraps in 32 bits for radius 181 and bright
    /// images: the first 16·⌊W/16⌋ columns give 255 (OpenCV 5.0.0 behaviour), the scalar columns
    /// give the mean.
    /// </summary>
    [Fact]
    public void StackBlur_LargeRadiusVectorColumns_ReproduceOverflow()
    {
        var src = new Mat<byte>(40, 450);
        src.Fill(250);
        Mat<byte> dst = Filters.StackBlur(src, new Size(1, 363));
        for (int x = 0; x < 40; x++)
        {
            Assert.Equal(x < 32 ? 255 : 250, dst.At(200, x));
        }
    }

    [Fact]
    public void StackBlur_ConstantFloatImage_StaysConstant()
    {
        var src = new Mat<float>(29, 21, 3);
        src.Fill(8f);
        foreach (float v in Filters.StackBlur(src, new Size(9, 31)).ToArray())
        {
            Assert.Equal(8f, v, 4);
        }
    }

    [Fact]
    public void StackBlur_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentNullException>(() => Filters.StackBlur((Mat<byte>)null!, new Size(3, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.StackBlur(new Mat<byte>(4, 4), new Size(2, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Filters.StackBlur(new Mat<byte>(4, 4), new Size(3, -1)));
        Assert.Throws<NotSupportedException>(() => Filters.StackBlur(new Mat<ushort>(4, 4), new Size(3, 3)));
    }

    [Fact]
    public void StackBlur_SourceUnchanged_AndDeterministic()
    {
        Mat<byte> src = PatternU8(333, 41, 3);
        byte[] before = src.ToArray();
        byte[] a = Filters.StackBlur(src, new Size(21, 9)).ToArray();
        Assert.Equal(a, Filters.StackBlur(src, new Size(21, 9)).ToArray());
        Assert.Equal(before, src.ToArray());
    }
}

/// <summary>
/// Runs alone: it clears the process-wide switch that selects the AVX2 bilateral blocks.
/// </summary>
[Collection(nameof(Geometrikal.OpenCV.Managed.Tests.Core.CvParallelCollection))]
public sealed class BilateralEmulationTests
{
    /// <summary>The scalar emulation of the bilateral vector formula equals the AVX2 blocks.</summary>
    [Fact]
    public void BilateralFilter_ScalarEmulation_MatchesVectorBlocks()
    {
        if (!Avx2.IsSupported)
        {
            return;
        }

        Mat<byte> b1 = AdvancedFilterBasicTests.PatternU8(77, 19, 1), b3 = AdvancedFilterBasicTests.PatternU8(45, 17, 3);
        Mat<float> f1 = AdvancedFilterBasicTests.PatternF32(77, 19, 1), f3 = AdvancedFilterBasicTests.PatternF32(45, 17, 3);
        f1.At(3, 5) = BitConverter.Int32BitsToSingle(0x7fc00000);
        f3.At(4, 9, 1) = BitConverter.Int32BitsToSingle(0x7fc00000);
        byte[][] vb = [Filters.BilateralFilter(b1, 5, 30, 3).ToArray(), Filters.BilateralFilter(b1, 9, 50, 3).ToArray(), Filters.BilateralFilter(b3, 7, 40, 3).ToArray()];
        float[][] vf = [Filters.BilateralFilter(f1, 7, 25, 3).ToArray(), Filters.BilateralFilter(f3, 5, 40, 3).ToArray()];
        try
        {
            BilateralEngine.VectorBodyEnabled = false;
            Assert.Equal(vb[0], Filters.BilateralFilter(b1, 5, 30, 3).ToArray());
            Assert.Equal(vb[1], Filters.BilateralFilter(b1, 9, 50, 3).ToArray());
            Assert.Equal(vb[2], Filters.BilateralFilter(b3, 7, 40, 3).ToArray());
            Assert.Equal(vf[0], Filters.BilateralFilter(f1, 7, 25, 3).ToArray());
            Assert.Equal(vf[1], Filters.BilateralFilter(f3, 5, 40, 3).ToArray());
        }
        finally
        {
            BilateralEngine.VectorBodyEnabled = true;
        }
    }
}
