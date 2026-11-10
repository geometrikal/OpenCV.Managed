namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Analytic behavioural tests for <see cref="Morphology"/> and
/// <see cref="Borders"/> that do not depend on the golden JSON: ROI/stride
/// correctness, source immutability, determinism, morphological identities,
/// OpenCV 5's diamond structuring element (absent from cv2 4.5.5 goldens),
/// the iteration-folding rules, and the validation contract.
/// </summary>
public sealed class MorphologyBasicTests
{
    // -- ROI / stride (MANDATORY) --------------------------------------------

    [Fact]
    public void Morphology_OnRoi_MatchesPackedMat_AndLeavesParentUntouched()
    {
        var parent = new Mat<byte>(30, 26);
        FillPattern(parent);
        Mat<byte> roi = parent.Roi(4, 3, 18, 16);
        Assert.False(roi.IsContinuous);

        Mat<byte> packed = roi.Clone();
        byte[] parentBefore = parent.ToArray();

        Mat<sbyte> ellipse = Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(5, 5));

        foreach ((string name, Func<Mat<byte>, Mat<byte>> run) in new (string, Func<Mat<byte>, Mat<byte>>)[]
        {
            ("erode", m => Morphology.Erode(m, kernel: null)),
            ("dilate-ellipse", m => Morphology.Dilate(m, ellipse)),
            ("open", m => Morphology.MorphologyEx(m, MorphOperation.Open, ellipse)),
            ("gradient", m => Morphology.MorphologyEx(m, MorphOperation.Gradient, null)),
            ("border", m => Borders.CopyMakeBorder(m, 2, 1, 3, 2, BorderType.Reflect101)),
        })
        {
            Mat<byte> fromRoi = run(roi);
            Mat<byte> fromPacked = run(packed);
            Assert.True(
                fromRoi.ToArray().AsSpan().SequenceEqual(fromPacked.ToArray()),
                $"ROI result differs from packed result for '{name}'.");
        }

        // Inputs (including the parent around the ROI) are never written.
        Assert.Equal(parentBefore, parent.ToArray());
    }

    // -- Non-mutation and determinism ----------------------------------------

    [Fact]
    public void Morphology_DoesNotModifySource_AndIsDeterministic()
    {
        var src = new Mat<byte>(20, 20);
        FillPattern(src);
        byte[] before = src.ToArray();

        Mat<byte> a = Morphology.MorphologyEx(src, MorphOperation.Close, null, iterations: 2);
        Mat<byte> b = Morphology.MorphologyEx(src, MorphOperation.Close, null, iterations: 2);

        Assert.Equal(before, src.ToArray());
        Assert.Equal(a.ToArray(), b.ToArray());
    }

    // -- Morphological identities ---------------------------------------------

    [Fact]
    public void ErodeDilate_DualityUnderComplement()
    {
        // erode(x) == 255 - dilate(255 - x) with default borders (erode pads
        // max, dilate pads min — the duality holds exactly).
        var src = new Mat<byte>(17, 13);
        FillPattern(src);

        Mat<sbyte> kernel = Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(5, 3));
        Mat<byte> eroded = Morphology.Erode(src, kernel);

        var inverted = new Mat<byte>(src.Width, src.Height);
        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                inverted.At(y, x) = (byte)(255 - src.At(y, x));
            }
        }

        Mat<byte> dualDilated = Morphology.Dilate(inverted, kernel);
        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                Assert.Equal(eroded.At(y, x), (byte)(255 - dualDilated.At(y, x)));
            }
        }
    }

    [Fact]
    public void Gradient_EqualsDilateMinusErode_AndOpenIsIdempotent()
    {
        var src = new Mat<byte>(19, 16);
        FillPattern(src);
        Mat<sbyte> kernel = Morphology.GetStructuringElement(MorphShape.Cross, new Size(3, 3));

        Mat<byte> gradient = Morphology.MorphologyEx(src, MorphOperation.Gradient, kernel);
        Mat<byte> dilated = Morphology.Dilate(src, kernel);
        Mat<byte> eroded = Morphology.Erode(src, kernel);
        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                Assert.Equal((byte)Math.Max(dilated.At(y, x) - eroded.At(y, x), 0), gradient.At(y, x));
            }
        }

        Mat<byte> opened = Morphology.MorphologyEx(src, MorphOperation.Open, kernel);
        Mat<byte> openedTwice = Morphology.MorphologyEx(opened, MorphOperation.Open, kernel);
        Assert.Equal(opened.ToArray(), openedTwice.ToArray());
    }

    [Fact]
    public void Erode_IterationFolding_MatchesRepeatedApplication()
    {
        // For all-ones kernels OpenCV folds iterations into one bigger kernel;
        // the result must equal both explicit repetition and the equivalent
        // expanded kernel.
        var src = new Mat<byte>(21, 18);
        FillPattern(src);
        Mat<sbyte> rect3 = Morphology.GetStructuringElement(MorphShape.Rect, new Size(3, 3));
        Mat<sbyte> rect5 = Morphology.GetStructuringElement(MorphShape.Rect, new Size(5, 5));

        Mat<byte> folded = Morphology.Erode(src, rect3, iterations: 2);
        Mat<byte> repeated = Morphology.Erode(Morphology.Erode(src, rect3), rect3);
        Mat<byte> expanded = Morphology.Erode(src, rect5);

        Assert.Equal(repeated.ToArray(), folded.ToArray());
        Assert.Equal(expanded.ToArray(), folded.ToArray());
    }

    [Fact]
    public void Erode_IterationsZeroOrOneByOneKernel_ReturnsCopy()
    {
        var src = new Mat<byte>(9, 7);
        FillPattern(src);

        Mat<byte> zeroIterations = Morphology.Erode(src, kernel: null, iterations: 0);
        Assert.Equal(src.ToArray(), zeroIterations.ToArray());

        var oneByOne = new Mat<sbyte>(1, 1);
        oneByOne.At(0, 0) = 1;
        Mat<byte> tiny = Morphology.Dilate(src, oneByOne);
        Assert.Equal(src.ToArray(), tiny.ToArray());
    }

    // -- Diamond structuring element (OpenCV 5 addition; no cv2 golden) -------

    [Fact]
    public void GetStructuringElement_Diamond5x5_IsManhattanBall()
    {
        Mat<sbyte> elem = Morphology.GetStructuringElement(MorphShape.Diamond, new Size(5, 5));

        // r = c = 2: cell set iff |x - 2| + |y - 2| <= 2.
        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 5; x++)
            {
                sbyte expected = Math.Abs(x - 2) + Math.Abs(y - 2) <= 2 ? (sbyte)1 : (sbyte)0;
                Assert.Equal(expected, elem.At(y, x));
            }
        }
    }

    // -- Validation ------------------------------------------------------------

    [Fact]
    public void Morphology_ValidationContract()
    {
        var src = new Mat<byte>(8, 8);
        var srcInt = new Mat<int>(8, 8);
        var src5Ch = new Mat<byte>(8, 8, channels: 5);
        Mat<sbyte> rect3 = Morphology.GetStructuringElement(MorphShape.Rect, new Size(3, 3));

        Assert.Throws<ArgumentNullException>(() => Morphology.Erode<byte>(null!, rect3));
        Assert.Throws<NotSupportedException>(() => Morphology.Erode(srcInt, rect3));
        Assert.Throws<NotSupportedException>(() => Morphology.Erode(src5Ch, rect3));
        Assert.Throws<ArgumentOutOfRangeException>(() => Morphology.Erode(src, rect3, iterations: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Morphology.Erode(src, rect3, anchor: new Point(3, 1)));
        Assert.Throws<NotSupportedException>(() =>
            Morphology.Erode(src, rect3, borderType: BorderType.Wrap));
        Assert.Throws<ArgumentException>(() =>
            Morphology.Erode(src, new Mat<sbyte>(3, 3, channels: 2)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Morphology.MorphologyEx(src, (MorphOperation)99, rect3));

        // Hit-or-miss requires a single-channel byte matrix.
        var srcU16 = new Mat<ushort>(8, 8);
        Assert.Throws<ArgumentException>(() =>
            Morphology.MorphologyEx(srcU16, MorphOperation.HitMiss, rect3));
        var src3Ch = new Mat<byte>(8, 8, channels: 3);
        Assert.Throws<ArgumentException>(() =>
            Morphology.MorphologyEx(src3Ch, MorphOperation.HitMiss, rect3));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Morphology.GetStructuringElement(MorphShape.Rect, new Size(0, 3)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Morphology.GetStructuringElement((MorphShape)9, new Size(3, 3)));

        Assert.Throws<ArgumentOutOfRangeException>(() => Borders.BorderInterpolate(1, 0, BorderType.Reflect));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Borders.CopyMakeBorder(src, -1, 0, 0, 0, BorderType.Replicate));
    }

    [Theory]
    [InlineData(BorderType.Constant, 1)]
    [InlineData(BorderType.Reflect101, 3)]
    [InlineData(BorderType.Wrap, 3)]
    [InlineData(BorderType.Replicate, 4)]
    public void CopyMakeBorder_Destination_EqualsAllocatingOverload(BorderType type, int channels)
    {
        var src = new Mat<byte>(9, 7, channels);
        FillPattern(src);
        var value = new Scalar(10, 20, 30, 40);
        Mat<byte> expected = Borders.CopyMakeBorder(src, 3, 2, 5, 4, type, value);

        var dst = new Mat<byte>(18, 12, channels);
        Borders.CopyMakeBorder(src, dst, 3, 2, 5, 4, type, value);
        Assert.Equal(expected.ToArray(), dst.ToArray());

        // A destination view that shares the source array gets the same result.
        var parent = new Mat<byte>(18, 19, channels);
        Mat<byte> roiSrc = parent.Roi(0, 12, 9, 7);
        src.CopyTo(roiSrc);
        Borders.CopyMakeBorder(roiSrc, parent.Roi(0, 0, 18, 12), 3, 2, 5, 4, type, value);
        Assert.Equal(expected.ToArray(), parent.Roi(0, 0, 18, 12).ToArray());

        Assert.Throws<ArgumentException>(() => Borders.CopyMakeBorder(src, new Mat<byte>(17, 12, channels), 3, 2, 5, 4, type, value));
    }

    [Fact]
    public void BorderInterpolate_ConstantOutOfRange_ReturnsMinusOne()
    {
        Assert.Equal(-1, Borders.BorderInterpolate(-3, 10, BorderType.Constant));
        Assert.Equal(-1, Borders.BorderInterpolate(12, 10, BorderType.Constant));
        Assert.Equal(4, Borders.BorderInterpolate(4, 10, BorderType.Constant));
    }

    private static void FillPattern(Mat<byte> m)
    {
        for (int y = 0; y < m.Height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                row[x] = (byte)(((x * 31) + (y * 57) + ((x * y) % 13)) & 0xFF);
            }
        }
    }
}
