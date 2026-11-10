namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins OpenCV 5.0 behaviour where cv2 4.5.5 (the golden reference) differs: the chamfer border
/// value <c>UINT_MAX - w</c> without the 8192 clamp (commit d752bac43f, OpenCV 4.9.0), and the
/// integer squares and 65536 cap of the precise transform (commit ddcbd2cc26, OpenCV 4.7.0, and
/// <c>PRECISE_DIST_MAX</c>). Expected values follow
/// <c>modules/imgproc/src/distransform.cpp</c> and <c>modules/imgproc/test/test_distancetransform.cpp</c>.
/// </summary>
public sealed class DistanceOpenCv5DifferenceTests
{
    /// <summary>
    /// Golden suspects on images without zero pixels: every pixel keeps the border value
    /// <c>(float)(UINT_MAX - w) / 65536</c>, with <c>w</c> the diagonal (3x3) or long (5x5) weight.
    /// cv2 4.5.5 gives 8192.
    /// </summary>
    [Theory]
    [InlineData(DistanceTypes.L2, DistanceTransformMasks.Mask3, 4294967295u - 89738u)]
    [InlineData(DistanceTypes.L1, DistanceTransformMasks.Mask3, 4294967295u - 131072u)]
    [InlineData(DistanceTypes.C, DistanceTransformMasks.Mask3, 4294967295u - 65536u)]
    [InlineData(DistanceTypes.L1, DistanceTransformMasks.Mask5, 4294967295u - 131072u)]
    [InlineData(DistanceTypes.L2, DistanceTransformMasks.Mask5, 4294967295u - 143976u)]
    public void DistanceTransform_NoZeroPixel_GivesBorderValue(DistanceTypes type, DistanceTransformMasks mask, uint border)
    {
        var src = new Mat<byte>(45, 7);
        src.Fill(3);
        float expected = (float)border / 65536;
        Assert.All(DistanceTransforms.DistanceTransform(src, type, mask).ToArray(), v => Assert.Equal(expected, v));
    }

    [Fact]
    public void DistanceTransform_NoZeroPixel_ExactValues()
    {
        var src = new Mat<byte>(40, 3);
        src.Fill(255);
        Assert.Equal(65534.62890625f, DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Mask3).At(1, 20));
        Assert.Equal(65533.8046875f, DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Mask5).At(1, 20));
        Assert.Equal(65534f, DistanceTransforms.DistanceTransform(src, DistanceTypes.L1, DistanceTransformMasks.Mask3).At(1, 20));
        Assert.Equal(65535f, DistanceTransforms.DistanceTransform(src, DistanceTypes.C, DistanceTransformMasks.Mask3).At(1, 20));
    }

    /// <summary>cv2 4.5.5 gives 31622776 (<c>sqrt(1e15)</c>); OpenCV 5.0 maps the missing column distance to <c>(float)UINT_MAX</c> and caps at 65536.</summary>
    [Fact]
    public void DistanceTransform_PreciseNoZeroPixel_Gives65536()
    {
        var src = new Mat<byte>(37, 5);
        src.Fill(1);
        Assert.All(DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Precise).ToArray(), v => Assert.Equal(65536f, v));
    }

    [Theory]
    [InlineData(DistanceTypes.L2, 65533.8046875f)]
    [InlineData(DistanceTypes.L1, 65533f)]
    [InlineData(DistanceTypes.C, 65534f)]
    public void DistanceTransformWithLabels_NoZeroPixel_GivesBorderValueAndLabelZero(DistanceTypes type, float expected)
    {
        var src = new Mat<byte>(33, 6);
        src.Fill(8);
        foreach (DistanceTransformLabelTypes labelType in new[] { DistanceTransformLabelTypes.ConnectedComponent, DistanceTransformLabelTypes.Pixel })
        {
            DistanceTransformResult r = DistanceTransforms.DistanceTransformWithLabels(src, type, DistanceTransformMasks.Mask5, labelType);
            Assert.All(r.Distances.ToArray(), v => Assert.Equal(expected, v));
            Assert.All(r.Labels.ToArray(), v => Assert.Equal(0, v));
        }
    }

    /// <summary>cv2 4.5.5 clamps chamfer distances at 8192 (<c>INT_MAX &gt;&gt; 2</c> in fixed point).</summary>
    [Fact]
    public void DistanceTransform_L1LongRow_HasNo8192Clamp()
    {
        var src = new Mat<byte>(10000, 1);
        src.Fill(1);
        src.At(0, 0) = 0;
        Mat<float> d = DistanceTransforms.DistanceTransform(src, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        for (int i = 0; i < 10000; i++)
        {
            Assert.Equal(i, d.At(0, i));
        }
    }

    /// <summary>OpenCV tests <c>max_distance_3x3</c>, <c>max_distance_5x5</c> and <c>max_distance_5x5_labels</c>; the values are the fixed-point sums.</summary>
    [Fact]
    public void DistanceTransform_MaxDistance_ReachesAbove65533()
    {
        var src = new Mat<byte>(70000, 1);
        src.Fill(1);
        src.At(0, 0) = 0;

        Mat<float> d3 = DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Mask3);
        Mat<float> d5 = DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Mask5);
        DistanceTransformResult labelled = DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L2, DistanceTransformMasks.Mask5);

        Assert.True(Max(d3) >= 65533);
        Assert.True(Max(d5) >= 65533);
        Assert.True(Max(labelled.Distances) >= 65533);
        for (int i = 0; i < 70000; i += 997)
        {
            // Horizontal steps only: i * HV, clamped at the border value in the forward pass.
            Assert.Equal((float)Math.Min(62587UL * (ulong)i, 4294967295UL - 89738UL) / 65536, d3.At(0, i));
            Assert.Equal((float)Math.Min(65536UL * (ulong)i, 4294967295UL - 143976UL) / 65536, d5.At(0, i));
            Assert.Equal(d5.At(0, i), labelled.Distances.At(0, i));
            // The labelled pass starts at the border value and keeps label 0 unless a candidate is smaller.
            Assert.Equal(65536UL * (ulong)i < 4294967295UL - 143976UL ? 1 : 0, labelled.Labels.At(0, i));
        }
    }

    /// <summary>OpenCV test <c>precise_long_dist</c>.</summary>
    [Fact]
    public void DistanceTransform_PreciseLongDistance_ClampsAt65536()
    {
        var src = new Mat<byte>(70000, 1);
        src.Fill(1);
        src.At(0, 0) = 0;
        Mat<float> d = DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Precise);
        for (int i = 0; i < 70000; i++)
        {
            Assert.Equal(Math.Min(i, 65536), d.At(0, i));
        }
    }

    /// <summary>
    /// OpenCV test <c>large_square_22732</c> at 3001 x 3006: squares above 2^24 are exact
    /// integers in OpenCV 5.0. cv2 4.5.5 squares in float.
    /// </summary>
    [Fact]
    public void DistanceTransform_PreciseLargeSquare_StaysWithinOnePixel()
    {
        const int w = 3006, h = 3001;
        Mat<float> zero = DistanceTransforms.DistanceTransform(new Mat<byte>(w, h), DistanceTypes.L2, DistanceTransformMasks.Precise);
        Assert.All(zero.ToArray(), v => Assert.Equal(0f, v));

        var src = new Mat<byte>(w, h);
        src.Fill(1);
        src.At(h - 1, w - 1) = 0;
        Mat<float> d = DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Precise);
        int errors = 0;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double dx = x - (w - 1), dy = y - (h - 1);
                if (Math.Abs((float)Math.Sqrt((dx * dx) + (dy * dy)) - d.At(y, x)) > 1)
                {
                    errors++;
                }
            }
        }

        Assert.Equal(0, errors);
    }

    private static float Max(Mat<float> m)
    {
        float max = float.MinValue;
        foreach (float v in m.ToArray())
        {
            max = Math.Max(max, v);
        }

        return max;
    }
}
