namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins OpenCV 5.0 behaviour where cv2 4.5.5 (the golden reference) differs or has no
/// equivalent. Expected values follow <c>modules/imgproc/src/resize.cpp</c>
/// (<c>resizeNN_bitexact_tab</c>), <c>modules/imgproc/src/imgwarp.cpp</c> (<c>remapBilinear</c>
/// with <c>BORDER_TRANSPARENT</c>, <c>WARP_RELATIVE_MAP</c>) and
/// <c>modules/geometry/src/geometry.cpp</c> (<c>getPerspectiveTransform</c>).
/// </summary>
public sealed class GeometryOpenCv5DifferenceTests
{
    private static Mat<T> Ramp<T>(int width, int height, int channels, Func<int, T> value)
        where T : unmanaged
    {
        var m = new Mat<T>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<T> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = value((y * row.Length) + i);
            }
        }

        return m;
    }

    /// <summary>
    /// Golden cases <c>resize_x_nnx_*_64x48_to_32x24_synth</c>: 5.0 samples
    /// <c>floor(2·(d + 0.5)) = 2d + 1</c>; 4.5.5 samples <c>(131072·d + 65535) &gt;&gt; 16 = 2d</c>.
    /// </summary>
    [Fact]
    public void ResizeNearestExact_HalfSize_SamplesOddPixels()
    {
        Mat<ushort> src = Ramp<ushort>(64, 48, 1, i => (ushort)i);
        Mat<ushort> dst = Geometry.Resize(src, new Size(32, 24), interpolation: InterpolationFlags.NearestExact);
        for (int y = 0; y < 24; y++)
        {
            for (int x = 0; x < 32; x++)
            {
                Assert.Equal(src.At((2 * y) + 1, (2 * x) + 1), dst.At(y, x));
            }
        }
    }

    /// <summary>
    /// Golden case <c>resize_x_nnx_u8_c1_40000x1_to_30000x1_synth</c>: 5.0 accumulates
    /// <c>src/dst</c> in double precision; 4.5.5 overflows its 32-bit product for
    /// <c>x ≥ 24576</c>.
    /// </summary>
    [Fact]
    public void ResizeNearestExact_WideRow_AccumulatesInDouble()
    {
        Mat<int> src = Ramp<int>(40000, 1, 1, i => i);
        Mat<int> dst = Geometry.Resize(src, new Size(30000, 1), interpolation: InterpolationFlags.NearestExact);
        double scale = 40000.0 / 30000, f = scale * 0.5;
        for (int x = 0; x < 30000; x++)
        {
            Assert.Equal(Math.Min((int)Math.Floor(f), 39999), dst.At(0, x));
            f += scale;
        }
    }

    /// <summary>
    /// Golden case <c>perspectiveTransform_collinear</c>: 4.5.5 returns
    /// <c>[0, …, 0, 1]</c>; 5.0 returns the unit singular vector of the smallest singular value
    /// of the homogeneous 8 × 9 system.
    /// </summary>
    [Fact]
    public void GetPerspectiveTransform_Collinear_UsesSvdFallback()
    {
        Point2f[] s = [new(0, 0), new(1, 1), new(2, 2), new(3, 3)];
        Point2f[] d = [new(0, 0), new(1, 0), new(2, 0), new(3, 0)];
        double[] m = Geometry.GetPerspectiveTransform(s, d).ToArray();
        double norm = 0;
        foreach (double v in m)
        {
            norm += v * v;
        }

        Assert.Equal(1.0, Math.Sqrt(norm), 12);
        Assert.NotEqual(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1 }, m);
        for (int i = 0; i < 4; i++)
        {
            double x = s[i].X, y = s[i].Y, u = d[i].X, v = d[i].Y;
            Assert.Equal(0, (m[0] * x) + (m[1] * y) + m[2] - (u * ((m[6] * x) + (m[7] * y) + m[8])), 9);
            Assert.Equal(0, (m[3] * x) + (m[4] * y) + m[5] - (v * ((m[6] * x) + (m[7] * y) + m[8])), 9);
        }
    }

    /// <summary>
    /// 5.0 renormalises a bilinear pixel at the right edge with <c>BORDER_TRANSPARENT</c>:
    /// map (2.5, 1) in a 3 × 3 image has weights {0.5, 0.5, 0, 0}; only the base tap is inside, so
    /// the result is <c>S(1, 2)·0.5 · 1 / 0.5 = S(1, 2)</c>. 4.5.5 leaves the pixel unchanged. A pixel
    /// whose base tap is outside stays unchanged in both versions.
    /// </summary>
    [Fact]
    public void RemapLinearTransparent_EdgePixel_UsesRenormalisedWeights()
    {
        Mat<byte> src = Ramp<byte>(3, 3, 1, i => (byte)(10 + (i * 20)));
        var mx = new Mat<float>([2.5f, 3.25f], 2, 1);
        var my = new Mat<float>([1f, 1f], 2, 1);
        var dst = new Mat<byte>(2, 1);
        dst.Fill(7);
        Geometry.Remap(src, dst, mx, my, InterpolationFlags.Linear, BorderType.Transparent);
        Assert.Equal(src.At(1, 2), dst.At(0, 0));
        Assert.Equal(7, dst.At(0, 1));

        Mat<float> srcF = Ramp<float>(3, 3, 1, i => 1.5f + i);
        var dstF = new Mat<float>(2, 1);
        dstF.Fill(-1);
        Geometry.Remap(srcF, dstF, mx, my, InterpolationFlags.Linear, BorderType.Transparent);
        Assert.Equal(srcF.At(1, 2), dstF.At(0, 0));
        Assert.Equal(-1, dstF.At(0, 1));
    }

    /// <summary>
    /// 5.0 bottom-edge renormalisation with all four taps weighted: map (0.25, 2.75) in a 3 × 3
    /// image keeps the two upper taps. For 8U the weights are the 16-bit table entry
    /// (i = 24, j = 8) and the result is <c>(int)(t0 · 32768f / (w0 + w1))</c> before the fixed-point cast.
    /// </summary>
    [Fact]
    public void RemapLinearTransparent_BottomEdge_MatchesFormula()
    {
        Mat<byte> src = Ramp<byte>(3, 3, 1, i => (byte)(200 - (i * 17)));
        var mx = new Mat<float>([0.25f], 1, 1);
        var my = new Mat<float>([2.75f], 1, 1);
        var dst = new Mat<byte>(1, 1);
        Geometry.Remap(src, dst, mx, my, InterpolationFlags.Linear, BorderType.Transparent);

        short[] w = Geometrikal.OpenCV.Managed.ImgProc.Internal.InterTables.Linear.Fixed.AsSpan(((24 * 32) + 8) * 4, 4).ToArray();
        int total = w[0] + w[1];
        int t0 = (src.At(2, 0) * w[0]) + (src.At(2, 1) * w[1]);
        int renormalised = (int)(t0 * (float)(w[0] + w[1] + w[2] + w[3]) / total);
        Assert.Equal((byte)Math.Clamp((renormalised + (1 << 14)) >> 15, 0, 255), dst.At(0, 0));
    }

    /// <summary>
    /// <c>WARP_RELATIVE_MAP</c> (5.0 only): the destination coordinates are added to the converted
    /// map coordinates. With offsets on the 1/32 grid this equals the absolute map.
    /// </summary>
    [Theory]
    [InlineData(InterpolationFlags.Nearest)]
    [InlineData(InterpolationFlags.Linear)]
    [InlineData(InterpolationFlags.Cubic)]
    [InlineData(InterpolationFlags.Lanczos4)]
    public void Remap_RelativeMap_EqualsAbsoluteMap(InterpolationFlags interpolation)
    {
        Mat<byte> src = Ramp<byte>(19, 15, 3, i => (byte)((i * 37) % 251));
        int w = 23, h = 17;
        var relX = new Mat<float>(w, h);
        var relY = new Mat<float>(w, h);
        var absX = new Mat<float>(w, h);
        var absY = new Mat<float>(w, h);
        float[] offsets = [-0.75f, -0.25f, 0.25f, 0.75f, 1.25f, -2.21875f, 3.40625f];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float ox = offsets[(x + (2 * y)) % offsets.Length], oy = offsets[((3 * x) + y) % offsets.Length];
                relX.At(y, x) = ox;
                relY.At(y, x) = oy;
                absX.At(y, x) = x + ox;
                absY.At(y, x) = y + oy;
            }
        }

        byte[] expected = Geometry.Remap(src, absX, absY, interpolation, BorderType.Reflect101).ToArray();
        Assert.Equal(expected, Geometry.Remap(src, relX, relY, interpolation, BorderType.Reflect101, default, WarpFlags.RelativeMap).ToArray());

        (Mat<short> relXy, Mat<ushort>? relFr) = Geometry.ConvertMapsToFixedPoint(relX, relY, interpolation == InterpolationFlags.Nearest);
        (Mat<short> absXy, Mat<ushort>? absFr) = Geometry.ConvertMapsToFixedPoint(absX, absY, interpolation == InterpolationFlags.Nearest);
        Assert.Equal(
            Geometry.Remap(src, absXy, absFr, interpolation, BorderType.Reflect101).ToArray(),
            Geometry.Remap(src, relXy, relFr, interpolation, BorderType.Reflect101, default, WarpFlags.RelativeMap).ToArray());
    }
}
