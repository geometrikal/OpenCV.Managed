namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

public sealed class GeometryBasicTests
{
    private static Mat<T> Random<T>(int width, int height, int channels, int seed)
        where T : unmanaged
    {
        var m = new Mat<T>(width, height, channels);
        var r = new Random(seed);
        for (int y = 0; y < height; y++)
        {
            Span<T> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                double v = r.Next(0, 256);
                row[i] = typeof(T) == typeof(float) ? (T)(object)(float)(v / 3) : typeof(T) == typeof(byte) ? (T)(object)(byte)v : (T)(object)(ushort)(v * 200);
            }
        }

        return m;
    }

    private static Mat<T> InRoi<T>(Mat<T> m)
        where T : unmanaged
    {
        var parent = new Mat<T>(m.Width + 7, m.Height + 5, m.Channels);
        Mat<T> roi = parent.Roi(3, 2, m.Width, m.Height);
        m.CopyTo(roi);
        return roi;
    }

    private static Mat<double> Affine() => Geometry.GetRotationMatrix2D(new Point2f(10.3f, 7.7f), 23, 0.85);

    // ------------------------------------------------------------------ resize

    [Theory]
    [InlineData(InterpolationFlags.Nearest)]
    [InlineData(InterpolationFlags.Linear)]
    [InlineData(InterpolationFlags.Cubic)]
    [InlineData(InterpolationFlags.Area)]
    [InlineData(InterpolationFlags.Lanczos4)]
    [InlineData(InterpolationFlags.LinearExact)]
    [InlineData(InterpolationFlags.NearestExact)]
    public void Resize_RoiWithStride_MatchesPackedImage(InterpolationFlags interpolation)
    {
        Mat<byte> src = Random<byte>(37, 23, 3, 1);
        foreach (Size size in new[] { new Size(61, 40), new Size(17, 9), new Size(18, 11) })
        {
            byte[] packed = Geometry.Resize(src, size, interpolation: interpolation).ToArray();
            Assert.Equal(packed, Geometry.Resize(InRoi(src), size, interpolation: interpolation).ToArray());

            var parent = new Mat<byte>(size.Width + 4, size.Height + 3, 3);
            Mat<byte> dst = parent.Roi(2, 1, size.Width, size.Height);
            Geometry.Resize(InRoi(src), dst, interpolation);
            Assert.Equal(packed, dst.ToArray());
        }
    }

    [Fact]
    public void Resize_DoesNotModifySource_AndIsDeterministic()
    {
        Mat<float> src = Random<float>(41, 29, 1, 2);
        float[] before = src.ToArray();
        float[] first = Geometry.Resize(src, new Size(19, 50), interpolation: InterpolationFlags.Cubic).ToArray();
        float[] second = Geometry.Resize(src, new Size(19, 50), interpolation: InterpolationFlags.Cubic).ToArray();
        Assert.Equal(before, src.ToArray());
        Assert.Equal(first, second);
    }

    [Fact]
    public void Resize_SameSize_CopiesSource()
    {
        Mat<ushort> src = Random<ushort>(9, 7, 2, 3);
        Assert.Equal(src.ToArray(), Geometry.Resize(src, new Size(9, 7), interpolation: InterpolationFlags.Lanczos4).ToArray());
    }

    [Fact]
    public void Resize_ScaleFactorsAndSize_UseDifferentCoordinates()
    {
        Mat<byte> src = Random<byte>(17, 13, 1, 4);
        Mat<byte> byScale = Geometry.Resize(src, default, 0.37, 0.41);
        Assert.Equal(6, byScale.Width);
        Assert.Equal(5, byScale.Height);
        Mat<byte> bySize = Geometry.Resize(src, new Size(6, 5));
        Assert.NotEqual(byScale.ToArray(), bySize.ToArray());
    }

    [Fact]
    public void Resize_SinglePixel_ReplicatesValue()
    {
        var src = new Mat<byte>([77], 1, 1);
        foreach (InterpolationFlags f in Enum.GetValues<InterpolationFlags>())
        {
            Assert.All(Geometry.Resize(src, new Size(5, 3), interpolation: f).ToArray(), v => Assert.Equal(77, v));
        }
    }

    [Fact]
    public void Resize_InvalidArguments_Throw()
    {
        Mat<byte> src = Random<byte>(8, 8, 1, 5);
        Assert.Throws<ArgumentNullException>(() => Geometry.Resize((Mat<byte>)null!, new Size(4, 4)));
        Assert.Throws<ArgumentException>(() => Geometry.Resize(src, new Size(-1, 4)));
        Assert.Throws<ArgumentException>(() => Geometry.Resize(src, new Size()));
        Assert.Throws<ArgumentException>(() => Geometry.Resize(src, default, 0.01, 0.01));
        Assert.Throws<ArgumentOutOfRangeException>(() => Geometry.Resize(src, new Size(4, 4), interpolation: (InterpolationFlags)7));
        Assert.Throws<NotSupportedException>(() => Geometry.Resize(new Mat<sbyte>(8, 8), new Size(4, 5), interpolation: InterpolationFlags.Linear));
        Assert.Throws<NotSupportedException>(() => Geometry.Resize(new Mat<byte>(9, 9, 5), new Size(4, 5), interpolation: InterpolationFlags.Area));
        Assert.Throws<ArgumentException>(() => Geometry.Resize(src, src, InterpolationFlags.Linear));
        Assert.Throws<ArgumentException>(() => Geometry.Resize(src, new Mat<byte>(4, 4, 3), InterpolationFlags.Linear));
    }

    [Fact]
    public void Resize_LinearExactFloat_EqualsLinear()
    {
        Mat<float> src = Random<float>(13, 11, 2, 6);
        Assert.Equal(
            Geometry.Resize(src, new Size(20, 7), interpolation: InterpolationFlags.Linear).ToArray(),
            Geometry.Resize(src, new Size(20, 7), interpolation: InterpolationFlags.LinearExact).ToArray());
    }

    // ------------------------------------------------------------------ flip, rotate, transpose

    [Fact]
    public void FlipRotateTranspose_RoiWithStride_MatchPackedImage()
    {
        Mat<byte> src = Random<byte>(37, 21, 3, 7);
        Mat<byte> roi = InRoi(src);
        for (int code = -1; code <= 1; code++)
        {
            Assert.Equal(MatrixTransforms.Flip(src, code).ToArray(), MatrixTransforms.Flip(roi, code).ToArray());
        }

        foreach (RotateFlags r in Enum.GetValues<RotateFlags>())
        {
            Assert.Equal(MatrixTransforms.Rotate(src, r).ToArray(), MatrixTransforms.Rotate(roi, r).ToArray());
        }

        Assert.Equal(MatrixTransforms.Transpose(src).ToArray(), MatrixTransforms.Transpose(roi).ToArray());
    }

    [Fact]
    public void Transpose_MovesEveryPixel()
    {
        Mat<byte> src = Random<byte>(45, 37, 1, 8);
        Mat<byte> t = MatrixTransforms.Transpose(src);
        Assert.Equal(37, t.Width);
        Assert.Equal(45, t.Height);
        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                Assert.Equal(src.At(y, x), t.At(x, y));
            }
        }

        Assert.Equal(src.ToArray(), MatrixTransforms.Transpose(t).ToArray());
    }

    [Fact]
    public void Rotate_Compositions_AreConsistent()
    {
        Mat<ushort> src = Random<ushort>(19, 33, 2, 9);
        Mat<ushort> cw = MatrixTransforms.Rotate(src, RotateFlags.Rotate90Clockwise);
        Mat<ushort> ccw = MatrixTransforms.Rotate(src, RotateFlags.Rotate90Counterclockwise);
        Assert.Equal(src.ToArray(), MatrixTransforms.Rotate(cw, RotateFlags.Rotate90Counterclockwise).ToArray());
        Assert.Equal(MatrixTransforms.Rotate(src, RotateFlags.Rotate180).ToArray(), MatrixTransforms.Rotate(cw, RotateFlags.Rotate90Clockwise).ToArray());
        Assert.Equal(MatrixTransforms.Flip(MatrixTransforms.Transpose(src), 0).ToArray(), ccw.ToArray());
        Assert.Equal(src.At(32, 0, 1), cw.At(0, 0, 1));
    }

    [Fact]
    public void FlipAndRotate_InPlace_MatchOutOfPlace()
    {
        Mat<byte> src = Random<byte>(23, 23, 3, 10);
        foreach (int code in new[] { -1, 0, 1 })
        {
            Mat<byte> copy = src.Clone();
            MatrixTransforms.Flip(copy, copy, code);
            Assert.Equal(MatrixTransforms.Flip(src, code).ToArray(), copy.ToArray());
        }

        Mat<byte> square = src.Clone();
        MatrixTransforms.Rotate(square, square, RotateFlags.Rotate90Clockwise);
        Assert.Equal(MatrixTransforms.Rotate(src, RotateFlags.Rotate90Clockwise).ToArray(), square.ToArray());
        Mat<byte> transposed = src.Clone();
        MatrixTransforms.Transpose(transposed, transposed);
        Assert.Equal(MatrixTransforms.Transpose(src).ToArray(), transposed.ToArray());
    }

    [Fact]
    public void Transforms_InvalidArguments_Throw()
    {
        Assert.Throws<NotSupportedException>(() => MatrixTransforms.Transpose(new Mat<double>(3, 3, 5)));
        Assert.Throws<ArgumentOutOfRangeException>(() => MatrixTransforms.Rotate(new Mat<byte>(3, 3), (RotateFlags)3));
        Assert.Throws<ArgumentException>(() => MatrixTransforms.Flip(new Mat<byte>(3, 4), new Mat<byte>(4, 3), 1));
        Assert.Throws<ArgumentException>(() => MatrixTransforms.Transpose(new Mat<byte>(3, 4), new Mat<byte>(3, 4)));
        Assert.Throws<ArgumentNullException>(() => MatrixTransforms.Flip<byte>(null!, 0));
    }

    // ------------------------------------------------------------------ warps and remap

    [Theory]
    [InlineData(InterpolationFlags.Nearest)]
    [InlineData(InterpolationFlags.Linear)]
    [InlineData(InterpolationFlags.Cubic)]
    [InlineData(InterpolationFlags.Lanczos4)]
    public void Warps_RoiWithStride_MatchPackedImage(InterpolationFlags interpolation)
    {
        Mat<byte> src = Random<byte>(23, 17, 3, 11);
        Mat<double> m = Affine();
        var p = new Mat<double>([.. m.ToArray(), 0.002, -0.001, 1.0], 3, 3);
        Assert.Equal(
            Geometry.WarpAffine(src, m, new Size(25, 19), interpolation, borderType: BorderType.Reflect).ToArray(),
            Geometry.WarpAffine(InRoi(src), m, new Size(25, 19), interpolation, borderType: BorderType.Reflect).ToArray());
        Assert.Equal(
            Geometry.WarpPerspective(src, p, new Size(25, 19), interpolation, borderType: BorderType.Wrap).ToArray(),
            Geometry.WarpPerspective(InRoi(src), p, new Size(25, 19), interpolation, borderType: BorderType.Wrap).ToArray());
    }

    [Fact]
    public void WarpAffine_Identity_CopiesSource()
    {
        Mat<byte> src = Random<byte>(21, 13, 3, 12);
        var identity = new Mat<double>([1.0, 0, 0, 0, 1.0, 0], 3, 2);
        Assert.Equal(src.ToArray(), Geometry.WarpAffine(src, identity, default, InterpolationFlags.Linear).ToArray());
        Assert.Equal(src.ToArray(), Geometry.WarpAffine(src, identity, default, InterpolationFlags.Nearest).ToArray());
        Assert.Equal(src.ToArray(), Geometry.WarpAffine(src, identity, default, InterpolationFlags.Cubic, borderType: BorderType.Replicate).ToArray());
    }

    [Fact]
    public void WarpAffine_InverseFlag_EqualsInvertedMatrix()
    {
        Mat<float> src = Random<float>(19, 15, 1, 13);
        Mat<double> m = Affine();
        Mat<float> a = Geometry.WarpAffine(src, m, new Size(20, 16), InterpolationFlags.Linear);
        Mat<float> b = Geometry.WarpAffine(src, Geometry.InvertAffineTransform(m), new Size(20, 16), InterpolationFlags.Linear, WarpFlags.InverseMap);
        Assert.Equal(a.ToArray(), b.ToArray());
    }

    [Fact]
    public void Warp_Transparent_KeepsDestinationOutsideSource()
    {
        Mat<byte> src = Random<byte>(10, 10, 1, 14);
        var shift = new Mat<double>([1.0, 0, 20, 0, 1.0, 0], 3, 2);
        var dst = new Mat<byte>(10, 10);
        dst.Fill(99);
        Geometry.WarpAffine(src, dst, shift, InterpolationFlags.Linear, WarpFlags.None, BorderType.Transparent);
        Assert.All(dst.ToArray(), v => Assert.Equal(99, v));
    }

    [Fact]
    public void Warp_InPlace_MatchesOutOfPlace()
    {
        Mat<ushort> src = Random<ushort>(16, 16, 2, 15);
        Mat<double> m = Affine();
        Mat<ushort> expected = Geometry.WarpAffine(src, m, default, InterpolationFlags.Cubic, borderType: BorderType.Reflect101);
        Mat<ushort> copy = src.Clone();
        Geometry.WarpAffine(copy, copy, m, InterpolationFlags.Cubic, borderType: BorderType.Reflect101);
        Assert.Equal(expected.ToArray(), copy.ToArray());
    }

    [Fact]
    public void Warp_DoesNotModifySource_AndIsDeterministic()
    {
        Mat<byte> src = Random<byte>(30, 20, 4, 16);
        byte[] before = src.ToArray();
        var p = new Mat<double>([0.9, 0.1, 2, -0.05, 1.1, -1, 0.001, 0.002, 1], 3, 3);
        byte[] a = Geometry.WarpPerspective(src, p, default, InterpolationFlags.Lanczos4).ToArray();
        byte[] b = Geometry.WarpPerspective(src, p, default, InterpolationFlags.Lanczos4).ToArray();
        Assert.Equal(before, src.ToArray());
        Assert.Equal(a, b);
    }

    [Fact]
    public void Warps_InvalidArguments_Throw()
    {
        var src = new Mat<byte>(8, 8);
        Mat<double> m = Affine();
        Assert.Throws<ArgumentException>(() => Geometry.WarpAffine(src, new Mat<double>(3, 3)));
        Assert.Throws<ArgumentException>(() => Geometry.WarpPerspective(src, m));
        Assert.Throws<ArgumentOutOfRangeException>(() => Geometry.WarpAffine(src, m, default, InterpolationFlags.LinearExact));
        Assert.Throws<ArgumentOutOfRangeException>(() => Geometry.WarpAffine(src, m, default, InterpolationFlags.Linear, WarpFlags.RelativeMap));
        Assert.Throws<NotSupportedException>(() => Geometry.WarpAffine(new Mat<sbyte>(8, 8), m, default, InterpolationFlags.Linear));
        Assert.Throws<NotSupportedException>(() => Geometry.WarpAffine(new Mat<byte>(8, 8, 5), m, default, InterpolationFlags.Cubic));
        Assert.Throws<ArgumentException>(() => Geometry.WarpAffine(src, new Mat<byte>(8, 8, 2), m));
        Assert.Throws<ArgumentOutOfRangeException>(() => Geometry.WarpAffine(src, m, default, InterpolationFlags.Linear, WarpFlags.None, (BorderType)6));
    }

    [Fact]
    public void Remap_InvalidArguments_Throw()
    {
        var src = new Mat<byte>(8, 8);
        var mx = new Mat<float>(4, 4);
        Assert.Throws<ArgumentException>(() => Geometry.Remap(src, mx, new Mat<float>(5, 4), InterpolationFlags.Linear));
        Assert.Throws<ArgumentException>(() => Geometry.Remap(src, mx, null, InterpolationFlags.Linear));
        Assert.Throws<ArgumentException>(() => Geometry.Remap(src, new Mat<short>(4, 4, 2), null, InterpolationFlags.Linear));
        Assert.Throws<ArgumentException>(() => Geometry.Remap(src, new Mat<short>(4, 4, 1), null, InterpolationFlags.Nearest));
        Assert.Throws<ArgumentException>(() => Geometry.Remap(src, new Mat<byte>(5, 4), mx, mx, InterpolationFlags.Linear));
        Assert.Throws<ArgumentOutOfRangeException>(() => Geometry.Remap(src, mx, mx, InterpolationFlags.Linear, flags: WarpFlags.InverseMap));
        Assert.Throws<ArgumentException>(() => Geometry.Remap(new Mat<byte>(32767, 1), mx, mx, InterpolationFlags.Linear));
    }

    [Fact]
    public void Remap_RoiMaps_MatchPackedMaps()
    {
        Mat<byte> src = Random<byte>(15, 13, 1, 17);
        var mx = new Mat<float>(9, 7);
        var my = new Mat<float>(9, 7);
        var r = new Random(18);
        for (int y = 0; y < 7; y++)
        {
            for (int x = 0; x < 9; x++)
            {
                mx.At(y, x) = (float)((r.NextDouble() * 19) - 2);
                my.At(y, x) = (float)((r.NextDouble() * 17) - 2);
            }
        }

        byte[] expected = Geometry.Remap(src, mx, my, InterpolationFlags.Linear, BorderType.Replicate).ToArray();
        Assert.Equal(expected, Geometry.Remap(InRoi(src), InRoi(mx), InRoi(my), InterpolationFlags.Linear, BorderType.Replicate).ToArray());
        (Mat<short> xy, Mat<ushort>? fr) = Geometry.ConvertMapsToFixedPoint(InRoi(mx), InRoi(my));
        Assert.Equal(expected, Geometry.Remap(src, InRoi(xy), InRoi(fr!), InterpolationFlags.Linear, BorderType.Replicate).ToArray());
    }

    [Fact]
    public void ConvertMaps_RoundTrip_PreservesQuantisedCoordinates()
    {
        var mx = new Mat<float>([0.5f, 1.25f, -3.03125f, 7f], 4, 1);
        var my = new Mat<float>([2f, -0.96875f, 4.5f, 0.0625f], 4, 1);
        (Mat<short> xy, Mat<ushort>? fr) = Geometry.ConvertMapsToFixedPoint(mx, my);
        (Mat<float> bx, Mat<float> by) = Geometry.ConvertMapsToFloat(xy, fr);
        Assert.Equal(mx.ToArray(), bx.ToArray());
        Assert.Equal(my.ToArray(), by.ToArray());
        Mat<float> interleaved = Geometry.ConvertMapsToFloatInterleaved(mx, my);
        (Mat<float> sx, Mat<float> sy) = Geometry.ConvertMapsToFloat(interleaved);
        Assert.Equal(mx.ToArray(), sx.ToArray());
        Assert.Equal(my.ToArray(), sy.ToArray());
        (Mat<short> nn, Mat<ushort>? none) = Geometry.ConvertMapsToFixedPoint(mx, my, nearestInterpolation: true);
        Assert.Null(none);
        Assert.Equal(new short[] { 0, 2, 1, -1, -3, 4, 7, 0 }, nn.ToArray());
    }

    // ------------------------------------------------------------------ matrix helpers

    [Fact]
    public void GetRotationMatrix2D_QuarterTurn_HasExpectedTerms()
    {
        double[] m = Geometry.GetRotationMatrix2D(new Point2f(4, 2), 90, 2).ToArray();
        Assert.Equal(0, m[0], 12);
        Assert.Equal(2, m[1], 12);
        Assert.Equal(-2, m[3], 12);
        Assert.Equal(((1 - m[0]) * 4) - (m[1] * 2), m[2]);
        Assert.Equal((m[1] * 4) + ((1 - m[0]) * 2), m[5]);
    }

    [Fact]
    public void GetAffineAndPerspectiveTransform_MapThePoints()
    {
        Point2f[] s = [new(1, 2), new(40, 5), new(7, 33), new(45, 41)];
        Point2f[] d = [new(3, 1), new(44, 9), new(2, 30), new(50, 47)];
        double[] a = Geometry.GetAffineTransform(s.AsSpan(0, 3), d.AsSpan(0, 3)).ToArray();
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(d[i].X, (a[0] * s[i].X) + (a[1] * s[i].Y) + a[2], 9);
            Assert.Equal(d[i].Y, (a[3] * s[i].X) + (a[4] * s[i].Y) + a[5], 9);
        }

        double[] p = Geometry.GetPerspectiveTransform(s, d).ToArray();
        Assert.Equal(1.0, p[8]);
        for (int i = 0; i < 4; i++)
        {
            double w = (p[6] * s[i].X) + (p[7] * s[i].Y) + p[8];
            Assert.Equal(d[i].X, ((p[0] * s[i].X) + (p[1] * s[i].Y) + p[2]) / w, 4);
            Assert.Equal(d[i].Y, ((p[3] * s[i].X) + (p[4] * s[i].Y) + p[5]) / w, 4);
        }

        Assert.Throws<ArgumentException>(() => Geometry.GetAffineTransform(s, d));
        Assert.Throws<ArgumentException>(() => Geometry.GetPerspectiveTransform(s.AsSpan(0, 3), d.AsSpan(0, 3)));
    }

    [Fact]
    public void InvertAffineTransform_ComposesToIdentity()
    {
        Mat<double> m = Affine();
        double[] a = m.ToArray(), i = Geometry.InvertAffineTransform(m).ToArray();
        Assert.Equal(1, (i[0] * a[0]) + (i[1] * a[3]), 12);
        Assert.Equal(0, (i[0] * a[1]) + (i[1] * a[4]), 12);
        Assert.Equal(0, (i[0] * a[2]) + (i[1] * a[5]) + i[2], 12);
        float[] f = Geometry.InvertAffineTransform(new Mat<float>(a.Select(v => (float)v).ToArray(), 3, 2)).ToArray();
        Assert.Equal(i[0], f[0], 5);
        Assert.Throws<ArgumentException>(() => Geometry.InvertAffineTransform(new Mat<double>(3, 3)));
    }

    // ------------------------------------------------------------------ interpolation tables

    [Fact]
    public void InterTables_FixedEntries_SumToScale()
    {
        foreach (InterTables t in new[] { InterTables.Linear, InterTables.Cubic, InterTables.Lanczos4 })
        {
            int kk = t.Ksize * t.Ksize;
            for (int e = 0; e < InterTables.Size2; e++)
            {
                int sum = 0;
                for (int i = 0; i < kk; i++)
                {
                    sum += t.Fixed[(e * kk) + i];
                }

                Assert.Equal(InterTables.CoefScale, sum);
            }
        }

        Assert.Equal(new short[] { 32767, 0, 0, 1 }, InterTables.Linear.Fixed.AsSpan(0, 4).ToArray());
    }

    [Theory]
    [InlineData(2, 1)]
    [InlineData(4, 657)]
    [InlineData(8, 856)]
    public void InterTables_CorrectedEntryCount_MatchesOpenCv(int ksize, int corrected)
    {
        InterTables t = ksize == 2 ? InterTables.Linear : ksize == 4 ? InterTables.Cubic : InterTables.Lanczos4;
        int kk = ksize * ksize, count = 0;
        for (int e = 0; e < InterTables.Size2; e++)
        {
            int sum = 0;
            for (int i = 0; i < kk; i++)
            {
                sum += (short)Math.Clamp((int)Math.Round(t.Float[(e * kk) + i] * InterTables.CoefScale, MidpointRounding.ToEven), short.MinValue, short.MaxValue);
            }

            count += sum != InterTables.CoefScale ? 1 : 0;
        }

        Assert.Equal(corrected, count);
    }

    [Fact]
    public void BorderTransparent_IsRejectedByCopyMakeBorder()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Borders.CopyMakeBorder(new Mat<byte>(3, 3), 1, 1, 1, 1, BorderType.Transparent));
        Assert.Throws<ArgumentOutOfRangeException>(() => Borders.BorderInterpolate(-1, 3, BorderType.Transparent));
    }
}
