namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>
/// Analytic, edge-case, ROI, validation and thread-count tests for <see cref="ShapeDescriptors"/>.
/// Vector paths are compared with scalar references of the OpenCV code at widths that are not a
/// multiple of the vector size.
/// </summary>
[Collection(nameof(CvParallelCollection))]
public sealed class ShapeDescriptorsBasicTests
{
    private static readonly Point[] Square = [new(0, 0), new(10, 0), new(10, 10), new(0, 10)];

    [Fact]
    public void EmptyPointSet_ReturnsZeroResults()
    {
        Assert.Empty(ShapeDescriptors.ApproxPolyDP(ReadOnlySpan<Point>.Empty, 1, true));
        Assert.Empty(ShapeDescriptors.ApproxPolyDP(ReadOnlySpan<Point2f>.Empty, 1, false));
        Assert.Equal(0, ShapeDescriptors.ArcLength(ReadOnlySpan<Point>.Empty, true));
        Assert.Equal(0, ShapeDescriptors.ContourArea(ReadOnlySpan<Point2f>.Empty));
        Assert.Equal(default, ShapeDescriptors.BoundingRect(ReadOnlySpan<Point>.Empty));
        Assert.Equal(default, ShapeDescriptors.BoundingRect(ReadOnlySpan<Point2f>.Empty));
        Assert.Equal(default, ShapeDescriptors.Moments(ReadOnlySpan<Point>.Empty));
    }

    [Fact]
    public void Square_GivesAnalyticDescriptors()
    {
        Assert.Equal(40, ShapeDescriptors.ArcLength(Square, true));
        Assert.Equal(30, ShapeDescriptors.ArcLength(Square, false));
        Assert.Equal(100, ShapeDescriptors.ContourArea(Square, true));
        Assert.Equal(-100, ShapeDescriptors.ContourArea(Square.Reverse().ToArray(), true));
        Assert.Equal(new Rect(0, 0, 11, 11), ShapeDescriptors.BoundingRect(Square));

        Moments m = ShapeDescriptors.Moments(Square);
        Assert.Equal(100, m.M00);
        Assert.Equal(500, m.M10, 9);
        Assert.Equal(500, m.M01, 9);
        Assert.Equal(0, m.Mu11, 9);
        Assert.Equal(1.0 / 12, m.Nu20, 12);
        Assert.Equal(m, ShapeDescriptors.Moments(Square.Reverse().ToArray()));
    }

    [Fact]
    public void ApproxPolyDP_SquareWithMidpoints_KeepsCorners()
    {
        Point[] curve = [new(0, 0), new(5, 0), new(10, 0), new(10, 5), new(10, 10), new(5, 10), new(0, 10), new(0, 5)];
        Point[] r = ShapeDescriptors.ApproxPolyDP(curve, 1, true);
        Assert.Equal(4, r.Length);
        Assert.All(r, p => Assert.True(p.X % 10 == 0 && p.Y % 10 == 0));
    }

    [Fact]
    public void ApproxPolyDP_DoesNotModifySource()
    {
        Point2f[] curve = Enumerable.Range(0, 50).Select(i => new Point2f(MathF.Cos(i * 0.13f) * 20, MathF.Sin(i * 0.13f) * 9)).ToArray();
        Point2f[] copy = curve.ToArray();
        Point2f[] first = ShapeDescriptors.ApproxPolyDP(curve, 0.7, true);
        Assert.Equal(copy, curve);
        Assert.Equal(first, ShapeDescriptors.ApproxPolyDP(curve, 0.7, true));
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(1e30)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ApproxPolyDP_InvalidEpsilon_Throws(double eps)
        => Assert.Throws<ArgumentOutOfRangeException>(() => ShapeDescriptors.ApproxPolyDP(Square, eps, true));

    [Fact]
    public void InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ShapeDescriptors.MatchShapes(Square, Square, (ShapeMatchModes)4));
        Assert.Throws<ArgumentNullException>(() => ShapeDescriptors.BoundingRect((Mat<byte>)null!));
        Assert.Throws<ArgumentException>(() => ShapeDescriptors.BoundingRect(new Mat<byte>(4, 4, 2)));
        Assert.Throws<ArgumentException>(() => ShapeDescriptors.Moments(new Mat<float>(4, 4, 3)));
        Assert.Throws<NotSupportedException>(() => ShapeDescriptors.Moments(new Mat<int>(4, 4)));
        Assert.Throws<ArgumentException>(() => ShapeDescriptors.HuMoments(default, new double[6]));
    }

    [Fact]
    public void MatchShapes_SameContour_ReturnsZero()
    {
        foreach (ShapeMatchModes mode in Enum.GetValues<ShapeMatchModes>())
        {
            Assert.Equal(0, ShapeDescriptors.MatchShapes(Square, Square, mode));
        }

        Point[] line = [new(0, 0), new(4, 4)];
        Assert.Equal(double.MaxValue, ShapeDescriptors.MatchShapes(line, Square, ShapeMatchModes.ContoursMatchI2));
        Assert.Equal(0, ShapeDescriptors.MatchShapes(line, line, ShapeMatchModes.ContoursMatchI2));
    }

    [Fact]
    public void ImageMoments_SinglePixel_GivesPixelPosition()
    {
        var img = new Mat<byte>(40, 35);
        img.At(20, 33) = 7;
        Moments m = ShapeDescriptors.Moments(img);
        Assert.Equal(7, m.M00);
        Assert.Equal(7 * 33, m.M10);
        Assert.Equal(7 * 20, m.M01);
        Assert.Equal(7 * 33 * 33 * 33, m.M30);
        Moments b = ShapeDescriptors.Moments(img, binaryImage: true);
        Assert.Equal(1, b.M00);
        Assert.Equal(default, ShapeDescriptors.Moments(new Mat<float>(9, 9)));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(31, 7)]
    [InlineData(33, 40)]
    [InlineData(47, 65)]
    [InlineData(70, 3)]
    [InlineData(129, 34)]
    public void ImageMoments_Byte_MatchesScalarReference(int width, int height)
    {
        var rng = new Random(width * 1000 + height);
        Mat<byte> img = RandomMat<byte>(rng, width, height, r => (byte)r.Next(256));
        for (int k = 0; k < width * height / 3; k++)
        {
            img.At(rng.Next(height), rng.Next(width)) = 0;
        }

        Assert.Equal(ReferenceMoments(img, false), ShapeDescriptors.Moments(img));
        Assert.Equal(ReferenceMoments(img, true), ShapeDescriptors.Moments(img, true));
    }

    [Fact]
    public void ImageMoments_OtherTypes_MatchScalarReference()
    {
        var rng = new Random(5);
        Mat<ushort> u16 = RandomMat<ushort>(rng, 45, 37, r => (ushort)r.Next(65536));
        Mat<short> s16 = RandomMat<short>(rng, 45, 37, r => (short)r.Next(-32768, 32768));
        Mat<float> f32 = RandomMat<float>(rng, 45, 37, r => r.Next(4) == 0 ? 0f : (float)(r.NextDouble() * 100 - 20));
        Mat<double> f64 = RandomMat<double>(rng, 45, 37, r => r.NextDouble());
        foreach (bool binary in new[] { false, true })
        {
            Assert.Equal(ReferenceMoments(u16, binary), ShapeDescriptors.Moments(u16, binary));
            Assert.Equal(ReferenceMoments(s16, binary), ShapeDescriptors.Moments(s16, binary));
            Assert.Equal(ReferenceMoments(f32, binary), ShapeDescriptors.Moments(f32, binary));
            Assert.Equal(ReferenceMoments(f64, binary), ShapeDescriptors.Moments(f64, binary));
        }
    }

    [Fact]
    public void ImageMoments_RoiWithStride_MatchesPackedImage()
    {
        var rng = new Random(11);
        Mat<byte> parent = RandomMat<byte>(rng, 90, 70, r => (byte)r.Next(256));
        Mat<byte> roi = parent.Roi(7, 5, 61, 43);
        Assert.Equal(ShapeDescriptors.Moments(roi.Clone()), ShapeDescriptors.Moments(roi));
        Assert.Equal(ShapeDescriptors.Moments(roi.Clone(), true), ShapeDescriptors.Moments(roi, true));

        Mat<float> fparent = RandomMat<float>(rng, 90, 70, r => (float)r.NextDouble());
        Mat<float> froi = fparent.Roi(3, 9, 50, 33);
        Assert.Equal(ShapeDescriptors.Moments(froi.Clone()), ShapeDescriptors.Moments(froi));
    }

    [Fact]
    public void ImageMoments_DoesNotModifySource()
    {
        var rng = new Random(3);
        Mat<float> img = RandomMat<float>(rng, 40, 40, r => (float)r.NextDouble());
        float[] before = img.ToArray();
        ShapeDescriptors.Moments(img, true);
        Assert.Equal(before, img.ToArray());
    }

    [Fact]
    public void LargeImage_Output_DoesNotDependOnThreadCount()
    {
        var rng = new Random(17);
        Mat<byte> u8 = RandomMat<byte>(rng, 1531, 1207, r => (byte)r.Next(256));
        Mat<float> f32 = RandomMat<float>(rng, 1531, 1207, r => (float)r.NextDouble());
        Mat<byte> mask = new(3001, 1999);
        mask.At(1500, 2999) = 1;
        mask.At(1998, 7) = 1;
        mask.At(3, 1000) = 1;

        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            Moments a8 = ShapeDescriptors.Moments(u8), af = ShapeDescriptors.Moments(f32), ab = ShapeDescriptors.Moments(f32, true);
            Rect ar = ShapeDescriptors.BoundingRect(mask);
            CvParallel.MaxDegreeOfParallelism = 8;
            Assert.Equal(a8, ShapeDescriptors.Moments(u8));
            Assert.Equal(af, ShapeDescriptors.Moments(f32));
            Assert.Equal(ab, ShapeDescriptors.Moments(f32, true));
            Assert.Equal(ar, ShapeDescriptors.BoundingRect(mask));
            Assert.Equal(new Rect(7, 3, 2993, 1996), ar);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(17, 5)]
    [InlineData(63, 9)]
    [InlineData(64, 9)]
    [InlineData(65, 30)]
    [InlineData(200, 50)]
    public void MaskBoundingRect_MatchesNaiveScan(int width, int height)
    {
        var rng = new Random(width + height);
        for (int trial = 0; trial < 20; trial++)
        {
            var mask = new Mat<byte>(width, height);
            int n = rng.Next(4);
            for (int k = 0; k < n; k++)
            {
                mask.At(rng.Next(height), rng.Next(width)) = (byte)rng.Next(1, 256);
            }

            Assert.Equal(NaiveMaskRect(mask), ShapeDescriptors.BoundingRect(mask));
        }
    }

    [Fact]
    public void MaskBoundingRect_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<byte>(100, 60);
        parent.At(10, 4) = 1;
        parent.At(30, 70) = 1;
        parent.At(50, 95) = 1;
        Mat<byte> roi = parent.Roi(3, 8, 80, 40);
        Assert.Equal(new Rect(1, 2, 67, 21), ShapeDescriptors.BoundingRect(roi));
        Assert.Equal(ShapeDescriptors.BoundingRect(roi.Clone()), ShapeDescriptors.BoundingRect(roi));
    }

    [Fact]
    public void PointSetBoundingRect_MatchesNaiveRange()
    {
        var rng = new Random(23);
        for (int n = 1; n <= 40; n++)
        {
            Point[] p = Enumerable.Range(0, n).Select(_ => new Point(rng.Next(-1000, 1000), rng.Next(-1000, 1000))).ToArray();
            int x0 = p.Min(q => q.X), x1 = p.Max(q => q.X), y0 = p.Min(q => q.Y), y1 = p.Max(q => q.Y);
            Assert.Equal(new Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1), ShapeDescriptors.BoundingRect(p));

            Point2f[] f = Enumerable.Range(0, n).Select(_ => new Point2f((float)(rng.NextDouble() * 50 - 25), (float)(rng.NextDouble() * 50 - 25))).ToArray();
            int fx0 = (int)Math.Floor(f.Min(q => q.X)), fx1 = (int)Math.Floor(f.Max(q => q.X));
            int fy0 = (int)Math.Floor(f.Min(q => q.Y)), fy1 = (int)Math.Floor(f.Max(q => q.Y));
            Assert.Equal(new Rect(fx0, fy0, fx1 - fx0 + 1, fy1 - fy0 + 1), ShapeDescriptors.BoundingRect(f));
        }
    }

    [Fact]
    public void HuMoments_SpanAndArrayOverloads_Agree()
    {
        Moments m = ShapeDescriptors.Moments(new Point[] { new(0, 0), new(13, 2), new(9, 11), new(1, 7) });
        Span<double> hu = stackalloc double[7];
        ShapeDescriptors.HuMoments(m, hu);
        Assert.Equal(ShapeDescriptors.HuMoments(m), hu.ToArray());
    }

    private static Mat<T> RandomMat<T>(Random rng, int width, int height, Func<Random, T> next)
        where T : unmanaged
    {
        var m = new Mat<T>(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                m.At(y, x) = next(rng);
            }
        }

        return m;
    }

    private static Rect NaiveMaskRect(Mat<byte> mask)
    {
        int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
        for (int y = 0; y < mask.Height; y++)
        {
            for (int x = 0; x < mask.Width; x++)
            {
                if (mask.At(y, x) != 0)
                {
                    x0 = Math.Min(x0, x);
                    x1 = Math.Max(x1, x);
                    y0 = Math.Min(y0, y);
                    y1 = Math.Max(y1, y);
                }
            }
        }

        return x1 < 0 ? default : new Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    }

    /// <summary>
    /// A direct transcription of the OpenCV scalar <c>momentsInTile</c> and tile accumulation, with
    /// all per-tile sums in <c>double</c> (exact for integer pixels).
    /// </summary>
    private static Moments ReferenceMoments<T>(Mat<T> img, bool binary)
        where T : unmanaged
    {
        double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m11 = 0, m02 = 0, m30 = 0, m21 = 0, m12 = 0, m03 = 0;
        var mom = new double[10];
        for (int y = 0; y < img.Height; y += 32)
        {
            for (int x = 0; x < img.Width; x += 32)
            {
                Array.Clear(mom);
                int th = Math.Min(32, img.Height - y), tw = Math.Min(32, img.Width - x);
                for (int ty = 0; ty < th; ty++)
                {
                    double x0 = 0, x1 = 0, x2 = 0, x3 = 0;
                    for (int tx = 0; tx < tw; tx++)
                    {
                        double p = Convert.ToDouble(img.At(y + ty, x + tx));
                        if (binary)
                        {
                            p = p != 0 ? 255 : 0;
                        }

                        double xp = tx * p;
                        x0 += p;
                        x1 += xp;
                        double xxp = xp * tx;
                        x2 += xxp;
                        x3 += xxp * tx;
                    }

                    double py = ty * x0, sy = ty * ty;
                    mom[9] += py * sy;
                    mom[8] += x1 * sy;
                    mom[7] += x2 * ty;
                    mom[6] += x3;
                    mom[5] += x0 * sy;
                    mom[4] += x1 * ty;
                    mom[3] += x2;
                    mom[2] += py;
                    mom[1] += x1;
                    mom[0] += x0;
                }

                if (binary)
                {
                    for (int k = 0; k < 10; k++)
                    {
                        mom[k] *= 1.0 / 255;
                    }
                }

                double xm = x * mom[0], ym = y * mom[0];
                m00 += mom[0];
                m10 += mom[1] + xm;
                m01 += mom[2] + ym;
                m20 += mom[3] + (x * ((mom[1] * 2) + xm));
                m11 += mom[4] + (x * (mom[2] + ym)) + (y * mom[1]);
                m02 += mom[5] + (y * ((mom[2] * 2) + ym));
                m30 += mom[6] + (x * ((3.0 * mom[3]) + (x * ((3.0 * mom[1]) + xm))));
                m21 += mom[7] + (x * ((2 * (mom[4] + (y * mom[1]))) + (x * (mom[2] + ym)))) + (y * mom[3]);
                m12 += mom[8] + (y * ((2 * (mom[4] + (x * mom[2]))) + (y * (mom[1] + xm)))) + (x * mom[5]);
                m03 += mom[9] + (y * ((3.0 * mom[5]) + (y * ((3.0 * mom[2]) + ym))));
            }
        }

        return new Moments(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    }
}
