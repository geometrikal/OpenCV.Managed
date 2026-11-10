namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching;
using Geometrikal.OpenCV.Managed.Tests.Core;

[Collection(nameof(CvParallelCollection))]
public sealed class StitcherBasicTests
{
    private const int SceneWidth = 640;
    private const int SceneHeight = 440;
    private const int TileWidth = 380;
    private const int TileHeight = 270;

    [Fact]
    public void Scans_FourTiles_RebuildsTheScene()
    {
        Mat<byte> scene = Scene(SceneWidth, SceneHeight);
        var origins = new[] { new Point(0, 0), new Point(260, 0), new Point(0, 170), new Point(260, 170) };
        var tiles = new List<Mat<byte>>();
        foreach (Point o in origins)
        {
            tiles.Add(scene.Roi(o.X, o.Y, TileWidth, TileHeight).Clone());
        }

        Stitcher stitcher = Stitcher.Create(StitcherMode.Scans);
        StitcherStatus status = stitcher.Stitch(tiles, out Mat<byte>? pano);

        Assert.Equal(StitcherStatus.Ok, status);
        Assert.NotNull(pano);
        Assert.True(
            stitcher.Component.Count == 4 && Math.Abs(pano!.Width - SceneWidth) <= 6 && Math.Abs(pano.Height - SceneHeight) <= 6,
            $"component {stitcher.Component.Count} size {pano!.Width}x{pano.Height} R {string.Join(" | ", System.Linq.Enumerable.Select(stitcher.Cameras, c => c.R.ToString()))}");
        Assert.InRange(pano!.Width, SceneWidth - 6, SceneWidth + 6);
        Assert.InRange(pano.Height, SceneHeight - 6, SceneHeight + 6);
        Assert.Equal(3, pano.Channels);

        // The panorama is the scene up to an integer shift; the overlap regions must match it.
        (double best, int dx, int dy) = BestShift(pano, stitcher.ResultMask!, scene);
        Assert.True(best < 8.0, $"mean absolute difference {best:F2} at shift ({dx}, {dy})");

        double overlap = MeanAbsDiff(pano, stitcher.ResultMask!, scene, dx, dy, new Rect(270, 180, 360 - 270, 260 - 180));
        Assert.True(overlap < 8.0, $"overlap mean absolute difference {overlap:F2}");
    }

    [Fact]
    public void Scans_Cameras_AreTranslations()
    {
        Mat<byte> scene = Scene(SceneWidth, SceneHeight);
        var tiles = new List<Mat<byte>>
        {
            scene.Roi(0, 0, TileWidth, TileHeight).Clone(),
            scene.Roi(200, 40, TileWidth, TileHeight).Clone(),
        };

        Stitcher stitcher = Stitcher.Create(StitcherMode.Scans);
        Assert.Equal(StitcherStatus.Ok, stitcher.EstimateTransform(tiles));
        IReadOnlyList<CameraParams> cams = stitcher.Cameras;
        Assert.Equal(2, cams.Count);

        // R1 maps tile 0 coordinates to tile 1 coordinates: about a translation of (-200, -40).
        // The tile centre must map to within one pixel.
        Mat<double> r0 = cams[0].R, r1 = cams[1].R;
        Assert.Equal(1.0, r0.At(0, 0));
        Assert.Equal(0.0, r0.At(0, 2));
        double cx = TileWidth / 2.0, cy = TileHeight / 2.0;
        double mx = (r1.At(0, 0) * cx) + (r1.At(0, 1) * cy) + r1.At(0, 2);
        double my = (r1.At(1, 0) * cx) + (r1.At(1, 1) * cy) + r1.At(1, 2);
        Assert.True(Math.Abs(mx - (cx - 200)) < 1.0 && Math.Abs(my - (cy - 40)) < 1.0, $"centre maps to ({mx}, {my})");
        Assert.InRange(r1.At(0, 0), 0.99, 1.01);
        Assert.InRange(r1.At(1, 0), -0.01, 0.01);
    }

    [Fact]
    public void Panorama_RotatingCamera_EstimatesFocalAndStitches()
    {
        // Views of a plane at depth f from a camera that turns about the y axis: H = K·R·K⁻¹.
        Mat<byte> scene = Scene(900, 400);
        const double F = 400;
        var tiles = new List<Mat<byte>>();
        foreach (double deg in new[] { -12.0, 0.0, 12.0 })
        {
            double a = deg * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
            double cx = 450, cy = 200, ox = 160, oy = 120;
            // Maps scene pixels to tile pixels: K_tile · R · K_scene⁻¹.
            double[] kTile = [F, 0, ox, 0, F, oy, 0, 0, 1];
            double[] r = [c, 0, s, 0, 1, 0, -s, 0, c];
            double[] kSceneInv = [1 / F, 0, -cx / F, 0, 1 / F, -cy / F, 0, 0, 1];
            var h = new Mat<double>(Mul3(Mul3(kTile, r), kSceneInv), 3, 3);
            tiles.Add(Geometry.WarpPerspective(scene, h, new Size(320, 240)));
        }

        Stitcher stitcher = Stitcher.Create(StitcherMode.Panorama);
        StitcherStatus status = stitcher.Stitch(tiles, out Mat<byte>? pano);
        Assert.Equal(StitcherStatus.Ok, status);
        Assert.NotNull(pano);
        Assert.Equal(3, stitcher.Component.Count);
        foreach (CameraParams cam in stitcher.Cameras)
        {
            Assert.InRange(cam.Focal, F * 0.9, F * 1.1);
        }

        Assert.InRange(pano!.Width, 400, 520);
        Assert.InRange(pano.Height, 220, 280);
    }

    private static double[] Mul3(double[] a, double[] b)
    {
        var r = new double[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                r[(i * 3) + j] = (a[i * 3] * b[j]) + (a[(i * 3) + 1] * b[3 + j]) + (a[(i * 3) + 2] * b[6 + j]);
            }
        }

        return r;
    }

    [Fact]
    public void Stitch_OneImage_NeedsMoreImages()
    {
        Mat<byte> scene = Scene(200, 150);
        Stitcher stitcher = Stitcher.Create(StitcherMode.Scans);
        Assert.Equal(StitcherStatus.ErrNeedMoreImgs, stitcher.Stitch(new[] { scene }, out Mat<byte>? pano));
        Assert.Null(pano);
    }

    [Fact]
    public void Stitch_UnrelatedImages_NeedsMoreImages()
    {
        Mat<byte> a = Scene(300, 220, seed: 1);
        Mat<byte> b = Scene(300, 220, seed: 2);
        Stitcher stitcher = Stitcher.Create(StitcherMode.Scans);
        Assert.Equal(StitcherStatus.ErrNeedMoreImgs, stitcher.Stitch(new[] { a, b }, out _));
    }

    [Fact]
    public void Create_Panorama_UsesOpenCvComponents()
    {
        Stitcher s = Stitcher.Create(StitcherMode.Panorama);
        Assert.IsType<HomographyBasedEstimator>(s.Estimator);
        Assert.IsType<BestOf2NearestMatcher>(s.FeaturesMatcher);
        Assert.IsType<BundleAdjusterRay>(s.BundleAdjuster);
        Assert.IsType<SphericalWarperCreator>(s.Warper);
        Assert.IsType<BlocksGainCompensator>(s.ExposureCompensator);
        Assert.IsType<GraphCutSeamFinder>(s.SeamFinder);
        Assert.IsType<MultiBandBlender>(s.Blender);
        Assert.True(s.WaveCorrection);
        Assert.Equal(0.6, s.RegistrationResol);
        Assert.Equal(0.1, s.SeamEstimationResol);
        Assert.Equal(Stitcher.OrigResol, s.CompositingResol);
    }

    internal static Mat<byte> Scene(int width, int height, int seed = 7)
    {
        var rng = new Random(seed);
        var gray = new Mat<byte>(width, height);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = gray.GetRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                row[x] = (byte)rng.Next(256);
            }
        }

        gray = Filters.GaussianBlur(gray, new Size(5, 5), 1.5);
        for (int i = 0; i < 120; i++)
        {
            var c = new Point(rng.Next(width), rng.Next(height));
            Drawing.Circle(gray, c, rng.Next(4, 22), new Scalar(rng.Next(256)), rng.Next(2) == 0 ? -1 : 2);
        }

        for (int i = 0; i < 60; i++)
        {
            var p = new Point(rng.Next(width), rng.Next(height));
            Drawing.Rectangle(gray, p, new Point(p.X + rng.Next(5, 40), p.Y + rng.Next(5, 40)), new Scalar(rng.Next(256)), rng.Next(2) == 0 ? -1 : 2);
        }

        var color = new Mat<byte>(width, height, 3);
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> g = gray.GetReadOnlyRowSpan(y);
            Span<byte> d = color.GetRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                d[3 * x] = g[x];
                d[(3 * x) + 1] = (byte)(255 - g[x]);
                d[(3 * x) + 2] = (byte)((g[x] + (x / 3)) & 255);
            }
        }

        return color;
    }

    private static (double Best, int Dx, int Dy) BestShift(Mat<byte> pano, Mat<byte> mask, Mat<byte> scene)
    {
        double best = double.MaxValue;
        int bx = 0, by = 0;
        for (int dy = -6; dy <= 6; dy++)
        {
            for (int dx = -6; dx <= 6; dx++)
            {
                double d = MeanAbsDiff(pano, mask, scene, dx, dy, new Rect(0, 0, scene.Width, scene.Height));
                if (d < best)
                {
                    best = d;
                    bx = dx;
                    by = dy;
                }
            }
        }

        return (best, bx, by);
    }

    // Compares scene pixel (x, y) in the region with panorama pixel (x + dx, y + dy).
    private static double MeanAbsDiff(Mat<byte> pano, Mat<byte> mask, Mat<byte> scene, int dx, int dy, Rect region)
    {
        long sum = 0, count = 0;
        for (int y = region.Y + 4; y < region.Y + region.Height - 4; y++)
        {
            int py = y + dy;
            if (py < 0 || py >= pano.Height)
            {
                continue;
            }

            for (int x = region.X + 4; x < region.X + region.Width - 4; x++)
            {
                int px = x + dx;
                if (px < 0 || px >= pano.Width || mask.At(py, px) == 0)
                {
                    continue;
                }

                for (int c = 0; c < 3; c++)
                {
                    sum += Math.Abs(pano.At(py, px, c) - scene.At(y, x, c));
                }

                count += 3;
            }
        }

        return count == 0 ? double.MaxValue : (double)sum / count;
    }
}
