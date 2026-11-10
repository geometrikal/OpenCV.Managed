namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

public sealed class SeamFinderBasicTests
{
    private static readonly Point[] PairCorners = [new(0, 0), new(30, 4)];

    private static Mat<float>[] PairImages()
    {
        var a = new Mat<float>(50, 40, 3);
        var b = new Mat<float>(50, 40, 3);
        for (int y = 0; y < 40; y++)
        {
            for (int x = 0; x < 50; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    a.At(y, x, c) = ((x * 7) + (y * 3) + (c * 40)) % 256;
                    b.At(y, x, c) = ((x * 5) + (y * 11) + (c * 30)) % 256;
                }
            }
        }

        return [a, b];
    }

    private static Mat<byte>[] FullMasks(int count = 2)
    {
        var masks = new Mat<byte>[count];
        for (int i = 0; i < count; i++)
        {
            masks[i] = new Mat<byte>(50, 40);
            masks[i].Fill(255);
        }

        return masks;
    }

    public static TheoryData<string> Finders => new() { "voronoi", "dp", "dpgrad", "gc", "gcgrad" };

    private static SeamFinder Create(string name) => name switch
    {
        "voronoi" => new VoronoiSeamFinder(),
        "dp" => new DpSeamFinder(),
        "dpgrad" => new DpSeamFinder(DpSeamCostFunction.ColorGrad),
        "gc" => new GraphCutSeamFinder(GraphCutCostType.Color),
        _ => new GraphCutSeamFinder(),
    };

    [Fact]
    public void CreateDefault_ReturnsTheOpenCvTypes()
    {
        Assert.IsType<NoSeamFinder>(SeamFinder.CreateDefault(SeamFinderType.No));
        Assert.IsType<VoronoiSeamFinder>(SeamFinder.CreateDefault(SeamFinderType.VoronoiSeam));
        var dp = Assert.IsType<DpSeamFinder>(SeamFinder.CreateDefault(SeamFinderType.DpSeam));
        Assert.Equal(DpSeamCostFunction.Color, dp.CostFunction);
        Assert.Throws<ArgumentOutOfRangeException>(() => SeamFinder.CreateDefault((SeamFinderType)7));
    }

    [Fact]
    public void Defaults_MatchOpenCv()
    {
        var gc = new GraphCutSeamFinder();
        Assert.Equal((GraphCutCostType.ColorGrad, 10000f, 1000f), (gc.CostType, gc.TerminalCost, gc.BadRegionPenalty));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GraphCutSeamFinder((GraphCutCostType)5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DpSeamFinder((DpSeamCostFunction)5));
    }

    [Fact]
    public void NoSeamFinder_LeavesMasksUnchanged()
    {
        Mat<byte>[] masks = FullMasks();
        new NoSeamFinder().Find(PairImages(), PairCorners, masks);
        Assert.All(masks, m => Assert.All(m.ToArray(), v => Assert.Equal(255, v)));
    }

    [Theory]
    [MemberData(nameof(Finders))]
    public void Find_EmptyInput_DoesNothing(string name)
        => Create(name).Find(Array.Empty<Mat<float>>(), Array.Empty<Point>(), Array.Empty<Mat<byte>>());

    [Theory]
    [MemberData(nameof(Finders))]
    public void Find_DisjointImages_LeavesMasksUnchanged(string name)
    {
        Mat<byte>[] masks = FullMasks();
        Create(name).Find(PairImages(), [new Point(0, 0), new Point(50, 0)], masks);
        Assert.All(masks, m => Assert.All(m.ToArray(), v => Assert.Equal(255, v)));
    }

    [Theory]
    [MemberData(nameof(Finders))]
    public void Find_InvalidArguments_Throw(string name)
    {
        SeamFinder f = Create(name);
        Mat<float>[] images = PairImages();
        Assert.Throws<ArgumentException>(() => f.Find(images, [new Point(0, 0)], FullMasks()));
        Assert.Throws<ArgumentException>(() => f.Find(images, PairCorners, [new Mat<byte>(50, 40), new Mat<byte>(49, 40)]));
        Assert.Throws<ArgumentException>(() => f.Find(images, PairCorners, [new Mat<byte>(50, 40), new Mat<byte>(50, 40, 3)]));
    }

    [Fact]
    public void Find_UnsupportedTypes_Throw()
    {
        var byteImages = PairImages().Select(ToByte).ToArray();
        Assert.Throws<NotSupportedException>(() => new GraphCutSeamFinder().Find(byteImages, PairCorners, FullMasks()));
        var gray = new[] { new Mat<float>(50, 40), new Mat<float>(50, 40) };
        Assert.Throws<NotSupportedException>(() => new GraphCutSeamFinder().Find(gray, PairCorners, FullMasks()));
        Assert.Throws<NotSupportedException>(() => new DpSeamFinder().Find(gray, PairCorners, FullMasks()));
        var shorts = new[] { new Mat<short>(50, 40, 3), new Mat<short>(50, 40, 3) };
        Assert.Throws<NotSupportedException>(() => new DpSeamFinder().Find(shorts, PairCorners, FullMasks()));

        // Voronoi does not read the pixels.
        new VoronoiSeamFinder().Find(shorts, PairCorners, FullMasks());
    }

    [Theory]
    [InlineData("voronoi")]
    [InlineData("gc")]
    [InlineData("gcgrad")]
    public void Find_FullMasks_AssignsEachOverlapPixelToOneImage(string name)
    {
        Mat<byte>[] masks = FullMasks();
        Create(name).Find(PairImages(), PairCorners, masks);
        for (int y = 4; y < 40; y++)
        {
            for (int x = 30; x < 50; x++)
            {
                bool a = masks[0].At(y, x) != 0;
                bool b = masks[1].At(y - 4, x - 30) != 0;
                Assert.True(a ^ b, $"pixel ({x}, {y})");
            }
        }

        Assert.All(masks[0].ToArray().Take(30), v => Assert.Equal(255, v));
    }

    [Fact]
    public void Voronoi_SizesOverload_EqualsImageOverload()
    {
        Mat<byte>[] a = FullMasks();
        Mat<byte>[] b = FullMasks();
        new VoronoiSeamFinder().Find(PairImages(), PairCorners, a);
        new VoronoiSeamFinder().Find([new Size(50, 40), new Size(50, 40)], PairCorners, b);
        Assert.Equal(a[0].ToArray(), b[0].ToArray());
        Assert.Equal(a[1].ToArray(), b[1].ToArray());
    }

    [Theory]
    [MemberData(nameof(Finders))]
    public void Find_RoiInputs_EqualContinuousInputs(string name)
    {
        Mat<float>[] images = PairImages();
        Mat<byte>[] expected = FullMasks();
        expected[0].Roi(10, 10, 8, 8).Fill(0);
        expected[1].Roi(0, 20, 12, 5).Fill(0);
        Mat<byte>[] input = expected.Select(m => m.Clone()).ToArray();
        Create(name).Find(images, PairCorners, expected);

        Mat<float>[] roiImages = images.Select(im =>
        {
            var parent = new Mat<float>(im.Width + 7, im.Height + 3, 3);
            Mat<float> view = parent.Roi(5, 2, im.Width, im.Height);
            im.CopyTo(view);
            return view;
        }).ToArray();
        Mat<byte>[] roiMasks = input.Select(m =>
        {
            var parent = new Mat<byte>(m.Width + 4, m.Height + 6);
            parent.Fill(77);
            Mat<byte> view = parent.Roi(3, 1, m.Width, m.Height);
            m.CopyTo(view);
            return view;
        }).ToArray();
        Create(name).Find(roiImages, PairCorners, roiMasks);

        Assert.Equal(expected[0].ToArray(), roiMasks[0].ToArray());
        Assert.Equal(expected[1].ToArray(), roiMasks[1].ToArray());
    }

    [Fact]
    public void Dp_ByteAndFloatImagesWithEqualValues_GiveEqualMasks()
    {
        Mat<float>[] images = PairImages();
        Mat<byte>[] a = FullMasks();
        Mat<byte>[] b = FullMasks();
        new DpSeamFinder().Find(images, PairCorners, a);
        new DpSeamFinder().Find(images.Select(ToByte).ToArray(), PairCorners, b);
        Assert.Equal(a[0].ToArray(), b[0].ToArray());
        Assert.Equal(a[1].ToArray(), b[1].ToArray());
    }

    [Fact]
    public void GcGraph_MaxFlow_SmallGraph()
    {
        // s -> 0 (5), s -> 1 (4); 0 -> t (3), 1 -> t (6); 0 -> 1 (2), 1 -> 0 (0). Max flow is 9.
        using var g = new GcGraph(2, 1);
        g.AddVtx();
        g.AddVtx();
        g.AddTermWeights(0, 5, 3);
        g.AddTermWeights(1, 4, 6);
        g.AddEdges(0, 1, 2, 0);
        Assert.Equal(9f, g.MaxFlow());
        Assert.True(g.InSourceSegment(0));
        Assert.False(g.InSourceSegment(1));
    }

    [Fact]
    public void GcGraph_InvalidEdges_Throw()
    {
        using var g = new GcGraph(2, 1);
        g.AddVtx();
        g.AddVtx();
        Assert.Throws<ArgumentException>(() => g.AddEdges(0, 0, 1, 1));
        Assert.Throws<ArgumentException>(() => g.AddEdges(0, 2, 1, 1));
        Assert.Throws<ArgumentException>(() => g.AddEdges(0, 1, -1, 1));
        Assert.Throws<InvalidOperationException>(() => g.MaxFlow());
    }

    [Theory]
    [InlineData(5)]
    [InlineData(33)]
    [InlineData(41)]
    [InlineData(300)]
    [InlineData(5000)]
    public void SeamPairSort_SortsAscending_AndKeepsItemsWithKeys(int n)
    {
        var rng = new Random(n);
        int[] keys = Enumerable.Range(0, n).Select(_ => rng.Next(0, 12)).ToArray();
        int[] items = Enumerable.Range(0, n).ToArray();
        int[] original = keys.ToArray();
        SeamPairSort.Sort(items, keys);
        for (int i = 1; i < n; i++)
        {
            Assert.True(keys[i - 1] <= keys[i]);
        }

        Assert.Equal(Enumerable.Range(0, n), items.OrderBy(i => i));
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(original[items[i]], keys[i]);
        }
    }

    [Fact]
    public void SeamPairSort_UpTo32Elements_IsStable()
    {
        int[] keys = [3, 1, 3, 0, 1, 3, 0];
        int[] items = [0, 1, 2, 3, 4, 5, 6];
        SeamPairSort.Sort(items, keys);
        Assert.Equal(new[] { 3, 6, 1, 4, 0, 2, 5 }, items);
    }

    private static Mat<byte> ToByte(Mat<float> im)
    {
        float[] s = im.ToArray();
        return new Mat<byte>(s.Select(v => (byte)v).ToArray(), im.Width, im.Height, im.Channels);
    }
}
