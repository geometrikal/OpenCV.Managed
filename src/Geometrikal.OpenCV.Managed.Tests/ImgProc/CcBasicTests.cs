namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Shape, validation and edge-case tests, an independent flood-fill reference for the label
/// numbering and statistics, and analytic tests for the OpenCV 5.0 <c>ushort</c> overflow rule
/// that cv2 4.5.5 cannot generate.
/// </summary>
public sealed class CcBasicTests
{
    private static readonly ConnectedComponentsAlgorithm[] Algorithms =
        Enum.GetValues<ConnectedComponentsAlgorithm>();

    private static readonly double QuietNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

    public static IEnumerable<object[]> AlgorithmsAndConnectivity()
        => Algorithms.SelectMany(a => new[] { 4, 8 }.Select(c => new object[] { a, c }));

    public static IEnumerable<object[]> AlgorithmValues() => Algorithms.Select(a => new object[] { a });

    private static Mat<byte> FromRows(params string[] rows)
    {
        var m = new Mat<byte>(rows[0].Length, rows.Length);
        for (int y = 0; y < rows.Length; y++)
        {
            for (int x = 0; x < rows[y].Length; x++)
            {
                m.At(y, x) = rows[y][x] == '#' ? (byte)255 : (byte)0;
            }
        }

        return m;
    }

    private static Mat<byte> Random(int w, int h, int seed, double density)
    {
        var rng = new Random(seed);
        var m = new Mat<byte>(w, h);
        for (int y = 0; y < h; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                row[x] = rng.NextDouble() < density ? (byte)rng.Next(1, 256) : (byte)0;
            }
        }

        return m;
    }

    [Theory]
    [InlineData(ConnectedComponentsAlgorithm.Wu, new[] { 0, 0, 0, 0, 1, 2, 0, 0, 0, 0 })]
    [InlineData(ConnectedComponentsAlgorithm.Sauf, new[] { 0, 0, 0, 0, 1, 2, 0, 0, 0, 0 })]
    [InlineData(ConnectedComponentsAlgorithm.Default, new[] { 0, 0, 0, 0, 2, 1, 0, 0, 0, 0 })]
    [InlineData(ConnectedComponentsAlgorithm.Grana, new[] { 0, 0, 0, 0, 2, 1, 0, 0, 0, 0 })]
    [InlineData(ConnectedComponentsAlgorithm.Bbdt, new[] { 0, 0, 0, 0, 2, 1, 0, 0, 0, 0 })]
    [InlineData(ConnectedComponentsAlgorithm.Bolelli, new[] { 0, 0, 0, 0, 2, 1, 0, 0, 0, 0 })]
    [InlineData(ConnectedComponentsAlgorithm.Spaghetti, new[] { 0, 0, 0, 0, 2, 1, 0, 0, 0, 0 })]
    public void ConnectedComponents_EightConnectivity_NumbersByAlgorithmOrder(ConnectedComponentsAlgorithm algorithm, int[] expected)
    {
        Mat<byte> image = FromRows("....#", "#....");
        ConnectedComponentsResult<int> r = ConnectedComponentLabeling.ConnectedComponents<int>(image, 8, algorithm);
        Assert.Equal(3, r.LabelCount);
        Assert.Equal(expected, r.Labels.ToArray());
    }

    [Theory]
    [MemberData(nameof(AlgorithmValues))]
    public void ConnectedComponents_FourConnectivity_NumbersInRasterOrder(ConnectedComponentsAlgorithm algorithm)
    {
        Mat<byte> image = FromRows("....#", "#....");
        ConnectedComponentsResult<int> r = ConnectedComponentLabeling.ConnectedComponents<int>(image, 4, algorithm);
        Assert.Equal(new[] { 0, 0, 0, 0, 1, 2, 0, 0, 0, 0 }, r.Labels.ToArray());
    }

    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponentsWithStats_AllBackground_ReturnsOneLabel(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        var image = new Mat<byte>(9, 5);
        ConnectedComponentsStatsResult<ushort> r = ConnectedComponentLabeling.ConnectedComponentsWithStats<ushort>(image, connectivity, algorithm);
        Assert.Equal(1, r.LabelCount);
        Assert.All(r.Labels.ToArray(), v => Assert.Equal(0, v));
        Assert.Equal(new[] { 0, 0, 9, 5, 45 }, r.Stats.ToArray());
        Assert.Equal(new[] { 4.0, 2.0 }, r.Centroids.ToArray());
    }

    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponentsWithStats_AllForeground_BackgroundRowIsEmpty(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        var image = new Mat<byte>(4, 3);
        image.Fill(1);
        ConnectedComponentsStatsResult<int> r = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(image, connectivity, algorithm);
        Assert.Equal(2, r.LabelCount);
        Assert.All(r.Labels.ToArray(), v => Assert.Equal(1, v));
        Assert.Equal(new[] { -1, int.MaxValue, 0, 0, 0, 0, 0, 4, 3, 12 }, r.Stats.ToArray());
        double[] c = r.Centroids.ToArray();
        Assert.Equal(0x7FF8000000000000, BitConverter.DoubleToInt64Bits(c[0]));
        Assert.Equal(0x7FF8000000000000, BitConverter.DoubleToInt64Bits(c[1]));
        Assert.Equal(new[] { 1.5, 1.0 }, c[2..]);
        Assert.Equal(new Rect(0, 0, 4, 3), r.GetBoundingBox(1));
        Assert.Equal(12, r.GetStat(1, ConnectedComponentsStat.Area));
    }

    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponentsWithStats_SinglePixel_ProducesExpectedRows(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        var fg = new Mat<byte>(1, 1);
        fg.At(0, 0) = 3;
        ConnectedComponentsStatsResult<int> r = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(fg, connectivity, algorithm);
        Assert.Equal(2, r.LabelCount);
        Assert.Equal(new[] { 1 }, r.Labels.ToArray());
        Assert.Equal(new[] { -1, int.MaxValue, 0, 0, 0, 0, 0, 1, 1, 1 }, r.Stats.ToArray());

        ConnectedComponentsStatsResult<int> b = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(new Mat<byte>(1, 1), connectivity, algorithm);
        Assert.Equal(1, b.LabelCount);
        Assert.Equal(new[] { 0, 0, 1, 1, 1 }, b.Stats.ToArray());
        Assert.Equal(new[] { 0.0, 0.0 }, b.Centroids.ToArray());
    }

    [Fact]
    public void ConnectedComponents_SByteNegativeValues_AreForeground()
    {
        var image = new Mat<sbyte>(new sbyte[] { 0, -1, 0, 0, 0, 0, 1, 0, -128 }, 3, 3);
        ConnectedComponentsResult<int> r = ConnectedComponentLabeling.ConnectedComponents<int>(image);
        Assert.Equal(4, r.LabelCount);
        Assert.Equal(new[] { 0, 1, 0, 0, 0, 0, 2, 0, 3 }, r.Labels.ToArray());
    }

    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponents_RoiWithStride_MatchesPackedImage(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        Mat<byte> packed = Random(37, 23, 3, 0.55);
        var parent = new Mat<byte>(50, 30);
        parent.Fill(9);
        Mat<byte> roi = parent.Roi(5, 3, 37, 23);
        packed.CopyTo(roi);

        ConnectedComponentsStatsResult<int> a = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(packed, connectivity, algorithm);
        ConnectedComponentsStatsResult<int> b = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(roi, connectivity, algorithm);
        Assert.Equal(a.LabelCount, b.LabelCount);
        Assert.Equal(a.Labels.ToArray(), b.Labels.ToArray());
        Assert.Equal(a.Stats.ToArray(), b.Stats.ToArray());
        Assert.Equal(a.Centroids.ToArray(), b.Centroids.ToArray());

        var labelParent = new Mat<ushort>(41, 29);
        labelParent.Fill(777);
        Mat<ushort> labelRoi = labelParent.Roi(2, 4, 37, 23);
        int n = ConnectedComponentLabeling.ConnectedComponents(roi, labelRoi, connectivity, algorithm);
        Assert.Equal(a.LabelCount, n);
        Assert.Equal(a.Labels.ToArray().Select(v => (ushort)v).ToArray(), labelRoi.ToArray());
        int outside = 0;
        for (int y = 0; y < labelParent.Height; y++)
        {
            for (int x = 0; x < labelParent.Width; x++)
            {
                bool inside = x >= 2 && x < 39 && y >= 4 && y < 27;
                if (!inside && labelParent.At(y, x) == 777)
                {
                    outside++;
                }
            }
        }

        Assert.Equal((41 * 29) - (37 * 23), outside);
    }

    [Fact]
    public void ConnectedComponents_DoesNotModifySource()
    {
        Mat<byte> image = Random(31, 17, 4, 0.5);
        byte[] before = image.ToArray();
        _ = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(image);
        _ = ConnectedComponentLabeling.ConnectedComponents<ushort>(image, 4, ConnectedComponentsAlgorithm.Wu);
        Assert.Equal(before, image.ToArray());
    }

    [Fact]
    public void ConnectedComponents_RepeatedCalls_AreDeterministic()
    {
        Mat<byte> image = Random(301, 211, 5, 0.6);
        ConnectedComponentsStatsResult<int> a = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(image);
        ConnectedComponentsStatsResult<int> b = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(image);
        Assert.Equal(a.Labels.ToArray(), b.Labels.ToArray());
        Assert.Equal(a.Stats.ToArray(), b.Stats.ToArray());
        Assert.Equal(a.Centroids.ToArray(), b.Centroids.ToArray());
    }

    /// <summary>The port must equal a flood-fill reference for every size parity and density.</summary>
    [Theory]
    [InlineData(1, 1, 0.5)]
    [InlineData(1, 40, 0.5)]
    [InlineData(40, 1, 0.5)]
    [InlineData(2, 3, 0.7)]
    [InlineData(65, 33, 0.3)]
    [InlineData(64, 64, 0.59)]
    [InlineData(127, 129, 0.45)]
    [InlineData(333, 211, 0.62)]
    [InlineData(300, 400, 0.95)]
    [InlineData(1001, 257, 0.5)]
    public void ConnectedComponentsWithStats_MatchesFloodFillReference(int w, int h, double density)
    {
        Mat<byte> image = Random(w, h, w * 31 + h, density);
        foreach (int connectivity in new[] { 4, 8 })
        {
            foreach (ConnectedComponentsAlgorithm algorithm in Algorithms)
            {
                bool block = connectivity == 8 && algorithm is not (ConnectedComponentsAlgorithm.Wu or ConnectedComponentsAlgorithm.Sauf);
                (int n, int[] labels, int[] stats, double[] centroids) = Reference(image, connectivity, block);
                ConnectedComponentsStatsResult<int> r = ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(image, connectivity, algorithm);
                Assert.Equal(n, r.LabelCount);
                Assert.Equal(labels, r.Labels.ToArray());
                Assert.Equal(stats, r.Stats.ToArray());
                Assert.Equal(centroids.Select(BitConverter.DoubleToInt64Bits), r.Centroids.ToArray().Select(BitConverter.DoubleToInt64Bits));

                if ((long)w * h <= 65534)
                {
                    ConnectedComponentsResult<ushort> u = ConnectedComponentLabeling.ConnectedComponents<ushort>(image, connectivity, algorithm);
                    Assert.Equal(n, u.LabelCount);
                    Assert.Equal(labels, u.Labels.ToArray().Select(v => (int)v));
                }
            }
        }
    }

    /// <summary>
    /// OpenCV 5.0 regression_27568: 65536 isolated components overflow <c>ushort</c> labels for
    /// every algorithm and connectivity; <c>int</c> labels succeed.
    /// </summary>
    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponents_UShortOverflow_Throws(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        Mat<byte> image = IsolatedPixels(512, 512, 65536);
        Assert.Throws<OverflowException>(() => ConnectedComponentLabeling.ConnectedComponents<ushort>(image, connectivity, algorithm));
        Assert.Throws<OverflowException>(() => ConnectedComponentLabeling.ConnectedComponentsWithStats<ushort>(image, connectivity, algorithm));
        Assert.Equal(65537, ConnectedComponentLabeling.ConnectedComponents<int>(image, connectivity, algorithm).LabelCount);
    }

    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponents_UShortLimit_AllowsLabel65534(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        Mat<byte> ok = IsolatedPixels(512, 512, 65534);
        ConnectedComponentsResult<ushort> r = ConnectedComponentLabeling.ConnectedComponents<ushort>(ok, connectivity, algorithm);
        Assert.Equal(65535, r.LabelCount);
        Assert.Equal(65534, r.Labels.ToArray().Max());

        Mat<byte> over = IsolatedPixels(512, 512, 65535);
        Assert.Throws<OverflowException>(() => ConnectedComponentLabeling.ConnectedComponents<ushort>(over, connectivity, algorithm));
    }

    /// <summary>
    /// The check counts provisional labels: every U shape creates two provisional labels that
    /// merge in its bottom row, so 33124 components overflow <c>ushort</c> labels.
    /// </summary>
    [Theory]
    [MemberData(nameof(AlgorithmsAndConnectivity))]
    public void ConnectedComponents_UShortProvisionalOverflow_Throws(ConnectedComponentsAlgorithm algorithm, int connectivity)
    {
        const int per = 182;
        var image = new Mat<byte>(per * 6, per * 4);
        for (int by = 0; by < per; by++)
        {
            for (int bx = 0; bx < per; bx++)
            {
                int x0 = bx * 6;
                int y0 = by * 4;
                for (int k = 0; k < 3; k++)
                {
                    image.At(y0 + k, x0) = 1;
                    image.At(y0 + k, x0 + 4) = 1;
                }

                for (int x = x0; x <= x0 + 4; x++)
                {
                    image.At(y0 + 2, x) = 1;
                }
            }
        }

        Assert.Throws<OverflowException>(() => ConnectedComponentLabeling.ConnectedComponents<ushort>(image, connectivity, algorithm));
        Assert.Equal((per * per) + 1, ConnectedComponentLabeling.ConnectedComponents<int>(image, connectivity, algorithm).LabelCount);
    }

    [Fact]
    public void ConnectedComponents_InvalidArguments_Throw()
    {
        var image = new Mat<byte>(4, 4);
        Assert.Throws<ArgumentNullException>(() => ConnectedComponentLabeling.ConnectedComponents<int>((Mat<byte>)null!));
        Assert.Throws<ArgumentNullException>(() => ConnectedComponentLabeling.ConnectedComponents(image, (Mat<int>)null!));
        Assert.Throws<ArgumentException>(() => ConnectedComponentLabeling.ConnectedComponents<int>(new Mat<byte>(4, 4, 3)));
        Assert.Throws<ArgumentException>(() => ConnectedComponentLabeling.ConnectedComponents(image, new Mat<int>(4, 4, 2)));
        Assert.Throws<ArgumentException>(() => ConnectedComponentLabeling.ConnectedComponents(image, new Mat<int>(4, 5)));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConnectedComponentLabeling.ConnectedComponents<int>(image, 6));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConnectedComponentLabeling.ConnectedComponents<int>(image, 8, (ConnectedComponentsAlgorithm)6));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConnectedComponentLabeling.ConnectedComponents<int>(image, 8, (ConnectedComponentsAlgorithm)(-2)));
        Assert.Throws<NotSupportedException>(() => ConnectedComponentLabeling.ConnectedComponents<byte>(image));
        Assert.Throws<NotSupportedException>(() => ConnectedComponentLabeling.ConnectedComponentsWithStats<short>(image));
    }

    /// <summary>Foreground at the first <paramref name="count"/> even (x, y) positions in raster order.</summary>
    private static Mat<byte> IsolatedPixels(int w, int h, int count)
    {
        var m = new Mat<byte>(w, h);
        int k = 0;
        for (int y = 0; y < h && k < count; y += 2)
        {
            for (int x = 0; x < w && k < count; x += 2, k++)
            {
                m.At(y, x) = 1;
            }
        }

        Assert.Equal(count, k);
        return m;
    }

    /// <summary>
    /// Flood fill; components are numbered by first pixel (raster) or first 2x2 block (block)
    /// in raster order. Statistics follow OpenCV <c>CCStatsOp</c>.
    /// </summary>
    private static (int N, int[] Labels, int[] Stats, double[] Centroids) Reference(Mat<byte> image, int connectivity, bool block)
    {
        int w = image.Width;
        int h = image.Height;
        var comp = new int[w * h];
        Array.Fill(comp, -1);
        var firstKey = new List<long>();
        var stack = new Stack<int>();
        int components = 0;
        for (int start = 0; start < w * h; start++)
        {
            if (image.At(start / w, start % w) == 0 || comp[start] >= 0)
            {
                continue;
            }

            long best = long.MaxValue;
            comp[start] = components;
            stack.Push(start);
            while (stack.Count > 0)
            {
                int p = stack.Pop();
                int y = p / w;
                int x = p % w;
                long key = block ? ((long)(y / 2) * ((w + 1) / 2)) + (x / 2) : p;
                best = Math.Min(best, key);
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if ((dx == 0 && dy == 0) || (connectivity == 4 && dx != 0 && dy != 0))
                        {
                            continue;
                        }

                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h || image.At(ny, nx) == 0 || comp[(ny * w) + nx] >= 0)
                        {
                            continue;
                        }

                        comp[(ny * w) + nx] = components;
                        stack.Push((ny * w) + nx);
                    }
                }
            }

            firstKey.Add(best);
            components++;
        }

        int[] order = Enumerable.Range(0, components).OrderBy(i => firstKey[i]).ToArray();
        var final = new int[components];
        for (int k = 0; k < components; k++)
        {
            final[order[k]] = k + 1;
        }

        int n = components + 1;
        var labels = new int[w * h];
        var minX = Enumerable.Repeat(int.MaxValue, n).ToArray();
        var minY = Enumerable.Repeat(int.MaxValue, n).ToArray();
        var maxX = Enumerable.Repeat(int.MinValue, n).ToArray();
        var maxY = Enumerable.Repeat(int.MinValue, n).ToArray();
        var area = new int[n];
        var sx = new ulong[n];
        var sy = new ulong[n];
        for (int p = 0; p < w * h; p++)
        {
            int l = comp[p] < 0 ? 0 : final[comp[p]];
            labels[p] = l;
            int x = p % w;
            int y = p / w;
            minX[l] = Math.Min(minX[l], x);
            maxX[l] = Math.Max(maxX[l], x);
            minY[l] = Math.Min(minY[l], y);
            maxY[l] = Math.Max(maxY[l], y);
            area[l]++;
            sx[l] += (ulong)x;
            sy[l] += (ulong)y;
        }

        var stats = new int[n * 5];
        var centroids = new double[n * 2];
        for (int l = 0; l < n; l++)
        {
            if (area[l] > 0)
            {
                stats[(l * 5) + 0] = minX[l];
                stats[(l * 5) + 1] = minY[l];
                stats[(l * 5) + 2] = maxX[l] - minX[l] + 1;
                stats[(l * 5) + 3] = maxY[l] - minY[l] + 1;
                stats[(l * 5) + 4] = area[l];
                centroids[2 * l] = sx[l] / (double)area[l];
                centroids[(2 * l) + 1] = sy[l] / (double)area[l];
            }
            else
            {
                stats[(l * 5) + 0] = -1;
                stats[(l * 5) + 1] = int.MaxValue;
                centroids[2 * l] = QuietNaN;
                centroids[(2 * l) + 1] = QuietNaN;
            }
        }

        return (n, labels, stats, centroids);
    }
}
