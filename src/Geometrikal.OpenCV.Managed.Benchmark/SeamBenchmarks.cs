using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Seam finders on four float tiles of <see cref="Size"/> x <see cref="Size"/> in a 2x2 grid with
/// a step of 3/4 tile. The tiles are cut from <c>Inputs.Color(Size * 7 / 4)</c>; tile i is scaled by
/// <c>1 + i / 16</c>. Each call starts from full masks. OpenCvSharp does not expose the seam finders,
/// so this suite has no OpenCvSharp twin.
/// </summary>
[MemoryDiagnoser]
public class SeamBenchmarks
{
    private Mat<float>[] _tiles = null!;
    private Point[] _corners = null!;
    private Mat<byte>[] _masks = null!;
    private readonly VoronoiSeamFinder _voronoi = new();
    private readonly DpSeamFinder _dpColorGrad = new(DpSeamCostFunction.ColorGrad);
    private readonly GraphCutSeamFinder _graphCutColor = new(GraphCutCostType.Color);

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        int step = Size * 3 / 4;
        Mat<byte> world = Inputs.Color(Size + step);
        _corners = [new Point(0, 0), new Point(step, 0), new Point(0, step), new Point(step, step)];
        _tiles = new Mat<float>[4];
        _masks = new Mat<byte>[4];
        for (int i = 0; i < 4; i++)
        {
            Mat<byte> roi = world.Roi(_corners[i].X, _corners[i].Y, Size, Size);
            var tile = new Mat<float>(Size, Size, 3);
            float gain = 1f + (i / 16f);
            for (int y = 0; y < Size; y++)
            {
                ReadOnlySpan<byte> s = roi.GetReadOnlyRowSpan(y);
                Span<float> d = tile.GetRowSpan(y);
                for (int x = 0; x < s.Length; x++)
                {
                    d[x] = s[x] * gain;
                }
            }

            _tiles[i] = tile;
            _masks[i] = new Mat<byte>(Size, Size);
        }
    }

    private Mat<byte>[] ResetMasks()
    {
        foreach (Mat<byte> m in _masks)
        {
            m.Fill(255);
        }

        return _masks;
    }

    [Benchmark]
    public Mat<byte>[] Voronoi_F32C3()
    {
        Mat<byte>[] masks = ResetMasks();
        _voronoi.Find(_tiles, _corners, masks);
        return masks;
    }

    [Benchmark]
    public Mat<byte>[] DpColorGrad_F32C3()
    {
        Mat<byte>[] masks = ResetMasks();
        _dpColorGrad.Find(_tiles, _corners, masks);
        return masks;
    }

    [Benchmark]
    public Mat<byte>[] GraphCutColor_F32C3()
    {
        Mat<byte>[] masks = ResetMasks();
        _graphCutColor.Find(_tiles, _corners, masks);
        return masks;
    }
}
