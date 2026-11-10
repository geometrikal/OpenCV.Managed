using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Stitching warpers, blenders and exposure compensation. <see cref="Size"/> is the tile size;
/// the blender and compensator cases use four tiles that overlap by 30 % in each direction.
/// OpenCvSharp does not expose the <c>detail</c> warpers, blenders or compensators, so this suite
/// has no OpenCvSharp twin.
/// </summary>
[MemoryDiagnoser]
public class StitchingBenchmarks
{
    private Mat<byte> _color = null!;
    private Mat<float> _k = null!;
    private Mat<float> _r = null!;
    private Point[] _corners = null!;
    private Size[] _sizes = null!;
    private Mat<byte>[] _tiles = null!;
    private Mat<short>[] _tiles16 = null!;
    private Mat<byte>[] _masks = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _color = Inputs.Color(Size);
        _k = new Mat<float>(3, 3);
        _k.At(0, 0) = Size;
        _k.At(1, 1) = Size;
        _k.At(0, 2) = Size / 2f;
        _k.At(1, 2) = Size / 2f;
        _k.At(2, 2) = 1;
        _r = RotationY(0.15);

        int step = Size * 7 / 10;
        _corners = [new Point(0, 0), new Point(step, 0), new Point(0, step), new Point(step, step)];
        _sizes = Enumerable.Repeat(new Size(Size, Size), 4).ToArray();
        _tiles = new Mat<byte>[4];
        _tiles16 = new Mat<short>[4];
        _masks = new Mat<byte>[4];
        for (int i = 0; i < 4; i++)
        {
            _tiles[i] = Inputs.Color(Size);
            _tiles16[i] = Conversions.ConvertTo<byte, short>(_tiles[i]);
            _masks[i] = new Mat<byte>(Size, Size);
            _masks[i].SetTo(Scalar.All(255));
        }
    }

    [Benchmark]
    public Mat<byte> PlaneWarper_Warp_U8C3()
    {
        new PlaneWarper(Size).Warp(_color, _k, _r, InterpolationFlags.Linear, BorderType.Reflect, out Mat<byte> dst);
        return dst;
    }

    [Benchmark]
    public Mat<float> SphericalWarper_BuildMaps()
    {
        new SphericalWarper(Size).BuildMaps(new Size(Size, Size), _k, _r, out Mat<float> mapX, out _);
        return mapX;
    }

    [Benchmark]
    public Mat<short> FeatherBlender_FeedBlend_4Tiles()
    {
        var blender = new FeatherBlender();
        blender.Prepare(_corners, _sizes);
        for (int i = 0; i < 4; i++)
        {
            blender.Feed(_tiles16[i], _masks[i], _corners[i]);
        }

        blender.Blend(out Mat<short> dst, out _);
        return dst;
    }

    [Benchmark]
    public Mat<short> MultiBandBlender5_FeedBlend_4Tiles()
    {
        var blender = new MultiBandBlender(5);
        blender.Prepare(_corners, _sizes);
        for (int i = 0; i < 4; i++)
        {
            blender.Feed(_tiles[i], _masks[i], _corners[i]);
        }

        blender.Blend(out Mat<short> dst, out _);
        return dst;
    }

    [Benchmark]
    public double[] GainCompensator_Feed_4Tiles()
    {
        var compensator = new GainCompensator();
        compensator.Feed(_corners, _tiles, _masks);
        return compensator.Gains;
    }

    private static Mat<float> RotationY(double angle)
    {
        var r = new Mat<float>(3, 3);
        r.At(0, 0) = (float)Math.Cos(angle);
        r.At(0, 2) = (float)Math.Sin(angle);
        r.At(1, 1) = 1;
        r.At(2, 0) = -(float)Math.Sin(angle);
        r.At(2, 2) = (float)Math.Cos(angle);
        return r;
    }
}
