namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/shapes_golden.json</c> through
/// approxPolyDP, arcLength, contourArea, boundingRect, moments, HuMoments and matchShapes. Points
/// must match exactly and every double must have the same bits (no tolerance).
/// Regenerate with <c>python tools/generate_shapes_golden.py</c>.
/// </summary>
public sealed class ShapeDescriptorsGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.True(Data.Cases.Length > 1000);
        Assert.Equal("5.0.0", Data.Generator["cv2"].GetString());
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void ShapeDescriptors_MatchOpenCvGolden(string name)
    {
        ShapeCase c = Data.Cases.Single(x => x.Name == name);
        double[] actual = Run(c, out ArrayData? points);
        Assert.Equal(c.Values.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            double expected = c.Values[i];
            Assert.True(
                BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual[i]) || (double.IsNaN(expected) && double.IsNaN(actual[i])),
                $"{name}[{i}]: expected {expected:R}, actual {actual[i]:R}");
        }

        if (c.Expected is int e)
        {
            ArrayData exp = Data.Arrays[e];
            Assert.NotNull(points);
            Assert.Equal(exp.Dtype, points!.Dtype);
            Assert.Equal(exp.Data, points.Data);
        }
    }

    private static double[] Run(ShapeCase c, out ArrayData? points)
    {
        points = null;
        bool B(string key) => c.Params[key].GetBoolean();
        ArrayData a0 = Data.Arrays[c.Inputs[0]];
        switch (c.Op)
        {
            case "arcLength":
                return [a0.Dtype == "i32" ? ShapeDescriptors.ArcLength(IntPoints(a0), B("closed")) : ShapeDescriptors.ArcLength(FloatPoints(a0), B("closed"))];
            case "contourArea":
                return [a0.Dtype == "i32" ? ShapeDescriptors.ContourArea(IntPoints(a0), B("oriented")) : ShapeDescriptors.ContourArea(FloatPoints(a0), B("oriented"))];
            case "boundingRect":
            {
                Rect r = a0.Dtype == "i32" ? ShapeDescriptors.BoundingRect(IntPoints(a0)) : ShapeDescriptors.BoundingRect(FloatPoints(a0));
                return [r.X, r.Y, r.Width, r.Height];
            }

            case "moments":
            {
                Moments m = a0.Dtype == "i32" ? ShapeDescriptors.Moments(IntPoints(a0)) : ShapeDescriptors.Moments(FloatPoints(a0));
                return WithHu(m);
            }

            case "approx":
            {
                double eps = c.Params["epsilon"].GetDouble();
                if (a0.Dtype == "i32")
                {
                    Point[] r = ShapeDescriptors.ApproxPolyDP(IntPoints(a0), eps, B("closed"));
                    points = Store("i32", MemoryMarshal.AsBytes(r.AsSpan()).ToArray(), r.Length);
                    return [r.Length];
                }
                else
                {
                    Point2f[] r = ShapeDescriptors.ApproxPolyDP(FloatPoints(a0), eps, B("closed"));
                    points = Store("f32", MemoryMarshal.AsBytes(r.AsSpan()).ToArray(), r.Length);
                    return [r.Length];
                }
            }

            case "imageMoments":
            {
                bool binary = B("binary");
                Moments m = a0.Dtype switch
                {
                    "u8" => ShapeDescriptors.Moments(Roi(c, LoadMat<byte>(a0)), binary),
                    "u16" => ShapeDescriptors.Moments(Roi(c, LoadMat<ushort>(a0)), binary),
                    "i16" => ShapeDescriptors.Moments(Roi(c, LoadMat<short>(a0)), binary),
                    "f32" => ShapeDescriptors.Moments(Roi(c, LoadMat<float>(a0)), binary),
                    "f64" => ShapeDescriptors.Moments(Roi(c, LoadMat<double>(a0)), binary),
                    _ => throw new InvalidOperationException(a0.Dtype),
                };
                return WithHu(m);
            }

            case "maskBoundingRect":
            {
                Rect r = ShapeDescriptors.BoundingRect(Roi(c, LoadMat<byte>(a0)));
                return [r.X, r.Y, r.Width, r.Height];
            }

            case "matchShapes":
            {
                var method = (ShapeMatchModes)c.Params["method"].GetInt32();
                ArrayData a1 = Data.Arrays[c.Inputs[1]];
                return [a0.Dtype == "i32"
                    ? ShapeDescriptors.MatchShapes(IntPoints(a0), IntPoints(a1), method)
                    : ShapeDescriptors.MatchShapes(FloatPoints(a0), FloatPoints(a1), method)];
            }

            case "matchShapesImage":
            {
                var method = (ShapeMatchModes)c.Params["method"].GetInt32();
                return [ShapeDescriptors.MatchShapes(LoadMat<byte>(a0), LoadMat<byte>(Data.Arrays[c.Inputs[1]]), method)];
            }

            default:
                throw new InvalidOperationException(c.Op);
        }
    }

    private static double[] WithHu(Moments m)
    {
        double[] hu = ShapeDescriptors.HuMoments(m);
        return
        [
            m.M00, m.M10, m.M01, m.M20, m.M11, m.M02, m.M30, m.M21, m.M12, m.M03,
            m.Mu20, m.Mu11, m.Mu02, m.Mu30, m.Mu21, m.Mu12, m.Mu03,
            m.Nu20, m.Nu11, m.Nu02, m.Nu30, m.Nu21, m.Nu12, m.Nu03,
            .. hu,
        ];
    }

    private static ArrayData Store(string dtype, byte[] bytes, int count)
        => new() { Dtype = dtype, Width = count, Height = 1, Channels = 2, Data = Convert.ToBase64String(bytes) };

    private static Mat<T> Roi<T>(ShapeCase c, Mat<T> m)
        where T : unmanaged
    {
        if (!c.Params.TryGetValue("roi", out JsonElement re))
        {
            return m;
        }

        int[] r = re.EnumerateArray().Select(e => e.GetInt32()).ToArray();
        return m.Roi(r[0], r[1], r[2], r[3]);
    }

    private static Point[] IntPoints(ArrayData a)
    {
        Assert.Equal("i32", a.Dtype);
        return MemoryMarshal.Cast<byte, Point>(Convert.FromBase64String(a.Data)).ToArray();
    }

    private static Point2f[] FloatPoints(ArrayData a)
    {
        Assert.Equal("f32", a.Dtype);
        return MemoryMarshal.Cast<byte, Point2f>(Convert.FromBase64String(a.Data)).ToArray();
    }

    private static Mat<T> LoadMat<T>(ArrayData a)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "shapes_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        })!;
    }

    private sealed class GoldenFile
    {
        public Dictionary<string, JsonElement> Generator { get; set; } = [];

        public ArrayData[] Arrays { get; set; } = [];

        public ShapeCase[] Cases { get; set; } = [];
    }

    private sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    private sealed class ShapeCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int[] Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public double[] Values { get; set; } = [];

        public int? Expected { get; set; }
    }
}
