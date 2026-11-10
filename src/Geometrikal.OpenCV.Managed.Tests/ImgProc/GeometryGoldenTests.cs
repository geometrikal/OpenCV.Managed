namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 4.5.5 reference outputs in <c>TestData/geometry_golden.json</c> through
/// <see cref="Geometry"/> and <see cref="MatrixTransforms"/>. Output buffers must match byte for
/// byte. Regenerate with <c>python tools/generate_geometry_golden.py</c>.
/// </summary>
public sealed class GeometryGoldenTests
{
    private static readonly GoldenFile Data = Load();

    /// <summary>
    /// Adjudicated cv2 4.5.5 versus OpenCV 5.0 differences. <c>INTER_NEAREST_EXACT</c>: OpenCV 5.0
    /// accumulates <c>src/dst</c> in double precision (pull request 28873); 4.5.5 uses a 16.16
    /// fixed-point step with 32-bit products. <c>getPerspectiveTransform</c> with collinear points:
    /// OpenCV 5.0 falls back to an SVD of the homogeneous system (pull request 26926). The OpenCV
    /// 5.0 results are pinned in <see cref="GeometryOpenCv5DifferenceTests"/>.
    /// </summary>
    private static readonly string[] AdjudicatedSuspects =
    [
        "perspectiveTransform_collinear",
        "resize_x_nnx_f64_c1_64x48_to_32x24_synth",
        "resize_x_nnx_f64_c4_70x12_to_97x9_synth",
        "resize_x_nnx_i32_c1_64x48_to_32x24_synth",
        "resize_x_nnx_i32_c4_70x12_to_97x9_synth",
        "resize_x_nnx_i8_c1_64x48_to_32x24_synth",
        "resize_x_nnx_i8_c4_70x12_to_97x9_synth",
        "resize_x_nnx_u16_c1_64x48_to_32x24_synth",
        "resize_x_nnx_u16_c4_70x12_to_97x9_synth",
        "resize_x_nnx_u8_c1_40000x1_to_30000x1_synth",
    ];

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = Data.Cases.Where(c => c.Suspect).Select(c => c.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.Equal(AdjudicatedSuspects.OrderBy(n => n, StringComparer.Ordinal), actual);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Geometry_MatchesOpenCvGolden(string name)
    {
        GeometryCase c = Data.Cases.Single(x => x.Name == name);
        byte[][] actual = c.Op switch
        {
            "rotationMatrix" or "affineTransform" or "perspectiveTransform" or "invertAffine" => RunMatrix(c),
            "convertMaps" => RunConvertMaps(c),
            _ => SourceType(c) switch
            {
                "u8" => Run<byte>(c),
                "i8" => Run<sbyte>(c),
                "u16" => Run<ushort>(c),
                "i16" => Run<short>(c),
                "i32" => Run<int>(c),
                "f32" => Run<float>(c),
                "f64" => Run<double>(c),
                string d => throw new InvalidOperationException(d),
            },
        };

        if (c.ExpectedSha256 is not null)
        {
            Assert.Equal(c.ExpectedSha256.Length, actual.Length);
            for (int k = 0; k < actual.Length; k++)
            {
                Assert.True(
                    c.ExpectedSha256[k] == Convert.ToHexStringLower(SHA256.HashData(actual[k])),
                    $"Golden hash mismatch for '{name}' output {k}.");
            }

            return;
        }

        Assert.Equal(c.Expected.Length, actual.Length);
        for (int k = 0; k < actual.Length; k++)
        {
            byte[] expected = Convert.FromBase64String(Data.Arrays[c.Expected[k]].Data);
            if (!expected.AsSpan().SequenceEqual(actual[k]))
            {
                int i = 0;
                while (i < expected.Length && i < actual[k].Length && expected[i] == actual[k][i])
                {
                    i++;
                }

                Assert.Fail($"Golden mismatch for '{name}' output {k} at byte {i} of {expected.Length} (actual length {actual[k].Length}).");
            }
        }
    }

    private static string SourceType(GeometryCase c)
        => c.Params.TryGetValue("synth", out JsonElement synth) ? synth.GetString()! : Data.Arrays[c.Inputs[0]!.Value].Dtype;

    /// <summary>The formula image of <c>synth</c> in the generator.</summary>
    private static Mat<T> Synthetic<T>(int width, int height, int channels)
        where T : unmanaged
    {
        var m = new Mat<T>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<T> row = m.GetRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    long b = ((x * 73L) + (y * 151L) + (c * 37L) + ((x * (long)y * 5) >> 2)) % 65536;
                    object v = typeof(T) == typeof(byte) ? (byte)(b & 255)
                        : typeof(T) == typeof(sbyte) ? (sbyte)((b & 255) - 128)
                        : typeof(T) == typeof(ushort) ? (ushort)b
                        : typeof(T) == typeof(short) ? (short)(b - 32768)
                        : typeof(T) == typeof(int) ? (int)((b - 32768) * 60001)
                        : typeof(T) == typeof(float) ? (float)((b * 0.0371) - 600.0)
                        : (object)((b * 0.0371) - 600.0);
                    row[(x * channels) + c] = (T)v;
                }
            }
        }

        return m;
    }

    private static byte[][] Run<T>(GeometryCase c)
        where T : unmanaged
    {
        Mat<T> src = c.Params.ContainsKey("synth") ? Synthetic<T>(I(c, "sw"), I(c, "sh"), I(c, "cn")) : Mat<T>(c.Inputs[0]);
        switch (c.Op)
        {
            case "resize":
                return [Bytes(Geometry.Resize(src, new Size(I(c, "dw"), I(c, "dh")), D(c, "fx"), D(c, "fy"), (InterpolationFlags)I(c, "interp")))];
            case "flip":
                return [Bytes(MatrixTransforms.Flip(src, I(c, "code")))];
            case "rotate":
                return [Bytes(MatrixTransforms.Rotate(src, (RotateFlags)I(c, "code")))];
            case "transpose":
                return [Bytes(MatrixTransforms.Transpose(src))];
            case "warpAffine":
            case "warpPerspective":
                return [Bytes(RunWarp(c, src))];
            case "remap":
                return [Bytes(RunRemap(c, src))];
            default:
                throw new InvalidOperationException(c.Op);
        }
    }

    private static Mat<T> RunWarp<T>(GeometryCase c, Mat<T> src)
        where T : unmanaged
    {
        Mat<double> m = Data.Arrays[c.Inputs[1]!.Value].Dtype == "f32" ? ToDouble(Mat<float>(c.Inputs[1])) : Mat<double>(c.Inputs[1]);
        var interp = (InterpolationFlags)I(c, "interp");
        WarpFlags flags = c.Params["inverse"].GetBoolean() ? WarpFlags.InverseMap : WarpFlags.None;
        var border = (BorderType)I(c, "border");
        Scalar value = BorderValue(c);
        bool affine = c.Op == "warpAffine";
        if (c.Inputs[2] is not null)
        {
            Mat<T> dst = Mat<T>(c.Inputs[2]);
            if (affine)
            {
                Geometry.WarpAffine(src, dst, m, interp, flags, border, value);
            }
            else
            {
                Geometry.WarpPerspective(src, dst, m, interp, flags, border, value);
            }

            return dst;
        }

        var size = new Size(I(c, "dw"), I(c, "dh"));
        if (c.Params["matrix32"].GetBoolean())
        {
            Mat<float> m32 = Mat<float>(c.Inputs[1]);
            return affine
                ? Geometry.WarpAffine(src, m32, size, interp, flags, border, value)
                : Geometry.WarpPerspective(src, m32, size, interp, flags, border, value);
        }

        return affine
            ? Geometry.WarpAffine(src, m, size, interp, flags, border, value)
            : Geometry.WarpPerspective(src, m, size, interp, flags, border, value);
    }

    private static Mat<T> RunRemap<T>(GeometryCase c, Mat<T> src)
        where T : unmanaged
    {
        var interp = (InterpolationFlags)I(c, "interp");
        var border = (BorderType)I(c, "border");
        Scalar value = BorderValue(c);
        string mapType = c.Params["maptype"].GetString()!;
        Mat<T>? dst = c.Inputs[3] is null ? null : Mat<T>(c.Inputs[3]);
        if (mapType.StartsWith("32F", StringComparison.Ordinal))
        {
            Mat<float> m1 = Mat<float>(c.Inputs[1]);
            Mat<float>? m2 = c.Inputs[2] is null ? null : Mat<float>(c.Inputs[2]);
            if (dst is null)
            {
                return Geometry.Remap(src, m1, m2, interp, border, value);
            }

            Geometry.Remap(src, dst, m1, m2, interp, border, value);
            return dst;
        }

        Mat<short> xy = Mat<short>(c.Inputs[1]);
        Mat<ushort>? fr = c.Inputs[2] is null ? null : Mat<ushort>(c.Inputs[2]);
        if (dst is null)
        {
            return Geometry.Remap(src, xy, fr, interp, border, value);
        }

        Geometry.Remap(src, dst, xy, fr, interp, border, value);
        return dst;
    }

    private static byte[][] RunConvertMaps(GeometryCase c)
    {
        string from = c.Params["from"].GetString()!, to = c.Params["to"].GetString()!;
        bool nn = c.Params["nn"].GetBoolean();
        switch (from, to)
        {
            case ("32FC1" or "32FC2", "16SC2"):
            {
                Mat<float> m1 = Mat<float>(c.Inputs[0]);
                Mat<float>? m2 = c.Inputs[1] is null ? null : Mat<float>(c.Inputs[1]);
                (Mat<short> xy, Mat<ushort>? fr) = Geometry.ConvertMapsToFixedPoint(m1, m2, nn);
                return fr is null ? [Bytes(xy)] : [Bytes(xy), Bytes(fr)];
            }

            case ("16SC2", "32FC1"):
            {
                (Mat<float> mx, Mat<float> my) = Geometry.ConvertMapsToFloat(Mat<short>(c.Inputs[0]), c.Inputs[1] is null ? null : Mat<ushort>(c.Inputs[1]));
                return [Bytes(mx), Bytes(my)];
            }

            case ("16SC2", "32FC2"):
                return [Bytes(Geometry.ConvertMapsToFloatInterleaved(Mat<short>(c.Inputs[0]), c.Inputs[1] is null ? null : Mat<ushort>(c.Inputs[1])))];
            case ("32FC2", "32FC1"):
            {
                (Mat<float> mx, Mat<float> my) = Geometry.ConvertMapsToFloat(Mat<float>(c.Inputs[0]));
                return [Bytes(mx), Bytes(my)];
            }

            case ("32FC1", "32FC2"):
                return [Bytes(Geometry.ConvertMapsToFloatInterleaved(Mat<float>(c.Inputs[0]), Mat<float>(c.Inputs[1])))];
            default:
                throw new InvalidOperationException($"{from} -> {to}");
        }
    }

    private static byte[][] RunMatrix(GeometryCase c)
    {
        switch (c.Op)
        {
            case "rotationMatrix":
            {
                float[] center = Mat<float>(c.Inputs[0]).ToArray();
                return [Bytes(Geometry.GetRotationMatrix2D(new Point2f(center[0], center[1]), D(c, "angle"), D(c, "scale")))];
            }

            case "affineTransform":
                return [Bytes(Geometry.GetAffineTransform(Points(c.Inputs[0]), Points(c.Inputs[1])))];
            case "perspectiveTransform":
                return [Bytes(Geometry.GetPerspectiveTransform(Points(c.Inputs[0]), Points(c.Inputs[1])))];
            default:
                return Data.Arrays[c.Inputs[0]!.Value].Dtype == "f32"
                    ? [Bytes(Geometry.InvertAffineTransform(Mat<float>(c.Inputs[0])))]
                    : [Bytes(Geometry.InvertAffineTransform(Mat<double>(c.Inputs[0])))];
        }
    }

    private static Point2f[] Points(int? index)
    {
        float[] v = Mat<float>(index).ToArray();
        var p = new Point2f[v.Length / 2];
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = new Point2f(v[2 * i], v[(2 * i) + 1]);
        }

        return p;
    }

    private static Scalar BorderValue(GeometryCase c)
    {
        double[] v = c.Params["borderValue"].EnumerateArray().Select(e => e.GetDouble()).ToArray();
        return new Scalar(v[0], v[1], v[2], v[3]);
    }

    private static Mat<double> ToDouble(Mat<float> m)
    {
        float[] f = m.ToArray();
        return new Mat<double>(f.Select(x => (double)x).ToArray(), m.Width, m.Height, m.Channels);
    }

    private static int I(GeometryCase c, string key) => c.Params[key].GetInt32();

    private static double D(GeometryCase c, string key) => c.Params[key].GetDouble();

    internal static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
        => MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray();

    private static Mat<T> Mat<T>(int? index)
        where T : unmanaged
    {
        ArrayData a = Data.Arrays[index!.Value];
        byte[] raw = Convert.FromBase64String(a.Data);
        var buffer = new T[a.Width * a.Height * a.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, a.Width, a.Height, a.Channels);
    }

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "geometry_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public GeometryCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class GeometryCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int?[] Inputs { get; set; } = [];

        public Dictionary<string, JsonElement> Params { get; set; } = [];

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }

        public int[] Expected { get; set; } = [];

        public string[]? ExpectedSha256 { get; set; }
    }
}
