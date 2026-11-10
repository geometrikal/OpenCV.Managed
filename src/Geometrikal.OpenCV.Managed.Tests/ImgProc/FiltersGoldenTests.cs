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
/// Data-driven runner replaying the OpenCV (cv2 4.5.5) filtering reference
/// outputs captured in <c>TestData/filters_golden.json</c>: GaussianBlur,
/// blur/boxFilter, medianBlur, filter2D, sepFilter2D, Sobel, Scharr,
/// Laplacian and getGaussianKernel must reproduce the reference
/// byte-for-byte.
///
/// Regenerate the golden data with <c>python tools/generate_filters_golden.py</c>
/// (see the generator header for the environment constraints the battery
/// must respect).
/// </summary>
public sealed class FiltersGoldenTests
{
    internal static readonly IReadOnlyDictionary<string, FilterCase> AllCases = LoadCases();

    public static IEnumerable<object[]> CaseNames()
    {
        foreach (string name in AllCases.Keys.OrderBy(n => n, StringComparer.Ordinal))
        {
            if (AllCases[name].Suspect)
            {
                continue;
            }

            yield return new object[] { name };
        }
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Filters_MatchOpenCvGolden(string caseName)
    {
        FilterCase c = AllCases[caseName];
        byte[] expected = Convert.FromBase64String(c.Expected!);
        byte[] actual = Run(c);

        if (!expected.AsSpan().SequenceEqual(actual))
        {
            Assert.Fail(
                $"Golden mismatch for '{caseName}' (op={c.Op}, dtype={c.Dtype}→{c.Ddepth}, " +
                $"{c.Width}x{c.Height}x{c.Channels}): first difference at byte " +
                $"{FirstDiff(expected, actual)} of {expected.Length} " +
                $"(actual length {actual.Length}).");
        }
    }

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.NotEmpty(AllCases);
    }

    /// <summary>
    /// The adjudicated cv2 4.5.5 versus OpenCV 5.0 differences. Each one is pinned in
    /// <see cref="FiltersOpenCv5DifferenceTests"/>.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
    {
        string[] actual = AllCases.Values
            .Where(c => c.Suspect)
            .Select(c => c.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(new[] { "lap_f64_k3_scale01", "lap_u8f64_k1_scale01" }, actual);
    }

    /// <summary>Replays one case through the managed API and returns the raw output bytes.</summary>
    internal static byte[] Run(FilterCase c)
        => c.Op switch
        {
            "getGaussianKernel" => MemoryMarshal.AsBytes(
                Filters.GetGaussianKernel(c.N, c.Sigma).AsSpan()).ToArray(),
            _ => c.Dtype switch
            {
                "u8" => RunCase<byte>(c),
                "u16" => RunCase<ushort>(c),
                "s16" => RunCase<short>(c),
                "f32" => RunCase<float>(c),
                "f64" => RunCase<double>(c),
                _ => throw new InvalidOperationException($"unknown dtype '{c.Dtype}'"),
            },
        };

    private static byte[] RunCase<TSrc>(FilterCase c)
        where TSrc : unmanaged
    {
        Mat<TSrc> src = BuildInput<TSrc>(c);
        return c.Op switch
        {
            "gaussianBlur" => RunGaussian(c, src),
            "boxFilter" => c.Ddepth switch
            {
                "u8" => RunBox<TSrc, byte>(c, src),
                "s16" => RunBox<TSrc, short>(c, src),
                "f32" => RunBox<TSrc, float>(c, src),
                _ => throw new InvalidOperationException($"unknown ddepth '{c.Ddepth}'"),
            },
            "medianBlur" => RawBytes(Filters.MedianBlur(src, c.Ksize![0])),
            "filter2D" => Dispatch(c, src, RunFilter2D<TSrc, byte>, RunFilter2D<TSrc, ushort>, RunFilter2D<TSrc, short>, RunFilter2D<TSrc, float>, RunFilter2D<TSrc, double>),
            "sepFilter2D" => Dispatch(c, src, RunSep<TSrc, byte>, RunSep<TSrc, ushort>, RunSep<TSrc, short>, RunSep<TSrc, float>, RunSep<TSrc, double>),
            "sobel" => Dispatch(c, src, RunSobel<TSrc, byte>, RunSobel<TSrc, ushort>, RunSobel<TSrc, short>, RunSobel<TSrc, float>, RunSobel<TSrc, double>),
            "scharr" => Dispatch(c, src, RunScharr<TSrc, byte>, RunScharr<TSrc, ushort>, RunScharr<TSrc, short>, RunScharr<TSrc, float>, RunScharr<TSrc, double>),
            "laplacian" => Dispatch(c, src, RunLaplacian<TSrc, byte>, RunLaplacian<TSrc, ushort>, RunLaplacian<TSrc, short>, RunLaplacian<TSrc, float>, RunLaplacian<TSrc, double>),
            _ => throw new InvalidOperationException($"unknown op '{c.Op}'"),
        };
    }

    private static byte[] Dispatch<TSrc>(
        FilterCase c,
        Mat<TSrc> src,
        Func<FilterCase, Mat<TSrc>, byte[]> u8,
        Func<FilterCase, Mat<TSrc>, byte[]> u16,
        Func<FilterCase, Mat<TSrc>, byte[]> s16,
        Func<FilterCase, Mat<TSrc>, byte[]> f32,
        Func<FilterCase, Mat<TSrc>, byte[]> f64)
        where TSrc : unmanaged
        => c.Ddepth switch
        {
            "u8" => u8(c, src),
            "u16" => u16(c, src),
            "s16" => s16(c, src),
            "f32" => f32(c, src),
            "f64" => f64(c, src),
            _ => throw new InvalidOperationException($"unknown ddepth '{c.Ddepth}'"),
        };

    private static byte[] RunGaussian<TSrc>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
    {
        var ksize = new Size(c.Ksize![0], c.Ksize[1]);
        if (typeof(TSrc) == typeof(byte))
        {
            return RawBytes(Filters.GaussianBlur(
                (Mat<byte>)(object)src, ksize, c.SigmaX, c.SigmaY, (BorderType)c.BorderType));
        }

        return RawBytes(Filters.GaussianBlur(
            (Mat<float>)(object)src, ksize, c.SigmaX, c.SigmaY, (BorderType)c.BorderType));
    }

    private static byte[] RunBox<TSrc, TDst>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
        => RawBytes(Filters.BoxFilter<TSrc, TDst>(
            src, new Size(c.Ksize![0], c.Ksize[1]), BuildAnchor(c), c.Normalize,
            (BorderType)c.BorderType));

    private static byte[] RunFilter2D<TSrc, TDst>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        float[] coeffs = FloatArray(c.Kernel!);
        var kernel = new Mat<float>(coeffs, c.Kw, c.Kh, 1);
        return RawBytes(Filters.Filter2D<TSrc, TDst>(
            src, kernel, BuildAnchor(c), c.Delta, (BorderType)c.BorderType));
    }

    private static byte[] RunSep<TSrc, TDst>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
        => RawBytes(Filters.SepFilter2D<TSrc, TDst>(
            src, FloatArray(c.KernelX!), FloatArray(c.KernelY!), BuildAnchor(c), c.Delta,
            (BorderType)c.BorderType));

    private static byte[] RunSobel<TSrc, TDst>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
        => RawBytes(Derivatives.Sobel<TSrc, TDst>(
            src, c.Dx, c.Dy, c.Ksize![0], c.Scale, c.Delta, (BorderType)c.BorderType));

    private static byte[] RunScharr<TSrc, TDst>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
        => RawBytes(Derivatives.Scharr<TSrc, TDst>(
            src, c.Dx, c.Dy, c.Scale, c.Delta, (BorderType)c.BorderType));

    private static byte[] RunLaplacian<TSrc, TDst>(FilterCase c, Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
        => RawBytes(Derivatives.Laplacian<TSrc, TDst>(
            src, c.Ksize![0], c.Scale, c.Delta, (BorderType)c.BorderType));

    private static Mat<T> BuildInput<T>(FilterCase c)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(c.Input!);
        var buffer = new T[c.Width * c.Height * c.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, c.Width, c.Height, c.Channels);
    }

    private static float[] FloatArray(string b64)
    {
        byte[] raw = Convert.FromBase64String(b64);
        var result = new float[raw.Length / 4];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(result.AsSpan()));
        return result;
    }

    private static Point? BuildAnchor(FilterCase c)
        => c.Anchor is null ? null : new Point(c.Anchor[0], c.Anchor[1]);

    private static byte[] RawBytes<T>(Mat<T> mat)
        where T : unmanaged
    {
        T[] packed = mat.ToArray();
        return MemoryMarshal.AsBytes(packed.AsSpan()).ToArray();
    }

    private static int FirstDiff(byte[] a, byte[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            if (a[i] != b[i])
            {
                return i;
            }
        }

        return n;
    }

    private static IReadOnlyDictionary<string, FilterCase> LoadCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "filters_golden.json");
        using FileStream stream = File.OpenRead(path);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        FilterGoldenFile file = JsonSerializer.Deserialize<FilterGoldenFile>(stream, options)
            ?? throw new InvalidOperationException("empty golden file");
        return file.Cases.ToDictionary(c => c.Name, c => c);
    }

    internal sealed class FilterGoldenFile
    {
        public List<FilterCase> Cases { get; set; } = [];
    }

    internal sealed class FilterCase
    {
        public string Name { get; set; } = "";

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }

        public string Op { get; set; } = "";

        public string Dtype { get; set; } = "";

        public string Ddepth { get; set; } = "";

        public int Channels { get; set; }

        public int Width { get; set; }

        public int Height { get; set; }

        public string? Input { get; set; }

        public string? Expected { get; set; }

        public int[]? Ksize { get; set; }

        public double SigmaX { get; set; }

        public double SigmaY { get; set; }

        public int BorderType { get; set; }

        public int[]? Anchor { get; set; }

        public bool Normalize { get; set; }

        public string? Kernel { get; set; }

        public int Kw { get; set; }

        public int Kh { get; set; }

        public string? KernelX { get; set; }

        public string? KernelY { get; set; }

        public double Delta { get; set; }

        public int Dx { get; set; }

        public int Dy { get; set; }

        public double Scale { get; set; }

        public int N { get; set; }

        public double Sigma { get; set; }
    }
}
