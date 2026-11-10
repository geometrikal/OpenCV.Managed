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
/// Data-driven runner replaying the OpenCV (cv2 4.5.5) morphology reference
/// outputs captured in <c>TestData/morphology_golden.json</c>: erode/dilate,
/// morphologyEx, getStructuringElement, copyMakeBorder, and borderInterpolate
/// must reproduce the reference byte-for-byte.
///
/// Regenerate the golden data with <c>python tools/generate_morphology_golden.py</c>.
/// </summary>
public sealed class MorphologyGoldenTests
{
    internal static readonly IReadOnlyDictionary<string, MorphologyCase> AllCases = LoadCases();

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
    public void Morphology_MatchesOpenCvGolden(string caseName)
    {
        MorphologyCase c = AllCases[caseName];

        if (c.Op == "borderInterpolate")
        {
            int actual = Borders.BorderInterpolate(c.P, c.Len, (BorderType)c.BorderType);
            Assert.Equal(c.ExpectedInt, actual);
            return;
        }

        byte[] expected = Convert.FromBase64String(c.Expected!);
        byte[] actualBytes = c.Dtype switch
        {
            "u8" => RunCase<byte>(c),
            "u16" => RunCase<ushort>(c),
            "s16" => RunCase<short>(c),
            "f32" => RunCase<float>(c),
            "f64" => RunCase<double>(c),
            _ => throw new InvalidOperationException($"unknown dtype '{c.Dtype}'"),
        };

        if (!expected.AsSpan().SequenceEqual(actualBytes))
        {
            Assert.Fail(
                $"Golden mismatch for '{caseName}' (op={c.Op}, dtype={c.Dtype}, " +
                $"{c.Width}x{c.Height}x{c.Channels}): first difference at byte " +
                $"{FirstDiff(expected, actualBytes)} of {expected.Length}.");
        }
    }

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty()
    {
        Assert.NotEmpty(AllCases);
    }

    /// <summary>
    /// Visibility guard mirroring the contours/drawing ports: the morphology
    /// golden battery ships with an EMPTY suspect set. Any future conflict
    /// must be adjudicated and added here deliberately.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreEmpty()
    {
        string[] actual = AllCases.Values
            .Where(c => c.Suspect)
            .Select(c => c.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(Array.Empty<string>(), actual);
    }

    private static byte[] RunCase<T>(MorphologyCase c)
        where T : unmanaged
    {
        switch (c.Op)
        {
            case "erode":
            case "dilate":
            {
                Mat<T> src = BuildInput<T>(c);
                Mat<T> result = c.Op == "erode"
                    ? Morphology.Erode(src, BuildKernel(c), BuildAnchor(c), c.Iterations,
                        (BorderType)c.BorderType, BuildBorderValue(c))
                    : Morphology.Dilate(src, BuildKernel(c), BuildAnchor(c), c.Iterations,
                        (BorderType)c.BorderType, BuildBorderValue(c));
                return RawBytes(result);
            }

            case "morphologyEx":
            {
                Mat<T> src = BuildInput<T>(c);
                Mat<T> result = Morphology.MorphologyEx(src, (MorphOperation)c.MorphOp,
                    BuildKernel(c), BuildAnchor(c), c.Iterations,
                    (BorderType)c.BorderType, BuildBorderValue(c));
                return RawBytes(result);
            }

            case "getStructuringElement":
            {
                Mat<sbyte> element = Morphology.GetStructuringElement(
                    (MorphShape)c.Shape,
                    new Size(c.Ksize![0], c.Ksize[1]),
                    BuildAnchor(c));
                sbyte[] raw = element.ToArray();
                var bytes = new byte[raw.Length];
                for (int i = 0; i < raw.Length; i++)
                {
                    bytes[i] = (byte)raw[i];
                }

                return bytes;
            }

            case "copyMakeBorder":
            {
                Mat<T> src = BuildInput<T>(c);
                Mat<T> result = Borders.CopyMakeBorder(
                    src, c.Top, c.Bottom, c.Left, c.Right,
                    (BorderType)c.BorderType, BuildBorderValue(c) ?? default);
                return RawBytes(result);
            }

            default:
                throw new InvalidOperationException($"unknown op '{c.Op}'");
        }
    }

    private static Mat<T> BuildInput<T>(MorphologyCase c)
        where T : unmanaged
    {
        byte[] raw = Convert.FromBase64String(c.Input!);
        var buffer = new T[c.Width * c.Height * c.Channels];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(buffer.AsSpan()));
        return new Mat<T>(buffer, c.Width, c.Height, c.Channels);
    }

    private static Mat<sbyte>? BuildKernel(MorphologyCase c)
    {
        if (c.KernelRows is null)
        {
            return null;
        }

        int kh = c.KernelRows.Count;
        int kw = c.KernelRows[0].Count;
        var kernel = new Mat<sbyte>(kw, kh);
        for (int y = 0; y < kh; y++)
        {
            Span<sbyte> row = kernel.GetRowSpan(y);
            for (int x = 0; x < kw; x++)
            {
                row[x] = (sbyte)c.KernelRows[y][x];
            }
        }

        return kernel;
    }

    private static Point? BuildAnchor(MorphologyCase c) =>
        c.Anchor is null ? null : new Point(c.Anchor[0], c.Anchor[1]);

    private static Scalar? BuildBorderValue(MorphologyCase c) =>
        c.BorderValue is null
            ? null
            : new Scalar(c.BorderValue[0], c.BorderValue[1], c.BorderValue[2], c.BorderValue[3]);

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

    private static IReadOnlyDictionary<string, MorphologyCase> LoadCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "morphology_golden.json");
        using FileStream stream = File.OpenRead(path);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        GoldenFile file = JsonSerializer.Deserialize<GoldenFile>(stream, options)
            ?? throw new InvalidOperationException("empty golden file");
        return file.Cases.ToDictionary(c => c.Name, c => c);
    }

    internal sealed class GoldenFile
    {
        public List<MorphologyCase> Cases { get; set; } = [];
    }

    internal sealed class MorphologyCase
    {
        public string Name { get; set; } = string.Empty;

        public string Op { get; set; } = string.Empty;

        public string Dtype { get; set; } = "u8";

        public int Channels { get; set; } = 1;

        public int Width { get; set; }

        public int Height { get; set; }

        public string? Input { get; set; }

        public string? Expected { get; set; }

        public int ExpectedInt { get; set; }

        public List<List<int>>? KernelRows { get; set; }

        public List<int>? Anchor { get; set; }

        public int Iterations { get; set; } = 1;

        public int BorderType { get; set; }

        public List<double>? BorderValue { get; set; }

        public int MorphOp { get; set; }

        public int Shape { get; set; }

        public List<int>? Ksize { get; set; }

        public int Top { get; set; }

        public int Bottom { get; set; }

        public int Left { get; set; }

        public int Right { get; set; }

        public int P { get; set; }

        public int Len { get; set; }

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }
    }
}
