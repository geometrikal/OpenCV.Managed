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
/// Replays the cv2 4.5.5 reference outputs (IPP off) in <c>TestData/distance_golden.json</c>
/// through <see cref="DistanceTransforms"/> and <see cref="Segmentation.Watershed"/>. Outputs must
/// match byte for byte. Cases with stored outputs also run on ROI inputs and ROI destinations.
/// Regenerate with <c>python tools/generate_distance_golden.py</c>.
/// </summary>
public sealed class DistanceGoldenTests
{
    private static readonly GoldenFile Data = Load();

    public static IEnumerable<object[]> CaseNames()
        => Data.Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.NotEmpty(Data.Cases);

    /// <summary>
    /// The suspects are exactly the chamfer and precise cases on images without zero pixels. cv2
    /// 4.5.5 gives 8192 or 31622776 there; OpenCV 5.0 gives (UINT_MAX - w) / 65536 or 65536
    /// (<see cref="DistanceOpenCv5DifferenceTests"/>).
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_AreTheNoZeroPixelImages()
    {
        string[] expected = Data.Cases
            .Where(c => c.Op is "dist" or "labels" && !LoadBytes(Data.Arrays[c.Input]).ToArray().Contains((byte)0))
            .Select(c => c.Name)
            .ToArray();
        Assert.Equal(60, expected.Length);
        Assert.Equal(expected, Data.Cases.Where(c => c.Suspect).Select(c => c.Name).ToArray());
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Distance_MatchesOpenCvGolden(string name)
    {
        DistCase c = Data.Cases.Single(x => x.Name == name);
        switch (c.Op)
        {
            case "dist":
                RunDist(c);
                break;
            case "dist8u":
                RunDist8u(c);
                break;
            case "labels":
                RunLabels(c);
                break;
            default:
                RunWatershed(c);
                break;
        }
    }

    private static void RunDist(DistCase c)
    {
        Mat<byte> src = LoadBytes(Data.Arrays[c.Input]);
        var type = (DistanceTypes)c.DistanceType;
        var mask = (DistanceTransformMasks)c.MaskSize;
        Compare(c, "dist", c.Dist, c.DistSha, Bytes(DistanceTransforms.DistanceTransform(src, type, mask)));
        if (c.Dist is null)
        {
            return;
        }

        Mat<byte> roi = RoiCopy(src, 1);
        Mat<float> dst = new Mat<float>(src.Width + 3, src.Height + 2).Roi(2, 1, src.Width, src.Height);
        DistanceTransforms.DistanceTransform(roi, dst, type, mask);
        Compare(c, "ROI dist", c.Dist, null, Bytes(dst));
    }

    private static void RunDist8u(DistCase c)
    {
        Mat<byte> src = LoadBytes(Data.Arrays[c.Input]);
        var type = (DistanceTypes)c.DistanceType;
        var mask = (DistanceTransformMasks)c.MaskSize;
        var full = new Mat<byte>(src.Width, src.Height);
        DistanceTransforms.DistanceTransform(src, full, type, mask);
        Compare(c, "dist", c.Dist, c.DistSha, Bytes(full));
        if (c.Dist is null)
        {
            return;
        }

        Mat<byte> roi = RoiCopy(src, 1);
        Mat<byte> dst = new Mat<byte>(src.Width + 3, src.Height + 2).Roi(2, 1, src.Width, src.Height);
        DistanceTransforms.DistanceTransform(roi, dst, type, mask);
        Compare(c, "ROI dist", c.Dist, null, Bytes(dst));

        // In place: the destination is the source.
        Mat<byte> inPlace = RoiCopy(src, 7);
        DistanceTransforms.DistanceTransform(inPlace, inPlace, type, mask);
        Compare(c, "in-place dist", c.Dist, null, Bytes(inPlace));
    }

    private static void RunLabels(DistCase c)
    {
        Mat<byte> src = LoadBytes(Data.Arrays[c.Input]);
        var type = (DistanceTypes)c.DistanceType;
        var mask = (DistanceTransformMasks)c.MaskSize;
        var labelType = (DistanceTransformLabelTypes)c.LabelType;
        DistanceTransformResult r = DistanceTransforms.DistanceTransformWithLabels(src, type, mask, labelType);
        Compare(c, "dist", c.Dist, c.DistSha, Bytes(r.Distances));
        Compare(c, "labels", c.Labels, c.LabelsSha, Bytes(r.Labels));
        if (c.Dist is null)
        {
            return;
        }

        Mat<byte> roi = RoiCopy(src, 0);
        Mat<float> dst = new Mat<float>(src.Width + 3, src.Height + 2).Roi(2, 1, src.Width, src.Height);
        Mat<int> labels = new Mat<int>(src.Width + 1, src.Height + 4).Roi(1, 3, src.Width, src.Height);
        DistanceTransforms.DistanceTransformWithLabels(roi, dst, labels, type, mask, labelType);
        Compare(c, "ROI dist", c.Dist, null, Bytes(dst));
        Compare(c, "ROI labels", c.Labels, null, Bytes(labels));
    }

    private static void RunWatershed(DistCase c)
    {
        ArrayData ia = Data.Arrays[c.Input];
        var image = new Mat<byte>(Convert.FromBase64String(ia.Data), ia.Width, ia.Height, 3);
        Mat<int> markers = LoadInts(Data.Arrays[c.Markers!.Value]);
        Mat<int> work = markers.Clone();
        Segmentation.Watershed(image, work);
        Compare(c, "markers", c.Labels, c.LabelsSha, Bytes(work));
        if (c.Labels is null)
        {
            return;
        }

        var imageParent = new Mat<byte>(image.Width + 2, image.Height + 3, 3);
        imageParent.Fill(77);
        Mat<byte> imageRoi = imageParent.Roi(1, 2, image.Width, image.Height);
        image.CopyTo(imageRoi);
        var markerParent = new Mat<int>(image.Width + 5, image.Height + 2);
        markerParent.Fill(123);
        Mat<int> markerRoi = markerParent.Roi(3, 1, image.Width, image.Height);
        markers.CopyTo(markerRoi);
        Segmentation.Watershed(imageRoi, markerRoi);
        Compare(c, "ROI markers", c.Labels, null, Bytes(markerRoi));
        Assert.Equal(123, markerParent.At(0, 0));
        Assert.Equal(123, markerParent.At(image.Height + 1, image.Width + 4));
    }

    /// <summary>Copies <paramref name="m"/> into an ROI with odd offsets, padded with <paramref name="fill"/>.</summary>
    private static Mat<byte> RoiCopy(Mat<byte> m, byte fill)
    {
        var parent = new Mat<byte>(m.Width + 5, m.Height + 3);
        parent.Fill(fill);
        Mat<byte> roi = parent.Roi(3, 2, m.Width, m.Height);
        m.CopyTo(roi);
        return roi;
    }

    private static void Compare(DistCase c, string what, int? index, string? sha, byte[] actual)
    {
        if (sha is not null)
        {
            Assert.True(sha == Convert.ToHexStringLower(SHA256.HashData(actual)), $"{what} hash mismatch for '{c.Name}'.");
            return;
        }

        byte[] expected = Convert.FromBase64String(Data.Arrays[index!.Value].Data);
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            int i = 0;
            while (i < expected.Length && i < actual.Length && expected[i] == actual[i])
            {
                i++;
            }

            Assert.Fail($"Golden {what} mismatch for '{c.Name}' at byte {i} of {expected.Length} (actual length {actual.Length}).");
        }
    }

    private static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
        => MemoryMarshal.AsBytes(m.ToArray().AsSpan()).ToArray();

    private static Mat<byte> LoadBytes(ArrayData a)
        => new(Convert.FromBase64String(a.Data), a.Width, a.Height);

    private static Mat<int> LoadInts(ArrayData a)
        => new(MemoryMarshal.Cast<byte, int>(Convert.FromBase64String(a.Data)).ToArray(), a.Width, a.Height);

    private static GoldenFile Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "distance_golden.json");
        return JsonSerializer.Deserialize<GoldenFile>(File.ReadAllText(path), new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
        })!;
    }

    internal sealed class GoldenFile
    {
        public ArrayData[] Arrays { get; set; } = [];

        public DistCase[] Cases { get; set; } = [];
    }

    internal sealed class ArrayData
    {
        public string Dtype { get; set; } = "";

        public int Width { get; set; }

        public int Height { get; set; }

        public int Channels { get; set; }

        public string Data { get; set; } = "";
    }

    internal sealed class DistCase
    {
        public string Name { get; set; } = "";

        public string Op { get; set; } = "";

        public int Input { get; set; }

        public int? Markers { get; set; }

        public int DistanceType { get; set; }

        public int MaskSize { get; set; }

        public int LabelType { get; set; }

        public bool Suspect { get; set; }

        public string? SuspectReason { get; set; }

        public int? Dist { get; set; }

        public string? DistSha { get; set; }

        public int? Labels { get; set; }

        public string? LabelsSha { get; set; }
    }
}
