namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

[CollectionDefinition(nameof(CvParallelCollection), DisableParallelization = true)]
public sealed class CvParallelCollection
{
}

/// <summary>
/// Row-stripe multithreading must not change any output element. Images are large enough
/// to split into several stripes.
/// </summary>
[Collection(nameof(CvParallelCollection))]
public sealed class CvParallelTests
{
    private const int Width = 517;
    private const int Height = 301;

    public static IEnumerable<object[]> Cases()
    {
        foreach (int cn in new[] { 1, 3 })
        {
            foreach (string name in Operations.Keys)
            {
                yield return [name, cn];
            }
        }
    }

    private static readonly Dictionary<string, Func<Mat<byte>, Mat<float>, byte[]>> Operations = new()
    {
        ["GaussianBlur U8 5x5"] = (b, f) => Bytes(Filters.GaussianBlur(b, new Size(5, 5), 1.5)),
        ["GaussianBlur U8 15x15"] = (b, f) => Bytes(Filters.GaussianBlur(b, new Size(15, 15), 3.0)),
        ["GaussianBlur F32 15x15"] = (b, f) => Bytes(Filters.GaussianBlur(f, new Size(15, 15), 3.0)),
        ["Blur U8 15x15"] = (b, f) => Bytes(Filters.Blur(b, new Size(15, 15))),
        ["Blur U8 21x15"] = (b, f) => Bytes(Filters.Blur(b, new Size(21, 15))),
        ["BoxFilter U8 to S16"] = (b, f) => Bytes(Filters.BoxFilter<byte, short>(b, new Size(7, 5), null, false)),
        ["Blur F32 9x9"] = (b, f) => Bytes(Filters.Blur(f, new Size(9, 9))),
        ["MedianBlur U8 3"] = (b, f) => Bytes(Filters.MedianBlur(b, 3)),
        ["MedianBlur U8 5"] = (b, f) => Bytes(Filters.MedianBlur(b, 5)),
        ["MedianBlur U8 15"] = (b, f) => Bytes(Filters.MedianBlur(b, 15)),
        ["MedianBlur F32 5"] = (b, f) => Bytes(Filters.MedianBlur(f, 5)),
        ["Filter2D U8 11x11"] = (b, f) => Bytes(Filters.Filter2D<byte, byte>(b, Kernel(11))),
        ["Filter2D F32 5x5"] = (b, f) => Bytes(Filters.Filter2D<float, float>(f, Kernel(5))),
        ["Filter2D U8 DFT 15x15"] = (b, f) => Bytes(Filters.Filter2D<byte, byte>(b, Kernel(15), null, 1.5)),
        ["Filter2D F32 DFT 31x31"] = (b, f) => Bytes(Filters.Filter2D<float, float>(f, Kernel(31), null, 0, BorderType.Wrap)),
        ["BoxFilter U8 to F32 normalized"] = (b, f) => Bytes(Filters.BoxFilter<byte, float>(b, new Size(6, 4))),
        ["SqrBoxFilter U8"] = (b, f) => Bytes(Filters.SqrBoxFilter(b, new Size(5, 5))),
        ["SqrBoxFilter F32"] = (b, f) => Bytes(Filters.SqrBoxFilter(f, new Size(9, 7))),
        ["Bilateral U8 d9"] = (b, f) => Bytes(Filters.BilateralFilter(b, 9, 40, 4)),
        ["Bilateral F32 d5"] = (b, f) => Bytes(Filters.BilateralFilter(f, 5, 10, 3)),
        ["GuidedFilter gray guide"] = (b, f) => Bytes(EdgePreservingFilters.GuidedFilter(Conversions.Split(b)[0], f, 8, 100)),
        ["GuidedFilter color guide"] = (b, f) => Bytes(EdgePreservingFilters.GuidedFilter(Color3(b), b, 4, 300)),
        ["GuidedFilter scaled"] = (b, f) => Bytes(EdgePreservingFilters.GuidedFilter(Color3(b), f, 4, 300, 0.5)),
        ["StackBlur U8 15x15"] = (b, f) => Bytes(Filters.StackBlur(b, new Size(15, 15))),
        ["StackBlur U8 5x5"] = (b, f) => Bytes(Filters.StackBlur(b, new Size(5, 5))),
        ["StackBlur F32 3x41"] = (b, f) => Bytes(Filters.StackBlur(f, new Size(3, 41))),
        ["SepFilter2D U8 to F32"] = (b, f) => Bytes(Filters.SepFilter2D<byte, float>(b, [0.1f, 0.2f, 0.4f, 0.2f, 0.1f], [0.25f, 0.5f, 0.25f])),
        ["Sobel U8 to S16 ksize 7"] = (b, f) => Bytes(Derivatives.Sobel<byte, short>(b, 1, 0, 7)),
        ["Sobel F32 dxy"] = (b, f) => Bytes(Derivatives.Sobel<float, float>(f, 1, 1)),
        ["Laplacian U8 to S16"] = (b, f) => Bytes(Derivatives.Laplacian<byte, short>(b, 3)),
        ["Sobel U16 to F32 ksize 5"] = (b, f) => Bytes(Derivatives.Sobel<ushort, float>(Conversions.ConvertTo<byte, ushort>(b, 200), 1, 0, 5)),
        ["Sobel F64 dxy"] = (b, f) => Bytes(Derivatives.Sobel<double, double>(Conversions.ConvertTo<float, double>(f, 0.1), 1, 1, 3, 0.3)),
        ["Laplacian S16 ksize 5"] = (b, f) => Bytes(Derivatives.Laplacian<short, short>(Conversions.ConvertTo<byte, short>(b, 9, -1000), 5)),
        ["Canny ap3 L1"] = (b, f) => Bytes(EdgeDetection.Canny(b, 50, 150)),
        ["Canny ap3 L2 blurred"] = (b, f) => Bytes(EdgeDetection.Canny(Filters.GaussianBlur(b, new Size(7, 7), 2.0), 8, 24, 3, true)),
        ["Canny ap5 L1 blurred"] = (b, f) => Bytes(EdgeDetection.Canny(Filters.GaussianBlur(b, new Size(7, 7), 2.0), 100, 300, 5)),
        ["Canny ap7 L2"] = (b, f) => Bytes(EdgeDetection.Canny(b, 2000, 6000, 7, true)),
        ["Canny Scharr L2 blurred"] = (b, f) => Bytes(EdgeDetection.Canny(Filters.GaussianBlur(b, new Size(5, 5), 1.5), 30, 90, -1, true)),
        ["Canny S16 derivatives L1"] = (b, f) => Bytes(CannyDerivatives(Filters.GaussianBlur(b, new Size(7, 7), 2.0))),
        ["Erode rect 15"] = (b, f) => Bytes(Morphology.Erode(b, Morphology.GetStructuringElement(MorphShape.Rect, new Size(15, 15)))),
        ["Dilate ellipse 9"] = (b, f) => Bytes(Morphology.Dilate(b, Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(9, 9)))),
        ["Gradient F32 cross 5"] = (b, f) => Bytes(Morphology.MorphologyEx(f, MorphOperation.Gradient, Morphology.GetStructuringElement(MorphShape.Cross, new Size(5, 5)))),
        ["AddWeighted U8"] = (b, f) => Bytes(Arithmetic.AddWeighted(b, 0.3, Bitwise.Not(b), 0.6, 2)),
        ["Subtract F32 masked"] = (b, f) => Bytes(MaskedSubtract(f, b)),
        ["Compare F32"] = (b, f) => Bytes(Comparison.Compare(f, 100.5, CmpType.Gt)),
        ["ConvertTo F32 to U8"] = (b, f) => Bytes(Conversions.ConvertTo<float, byte>(f, 0.7, -3)),
        ["Sum F32"] = (b, f) => BitConverter.GetBytes(Reductions.Sum(f).V0),
        ["MeanStdDev U8 masked"] = (b, f) => BitConverter.GetBytes(StdDev(b)),
        ["Norm L2 F32"] = (b, f) => BitConverter.GetBytes(Reductions.Norm(f, Arithmetic.Multiply(f, new Scalar(0.5, 0.5, 0.5)))),
        ["MeanStdDev U8"] = (b, f) => BitConverter.GetBytes(StdDevUnmasked(b)),
        ["Norm Inf U8 diff"] = (b, f) => BitConverter.GetBytes(Reductions.Norm(b, Bitwise.Not(b), NormType.L1)),
        ["CountNonZero U8"] = (b, f) => BitConverter.GetBytes(Reductions.CountNonZero(Conversions.Split(b)[0])),
        ["InRange U8"] = (b, f) => Bytes(Comparison.InRange(b, new Scalar(30, 40, 50), new Scalar(200, 210, 220))),
        ["CvtColor U8 to HSV"] = (b, f) => Bytes(ColorConversions.CvtColor(Color3(b), ColorConversionCode.Bgr2Hsv)),
        ["CvtColor F32 to Lab"] = (b, f) => Bytes(ColorConversions.CvtColor(Conversions.ConvertTo<byte, float>(Color3(b), 1.0 / 255), ColorConversionCode.Lbgr2Lab)),
        ["MinMaxLoc U8"] = (b, f) => BitConverter.GetBytes(Reductions.MinMaxLoc(Conversions.Split(b)[0]).MaxLoc.Y),
        ["Threshold U8 binary"] = (b, f) => Bytes(Thresholding.Threshold(b, 100.5, 200, ThresholdType.Binary)),
        ["Threshold F32 trunc"] = (b, f) => Bytes(Thresholding.Threshold(f, 17.25, 0, ThresholdType.Trunc)),
        ["Threshold F32 tozero masked"] = (b, f) => Bytes(MaskedThreshold(f, b)),
        ["Threshold U8 otsu"] = (b, f) => Bytes(Thresholding.Threshold(Conversions.Split(b)[0], 0, 255, ThresholdType.BinaryInv | ThresholdType.Otsu)),
        ["AdaptiveThreshold mean"] = (b, f) => Bytes(Thresholding.AdaptiveThreshold(Conversions.Split(b)[0], 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 15, 2.5)),
        ["AdaptiveThreshold gaussian"] = (b, f) => Bytes(Thresholding.AdaptiveThreshold(Conversions.Split(b)[0], 200, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 11, -1.5)),
        ["Resize U8 linear down"] = (b, f) => Bytes(Geometry.Resize(b, new Size(301, 177))),
        ["Resize U8 linear up"] = (b, f) => Bytes(Geometry.Resize(b, new Size(800, 450))),
        ["Resize U8 cubic"] = (b, f) => Bytes(Geometry.Resize(b, new Size(700, 222), interpolation: InterpolationFlags.Cubic)),
        ["Resize F32 lanczos"] = (b, f) => Bytes(Geometry.Resize(f, new Size(600, 350), interpolation: InterpolationFlags.Lanczos4)),
        ["Resize U8 area decimate"] = (b, f) => Bytes(Geometry.Resize(b, new Size(201, 111), interpolation: InterpolationFlags.Area)),
        ["Resize F32 area decimate"] = (b, f) => Bytes(Geometry.Resize(f, new Size(333, 190), interpolation: InterpolationFlags.Area)),
        ["Resize U8 area 2x"] = (b, f) => Bytes(Geometry.Resize(b, default, 0.5, 0.5, InterpolationFlags.Area)),
        ["Resize U8 linear exact"] = (b, f) => Bytes(Geometry.Resize(b, new Size(390, 260), interpolation: InterpolationFlags.LinearExact)),
        ["Resize U8 nearest exact"] = (b, f) => Bytes(Geometry.Resize(b, new Size(390, 610), interpolation: InterpolationFlags.NearestExact)),
        ["WarpAffine U8 linear"] = (b, f) => Bytes(Geometry.WarpAffine(b, WarpMatrix(), default, InterpolationFlags.Linear)),
        ["WarpAffine F32 cubic"] = (b, f) => Bytes(Geometry.WarpAffine(f, WarpMatrix(), default, InterpolationFlags.Cubic, borderType: BorderType.Reflect)),
        ["WarpPerspective U8 lanczos"] = (b, f) => Bytes(Geometry.WarpPerspective(b, PerspectiveMatrix(), default, InterpolationFlags.Lanczos4)),
        ["WarpPerspective U8 linear"] = (b, f) => Bytes(Geometry.WarpPerspective(b, PerspectiveMatrix(), default, InterpolationFlags.Linear, borderType: BorderType.Replicate)),
        ["Remap U8 linear"] = (b, f) => Bytes(Geometry.Remap(b, MapX(), MapY(), InterpolationFlags.Linear, BorderType.Wrap)),
        ["Flip U8 both"] = (b, f) => Bytes(MatrixTransforms.Flip(b, -1)),
        ["Rotate U8 clockwise"] = (b, f) => Bytes(MatrixTransforms.Rotate(b, RotateFlags.Rotate90Clockwise)),
        ["Transpose F32"] = (b, f) => Bytes(MatrixTransforms.Transpose(f)),
        ["ConnectedComponents block 8 stats"] = (b, f) => CcBytes<int>(Binary(b), 8, ConnectedComponentsAlgorithm.Default),
        ["ConnectedComponents pixel 8 stats"] = (b, f) => CcBytes<int>(Binary(b), 8, ConnectedComponentsAlgorithm.Wu),
        ["ConnectedComponents pixel 4 stats"] = (b, f) => CcBytes<int>(Binary(b), 4, ConnectedComponentsAlgorithm.Default),
        ["ConnectedComponents block 8 ushort"] = (b, f) => CcBytes<ushort>(Binary(b), 8, ConnectedComponentsAlgorithm.Grana),
        ["ConnectedComponents pixel 4 ushort"] = (b, f) => CcBytes<ushort>(Binary(b), 4, ConnectedComponentsAlgorithm.Sauf),
        ["DistanceTransform precise"] = (b, f) => Bytes(DistanceTransforms.DistanceTransform(Comparison.Compare(Conversions.Split(b)[0], 3, CmpType.Gt), DistanceTypes.L2, DistanceTransformMasks.Precise)),
        ["DistanceTransformWithLabels ccomp"] = (b, f) => DistanceLabelBytes(Binary(b)),
        ["DistanceTransform L1 byte"] = (b, f) => DistanceByteBytes(Comparison.Compare(Conversions.Split(b)[0], 3, CmpType.Gt)),
        ["CalcHist U8 1D"] = (b, f) => HistBytes(Histograms.CalcHist([b], [b.Channels - 1], null, [256], [[0, 256]])),
        ["CalcHist U8 1D masked"] = (b, f) => HistBytes(Histograms.CalcHist([b], [0], Binary(b), [40], [[3, 250]])),
        ["CalcHist U8 3D"] = (b, f) => HistBytes(Histograms.CalcHist([Color3(b)], [0, 1, 2], null, [8, 8, 8], [[0, 256], [0, 256], [0, 256]])),
        ["CalcHist U8 2D masked"] = (b, f) => HistBytes(Histograms.CalcHist([Color3(b)], [2, 0], Binary(b), [30, 32], [[0, 180], [0, 256]])),
        ["CalcHist F32"] = (b, f) => HistBytes(Histograms.CalcHist([f], [0], null, [97], [[1.5f, 30]])),
        ["CalcHist F32 2D"] = (b, f) => HistBytes(Histograms.CalcHist([f, f], [0, f.Channels], null, [9, 11], [[0, 36], [2, 30]])),
        ["EqualizeHist U8"] = (b, f) => Bytes(Histograms.EqualizeHist(Conversions.Split(b)[0])),
        ["Clahe U8"] = (b, f) => Bytes(new Clahe(3.0, new Size(8, 8)).Apply(Conversions.Split(b)[0])),
        ["Clahe U16"] = (b, f) => Bytes(new Clahe(2.0, new Size(5, 7)).Apply(Conversions.ConvertTo<byte, ushort>(Conversions.Split(b)[0], 251))),
        ["BackProject U8 2D"] = (b, f) => Bytes(Histograms.CalcBackProject([Color3(b)], [0, 1], HistOf(Color3(b)), [[0, 180], [0, 256]], 1.5)),
        ["BackProject F32 1D"] = (b, f) => Bytes(Histograms.CalcBackProject([f], [0], Histograms.CalcHist(f, 0, 50, 0, 36), [[0, 36]], 0.25)),
        ["Integral U8 sqsum"] = (b, f) => IntegralBytes(b),
        ["Integral F32 to F64"] = (b, f) => Bytes(IntegralImages.Integral<float, double>(f)),
    };

    private static Mat<byte> CannyDerivatives(Mat<byte> b)
    {
        Mat<short> dx = Derivatives.Sobel<byte, short>(b, 1, 0, 5, 1, 0, BorderType.Replicate);
        Mat<short> dy = Derivatives.Sobel<byte, short>(b, 0, 1, 5, 1, 0, BorderType.Replicate);
        return EdgeDetection.Canny(dx, dy, 150, 400);
    }

    private static byte[] DistanceByteBytes(Mat<byte> image)
    {
        var d = new Mat<byte>(image.Width, image.Height);
        DistanceTransforms.DistanceTransform(image, d, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        return Bytes(d);
    }

    private static byte[] DistanceLabelBytes(Mat<byte> image)
    {
        DistanceTransformResult r = DistanceTransforms.DistanceTransformWithLabels(image, DistanceTypes.L2, DistanceTransformMasks.Mask5);
        return [.. Bytes(r.Distances), .. Bytes(r.Labels)];
    }

    private static byte[] HistBytes(DenseHistogram h) => MemoryMarshal.AsBytes(h.Values).ToArray();

    private static DenseHistogram HistOf(Mat<byte> c3) => Histograms.CalcHist([c3], [0, 1], null, [30, 32], [[0, 180], [0, 256]]);

    private static byte[] IntegralBytes(Mat<byte> b)
    {
        Mat<int> s = IntegralImages.Integral<byte, int, double>(b, out Mat<double> q);
        return [.. Bytes(s), .. Bytes(q)];
    }

    private static Mat<double> WarpMatrix() => Geometry.GetRotationMatrix2D(new Point2f(250.5f, 150.25f), 23, 0.93);

    private static Mat<double> PerspectiveMatrix()
    {
        double[] a = WarpMatrix().ToArray();
        return new Mat<double>([a[0], a[1], a[2], a[3], a[4], a[5], 2e-4, -1e-4, 1.0], 3, 3);
    }

    private static Mat<float> MapX()
    {
        var m = new Mat<float>(Width, Height);
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                m.At(y, x) = (x * 0.9f) + (y * 0.07f) - 3.3f;
            }
        }

        return m;
    }

    private static Mat<float> MapY()
    {
        var m = new Mat<float>(Width, Height);
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                m.At(y, x) = (y * 1.1f) - (x * 0.05f) + 2.6f;
            }
        }

        return m;
    }

    /// <summary>Foreground density about 0.6, near the percolation threshold, with many merges.</summary>
    private static Mat<byte> Binary(Mat<byte> b) => Comparison.Compare(Conversions.Split(b)[0], 100, CmpType.Gt);

    private static byte[] CcBytes<TLabel>(Mat<byte> image, int connectivity, ConnectedComponentsAlgorithm algorithm)
        where TLabel : unmanaged
    {
        ConnectedComponentsStatsResult<TLabel> r = ConnectedComponentLabeling.ConnectedComponentsWithStats<TLabel>(image, connectivity, algorithm);
        return [.. BitConverter.GetBytes(r.LabelCount), .. Bytes(r.Labels), .. Bytes(r.Stats), .. Bytes(r.Centroids)];
    }

    private static Mat<float> MaskedSubtract(Mat<float> f, Mat<byte> b)
    {
        Mat<float> d = f.Clone();
        Arithmetic.Subtract(f, new Scalar(3.25, 1, 2), d, Comparison.Compare(Conversions.Split(b)[0], 128, CmpType.Lt));
        return d;
    }

    private static Mat<byte> Color3(Mat<byte> b) => b.Channels == 3 ? b : Conversions.Merge(b, Bitwise.Not(b), b);

    private static double StdDevUnmasked(Mat<byte> b)
    {
        Reductions.MeanStdDev(b, out _, out Scalar sd);
        return sd.V0;
    }

    private static Mat<float> MaskedThreshold(Mat<float> f, Mat<byte> b)
    {
        Mat<float> d = f.Clone();
        Thresholding.Threshold(f, d, 12.5, 0, ThresholdType.ToZero, Comparison.Compare(Conversions.Split(b)[0], 128, CmpType.Lt));
        return d;
    }

    private static double StdDev(Mat<byte> b)
    {
        Reductions.MeanStdDev(b, out _, out Scalar sd, Comparison.Compare(Conversions.Split(b)[0], 50, CmpType.Gt));
        return sd.V0;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Output_DoesNotDependOnThreadCount(string operation, int channels)
    {
        Mat<byte> bytes = RandomBytes(channels);
        Mat<float> floats = ToFloat(bytes);
        Func<Mat<byte>, Mat<float>, byte[]> run = Operations[operation];

        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            byte[] single = run(bytes, floats);
            CvParallel.MaxDegreeOfParallelism = 8;
            byte[] multi = run(bytes, floats);
            Assert.Equal(single, multi);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    /// <summary>
    /// Memory-bound vector loops split into stripes only above several million elements, so
    /// these cases use larger images than <see cref="Output_DoesNotDependOnThreadCount"/>.
    /// </summary>
    [Theory]
    [InlineData("Threshold U8 tozero")]
    [InlineData("Threshold U8 binary masked")]
    [InlineData("Threshold F32 binary inv")]
    [InlineData("AdaptiveThreshold compare")]
    [InlineData("Add U8 masked")]
    [InlineData("AddWeighted U8")]
    [InlineData("Compare F32")]
    [InlineData("ConvertTo F32 to U8")]
    [InlineData("Sum and Norm F32")]
    [InlineData("MinMaxLoc U8")]
    [InlineData("Resize U8 nearest")]
    [InlineData("Resize U8 area 2x")]
    [InlineData("Flip U8")]
    [InlineData("Transpose U8")]
    [InlineData("Rotate U8")]
    [InlineData("ConnectedComponents block 8")]
    [InlineData("ConnectedComponents pixel 8")]
    [InlineData("ConnectedComponents pixel 4")]
    [InlineData("ConnectedComponents block 8 ushort")]
    [InlineData("ConnectedComponents sparse ushort")]
    [InlineData("SpatialGradient U8")]
    [InlineData("Canny ap3 L1 blurred")]
    [InlineData("Canny ap5 L2 blurred")]
    [InlineData("DistanceTransform precise")]
    [InlineData("DistanceTransformWithLabels ccomp")]
    [InlineData("DistanceTransform L1 byte")]
    [InlineData("BoxFilter U8 to F32")]
    [InlineData("SqrBoxFilter U8 to F32")]
    [InlineData("Blur F32 3x3")]
    [InlineData("StackBlur U8 3x3")]
    [InlineData("StackBlur F32 3x3")]
    [InlineData("GuidedFilter gray r1")]
    [InlineData("CalcHist U8 1D")]
    [InlineData("CalcHist F32")]
    [InlineData("EqualizeHist U8")]
    [InlineData("BackProject U8 1D")]
    [InlineData("Clahe U8")]
    [InlineData("Integral U8 sqsum")]
    [InlineData("Integral F32 to F32")]
    public void LargeImage_Output_DoesNotDependOnThreadCount(string operation)
    {
        const int w = 4099, h = 2049;
        Mat<byte> bytes = new(w, h);
        new Random(5).NextBytes(bytes.GetSpan());
        Func<byte[]> run = operation switch
        {
            "Threshold U8 tozero" => () => Bytes(Thresholding.Threshold(bytes, 99.5, 0, ThresholdType.ToZero)),
            "Threshold U8 binary masked" => () =>
            {
                Mat<byte> d = new(w, h);
                Thresholding.Threshold(bytes, d, 140, 7, ThresholdType.Binary, Comparison.Compare(bytes, 64, CmpType.Gt));
                return Bytes(d);
            },
            "Threshold F32 binary inv" => () => Bytes(Thresholding.Threshold(ToFloat(bytes), 20.5, 3, ThresholdType.BinaryInv)),
            "Add U8 masked" => () =>
            {
                Mat<byte> d = Bitwise.Not(bytes);
                Arithmetic.Add(bytes, new Scalar(77), d, Comparison.Compare(bytes, 100, CmpType.Gt));
                return Bytes(d);
            },
            "AddWeighted U8" => () => Bytes(Arithmetic.AddWeighted(bytes, 0.3, Bitwise.Not(bytes), 0.6, 2)),
            "Compare F32" => () => Bytes(Comparison.Compare(ToFloat(bytes), 100.5, CmpType.Gt)),
            "ConvertTo F32 to U8" => () => Bytes(Conversions.ConvertTo<float, byte>(ToFloat(bytes), 0.7, -3)),
            "Sum and Norm F32" => () =>
            [
                .. BitConverter.GetBytes(Reductions.Sum(ToFloat(bytes)).V0),
                .. BitConverter.GetBytes(Reductions.Norm(ToFloat(bytes), NormType.L2)),
            ],
            "MinMaxLoc U8" => () => BitConverter.GetBytes(Reductions.MinMaxLoc(bytes).MaxLoc.Y),
            "Resize U8 nearest" => () => Bytes(Geometry.Resize(bytes, new Size(5001, 3001), interpolation: InterpolationFlags.Nearest)),
            "Resize U8 area 2x" => () => Bytes(Geometry.Resize(bytes, default, 0.5, 0.5, InterpolationFlags.Area)),
            "Flip U8" => () => Bytes(MatrixTransforms.Flip(bytes, -1)),
            "Transpose U8" => () => Bytes(MatrixTransforms.Transpose(bytes)),
            "Rotate U8" => () => Bytes(MatrixTransforms.Rotate(bytes, RotateFlags.Rotate90Counterclockwise)),
            "ConnectedComponents block 8" => () => CcBytes<int>(Comparison.Compare(bytes, 100, CmpType.Gt), 8, ConnectedComponentsAlgorithm.Default),
            "ConnectedComponents pixel 8" => () => CcBytes<int>(Comparison.Compare(bytes, 100, CmpType.Gt), 8, ConnectedComponentsAlgorithm.Sauf),
            "ConnectedComponents pixel 4" => () => CcBytes<int>(Comparison.Compare(bytes, 120, CmpType.Gt), 4, ConnectedComponentsAlgorithm.Wu),
            "ConnectedComponents block 8 ushort" => () => CcBytes<ushort>(Comparison.Compare(bytes, 15, CmpType.Gt), 8, ConnectedComponentsAlgorithm.Bbdt),
            "ConnectedComponents sparse ushort" => () => CcBytes<ushort>(Comparison.Compare(bytes, 253, CmpType.Gt), 4, ConnectedComponentsAlgorithm.Default),
            "SpatialGradient U8" => () =>
            {
                EdgeDetection.SpatialGradient(bytes, out Mat<short> dx, out Mat<short> dy, 3, BorderType.Replicate);
                return [.. Bytes(dx), .. Bytes(dy)];
            },
            "Canny ap3 L1 blurred" => () => Bytes(EdgeDetection.Canny(Filters.GaussianBlur(bytes, new Size(9, 9), 2.5), 6, 20)),
            "Canny ap5 L2 blurred" => () => Bytes(EdgeDetection.Canny(Filters.GaussianBlur(bytes, new Size(9, 9), 2.5), 60, 200, 5, true)),
            "DistanceTransform precise" => () => Bytes(DistanceTransforms.DistanceTransform(Comparison.Compare(bytes, 1, CmpType.Gt), DistanceTypes.L2, DistanceTransformMasks.Precise)),
            "DistanceTransformWithLabels ccomp" => () => DistanceLabelBytes(Comparison.Compare(bytes, 120, CmpType.Gt)),
            "DistanceTransform L1 byte" => () => DistanceByteBytes(Comparison.Compare(bytes, 0, CmpType.Gt)),
            "BoxFilter U8 to F32" => () => Bytes(Filters.BoxFilter<byte, float>(bytes, new Size(3, 3))),
            "SqrBoxFilter U8 to F32" => () => Bytes(Filters.SqrBoxFilter(bytes, new Size(3, 3))),
            "Blur F32 3x3" => () => Bytes(Filters.Blur(ToFloat(bytes), new Size(3, 3))),
            "StackBlur U8 3x3" => () => Bytes(Filters.StackBlur(bytes, new Size(3, 3))),
            "StackBlur F32 3x3" => () => Bytes(Filters.StackBlur(ToFloat(bytes), new Size(3, 3))),
            "GuidedFilter gray r1" => () => Bytes(EdgePreservingFilters.GuidedFilter(bytes, bytes, 1, 50)),
            "CalcHist U8 1D" => () => HistBytes(Histograms.CalcHist(bytes, 0, 256, 0, 256)),
            "CalcHist F32" => () => HistBytes(Histograms.CalcHist(ToFloat(bytes), 0, 1000, 0.5f, 36)),
            "EqualizeHist U8" => () => Bytes(Histograms.EqualizeHist(bytes)),
            "BackProject U8 1D" => () => Bytes(Histograms.CalcBackProject([bytes], [0], Histograms.CalcHist(bytes, 0, 64, 0, 256), [[0, 256]], 0.01)),
            "Clahe U8" => () => Bytes(new Clahe(4.0, new Size(16, 9)).Apply(bytes)),
            "Integral U8 sqsum" => () => IntegralBytes(bytes),
            "Integral F32 to F32" => () =>
            {
                Mat<float> s = IntegralImages.Integral<float, float, float>(ToFloat(bytes), out Mat<float> q);
                return [.. Bytes(s), .. Bytes(q)];
            },
            _ => () => Bytes(Thresholding.AdaptiveThreshold(bytes, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 3, 0)),
        };

        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            byte[] single = run();
            CvParallel.MaxDegreeOfParallelism = 8;
            byte[] multi = run();
            Assert.Equal(single, multi);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    /// <summary>
    /// The parallel <c>ushort</c> scan must throw exactly when the sequential OpenCV scan creates
    /// provisional label 65535. 258 vertical bars cross every stripe boundary; the stripe scans
    /// create extra labels for them that the sequential scan does not create. The bar at column
    /// 517 occupies only the right column of its blocks, next to the bar at column 515.
    /// </summary>
    [Theory]
    [InlineData(4, ConnectedComponentsAlgorithm.Default)]
    [InlineData(8, ConnectedComponentsAlgorithm.Default)]
    [InlineData(8, ConnectedComponentsAlgorithm.Wu)]
    public void ConnectedComponents_UShortOverflow_DoesNotDependOnThreadCount(int connectivity, ConnectedComponentsAlgorithm algorithm)
    {
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            foreach (int threads in new[] { 1, 8 })
            {
                CvParallel.MaxDegreeOfParallelism = threads;
                ConnectedComponentsResult<ushort> ok = ConnectedComponentLabeling.ConnectedComponents<ushort>(BarsAndDots(65534 - 258), connectivity, algorithm);
                Assert.Equal(65535, ok.LabelCount);
                Assert.Throws<OverflowException>(() => ConnectedComponentLabeling.ConnectedComponents<ushort>(BarsAndDots(65535 - 258), connectivity, algorithm));
            }
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    /// <summary>Bars at even columns 0..510 and at columns 515 and 517 over all rows; isolated pixels at even positions from column 520.</summary>
    private static Mat<byte> BarsAndDots(int dots)
    {
        var m = new Mat<byte>(2048, 512);
        for (int y = 0; y < 512; y++)
        {
            for (int x = 0; x <= 510; x += 2)
            {
                m.At(y, x) = 1;
            }

            m.At(y, 515) = 1;
            m.At(y, 517) = 1;
        }

        int k = 0;
        for (int y = 0; y < 512 && k < dots; y += 2)
        {
            for (int x = 520; x < 2048 && k < dots; x += 2, k++)
            {
                m.At(y, x) = 1;
            }
        }

        Assert.Equal(dots, k);
        return m;
    }

    /// <summary>
    /// Spectral transforms split into stripes only on large matrices: the row and column batch
    /// passes, the conjugate completion of complex output and the spectrum product.
    /// </summary>
    [Theory]
    [InlineData("Dft real")]
    [InlineData("Dft inverse CCS")]
    [InlineData("Dft 2-channel inverse")]
    [InlineData("Dft F64 real")]
    [InlineData("Dct 2D")]
    [InlineData("Idct F64 rows")]
    [InlineData("Dft complex output")]
    [InlineData("Dft rows complex output")]
    [InlineData("MulSpectrums CCS")]
    [InlineData("MulSpectrums complex conj")]
    [InlineData("MulSpectrums rows")]
    public void Spectral_LargeImage_Output_DoesNotDependOnThreadCount(string operation)
    {
        const int w = 1536, h = 1200;
        var x = new Mat<float>(w, h);
        var random = new Random(9);
        foreach (ref float v in x.GetSpan())
        {
            v = (float)random.NextDouble();
        }

        Mat<float> ccs = SpectralTransforms.Dft(x);
        Mat<float> cplx = Conversions.Merge(x, ccs);
        Func<byte[]> run = operation switch
        {
            "Dft real" => () => Bytes(SpectralTransforms.Dft(x)),
            "Dft inverse CCS" => () => Bytes(SpectralTransforms.Idft(x, DftFlags.Scale)),
            "Dft 2-channel inverse" => () => Bytes(SpectralTransforms.Idft(cplx, DftFlags.Scale)),
            "Dft F64 real" => () => Bytes(SpectralTransforms.Dft(Conversions.ConvertTo<float, double>(x))),
            "Dct 2D" => () => Bytes(SpectralTransforms.Dct(x)),
            "Idct F64 rows" => () => Bytes(SpectralTransforms.Idct(Conversions.ConvertTo<float, double>(x), DctFlags.Rows)),
            "Dft complex output" => () => Bytes(SpectralTransforms.Dft(x, DftFlags.ComplexOutput)),
            "Dft rows complex output" => () => Bytes(SpectralTransforms.Dft(x, DftFlags.ComplexOutput | DftFlags.Rows)),
            "MulSpectrums CCS" => () => Bytes(SpectralTransforms.MulSpectrums(ccs, x)),
            "MulSpectrums complex conj" => () => Bytes(SpectralTransforms.MulSpectrums(cplx, cplx, DftFlags.None, true)),
            _ => () => Bytes(SpectralTransforms.MulSpectrums(ccs, x, DftFlags.Rows)),
        };

        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            byte[] single = run();
            CvParallel.MaxDegreeOfParallelism = 8;
            byte[] multi = run();
            Assert.Equal(single, multi);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void MaxDegreeOfParallelism_BelowOne_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => CvParallel.MaxDegreeOfParallelism = 0);
    }

    private static Mat<byte> RandomBytes(int channels)
    {
        var m = new Mat<byte>(Width, Height, channels);
        var random = new Random(channels);
        for (int y = 0; y < Height; y++)
        {
            random.NextBytes(m.GetRowSpan(y));
        }

        return m;
    }

    private static Mat<float> ToFloat(Mat<byte> source)
    {
        var m = new Mat<float>(source.Width, source.Height, source.Channels);
        for (int y = 0; y < source.Height; y++)
        {
            ReadOnlySpan<byte> s = source.GetReadOnlyRowSpan(y);
            Span<float> d = m.GetRowSpan(y);
            for (int i = 0; i < s.Length; i++)
            {
                d[i] = s[i] / 7f;
            }
        }

        return m;
    }

    private static Mat<float> Kernel(int size)
    {
        var values = new float[size * size];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = ((i * 37) % 11 - 5) / 61f;
        }

        return new Mat<float>(values, size, size);
    }

    private static byte[] Bytes<T>(Mat<T> m)
        where T : unmanaged
    {
        var result = new List<byte>();
        for (int y = 0; y < m.Height; y++)
        {
            result.AddRange(MemoryMarshal.AsBytes(m.GetReadOnlyRowSpan(y)).ToArray());
        }

        return result.ToArray();
    }
}
