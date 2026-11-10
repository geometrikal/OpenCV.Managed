namespace Geometrikal.OpenCV.Managed.Tests.Regression;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Deterministic cases on wide, odd-sized images that exercise vectorized loops and their
/// scalar tails. <c>TestData/regression_hashes.json</c> holds the output hash of each case
/// as produced by the scalar implementation that passed the cv2 golden tests.
/// </summary>
/// <remarks>
/// Regenerate the hashes only from a build that is known to be correct; see
/// <c>tools/README.md</c>.
/// </remarks>
public static class RegressionCases
{
    private static readonly (int Width, int Height)[] Sizes = [(67, 13), (130, 9)];

    public static IEnumerable<(string Name, Func<string> Run)> All()
    {
        foreach ((int w, int h) in Sizes)
        {
            foreach (int cn in new[] { 1, 3 })
            {
                string tag = $"{w}x{h}c{cn}";
                foreach (var c in MedianCases(w, h, cn, tag)) yield return c;
                foreach (var c in MorphologyCases(w, h, cn, tag)) yield return c;
                foreach (var c in FilterCases(w, h, cn, tag)) yield return c;
                foreach (var c in AdvancedFilterCases(w, h, cn, tag)) yield return c;
                foreach (var c in BorderCases(w, h, cn, tag)) yield return c;
                foreach (var c in EdgeCases(w, h, cn, tag)) yield return c;
            }
        }

        foreach (var c in RoiCases()) yield return c;
    }

    private static IEnumerable<(string, Func<string>)> MedianCases(int w, int h, int cn, string tag)
    {
        foreach (int k in new[] { 3, 5, 7 })
        {
            yield return ($"median_u8_k{k}_{tag}", () => Hash(Filters.MedianBlur(RandomBytes(w, h, cn, 1), k)));
        }

        foreach (int k in new[] { 3, 5 })
        {
            yield return ($"median_f32_k{k}_{tag}", () => Hash(Filters.MedianBlur(RandomFloats(w, h, cn, 2, specials: true), k)));
        }
    }

    private static IEnumerable<(string, Func<string>)> MorphologyCases(int w, int h, int cn, string tag)
    {
        var kernels = new (string Name, Mat<sbyte>? Kernel, Point? Anchor)[]
        {
            ("default", null, null),
            ("rect5x3", Morphology.GetStructuringElement(MorphShape.Rect, new Size(5, 3)), new Point(1, 2)),
            ("ellipse5", Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(5, 5)), null),
            ("cross3", Morphology.GetStructuringElement(MorphShape.Cross, new Size(3, 3)), null),
            ("diamond7", Morphology.GetStructuringElement(MorphShape.Diamond, new Size(7, 7)), null),
        };
        var borders = new (string Name, BorderType Type, Scalar? Value)[]
        {
            ("default", BorderType.Constant, null),
            ("const77", BorderType.Constant, Scalar.All(77)),
            ("replicate", BorderType.Replicate, null),
            ("reflect101", BorderType.Reflect101, null),
        };

        foreach (var (kn, kernel, anchor) in kernels)
        {
            foreach (var (bn, border, value) in borders)
            {
                foreach (int it in new[] { 1, 2 })
                {
                    string p = $"{kn}_{bn}_it{it}_{tag}";
                    yield return ($"erode_u8_{p}", () => Hash(Morphology.Erode(RandomBytes(w, h, cn, 3), kernel, anchor, it, border, value)));
                    yield return ($"dilate_u8_{p}", () => Hash(Morphology.Dilate(RandomBytes(w, h, cn, 4), kernel, anchor, it, border, value)));
                }
            }

            string q = $"{kn}_{tag}";
            yield return ($"erode_u16_{q}", () => Hash(Morphology.Erode(RandomUShorts(w, h, cn, 5), kernel, anchor)));
            yield return ($"dilate_s16_{q}", () => Hash(Morphology.Dilate(RandomShorts(w, h, cn, 6), kernel, anchor)));
            yield return ($"erode_f32_{q}", () => Hash(Morphology.Erode(RandomFloats(w, h, cn, 7, specials: true), kernel, anchor)));
            yield return ($"dilate_f32_{q}", () => Hash(Morphology.Dilate(RandomFloats(w, h, cn, 8, specials: true), kernel, anchor, 1, BorderType.Reflect101)));
            yield return ($"dilate_f64_{q}", () => Hash(Morphology.Dilate(RandomDoubles(w, h, cn, 9), kernel, anchor)));

            foreach (MorphOperation op in new[] { MorphOperation.Open, MorphOperation.Close, MorphOperation.Gradient, MorphOperation.TopHat, MorphOperation.BlackHat })
            {
                yield return ($"morphEx_{op}_u8_{q}", () => Hash(Morphology.MorphologyEx(RandomBytes(w, h, cn, 10), op, kernel, anchor)));
                yield return ($"morphEx_{op}_s16_{q}", () => Hash(Morphology.MorphologyEx(RandomShorts(w, h, cn, 11), op, kernel, anchor)));
            }
        }

        if (cn == 1)
        {
            var hitMiss = new Mat<sbyte>([0, 1, 0, 1, -1, 1, 0, 1, 0], 3, 3);
            yield return ($"hitMiss_{tag}", () => Hash(Morphology.MorphologyEx(RandomBinary(w, h, 12), MorphOperation.HitMiss, hitMiss)));
        }
    }

    private static IEnumerable<(string, Func<string>)> AdvancedFilterCases(int w, int h, int cn, string tag)
    {
        var k1311 = new Mat<float>(11, 13);
        Span<float> kv = k1311.GetSpan();
        for (int i = 0; i < kv.Length; i++)
        {
            kv[i] = (((i * 29) % 17) - 8) / 143f;
        }

        yield return ($"filter2d_dft_u8u8_{tag}", () => Hash(Filters.Filter2D<byte, byte>(RandomBytes(w, h, cn, 90), k1311, null, 1.5)));
        yield return ($"filter2d_dft_f32f32_{tag}", () => Hash(Filters.Filter2D<float, float>(RandomFloats(w, h, cn, 91, specials: false), k1311, new Point(0, 12), 0, BorderType.Wrap)));
        yield return ($"sqrbox_u8f32_{tag}", () => Hash(Filters.SqrBoxFilter(RandomBytes(w, h, cn, 92), new Size(5, 4))));
        yield return ($"sqrbox_f32f64_{tag}", () => Hash(Filters.SqrBoxFilter(RandomFloats(w, h, cn, 93, specials: false), new Size(7, 3))));
        yield return ($"bilateral_u8_{tag}", () => Hash(Filters.BilateralFilter(RandomBytes(w, h, cn, 94), 5, 40, 3)));
        yield return ($"bilateral_f32_{tag}", () => Hash(Filters.BilateralFilter(RandomFloats(w, h, cn, 95, specials: false), 7, 30, 2)));
        yield return ($"guided_u8_{tag}", () => Hash(EdgePreservingFilters.GuidedFilter(RandomBytes(w, h, 3, 96), RandomBytes(w, h, cn, 97), 4, 300)));
        yield return ($"stackblur_u8_{tag}", () => Hash(Filters.StackBlur(RandomBytes(w, h, cn, 98), new Size(7, 5))));
        yield return ($"stackblur_f32_{tag}", () => Hash(Filters.StackBlur(RandomFloats(w, h, cn, 99, specials: false), new Size(21, 3))));
    }

    private static IEnumerable<(string, Func<string>)> FilterCases(int w, int h, int cn, string tag)
    {
        foreach (var (name, ksize, sigma) in new[] { ("5x5s1.5", new Size(5, 5), 1.5), ("3x3s0", new Size(3, 3), 0.8), ("7x5s2", new Size(7, 5), 2.0), ("auto", new Size(0, 0), 1.2) })
        {
            yield return ($"gaussian_u8_{name}_{tag}", () => Hash(Filters.GaussianBlur(RandomBytes(w, h, cn, 20), ksize, sigma)));
            yield return ($"gaussian_f32_{name}_{tag}", () => Hash(Filters.GaussianBlur(RandomFloats(w, h, cn, 21, specials: false), ksize, sigma)));
        }

        yield return ($"gaussian_u8_replicate_{tag}", () => Hash(Filters.GaussianBlur(RandomBytes(w, h, cn, 22), new Size(5, 5), 1.5, 0, BorderType.Replicate)));
        yield return ($"gaussian_u8_constant_{tag}", () => Hash(Filters.GaussianBlur(RandomBytes(w, h, cn, 23), new Size(5, 3), 1.1, 0.7, BorderType.Constant)));

        yield return ($"blur_u8_5x5_{tag}", () => Hash(Filters.Blur(RandomBytes(w, h, cn, 30), new Size(5, 5))));
        yield return ($"blur_u8_3x7_anchor_{tag}", () => Hash(Filters.Blur(RandomBytes(w, h, cn, 31), new Size(3, 7), new Point(0, 2), BorderType.Replicate)));
        yield return ($"blur_f32_5x3_{tag}", () => Hash(Filters.Blur(RandomFloats(w, h, cn, 32, specials: false), new Size(5, 3))));
        yield return ($"box_u8s16_unnorm_{tag}", () => Hash(Filters.BoxFilter<byte, short>(RandomBytes(w, h, cn, 33), new Size(4, 6), null, false)));
        yield return ($"box_u8f32_{tag}", () => Hash(Filters.BoxFilter<byte, float>(RandomBytes(w, h, cn, 34), new Size(6, 4))));
        yield return ($"box_u8u8_big_{tag}", () => Hash(Filters.Blur(RandomBytes(w, h, cn, 35), new Size(21, 15))));

        var sharpen = new Mat<float>([0, -1, 0, -1, 5, -1, 0, -1, 0], 3, 3);
        var k53 = new Mat<float>([0.1f, -0.2f, 0.3f, 0, 0.5f, 0.25f, -0.125f, 0.05f, 0, 0.7f, -0.3f, 0.2f, 0.1f, 0, 0.4f], 5, 3);
        yield return ($"filter2d_u8u8_sharpen_{tag}", () => Hash(Filters.Filter2D<byte, byte>(RandomBytes(w, h, cn, 40), sharpen)));
        yield return ($"filter2d_u8s16_k53_{tag}", () => Hash(Filters.Filter2D<byte, short>(RandomBytes(w, h, cn, 41), k53, null, 3.5)));
        yield return ($"filter2d_u8f32_k53_{tag}", () => Hash(Filters.Filter2D<byte, float>(RandomBytes(w, h, cn, 42), k53, new Point(1, 1))));
        yield return ($"filter2d_f32f32_k53_{tag}", () => Hash(Filters.Filter2D<float, float>(RandomFloats(w, h, cn, 43, specials: false), k53, null, -1.25, BorderType.Reflect)));

        // Exact .5 sums pin round-half-to-even; huge coefficients pin the out-of-int-range cast.
        var quarter = new Mat<float>([0.25f, 0, 0.25f, 0, 0.5f, 0, 0.25f, 0, 0.25f], 3, 3);
        var huge = new Mat<float>([1e9f, -3e9f, 0, 2e9f], 2, 2);
        yield return ($"filter2d_u8u8_ties_{tag}", () => Hash(Filters.Filter2D<byte, byte>(RandomBytes(w, h, cn, 44), quarter)));
        yield return ($"filter2d_u8s16_ties_{tag}", () => Hash(Filters.Filter2D<byte, short>(RandomBytes(w, h, cn, 45), quarter, null, -100.5)));
        yield return ($"filter2d_u8u8_huge_{tag}", () => Hash(Filters.Filter2D<byte, byte>(RandomBytes(w, h, cn, 46), huge)));
        yield return ($"filter2d_u8s16_huge_{tag}", () => Hash(Filters.Filter2D<byte, short>(RandomBytes(w, h, cn, 47), huge)));
        yield return ($"laplacian_u8u8_half_{tag}", () => Hash(Derivatives.Laplacian<byte, byte>(RandomBytes(w, h, cn, 48), 3, 0.125, 64)));
        yield return ($"sep_u8s16_ties_{tag}", () => Hash(Filters.SepFilter2D<byte, short>(RandomBytes(w, h, cn, 49), [0.5f, 0.25f], [0.5f, 1f, 0.25f])));

        float[] binom5 = [1f / 16, 4f / 16, 6f / 16, 4f / 16, 1f / 16];
        float[] intK = [1, -2, 3, -2, 1];
        float[] asym = [0.3f, -0.1f, 0.25f, 0.6f];
        yield return ($"sep_u8u8_binom_{tag}", () => Hash(Filters.SepFilter2D<byte, byte>(RandomBytes(w, h, cn, 50), binom5, binom5)));
        yield return ($"sep_u8s16_int_{tag}", () => Hash(Filters.SepFilter2D<byte, short>(RandomBytes(w, h, cn, 51), intK, intK, null, 2)));
        yield return ($"sep_u8f32_asym_{tag}", () => Hash(Filters.SepFilter2D<byte, float>(RandomBytes(w, h, cn, 52), asym, binom5)));
        yield return ($"sep_u8u8_asym_{tag}", () => Hash(Filters.SepFilter2D<byte, byte>(RandomBytes(w, h, cn, 53), binom5, asym, null, 1.5)));
        yield return ($"sep_f32f32_asym_{tag}", () => Hash(Filters.SepFilter2D<float, float>(RandomFloats(w, h, cn, 54, specials: false), asym, intK)));
        yield return ($"sep_f32f32_binom_{tag}", () => Hash(Filters.SepFilter2D<float, float>(RandomFloats(w, h, cn, 55, specials: false), binom5, binom5, null, 0.5)));

        foreach (int k in new[] { 1, 3, 5 })
        {
            yield return ($"sobel_u8s16_dx_k{k}_{tag}", () => Hash(Derivatives.Sobel<byte, short>(RandomBytes(w, h, cn, 60), 1, 0, k)));
            yield return ($"sobel_u8s16_dy_k{k}_{tag}", () => Hash(Derivatives.Sobel<byte, short>(RandomBytes(w, h, cn, 61), 0, 1, k, 2, 3)));
            yield return ($"sobel_u8f32_dxy_k{k}_{tag}", () => Hash(Derivatives.Sobel<byte, float>(RandomBytes(w, h, cn, 62), 1, 1, k == 1 ? 3 : k)));
            yield return ($"sobel_f32f32_dx_k{k}_{tag}", () => Hash(Derivatives.Sobel<float, float>(RandomFloats(w, h, cn, 63, specials: false), 1, 0, k, 0.5)));
            yield return ($"sobel_u8u8_dx_k{k}_{tag}", () => Hash(Derivatives.Sobel<byte, byte>(RandomBytes(w, h, cn, 64), 1, 0, k, 0.25, 128)));
            yield return ($"laplacian_u8s16_k{k}_{tag}", () => Hash(Derivatives.Laplacian<byte, short>(RandomBytes(w, h, cn, 65), k)));
            yield return ($"laplacian_f32f32_k{k}_{tag}", () => Hash(Derivatives.Laplacian<float, float>(RandomFloats(w, h, cn, 66, specials: false), k, 0.5, 1)));
        }

        yield return ($"scharr_u8s16_dx_{tag}", () => Hash(Derivatives.Scharr<byte, short>(RandomBytes(w, h, cn, 70), 1, 0)));
        yield return ($"scharr_u8f32_dy_{tag}", () => Hash(Derivatives.Scharr<byte, float>(RandomBytes(w, h, cn, 71), 0, 1, 0.5, 2)));
        yield return ($"scharr_f32f32_dx_{tag}", () => Hash(Derivatives.Scharr<float, float>(RandomFloats(w, h, cn, 72, specials: false), 1, 0)));
    }

    private static IEnumerable<(string, Func<string>)> BorderCases(int w, int h, int cn, string tag)
    {
        foreach (BorderType type in new[] { BorderType.Constant, BorderType.Replicate, BorderType.Reflect, BorderType.Wrap, BorderType.Reflect101 })
        {
            yield return ($"copyMakeBorder_u8_{type}_{tag}", () => Hash(Borders.CopyMakeBorder(RandomBytes(w, h, cn, 80), 3, 5, 7, 2, type, new Scalar(10, 20, 30, 40))));
            yield return ($"copyMakeBorder_f32_{type}_{tag}", () => Hash(Borders.CopyMakeBorder(RandomFloats(w, h, cn, 81, specials: false), 6, 0, 0, 9, type, Scalar.All(-2.5))));
        }
    }

    private static IEnumerable<(string, Func<string>)> EdgeCases(int w, int h, int cn, string tag)
    {
        yield return ($"canny_ap3_l1_{tag}", () => Hash(EdgeDetection.Canny(RandomBytes(w, h, cn, 60), 60, 180)));
        yield return ($"canny_ap3_l2_blur_{tag}", () => Hash(EdgeDetection.Canny(Filters.GaussianBlur(RandomBytes(w, h, cn, 61), new Size(5, 5), 1.2), 10, 30, 3, true)));
        yield return ($"canny_ap5_l2_{tag}", () => Hash(EdgeDetection.Canny(RandomBytes(w, h, cn, 62), 400, 1500, 5, true)));
        yield return ($"canny_ap7_l1_{tag}", () => Hash(EdgeDetection.Canny(RandomBytes(w, h, cn, 63), 3000, 9000, 7)));
        yield return ($"canny_scharr_l2_{tag}", () => Hash(EdgeDetection.Canny(RandomBytes(w, h, cn, 64), 300, 900, -1, true)));
        yield return ($"cannyd_l2_{tag}", () => Hash(EdgeDetection.Canny(RandomShorts(w, h, cn, 65), RandomShorts(w, h, cn, 66), 5000, 20000, true)));
        yield return ($"cannyd_l1_{tag}", () => Hash(EdgeDetection.Canny(RandomShorts(w, h, cn, 67), RandomShorts(w, h, cn, 68), 5000, 20000)));
        if (cn == 1)
        {
            yield return ($"spatialGradient_reflect101_{tag}", () => HashGradient(RandomBytes(w, h, cn, 69), BorderType.Reflect101));
            yield return ($"spatialGradient_replicate_{tag}", () => HashGradient(RandomBytes(w, h, cn, 70), BorderType.Replicate));
        }

        yield return ($"sobel_u16f32_k5_{tag}", () => Hash(Derivatives.Sobel<ushort, float>(RandomUShorts(w, h, cn, 71), 1, 0, 5)));
        yield return ($"sobel_s16s16_k3_{tag}", () => Hash(Derivatives.Sobel<short, short>(RandomShorts(w, h, cn, 72), 0, 1, 3, 0.5, 3)));
        yield return ($"sobel_u8u16_k3_{tag}", () => Hash(Derivatives.Sobel<byte, ushort>(RandomBytes(w, h, cn, 73), 1, 0, 3, 1, 300)));
        yield return ($"sobel_f64f64_k5_{tag}", () => Hash(Derivatives.Sobel<double, double>(RandomDoubles(w, h, cn, 74), 1, 1, 5, 0.3, 0.1)));
        yield return ($"sobel_u8f64_k3_{tag}", () => Hash(Derivatives.Sobel<byte, double>(RandomBytes(w, h, cn, 75), 0, 1)));
        yield return ($"laplacian_s16f32_k3_{tag}", () => Hash(Derivatives.Laplacian<short, float>(RandomShorts(w, h, cn, 76), 3)));
        yield return ($"laplacian_u16u16_k5_{tag}", () => Hash(Derivatives.Laplacian<ushort, ushort>(RandomUShorts(w, h, cn, 77), 5, 0.01)));
        yield return ($"laplacian_f64f64_k3_{tag}", () => Hash(Derivatives.Laplacian<double, double>(RandomDoubles(w, h, cn, 78), 3, 0.1)));
    }

    private static string HashGradient(Mat<byte> src, BorderType border)
    {
        EdgeDetection.SpatialGradient(src, out Mat<short> dx, out Mat<short> dy, 3, border);
        return Hash(dx) + Hash(dy);
    }

    private static IEnumerable<(string, Func<string>)> RoiCases()
    {
        Mat<byte> RoiBytes(int cn, int seed) => RandomBytes(150, 30, cn, seed).Roi(11, 5, 113, 19);
        Mat<float> RoiFloats(int cn, int seed) => RandomFloats(150, 30, cn, seed, specials: true).Roi(11, 5, 113, 19);
        var ellipse = Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(5, 5));

        foreach (int cn in new[] { 1, 4 })
        {
            string tag = $"roi_c{cn}";
            yield return ($"median_u8_k3_{tag}", () => Hash(Filters.MedianBlur(RoiBytes(cn, 90), 3)));
            yield return ($"median_u8_k5_{tag}", () => Hash(Filters.MedianBlur(RoiBytes(cn, 91), 5)));
            yield return ($"median_f32_k5_{tag}", () => Hash(Filters.MedianBlur(RoiFloats(cn, 92), 5)));
            yield return ($"erode_u8_{tag}", () => Hash(Morphology.Erode(RoiBytes(cn, 93), null)));
            yield return ($"dilate_f32_ellipse_{tag}", () => Hash(Morphology.Dilate(RoiFloats(cn, 94), ellipse)));
            yield return ($"gaussian_u8_{tag}", () => Hash(Filters.GaussianBlur(RoiBytes(cn, 95), new Size(5, 5), 1.5)));
            yield return ($"sobel_u8s16_{tag}", () => Hash(Derivatives.Sobel<byte, short>(RoiBytes(cn, 96), 1, 0)));
            yield return ($"blur_u8_{tag}", () => Hash(Filters.Blur(RoiBytes(cn, 97), new Size(5, 5))));
            yield return ($"copyMakeBorder_u8_{tag}", () => Hash(Borders.CopyMakeBorder(RoiBytes(cn, 98), 2, 2, 4, 4, BorderType.Reflect101)));
            yield return ($"canny_ap3_{tag}", () => Hash(EdgeDetection.Canny(RoiBytes(cn, 99), 60, 180)));
        }
    }

    // ------------------------------------------------------------------ inputs

    private static Mat<byte> RandomBytes(int w, int h, int cn, int seed)
    {
        var random = new Random(seed);
        var data = new byte[w * h * cn];
        random.NextBytes(data);
        return new Mat<byte>(data, w, h, cn);
    }

    private static Mat<byte> RandomBinary(int w, int h, int seed)
    {
        var random = new Random(seed);
        var data = new byte[w * h];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.Next(3) == 0 ? (byte)0 : (byte)255;
        }

        return new Mat<byte>(data, w, h);
    }

    private static Mat<ushort> RandomUShorts(int w, int h, int cn, int seed)
    {
        var random = new Random(seed);
        var data = new ushort[w * h * cn];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (ushort)random.Next(65536);
        }

        return new Mat<ushort>(data, w, h, cn);
    }

    private static Mat<short> RandomShorts(int w, int h, int cn, int seed)
    {
        var random = new Random(seed);
        var data = new short[w * h * cn];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (short)random.Next(short.MinValue, short.MaxValue + 1);
        }

        return new Mat<short>(data, w, h, cn);
    }

    /// <summary>
    /// Floats in [-1000, 1000). With <paramref name="specials"/>, about 3% of values are NaN,
    /// ±infinity, +0 or -0 to pin comparison order.
    /// </summary>
    private static Mat<float> RandomFloats(int w, int h, int cn, int seed, bool specials)
    {
        var random = new Random(seed);
        var data = new float[w * h * cn];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((random.NextDouble() * 2000) - 1000);
            if (specials && random.Next(32) == 0)
            {
                data[i] = random.Next(5) switch
                {
                    0 => float.NaN,
                    1 => float.PositiveInfinity,
                    2 => float.NegativeInfinity,
                    3 => 0f,
                    _ => -0f,
                };
            }
        }

        return new Mat<float>(data, w, h, cn);
    }

    private static Mat<double> RandomDoubles(int w, int h, int cn, int seed)
    {
        var random = new Random(seed);
        var data = new double[w * h * cn];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.Next(64) == 0 ? double.NaN : (random.NextDouble() * 2000) - 1000;
        }

        return new Mat<double>(data, w, h, cn);
    }

    private static string Hash<T>(Mat<T> mat)
        where T : unmanaged
    {
        T[] data = mat.ToArray();
        byte[] header = BitConverter.GetBytes(((long)mat.Width << 40) | ((long)mat.Height << 8) | (long)mat.Channels);
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        sha.AppendData(header);
        sha.AppendData(MemoryMarshal.AsBytes(data.AsSpan()));
        return Convert.ToHexString(sha.GetHashAndReset())[..16];
    }
}
