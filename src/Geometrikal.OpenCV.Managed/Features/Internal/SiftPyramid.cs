// This implementation is ported from the OpenCV SIFT detector.
//
// Original sources:
//   modules/features/src/sift.dispatch.cpp (createInitialImage, buildGaussianPyramid,
//   buildDoGPyramid)
//   modules/features/src/sift.simd.hpp (constants)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>SIFT constants of OpenCV <c>sift.simd.hpp</c> (float DoG type, fixed-point scale 1).</summary>
internal static class SiftConstants
{
    public const int DescrWidth = 4;
    public const int DescrHistBins = 8;
    public const float InitSigma = 0.5f;
    public const int ImgBorder = 5;
    public const int MaxInterpSteps = 5;
    public const int OriHistBins = 36;
    public const float OriSigFctr = 1.5f;
    public const float OriRadius = 4.5f;
    public const float OriPeakRatio = 0.8f;
    public const float DescrSclFctr = 3f;
    public const float DescrMagThr = 0.2f;
    public const float IntDescrFctr = 512f;
    public const float FltEpsilon = 1.1920929e-07f;
}

/// <summary>
/// Gaussian and difference-of-Gaussian pyramids of OpenCV SIFT. Every level is a pooled matrix
/// that the method adds to <c>rented</c>; the caller disposes the list after the last use of a level.
/// </summary>
internal static class SiftPyramid
{
    /// <summary>
    /// OpenCV <c>createInitialImage</c>: gray float image, optionally upscaled by 2, blurred to
    /// <paramref name="sigma"/>.
    /// </summary>
    public static Mat<float> CreateInitialImage(Mat<byte> image, bool doubleImageSize, float sigma, bool preciseUpscale, List<PooledMat<float>> rented)
    {
        Mat<byte> gray = image.Channels switch
        {
            1 => image,
            3 => ColorConversions.CvtColor(image, ColorConversionCode.Bgr2Gray),
            4 => ColorConversions.CvtColor(image, ColorConversionCode.Bgra2Gray),
            _ => throw new ArgumentException("The image must have 1, 3 or 4 channels.", nameof(image)),
        };
        using PooledMat<float> grayFpt = PooledMat<float>.Rent(gray.Width, gray.Height, 1);
        ToFloat(gray, grayFpt.Mat);

        if (doubleImageSize)
        {
            float sigDiff = MathF.Sqrt(StdMax((sigma * sigma) - (SiftConstants.InitSigma * SiftConstants.InitSigma * 4), 0.01f));
            using PooledMat<float> dbl = PooledMat<float>.Rent(gray.Width * 2, gray.Height * 2, 1);
            if (preciseUpscale)
            {
                var h = new Mat<double>(new[] { 0.5, 0, 0, 0, 0.5, 0 }, 3, 2);
                Geometry.WarpAffine(grayFpt.Mat, dbl.Mat, h, InterpolationFlags.Linear, WarpFlags.InverseMap, BorderType.Reflect);
            }
            else
            {
                Geometry.Resize(grayFpt.Mat, dbl.Mat, InterpolationFlags.Linear);
            }

            Mat<float> result = Rent(rented, dbl.Mat.Width, dbl.Mat.Height);
            Filters.GaussianBlurInto(dbl.Mat, result, default, sigDiff, sigDiff);
            return result;
        }
        else
        {
            float sigDiff = MathF.Sqrt(StdMax((sigma * sigma) - (SiftConstants.InitSigma * SiftConstants.InitSigma), 0.01f));
            Mat<float> result = Rent(rented, gray.Width, gray.Height);
            Filters.GaussianBlurInto(grayFpt.Mat, result, default, sigDiff, sigDiff);
            return result;
        }
    }

    /// <summary>OpenCV <c>SIFT_Impl::buildGaussianPyramid</c>.</summary>
    public static Mat<float>[] BuildGaussianPyramid(Mat<float> baseImage, int nOctaves, int nOctaveLayers, double sigma, List<PooledMat<float>> rented)
    {
        int levels = nOctaveLayers + 3;
        var sig = new double[levels];
        var pyr = new Mat<float>[nOctaves * levels];

        sig[0] = sigma;
        double k = Math.Pow(2, 1.0 / nOctaveLayers);
        for (int i = 1; i < levels; i++)
        {
            double sigPrev = Math.Pow(k, i - 1) * sigma;
            double sigTotal = sigPrev * k;
            sig[i] = Math.Sqrt((sigTotal * sigTotal) - (sigPrev * sigPrev));
        }

        for (int o = 0; o < nOctaves; o++)
        {
            for (int i = 0; i < levels; i++)
            {
                if (o == 0 && i == 0)
                {
                    pyr[0] = baseImage;
                }
                else if (i == 0)
                {
                    Mat<float> src = pyr[((o - 1) * levels) + nOctaveLayers];
                    Mat<float> dst = Rent(rented, src.Width / 2, src.Height / 2);
                    Geometry.Resize(src, dst, InterpolationFlags.Nearest);
                    pyr[o * levels] = dst;
                }
                else
                {
                    Mat<float> src = pyr[(o * levels) + i - 1];
                    Mat<float> dst = Rent(rented, src.Width, src.Height);
                    Filters.GaussianBlurInto(src, dst, default, sig[i], sig[i]);
                    pyr[(o * levels) + i] = dst;
                }
            }
        }

        return pyr;
    }

    /// <summary>OpenCV <c>SIFT_Impl::buildDoGPyramid</c>: <c>dog[o][i] = gauss[o][i + 1] − gauss[o][i]</c>.</summary>
    public static Mat<float>[] BuildDoGPyramid(Mat<float>[] gpyr, int nOctaveLayers, List<PooledMat<float>> rented)
    {
        int nOctaves = gpyr.Length / (nOctaveLayers + 3);
        var dogpyr = new Mat<float>[nOctaves * (nOctaveLayers + 2)];
        for (int a = 0; a < dogpyr.Length; a++)
        {
            Mat<float> g = gpyr[(a / (nOctaveLayers + 2)) * (nOctaveLayers + 3)];
            dogpyr[a] = Rent(rented, g.Width, g.Height);
        }

        ParallelRows.ForEachSerial(dogpyr.Length, a =>
        {
            int o = a / (nOctaveLayers + 2);
            int i = a % (nOctaveLayers + 2);
            Mat<float> src1 = gpyr[(o * (nOctaveLayers + 3)) + i];
            Mat<float> src2 = gpyr[(o * (nOctaveLayers + 3)) + i + 1];
            Arithmetic.Subtract(src2, src1, dogpyr[a]);
        });
        return dogpyr;
    }

    private static Mat<float> Rent(List<PooledMat<float>> rented, int width, int height)
    {
        PooledMat<float> m = PooledMat<float>.Rent(width, height, 1);
        rented.Add(m);
        return m.Mat;
    }

    /// <summary>Exact <c>byte</c> to <c>float</c> conversion (<c>convertTo(CV_32F)</c>).</summary>
    private static void ToFloat(Mat<byte> src, Mat<float> dst)
    {
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<byte> s = src.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            for (int x = 0; x < s.Length; x++)
            {
                d[x] = s[x];
            }
        }
    }

    private static float StdMax(float a, float b) => a < b ? b : a;
}
