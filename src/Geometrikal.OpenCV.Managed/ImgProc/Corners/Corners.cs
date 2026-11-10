// This implementation is ported from the OpenCV corner detectors.
//
// Original sources:
//   modules/imgproc/src/corner.cpp (cornerMinEigenVal, cornerHarris, cornerEigenValsAndVecs,
//     preCornerDetect)
//   modules/imgproc/src/cornersubpix.cpp (cornerSubPix)
//   modules/features/src/featureselect.cpp (goodFeaturesToTrack, greaterThanPtr)
//   modules/imgproc/include/opencv2/imgproc.hpp, modules/features/include/opencv2/features.hpp
//     (declarations)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of the OpenCV corner functions: <c>cornerMinEigenVal</c>, <c>cornerHarris</c>,
/// <c>cornerEigenValsAndVecs</c>, <c>goodFeaturesToTrack</c> and <c>cornerSubPix</c>.
/// </summary>
/// <remarks>
/// <para><b>Sources.</b> Single-channel <c>byte</c> or <c>float</c> images. Other element types
/// and channel counts throw <see cref="NotSupportedException"/>.</para>
/// <para><b>Gradients.</b> <c>ksize</c> 1, 3, 5 or 7 selects a Sobel aperture; −1 selects the
/// 3×3 Scharr filter. The derivatives are scaled by
/// <c>1 / (2^(ksize−1) · blockSize [· 2 for Scharr] [· 255 for byte])</c>, as OpenCV.</para>
/// <para><b>Borders.</b> <c>borderType</c> applies to the derivative filter and to the
/// <c>blockSize × blockSize</c> box sum. A region of interest is an isolated image.
/// <see cref="BorderType.Wrap"/> and <see cref="BorderType.Transparent"/> are not supported by
/// the filters.</para>
/// <para><b>Exactness.</b> The results are bit-identical to OpenCV 5.0 on its SSE3 baseline with
/// IPP disabled. <see cref="CornerHarris{T}"/> reproduces the OpenCV split between the 4-lane
/// vector loop (float arithmetic) and the scalar tail (double arithmetic for the
/// <c>k</c> term) over the continuous output.</para>
/// <para><b>Non-mutating.</b> All functions except <see cref="CornerSubPix{T}"/> return new
/// arrays. <see cref="CornerSubPix{T}"/> changes the corner array in place.</para>
/// </remarks>
public static class Corners
{
    /// <summary>
    /// Minimal eigenvalue of the gradient covariance matrix for each pixel, as OpenCV
    /// <c>cornerMinEigenVal</c>.
    /// </summary>
    /// <param name="source">Single-channel byte or float image.</param>
    /// <param name="blockSize">Neighbourhood size of the covariance sum.</param>
    /// <param name="ksize">Sobel aperture (1, 3, 5, 7) or −1 for Scharr.</param>
    /// <param name="borderType">Border extrapolation.</param>
    /// <returns>A new float matrix of the source size.</returns>
    public static Mat<float> CornerMinEigenVal<T>(Mat<T> source, int blockSize, int ksize = 3, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        using PooledMat<float> cov = Covariance(source, blockSize, ksize, borderType);
        var dst = new Mat<float>(source.Width, source.Height, 1, clear: false);
        CornerEngine.MinEigenVal(cov.Mat, dst);
        return dst;
    }

    /// <summary>
    /// Harris response <c>det(M) − k·trace(M)²</c> for each pixel, as OpenCV <c>cornerHarris</c>.
    /// </summary>
    /// <param name="source">Single-channel byte or float image.</param>
    /// <param name="blockSize">Neighbourhood size of the covariance sum.</param>
    /// <param name="ksize">Sobel aperture (1, 3, 5, 7) or −1 for Scharr.</param>
    /// <param name="k">Harris free parameter.</param>
    /// <param name="borderType">Border extrapolation.</param>
    /// <returns>A new float matrix of the source size.</returns>
    public static Mat<float> CornerHarris<T>(Mat<T> source, int blockSize, int ksize, double k, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        using PooledMat<float> cov = Covariance(source, blockSize, ksize, borderType);
        var dst = new Mat<float>(source.Width, source.Height, 1, clear: false);
        CornerEngine.Harris(cov.Mat, k, dst);
        return dst;
    }

    /// <summary>
    /// Eigenvalues and eigenvectors of the gradient covariance matrix for each pixel, as OpenCV
    /// <c>cornerEigenValsAndVecs</c>.
    /// </summary>
    /// <param name="source">Single-channel byte or float image.</param>
    /// <param name="blockSize">Neighbourhood size of the covariance sum.</param>
    /// <param name="ksize">Sobel aperture (1, 3, 5, 7) or −1 for Scharr.</param>
    /// <param name="borderType">Border extrapolation.</param>
    /// <returns>
    /// A new 6-channel float matrix with <c>(λ1, λ2, x1, y1, x2, y2)</c> per pixel: λ1 ≥ λ2 and the
    /// unit eigenvectors of λ1 and λ2.
    /// </returns>
    public static Mat<float> CornerEigenValsAndVecs<T>(Mat<T> source, int blockSize, int ksize, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        using PooledMat<float> cov = Covariance(source, blockSize, ksize, borderType);
        var dst = new Mat<float>(source.Width, source.Height, 6, clear: false);
        CornerEngine.EigenValsVecs(cov.Mat, dst);
        return dst;
    }

    /// <summary>
    /// Strongest corners, as OpenCV <c>goodFeaturesToTrack</c>.
    /// </summary>
    /// <param name="image">Single-channel byte or float image.</param>
    /// <param name="maxCorners">Maximum number of corners; 0 or less means no limit.</param>
    /// <param name="qualityLevel">
    /// Fraction of the largest response (inside the mask) below which corners are rejected; positive.
    /// </param>
    /// <param name="minDistance">
    /// Minimum Euclidean distance between returned corners. Values below 1 disable the distance test.
    /// </param>
    /// <param name="mask">Optional byte mask of the image size; zero excludes a pixel.</param>
    /// <param name="blockSize">Neighbourhood size of the covariance sum.</param>
    /// <param name="gradientSize">Sobel aperture of the derivatives (or −1 for Scharr).</param>
    /// <param name="useHarrisDetector">True for the Harris response, false for the minimal eigenvalue.</param>
    /// <param name="k">Harris free parameter.</param>
    /// <returns>Corners with integer coordinates, by decreasing response.</returns>
    /// <remarks>
    /// <para>Candidates are local maxima of the thresholded response in a 3×3 neighbourhood,
    /// excluding the outermost image rows and columns. OpenCV sorts the candidates by decreasing
    /// response; equal responses are ordered by decreasing address, so the candidate that comes
    /// later in row-major order comes first. The port uses the same total order, so the result
    /// does not depend on the sort algorithm.</para>
    /// <para>With <paramref name="minDistance"/> ≥ 1, a candidate is kept when no kept corner in
    /// the 3×3 neighbouring grid cells of size <c>cvRound(minDistance)</c> is closer than
    /// <paramref name="minDistance"/> (float squared distance compared with the double squared
    /// limit).</para>
    /// </remarks>
    public static Point2f[] GoodFeaturesToTrack<T>(
        Mat<T> image,
        int maxCorners,
        double qualityLevel,
        double minDistance,
        Mat<byte>? mask = null,
        int blockSize = 3,
        int gradientSize = 3,
        bool useHarrisDetector = false,
        double k = 0.04)
        where T : unmanaged
        => GoodFeatures.Find(image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k).Corners;

    /// <summary>
    /// <see cref="GoodFeaturesToTrack{T}"/> that also returns the response of each corner, as the
    /// OpenCV overload with <c>cornersQuality</c>.
    /// </summary>
    public static (Point2f[] Corners, float[] Quality) GoodFeaturesToTrackWithQuality<T>(
        Mat<T> image,
        int maxCorners,
        double qualityLevel,
        double minDistance,
        Mat<byte>? mask = null,
        int blockSize = 3,
        int gradientSize = 3,
        bool useHarrisDetector = false,
        double k = 0.04)
        where T : unmanaged
        => GoodFeatures.Find(image, maxCorners, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k);

    /// <summary>
    /// Refines corner locations to sub-pixel accuracy, as OpenCV <c>cornerSubPix</c>.
    /// </summary>
    /// <param name="image">Single-channel byte or float image.</param>
    /// <param name="corners">Initial corners; refined in place.</param>
    /// <param name="winSize">Half size of the search window; the window is <c>2·winSize + 1</c>.</param>
    /// <param name="zeroZone">
    /// Half size of the central region excluded from the sums, or (−1, −1) for none.
    /// </param>
    /// <param name="criteria">Iteration limit (at most 100) and squared-step tolerance.</param>
    /// <remarks>
    /// A corner keeps its last position when the next step leaves the image or the system is
    /// singular, and returns to its initial position when it moves more than the half window.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">A corner is outside the image, or the image is smaller than <c>2·winSize + 5</c>.</exception>
    public static void CornerSubPix<T>(Mat<T> image, Span<Point2f> corners, Size winSize, Size zeroZone, TermCriteria criteria)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ValidateSource(image);
        SubPixRefiner.Refine(image, corners, winSize, zeroZone, criteria);
    }

    /// <summary>
    /// <see cref="CornerHarris{T}"/> or <see cref="CornerMinEigenVal{T}"/> with the default border
    /// into <paramref name="dst"/>, which has the source size and one channel.
    /// </summary>
    internal static void CornerResponseInto<T>(Mat<T> source, int blockSize, int ksize, bool harris, double k, Mat<float> dst)
        where T : unmanaged
    {
        using PooledMat<float> cov = Covariance(source, blockSize, ksize, BorderType.Reflect101);
        if (harris)
        {
            CornerEngine.Harris(cov.Mat, k, dst);
        }
        else
        {
            CornerEngine.MinEigenVal(cov.Mat, dst);
        }
    }


    /// <summary>
    /// Feature map for corner detection, as OpenCV <c>preCornerDetect</c>:
    /// <c>(Dx²·Dyy + Dy²·Dxx − 2·Dx·Dy·Dxy) / (2^(ksize−1) [· 255 for byte])³</c>.
    /// </summary>
    /// <param name="source">Single-channel byte or float image.</param>
    /// <param name="ksize">Sobel aperture (1, 3, 5 or 7).</param>
    /// <param name="borderType">Border extrapolation of the Sobel filters.</param>
    /// <returns>A new float matrix of the source size.</returns>
    /// <remarks>
    /// The derivatives are float Sobel outputs. OpenCV computes whole 4-element blocks of each row in
    /// float and scales by <c>(float)factor</c>; the remaining elements scale the float sum by the
    /// <c>double</c> factor. The port uses the same split.
    /// </remarks>
    public static Mat<float> PreCornerDetect<T>(Mat<T> source, int ksize = 3, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException("Only byte and float images are supported.");
        }

        if (source.Channels != 1)
        {
            throw new NotSupportedException("Only single-channel images are supported.");
        }

        if (ksize is not (1 or 3 or 5 or 7))
        {
            throw new ArgumentOutOfRangeException(nameof(ksize), "The aperture must be 1, 3, 5 or 7.");
        }

        var dx = Derivatives.Sobel<T, float>(source, 1, 0, ksize, 1, 0, borderType);
        var dy = Derivatives.Sobel<T, float>(source, 0, 1, ksize, 1, 0, borderType);
        var d2x = Derivatives.Sobel<T, float>(source, 2, 0, ksize, 1, 0, borderType);
        var d2y = Derivatives.Sobel<T, float>(source, 0, 2, ksize, 1, 0, borderType);
        var dxy = Derivatives.Sobel<T, float>(source, 1, 1, ksize, 1, 0, borderType);

        double factor = 1 << (ksize - 1);
        if (typeof(T) == typeof(byte))
        {
            factor *= 255;
        }

        factor = 1.0 / (factor * factor * factor);
        float factorF = (float)factor;
        var dst = new Mat<float>(source.Width, source.Height, clear: false);
        int width = source.Width;
        int vectorEnd = width / 4 * 4;
        for (int y = 0; y < source.Height; y++)
        {
            var o = dst.GetRowSpan(y);
            var x1 = dx.GetReadOnlyRowSpan(y);
            var y1 = dy.GetReadOnlyRowSpan(y);
            var x2 = d2x.GetReadOnlyRowSpan(y);
            var y2 = d2y.GetReadOnlyRowSpan(y);
            var xy = dxy.GetReadOnlyRowSpan(y);
            for (int j = 0; j < width; j++)
            {
                float a = x1[j], b = y1[j];
                float sum = (a * a * y2[j]) + (b * b * x2[j]) - (2 * a * b * xy[j]);
                o[j] = j < vectorEnd ? sum * factorF : (float)(factor * sum);
            }
        }

        return dst;
    }

    private static PooledMat<float> Covariance<T>(Mat<T> source, int blockSize, int ksize, BorderType borderType)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateSource(source);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        if (ksize is not (-1 or 1 or 3 or 5 or 7))
        {
            throw new ArgumentOutOfRangeException(nameof(ksize), ksize, "ksize must be 1, 3, 5, 7 or -1 (Scharr).");
        }

        return CornerEngine.Covariance(source, blockSize, ksize, borderType);
    }

    internal static void ValidateSource<T>(Mat<T> source)
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"Element type {typeof(T).Name} is not supported; use byte or float.");
        }

        if (source.Channels != 1)
        {
            throw new NotSupportedException("Only single-channel images are supported.");
        }
    }
}
