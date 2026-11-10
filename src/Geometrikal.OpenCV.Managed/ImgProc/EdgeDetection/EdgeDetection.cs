// This implementation is ported from the OpenCV edge detection functions.
//
// Original sources:
//   modules/imgproc/src/canny.cpp (Canny)
//   modules/imgproc/src/spatialgradient.cpp (spatialGradient)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>Canny</c> and <c>spatialGradient</c>.
///
/// <para><b>Non-mutating.</b> Every operation returns new matrices; the sources are not
/// modified.</para>
///
/// <para><b>Regions of interest.</b> A region-of-interest view is processed as an isolated
/// image: pixels outside the view are never read. OpenCV's <c>Canny</c> also isolates its
/// input; its <c>spatialGradient</c> reads only the source matrix.</para>
///
/// <para><b>Threads.</b> The result does not depend on <see cref="CvParallel.MaxDegreeOfParallelism"/>.</para>
/// </summary>
public static class EdgeDetection
{
    private const int LegacyL2Flag = int.MinValue;

    /// <summary>
    /// Finds edges with the Canny algorithm, as OpenCV
    /// <c>Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient)</c>.
    /// </summary>
    /// <remarks>
    /// <para>The derivatives are <c>Sobel(image, CV_16S, 1, 0)</c> and <c>(0, 1)</c> with
    /// <paramref name="apertureSize"/> and <see cref="BorderType.Replicate"/>. Aperture 7 uses
    /// <c>scale = 1/16</c> (round half to even). Aperture -1 is the 3×3 Scharr operator.
    /// For a multi-channel image each pixel takes the gradient of the first channel with the
    /// largest magnitude.</para>
    /// <para>Thresholds are converted as in OpenCV: for aperture 7 both are divided by 16;
    /// then a larger <paramref name="threshold1"/> is swapped with
    /// <paramref name="threshold2"/>; for the L2 norm each is clamped to 32767 and squared when
    /// positive; then each is converted with x86 <c>cvFloor</c> (values at or above 2^31 give
    /// <c>int.MinValue</c>, so every local maximum becomes an edge). A pixel is a candidate when
    /// its magnitude is greater than the low integer threshold and an edge seed when it is
    /// greater than the high integer threshold.</para>
    /// <para>The magnitude is <c>|dx| + |dy|</c> (L1) or <c>dx² + dy²</c> (L2, 32-bit
    /// wrap). Non-maximum suppression quantizes the gradient direction with
    /// <c>TG22 = 13573</c> (tan 22.5° in Q15), as OpenCV does.</para>
    /// <para>Legacy flag: when <paramref name="l2Gradient"/> is false and bit 31 of
    /// <paramref name="apertureSize"/> is set, the bit is cleared and the L2 norm is used.
    /// Consequently, aperture -1 is accepted only with <paramref name="l2Gradient"/> true:
    /// with false, -1 becomes <c>0x7FFFFFFF</c> and fails the range check, as in OpenCV.</para>
    /// </remarks>
    /// <param name="image">8-bit source image with any number of channels.</param>
    /// <param name="threshold1">First hysteresis threshold.</param>
    /// <param name="threshold2">Second hysteresis threshold.</param>
    /// <param name="apertureSize">Sobel aperture: 3, 5 or 7, or -1 (Scharr, requires the L2 norm).</param>
    /// <param name="l2Gradient">True for the L2 gradient norm; false for the L1 norm.</param>
    /// <returns>A new single-channel image of the source size with values 0 and 255.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The aperture size is not supported.</exception>
    public static Mat<byte> Canny(
        Mat<byte> image, double threshold1, double threshold2, int apertureSize = 3, bool l2Gradient = false)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (!l2Gradient && (apertureSize & LegacyL2Flag) == LegacyL2Flag)
        {
            apertureSize &= ~LegacyL2Flag;
            l2Gradient = true;
        }

        if ((apertureSize & 1) == 0 || (apertureSize != -1 && (apertureSize < 3 || apertureSize > 7)))
        {
            throw new ArgumentOutOfRangeException(
                nameof(apertureSize), apertureSize, "Aperture size should be odd between 3 and 7, or -1 with the L2 norm.");
        }

        if (apertureSize == 7)
        {
            threshold1 /= 16.0;
            threshold2 /= 16.0;
        }

        (int low, int high) = IntegerThresholds(threshold1, threshold2, l2Gradient);
        return CannyEngine.FromImage(image, apertureSize, l2Gradient, low, high);
    }

    /// <summary>
    /// Finds edges with the Canny algorithm from precomputed derivatives, as OpenCV
    /// <c>Canny(dx, dy, edges, threshold1, threshold2, L2gradient)</c>.
    /// </summary>
    /// <remarks>
    /// <para>Threshold conversion, magnitude, non-maximum suppression and hysteresis are those of
    /// <see cref="Canny(Mat{byte}, double, double, int, bool)"/> without the aperture-7 division.
    /// Integer arithmetic wraps as in OpenCV: the L2 magnitude of <c>dx = dy = -32768</c> is
    /// <c>int.MinValue</c>, and the direction test treats <c>|dx| ≥ 27146</c> as vertical.</para>
    /// <para>Difference from OpenCV: the L1 magnitude uses <c>|-32768| = 32768</c> for every
    /// element. OpenCV's SIMD loop uses -32768 for elements in its vector body and 32768 in its
    /// scalar tail, so its result depends on the column and the vector width.</para>
    /// </remarks>
    /// <param name="dx">Horizontal derivative, 1 or 3 channels.</param>
    /// <param name="dy">Vertical derivative with the size and channel count of <paramref name="dx"/>.</param>
    /// <param name="threshold1">First hysteresis threshold.</param>
    /// <param name="threshold2">Second hysteresis threshold.</param>
    /// <param name="l2Gradient">True for the L2 gradient norm; false for the L1 norm.</param>
    /// <returns>A new single-channel image of the derivative size with values 0 and 255.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="dx"/> or <paramref name="dy"/> is null.</exception>
    /// <exception cref="ArgumentException">The channel count is not 1 or 3, or the matrices differ in size or channels.</exception>
    public static Mat<byte> Canny(
        Mat<short> dx, Mat<short> dy, double threshold1, double threshold2, bool l2Gradient = false)
    {
        ArgumentNullException.ThrowIfNull(dx);
        ArgumentNullException.ThrowIfNull(dy);
        if (dx.Channels != 1 && dx.Channels != 3)
        {
            throw new ArgumentException("The derivatives must have 1 or 3 channels.", nameof(dx));
        }

        if (dy.Channels != dx.Channels || dy.Width != dx.Width || dy.Height != dx.Height)
        {
            throw new ArgumentException("dy must have the size and channel count of dx.", nameof(dy));
        }

        (int low, int high) = IntegerThresholds(threshold1, threshold2, l2Gradient);
        return CannyEngine.FromDerivatives(dx, dy, l2Gradient, low, high);
    }

    /// <summary>
    /// Computes the first-order image derivatives with the 3×3 Sobel operator in one pass, as
    /// OpenCV <c>spatialGradient(src, dx, dy, ksize, borderType)</c>. The results equal
    /// <c>Sobel(src, CV_16S, 1, 0, 3)</c> and <c>Sobel(src, CV_16S, 0, 1, 3)</c> with the same
    /// border.
    /// </summary>
    /// <param name="source">Single-channel 8-bit source image.</param>
    /// <param name="dx">Receives the horizontal derivative (new matrix).</param>
    /// <param name="dy">Receives the vertical derivative (new matrix).</param>
    /// <param name="ksize">Aperture size; must be 3.</param>
    /// <param name="borderType"><see cref="BorderType.Reflect101"/> (OpenCV's default) or <see cref="BorderType.Replicate"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="ArgumentException">The source is not single-channel or the border type is not supported.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ksize"/> is not 3.</exception>
    public static void SpatialGradient(
        Mat<byte> source,
        out Mat<short> dx,
        out Mat<short> dy,
        int ksize = 3,
        BorderType borderType = BorderType.Reflect101)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels != 1)
        {
            throw new ArgumentException("spatialGradient requires a single-channel image.", nameof(source));
        }

        if (borderType != BorderType.Reflect101 && borderType != BorderType.Replicate)
        {
            throw new ArgumentException("spatialGradient supports only Reflect101 and Replicate borders.", nameof(borderType));
        }

        if (ksize != 3)
        {
            throw new ArgumentOutOfRangeException(nameof(ksize), ksize, "spatialGradient supports only ksize 3.");
        }

        dx = new Mat<short>(source.Width, source.Height, 1, clear: false);
        dy = new Mat<short>(source.Width, source.Height, 1, clear: false);
        SobelGradient3x3.Apply(source, dx, dy, borderType);
    }

    /// <summary>
    /// Swap, L2 clamp and square, then x86 <c>cvFloor</c> (canny.cpp:794-817).
    /// <c>std::min(32767.0, t)</c> returns 32767 for NaN.
    /// </summary>
    private static (int Low, int High) IntegerThresholds(double low, double high, bool l2)
    {
        if (low > high)
        {
            (low, high) = (high, low);
        }

        if (l2)
        {
            low = low < 32767.0 ? low : 32767.0;
            high = high < 32767.0 ? high : 32767.0;
            if (low > 0)
            {
                low *= low;
            }

            if (high > 0)
            {
                high *= high;
            }
        }

        return (ScalarConverter.CvFloor(low), ScalarConverter.CvFloor(high));
    }
}
