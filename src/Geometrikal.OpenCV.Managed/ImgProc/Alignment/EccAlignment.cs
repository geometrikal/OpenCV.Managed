// This API is based on the OpenCV ECC image alignment.
//
// Original sources:
//   modules/video/include/opencv2/video/tracking.hpp (declarations)
//   modules/video/src/ecc.cpp (findTransformECC, findTransformECCWithMask, computeECC)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/video/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Enhanced correlation coefficient (ECC) image alignment (Evangelidis and Psarakis, 2008), as
/// OpenCV <c>findTransformECC</c>, <c>findTransformECCWithMask</c> and <c>computeECC</c>. The
/// functions read the images and masks and do not change them.
/// </summary>
/// <remarks>
/// Images are <c>byte</c>, <c>ushort</c>, <c>float</c> or <c>double</c> with 1 or 3 channels.
/// Masks are 1-channel <c>byte</c> matrices; nonzero elements select pixels.
/// </remarks>
public static class EccAlignment
{
    /// <summary>OpenCV default criteria of <c>findTransformECC</c>: 50 iterations or a change of at most 0.001.</summary>
    public static readonly TermCriteria DefaultCriteria = new(TermCriteriaType.MaxIter | TermCriteriaType.Eps, 50, 0.001);

    /// <summary>
    /// Computes the enhanced correlation coefficient of two images, as OpenCV
    /// <c>computeECC(templateImage, inputImage, inputMask)</c>.
    /// </summary>
    /// <param name="templateImage">Template image.</param>
    /// <param name="inputImage">Image of the template type and size.</param>
    /// <param name="inputMask">Optional mask of the image size.</param>
    /// <returns>The correlation coefficient in [-1, 1].</returns>
    /// <remarks>
    /// As OpenCV, <c>byte</c> and <c>ushort</c> images subtract the mean rounded to an integer.
    /// </remarks>
    public static double ComputeEcc<T>(Mat<T> templateImage, Mat<T> inputImage, Mat<byte>? inputMask = null)
        where T : unmanaged
    {
        ValidateImages(templateImage, inputImage, sameSize: true);
        ValidateMask(inputMask, inputImage, nameof(inputMask));
        return EccEngine.ComputeEcc(templateImage, inputImage, inputMask);
    }

    /// <summary>
    /// Finds the geometric transform that maximises the ECC between a template and a warped
    /// input image, as OpenCV <c>findTransformECC(templateImage, inputImage, warpMatrix,
    /// motionType, criteria, inputMask, gaussFiltSize)</c>.
    /// </summary>
    /// <param name="templateImage">Template image.</param>
    /// <param name="inputImage">Input image of the template type. The size can differ.</param>
    /// <param name="warpMatrix">
    /// Initial estimate and result: a <c>float</c> matrix of width 3 and height 2 (height 3 for
    /// <see cref="MotionTypes.Homography"/>). It maps template coordinates to input coordinates
    /// (use it with <see cref="WarpFlags.InverseMap"/> to warp the input onto the template).
    /// The function updates it in place.
    /// </param>
    /// <param name="motionType">Motion model.</param>
    /// <param name="criteria">Termination criteria, or null for <see cref="DefaultCriteria"/>.</param>
    /// <param name="inputMask">Optional mask of the input image size.</param>
    /// <param name="gaussFiltSize">Size of the Gaussian smoothing kernel (odd and positive; 1 disables smoothing).</param>
    /// <returns>The final correlation coefficient.</returns>
    /// <exception cref="InvalidOperationException">
    /// The correlation becomes NaN, or the algorithm stops because the images are uncorrelated
    /// (OpenCV <c>StsNoConv</c>).
    /// </exception>
    public static double FindTransformEcc<T>(
        Mat<T> templateImage,
        Mat<T> inputImage,
        Mat<float> warpMatrix,
        MotionTypes motionType = MotionTypes.Affine,
        TermCriteria? criteria = null,
        Mat<byte>? inputMask = null,
        int gaussFiltSize = 5)
        where T : unmanaged
        => FindTransformEccWithMask(templateImage, inputImage, null, inputMask, warpMatrix, motionType, criteria, gaussFiltSize);

    /// <summary>
    /// Finds the geometric transform that maximises the ECC with masks for the template and the
    /// input image, as OpenCV <c>findTransformECCWithMask</c>.
    /// </summary>
    /// <param name="templateImage">Template image.</param>
    /// <param name="inputImage">Input image of the template type. The size can differ.</param>
    /// <param name="templateMask">Optional mask of the template size.</param>
    /// <param name="inputMask">Optional mask of the input image size.</param>
    /// <param name="warpMatrix">Initial estimate and result, updated in place (see <see cref="FindTransformEcc{T}"/>).</param>
    /// <param name="motionType">Motion model.</param>
    /// <param name="criteria">Termination criteria, or null for <see cref="DefaultCriteria"/>.</param>
    /// <param name="gaussFiltSize">Size of the Gaussian smoothing kernel (odd and positive).</param>
    /// <returns>The final correlation coefficient.</returns>
    /// <exception cref="InvalidOperationException">The algorithm does not converge (OpenCV <c>StsNoConv</c>).</exception>
    public static double FindTransformEccWithMask<T>(
        Mat<T> templateImage,
        Mat<T> inputImage,
        Mat<byte>? templateMask,
        Mat<byte>? inputMask,
        Mat<float> warpMatrix,
        MotionTypes motionType = MotionTypes.Affine,
        TermCriteria? criteria = null,
        int gaussFiltSize = 5)
        where T : unmanaged
    {
        ValidateImages(templateImage, inputImage, sameSize: false);
        ValidateMask(templateMask, templateImage, nameof(templateMask));
        ValidateMask(inputMask, inputImage, nameof(inputMask));
        ArgumentNullException.ThrowIfNull(warpMatrix);
        if (motionType is < MotionTypes.Translation or > MotionTypes.Homography)
        {
            throw new ArgumentOutOfRangeException(nameof(motionType));
        }

        int rows = motionType == MotionTypes.Homography ? 3 : 2;
        if (warpMatrix.Channels != 1 || warpMatrix.Width != 3 || warpMatrix.Height != rows)
        {
            throw new ArgumentException($"The warp matrix must be a 1-channel float matrix of width 3 and height {rows}.", nameof(warpMatrix));
        }

        TermCriteria c = criteria ?? DefaultCriteria;
        if ((c.Type & (TermCriteriaType.MaxIter | TermCriteriaType.Eps)) == 0)
        {
            throw new ArgumentException("The criteria must set MaxIter, Eps or both.", nameof(criteria));
        }

        if (gaussFiltSize <= 0 || gaussFiltSize % 2 == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gaussFiltSize), "The Gaussian kernel size must be odd and positive.");
        }

        return EccEngine.FindTransform(templateImage, inputImage, templateMask, inputMask, warpMatrix, motionType, c, gaussFiltSize);
    }

    private static void ValidateImages<T>(Mat<T> templ, Mat<T> input, bool sameSize)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(templ);
        ArgumentNullException.ThrowIfNull(input);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException("ECC supports byte, ushort, float and double images only.");
        }

        if ((templ.Channels != 1 && templ.Channels != 3) || templ.Channels != input.Channels)
        {
            throw new ArgumentException("The images must have the same channel count, 1 or 3.", nameof(input));
        }

        if (sameSize && (templ.Width != input.Width || templ.Height != input.Height))
        {
            throw new ArgumentException("The images must have the same size.", nameof(input));
        }
    }

    private static void ValidateMask<T>(Mat<byte>? mask, Mat<T> image, string name)
        where T : unmanaged
    {
        if (mask is not null && (mask.Channels != 1 || mask.Width != image.Width || mask.Height != image.Height))
        {
            throw new ArgumentException("The mask must be 1-channel and of the image size.", name);
        }
    }
}
