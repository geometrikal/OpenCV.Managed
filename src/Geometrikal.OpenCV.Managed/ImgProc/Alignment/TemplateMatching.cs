// This API is based on OpenCV template matching.
//
// Original sources:
//   modules/imgproc/include/opencv2/imgproc.hpp (declaration)
//   modules/imgproc/src/templmatch.cpp (matchTemplate, matchTemplateMask, common_matchTemplate)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Template matching, as OpenCV <c>matchTemplate</c> (CPU path, IPP off). The functions read the
/// inputs and do not change them.
/// </summary>
/// <remarks>
/// <para>Without a mask, the cross-correlation is the DFT block correlation of OpenCV
/// <c>crossCorr</c> for every template size (the OpenCV CPU path has no naive loop; the naive
/// kernels exist only in the OpenCL path). The mode-specific normalisation then uses
/// float64 integral images, as <c>common_matchTemplate</c>.</para>
/// <para>With a mask, the result follows OpenCV 5.0 <c>matchTemplateMask</c>: byte inputs are
/// converted to float, a byte mask is binarised to 0 or 1, a float mask is used as a weight, and
/// a 1-channel mask applies to all channels.</para>
/// </remarks>
public static class TemplateMatching
{
    /// <summary>
    /// Compares <paramref name="templ"/> with every overlapping window of <paramref name="image"/>.
    /// </summary>
    /// <param name="image">Image: <c>byte</c> or <c>float</c>, 1 to 4 channels.</param>
    /// <param name="templ">
    /// Template of the image type, not larger than the image. When both template dimensions are
    /// greater than or equal to the image dimensions (and no mask is given), the two inputs are
    /// swapped, as OpenCV does.
    /// </param>
    /// <param name="method">Comparison method.</param>
    /// <param name="mask">
    /// Optional byte mask of the template size, with 1 channel or the template channel count.
    /// Nonzero elements select template pixels.
    /// </param>
    /// <returns>
    /// A 1-channel <c>float</c> result of size <c>(W - w + 1) × (H - h + 1)</c>.
    /// </returns>
    public static Mat<float> MatchTemplate<T>(Mat<T> image, Mat<T> templ, TemplateMatchModes method, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateInputs(image, templ, method);
        if (mask is not null)
        {
            ValidateMask(templ, mask.Width, mask.Height, mask.Channels);
            var maskBin = Thresholding.Threshold(mask, 0, 1, ThresholdType.Binary);
            return RunMasked(image, templ, Conversions.ConvertTo<byte, float>(maskBin), method);
        }

        return MatchNoMask(image, templ, method);
    }

    /// <summary>
    /// Compares <paramref name="templ"/> with every overlapping window of <paramref name="image"/>
    /// with a float weight mask, as OpenCV <c>matchTemplate</c> with a <c>CV_32F</c> mask.
    /// </summary>
    /// <param name="image">Image: <c>byte</c> or <c>float</c>, 1 to 4 channels.</param>
    /// <param name="templ">Template of the image type, not larger than the image.</param>
    /// <param name="method">Comparison method.</param>
    /// <param name="mask">
    /// Float mask of the template size, with 1 channel or the template channel count. Each element
    /// weights the template pixel; the mask is not binarised.
    /// </param>
    /// <returns>A 1-channel <c>float</c> result of size <c>(W - w + 1) × (H - h + 1)</c>.</returns>
    public static Mat<float> MatchTemplate<T>(Mat<T> image, Mat<T> templ, TemplateMatchModes method, Mat<float> mask)
        where T : unmanaged
    {
        ValidateInputs(image, templ, method);
        ArgumentNullException.ThrowIfNull(mask);
        ValidateMask(templ, mask.Width, mask.Height, mask.Channels);
        return RunMasked(image, templ, mask, method);
    }

    private static void ValidateInputs<T>(Mat<T> image, Mat<T> templ, TemplateMatchModes method)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(templ);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException("matchTemplate supports byte and float images only.");
        }

        if (method < TemplateMatchModes.SqDiff || method > TemplateMatchModes.CCoeffNormed)
        {
            throw new ArgumentOutOfRangeException(nameof(method));
        }

        if (image.Channels > 4 || image.Channels != templ.Channels)
        {
            throw new ArgumentException("The image and template must have the same channel count (1 to 4).", nameof(templ));
        }
    }

    private static void ValidateMask<T>(Mat<T> templ, int width, int height, int channels)
        where T : unmanaged
    {
        if (width != templ.Width || height != templ.Height)
        {
            throw new ArgumentException("The mask must have the template size.", "mask");
        }

        if (channels != 1 && channels != templ.Channels)
        {
            throw new ArgumentException("The mask must have 1 channel or the template channel count.", "mask");
        }
    }

    private static Mat<float> RunMasked<T>(Mat<T> image, Mat<T> templ, Mat<float> mask, TemplateMatchModes method)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return TemplateMatchMask.Run((Mat<float>)(object)image, (Mat<float>)(object)templ, mask, method);
        }

        var src = (Mat<byte>)(object)image;
        using PooledMat<float> imageF = PooledMat<float>.Rent(src.Width, src.Height, src.Channels);
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<byte> s = src.GetReadOnlyRowSpan(y);
            Span<float> d = imageF.Mat.GetRowSpan(y);
            for (int i = 0; i < s.Length; i++)
            {
                d[i] = s[i];
            }
        }

        return TemplateMatchMask.Run(imageF.Mat, ToFloat(templ), mask, method);
    }

    private static Mat<float> ToFloat<T>(Mat<T> m)
        where T : unmanaged
        => typeof(T) == typeof(float) ? (Mat<float>)(object)m : Conversions.ConvertTo<T, float>(m);

    private static Mat<float> MatchNoMask<T>(Mat<T> image, Mat<T> templ, TemplateMatchModes method)
        where T : unmanaged
    {
        bool needSwap = image.Height < templ.Height || image.Width < templ.Width;
        if (needSwap)
        {
            if (image.Height > templ.Height || image.Width > templ.Width)
            {
                throw new ArgumentException("The template must not be larger than the image in one dimension and smaller in the other.", nameof(templ));
            }

            (image, templ) = (templ, image);
        }

        var result = new Mat<float>(image.Width - templ.Width + 1, image.Height - templ.Height + 1, 1, clear: false);
        TemplateCrossCorr.Run(image, templ, result);
        TemplateNormalization.Run(image, templ, result, method);
        return result;
    }
}
