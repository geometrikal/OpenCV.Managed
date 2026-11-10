// This implementation is ported from the OpenCV colour conversion dispatcher.
//
// Original sources:
//   modules/imgproc/src/color.cpp (cvtColor)
//   modules/imgproc/src/color.hpp (swapBlue, CvtHelper channel and depth checks)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed port of OpenCV's <c>cvtColor</c> for RGB/BGR reordering, gray, HSV, CIE L*a*b*,
/// CIE XYZ and YCrCb.
/// </summary>
/// <remarks>
/// <para><b>Non-mutating.</b> The result is a new matrix; the source is not modified.</para>
/// <para><b>Element types.</b> <c>byte</c>, <c>ushort</c> and <c>float</c>. HSV and Lab
/// support <c>byte</c> and <c>float</c> only, as in OpenCV. Float colour values are in
/// [0, 1]; float hue is in degrees; float L*a*b* uses L in [0, 100].</para>
/// <para><b>Channels.</b> Colour sources have 3 or 4 channels (the fourth is ignored, except
/// that RGBA-to-BGRA reordering keeps it). Conversions to colour produce 3 channels by default
/// or 4 when <c>dstCn</c> is 4; the added alpha is 255, 65535 or 1.0.</para>
/// <para><b>Bit-exactness.</b> Results match OpenCV built for 128-bit vectors (SSE baseline).
/// Some OpenCV kernels use different float formulas in the vector body and the scalar tail of a
/// row; this port applies the same split per row. The split points depend on the vector width,
/// so OpenCV builds with 256-bit vectors (AVX2) give different results for those kernels.</para>
/// </remarks>
public static class ColorConversions
{
    /// <summary>Converts an image from one colour space to another, as OpenCV <c>cvtColor</c>.</summary>
    /// <typeparam name="T">Element type: <c>byte</c>, <c>ushort</c> or <c>float</c>.</typeparam>
    /// <param name="source">Source image.</param>
    /// <param name="code">Conversion code.</param>
    /// <param name="dstCn">Destination channel count, or 0 for the default of the code.</param>
    /// <returns>A new converted image.</returns>
    /// <exception cref="ArgumentException">The source channel count does not fit the code, or <paramref name="dstCn"/> is invalid.</exception>
    /// <exception cref="NotSupportedException">The element type is not supported for the code.</exception>
    public static Mat<T> CvtColor<T>(Mat<T> source, ColorConversionCode code, int dstCn = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"cvtColor does not support element type '{typeof(T).Name}'.");
        }

        int scn = source.Channels;
        int bidx = SwapBlue(code) ? 2 : 0;
        switch (code)
        {
            case ColorConversionCode.Bgr2Bgra:
            case ColorConversionCode.Bgra2Bgr:
            case ColorConversionCode.Bgr2Rgba:
            case ColorConversionCode.Rgba2Bgr:
            case ColorConversionCode.Bgr2Rgb:
            case ColorConversionCode.Bgra2Rgba:
            {
                RequireChannels(scn, 3, 4);
                int dcn = dstCn > 0 ? dstCn
                    : code is ColorConversionCode.Bgr2Bgra or ColorConversionCode.Bgr2Rgba or ColorConversionCode.Bgra2Rgba ? 4 : 3;
                RequireDstCn(dcn);
                return Run(source, dcn, new SwapRow<T>(scn, dcn, bidx));
            }

            case ColorConversionCode.Bgr2Gray:
            case ColorConversionCode.Rgb2Gray:
            case ColorConversionCode.Bgra2Gray:
            case ColorConversionCode.Rgba2Gray:
                RequireChannels(scn, 3, 4);
                return typeof(T) == typeof(float)
                    ? As<T>(Run(As<float>(source), 1, new GrayFloatRow(scn, bidx)))
                    : Run(source, 1, new GrayIntRow<T>(scn, bidx));

            case ColorConversionCode.Gray2Bgr:
            case ColorConversionCode.Gray2Bgra:
            {
                RequireChannels(scn, 1, 1);
                int dcn = dstCn > 0 ? dstCn : code == ColorConversionCode.Gray2Bgra ? 4 : 3;
                RequireDstCn(dcn);
                return Run(source, dcn, new GrayToColorRow<T>(dcn));
            }

            case ColorConversionCode.Bgr2Hsv:
            case ColorConversionCode.Rgb2Hsv:
            case ColorConversionCode.Bgr2HsvFull:
            case ColorConversionCode.Rgb2HsvFull:
            {
                RequireChannels(scn, 3, 4);
                RequireByteOrFloat<T>(code);
                int hrange = code is ColorConversionCode.Bgr2HsvFull or ColorConversionCode.Rgb2HsvFull ? 256 : 180;
                return typeof(T) == typeof(byte)
                    ? As<T>(Run(As<byte>(source), 3, new HsvByteRow(scn, bidx, hrange)))
                    : As<T>(Run(As<float>(source), 3, new HsvFloatRow(scn, bidx)));
            }

            case ColorConversionCode.Hsv2Bgr:
            case ColorConversionCode.Hsv2Rgb:
            case ColorConversionCode.Hsv2BgrFull:
            case ColorConversionCode.Hsv2RgbFull:
            {
                RequireChannels(scn, 3, 3);
                RequireByteOrFloat<T>(code);
                int dcn = DefaultColorDcn(dstCn);
                int hrange = code is ColorConversionCode.Hsv2BgrFull or ColorConversionCode.Hsv2RgbFull ? 255 : 180;
                return typeof(T) == typeof(byte)
                    ? As<T>(Run(As<byte>(source), dcn, new HsvToRgbByteRow(dcn, bidx, hrange)))
                    : As<T>(Run(As<float>(source), dcn, new HsvToRgbFloatRow(dcn, bidx)));
            }

            case ColorConversionCode.Bgr2Lab:
            case ColorConversionCode.Rgb2Lab:
            case ColorConversionCode.Lbgr2Lab:
            case ColorConversionCode.Lrgb2Lab:
            {
                RequireChannels(scn, 3, 4);
                RequireByteOrFloat<T>(code);
                bool srgb = code is ColorConversionCode.Bgr2Lab or ColorConversionCode.Rgb2Lab;
                return typeof(T) == typeof(byte)
                    ? As<T>(Run(As<byte>(source), 3, new LabByteRow(scn, bidx, srgb)))
                    : As<T>(Run(As<float>(source), 3, new LabFloatRow(scn, bidx, srgb)));
            }

            case ColorConversionCode.Lab2Bgr:
            case ColorConversionCode.Lab2Rgb:
            case ColorConversionCode.Lab2Lbgr:
            case ColorConversionCode.Lab2Lrgb:
            {
                RequireChannels(scn, 3, 3);
                RequireByteOrFloat<T>(code);
                int dcn = DefaultColorDcn(dstCn);
                bool srgb = code is ColorConversionCode.Lab2Bgr or ColorConversionCode.Lab2Rgb;
                return typeof(T) == typeof(byte)
                    ? As<T>(Run(As<byte>(source), dcn, new LabToRgbByteRow(dcn, bidx, srgb)))
                    : As<T>(Run(As<float>(source), dcn, new LabToRgbFloatRow(dcn, bidx, srgb)));
            }

            case ColorConversionCode.Bgr2Xyz:
            case ColorConversionCode.Rgb2Xyz:
                RequireChannels(scn, 3, 4);
                return typeof(T) == typeof(float)
                    ? As<T>(Run(As<float>(source), 3, new MatrixFloatRow(scn, 3, XyzCoefficients.ForwardFloat(bidx))))
                    : Run(source, 3, new MatrixIntRow<T>(scn, 3, XyzCoefficients.ForwardInt(bidx)));

            case ColorConversionCode.Xyz2Bgr:
            case ColorConversionCode.Xyz2Rgb:
            {
                RequireChannels(scn, 3, 3);
                int dcn = DefaultColorDcn(dstCn);
                return typeof(T) == typeof(float)
                    ? As<T>(Run(As<float>(source), dcn, new MatrixFloatRow(3, dcn, XyzCoefficients.InverseFloat(bidx))))
                    : Run(source, dcn, new MatrixIntRow<T>(3, dcn, XyzCoefficients.InverseInt(bidx)));
            }

            case ColorConversionCode.Bgr2YCrCb:
            case ColorConversionCode.Rgb2YCrCb:
                RequireChannels(scn, 3, 4);
                return typeof(T) == typeof(float)
                    ? As<T>(Run(As<float>(source), 3, new YCrCbFloatRow(scn, bidx)))
                    : Run(source, 3, new YCrCbIntRow<T>(scn, bidx));

            case ColorConversionCode.YCrCb2Bgr:
            case ColorConversionCode.YCrCb2Rgb:
            {
                RequireChannels(scn, 3, 3);
                int dcn = DefaultColorDcn(dstCn);
                return typeof(T) == typeof(float)
                    ? As<T>(Run(As<float>(source), dcn, new YCrCbToRgbFloatRow(dcn, bidx)))
                    : Run(source, dcn, new YCrCbToRgbIntRow<T>(dcn, bidx));
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(code), code, "Unsupported colour conversion code.");
        }
    }

    private static bool SwapBlue(ColorConversionCode code) => code switch
    {
        ColorConversionCode.Bgr2Rgba or ColorConversionCode.Rgba2Bgr or ColorConversionCode.Bgr2Rgb or
        ColorConversionCode.Bgra2Rgba or ColorConversionCode.Rgb2Gray or ColorConversionCode.Rgba2Gray or
        ColorConversionCode.Rgb2Xyz or ColorConversionCode.Xyz2Rgb or ColorConversionCode.Rgb2YCrCb or
        ColorConversionCode.YCrCb2Rgb or ColorConversionCode.Rgb2Hsv or ColorConversionCode.Rgb2Lab or
        ColorConversionCode.Hsv2Rgb or ColorConversionCode.Lab2Rgb or ColorConversionCode.Rgb2HsvFull or
        ColorConversionCode.Hsv2RgbFull or ColorConversionCode.Lrgb2Lab or ColorConversionCode.Lab2Lrgb => true,
        _ => false,
    };

    private static void RequireChannels(int scn, int min, int max)
    {
        if (scn < min || scn > max)
        {
            throw new ArgumentException(
                min == max
                    ? $"The source must have {min} channel(s); it has {scn}."
                    : $"The source must have {min} to {max} channels; it has {scn}.",
                "source");
        }
    }

    private static void RequireDstCn(int dcn)
    {
        if (dcn is not (3 or 4))
        {
            throw new ArgumentException("The destination channel count must be 3 or 4.", nameof(dcn));
        }
    }

    private static int DefaultColorDcn(int dstCn)
    {
        int dcn = dstCn > 0 ? dstCn : 3;
        RequireDstCn(dcn);
        return dcn;
    }

    private static void RequireByteOrFloat<T>(ColorConversionCode code)
        where T : unmanaged
    {
        if (typeof(T) == typeof(ushort))
        {
            throw new NotSupportedException($"{code} supports byte and float images only.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Mat<TTo> As<TTo>(object mat)
        where TTo : unmanaged
        => (Mat<TTo>)mat;

    private static Mat<T> Run<T, TRow>(Mat<T> source, int dcn, TRow row)
        where T : unmanaged
        where TRow : struct, IColorRow<T>
    {
        var dst = new Mat<T>(source.Width, source.Height, dcn, clear: false);
        ParallelRows.For(source.Height, (long)source.Width * 8, (y0, y1) => Rows(source, dst, row, y0, y1));
        return dst;
    }

    private static void Rows<T, TRow>(Mat<T> source, Mat<T> dst, TRow row, int y0, int y1)
        where T : unmanaged
        where TRow : struct, IColorRow<T>
    {
        int width = source.Width;
        for (int y = y0; y < y1; y++)
        {
            row.Apply(source.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), width);
        }
    }
}
