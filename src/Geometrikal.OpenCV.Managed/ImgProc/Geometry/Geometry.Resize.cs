// This implementation is ported from the OpenCV resize.
//
// Original source:
//   modules/imgproc/src/resize.cpp (cv::resize, hal::resize)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's geometric image transforms: <c>resize</c>, <c>warpAffine</c>,
/// <c>warpPerspective</c>, <c>remap</c>, <c>convertMaps</c> and the transform-matrix helpers.
/// </summary>
/// <remarks>
/// <para><b>Results.</b> Overloads without a destination return a new matrix and do not modify
/// the source. Destination overloads write every destination pixel, except pixels that
/// <see cref="BorderType.Transparent"/> leaves unchanged. A warp or remap destination may be the
/// source matrix (the source is copied first); a resize destination must not share its array with
/// the source. New matrices of the warps and remap start as zeros with
/// <see cref="BorderType.Transparent"/>.</para>
/// <para><b>Bit exactness.</b> Results equal the OpenCV CPU kernels of a 128-bit SIMD build
/// (SSE4.1 dispatch, no AVX2 or FMA dispatch), with IPP and OpenCL disabled. They do not depend on
/// the managed vector width or on <see cref="CvParallel.MaxDegreeOfParallelism"/>.</para>
/// </remarks>
public static partial class Geometry
{
    /// <summary>
    /// Resizes an image, as OpenCV <c>resize</c>. Either <paramref name="dsize"/> is non-empty, or
    /// both scale factors are positive and the size is <c>round(width·fx) × round(height·fy)</c>.
    /// </summary>
    /// <remarks>
    /// <para>The scale used for the coordinate mapping is <c>dsize.Width / width</c> when
    /// <paramref name="dsize"/> is given, and <paramref name="fx"/> otherwise. The same output size
    /// can therefore give different pixels for the two forms, as in OpenCV.</para>
    /// <para>Element types: <see cref="InterpolationFlags.Nearest"/> and
    /// <see cref="InterpolationFlags.NearestExact"/> accept every element type and channel count.
    /// <see cref="InterpolationFlags.Linear"/>, <see cref="InterpolationFlags.Cubic"/>,
    /// <see cref="InterpolationFlags.Lanczos4"/> and <see cref="InterpolationFlags.Area"/> accept
    /// <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> and <c>double</c>; area decimation by
    /// a non-integer factor accepts at most 4 channels. <see cref="InterpolationFlags.LinearExact"/>
    /// accepts <c>byte</c>, <c>sbyte</c>, <c>ushort</c>, <c>short</c> and <c>int</c>, and uses
    /// <see cref="InterpolationFlags.Linear"/> for <c>float</c> and <c>double</c>.</para>
    /// <para><see cref="InterpolationFlags.NearestExact"/> follows OpenCV 5.0: the source index is
    /// <c>floor(scale·(d + 0.5))</c> accumulated in double precision. OpenCV 4.x uses a 16.16
    /// fixed-point formula and gives other indices for some sizes.</para>
    /// <para>The 16-bit and 32-bit float Lanczos4 and bicubic kernels sum the vertical taps
    /// right to left in blocks of 8 (16-bit) or 4 (float) elements and left to right in the
    /// remainder of each row, as the OpenCV 128-bit vector code. The 8-bit bicubic kernel uses
    /// single precision for blocks of 8 elements and integer arithmetic for the remainder.</para>
    /// </remarks>
    /// <param name="source">Source image.</param>
    /// <param name="dsize">Destination size, or <c>default</c> to use the scale factors.</param>
    /// <param name="fx">Horizontal scale factor; used when <paramref name="dsize"/> is empty.</param>
    /// <param name="fy">Vertical scale factor; used when <paramref name="dsize"/> is empty.</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <returns>A new resized image.</returns>
    /// <exception cref="ArgumentException">The size or scale factors are invalid.</exception>
    /// <exception cref="NotSupportedException">The element type or channel count is not supported for the method.</exception>
    public static Mat<T> Resize<T>(
        Mat<T> source,
        Size dsize,
        double fx = 0,
        double fy = 0,
        InterpolationFlags interpolation = InterpolationFlags.Linear)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        (Size size, double invX, double invY) = ResizeScale(source, dsize, fx, fy);
        var dst = new Mat<T>(size.Width, size.Height, source.Channels, clear: false);
        ResizeCore(source, dst, invX, invY, interpolation);
        return dst;
    }

    /// <summary>
    /// Resizes an image into <paramref name="destination"/>, as OpenCV <c>resize</c> with
    /// <c>dsize = destination size</c>.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="destination">Destination with the channel count of the source. It must not share its array with the source.</param>
    /// <param name="interpolation">Interpolation method.</param>
    public static void Resize<T>(Mat<T> source, Mat<T> destination, InterpolationFlags interpolation = InterpolationFlags.Linear)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Channels != source.Channels)
        {
            throw new ArgumentException("The destination must have the channel count of the source.", nameof(destination));
        }

        RequireDistinct(source, destination);
        (_, double invX, double invY) = ResizeScale(source, new Size(destination.Width, destination.Height), 0, 0);
        ResizeCore(source, destination, invX, invY, interpolation);
    }

    private static (Size Size, double InvX, double InvY) ResizeScale<T>(Mat<T> source, Size dsize, double fx, double fy)
        where T : unmanaged
    {
        if (dsize.Width <= 0 || dsize.Height <= 0)
        {
            if (dsize.Width != 0 || dsize.Height != 0)
            {
                throw new ArgumentException("The destination size must be positive or empty.", nameof(dsize));
            }

            if (!(fx > 0) || !(fy > 0))
            {
                throw new ArgumentException("Scale factors must be positive when the destination size is empty.");
            }

            var size = new Size(ScalarConverter.CvRound(source.Width * fx), ScalarConverter.CvRound(source.Height * fy));
            if (size.Width <= 0 || size.Height <= 0)
            {
                throw new ArgumentException("The computed destination size is empty.");
            }

            return (size, fx, fy);
        }

        return (dsize, (double)dsize.Width / source.Width, (double)dsize.Height / source.Height);
    }

    private static void ResizeCore<T>(Mat<T> src, Mat<T> dst, double invX, double invY, InterpolationFlags interpolation)
        where T : unmanaged
    {
        if (interpolation is < InterpolationFlags.Nearest or > InterpolationFlags.NearestExact)
        {
            throw new ArgumentOutOfRangeException(nameof(interpolation), interpolation, "Unknown interpolation method.");
        }

        bool floating = typeof(T) == typeof(float) || typeof(T) == typeof(double);
        if (interpolation == InterpolationFlags.LinearExact && floating)
        {
            interpolation = InterpolationFlags.Linear;
        }

        if (dst.Width == src.Width && dst.Height == src.Height)
        {
            src.CopyTo(dst);
            return;
        }

        int cn = src.Channels;
        double scaleX = 1.0 / invX, scaleY = 1.0 / invY;
        int iscaleX = ScalarConverter.CvRound(scaleX), iscaleY = ScalarConverter.CvRound(scaleY);
        bool areaFast = Math.Abs(scaleX - iscaleX) < Conversions.DblEpsilon && Math.Abs(scaleY - iscaleY) < Conversions.DblEpsilon;

        if (interpolation == InterpolationFlags.LinearExact)
        {
            if (areaFast && iscaleX == 2 && iscaleY == 2 && cn != 2)
            {
                interpolation = InterpolationFlags.Area;
            }
            else
            {
                RunLinearExact(src, dst, invX, invY);
                return;
            }
        }

        if (interpolation is InterpolationFlags.Nearest or InterpolationFlags.NearestExact)
        {
            int[] xofs = ResizeNearest.Rent(dst.Width);
            int[] yofs = ResizeNearest.Rent(dst.Height);
            try
            {
                if (interpolation == InterpolationFlags.Nearest)
                {
                    ResizeNearest.Offsets(src.Width, dst.Width, invX, xofs);
                    ResizeNearest.Offsets(src.Height, dst.Height, invY, yofs);
                }
                else
                {
                    ResizeNearest.ExactOffsets(src.Width, dst.Width, xofs);
                    ResizeNearest.ExactOffsets(src.Height, dst.Height, yofs);
                }

                ResizeNearest.Run(src, dst, xofs, yofs);
            }
            finally
            {
                ResizeNearest.Return(xofs);
                ResizeNearest.Return(yofs);
            }

            return;
        }

        EnsureResizeType<T>();
        if (interpolation == InterpolationFlags.Linear && areaFast && iscaleX == 2 && iscaleY == 2)
        {
            interpolation = InterpolationFlags.Area;
        }

        if (interpolation == InterpolationFlags.Area && scaleX >= 1 && scaleY >= 1)
        {
            if (areaFast)
            {
                ResizeArea.RunFast(src, dst, iscaleX, iscaleY);
                return;
            }

            if (cn > 4)
            {
                throw new NotSupportedException("Area decimation supports at most 4 channels.");
            }

            ResizeArea.RunDecimate(src, dst, scaleX, scaleY);
            return;
        }

        ResizeKernel kernel = interpolation switch
        {
            InterpolationFlags.Cubic => ResizeKernel.Cubic,
            InterpolationFlags.Lanczos4 => ResizeKernel.Lanczos4,
            _ => ResizeKernel.Linear,
        };

        using ResizeCoefficients c = ResizeCoefficients.Create(
            src.Width, dst.Width, dst.Height, cn, invX, invY, kernel, interpolation == InterpolationFlags.Area);
        if (typeof(T) == typeof(byte))
        {
            using var plan = new ResizeRows8uPlan(c, src.Width, dst.Width, cn);
            ResizeGeneric.Run<byte, int, ResizeRows8u>(
                (Mat<byte>)(object)src, (Mat<byte>)(object)dst, new ResizeRows8u(plan, c), c.YOffsets, c.Ksize);
        }
        else if (typeof(T) == typeof(double))
        {
            ResizeGeneric.Run<double, double, ResizeRowsDouble>(
                (Mat<double>)(object)src, (Mat<double>)(object)dst, new ResizeRowsDouble(c, src.Width, dst.Width, cn), c.YOffsets, c.Ksize);
        }
        else
        {
            using ResizeLinearGather? gather = typeof(T) == typeof(float) && c.Ksize <= 4 ? ResizeLinearGather.Create(c, src.Width, cn) : null;
            ResizeGeneric.Run<T, float, ResizeRowsFloat<T>>(src, dst, new ResizeRowsFloat<T>(c, src.Width, dst.Width, cn, gather), c.YOffsets, c.Ksize);
        }
    }

    private static void RunLinearExact<T>(Mat<T> src, Mat<T> dst, double invX, double invY)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            ResizeLinearExact.Run<byte, FixedU8>((Mat<byte>)(object)src, (Mat<byte>)(object)dst, invX, invY);
        }
        else if (typeof(T) == typeof(sbyte))
        {
            ResizeLinearExact.Run<sbyte, FixedS32Rows<sbyte>>((Mat<sbyte>)(object)src, (Mat<sbyte>)(object)dst, invX, invY);
        }
        else if (typeof(T) == typeof(ushort))
        {
            ResizeLinearExact.Run<ushort, FixedU16>((Mat<ushort>)(object)src, (Mat<ushort>)(object)dst, invX, invY);
        }
        else if (typeof(T) == typeof(short))
        {
            ResizeLinearExact.Run<short, FixedS32Rows<short>>((Mat<short>)(object)src, (Mat<short>)(object)dst, invX, invY);
        }
        else if (typeof(T) == typeof(int))
        {
            ResizeLinearExact.Run<int, FixedS32>((Mat<int>)(object)src, (Mat<int>)(object)dst, invX, invY);
        }
        else
        {
            throw new NotSupportedException($"INTER_LINEAR_EXACT does not support the element type '{typeof(T)}'.");
        }
    }

    private static void EnsureResizeType<T>()
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(short) &&
            typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"This interpolation does not support the element type '{typeof(T)}'.");
        }
    }

    private static void RequireDistinct<T>(Mat<T> source, Mat<T> destination)
        where T : unmanaged
    {
        if (TransformKernels.SharesArray(source, destination))
        {
            throw new ArgumentException("The destination must not share its array with the source.", nameof(destination));
        }
    }
}
