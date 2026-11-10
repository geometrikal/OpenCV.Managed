// This implementation is ported from the OpenCV flood fill.
//
// Original sources:
//   modules/imgproc/src/floodfill.cpp (cv::floodFill)
//   modules/imgproc/include/opencv2/imgproc.hpp
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/floodfill.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>Managed port of OpenCV <c>floodFill</c>.</summary>
/// <remarks>
/// <para><b>Region.</b> The fill starts at the seed pixel. A neighbour pixel <c>p</c> joins the
/// region when, in every channel, <c>q - loDiff &lt;= p &lt;= q + upDiff</c>. Without
/// <see cref="FloodFillFlags.FixedRange"/>, <c>q</c> is a pixel of the region next to <c>p</c>.
/// With <see cref="FloodFillFlags.FixedRange"/>, <c>q</c> is the seed pixel. Pixels that are
/// non-zero in the mask do not join the region.</para>
/// <para><b>Arithmetic.</b> For <see cref="byte"/> images, the differences are rounded down
/// (<c>cvFloor</c>) and saturated to 0-255. For <see cref="int"/> images, the differences are
/// rounded down and the pixel difference wraps on overflow. For <see cref="float"/> images, the
/// differences are converted to <see cref="float"/>; a NaN or infinite pixel difference fails the
/// test. The new value is converted as OpenCV <c>saturate_cast</c>.</para>
/// <para><b>Order.</b> The scan-line segment order is that of OpenCV 5.0, so the region, the area
/// and the rectangle are identical to OpenCV for every input. The function is single-threaded.</para>
/// <para><b>Supported types.</b> <see cref="byte"/>, <see cref="int"/> and <see cref="float"/>
/// images with one or three channels. Image width and height must not exceed 65535: OpenCV
/// stores segment coordinates in 16 bits.</para>
/// </remarks>
public static class FloodFilling
{
    private const int MaskOnlyFlag = (int)FloodFillFlags.MaskOnly;

    /// <summary>
    /// Returns the flag bits that set the value written to the mask for filled pixels
    /// (bits 8-15 of the OpenCV flags). The value 0 selects the default value 1.
    /// </summary>
    /// <param name="value">Mask fill value.</param>
    /// <returns>The flag bits to combine with the other <see cref="FloodFillFlags"/>.</returns>
    public static FloodFillFlags MaskFillValue(byte value) => (FloodFillFlags)(value << 8);

    /// <summary>
    /// Fills the connected region of the seed pixel with <paramref name="newValue"/>, as OpenCV
    /// <c>floodFill</c> without a mask. <paramref name="image"/> is modified in place.
    /// </summary>
    /// <typeparam name="T"><see cref="byte"/>, <see cref="int"/> or <see cref="float"/>.</typeparam>
    /// <param name="image">Image with one or three channels, read and written. It may be an ROI; pixels outside the ROI are not read.</param>
    /// <param name="seedPoint">Start pixel. It must be inside the image.</param>
    /// <param name="newValue">Value of the filled pixels, one component per channel.</param>
    /// <param name="loDiff">Maximum lower difference per channel. Must not be negative. The default is 0.</param>
    /// <param name="upDiff">Maximum upper difference per channel. Must not be negative. The default is 0.</param>
    /// <param name="flags">Connectivity, <see cref="FloodFillFlags.FixedRange"/> and <see cref="FloodFillFlags.MaskOnly"/>. With <see cref="FloodFillFlags.MaskOnly"/> the image does not change and only the result is computed.</param>
    /// <returns>The number of filled pixels and their bounding rectangle.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">The channel count is not 1 or 3, the connectivity is not 0, 4 or 8, a difference is negative, or a dimension exceeds 65535.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="seedPoint"/> is outside the image.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not supported.</exception>
    public static FloodFillResult FloodFill<T>(
        Mat<T> image,
        Point seedPoint,
        Scalar newValue,
        Scalar loDiff = default,
        Scalar upDiff = default,
        FloodFillFlags flags = FloodFillFlags.Connectivity4)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        return Run(image, null, seedPoint, newValue, loDiff, upDiff, (int)flags);
    }

    /// <summary>
    /// Fills the connected region of the seed pixel with <paramref name="newValue"/> and marks it
    /// in <paramref name="mask"/>, as OpenCV <c>floodFill</c> with a mask. <paramref name="image"/>
    /// and <paramref name="mask"/> are modified in place.
    /// </summary>
    /// <remarks>
    /// Mask pixel (x + 1, y + 1) belongs to image pixel (x, y). The fill does not enter pixels that
    /// are non-zero in the mask, and it writes the mask fill value (bits 8-15 of
    /// <paramref name="flags"/>, 1 by default) to the mask for each filled pixel. As in OpenCV, the
    /// one-pixel frame of the mask is set to 1 before the fill.
    /// </remarks>
    /// <typeparam name="T"><see cref="byte"/>, <see cref="int"/> or <see cref="float"/>.</typeparam>
    /// <param name="image">Image with one or three channels, read and written unless <see cref="FloodFillFlags.MaskOnly"/> is set. It may be an ROI.</param>
    /// <param name="mask">Single-channel mask of (width + 2) x (height + 2), read and written. It may be an ROI.</param>
    /// <param name="seedPoint">Start pixel. It must be inside the image.</param>
    /// <param name="newValue">Value of the filled pixels, one component per channel.</param>
    /// <param name="loDiff">Maximum lower difference per channel. Must not be negative.</param>
    /// <param name="upDiff">Maximum upper difference per channel. Must not be negative.</param>
    /// <param name="flags">Connectivity, mask fill value, <see cref="FloodFillFlags.FixedRange"/> and <see cref="FloodFillFlags.MaskOnly"/>.</param>
    /// <returns>The number of filled pixels and their bounding rectangle. The area is 0 when the seed pixel is non-zero in the mask.</returns>
    /// <exception cref="ArgumentNullException">An argument is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException">The channel count is not 1 or 3, the mask has the wrong size or channel count, the connectivity is not 0, 4 or 8, a difference is negative, or a dimension exceeds 65535.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="seedPoint"/> is outside the image.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not supported.</exception>
    public static FloodFillResult FloodFill<T>(
        Mat<T> image,
        Mat<byte> mask,
        Point seedPoint,
        Scalar newValue,
        Scalar loDiff = default,
        Scalar upDiff = default,
        FloodFillFlags flags = FloodFillFlags.Connectivity4)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(mask);
        return Run(image, mask, seedPoint, newValue, loDiff, upDiff, (int)flags);
    }

    private static FloodFillResult Run<T>(Mat<T> image, Mat<byte>? mask, Point seed, Scalar newValue, Scalar loDiff, Scalar upDiff, int flags)
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(int) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"The element type '{typeof(T)}' is not supported. Use byte, int or float.");
        }

        int cn = image.Channels;
        if (cn != 1 && cn != 3)
        {
            throw new ArgumentException("The image must have one or three channels.", nameof(image));
        }

        int connectivity = flags & 255;
        if (connectivity != 0 && connectivity != 4 && connectivity != 8)
        {
            throw new ArgumentException("The connectivity must be 4, 0 (= 4) or 8.", nameof(flags));
        }

        int width = image.Width;
        int height = image.Height;
        if (width > ushort.MaxValue || height > ushort.MaxValue)
        {
            throw new ArgumentException("The image width and height must not exceed 65535.", nameof(image));
        }

        if (mask is not null && (mask.Channels != 1 || mask.Width != width + 2 || mask.Height != height + 2))
        {
            throw new ArgumentException("The mask must be single-channel with size (width + 2) x (height + 2).", nameof(mask));
        }

        for (int c = 0; c < cn; c++)
        {
            if (loDiff[c] < 0 || upDiff[c] < 0)
            {
                throw new ArgumentException("loDiff and upDiff must not be negative.", loDiff[c] < 0 ? nameof(loDiff) : nameof(upDiff));
            }
        }

        if ((uint)seed.X >= (uint)width || (uint)seed.Y >= (uint)height)
        {
            throw new ArgumentOutOfRangeException("seedPoint", "The seed point is outside of the image.");
        }

        Span<T> nv = stackalloc T[3];
        for (int c = 0; c < cn; c++)
        {
            nv[c] = ScalarConverter.SaturateCast<T>(newValue[c]);
        }

        nv = nv[..cn];
        ref T img = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(image.DangerousGetArray()), image.DangerousGetOffset());
        int stride = image.Stride;

        if (mask is not null)
        {
            SetMaskFrame(mask);
            ref byte m = ref mask.AtElement(1, 1);
            return Gradient(ref img, stride, width, height, ref m, mask.Stride, seed, nv, loDiff, upDiff, flags);
        }

        if ((flags & MaskOnlyFlag) == 0 && typeof(T) != typeof(float) && ZeroDiffs<T>(loDiff, upDiff, cn)
            && !MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref Unsafe.Add(ref img, (seed.Y * stride) + (seed.X * cn)), cn)).SequenceEqual(MemoryMarshal.AsBytes(nv)))
        {
            // Without a caller mask, the equality fill gives the region, area and rectangle of the gradient fill.
            if (cn == 3)
            {
                return FloodFillScan.Simple<T, FloodEqualC3<T>>(ref img, stride, width, height, seed, nv, flags);
            }

            return typeof(T) == typeof(byte)
                ? FloodFillScan.SimpleC1(ref Unsafe.As<T, byte>(ref img), stride, width, height, seed, Unsafe.As<T, byte>(ref nv[0]), flags)
                : FloodFillScan.SimpleC1(ref Unsafe.As<T, int>(ref img), stride, width, height, seed, Unsafe.As<T, int>(ref nv[0]), flags);
        }

        int maskWidth = width + 2;
        int maskLength = maskWidth * (height + 2);
        byte[] pooled = ArrayPool<byte>.Shared.Rent(maskLength);
        try
        {
            var temp = new Mat<byte>(pooled, maskWidth, height + 2);
            pooled.AsSpan(0, maskLength).Clear();
            SetMaskFrame(temp);
            return Gradient(ref img, stride, width, height, ref pooled[maskWidth + 1], maskWidth, seed, nv, loDiff, upDiff, flags);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(pooled);
        }
    }

    /// <summary>Sets the one-pixel frame of the mask to 1 (OpenCV <c>copyMakeBorder</c> with <c>BORDER_ISOLATED | BORDER_CONSTANT</c>).</summary>
    private static void SetMaskFrame(Mat<byte> mask)
    {
        int last = mask.Width - 1;
        mask.GetRowSpan(0).Fill(1);
        mask.GetRowSpan(mask.Height - 1).Fill(1);
        for (int y = 1; y < mask.Height - 1; y++)
        {
            Span<byte> row = mask.GetRowSpan(y);
            row[0] = 1;
            row[last] = 1;
        }
    }

    /// <summary>True when the converted 8U or 32S differences are 0, so the gradient test is pixel equality.</summary>
    private static bool ZeroDiffs<T>(Scalar loDiff, Scalar upDiff, int cn)
        where T : unmanaged
    {
        for (int c = 0; c < cn; c++)
        {
            bool zero = typeof(T) == typeof(byte)
                ? FloorByte(loDiff[c]) == 0 && FloorByte(upDiff[c]) == 0
                : ScalarConverter.CvFloor(loDiff[c]) == 0 && ScalarConverter.CvFloor(upDiff[c]) == 0;
            if (!zero)
            {
                return false;
            }
        }

        return true;
    }

    private static byte FloorByte(double v) => (byte)Math.Clamp(ScalarConverter.CvFloor(v), 0, 255);

    private static FloodFillResult Gradient<T>(
        ref T img, int stride, int width, int height, ref byte mask, int maskStride, Point seed,
        ReadOnlySpan<T> nv, Scalar lo, Scalar up, int flags)
        where T : unmanaged
    {
        byte nm = (flags & 0xff00) == 0 ? (byte)1 : (byte)((flags >> 8) & 255);
        int cn = nv.Length;
        if (typeof(T) == typeof(byte))
        {
            ref byte b = ref Unsafe.As<T, byte>(ref img);
            ReadOnlySpan<byte> v = MemoryMarshal.Cast<T, byte>(nv);
            return cn == 1
                ? FloodFillScan.Gradient(ref b, stride, width, height, ref mask, maskStride, seed, v, nm, new FloodDiff8uC1(FloorByte(lo[0]), FloorByte(up[0])), flags)
                : FloodFillScan.Gradient(ref b, stride, width, height, ref mask, maskStride, seed, v, nm,
                    new FloodDiff8uC3(FloorByte(lo[0]), FloorByte(lo[1]), FloorByte(lo[2]), FloorByte(up[0]), FloorByte(up[1]), FloorByte(up[2])), flags);
        }

        if (typeof(T) == typeof(int))
        {
            ref int s = ref Unsafe.As<T, int>(ref img);
            ReadOnlySpan<int> v = MemoryMarshal.Cast<T, int>(nv);
            return cn == 1
                ? FloodFillScan.Gradient(ref s, stride, width, height, ref mask, maskStride, seed, v, nm,
                    new FloodDiff32sC1(ScalarConverter.CvFloor(lo[0]), ScalarConverter.CvFloor(up[0])), flags)
                : FloodFillScan.Gradient(ref s, stride, width, height, ref mask, maskStride, seed, v, nm,
                    new FloodDiff32sC3(
                        ScalarConverter.CvFloor(lo[0]), ScalarConverter.CvFloor(lo[1]), ScalarConverter.CvFloor(lo[2]),
                        ScalarConverter.CvFloor(up[0]), ScalarConverter.CvFloor(up[1]), ScalarConverter.CvFloor(up[2])), flags);
        }

        ref float f = ref Unsafe.As<T, float>(ref img);
        ReadOnlySpan<float> fv = MemoryMarshal.Cast<T, float>(nv);
        return cn == 1
            ? FloodFillScan.Gradient(ref f, stride, width, height, ref mask, maskStride, seed, fv, nm, new FloodDiff32fC1((float)lo[0], (float)up[0]), flags)
            : FloodFillScan.Gradient(ref f, stride, width, height, ref mask, maskStride, seed, fv, nm,
                new FloodDiff32fC3((float)lo[0], (float)lo[1], (float)lo[2], (float)up[0], (float)up[1], (float)up[2]), flags);
    }
}
