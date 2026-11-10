// This implementation is ported from the OpenCV affine and perspective warps.
//
// Original sources:
//   modules/imgproc/src/imgwarp.cpp (cv::warpAffine, cv::warpPerspective)
//   modules/core/src/lapack.cpp (invert, DECOMP_LU for 3x3 matrices)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <content>Affine and perspective warps.</content>
public static partial class Geometry
{
    /// <summary>
    /// Applies an affine transformation, as OpenCV <c>warpAffine</c>:
    /// <c>dst(x, y) = src(M11·x + M12·y + M13, M21·x + M22·y + M23)</c> with the inverse of
    /// <paramref name="matrix"/>, or with <paramref name="matrix"/> itself for
    /// <see cref="WarpFlags.InverseMap"/>.
    /// </summary>
    /// <remarks>
    /// <para>Source coordinates are computed on a 1/1024-pixel grid and quantised to 1/32 pixel.
    /// Interpolating kernels use the 32 × 32 OpenCV weight tables (16-bit weights for <c>byte</c>,
    /// single-precision weights otherwise) for every element type and interpolation. This is the
    /// OpenCV 4.x algorithm. OpenCV 5.0 uses floating-point coordinates without quantisation for
    /// nearest and bilinear interpolation of 8U, 16U and 32F images with 1, 3 or 4 channels, and a
    /// floating-point bicubic kernel; their results depend on the CPU instruction set (FMA and
    /// vector width), so this port does not follow them.</para>
    /// <para><see cref="InterpolationFlags.Area"/> is treated as
    /// <see cref="InterpolationFlags.Linear"/>. <c>sbyte</c> and <c>int</c> images support only
    /// <see cref="InterpolationFlags.Nearest"/>. Cubic and Lanczos4 support at most 4 channels.</para>
    /// <para>Border handling: see <see cref="Remap{T}(Mat{T}, Mat{float}, Mat{float}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>.</para>
    /// </remarks>
    /// <param name="source">Source image.</param>
    /// <param name="matrix">2 × 3 transformation matrix (width 3, height 2).</param>
    /// <param name="dsize">Destination size, or <c>default</c> for the source size.</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <param name="flags"><see cref="WarpFlags.InverseMap"/> or <see cref="WarpFlags.None"/>.</param>
    /// <param name="borderType">Border rule for source pixels outside the image.</param>
    /// <param name="borderValue">Value for <see cref="BorderType.Constant"/>; component <c>c &amp; 3</c> is used for channel <c>c</c>.</param>
    /// <returns>A new warped image.</returns>
    public static Mat<T> WarpAffine<T>(
        Mat<T> source,
        Mat<double> matrix,
        Size dsize = default,
        InterpolationFlags interpolation = InterpolationFlags.Linear,
        WarpFlags flags = WarpFlags.None,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Size size = DefaultSize(source, dsize);
        var dst = new Mat<T>(size.Width, size.Height, source.Channels, clear: borderType == BorderType.Transparent);
        WarpAffineCore(source, dst, Geometry.ReadAffine(matrix), interpolation, flags, borderType, borderValue);
        return dst;
    }

    /// <summary>Applies an affine transformation with a single-precision matrix, as OpenCV <c>warpAffine</c>.</summary>
    /// <inheritdoc cref="WarpAffine{T}(Mat{T}, Mat{double}, Size, InterpolationFlags, WarpFlags, BorderType, Scalar)"/>
    public static Mat<T> WarpAffine<T>(
        Mat<T> source,
        Mat<float> matrix,
        Size dsize = default,
        InterpolationFlags interpolation = InterpolationFlags.Linear,
        WarpFlags flags = WarpFlags.None,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default)
        where T : unmanaged
        => WarpAffine(source, ToDouble(matrix), dsize, interpolation, flags, borderType, borderValue);

    /// <summary>
    /// Applies an affine transformation into <paramref name="destination"/>, as OpenCV
    /// <c>warpAffine</c> with <c>dsize = destination size</c>. With
    /// <see cref="BorderType.Transparent"/>, destination pixels that map outside the source keep
    /// their values.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="destination">Destination with the channel count of the source. It may be the source matrix.</param>
    /// <param name="matrix">2 × 3 transformation matrix (width 3, height 2).</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <param name="flags"><see cref="WarpFlags.InverseMap"/> or <see cref="WarpFlags.None"/>.</param>
    /// <param name="borderType">Border rule.</param>
    /// <param name="borderValue">Value for <see cref="BorderType.Constant"/>.</param>
    public static void WarpAffine<T>(
        Mat<T> source,
        Mat<T> destination,
        Mat<double> matrix,
        InterpolationFlags interpolation = InterpolationFlags.Linear,
        WarpFlags flags = WarpFlags.None,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        RequireChannels(source, destination);
        double[] m = Geometry.ReadAffine(matrix);
        WithDetachedSource(source, destination, src => WarpAffineCore(src, destination, m, interpolation, flags, borderType, borderValue));
    }

    /// <summary>
    /// Applies a perspective transformation, as OpenCV <c>warpPerspective</c>:
    /// <c>dst(x, y) = src((M11·x + M12·y + M13) / W, (M21·x + M22·y + M23) / W)</c> with
    /// <c>W = M31·x + M32·y + M33</c>, using the inverse of <paramref name="matrix"/> (LU
    /// cofactor formula) or <paramref name="matrix"/> itself for <see cref="WarpFlags.InverseMap"/>.
    /// </summary>
    /// <remarks>
    /// <para>Coordinates are computed in double precision per destination pixel and quantised to
    /// 1/32 pixel. A pixel with <c>W = 0</c> samples source pixel (0, 0). A singular matrix
    /// inverts to the zero matrix. Other rules: see
    /// <see cref="WarpAffine{T}(Mat{T}, Mat{double}, Size, InterpolationFlags, WarpFlags, BorderType, Scalar)"/>.</para>
    /// </remarks>
    /// <param name="source">Source image.</param>
    /// <param name="matrix">3 × 3 transformation matrix.</param>
    /// <param name="dsize">Destination size, or <c>default</c> for the source size.</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <param name="flags"><see cref="WarpFlags.InverseMap"/> or <see cref="WarpFlags.None"/>.</param>
    /// <param name="borderType">Border rule.</param>
    /// <param name="borderValue">Value for <see cref="BorderType.Constant"/>.</param>
    /// <returns>A new warped image.</returns>
    public static Mat<T> WarpPerspective<T>(
        Mat<T> source,
        Mat<double> matrix,
        Size dsize = default,
        InterpolationFlags interpolation = InterpolationFlags.Linear,
        WarpFlags flags = WarpFlags.None,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Size size = DefaultSize(source, dsize);
        var dst = new Mat<T>(size.Width, size.Height, source.Channels, clear: borderType == BorderType.Transparent);
        WarpPerspectiveCore(source, dst, ReadPerspective(matrix), interpolation, flags, borderType, borderValue);
        return dst;
    }

    /// <summary>Applies a perspective transformation with a single-precision matrix, as OpenCV <c>warpPerspective</c>.</summary>
    /// <inheritdoc cref="WarpPerspective{T}(Mat{T}, Mat{double}, Size, InterpolationFlags, WarpFlags, BorderType, Scalar)"/>
    public static Mat<T> WarpPerspective<T>(
        Mat<T> source,
        Mat<float> matrix,
        Size dsize = default,
        InterpolationFlags interpolation = InterpolationFlags.Linear,
        WarpFlags flags = WarpFlags.None,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default)
        where T : unmanaged
        => WarpPerspective(source, ToDouble(matrix), dsize, interpolation, flags, borderType, borderValue);

    /// <summary>
    /// Applies a perspective transformation into <paramref name="destination"/>, as OpenCV
    /// <c>warpPerspective</c> with <c>dsize = destination size</c>.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="destination">Destination with the channel count of the source. It may be the source matrix.</param>
    /// <param name="matrix">3 × 3 transformation matrix.</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <param name="flags"><see cref="WarpFlags.InverseMap"/> or <see cref="WarpFlags.None"/>.</param>
    /// <param name="borderType">Border rule.</param>
    /// <param name="borderValue">Value for <see cref="BorderType.Constant"/>.</param>
    public static void WarpPerspective<T>(
        Mat<T> source,
        Mat<T> destination,
        Mat<double> matrix,
        InterpolationFlags interpolation = InterpolationFlags.Linear,
        WarpFlags flags = WarpFlags.None,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        RequireChannels(source, destination);
        double[] m = ReadPerspective(matrix);
        WithDetachedSource(source, destination, src => WarpPerspectiveCore(src, destination, m, interpolation, flags, borderType, borderValue));
    }

    private static void WarpAffineCore<T>(
        Mat<T> src, Mat<T> dst, double[] m, InterpolationFlags interpolation, WarpFlags flags, BorderType borderType, Scalar borderValue)
        where T : unmanaged
    {
        interpolation = WarpEngine.Validate<T>(interpolation, src.Channels);
        WarpEngine.ValidateBorder(borderType);
        ValidateWarpFlags(flags);
        if ((flags & WarpFlags.InverseMap) == 0)
        {
            double d = (m[0] * m[4]) - (m[1] * m[3]);
            d = d != 0 ? 1.0 / d : 0;
            double a11 = m[4] * d, a22 = m[0] * d;
            m[0] = a11;
            m[1] *= -d;
            m[3] *= -d;
            m[4] = a22;
            double b1 = (-m[0] * m[2]) - (m[1] * m[5]);
            double b2 = (-m[3] * m[2]) - (m[4] * m[5]);
            m[2] = b1;
            m[5] = b2;
        }

        using var coords = new AffineCoords(m, dst.Width);
        WarpEngine.Run(src, dst, coords, interpolation, borderType, borderValue, relative: false);
    }

    private static void WarpPerspectiveCore<T>(
        Mat<T> src, Mat<T> dst, double[] m, InterpolationFlags interpolation, WarpFlags flags, BorderType borderType, Scalar borderValue)
        where T : unmanaged
    {
        interpolation = WarpEngine.Validate<T>(interpolation, src.Channels);
        WarpEngine.ValidateBorder(borderType);
        ValidateWarpFlags(flags);
        if ((flags & WarpFlags.InverseMap) == 0)
        {
            m = Invert3x3(m);
        }

        WarpEngine.Run(src, dst, new PerspectiveCoords(m, dst.Width, dst.Height), interpolation, borderType, borderValue, relative: false);
    }

    /// <summary><c>invert(M, M, DECOMP_LU)</c> for 3 × 3: cofactor formula; a zero determinant gives the zero matrix.</summary>
    private static double[] Invert3x3(double[] s)
    {
        double d = (s[0] * ((s[4] * s[8]) - (s[5] * s[7]))) - (s[1] * ((s[3] * s[8]) - (s[5] * s[6]))) + (s[2] * ((s[3] * s[7]) - (s[4] * s[6])));
        var t = new double[9];
        if (d == 0)
        {
            return t;
        }

        d = 1.0 / d;
        t[0] = ((s[4] * s[8]) - (s[5] * s[7])) * d;
        t[1] = ((s[2] * s[7]) - (s[1] * s[8])) * d;
        t[2] = ((s[1] * s[5]) - (s[2] * s[4])) * d;
        t[3] = ((s[5] * s[6]) - (s[3] * s[8])) * d;
        t[4] = ((s[0] * s[8]) - (s[2] * s[6])) * d;
        t[5] = ((s[2] * s[3]) - (s[0] * s[5])) * d;
        t[6] = ((s[3] * s[7]) - (s[4] * s[6])) * d;
        t[7] = ((s[1] * s[6]) - (s[0] * s[7])) * d;
        t[8] = ((s[0] * s[4]) - (s[1] * s[3])) * d;
        return t;
    }

    private static void ValidateWarpFlags(WarpFlags flags)
    {
        if ((flags & ~WarpFlags.InverseMap) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(flags), flags, "The warps accept only WarpFlags.InverseMap.");
        }
    }

    private static double[] ReadPerspective(Mat<double> matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Width != 3 || matrix.Height != 3 || matrix.Channels != 1)
        {
            throw new ArgumentException("The matrix must be 3x3 with one channel.", nameof(matrix));
        }

        return matrix.ToArray();
    }

    private static Mat<double> ToDouble(Mat<float> matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        float[] f = matrix.ToArray();
        var d = new double[f.Length];
        for (int i = 0; i < f.Length; i++)
        {
            d[i] = f[i];
        }

        return new Mat<double>(d, matrix.Width, matrix.Height, matrix.Channels);
    }

    private static Size DefaultSize<T>(Mat<T> source, Size dsize)
        where T : unmanaged
    {
        if (dsize.Width == 0 && dsize.Height == 0)
        {
            return new Size(source.Width, source.Height);
        }

        if (dsize.Width <= 0 || dsize.Height <= 0)
        {
            throw new ArgumentException("The destination size must be positive or empty.", nameof(dsize));
        }

        return dsize;
    }

    private static void RequireChannels<T>(Mat<T> source, Mat<T> destination)
        where T : unmanaged
    {
        if (destination.Channels != source.Channels)
        {
            throw new ArgumentException("The destination must have the channel count of the source.", nameof(destination));
        }
    }

    private static void WithDetachedSource<T>(Mat<T> source, Mat<T> destination, Action<Mat<T>> run)
        where T : unmanaged
    {
        if (TransformKernels.SharesArray(source, destination))
        {
            using PooledMat<T> copy = TransformKernels.Detach(source);
            run(copy.Mat);
            return;
        }

        run(source);
    }
}
