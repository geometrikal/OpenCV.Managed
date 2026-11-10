// This implementation is ported from the OpenCV image pyramid functions.
//
// Original sources:
//   modules/imgproc/src/pyramids.cpp (pyrDown, pyrUp, buildPyramid)
//   modules/imgproc/include/opencv2/imgproc.hpp (declarations)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's Gaussian pyramid functions: <c>pyrDown</c>, <c>pyrUp</c> and
/// <c>buildPyramid</c>.
/// </summary>
/// <remarks>
/// <para><b>Element types.</b> <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> and
/// <c>double</c>, with any channel count. Other element types throw
/// <see cref="NotSupportedException"/>.</para>
/// <para><b>Arithmetic.</b> Integer types use the OpenCV fixed-point sums with the cast
/// <c>(s + 128) >> 8</c> (down) and <c>(s + 32) >> 6</c> (up); the results are exact.
/// Floating-point sums follow the operation order of OpenCV 5.0 on its SSE3 baseline: the
/// OpenCV 128-bit vector loops and the scalar loops group the sums differently, and each
/// element uses the order of the OpenCV loop that computes it.</para>
/// <para><b>Regions of interest.</b> A region of interest is always treated as an isolated
/// image; pixels outside the view are never read. OpenCV without a HAL also ignores the parent
/// image.</para>
/// <para><b>Non-mutating.</b> The allocating overloads return a new matrix. The destination
/// overloads write every destination element. A destination that shares its array with the
/// source is supported; the source is copied first.</para>
/// </remarks>
public static class Pyramids
{
    /// <summary>
    /// Blurs an image with the 5×5 Gaussian kernel <c>[1 4 6 4 1]ᵀ[1 4 6 4 1]/256</c> and
    /// removes every second row and column, as OpenCV <c>pyrDown</c>.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="dstSize">
    /// Destination size. A size with a component of zero or less selects
    /// <c>((width + 1) / 2, (height + 1) / 2)</c>. Otherwise
    /// <c>|2·dstSize.Width − width| ≤ 2</c> and <c>|2·dstSize.Height − height| ≤ 2</c> must hold.
    /// </param>
    /// <param name="borderType">
    /// <see cref="BorderType.Reflect101"/> (OpenCV <c>BORDER_DEFAULT</c>),
    /// <see cref="BorderType.Reflect"/>, <see cref="BorderType.Replicate"/> or
    /// <see cref="BorderType.Wrap"/>.
    /// </param>
    /// <returns>A new image of the destination size.</returns>
    /// <remarks>
    /// When <c>dstSize.Width</c> exceeds <c>(width − 1) / 2</c> by two (an even source width
    /// <c>2m</c> and a destination width <c>m + 1</c>), the last column uses the OpenCV
    /// right-border table shifted by one source column, as OpenCV does.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="ArgumentException">The destination size is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The border type is not supported.</exception>
    /// <exception cref="NotSupportedException">The element type is not supported.</exception>
    public static Mat<T> PyrDown<T>(Mat<T> source, Size dstSize = default, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        RequireType<T>();
        RequireDownBorder(borderType);
        Size size = dstSize.Width <= 0 || dstSize.Height <= 0
            ? new Size((source.Width + 1) / 2, (source.Height + 1) / 2)
            : dstSize;
        RequireDownSize(source, size);
        var dst = new Mat<T>(size.Width, size.Height, source.Channels, clear: false);
        PyrDownEngine.Run(source, dst, borderType);
        return dst;
    }

    /// <summary>
    /// <see cref="PyrDown{T}(Mat{T}, Size, BorderType)"/> into <paramref name="destination"/>,
    /// whose size is the destination size.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="destination">Destination with the channel count of the source.</param>
    /// <param name="borderType">Border type; see <see cref="PyrDown{T}(Mat{T}, Size, BorderType)"/>.</param>
    public static void PyrDown<T>(Mat<T> source, Mat<T> destination, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        RequireType<T>();
        RequireDownBorder(borderType);
        RequireChannels(source, destination);
        RequireDownSize(source, new Size(destination.Width, destination.Height));
        PyrDownEngine.Run(source, destination, borderType);
    }

    /// <summary>
    /// Doubles the image size by inserting zero rows and columns and blurs the result with the
    /// 5×5 Gaussian kernel <c>[1 4 6 4 1]ᵀ[1 4 6 4 1]/64</c>, as OpenCV <c>pyrUp</c>.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="dstSize">
    /// Destination size. A size with a component of zero or less selects
    /// <c>(2·width, 2·height)</c>. Otherwise <c>|dstSize.Width − 2·width| = dstSize.Width mod 2</c>
    /// and the same for the height must hold, so each component is <c>2n − 1</c>, <c>2n</c> or
    /// <c>2n + 1</c>.
    /// </param>
    /// <param name="borderType">Only <see cref="BorderType.Reflect101"/> (OpenCV <c>BORDER_DEFAULT</c>).</param>
    /// <returns>A new image of the destination size.</returns>
    /// <remarks>
    /// The source is extended as OpenCV <c>pyrUp_</c> does: the left column and the top row
    /// reflect, the right column replicates. For a destination of <c>2n + 1</c> rows, OpenCV
    /// copies row <c>2n − 2</c> to row <c>2n</c>; for <c>2n + 1</c> columns it copies column
    /// <c>2n − 1</c> to column <c>2n</c>. The port does the same. For a source of one column and
    /// a destination of three columns OpenCV leaves column 2 uninitialized; the port copies
    /// column 1 to it.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="ArgumentException">The destination size is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The border type is not <see cref="BorderType.Reflect101"/>.</exception>
    /// <exception cref="NotSupportedException">The element type is not supported.</exception>
    public static Mat<T> PyrUp<T>(Mat<T> source, Size dstSize = default, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        RequireType<T>();
        RequireUpBorder(borderType);
        Size size = dstSize.Width <= 0 || dstSize.Height <= 0
            ? new Size(checked(source.Width * 2), checked(source.Height * 2))
            : dstSize;
        RequireUpSize(source, size);
        var dst = new Mat<T>(size.Width, size.Height, source.Channels, clear: false);
        PyrUpEngine.Run(source, dst);
        return dst;
    }

    /// <summary>
    /// <see cref="PyrUp{T}(Mat{T}, Size, BorderType)"/> into <paramref name="destination"/>,
    /// whose size is the destination size.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="destination">Destination with the channel count of the source.</param>
    /// <param name="borderType">Only <see cref="BorderType.Reflect101"/>.</param>
    public static void PyrUp<T>(Mat<T> source, Mat<T> destination, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        RequireType<T>();
        RequireUpBorder(borderType);
        RequireChannels(source, destination);
        RequireUpSize(source, new Size(destination.Width, destination.Height));
        PyrUpEngine.Run(source, destination);
    }

    /// <summary>
    /// Builds a Gaussian pyramid, as OpenCV <c>buildPyramid</c>: level 0 is
    /// <paramref name="source"/> itself (not a copy), and level <c>i</c> is
    /// <see cref="PyrDown{T}(Mat{T}, Size, BorderType)"/> of level <c>i − 1</c> with the
    /// default size.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="maxLevel">Index of the last level; zero or more.</param>
    /// <param name="borderType">Border type; see <see cref="PyrDown{T}(Mat{T}, Size, BorderType)"/>.</param>
    /// <returns><c>maxLevel + 1</c> images.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLevel"/> is negative, or the border type is not supported.</exception>
    public static Mat<T>[] BuildPyramid<T>(Mat<T> source, int maxLevel, BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfNegative(maxLevel);
        RequireType<T>();
        RequireDownBorder(borderType);
        var levels = new Mat<T>[maxLevel + 1];
        levels[0] = source;
        for (int i = 1; i <= maxLevel; i++)
        {
            levels[i] = PyrDown(levels[i - 1], new Size(0, 0), borderType);
        }

        return levels;
    }

    private static void RequireType<T>()
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(short)
            && typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"Image pyramids do not support element type {typeof(T).Name}.");
        }
    }

    private static void RequireDownBorder(BorderType borderType)
    {
        if (borderType is not (BorderType.Reflect101 or BorderType.Reflect or BorderType.Replicate or BorderType.Wrap))
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "pyrDown supports Reflect101, Reflect, Replicate and Wrap borders.");
        }
    }

    private static void RequireUpBorder(BorderType borderType)
    {
        if (borderType != BorderType.Reflect101)
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "pyrUp supports only the Reflect101 (default) border.");
        }
    }

    private static void RequireChannels<T>(Mat<T> source, Mat<T> destination)
        where T : unmanaged
    {
        if (destination.Channels != source.Channels)
        {
            throw new ArgumentException("The destination must have the channel count of the source.", nameof(destination));
        }
    }

    private static void RequireDownSize<T>(Mat<T> source, Size size)
        where T : unmanaged
    {
        if (Math.Abs((2L * size.Width) - source.Width) > 2 || Math.Abs((2L * size.Height) - source.Height) > 2)
        {
            throw new ArgumentException($"pyrDown requires |2·dst − src| ≤ 2 in each dimension (source {source.Width}×{source.Height}, destination {size.Width}×{size.Height}).");
        }
    }

    private static void RequireUpSize<T>(Mat<T> source, Size size)
        where T : unmanaged
    {
        if (Math.Abs(size.Width - (2L * source.Width)) != size.Width % 2 || Math.Abs(size.Height - (2L * source.Height)) != size.Height % 2)
        {
            throw new ArgumentException($"pyrUp requires |dst − 2·src| = dst mod 2 in each dimension (source {source.Width}×{source.Height}, destination {size.Width}×{size.Height}).");
        }
    }
}
