// This implementation is ported from the OpenCV morphology implementation.
//
// Original sources:
//   modules/imgproc/src/morph.dispatch.cpp (in this repo: src/OpenCV/modules/imgproc/src/morph.dispatch.cpp)
//   modules/imgproc/src/morph.simd.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed port of OpenCV's morphological operations
/// (<c>modules/imgproc/src/morph.dispatch.cpp</c>): <c>erode</c>,
/// <c>dilate</c>, <c>morphologyEx</c>, and <c>getStructuringElement</c>.
///
/// <para><b>Non-mutating.</b> Every operation returns a new matrix; the source
/// is never modified (OpenCV's in-place <c>dst</c> aliasing is unnecessary
/// here).</para>
///
/// <para><b>Element types.</b> Matrices of <c>byte</c>, <c>ushort</c>,
/// <c>short</c>, <c>float</c>, and <c>double</c> with 1..4 channels are
/// supported — the same depth set as OpenCV's morphology. Channels are
/// processed independently.</para>
///
/// <para><b>Kernels.</b> Structuring elements are single-channel
/// <see cref="Mat{T}"/> of <c>sbyte</c>; any non-zero cell belongs to the
/// element (OpenCV accepts any kernel depth — a difference in type only).
/// Dilation uses the same, unreflected kernel geometry as erosion, exactly
/// like OpenCV. A null kernel means the OpenCV default (see each method).</para>
///
/// <para><b>Borders.</b> The default is <see cref="BorderType.Constant"/> with
/// OpenCV's <c>morphologyDefaultBorderValue()</c> substitution: erosion pads
/// with the type maximum and dilation with the type minimum, so borders never
/// influence the result. Region-of-interest views are always treated as
/// isolated images (OpenCV samples the parent image outside a submatrix ROI
/// unless <c>BORDER_ISOLATED</c> is set; this port always behaves as if
/// isolated).</para>
/// </summary>
public static class Morphology
{
    /// <summary>
    /// Creates a structuring element of the given shape and size, mirroring
    /// OpenCV's <c>getStructuringElement</c>. Cells belonging to the element
    /// are 1, others 0.
    /// </summary>
    /// <param name="shape">Element shape. <see cref="MorphShape.Cross"/> uses the anchor row/column.</param>
    /// <param name="size">Element size; both dimensions must be positive.</param>
    /// <param name="anchor">
    /// Anchor position, or null (or a negative component) for the centre.
    /// Only the cross shape depends on the anchor value; it is validated to be
    /// inside the element for all shapes.
    /// </param>
    /// <returns>A new single-channel <c>sbyte</c> matrix of zeros and ones.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The size is not positive or the anchor lies outside the element.</exception>
    public static Mat<sbyte> GetStructuringElement(MorphShape shape, Size size, Point? anchor = null)
    {
        if (shape is not (MorphShape.Rect or MorphShape.Cross or MorphShape.Ellipse or MorphShape.Diamond))
        {
            throw new ArgumentOutOfRangeException(nameof(shape), shape, "Unknown structuring element shape.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size.Width, nameof(size));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size.Height, nameof(size));

        Point a = NormalizeAnchor(anchor, size);

        if (size is { Width: 1, Height: 1 })
        {
            shape = MorphShape.Rect;
        }

        int r = 0;
        int c = 0;
        double invR2 = 0;
        if (shape == MorphShape.Ellipse)
        {
            r = size.Height / 2;
            c = size.Width / 2;
            invR2 = r != 0 ? 1.0 / ((double)r * r) : 0;
        }
        else if (shape == MorphShape.Diamond)
        {
            r = size.Height / 2;
            c = size.Width / 2;
        }

        var element = new Mat<sbyte>(size.Width, size.Height);

        for (int i = 0; i < size.Height; i++)
        {
            Span<sbyte> row = element.GetRowSpan(i);
            int j1 = 0;
            int j2 = 0;

            if (shape == MorphShape.Rect || (shape == MorphShape.Cross && i == a.Y))
            {
                j2 = size.Width;
            }
            else if (shape == MorphShape.Cross)
            {
                j1 = a.X;
                j2 = j1 + 1;
            }
            else if (shape == MorphShape.Diamond)
            {
                int dy = Math.Abs(i - r);
                if (dy <= r)
                {
                    int dx = r - dy;
                    j1 = Math.Max(c - dx, 0);
                    j2 = Math.Min(c + dx + 1, size.Width);
                }
            }
            else
            {
                int dy = i - r;
                if (Math.Abs(dy) <= r)
                {
                    int dx = ScalarConverter.CvRound(c * Math.Sqrt(((r * r) - (dy * dy)) * invR2));
                    j1 = Math.Max(c - dx, 0);
                    j2 = Math.Min(c + dx + 1, size.Width);
                }
            }

            for (int j = j1; j < j2; j++)
            {
                row[j] = 1;
            }
        }

        return element;
    }

    /// <summary>
    /// Erodes the source image: each output pixel is the minimum of the source
    /// pixels under the structuring element. Mirrors OpenCV's <c>erode</c>.
    /// </summary>
    /// <typeparam name="T">Element type: byte, ushort, short, float, or double.</typeparam>
    /// <param name="source">Source image; never modified.</param>
    /// <param name="kernel">
    /// Structuring element (non-zero cells), or null for the OpenCV default: a
    /// rectangle of size <c>(1 + 2·iterations)</c> squared with a central anchor.
    /// </param>
    /// <param name="anchor">Anchor within the kernel, or null (or negative components) for the centre.</param>
    /// <param name="iterations">
    /// Number of times erosion is applied; 0 returns a copy. For all-ones
    /// kernels, iterating is folded into one equivalent larger kernel exactly
    /// as OpenCV does.
    /// </param>
    /// <param name="borderType">Pixel extrapolation method.</param>
    /// <param name="borderValue">
    /// Constant border value, or null for OpenCV's default (the type maximum,
    /// so the border never wins the minimum).
    /// </param>
    /// <returns>A new eroded matrix of the same size and type.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="NotSupportedException">Unsupported element type or channel count.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Negative iterations, anchor outside the kernel, or bad border type.</exception>
    /// <exception cref="ArgumentException">The kernel is not single-channel.</exception>
    public static Mat<T> Erode<T>(
        Mat<T> source,
        Mat<sbyte>? kernel = null,
        Point? anchor = null,
        int iterations = 1,
        BorderType borderType = BorderType.Constant,
        Scalar? borderValue = null)
        where T : unmanaged
    {
        return MorphOp(source, erode: true, kernel, anchor, iterations, borderType, borderValue);
    }

    /// <summary>
    /// Dilates the source image: each output pixel is the maximum of the
    /// source pixels under the structuring element (same, unreflected kernel
    /// geometry as erosion — exactly like OpenCV's implementation). Mirrors
    /// OpenCV's <c>dilate</c>.
    /// </summary>
    /// <typeparam name="T">Element type: byte, ushort, short, float, or double.</typeparam>
    /// <param name="source">Source image; never modified.</param>
    /// <param name="kernel">Structuring element, or null for the OpenCV default (see <see cref="Erode{T}"/>).</param>
    /// <param name="anchor">Anchor within the kernel, or null for the centre.</param>
    /// <param name="iterations">Number of applications; 0 returns a copy.</param>
    /// <param name="borderType">Pixel extrapolation method.</param>
    /// <param name="borderValue">Constant border value, or null for OpenCV's default (the type minimum).</param>
    /// <returns>A new dilated matrix of the same size and type.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="NotSupportedException">Unsupported element type or channel count.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Negative iterations, anchor outside the kernel, or bad border type.</exception>
    /// <exception cref="ArgumentException">The kernel is not single-channel.</exception>
    public static Mat<T> Dilate<T>(
        Mat<T> source,
        Mat<sbyte>? kernel = null,
        Point? anchor = null,
        int iterations = 1,
        BorderType borderType = BorderType.Constant,
        Scalar? borderValue = null)
        where T : unmanaged
    {
        return MorphOp(source, erode: false, kernel, anchor, iterations, borderType, borderValue);
    }

    /// <summary>
    /// Performs a compound morphological operation, mirroring OpenCV's
    /// <c>morphologyEx</c>: opening, closing, gradient, top-hat, black-hat,
    /// the plain erode/dilate, and the hit-or-miss transform.
    /// </summary>
    /// <typeparam name="T">Element type: byte, ushort, short, float, or double. <see cref="MorphOperation.HitMiss"/> requires a single-channel byte matrix.</typeparam>
    /// <param name="source">Source image; never modified.</param>
    /// <param name="operation">The operation to perform.</param>
    /// <param name="kernel">
    /// Structuring element, or null for OpenCV's <c>morphologyEx</c> default:
    /// a 3×3 rectangle with anchor (1,1) — note this differs from the null
    /// kernel rule of <see cref="Erode{T}"/>/<see cref="Dilate{T}"/>. For
    /// <see cref="MorphOperation.HitMiss"/>, cells of 1 must match foreground
    /// and cells of -1 must match background.
    /// </param>
    /// <param name="anchor">Anchor within the kernel, or null for the centre.</param>
    /// <param name="iterations">Number of times erosion and dilation are applied; 0 returns a copy.</param>
    /// <param name="borderType">Pixel extrapolation method.</param>
    /// <param name="borderValue">Constant border value, or null for OpenCV's per-operation default.</param>
    /// <returns>A new matrix of the same size and type.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="NotSupportedException">Unsupported element type or channel count.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Unknown operation, negative iterations, or anchor outside the kernel.</exception>
    /// <exception cref="ArgumentException"><see cref="MorphOperation.HitMiss"/> on a non-byte or multi-channel matrix, or a multi-channel kernel.</exception>
    public static Mat<T> MorphologyEx<T>(
        Mat<T> source,
        MorphOperation operation,
        Mat<sbyte>? kernel,
        Point? anchor = null,
        int iterations = 1,
        BorderType borderType = BorderType.Constant,
        Scalar? borderValue = null)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);

        // morphologyEx substitutes a 3x3 rectangle for an empty kernel BEFORE
        // dispatching (morph.dispatch.cpp:1210-1213) — unlike erode/dilate,
        // whose empty-kernel rule depends on the iteration count.
        kernel ??= GetStructuringElement(MorphShape.Rect, new Size(3, 3), new Point(1, 1));

        switch (operation)
        {
            case MorphOperation.Erode:
                return Erode(source, kernel, anchor, iterations, borderType, borderValue);

            case MorphOperation.Dilate:
                return Dilate(source, kernel, anchor, iterations, borderType, borderValue);

            case MorphOperation.Open:
            {
                Mat<T> dst = Erode(source, kernel, anchor, iterations, borderType, borderValue);
                MorphInto(dst, dst, erode: false, kernel, anchor, iterations, borderType, borderValue);
                return dst;
            }

            case MorphOperation.Close:
            {
                Mat<T> dst = Dilate(source, kernel, anchor, iterations, borderType, borderValue);
                MorphInto(dst, dst, erode: true, kernel, anchor, iterations, borderType, borderValue);
                return dst;
            }

            case MorphOperation.Gradient:
            {
                Mat<T> dst = Dilate(source, kernel, anchor, iterations, borderType, borderValue);
                using PooledMat<T> eroded = PooledMat<T>.Rent(source.Width, source.Height, source.Channels);
                MorphInto(source, eroded.Mat, erode: true, kernel, anchor, iterations, borderType, borderValue);
                MorphEngine.SubtractInto(dst, eroded.Mat, dst);
                return dst;
            }

            case MorphOperation.TopHat:
            {
                Mat<T> dst = Erode(source, kernel, anchor, iterations, borderType, borderValue);
                MorphInto(dst, dst, erode: false, kernel, anchor, iterations, borderType, borderValue);
                MorphEngine.SubtractInto(source, dst, dst);
                return dst;
            }

            case MorphOperation.BlackHat:
            {
                Mat<T> dst = Dilate(source, kernel, anchor, iterations, borderType, borderValue);
                MorphInto(dst, dst, erode: true, kernel, anchor, iterations, borderType, borderValue);
                MorphEngine.SubtractInto(dst, source, dst);
                return dst;
            }

            case MorphOperation.HitMiss:
                return HitMiss(source, kernel, anchor, iterations, borderType, borderValue);

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(operation), operation, "Unknown morphological operation.");
        }
    }

    /// <summary>
    /// <see cref="Erode{T}"/> into <paramref name="dst"/>, which has the source size and channel
    /// count. <paramref name="dst"/> can be <paramref name="source"/>.
    /// </summary>
    internal static void ErodeInto<T>(
        Mat<T> source,
        Mat<T> dst,
        Mat<sbyte>? kernel = null,
        Point? anchor = null,
        int iterations = 1,
        BorderType borderType = BorderType.Constant,
        Scalar? borderValue = null)
        where T : unmanaged
        => MorphInto(source, dst, erode: true, kernel, anchor, iterations, borderType, borderValue);

    /// <summary>
    /// <see cref="Dilate{T}"/> into <paramref name="dst"/>, which has the source size and channel
    /// count. <paramref name="dst"/> can be <paramref name="source"/>.
    /// </summary>
    internal static void DilateInto<T>(
        Mat<T> source,
        Mat<T> dst,
        Mat<sbyte>? kernel = null,
        Point? anchor = null,
        int iterations = 1,
        BorderType borderType = BorderType.Constant,
        Scalar? borderValue = null)
        where T : unmanaged
        => MorphInto(source, dst, erode: false, kernel, anchor, iterations, borderType, borderValue);

    private static Mat<T> MorphOp<T>(
        Mat<T> source,
        bool erode,
        Mat<sbyte>? kernel,
        Point? anchor,
        int iterations,
        BorderType borderType,
        Scalar? borderValue)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
        MorphInto(source, dst, erode, kernel, anchor, iterations, borderType, borderValue);
        return dst;
    }

    /// <summary>
    /// Erosion or dilation into <paramref name="dst"/>. The engine pads the source before it
    /// writes, so <paramref name="dst"/> can be <paramref name="source"/>.
    /// </summary>
    private static void MorphInto<T>(
        Mat<T> source,
        Mat<T> dst,
        bool erode,
        Mat<sbyte>? kernel,
        Point? anchor,
        int iterations,
        BorderType borderType,
        Scalar? borderValue)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ThrowIfUnsupported<T>(source);
        ArgumentOutOfRangeException.ThrowIfNegative(iterations);
        if (borderType == BorderType.Wrap)
        {
            // OpenCV's FilterEngine asserts columnBorderType != BORDER_WRAP, so
            // morphology with a wrapping border throws in OpenCV as well.
            throw new NotSupportedException(
                "BorderType.Wrap is not supported by morphological operations (OpenCV FilterEngine restriction).");
        }

        if (kernel is not null && kernel.Channels != 1)
        {
            throw new ArgumentException("The structuring element must be single-channel.", nameof(kernel));
        }

        var ksize = kernel is not null ? new Size(kernel.Width, kernel.Height) : new Size(3, 3);
        Point a = NormalizeAnchor(anchor, ksize);

        // morph.dispatch.cpp:998-1002.
        if (iterations == 0 || (kernel is not null && kernel.Width * kernel.Height == 1))
        {
            if (!ReferenceEquals(source, dst))
            {
                source.CopyTo(dst);
            }

            return;
        }

        // morph.dispatch.cpp:1004-1018: the empty-kernel and rectangular
        // iteration-folding rules.
        if (kernel is null)
        {
            kernel = GetStructuringElement(
                MorphShape.Rect, new Size(1 + (iterations * 2), 1 + (iterations * 2)));
            a = new Point(iterations, iterations);
            iterations = 1;
        }
        else if (iterations > 1 && MorphEngine.CountNonZero(kernel) == kernel.Width * kernel.Height)
        {
            a = new Point(a.X * iterations, a.Y * iterations);
            kernel = GetStructuringElement(
                MorphShape.Rect,
                new Size(
                    kernel.Width + ((iterations - 1) * (kernel.Width - 1)),
                    kernel.Height + ((iterations - 1) * (kernel.Height - 1))),
                a);
            iterations = 1;
        }

        MorphEngine.ApplyInto(source, dst, kernel, a, erode, borderType, borderValue);
        for (int i = 1; i < iterations; i++)
        {
            MorphEngine.ApplyInto(dst, dst, kernel, a, erode, borderType, borderValue);
        }
    }

    /// <summary>
    /// Hit-or-miss transform (morph.dispatch.cpp:1267-1294): erode the source
    /// with the "hit" cells (1), erode the complement with the "miss" cells
    /// (-1), and combine with bitwise AND.
    /// </summary>
    private static Mat<T> HitMiss<T>(
        Mat<T> source,
        Mat<sbyte> kernel,
        Point? anchor,
        int iterations,
        BorderType borderType,
        Scalar? borderValue)
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) || source.Channels != 1)
        {
            throw new ArgumentException(
                "The hit-or-miss transform requires a single-channel byte matrix.", nameof(source));
        }

        var src = (Mat<byte>)(object)source;

        if (MorphEngine.CountNonZero(kernel) <= 0)
        {
            return source.Clone();
        }

        Mat<sbyte> k1 = MaskEquals(kernel, 1);
        Mat<sbyte> k2 = MaskEquals(kernel, -1);

        var result = new Mat<byte>(src.Width, src.Height, 1, clear: false);
        if (MorphEngine.CountNonZero(k1) > 0)
        {
            ErodeInto(src, result, k1, anchor, iterations, borderType, borderValue);
        }
        else
        {
            result.Fill(255);
        }

        if (MorphEngine.CountNonZero(k2) > 0)
        {
            using PooledMat<byte> complement = PooledMat<byte>.Rent(src.Width, src.Height, 1);
            BitwiseNot(src, complement.Mat);
            ErodeInto(complement.Mat, complement.Mat, k2, anchor, iterations, borderType, borderValue);
            BitwiseAnd(result, complement.Mat, result);
        }

        return (Mat<T>)(object)result;
    }

    private static Mat<sbyte> MaskEquals(Mat<sbyte> kernel, sbyte match)
    {
        var mask = new Mat<sbyte>(kernel.Width, kernel.Height);
        for (int y = 0; y < kernel.Height; y++)
        {
            ReadOnlySpan<sbyte> src = kernel.GetReadOnlyRowSpan(y);
            Span<sbyte> dst = mask.GetRowSpan(y);
            for (int x = 0; x < src.Length; x++)
            {
                dst[x] = src[x] == match ? (sbyte)1 : (sbyte)0;
            }
        }

        return mask;
    }

    private static void BitwiseNot(Mat<byte> src, Mat<byte> dst)
    {
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<byte> s = src.GetReadOnlyRowSpan(y);
            Span<byte> d = dst.GetRowSpan(y);
            for (int i = 0; i < s.Length; i++)
            {
                d[i] = (byte)~s[i];
            }
        }
    }

    private static void BitwiseAnd(Mat<byte> a, Mat<byte> b, Mat<byte> dst)
    {
        for (int y = 0; y < a.Height; y++)
        {
            ReadOnlySpan<byte> ra = a.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> rb = b.GetReadOnlyRowSpan(y);
            Span<byte> rd = dst.GetRowSpan(y);
            for (int i = 0; i < ra.Length; i++)
            {
                rd[i] = (byte)(ra[i] & rb[i]);
            }
        }
    }

    private static Point NormalizeAnchor(Point? anchor, Size ksize)
    {
        Point a = anchor ?? new Point(-1, -1);
        int x = a.X == -1 ? ksize.Width / 2 : a.X;
        int y = a.Y == -1 ? ksize.Height / 2 : a.Y;
        if ((uint)x >= (uint)ksize.Width || (uint)y >= (uint)ksize.Height)
        {
            throw new ArgumentOutOfRangeException(
                nameof(anchor), a, "The anchor must lie inside the structuring element.");
        }

        return new Point(x, y);
    }

    private static void ThrowIfUnsupported<T>(Mat<T> source)
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(short)
            && typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException(
                $"Morphology supports byte, ushort, short, float, and double matrices; got '{typeof(T)}'.");
        }

        if (source.Channels > 4)
        {
            throw new NotSupportedException("Scalar supports at most 4 channels.");
        }
    }
}
