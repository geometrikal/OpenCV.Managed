// This implementation is ported from the OpenCV matrix transforms.
//
// Original source:
//   modules/core/src/matrix_transform.cpp (flip, transpose, rotate)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>flip</c>, <c>transpose</c> and <c>rotate</c>. The operations
/// move whole pixels and do no arithmetic, so they accept every element type and channel count.
/// </summary>
/// <remarks>
/// Overloads without a destination return a new matrix and do not modify the source. A
/// destination may be the source matrix or share its array; the source is then copied first.
/// </remarks>
public static class MatrixTransforms
{
    /// <summary>Flips a matrix, as OpenCV <c>flip</c>.</summary>
    /// <param name="source">Source matrix.</param>
    /// <param name="flipCode">0: around the x-axis (rows reversed); positive: around the y-axis (columns reversed); negative: both.</param>
    /// <returns>A new matrix of the source size.</returns>
    public static Mat<T> Flip<T>(Mat<T> source, int flipCode)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
        FlipInto(source, dst, flipCode);
        return dst;
    }

    /// <summary>Flips a matrix into a destination of the same shape, as OpenCV <c>flip</c>.</summary>
    /// <param name="source">Source matrix.</param>
    /// <param name="destination">Destination with the source shape. It may be the source matrix.</param>
    /// <param name="flipCode">0: around the x-axis; positive: around the y-axis; negative: both.</param>
    public static void Flip<T>(Mat<T> source, Mat<T> destination, int flipCode)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        EnsureShape(destination, source.Width, source.Height, source.Channels);
        if (TransformKernels.SharesArray(source, destination))
        {
            using PooledMat<T> copy = TransformKernels.Detach(source);
            FlipInto(copy.Mat, destination, flipCode);
            return;
        }

        FlipInto(source, destination, flipCode);
    }

    /// <summary>Transposes a matrix, as OpenCV <c>transpose</c>: <c>dst(i, j) = src(j, i)</c>.</summary>
    /// <param name="source">Source matrix. A pixel may have at most 32 bytes.</param>
    /// <returns>A new matrix with the width and height exchanged.</returns>
    /// <exception cref="NotSupportedException">A pixel has more than 32 bytes.</exception>
    public static Mat<T> Transpose<T>(Mat<T> source)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        EnsurePixelSize(source);
        var dst = new Mat<T>(source.Height, source.Width, source.Channels, clear: false);
        TransposeInto(source, dst, false, false);
        return dst;
    }

    /// <summary>Transposes a matrix into a destination, as OpenCV <c>transpose</c>.</summary>
    /// <param name="source">Source matrix. A pixel may have at most 32 bytes.</param>
    /// <param name="destination">
    /// Destination with the width and height of the source exchanged. It may be the source matrix
    /// when the source is square.
    /// </param>
    public static void Transpose<T>(Mat<T> source, Mat<T> destination)
        where T : unmanaged
        => TransformInto(source, destination, false, false);

    /// <summary>Rotates a matrix by a multiple of 90 degrees, as OpenCV <c>rotate</c>.</summary>
    /// <param name="source">Source matrix. A pixel may have at most 32 bytes.</param>
    /// <param name="rotateCode">The rotation.</param>
    /// <returns>A new matrix. 90-degree rotations exchange the width and height.</returns>
    public static Mat<T> Rotate<T>(Mat<T> source, RotateFlags rotateCode)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateRotate(rotateCode);
        if (rotateCode == RotateFlags.Rotate180)
        {
            return Flip(source, -1);
        }

        EnsurePixelSize(source);
        var dst = new Mat<T>(source.Height, source.Width, source.Channels, clear: false);
        TransposeInto(source, dst, rotateCode == RotateFlags.Rotate90Clockwise, rotateCode == RotateFlags.Rotate90Counterclockwise);
        return dst;
    }

    /// <summary>Rotates a matrix by a multiple of 90 degrees into a destination, as OpenCV <c>rotate</c>.</summary>
    /// <param name="source">Source matrix. A pixel may have at most 32 bytes.</param>
    /// <param name="destination">Destination with the shape of the rotated source. It may be the source matrix.</param>
    /// <param name="rotateCode">The rotation.</param>
    public static void Rotate<T>(Mat<T> source, Mat<T> destination, RotateFlags rotateCode)
        where T : unmanaged
    {
        ValidateRotate(rotateCode);
        if (rotateCode == RotateFlags.Rotate180)
        {
            Flip(source, destination, -1);
            return;
        }

        TransformInto(source, destination, rotateCode == RotateFlags.Rotate90Clockwise, rotateCode == RotateFlags.Rotate90Counterclockwise);
    }

    private static void TransformInto<T>(Mat<T> source, Mat<T> destination, bool reverseRows, bool reverseCols)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        EnsurePixelSize(source);
        EnsureShape(destination, source.Height, source.Width, source.Channels);
        if (TransformKernels.SharesArray(source, destination))
        {
            using PooledMat<T> copy = TransformKernels.Detach(source);
            TransposeInto(copy.Mat, destination, reverseRows, reverseCols);
            return;
        }

        TransposeInto(source, destination, reverseRows, reverseCols);
    }

    private static void FlipInto<T>(Mat<T> source, Mat<T> dst, int flipCode)
        where T : unmanaged
    {
        var s = new TransformKernels.Plane<T>(source);
        var d = new TransformKernels.Plane<T>(dst);
        int ps = Unsafe.SizeOf<T>() * source.Channels;
        int width = source.Width;
        int height = source.Height;
        bool horizontal = flipCode != 0;
        bool vertical = flipCode <= 0;
        ParallelRows.For(height, (long)width * ps / (horizontal ? 8 : 32), (y0, y1) =>
            TransformKernels.FlipRows(s, d, width, height, ps, horizontal, vertical, y0, y1));
    }

    private static void TransposeInto<T>(Mat<T> source, Mat<T> dst, bool reverseRows, bool reverseCols)
        where T : unmanaged
    {
        var s = new TransformKernels.Plane<T>(source);
        var d = new TransformKernels.Plane<T>(dst);
        int ps = Unsafe.SizeOf<T>() * source.Channels;
        int width = source.Width;
        int height = source.Height;
        ParallelRows.For(width, (long)height * ps / 8, (i0, i1) =>
            TransformKernels.Transpose(s, d, width, height, ps, reverseRows, reverseCols, i0, i1));
    }

    private static void ValidateRotate(RotateFlags rotateCode)
    {
        if (rotateCode is not (RotateFlags.Rotate90Clockwise or RotateFlags.Rotate180 or RotateFlags.Rotate90Counterclockwise))
        {
            throw new ArgumentOutOfRangeException(nameof(rotateCode), rotateCode, "Unknown rotation.");
        }
    }

    private static void EnsurePixelSize<T>(Mat<T> source)
        where T : unmanaged
    {
        if (Unsafe.SizeOf<T>() * source.Channels > 32)
        {
            throw new NotSupportedException("Transpose supports pixels of at most 32 bytes.");
        }
    }

    private static void EnsureShape<T>(Mat<T> m, int width, int height, int channels)
        where T : unmanaged
    {
        if (m.Width != width || m.Height != height || m.Channels != channels)
        {
            throw new ArgumentException(
                $"The destination must be {width}x{height} with {channels} channel(s).", nameof(m));
        }
    }
}
