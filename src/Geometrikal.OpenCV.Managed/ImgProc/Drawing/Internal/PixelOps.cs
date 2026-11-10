// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Low-level per-pixel and per-row write helpers shared by the drawing
/// primitives, plus the scalar-to-raw-colour conversion (OpenCV
/// <c>scalarToRawData</c>).
///
/// <para>All writes go through <see cref="Mat{T}.GetRowSpan(int)"/> so that ROI
/// (view) targets write through to their parent matrix. The pixel-write helpers
/// perform no bounds checks: callers must guarantee the coordinates are inside
/// the matrix (the drawing primitives clip beforehand).</para>
/// </summary>
internal static class PixelOps
{
    /// <summary>Run length in pixels below which <see cref="HLine{T}"/> writes each pixel instead of doubling copies.</summary>
    private const int ShortRun = 16;

    /// <summary>
    /// Converts a <see cref="Scalar"/> to a per-channel raw colour array in the
    /// matrix element type, mirroring OpenCV's <c>scalarToRawData</c> saturating
    /// conversion semantics.
    /// </summary>
    /// <typeparam name="T">
    /// Matrix element type; one of byte, sbyte, ushort, short, int, float or
    /// double.
    /// </typeparam>
    /// <param name="color">The source colour; component <c>c</c> feeds channel <c>c</c>.</param>
    /// <param name="channels">Number of channels to produce (matrix channel count).</param>
    /// <returns>An array of length <paramref name="channels"/> holding the converted colour.</returns>
    /// <exception cref="NotSupportedException">Thrown for unsupported element types.</exception>
    public static T[] ToRawColor<T>(Scalar color, int channels)
        where T : unmanaged
    {
        return ScalarConverter.ToRawColor<T>(color, channels);
    }

    /// <summary>
    /// Writes a single pixel. The caller guarantees
    /// <c>0 &lt;= x &lt; Width</c> and <c>0 &lt;= y &lt; Height</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PutPixel<T>(Mat<T> img, int y, int x, ReadOnlySpan<T> color)
        where T : unmanaged
    {
        Span<T> row = img.GetRowSpan(y);
        int channels = img.Channels;

        if (channels == 1)
        {
            row[x] = color[0];
            return;
        }

        int baseIndex = x * channels;
        for (int c = 0; c < channels; c++)
        {
            row[baseIndex + c] = color[c];
        }
    }

    /// <summary>
    /// Fills the inclusive horizontal run [<paramref name="x1"/>..<paramref name="x2"/>]
    /// on row <paramref name="y"/> with <paramref name="color"/>. Ports OpenCV's
    /// <c>ICV_HLINE_X</c>: single-channel fills directly; multi-channel writes the
    /// first pixel then doubles the filled prefix onto the remainder. The caller
    /// guarantees the row and column range are inside the matrix.
    /// </summary>
    public static void HLine<T>(Mat<T> img, int y, int x1, int x2, ReadOnlySpan<T> color)
        where T : unmanaged
    {
        // Scanline fills emit empty runs (x2 == x1 - 1); ICV_HLINE_X writes nothing for them.
        if (x2 < x1)
        {
            return;
        }

        Span<T> row = img.GetRowSpan(y);
        int channels = img.Channels;

        if (channels == 1)
        {
            row.Slice(x1, x2 - x1 + 1).Fill(color[0]);
            return;
        }

        // Multi-channel: the active elements are [x1*ch .. (x2+1)*ch).
        int start = x1 * channels;
        int end = (x2 + 1) * channels;
        Span<T> segment = row.Slice(start, end - start);
        if (channels == 3 && x2 - x1 < ShortRun)
        {
            T c0 = color[0], c1 = color[1], c2 = color[2];
            for (int p = 0; p + 2 < segment.Length; p += 3)
            {
                segment[p] = c0;
                segment[p + 1] = c1;
                segment[p + 2] = c2;
            }

            return;
        }


        // Write the first pixel, then repeatedly copy the already-filled prefix
        // onto the rest, doubling the copy length each time (same effect as the
        // C++ memcpy-doubling loop).
        color.Slice(0, channels).CopyTo(segment);
        int filled = channels;
        while (filled < segment.Length)
        {
            int copyLength = Math.Min(filled, segment.Length - filled);
            segment.Slice(0, copyLength).CopyTo(segment.Slice(filled, copyLength));
            filled += copyLength;
        }
    }

}
