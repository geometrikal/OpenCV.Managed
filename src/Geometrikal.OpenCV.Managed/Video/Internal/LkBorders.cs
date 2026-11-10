// This implementation is ported from the OpenCV core border functions as used by the
// optical flow pyramid.
//
// Original sources:
//   modules/core/src/copy.cpp (copyMakeBorder, Mat::locateROI semantics)
//   modules/video/src/lkpyramid.cpp (buildOpticalFlowPyramid)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Video.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Region-of-interest location and in-place border fill for padded pyramid buffers. The fill
/// supports any channel count.
/// </summary>
internal static class LkBorders
{
    /// <summary>
    /// Returns the size of the backing array as a matrix and the position of <paramref name="m"/>
    /// in it, as OpenCV <c>Mat::locateROI</c> with the array start and end as
    /// <c>datastart</c> and <c>dataend</c>.
    /// </summary>
    public static (Size Whole, Point Offset) LocateRoi<T>(Mat<T> m)
        where T : unmanaged
    {
        int step = m.Stride;
        int cn = m.Channels;
        long delta1 = m.DangerousGetOffset();
        long delta2 = m.DangerousGetArray().Length;
        int ofsY = (int)(delta1 / step);
        int ofsX = (int)((delta1 - ((long)step * ofsY)) / cn);
        long minStep = (long)(ofsX + m.Width) * cn;
        int wholeH = (int)(((delta2 - minStep) / step) + 1);
        wholeH = Math.Max(wholeH, ofsY + m.Height);
        int wholeW = (int)((delta2 - ((long)step * (wholeH - 1))) / cn);
        wholeW = Math.Max(wholeW, ofsX + m.Width);
        return (new Size(wholeW, wholeH), new Point(ofsX, ofsY));
    }

    /// <summary>True when the backing array holds elements outside <paramref name="m"/>.</summary>
    public static bool IsSubmatrix<T>(Mat<T> m)
        where T : unmanaged
    {
        (Size whole, Point ofs) = LocateRoi(m);
        return ofs.X != 0 || ofs.Y != 0 || whole.Width != m.Width || whole.Height != m.Height;
    }

    /// <summary>
    /// Returns a view of the same array that extends <paramref name="m"/> by the given counts.
    /// The caller guarantees that the extended view is inside the array.
    /// </summary>
    public static Mat<T> Extend<T>(Mat<T> m, int top, int bottom, int left, int right)
        where T : unmanaged
    {
        int offset = m.DangerousGetOffset() - (top * m.Stride) - (left * m.Channels);
        return new Mat<T>(m.DangerousGetArray(), m.Width + left + right, m.Height + top + bottom, m.Channels, m.Stride, offset);
    }

    /// <summary>
    /// OpenCV <c>copyMakeBorder(src, dst, top, bottom, left, right, border)</c> without
    /// <c>BORDER_ISOLATED</c>: pixels of the parent array around a submatrix source replace
    /// border pixels where they exist. The constant border value is 0.
    /// </summary>
    public static void CopyMakeBorder<T>(Mat<T> src, Mat<T> dst, int top, int left, BorderType border)
        where T : unmanaged
    {
        int bottom = dst.Height - src.Height - top;
        int right = dst.Width - src.Width - left;
        if (IsSubmatrix(src))
        {
            (Size whole, Point ofs) = LocateRoi(src);
            int dtop = Math.Min(ofs.Y, top);
            int dbottom = Math.Min(whole.Height - src.Height - ofs.Y, bottom);
            int dleft = Math.Min(ofs.X, left);
            int dright = Math.Min(whole.Width - src.Width - ofs.X, right);
            src = Extend(src, dtop, dbottom, dleft, dright);
            top -= dtop;
            left -= dleft;
        }

        src.CopyTo(dst.Roi(left, top, src.Width, src.Height));
        FillBorder(dst, top, left, src.Width, src.Height, border);
    }

    /// <summary>
    /// Fills the pixels of <paramref name="dst"/> outside the rectangle
    /// (<paramref name="left"/>, <paramref name="top"/>, <paramref name="width"/>,
    /// <paramref name="height"/>) from that rectangle, as OpenCV <c>copyMakeBorder</c> with
    /// <c>BORDER_ISOLATED</c>. The constant border value is 0.
    /// </summary>
    public static void FillBorder<T>(Mat<T> dst, int top, int left, int width, int height, BorderType border)
        where T : unmanaged
    {
        int cn = dst.Channels;
        int dw = dst.Width;
        int[] xmapBuffer = ArrayPool<int>.Shared.Rent(dw);
        try
        {
            Span<int> xmap = xmapBuffer.AsSpan(0, dw);
            for (int x = 0; x < dw; x++)
            {
                xmap[x] = border == BorderType.Constant && (x < left || x >= left + width)
                    ? -1
                    : left + Borders.BorderInterpolate(x - left, width, border);
            }

            for (int y = top; y < top + height; y++)
            {
                Span<T> row = dst.GetRowSpan(y);
                FillRowEnds(row, xmap, left, width, cn);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(xmapBuffer);
        }

        for (int y = 0; y < dst.Height; y++)
        {
            if (y >= top && y < top + height)
            {
                continue;
            }

            Span<T> row = dst.GetRowSpan(y);
            if (border == BorderType.Constant)
            {
                row.Clear();
                continue;
            }

            int sy = top + Borders.BorderInterpolate(y - top, height, border);
            dst.GetRowSpan(sy).CopyTo(row);
        }
    }

    private static void FillRowEnds<T>(Span<T> row, ReadOnlySpan<int> xmap, int left, int width, int cn)
        where T : unmanaged
    {
        for (int x = 0; x < xmap.Length; x++)
        {
            if (x >= left && x < left + width)
            {
                x = left + width - 1;
                continue;
            }

            int sx = xmap[x];
            for (int c = 0; c < cn; c++)
            {
                row[(x * cn) + c] = sx < 0 ? default : row[(sx * cn) + c];
            }
        }
    }
}
