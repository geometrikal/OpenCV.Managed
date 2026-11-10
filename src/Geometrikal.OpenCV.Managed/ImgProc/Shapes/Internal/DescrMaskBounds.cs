// This implementation is ported from the OpenCV shape descriptors.
//
// Original source:
//   modules/geometry/src/geometry.cpp (maskBoundingRect)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>maskBoundingRect</c>: the bounding rectangle of the nonzero pixels of an 8-bit mask.
/// </summary>
/// <remarks>
/// Each row scans only the columns outside the current column range, as OpenCV does: the left part
/// for a new minimum, the right part for a new maximum, and the middle part only when the outer
/// parts contain no nonzero pixel. Stripes keep their own range; the merged range is exact, so the
/// result does not depend on the thread count.
/// </remarks>
internal static class DescrMaskBounds
{
    public static Rect BoundingRect(Mat<byte> mask)
    {
        int width = mask.Width, height = mask.Height;
        var state = new Bounds { XMin = width, XMax = -1, YMin = int.MaxValue, YMax = -1 };
        object gate = new();
        ParallelRows.For(height, Math.Max(1, width / Vector<byte>.Count), (start, end) =>
        {
            Bounds local = Scan(mask, start, end);
            lock (gate)
            {
                state.XMin = Math.Min(state.XMin, local.XMin);
                state.XMax = Math.Max(state.XMax, local.XMax);
                state.YMin = Math.Min(state.YMin, local.YMin);
                state.YMax = Math.Max(state.YMax, local.YMax);
            }
        });

        if (state.XMax < 0)
        {
            return default;
        }

        return new Rect(state.XMin, state.YMin, state.XMax - state.XMin + 1, state.YMax - state.YMin + 1);
    }

    private static Bounds Scan(Mat<byte> mask, int startRow, int endRow)
    {
        int width = mask.Width;
        int xmin = width, xmax = -1, ymin = int.MaxValue, ymax = -1;
        byte[] buffer = mask.DangerousGetArray();
        int offset = mask.DangerousGetOffset();
        int stride = mask.Stride;
        for (int y = startRow; y < endRow; y++)
        {
            ref byte row = ref buffer[offset + (y * stride)];
            bool haveNz = false;
            if (xmin > 0)
            {
                int j = FirstNonZero(ref row, 0, xmin);
                if (j >= 0)
                {
                    xmin = j;
                    xmax = Math.Max(xmax, j);
                    haveNz = true;
                }
            }

            int k = LastNonZero(ref row, xmax + 1, width);
            if (k >= 0)
            {
                xmax = k;
                haveNz = true;
            }

            if (!haveNz && xmax >= xmin)
            {
                haveNz = FirstNonZero(ref row, xmin, xmax + 1) >= 0;
            }

            if (haveNz)
            {
                ymin = Math.Min(ymin, y);
                ymax = y;
            }
        }

        return new Bounds { XMin = xmin, XMax = xmax, YMin = ymin, YMax = ymax };
    }

    /// <summary>Returns the first index in <c>[start, end)</c> with a nonzero byte, or -1.</summary>
    private static int FirstNonZero(ref byte row, int start, int end)
    {
        int i = start;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= end - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                if (Vector.LoadUnsafe(ref row, (nuint)i) != Vector<byte>.Zero)
                {
                    break;
                }
            }
        }

        for (; i < end; i++)
        {
            if (Unsafe.Add(ref row, i) != 0)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>Returns the last index in <c>[start, end)</c> with a nonzero byte, or -1.</summary>
    private static int LastNonZero(ref byte row, int start, int end)
    {
        int i = end;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i - Vector<byte>.Count >= start; i -= Vector<byte>.Count)
            {
                if (Vector.LoadUnsafe(ref row, (nuint)(i - Vector<byte>.Count)) != Vector<byte>.Zero)
                {
                    break;
                }
            }
        }

        for (i--; i >= start; i--)
        {
            if (Unsafe.Add(ref row, i) != 0)
            {
                return i;
            }
        }

        return -1;
    }

    private struct Bounds
    {
        public int XMin;
        public int XMax;
        public int YMin;
        public int YMax;
    }
}
