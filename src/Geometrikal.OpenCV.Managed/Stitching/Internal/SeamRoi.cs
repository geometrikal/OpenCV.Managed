// This implementation is ported from the OpenCV stitching utilities.
//
// Original sources:
//   modules/stitching/src/util.cpp (overlapRoi)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Region helpers of the seam finders.</summary>
internal static class SeamRoi
{
    /// <summary>OpenCV <c>overlapRoi</c>: the intersection of two placed rectangles.</summary>
    /// <returns><see langword="true"/> when the intersection is not empty.</returns>
    public static bool OverlapRoi(Point tl1, Point tl2, Size sz1, Size sz2, out Rect roi)
    {
        int xTl = Math.Max(tl1.X, tl2.X);
        int yTl = Math.Max(tl1.Y, tl2.Y);
        int xBr = Math.Min(tl1.X + sz1.Width, tl2.X + sz2.Width);
        int yBr = Math.Min(tl1.Y + sz1.Height, tl2.Y + sz2.Height);
        if (xTl < xBr && yTl < yBr)
        {
            roi = new Rect(xTl, yTl, xBr - xTl, yBr - yTl);
            return true;
        }

        roi = default;
        return false;
    }

    /// <summary>
    /// Copies the part of <paramref name="mask"/> that falls into the window at
    /// (<paramref name="x0"/>, <paramref name="y0"/>) of the mask coordinates. Pixels outside the
    /// mask are 0.
    /// </summary>
    public static void CutWindow(Mat<byte> mask, int x0, int y0, Span<byte> window, int width, int height)
    {
        window[..(width * height)].Clear();
        int xs = Math.Max(0, -x0);
        int xe = Math.Min(width, mask.Width - x0);
        if (xs >= xe)
        {
            return;
        }

        int ys = Math.Max(0, -y0);
        int ye = Math.Min(height, mask.Height - y0);
        for (int y = ys; y < ye; y++)
        {
            mask.GetReadOnlyRowSpan(y0 + y).Slice(x0 + xs, xe - xs).CopyTo(window.Slice((y * width) + xs, xe - xs));
        }
    }
}
