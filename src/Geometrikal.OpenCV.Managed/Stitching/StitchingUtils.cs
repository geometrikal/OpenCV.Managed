// This implementation is ported from the OpenCV stitching utilities.
//
// Original source:
//   modules/stitching/src/util.cpp (overlapRoi, resultRoi, resultRoiIntersection, resultTl)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Region helpers of the stitching pipeline, as OpenCV <c>detail</c> utility functions.</summary>
public static class StitchingUtils
{
    /// <summary>Computes the overlap of two placed images, as OpenCV <c>detail::overlapRoi</c>.</summary>
    /// <param name="tl1">Top-left corner of the first image.</param>
    /// <param name="tl2">Top-left corner of the second image.</param>
    /// <param name="size1">Size of the first image.</param>
    /// <param name="size2">Size of the second image.</param>
    /// <param name="roi">Receives the overlap when the images overlap; otherwise the default value.</param>
    /// <returns><c>true</c> when the overlap is not empty.</returns>
    public static bool OverlapRoi(Point tl1, Point tl2, Size size1, Size size2, out Rect roi)
    {
        int xTl = Math.Max(tl1.X, tl2.X);
        int yTl = Math.Max(tl1.Y, tl2.Y);
        int xBr = Math.Min(tl1.X + size1.Width, tl2.X + size2.Width);
        int yBr = Math.Min(tl1.Y + size1.Height, tl2.Y + size2.Height);
        if (xTl < xBr && yTl < yBr)
        {
            roi = new Rect(xTl, yTl, xBr - xTl, yBr - yTl);
            return true;
        }

        roi = default;
        return false;
    }

    /// <summary>Computes the bounding box of placed images, as OpenCV <c>detail::resultRoi</c>.</summary>
    /// <param name="corners">Top-left corners.</param>
    /// <param name="sizes">Image sizes, one per corner.</param>
    /// <returns>The bounding box.</returns>
    /// <exception cref="ArgumentException">The lists have different lengths.</exception>
    public static Rect ResultRoi(IReadOnlyList<Point> corners, IReadOnlyList<Size> sizes)
    {
        ArgumentNullException.ThrowIfNull(corners);
        ArgumentNullException.ThrowIfNull(sizes);
        if (corners.Count != sizes.Count)
        {
            throw new ArgumentException("The corner and size lists must have the same length.", nameof(sizes));
        }

        int tlx = int.MaxValue, tly = int.MaxValue, brx = int.MinValue, bry = int.MinValue;
        for (int i = 0; i < corners.Count; i++)
        {
            tlx = Math.Min(tlx, corners[i].X);
            tly = Math.Min(tly, corners[i].Y);
            brx = Math.Max(brx, corners[i].X + sizes[i].Width);
            bry = Math.Max(bry, corners[i].Y + sizes[i].Height);
        }

        return new Rect(tlx, tly, brx - tlx, bry - tly);
    }

    /// <summary>Computes the intersection of placed images, as OpenCV <c>detail::resultRoiIntersection</c>.</summary>
    /// <param name="corners">Top-left corners.</param>
    /// <param name="sizes">Image sizes, one per corner.</param>
    /// <returns>The intersection. The width or height is negative or zero when the intersection is empty, as OpenCV.</returns>
    /// <exception cref="ArgumentException">The lists have different lengths.</exception>
    public static Rect ResultRoiIntersection(IReadOnlyList<Point> corners, IReadOnlyList<Size> sizes)
    {
        ArgumentNullException.ThrowIfNull(corners);
        ArgumentNullException.ThrowIfNull(sizes);
        if (corners.Count != sizes.Count)
        {
            throw new ArgumentException("The corner and size lists must have the same length.", nameof(sizes));
        }

        int tlx = int.MinValue, tly = int.MinValue, brx = int.MaxValue, bry = int.MaxValue;
        for (int i = 0; i < corners.Count; i++)
        {
            tlx = Math.Max(tlx, corners[i].X);
            tly = Math.Max(tly, corners[i].Y);
            brx = Math.Min(brx, corners[i].X + sizes[i].Width);
            bry = Math.Min(bry, corners[i].Y + sizes[i].Height);
        }

        return new Rect(tlx, tly, brx - tlx, bry - tly);
    }

    /// <summary>Computes the minimum of the corners, as OpenCV <c>detail::resultTl</c>.</summary>
    /// <param name="corners">Top-left corners.</param>
    /// <returns>The component-wise minimum.</returns>
    public static Point ResultTl(IReadOnlyList<Point> corners)
    {
        ArgumentNullException.ThrowIfNull(corners);
        int x = int.MaxValue, y = int.MaxValue;
        foreach (Point p in corners)
        {
            x = Math.Min(x, p.X);
            y = Math.Min(y, p.Y);
        }

        return new Point(x, y);
    }
}
