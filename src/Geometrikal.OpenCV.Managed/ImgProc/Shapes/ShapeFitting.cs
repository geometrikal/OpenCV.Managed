// This API is based on OpenCV structural analysis functions.
//
// Original sources:
//   modules/geometry/include/opencv2/geometry.hpp (declarations)
//   modules/geometry/src/convhull.cpp, rotcalipers.cpp, shapedescr.cpp, linefit.cpp, geometry.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Buffers;
using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Convex hulls, enclosing shapes, ellipse and line fitting and point-in-contour tests, ported
/// from the OpenCV 5.0 <c>geometry</c> module. All functions read the input points and do not
/// change them. Coordinates are image coordinates (X to the right, Y downward).
/// </summary>
public static class ShapeFitting
{
    /// <summary>
    /// Finds the convex hull of a point set with the Sklansky algorithm, as OpenCV
    /// <c>convexHull(points, hull, clockwise, returnPoints=true)</c>.
    /// </summary>
    /// <param name="points">Input points. The order and duplicates are permitted.</param>
    /// <param name="clockwise">
    /// True gives the clockwise order in a Y-up system. In image coordinates (Y downward)
    /// true gives counter-clockwise order on screen.
    /// </param>
    /// <returns>
    /// The hull vertices in OpenCV order, or an empty array for an empty input. When all points
    /// are equal, the result is the first point.
    /// </returns>
    public static Point[] ConvexHull(ReadOnlySpan<Point> points, bool clockwise = false)
        => HullPoints(points, HullSklansky.Coords(points), clockwise);

    /// <summary>
    /// Finds the convex hull of a point set, as <see cref="ConvexHull(ReadOnlySpan{Point}, bool)"/>.
    /// </summary>
    /// <param name="points">Input points.</param>
    /// <param name="clockwise">Output orientation, as in the integer overload.</param>
    /// <returns>The hull vertices in OpenCV order.</returns>
    public static Point2f[] ConvexHull(ReadOnlySpan<Point2f> points, bool clockwise = false)
        => HullPoints(points, HullSklansky.Coords(points), clockwise);

    /// <summary>
    /// Finds the convex hull of a point set and returns the indices of its vertices in
    /// <paramref name="points"/>, as OpenCV <c>convexHull(points, hull, clockwise, returnPoints=false)</c>.
    /// For duplicate points, the indices are selected to keep the index sequence monotonous, as
    /// OpenCV does.
    /// </summary>
    /// <param name="points">Input points.</param>
    /// <param name="clockwise">Output orientation, as in <see cref="ConvexHull(ReadOnlySpan{Point}, bool)"/>.</param>
    /// <returns>The hull vertex indices, or an empty array for an empty input.</returns>
    public static int[] ConvexHullIndices(ReadOnlySpan<Point> points, bool clockwise = false)
        => HullIndices(HullSklansky.Coords(points), clockwise, false);

    /// <summary>Finds the hull vertex indices of a float point set, as the integer overload.</summary>
    /// <param name="points">Input points.</param>
    /// <param name="clockwise">Output orientation.</param>
    /// <returns>The hull vertex indices.</returns>
    public static int[] ConvexHullIndices(ReadOnlySpan<Point2f> points, bool clockwise = false)
        => HullIndices(HullSklansky.Coords(points), clockwise, false);

    /// <summary>
    /// Tests whether a contour is convex, as OpenCV <c>isContourConvex</c>. The contour must be
    /// simple (without self-intersections); otherwise the result is not defined.
    /// </summary>
    /// <param name="contour">Contour vertices.</param>
    /// <returns>True when the contour is convex. False for an empty contour.</returns>
    public static bool IsContourConvex(ReadOnlySpan<Point> contour)
        => !contour.IsEmpty && HullSklansky.IsContourConvex(HullSklansky.Coords(contour));

    /// <summary>Tests whether a float contour is convex, as the integer overload.</summary>
    /// <param name="contour">Contour vertices.</param>
    /// <returns>True when the contour is convex.</returns>
    public static bool IsContourConvex(ReadOnlySpan<Point2f> contour)
        => !contour.IsEmpty && HullSklansky.IsContourConvex(HullSklansky.Coords(contour));

    /// <summary>
    /// Finds the convexity defects of a contour, as OpenCV <c>convexityDefects</c>.
    /// </summary>
    /// <param name="contour">Contour vertices.</param>
    /// <param name="hullIndices">
    /// Hull vertex indices from <see cref="ConvexHullIndices(ReadOnlySpan{Point}, bool)"/>.
    /// </param>
    /// <returns>
    /// The defects, in hull order. The result is empty when the contour has 3 points or fewer, or
    /// the hull has fewer than 3 points.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// The hull is empty, or the hull indices are not monotonous (a self-intersecting contour).
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">A hull index is outside the contour.</exception>
    public static ConvexityDefect[] ConvexityDefects(ReadOnlySpan<Point> contour, ReadOnlySpan<int> hullIndices)
        => HullSklansky.ConvexityDefects(contour, hullIndices);

    /// <summary>
    /// Finds the rotated rectangle of minimum area that encloses a point set, with the rotating
    /// calipers, as OpenCV 5.0 <c>minAreaRect</c>.
    /// </summary>
    /// <param name="points">Input points.</param>
    /// <returns>
    /// The rectangle. The angle is in [-90, 0) degrees. An empty input gives a zero rectangle with
    /// angle -90.
    /// </returns>
    public static RotatedRect MinAreaRect(ReadOnlySpan<Point> points)
    {
        int total = points.Length;
        if (total == 0)
        {
            return CalipersMinAreaRect.FromHull([]);
        }

        int[] idx = ArrayPool<int>.Shared.Rent(total);
        Point2f[] hull = ArrayPool<Point2f>.Shared.Rent(total);
        try
        {
            int nout = HullSklansky.ConvexHull(HullSklansky.Coords(points), false, true, idx);
            for (int i = 0; i < nout; i++)
            {
                hull[i] = new Point2f(points[idx[i]].X, points[idx[i]].Y);
            }

            return CalipersMinAreaRect.FromHull(hull.AsSpan(0, nout));
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(hull);
            ArrayPool<int>.Shared.Return(idx);
        }
    }

    /// <summary>Finds the minimum-area rotated rectangle of a float point set, as the integer overload.</summary>
    /// <param name="points">Input points.</param>
    /// <returns>The rectangle.</returns>
    public static RotatedRect MinAreaRect(ReadOnlySpan<Point2f> points)
        => CalipersMinAreaRect.FromHull(ConvexHull(points, false));

    /// <summary>Returns the four corners of a rotated rectangle, as OpenCV <c>boxPoints</c>.</summary>
    /// <param name="box">The rectangle.</param>
    /// <returns>A new array with the corners in the order of <see cref="RotatedRect.Points()"/>.</returns>
    public static Point2f[] BoxPoints(RotatedRect box) => box.Points();

    /// <summary>
    /// Finds the minimum enclosing circle of a point set, as OpenCV 5.0 <c>minEnclosingCircle</c>.
    /// For more than 10 points, the Welzl algorithm processes the points in a pseudo-random order
    /// seeded from the point count and the first and last points; the result is deterministic.
    /// </summary>
    /// <param name="points">Input points.</param>
    /// <returns>
    /// The circle. The radius includes a margin of 1e-4. An empty input gives a zero circle; one
    /// point gives radius 1e-4.
    /// </returns>
    public static EnclosingCircle MinEnclosingCircle(ReadOnlySpan<Point> points)
    {
        (Point2f c, float r) = FitEnclosingCircle.Compute(HullSklansky.Coords(points));
        return new EnclosingCircle(c, r);
    }

    /// <summary>Finds the minimum enclosing circle of a float point set, as the integer overload.</summary>
    /// <param name="points">Input points.</param>
    /// <returns>The circle.</returns>
    public static EnclosingCircle MinEnclosingCircle(ReadOnlySpan<Point2f> points)
    {
        (Point2f c, float r) = FitEnclosingCircle.Compute(HullSklansky.Coords(points));
        return new EnclosingCircle(c, r);
    }

    /// <summary>
    /// Fits an ellipse to a point set in the least-squares sense, as OpenCV <c>fitEllipse</c>.
    /// Five points use the Direct fit; more points use the general conic fit with a Jacobi SVD.
    /// </summary>
    /// <param name="points">At least 5 input points.</param>
    /// <returns>
    /// The ellipse as a rotated rectangle: the size holds the full axes (width ≤ height) and the
    /// angle is in degrees.
    /// </returns>
    /// <exception cref="ArgumentException">There are fewer than 5 points.</exception>
    /// <remarks>
    /// When the point set is degenerate, the fit adds small offsets from the per-thread OpenCV
    /// random generator (<c>theRNG</c>), as OpenCV does. OpenCV builds with LAPACK use a different
    /// SVD for 25 points or more; the results can then differ in the last bits.
    /// </remarks>
    public static RotatedRect FitEllipse(ReadOnlySpan<Point> points) => FitEllipseCore.Fit(ToFloat(points));

    /// <summary>Fits an ellipse to a float point set, as the integer overload.</summary>
    /// <param name="points">At least 5 input points.</param>
    /// <returns>The ellipse.</returns>
    public static RotatedRect FitEllipse(ReadOnlySpan<Point2f> points) => FitEllipseCore.Fit(points);

    /// <summary>
    /// Fits an ellipse with the Approximate Mean Square method, as OpenCV <c>fitEllipseAMS</c>.
    /// When the result is not an ellipse, the function returns the Direct fit.
    /// </summary>
    /// <param name="points">At least 5 input points.</param>
    /// <returns>The ellipse.</returns>
    /// <exception cref="ArgumentException">There are fewer than 5 points.</exception>
    public static RotatedRect FitEllipseAms(ReadOnlySpan<Point> points) => FitEllipseCore.Ams(ToFloat(points));

    /// <summary>Fits an AMS ellipse to a float point set, as the integer overload.</summary>
    /// <param name="points">At least 5 input points.</param>
    /// <returns>The ellipse.</returns>
    public static RotatedRect FitEllipseAms(ReadOnlySpan<Point2f> points) => FitEllipseCore.Ams(points);

    /// <summary>Fits an ellipse with the Direct least-squares method, as OpenCV <c>fitEllipseDirect</c>.</summary>
    /// <param name="points">At least 5 input points.</param>
    /// <returns>The ellipse.</returns>
    /// <exception cref="ArgumentException">There are fewer than 5 points.</exception>
    public static RotatedRect FitEllipseDirect(ReadOnlySpan<Point> points) => FitEllipseCore.Direct(ToFloat(points));

    /// <summary>Fits a Direct ellipse to a float point set, as the integer overload.</summary>
    /// <param name="points">At least 5 input points.</param>
    /// <returns>The ellipse.</returns>
    public static RotatedRect FitEllipseDirect(ReadOnlySpan<Point2f> points) => FitEllipseCore.Direct(points);

    /// <summary>
    /// Fits a line to a 2-D point set, as OpenCV <c>fitLine</c>. <see cref="DistanceTypes.L2"/>
    /// is a least-squares fit. The other types use iteratively reweighted least squares from 20
    /// seeded random starts (OpenCV <c>RNG(-1)</c>), so the result is deterministic.
    /// </summary>
    /// <param name="points">At least one input point.</param>
    /// <param name="distType">
    /// <see cref="DistanceTypes.L2"/>, <see cref="DistanceTypes.L1"/>, <see cref="DistanceTypes.L12"/>,
    /// <see cref="DistanceTypes.Fair"/>, <see cref="DistanceTypes.Welsch"/> or <see cref="DistanceTypes.Huber"/>.
    /// </param>
    /// <param name="param">Numerical parameter C of Fair, Welsch and Huber; 0 selects the OpenCV default.</param>
    /// <param name="reps">Accuracy for the distance between the origin and the line; 0 selects 1.</param>
    /// <param name="aeps">Accuracy for the angle; 0 selects 0.01.</param>
    /// <returns>The line.</returns>
    /// <exception cref="ArgumentException">The point set is empty, or the distance type is not supported.</exception>
    public static FittedLine FitLine(ReadOnlySpan<Point> points, DistanceTypes distType, double param, double reps, double aeps)
        => FitLine(ToFloat(points), distType, param, reps, aeps);

    /// <summary>Fits a line to a float point set, as the integer overload.</summary>
    /// <param name="points">At least one input point.</param>
    /// <param name="distType">Distance type.</param>
    /// <param name="param">Numerical parameter C.</param>
    /// <param name="reps">Radius accuracy.</param>
    /// <param name="aeps">Angle accuracy.</param>
    /// <returns>The line.</returns>
    public static FittedLine FitLine(ReadOnlySpan<Point2f> points, DistanceTypes distType, double param, double reps, double aeps)
    {
        if (points.IsEmpty)
        {
            throw new ArgumentException("The point set must not be empty.", nameof(points));
        }

        Span<float> line = stackalloc float[4];
        FitLine2D.Fit(points, distType, (float)param, (float)reps, (float)aeps, line);
        return new FittedLine(line[0], line[1], line[2], line[3]);
    }

    /// <summary>
    /// Tests the position of a point relative to a contour, as OpenCV <c>pointPolygonTest</c>.
    /// </summary>
    /// <param name="contour">Contour vertices; the contour is closed.</param>
    /// <param name="pt">The point.</param>
    /// <param name="measureDist">
    /// False: return +1 (inside), -1 (outside) or 0 (on an edge). True: return the signed
    /// distance to the nearest edge (positive inside).
    /// </param>
    /// <returns>
    /// The test result. An empty contour gives -1, or <c>-double.MaxValue</c> when
    /// <paramref name="measureDist"/> is true.
    /// </returns>
    public static double PointPolygonTest(ReadOnlySpan<Point> contour, Point2f pt, bool measureDist)
        => FitPointPolygonTest.Compute(HullSklansky.Coords(contour), pt, measureDist);

    /// <summary>Tests the position of a point relative to a float contour, as the integer overload.</summary>
    /// <param name="contour">Contour vertices.</param>
    /// <param name="pt">The point.</param>
    /// <param name="measureDist">Return the signed distance instead of +1/-1/0.</param>
    /// <returns>The test result.</returns>
    public static double PointPolygonTest(ReadOnlySpan<Point2f> contour, Point2f pt, bool measureDist)
        => FitPointPolygonTest.Compute(HullSklansky.Coords(contour), pt, measureDist);

    private static int[] HullIndices<T>(ReadOnlySpan<T> xy, bool clockwise, bool returnPoints)
        where T : unmanaged, INumber<T>
    {
        int total = xy.Length / 2;
        if (total == 0)
        {
            return [];
        }

        int[] rented = ArrayPool<int>.Shared.Rent(total);
        try
        {
            int nout = HullSklansky.ConvexHull(xy, clockwise, returnPoints, rented);
            return rented.AsSpan(0, nout).ToArray();
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    private static TPoint[] HullPoints<TPoint, T>(ReadOnlySpan<TPoint> points, ReadOnlySpan<T> xy, bool clockwise)
        where T : unmanaged, INumber<T>
    {
        int total = points.Length;
        if (total == 0)
        {
            return [];
        }

        int[] idx = ArrayPool<int>.Shared.Rent(total);
        try
        {
            int nout = HullSklansky.ConvexHull(xy, clockwise, true, idx);
            var result = new TPoint[nout];
            for (int i = 0; i < nout; i++)
            {
                result[i] = points[idx[i]];
            }

            return result;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(idx);
        }
    }

    private static Point2f[] ToFloat(ReadOnlySpan<Point> points)
    {
        var result = new Point2f[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            result[i] = new Point2f(points[i].X, points[i].Y);
        }

        return result;
    }
}
