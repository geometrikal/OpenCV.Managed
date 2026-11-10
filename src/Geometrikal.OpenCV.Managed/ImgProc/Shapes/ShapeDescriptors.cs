// This implementation is ported from the OpenCV shape descriptors.
//
// Original sources:
//   modules/geometry/src/approx.cpp        (approxPolyDP)
//   modules/geometry/src/geometry.cpp      (contourArea, arcLength, boundingRect)
//   modules/geometry/src/moments.cpp       (moments, HuMoments)
//   modules/geometry/src/matchcontours.cpp (matchShapes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Contour and image shape descriptors: polygon approximation, perimeter, area, bounding
/// rectangle, moments, Hu invariants and Hu-moment shape matching. A managed port of the OpenCV 5.0
/// <c>geometry</c> module functions <c>approxPolyDP</c>, <c>arcLength</c>, <c>contourArea</c>,
/// <c>boundingRect</c>, <c>moments</c>, <c>HuMoments</c> and <c>matchShapes</c>.
/// </summary>
/// <remarks>
/// Point sequences use image coordinates (X to the right, Y downward). No function modifies its
/// input. Results are bit-exact with OpenCV 5.0 built without IPP and without a HAL replacement.
/// </remarks>
public static class ShapeDescriptors
{
    /// <summary>
    /// Approximates a polygonal curve with the Ramer-Douglas-Peucker algorithm, as OpenCV
    /// <c>approxPolyDP</c>.
    /// </summary>
    /// <param name="curve">The input points. An empty curve gives an empty result.</param>
    /// <param name="epsilon">
    /// The maximum distance between the curve and its approximation. It must be in [0, 1e30).
    /// </param>
    /// <param name="closed">
    /// <c>true</c> when the curve is closed. An open curve whose first and last points are equal is
    /// processed as a closed curve, and its last point is kept.
    /// </param>
    /// <returns>A new array with a subset of the input points, in input order.</returns>
    /// <remarks>
    /// For a closed curve, OpenCV starts from the point found by three alternate farthest-point
    /// searches from point 0. A final pass removes points on almost straight lines. The port keeps
    /// the start point, the slice stack order and the clean-up pass, so the output points and their
    /// order match OpenCV.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is out of range.</exception>
    public static Point[] ApproxPolyDP(ReadOnlySpan<Point> curve, double epsilon, bool closed)
        => ApproxPolyDP<Point, DescrIntPointOps>(curve, epsilon, closed);

    /// <inheritdoc cref="ApproxPolyDP(ReadOnlySpan{Point}, double, bool)"/>
    public static Point2f[] ApproxPolyDP(ReadOnlySpan<Point2f> curve, double epsilon, bool closed)
        => ApproxPolyDP<Point2f, DescrFloatPointOps>(curve, epsilon, closed);

    /// <summary>
    /// Calculates the perimeter of a closed curve or the length of an open curve, as OpenCV
    /// <c>arcLength</c>.
    /// </summary>
    /// <param name="curve">The input points.</param>
    /// <param name="closed"><c>true</c> to include the segment from the last point to the first point.</param>
    /// <returns>The length. A curve with 0 or 1 point has length 0.</returns>
    /// <remarks>
    /// The coordinates convert to <c>float</c>. Each segment length is a <c>float</c> square root;
    /// the sum is <c>double</c>.
    /// </remarks>
    public static double ArcLength(ReadOnlySpan<Point> curve, bool closed)
        => DescrPolygon.ArcLength<Point, DescrIntPointOps>(curve, closed);

    /// <inheritdoc cref="ArcLength(ReadOnlySpan{Point}, bool)"/>
    public static double ArcLength(ReadOnlySpan<Point2f> curve, bool closed)
        => DescrPolygon.ArcLength<Point2f, DescrFloatPointOps>(curve, closed);

    /// <summary>
    /// Calculates the area of a polygon with the shoelace formula, as OpenCV <c>contourArea</c>.
    /// </summary>
    /// <param name="contour">The polygon vertices. The polygon closes from the last vertex to the first vertex.</param>
    /// <param name="oriented">
    /// <c>true</c> to return the signed area <c>0.5 * sum(x[i-1] * y[i] - y[i-1] * x[i])</c>. The
    /// area is positive for a contour that is clockwise on screen (X to the right, Y downward).
    /// <c>false</c> returns the absolute value.
    /// </param>
    /// <returns>The area. An empty contour has area 0.</returns>
    /// <remarks>
    /// The coordinates convert to <c>float</c> and the products accumulate in <c>double</c>. A
    /// self-intersecting polygon gives the sum of the signed areas of its parts.
    /// </remarks>
    public static double ContourArea(ReadOnlySpan<Point> contour, bool oriented = false)
        => DescrPolygon.ContourArea<Point, DescrIntPointOps>(contour, oriented);

    /// <inheritdoc cref="ContourArea(ReadOnlySpan{Point}, bool)"/>
    public static double ContourArea(ReadOnlySpan<Point2f> contour, bool oriented = false)
        => DescrPolygon.ContourArea<Point2f, DescrFloatPointOps>(contour, oriented);

    /// <summary>
    /// Calculates the smallest upright integer rectangle that contains a point set, as OpenCV
    /// <c>boundingRect</c>.
    /// </summary>
    /// <param name="points">The input points.</param>
    /// <returns>
    /// <c>Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)</c>. An empty set gives
    /// <c>Rect(0, 0, 0, 0)</c>.
    /// </returns>
    public static Rect BoundingRect(ReadOnlySpan<Point> points) => DescrPolygon.BoundingRect(points);

    /// <summary>
    /// Calculates an upright integer rectangle for a <c>float</c> point set, as OpenCV
    /// <c>boundingRect</c>.
    /// </summary>
    /// <param name="points">The input points. NaN coordinates are not supported.</param>
    /// <returns>
    /// <c>Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)</c> with all four limits rounded down
    /// (<c>cvFloor</c>). A point with a fractional coordinate at the maximum can therefore lie
    /// outside the rectangle, as in OpenCV. An empty set gives <c>Rect(0, 0, 0, 0)</c>.
    /// </returns>
    public static Rect BoundingRect(ReadOnlySpan<Point2f> points) => DescrPolygon.BoundingRect(points);

    /// <summary>
    /// Calculates the bounding rectangle of the nonzero pixels of a mask, as OpenCV
    /// <c>boundingRect</c> on an 8-bit image.
    /// </summary>
    /// <param name="mask">A single-channel mask. Regions of interest are supported.</param>
    /// <returns>
    /// The smallest rectangle that contains all nonzero pixels, in mask coordinates. A mask without
    /// nonzero pixels gives <c>Rect(0, 0, 0, 0)</c>.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="mask"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="mask"/> has more than one channel.</exception>
    public static Rect BoundingRect(Mat<byte> mask)
    {
        ArgumentNullException.ThrowIfNull(mask);
        RequireSingleChannel(mask.Channels, nameof(mask));
        return DescrMaskBounds.BoundingRect(mask);
    }

    /// <summary>
    /// Calculates the moments of a polygon with Green's theorem, as OpenCV <c>moments</c> on a
    /// point vector.
    /// </summary>
    /// <param name="contour">The polygon vertices. The polygon closes from the last vertex to the first vertex.</param>
    /// <returns>
    /// The moments of the polygon area. The sign does not depend on the orientation. When the
    /// absolute area is not greater than <c>FLT_EPSILON</c> (for example fewer than 3 points or
    /// collinear points), all moments are 0.
    /// </returns>
    public static Moments Moments(ReadOnlySpan<Point> contour)
        => DescrPolygon.ContourMoments<Point, DescrIntPointOps>(contour);

    /// <inheritdoc cref="Moments(ReadOnlySpan{Point})"/>
    public static Moments Moments(ReadOnlySpan<Point2f> contour)
        => DescrPolygon.ContourMoments<Point2f, DescrFloatPointOps>(contour);

    /// <summary>
    /// Calculates the raster moments of a single-channel image, as OpenCV <c>moments</c> on an
    /// image.
    /// </summary>
    /// <typeparam name="T"><c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <param name="image">A single-channel image. Regions of interest are supported.</param>
    /// <param name="binaryImage">
    /// <c>true</c> to use 1 for each nonzero pixel and 0 for each zero pixel. NaN counts as nonzero.
    /// </param>
    /// <returns>The moments. Pixel (x, y) has coordinates x and y.</returns>
    /// <remarks>
    /// The result is bit-exact with OpenCV for all pixel types and does not depend on the thread
    /// count. OpenCV treats a continuous single-channel <c>float</c> or <c>int</c> matrix with two
    /// columns as a point vector; this method always treats <paramref name="image"/> as an image.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="image"/> has more than one channel.</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not a supported pixel type.</exception>
    public static Moments Moments<T>(Mat<T> image, bool binaryImage = false)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        RequireSingleChannel(image.Channels, nameof(image));
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(short)
            && typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"Image moments do not support the pixel type {typeof(T).Name}.");
        }

        return DescrImageMoments.Compute(image, binaryImage);
    }

    /// <summary>Calculates the seven Hu invariants, as OpenCV <c>HuMoments</c>.</summary>
    /// <param name="moments">Moments from <see cref="Moments(ReadOnlySpan{Point})"/> or an overload.</param>
    /// <returns>A new array of 7 values. Hu invariant 7 changes sign under reflection.</returns>
    public static double[] HuMoments(in Moments moments)
    {
        double[] hu = new double[7];
        DescrHuMatch.HuMoments(moments, hu);
        return hu;
    }

    /// <summary>Writes the seven Hu invariants to <paramref name="destination"/>.</summary>
    /// <param name="moments">The moments.</param>
    /// <param name="destination">A span of at least 7 elements.</param>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is shorter than 7.</exception>
    public static void HuMoments(in Moments moments, Span<double> destination)
    {
        if (destination.Length < 7)
        {
            throw new ArgumentException("The destination must hold 7 values.", nameof(destination));
        }

        DescrHuMatch.HuMoments(moments, destination);
    }

    /// <summary>
    /// Compares two contours with their Hu invariants, as OpenCV <c>matchShapes</c>.
    /// </summary>
    /// <param name="contour1">The first contour.</param>
    /// <param name="contour2">The second contour.</param>
    /// <param name="method">The comparison method.</param>
    /// <returns>
    /// The distance. 0 for identical invariants. <see cref="double.MaxValue"/> when exactly one
    /// shape has all Hu invariants equal to 0.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="method"/> is not defined.</exception>
    public static double MatchShapes(ReadOnlySpan<Point> contour1, ReadOnlySpan<Point> contour2, ShapeMatchModes method)
    {
        RequireMethod(method);
        return DescrHuMatch.MatchShapes(Moments(contour1), Moments(contour2), method);
    }

    /// <inheritdoc cref="MatchShapes(ReadOnlySpan{Point}, ReadOnlySpan{Point}, ShapeMatchModes)"/>
    public static double MatchShapes(ReadOnlySpan<Point2f> contour1, ReadOnlySpan<Point2f> contour2, ShapeMatchModes method)
    {
        RequireMethod(method);
        return DescrHuMatch.MatchShapes(Moments(contour1), Moments(contour2), method);
    }

    /// <summary>
    /// Compares two grayscale images with the Hu invariants of their raster moments (not binary),
    /// as OpenCV <c>matchShapes</c> on images.
    /// </summary>
    /// <typeparam name="T"><c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <param name="image1">The first single-channel image.</param>
    /// <param name="image2">The second single-channel image.</param>
    /// <param name="method">The comparison method.</param>
    /// <returns>The distance, as <see cref="MatchShapes(ReadOnlySpan{Point}, ReadOnlySpan{Point}, ShapeMatchModes)"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="method"/> is not defined.</exception>
    public static double MatchShapes<T>(Mat<T> image1, Mat<T> image2, ShapeMatchModes method)
        where T : unmanaged
    {
        RequireMethod(method);
        return DescrHuMatch.MatchShapes(Moments(image1), Moments(image2), method);
    }

    /// <summary>
    /// Compares two shapes from their moments. Use this overload to compare a contour with an
    /// image.
    /// </summary>
    /// <param name="moments1">The moments of the first shape.</param>
    /// <param name="moments2">The moments of the second shape.</param>
    /// <param name="method">The comparison method.</param>
    /// <returns>The distance, as <see cref="MatchShapes(ReadOnlySpan{Point}, ReadOnlySpan{Point}, ShapeMatchModes)"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="method"/> is not defined.</exception>
    public static double MatchShapes(in Moments moments1, in Moments moments2, ShapeMatchModes method)
    {
        RequireMethod(method);
        return DescrHuMatch.MatchShapes(moments1, moments2, method);
    }

    private static TP[] ApproxPolyDP<TP, TOps>(ReadOnlySpan<TP> curve, double epsilon, bool closed)
        where TP : unmanaged
        where TOps : struct, IDescrPointOps<TP>
    {
        if (epsilon < 0.0 || !(epsilon < 1e30))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be in [0, 1e30).");
        }

        if (curve.IsEmpty)
        {
            return [];
        }

        TP[] buffer = new TP[curve.Length];
        int n = DescrApproxPoly.Run<TP, TOps>(curve, buffer, closed, epsilon);
        if (n == buffer.Length)
        {
            return buffer;
        }

        return buffer.AsSpan(0, n).ToArray();
    }

    private static void RequireSingleChannel(int channels, string name)
    {
        if (channels != 1)
        {
            throw new ArgumentException("The image must have one channel.", name);
        }
    }

    private static void RequireMethod(ShapeMatchModes method)
    {
        if (method is not (ShapeMatchModes.ContoursMatchI1 or ShapeMatchModes.ContoursMatchI2 or ShapeMatchModes.ContoursMatchI3))
        {
            throw new ArgumentOutOfRangeException(nameof(method), method, "Unknown comparison method.");
        }
    }
}
