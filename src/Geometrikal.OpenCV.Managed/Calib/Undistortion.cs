// This implementation is ported from the OpenCV lens undistortion functions.
//
// Original sources:
//   modules/imgproc/src/undistort.dispatch.cpp (initUndistortRectifyMap, undistort)
//   modules/imgproc/src/undistort.simd.hpp (initUndistortRectifyMapComputer)
//   modules/geometry/src/pinhole.cpp (getDefaultNewCameraMatrix, undistortPoints, undistortImagePoints)
//   modules/geometry/src/calibration_base.cpp (getUndistortRectangles, getOptimalNewCameraMatrix)
//   modules/geometry/src/distortion_model.hpp (computeTiltProjectionMatrix)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/, src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Lens distortion correction for the pinhole camera model, as the OpenCV functions
/// <c>initUndistortRectifyMap</c>, <c>undistort</c>, <c>undistortPoints</c>,
/// <c>undistortImagePoints</c>, <c>getDefaultNewCameraMatrix</c> and
/// <c>getOptimalNewCameraMatrix</c>.
/// </summary>
/// <remarks>
/// <para><b>Camera model.</b> The camera matrix is a 3 × 3 <c>double</c> matrix
/// <c>[fx 0 cx; 0 fy cy; 0 0 1]</c>. The distortion coefficients are
/// <c>(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]])</c>: 4, 5, 8, 12 or 14
/// values. An empty span means no distortion.</para>
/// <para><b>Bit-exactness.</b> The maps follow the OpenCV 5.0 baseline SSE build (AVX disabled):
/// pixels in complete blocks of four use the 128-bit <c>v_float64</c> operation order, and the
/// remaining pixels of a row use the scalar order. See <c>Docs/Notes/UndistortPortNotes.md</c>.</para>
/// <para>No function modifies its inputs.</para>
/// </remarks>
public static class Undistortion
{
    /// <summary>The OpenCV <c>undistortPoints</c> default: 5 iterations (<c>MAX_ITER</c>, epsilon 0.01 unused).</summary>
    public static readonly TermCriteria DefaultPointCriteria = new(TermCriteriaType.MaxIter, 5, 0.01);

    /// <summary>
    /// Computes the undistortion and rectification maps as two <c>CV_32FC1</c> maps, as OpenCV
    /// <c>initUndistortRectifyMap</c> with <c>m1type = CV_32FC1</c>.
    /// </summary>
    /// <param name="cameraMatrix">Camera matrix A (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="r">Rectification rotation R (3 × 3), or null for the identity.</param>
    /// <param name="newCameraMatrix">
    /// New camera matrix (3 × 3, or 3 × 4 of which the left 3 × 3 block is used), or null for
    /// <see cref="GetDefaultNewCameraMatrix"/> of A with a centred principal point.
    /// </param>
    /// <param name="size">Map size (the undistorted image size).</param>
    /// <returns>The x map and the y map: for each destination pixel, the source coordinates.</returns>
    public static (Mat<float> MapX, Mat<float> MapY) InitUndistortRectifyMap(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r,
        Mat<double>? newCameraMatrix,
        Size size)
    {
        UndistortMapParams p = CreateMapParams(cameraMatrix, distCoeffs, r, newCameraMatrix, size);
        var mapX = new Mat<float>(size.Width, size.Height, 1, clear: false);
        var mapY = new Mat<float>(size.Width, size.Height, 1, clear: false);
        UndistortMapKernel.Run(p, size.Width, size.Height, UndistortMapKind.FloatPair, mapX, mapY, null, null);
        return (mapX, mapY);
    }

    /// <summary>
    /// Computes the undistortion and rectification map as one two-channel (x, y) map, as OpenCV
    /// <c>initUndistortRectifyMap</c> with <c>m1type = CV_32FC2</c>.
    /// </summary>
    /// <inheritdoc cref="InitUndistortRectifyMap"/>
    /// <returns>The interleaved (x, y) map.</returns>
    public static Mat<float> InitUndistortRectifyMapInterleaved(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r,
        Mat<double>? newCameraMatrix,
        Size size)
    {
        UndistortMapParams p = CreateMapParams(cameraMatrix, distCoeffs, r, newCameraMatrix, size);
        var map = new Mat<float>(size.Width, size.Height, 2, clear: false);
        UndistortMapKernel.Run(p, size.Width, size.Height, UndistortMapKind.FloatInterleaved, map, null, null, null);
        return map;
    }

    /// <summary>
    /// Computes the undistortion and rectification maps in fixed point, as OpenCV
    /// <c>initUndistortRectifyMap</c> with <c>m1type = CV_16SC2</c>. The maps are input for
    /// <see cref="Geometry.Remap{T}(Mat{T}, Mat{short}, Mat{ushort}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>.
    /// </summary>
    /// <remarks>
    /// Each coordinate is <c>cvRound(c·32)</c>. The integer pair holds <c>c·32 &gt;&gt; 5</c>; the
    /// fraction map holds <c>(iy &amp; 31)·32 + (ix &amp; 31)</c>. In blocks of four pixels the
    /// integer parts saturate to the 16-bit range; in the last <c>width mod 4</c> pixels of a row
    /// they wrap, as in OpenCV.
    /// </remarks>
    /// <inheritdoc cref="InitUndistortRectifyMap"/>
    /// <returns>The two-channel integer map and the fraction map.</returns>
    public static (Mat<short> Xy, Mat<ushort> Fractions) InitUndistortRectifyMapFixedPoint(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r,
        Mat<double>? newCameraMatrix,
        Size size)
    {
        UndistortMapParams p = CreateMapParams(cameraMatrix, distCoeffs, r, newCameraMatrix, size);
        var xy = new Mat<short>(size.Width, size.Height, 2, clear: false);
        var fractions = new Mat<ushort>(size.Width, size.Height, 1, clear: false);
        UndistortMapKernel.Run(p, size.Width, size.Height, UndistortMapKind.FixedPoint, null, null, xy, fractions);
        return (xy, fractions);
    }

    /// <summary>
    /// Corrects the lens distortion of an image, as OpenCV <c>undistort</c>.
    /// </summary>
    /// <remarks>
    /// As OpenCV, the function processes stripes of <c>max(1, 4096 / width)</c> rows. For each
    /// stripe it computes fixed-point maps with the principal point y of the new camera matrix
    /// shifted by the stripe start, then remaps with bilinear interpolation and a constant zero
    /// border. The result can therefore differ in the last bits of the map from one full
    /// <see cref="InitUndistortRectifyMapFixedPoint"/> map followed by one remap.
    /// Supported element types are those of <see cref="Geometry.Remap{T}(Mat{T}, Mat{short}, Mat{ushort}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>.
    /// </remarks>
    /// <param name="source">Source image. It is not modified.</param>
    /// <param name="cameraMatrix">Camera matrix A (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="newCameraMatrix">New camera matrix (3 × 3), or null to use A.</param>
    /// <returns>A new undistorted image of the source size and type.</returns>
    public static Mat<T> Undistort<T>(
        Mat<T> source,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? newCameraMatrix = null)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        double[] k = CameraDistortion.Expand(distCoeffs, nameof(distCoeffs));
        double[] ar = newCameraMatrix is null ? (double[])a.Clone() : CameraDistortion.Read3x3(newCameraMatrix, nameof(newCameraMatrix));
        var dst = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);

        int cols = source.Width, rows = source.Height;
        int stripe0 = Math.Min(Math.Max(1, (1 << 12) / Math.Max(cols, 1)), rows);
        int stripes = (rows + stripe0 - 1) / stripe0;
        int chunks = Math.Min(stripes, Math.Max(1, CvParallel.MaxDegreeOfParallelism) * 4);
        double v0 = ar[5];
        double[] identity = CameraDistortion.Identity();

        ParallelRows.ForEachSerial(chunks, c =>
        {
            int first = (int)((long)stripes * c / chunks);
            int last = (int)((long)stripes * (c + 1) / chunks);
            using PooledMat<short> xyPooled = PooledMat<short>.Rent(cols, stripe0, 2);
            using PooledMat<ushort> fractionsPooled = PooledMat<ushort>.Rent(cols, stripe0, 1);
            Mat<short> xy = xyPooled.Mat;
            Mat<ushort> fractions = fractionsPooled.Mat;
            var arStripe = (double[])ar.Clone();
            for (int s = first; s < last; s++)
            {
                int y = s * stripe0;
                int stripe = Math.Min(stripe0, rows - y);
                arStripe[5] = v0 - y;
                UndistortMapParams p = UndistortMapParams.Create(a, k, identity, arStripe);
                Mat<short> xyPart = stripe == stripe0 ? xy : xy.Roi(0, 0, cols, stripe);
                Mat<ushort> fPart = stripe == stripe0 ? fractions : fractions.Roi(0, 0, cols, stripe);
                UndistortMapKernel.Run(p, cols, stripe, UndistortMapKind.FixedPoint, null, null, xyPart, fPart);
                Geometry.Remap(source, dst.Roi(0, y, cols, stripe), xyPart, fPart, InterpolationFlags.Linear, BorderType.Constant);
            }
        });

        return dst;
    }

    /// <summary>
    /// Computes ideal point coordinates from observed (distorted) point coordinates, as OpenCV
    /// <c>undistortPoints</c>.
    /// </summary>
    /// <remarks>
    /// The distortion is inverted with a damped fixed-point iteration. The iteration stops after
    /// <see cref="TermCriteria.MaxCount"/> iterations when <see cref="TermCriteriaType.MaxIter"/>
    /// is set, and when the reprojection error is below <see cref="TermCriteria.Epsilon"/> when
    /// <see cref="TermCriteriaType.Eps"/> is set. Without <paramref name="r"/> and
    /// <paramref name="p"/> the result is in normalized coordinates. With them the result is
    /// <c>P·R·(x, y, 1)</c> after division by the third component.
    /// </remarks>
    /// <param name="source">Observed point coordinates.</param>
    /// <param name="destination">Output points; same length as <paramref name="source"/>. It may be the source span.</param>
    /// <param name="cameraMatrix">Camera matrix A (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="r">Rectification rotation R (3 × 3), or null.</param>
    /// <param name="p">New camera or projection matrix P (3 × 3 or 3 × 4), or null.</param>
    /// <param name="criteria">Termination criteria, or null for <see cref="DefaultPointCriteria"/>.</param>
    public static void UndistortPoints(
        ReadOnlySpan<Point2f> source,
        Span<Point2f> destination,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r = null,
        Mat<double>? p = null,
        TermCriteria? criteria = null)
    {
        RequireLength(source.Length, destination.Length);
        var kernel = new UndistortPointsKernel(cameraMatrix, distCoeffs, r, p, criteria ?? DefaultPointCriteria);
        RunPoints(kernel, source, destination);
    }

    /// <inheritdoc cref="UndistortPoints(ReadOnlySpan{Point2f}, Span{Point2f}, Mat{double}, ReadOnlySpan{double}, Mat{double}, Mat{double}, TermCriteria?)"/>
    public static void UndistortPoints(
        ReadOnlySpan<Point2d> source,
        Span<Point2d> destination,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r = null,
        Mat<double>? p = null,
        TermCriteria? criteria = null)
    {
        RequireLength(source.Length, destination.Length);
        var kernel = new UndistortPointsKernel(cameraMatrix, distCoeffs, r, p, criteria ?? DefaultPointCriteria);
        RunPoints(kernel, source, destination);
    }

    /// <summary>Computes ideal point coordinates, as OpenCV <c>undistortPoints</c>.</summary>
    /// <inheritdoc cref="UndistortPoints(ReadOnlySpan{Point2f}, Span{Point2f}, Mat{double}, ReadOnlySpan{double}, Mat{double}, Mat{double}, TermCriteria?)"/>
    /// <returns>A new array with the undistorted points.</returns>
    public static Point2f[] UndistortPoints(
        ReadOnlySpan<Point2f> source,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r = null,
        Mat<double>? p = null,
        TermCriteria? criteria = null)
    {
        var dst = new Point2f[source.Length];
        UndistortPoints(source, dst, cameraMatrix, distCoeffs, r, p, criteria);
        return dst;
    }

    /// <summary>Computes ideal point coordinates, as OpenCV <c>undistortPoints</c>.</summary>
    /// <inheritdoc cref="UndistortPoints(ReadOnlySpan{Point2f}, Span{Point2f}, Mat{double}, ReadOnlySpan{double}, Mat{double}, Mat{double}, TermCriteria?)"/>
    /// <returns>A new array with the undistorted points.</returns>
    public static Point2d[] UndistortPoints(
        ReadOnlySpan<Point2d> source,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r = null,
        Mat<double>? p = null,
        TermCriteria? criteria = null)
    {
        var dst = new Point2d[source.Length];
        UndistortPoints(source, dst, cameraMatrix, distCoeffs, r, p, criteria);
        return dst;
    }

    /// <summary>
    /// Computes undistorted pixel coordinates, as OpenCV <c>undistortImagePoints</c>:
    /// <see cref="UndistortPoints(ReadOnlySpan{Point2f}, Mat{double}, ReadOnlySpan{double}, Mat{double}, Mat{double}, TermCriteria?)"/>
    /// with no R and P equal to the camera matrix.
    /// </summary>
    /// <param name="source">Observed point coordinates.</param>
    /// <param name="cameraMatrix">Camera matrix A (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="criteria">Termination criteria, or null for <see cref="DefaultPointCriteria"/>.</param>
    /// <returns>A new array with the undistorted pixel coordinates.</returns>
    public static Point2f[] UndistortImagePoints(
        ReadOnlySpan<Point2f> source,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        TermCriteria? criteria = null)
        => UndistortPoints(source, cameraMatrix, distCoeffs, null, cameraMatrix, criteria);

    /// <inheritdoc cref="UndistortImagePoints(ReadOnlySpan{Point2f}, Mat{double}, ReadOnlySpan{double}, TermCriteria?)"/>
    public static Point2d[] UndistortImagePoints(
        ReadOnlySpan<Point2d> source,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        TermCriteria? criteria = null)
        => UndistortPoints(source, cameraMatrix, distCoeffs, null, cameraMatrix, criteria);

    /// <summary>
    /// Returns the default new camera matrix, as OpenCV <c>getDefaultNewCameraMatrix</c>.
    /// </summary>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="imageSize">Image size; used only when <paramref name="centerPrincipalPoint"/> is true.</param>
    /// <param name="centerPrincipalPoint">
    /// True: the principal point becomes <c>((width − 1)/2, (height − 1)/2)</c>. False: the
    /// result is a copy of the camera matrix.
    /// </param>
    /// <returns>A new 3 × 3 matrix.</returns>
    public static Mat<double> GetDefaultNewCameraMatrix(Mat<double> cameraMatrix, Size imageSize = default, bool centerPrincipalPoint = false)
    {
        double[] m = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        if (centerPrincipalPoint)
        {
            m[2] = (imageSize.Width - 1) * 0.5;
            m[5] = (imageSize.Height - 1) * 0.5;
        }

        return new Mat<double>(m, 3, 3);
    }

    /// <summary>
    /// Returns the new camera matrix for a free scaling parameter, as OpenCV
    /// <c>getOptimalNewCameraMatrix</c>.
    /// </summary>
    /// <remarks>
    /// The function undistorts 32 points on the image border (a 9 × 9 grid) with the default
    /// criteria and computes the inscribed and the circumscribed rectangles of the undistorted
    /// image. <paramref name="alpha"/> = 0 maps the inscribed rectangle to the new image (only
    /// valid pixels); <paramref name="alpha"/> = 1 maps the circumscribed rectangle (all source
    /// pixels).
    /// </remarks>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="imageSize">Original image size.</param>
    /// <param name="alpha">Free scaling parameter, from 0 to 1.</param>
    /// <param name="newImageSize">New image size; an empty size means <paramref name="imageSize"/>.</param>
    /// <param name="validPixRoi">Rectangle of the new image that contains only valid pixels, clipped to the new image.</param>
    /// <param name="centerPrincipalPoint">True: the principal point of the result is the new image centre.</param>
    /// <returns>A new 3 × 3 camera matrix.</returns>
    public static Mat<double> GetOptimalNewCameraMatrix(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Size imageSize,
        double alpha,
        Size newImageSize,
        out Rect validPixRoi,
        bool centerPrincipalPoint = false)
    {
        newImageSize = newImageSize.Width * newImageSize.Height != 0 ? newImageSize : imageSize;
        double[] m = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));

        if (centerPrincipalPoint)
        {
            double cx0 = m[2];
            double cy0 = m[5];
            double cx = (newImageSize.Width - 1) * 0.5;
            double cy = (newImageSize.Height - 1) * 0.5;

            GetUndistortRectangles(cameraMatrix, distCoeffs, cameraMatrix, imageSize, out RectD inner, out RectD outer);
            double s0 = StdMax(StdMax(StdMax(cx / (cx0 - inner.X), cy / (cy0 - inner.Y)), cx / (inner.X + inner.Width - cx0)), cy / (inner.Y + inner.Height - cy0));
            double s1 = StdMin(StdMin(StdMin(cx / (cx0 - outer.X), cy / (cy0 - outer.Y)), cx / (outer.X + outer.Width - cx0)), cy / (outer.Y + outer.Height - cy0));
            double s = s0 * (1 - alpha) + s1 * alpha;

            m[0] *= s;
            m[4] *= s;
            m[2] = cx;
            m[5] = cy;

            double ix = (inner.X - cx0) * s + cx;
            double iy = (inner.Y - cy0) * s + cy;
            double iw = inner.Width * s;
            double ih = inner.Height * s;
            var roi = new Rect(ScalarConverter.CvCeil(ix), ScalarConverter.CvCeil(iy), ScalarConverter.CvFloor(iw), ScalarConverter.CvFloor(ih));
            validPixRoi = Intersect(roi, new Rect(0, 0, newImageSize.Width, newImageSize.Height));
        }
        else
        {
            GetUndistortRectangles(cameraMatrix, distCoeffs, null, imageSize, out RectD inner, out RectD outer);

            double fx0 = (newImageSize.Width - 1) / inner.Width;
            double fy0 = (newImageSize.Height - 1) / inner.Height;
            double cx0 = -fx0 * inner.X;
            double cy0 = -fy0 * inner.Y;

            double fx1 = (newImageSize.Width - 1) / outer.Width;
            double fy1 = (newImageSize.Height - 1) / outer.Height;
            double cx1 = -fx1 * outer.X;
            double cy1 = -fy1 * outer.Y;

            m[0] = fx0 * (1 - alpha) + fx1 * alpha;
            m[4] = fy0 * (1 - alpha) + fy1 * alpha;
            m[2] = cx0 * (1 - alpha) + cx1 * alpha;
            m[5] = cy0 * (1 - alpha) + cy1 * alpha;

            GetUndistortRectangles(cameraMatrix, distCoeffs, new Mat<double>((double[])m.Clone(), 3, 3), imageSize, out RectD inner2, out _);
            var roi = new Rect(
                ScalarConverter.CvRound(inner2.X),
                ScalarConverter.CvRound(inner2.Y),
                ScalarConverter.CvRound(inner2.Width),
                ScalarConverter.CvRound(inner2.Height));
            validPixRoi = Intersect(roi, new Rect(0, 0, newImageSize.Width, newImageSize.Height));
        }

        return new Mat<double>(m, 3, 3);
    }

    /// <inheritdoc cref="GetOptimalNewCameraMatrix(Mat{double}, ReadOnlySpan{double}, Size, double, Size, out Rect, bool)"/>
    public static Mat<double> GetOptimalNewCameraMatrix(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Size imageSize,
        double alpha,
        Size newImageSize = default,
        bool centerPrincipalPoint = false)
        => GetOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha, newImageSize, out _, centerPrincipalPoint);

    private static UndistortMapParams CreateMapParams(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? r,
        Mat<double>? newCameraMatrix,
        Size size)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size.Width, nameof(size));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size.Height, nameof(size));
        double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        double[] k = CameraDistortion.Expand(distCoeffs, nameof(distCoeffs));
        double[] ar = newCameraMatrix is null
            ? CameraDistortion.Read3x3(GetDefaultNewCameraMatrix(cameraMatrix, size, true), nameof(newCameraMatrix))
            : CameraDistortion.ReadLeft3x3(newCameraMatrix, nameof(newCameraMatrix));
        double[] rr = r is null ? CameraDistortion.Identity() : CameraDistortion.Read3x3(r, nameof(r));
        return UndistortMapParams.Create(a, k, rr, ar);
    }

    private static void RunPoints(UndistortPointsKernel kernel, ReadOnlySpan<Point2f> source, Span<Point2f> destination)
    {
        int n = source.Length;
        if (ParallelRows.IsSerial(n, PointWork))
        {
            for (int i = 0; i < n; i++)
            {
                kernel.Undistort(source[i].X, source[i].Y, out double x, out double y);
                destination[i] = new Point2f((float)x, (float)y);
            }

            return;
        }

        Point2f[] buffer = ArrayPool<Point2f>.Shared.Rent(n);
        try
        {
            source.CopyTo(buffer);
            ParallelRows.For(n, PointWork, (start, end) =>
            {
                for (int i = start; i < end; i++)
                {
                    kernel.Undistort(buffer[i].X, buffer[i].Y, out double x, out double y);
                    buffer[i] = new Point2f((float)x, (float)y);
                }
            });
            buffer.AsSpan(0, n).CopyTo(destination);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(buffer);
        }
    }

    private static void RunPoints(UndistortPointsKernel kernel, ReadOnlySpan<Point2d> source, Span<Point2d> destination)
    {
        int n = source.Length;
        if (ParallelRows.IsSerial(n, PointWork))
        {
            for (int i = 0; i < n; i++)
            {
                kernel.Undistort(source[i].X, source[i].Y, out double x, out double y);
                destination[i] = new Point2d(x, y);
            }

            return;
        }

        Point2d[] buffer = ArrayPool<Point2d>.Shared.Rent(n);
        try
        {
            source.CopyTo(buffer);
            ParallelRows.For(n, PointWork, (start, end) =>
            {
                for (int i = start; i < end; i++)
                {
                    kernel.Undistort(buffer[i].X, buffer[i].Y, out double x, out double y);
                    buffer[i] = new Point2d(x, y);
                }
            });
            buffer.AsSpan(0, n).CopyTo(destination);
        }
        finally
        {
            ArrayPool<Point2d>.Shared.Return(buffer);
        }
    }

    private const long PointWork = 256;

    private static void RequireLength(int source, int destination)
    {
        if (source != destination)
        {
            throw new ArgumentException("The destination must have the source length.", nameof(destination));
        }
    }

    /// <summary>OpenCV <c>getUndistortRectangles</c> with no R.</summary>
    private static void GetUndistortRectangles(
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Mat<double>? newCameraMatrix,
        Size imageSize,
        out RectD inner,
        out RectD outer)
    {
        const int N = 9;
        Span<Point2d> pts = stackalloc Point2d[4 * (N - 1)];
        double stepX = (imageSize.Width - 1) / (double)(N - 1);
        double stepY = (imageSize.Height - 1) / (double)(N - 1);
        int k = 0;
        for (int y = 0; y < N; y++)
        {
            for (int x = 0; x < N; x++)
            {
                if (x != 0 && x != N - 1 && y != 0 && y != N - 1)
                {
                    continue;
                }

                pts[k++] = new Point2d(x * stepX, y * stepY);
            }
        }

        var kernel = new UndistortPointsKernel(cameraMatrix, distCoeffs, null, newCameraMatrix, DefaultPointCriteria);
        for (int i = 0; i < pts.Length; i++)
        {
            kernel.Undistort(pts[i].X, pts[i].Y, out double ux, out double uy);
            pts[i] = new Point2d(ux, uy);
        }

        double iX0 = -float.MaxValue, iX1 = float.MaxValue, iY0 = -float.MaxValue, iY1 = float.MaxValue;
        double oX0 = float.MaxValue, oX1 = -float.MaxValue, oY0 = float.MaxValue, oY1 = -float.MaxValue;
        k = 0;
        for (int y = 0; y < N; y++)
        {
            for (int x = 0; x < N; x++)
            {
                if (x != 0 && x != N - 1 && y != 0 && y != N - 1)
                {
                    continue;
                }

                Point2d p = pts[k++];
                oX0 = MacroMin(oX0, p.X);
                oX1 = MacroMax(oX1, p.X);
                oY0 = MacroMin(oY0, p.Y);
                oY1 = MacroMax(oY1, p.Y);

                if (x == 0)
                {
                    iX0 = MacroMax(iX0, p.X);
                }

                if (x == N - 1)
                {
                    iX1 = MacroMin(iX1, p.X);
                }

                if (y == 0)
                {
                    iY0 = MacroMax(iY0, p.Y);
                }

                if (y == N - 1)
                {
                    iY1 = MacroMin(iY1, p.Y);
                }
            }
        }

        inner = new RectD(iX0, iY0, iX1 - iX0, iY1 - iY0);
        outer = new RectD(oX0, oY0, oX1 - oX0, oY1 - oY0);
    }

    private static double MacroMin(double a, double b) => a > b ? b : a;

    private static double MacroMax(double a, double b) => a < b ? b : a;

    private static double StdMax(double a, double b) => a < b ? b : a;

    private static double StdMin(double a, double b) => b < a ? b : a;

    /// <summary>OpenCV <c>Rect &amp;= Rect</c>.</summary>
    private static Rect Intersect(Rect a, Rect b)
    {
        if (a.Width <= 0 || a.Height <= 0 || b.Width <= 0 || b.Height <= 0)
        {
            return default;
        }

        Rect xMin = a.X < b.X ? a : b, xMax = a.X < b.X ? b : a;
        Rect yMin = a.Y < b.Y ? a : b, yMax = a.Y < b.Y ? b : a;
        if ((xMin.X < 0 && xMin.X + xMin.Width < xMax.X) || (yMin.Y < 0 && yMin.Y + yMin.Height < yMax.Y))
        {
            return default;
        }

        int w = Math.Min(xMin.Width - (xMax.X - xMin.X), xMax.Width);
        int h = Math.Min(yMin.Height - (yMax.Y - yMin.Y), yMax.Height);
        return w <= 0 || h <= 0 ? default : new Rect(xMax.X, yMax.Y, w, h);
    }

    private readonly record struct RectD(double X, double Y, double Width, double Height);
}
