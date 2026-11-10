// This API is based on the OpenCV two-view geometry functions.
//
// Original sources:
//   modules/geometry/include/opencv2/geometry/3d.hpp (declarations)
//   modules/geometry/src/fundam.cpp (findFundamentalMat, computeCorrespondEpilines,
//   convertPointsToHomogeneous, convertPointsFromHomogeneous)
//   modules/geometry/src/five-point.cpp (findEssentialMat, recoverPose, decomposeEssentialMat)
//   modules/geometry/src/triangulate.cpp (triangulatePoints)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>The method of <see cref="EpipolarGeometry.FindFundamentalMat(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, FundamentalMatMethod, double, double, int)"/> (OpenCV <c>FM_*</c>).</summary>
public enum FundamentalMatMethod
{
    /// <summary><c>FM_7POINT</c>: the 7-point algorithm. Needs exactly 7 pairs; with more pairs OpenCV runs LMEDS.</summary>
    SevenPoint = 1,

    /// <summary><c>FM_8POINT</c>: the normalized 8-point algorithm over all pairs.</summary>
    EightPoint = 2,

    /// <summary><c>FM_LMEDS</c>: least median of squares with 7-point samples.</summary>
    Lmeds = 4,

    /// <summary><c>FM_RANSAC</c>: RANSAC with 7-point samples. With fewer than 15 pairs OpenCV runs LMEDS.</summary>
    Ransac = 8,
}

/// <summary>
/// The result of <c>findFundamentalMat</c> or <c>findEssentialMat</c>.
/// </summary>
/// <param name="Model">
/// The 3 × 3 matrix, or a 3k × 3 matrix with k solutions stacked vertically (7-point algorithm,
/// or five-point algorithm on exactly 5 pairs), or null when the estimation failed.
/// </param>
/// <param name="InlierMask">One byte per input pair: 1 for an inlier and 0 for an outlier. All bytes are 0 when no mask was computed.</param>
public sealed record EpipolarEstimationResult(Mat<double>? Model, byte[] InlierMask)
{
    /// <summary>Gets a value indicating whether a matrix was found.</summary>
    public bool Success => Model is not null;
}

/// <summary>The result of <c>recoverPose</c>.</summary>
/// <param name="InlierCount">The number of pairs that pass the chirality test for the selected pose.</param>
/// <param name="R">The 3 × 3 rotation from the first to the second camera.</param>
/// <param name="T">The 3 × 1 unit translation from the first to the second camera.</param>
/// <param name="Mask">One byte per pair. Without an input mask, 255 marks a pair that passes the chirality test. With an input mask, the value is the input mask value AND 255.</param>
/// <param name="TriangulatedPoints">The 4 × N homogeneous points of the selected pose (first camera frame), or null when not requested.</param>
public sealed record RecoverPoseResult(int InlierCount, Mat<double> R, Mat<double> T, byte[] Mask, Mat<double>? TriangulatedPoints);

/// <summary>
/// Two-view epipolar geometry ported from the OpenCV 5.0 <c>geometry</c> module:
/// <c>findFundamentalMat</c>, <c>findEssentialMat</c>, <c>decomposeEssentialMat</c>,
/// <c>recoverPose</c>, <c>triangulatePoints</c>, <c>computeCorrespondEpilines</c>,
/// <c>convertPointsToHomogeneous</c> and <c>convertPointsFromHomogeneous</c>.
/// </summary>
/// <remarks>
/// <para>
/// The functions do not change their inputs. RANSAC and LMEDS create the OpenCV generator
/// <c>RNG((uint64)-1)</c> in every call: the results are deterministic and the functions are
/// thread-safe.
/// </para>
/// <para>
/// <c>findFundamentalMat</c> rounds the points to <c>float</c>; <c>findEssentialMat</c> and
/// <c>recoverPose</c> widen them to <c>double</c>, as OpenCV. USAC methods are not supported.
/// See <c>Docs/Notes/EpipolarPortNotes.md</c>.
/// </para>
/// </remarks>
public static class EpipolarGeometry
{
    /// <summary>Default distance threshold of <c>recoverPose</c> without the <c>distanceThresh</c> argument.</summary>
    public const double DefaultDistanceThreshold = 50.0;

    /// <summary>
    /// Calculates the fundamental matrix from corresponding points, as OpenCV
    /// <c>findFundamentalMat</c> (method form).
    /// </summary>
    /// <param name="points1">Points in the first image.</param>
    /// <param name="points2">Points in the second image. The length must equal the length of <paramref name="points1"/>.</param>
    /// <param name="method">Estimation method.</param>
    /// <param name="ransacReprojThreshold">Maximum distance from a point to its epipolar line for a RANSAC inlier, in pixels. A value ≤ 0 is replaced with 3.</param>
    /// <param name="confidence">Confidence level. A value outside (DBL_EPSILON, 1 - DBL_EPSILON) is replaced with 0.99.</param>
    /// <param name="maxIters">Maximum number of robust iterations.</param>
    /// <returns>
    /// The matrix and the inlier mask. The model is null for fewer than 7 pairs or when the
    /// estimation failed. For exactly 7 pairs or <see cref="FundamentalMatMethod.EightPoint"/>, the
    /// mask is all ones.
    /// </returns>
    public static EpipolarEstimationResult FindFundamentalMat(
        ReadOnlySpan<Point2f> points1,
        ReadOnlySpan<Point2f> points2,
        FundamentalMatMethod method = FundamentalMatMethod.Ransac,
        double ransacReprojThreshold = 3.0,
        double confidence = 0.99,
        int maxIters = 1000)
    {
        RequireSameLength(points1.Length, points2.Length);
        int npoints = points1.Length;
        byte[] mask = new byte[npoints];
        if (npoints < 7)
        {
            return new EpipolarEstimationResult(null, mask);
        }

        var kernel = default(FundamentalKernel);
        double[] f = new double[27];
        int result;
        if (npoints == 7 || method == FundamentalMatMethod.EightPoint)
        {
            result = kernel.RunKernel(points1, points2, f);
            mask.AsSpan().Fill(1);
        }
        else
        {
            if (ransacReprojThreshold <= 0)
            {
                ransacReprojThreshold = 3;
            }

            if (confidence < 2.220446049250313e-16 || confidence > 1 - 2.220446049250313e-16)
            {
                confidence = 0.99;
            }

            if (((int)method & ~3) == (int)FundamentalMatMethod.Ransac && npoints >= 15)
            {
                result = MultiModelRegistrator.Ransac<Point2f, FundamentalKernel>(kernel, points1, points2, ransacReprojThreshold, confidence, maxIters, f, mask);
            }
            else
            {
                int n = MultiModelRegistrator.Lmeds<Point2f, FundamentalKernel>(kernel, points1, points2, confidence, maxIters, f, mask, out bool ok);
                result = ok ? n : 0;
            }
        }

        if (result <= 0)
        {
            return new EpipolarEstimationResult(null, mask);
        }

        return new EpipolarEstimationResult(new Mat<double>(f.AsSpan(0, result * 9).ToArray(), 3, result * 3), mask);
    }

    /// <summary>
    /// Calculates the fundamental matrix; the points are rounded to <c>float</c> first.
    /// </summary>
    /// <inheritdoc cref="FindFundamentalMat(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, FundamentalMatMethod, double, double, int)"/>
    public static EpipolarEstimationResult FindFundamentalMat(
        ReadOnlySpan<Point2d> points1,
        ReadOnlySpan<Point2d> points2,
        FundamentalMatMethod method = FundamentalMatMethod.Ransac,
        double ransacReprojThreshold = 3.0,
        double confidence = 0.99,
        int maxIters = 1000)
    {
        Point2f[] f1 = ToFloatRented(points1), f2 = ToFloatRented(points2);
        try
        {
            return FindFundamentalMat(f1.AsSpan(0, points1.Length), f2.AsSpan(0, points2.Length), method, ransacReprojThreshold, confidence, maxIters);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(f1);
            ArrayPool<Point2f>.Shared.Return(f2);
        }
    }

    /// <summary>
    /// Calculates the essential matrix from corresponding points with the five-point algorithm, as
    /// OpenCV <c>findEssentialMat</c> with a camera matrix.
    /// </summary>
    /// <param name="points1">Points in the first image, in pixels. The camera matrix applies to both images.</param>
    /// <param name="points2">Points in the second image. The length must equal the length of <paramref name="points1"/>.</param>
    /// <param name="cameraMatrix">3 × 3 camera matrix. Only fx, fy, cx and cy are used.</param>
    /// <param name="method"><see cref="RobustEstimationMethod.Ransac"/>; any other non-USAC value runs LMEDS, as OpenCV.</param>
    /// <param name="prob">Confidence level in (0, 1).</param>
    /// <param name="threshold">Maximum Sampson distance of a RANSAC inlier, in pixels. It is divided by (fx + fy)/2.</param>
    /// <param name="maxIters">Maximum number of robust iterations.</param>
    /// <returns>
    /// The matrix and the inlier mask. The model is null for fewer than 5 pairs or when RANSAC
    /// finds no model. For exactly 5 pairs the model can stack up to 10 solutions. LMEDS returns
    /// its best model also when fewer than 5 pairs are inliers, as OpenCV.
    /// </returns>
    public static EpipolarEstimationResult FindEssentialMat(
        ReadOnlySpan<Point2d> points1,
        ReadOnlySpan<Point2d> points2,
        Mat<double> cameraMatrix,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double prob = 0.999,
        double threshold = 1.0,
        int maxIters = 1000)
    {
        RequireSameLength(points1.Length, points2.Length);
        RequireCameraMatrix(cameraMatrix);
        if (method >= RobustEstimationMethod.UsacDefault && method <= RobustEstimationMethod.UsacMagsac)
        {
            throw new NotSupportedException("USAC methods are not supported.");
        }

        int npoints = points1.Length;
        Point2d[] n1 = ArrayPool<Point2d>.Shared.Rent(npoints);
        Point2d[] n2 = ArrayPool<Point2d>.Shared.Rent(npoints);
        byte[] mask = new byte[npoints];
        double[] e = new double[90];
        int n;
        try
        {
            ReadOnlySpan<Point2d> s1 = n1.AsSpan(0, npoints), s2 = n2.AsSpan(0, npoints);
            TriangulationKernels.Normalize(points1, cameraMatrix, n1);
            TriangulationKernels.Normalize(points2, cameraMatrix, n2);
            double fx = cameraMatrix.At(0, 0), fy = cameraMatrix.At(1, 1);
            threshold /= (fx + fy) / 2;

            var kernel = default(EssentialFivePointKernel);
            n = method == RobustEstimationMethod.Ransac
                ? MultiModelRegistrator.Ransac<Point2d, EssentialFivePointKernel>(kernel, s1, s2, threshold, prob, maxIters, e, mask)
                : MultiModelRegistrator.Lmeds<Point2d, EssentialFivePointKernel>(kernel, s1, s2, prob, maxIters, e, mask, out _);
        }
        finally
        {
            ArrayPool<Point2d>.Shared.Return(n1);
            ArrayPool<Point2d>.Shared.Return(n2);
        }

        if (n <= 0)
        {
            return new EpipolarEstimationResult(null, mask);
        }

        return new EpipolarEstimationResult(new Mat<double>(e.AsSpan(0, n * 9).ToArray(), 3, n * 3), mask);
    }

    /// <summary>Calculates the essential matrix; <c>float</c> points are widened to <c>double</c>.</summary>
    /// <inheritdoc cref="FindEssentialMat(ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d}, Mat{double}, RobustEstimationMethod, double, double, int)"/>
    public static EpipolarEstimationResult FindEssentialMat(
        ReadOnlySpan<Point2f> points1,
        ReadOnlySpan<Point2f> points2,
        Mat<double> cameraMatrix,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double prob = 0.999,
        double threshold = 1.0,
        int maxIters = 1000)
        => FindEssentialMat(ToDouble(points1), ToDouble(points2), cameraMatrix, method, prob, threshold, maxIters);

    /// <summary>
    /// Calculates the essential matrix for a camera with focal length <paramref name="focal"/> and
    /// principal point <paramref name="pp"/>, as the OpenCV overload with <c>focal</c> and <c>pp</c>.
    /// </summary>
    /// <inheritdoc cref="FindEssentialMat(ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d}, Mat{double}, RobustEstimationMethod, double, double, int)"/>
    public static EpipolarEstimationResult FindEssentialMat(
        ReadOnlySpan<Point2d> points1,
        ReadOnlySpan<Point2d> points2,
        double focal,
        Point2d pp,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double prob = 0.999,
        double threshold = 1.0,
        int maxIters = 1000)
        => FindEssentialMat(points1, points2, CameraMatrix(focal, pp), method, prob, threshold, maxIters);

    /// <inheritdoc cref="FindEssentialMat(ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d}, double, Point2d, RobustEstimationMethod, double, double, int)"/>
    public static EpipolarEstimationResult FindEssentialMat(
        ReadOnlySpan<Point2f> points1,
        ReadOnlySpan<Point2f> points2,
        double focal,
        Point2d pp,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double prob = 0.999,
        double threshold = 1.0,
        int maxIters = 1000)
        => FindEssentialMat(ToDouble(points1), ToDouble(points2), CameraMatrix(focal, pp), method, prob, threshold, maxIters);

    /// <summary>
    /// Decomposes an essential matrix into two possible rotations and a translation, as OpenCV
    /// <c>decomposeEssentialMat</c>.
    /// </summary>
    /// <param name="essential">3 × 3 essential matrix.</param>
    /// <returns><c>R1 = U·W·Vᵀ</c>, <c>R2 = U·Wᵀ·Vᵀ</c> and the unit translation <c>t = U.col(2)</c> (3 × 1).</returns>
    public static (Mat<double> R1, Mat<double> R2, Mat<double> T) DecomposeEssentialMat(Mat<double> essential)
    {
        Require3x3(essential, nameof(essential));
        TriangulationKernels.DecomposeEssential(essential, out Mat<double> r1, out Mat<double> r2, out Mat<double> t);
        return (r1, r2, t);
    }

    /// <summary>
    /// Recovers the relative camera rotation and translation from an essential matrix with the
    /// chirality test, as OpenCV <c>recoverPose</c> with the distance threshold 50.
    /// </summary>
    /// <param name="essential">3 × 3 essential matrix.</param>
    /// <param name="points1">Points in the first image, in pixels.</param>
    /// <param name="points2">Points in the second image.</param>
    /// <param name="cameraMatrix">3 × 3 camera matrix (fx, fy, cx, cy).</param>
    /// <param name="mask">Optional input mask (for example the inlier mask of <c>findEssentialMat</c>). Pairs with a zero byte are not counted.</param>
    /// <returns>The pose, the inlier count and the output mask.</returns>
    public static RecoverPoseResult RecoverPose(
        Mat<double> essential,
        ReadOnlySpan<Point2d> points1,
        ReadOnlySpan<Point2d> points2,
        Mat<double> cameraMatrix,
        ReadOnlySpan<byte> mask = default)
        => RecoverPose(essential, points1, points2, cameraMatrix, DefaultDistanceThreshold, mask, false);

    /// <summary>
    /// Recovers the relative camera pose, as OpenCV <c>recoverPose</c> with
    /// <c>distanceThresh</c> and <c>triangulatedPoints</c>.
    /// </summary>
    /// <param name="essential">3 × 3 essential matrix.</param>
    /// <param name="points1">Points in the first image, in pixels.</param>
    /// <param name="points2">Points in the second image.</param>
    /// <param name="cameraMatrix">3 × 3 camera matrix (fx, fy, cx, cy).</param>
    /// <param name="distanceThresh">Points with a depth at or above this value (in units of the translation) fail the test.</param>
    /// <param name="mask">Optional input mask. Pairs with a zero byte are not counted.</param>
    /// <param name="triangulate">True to return the triangulated points of the selected pose.</param>
    /// <returns>The pose, the inlier count, the output mask and the optional points.</returns>
    public static RecoverPoseResult RecoverPose(
        Mat<double> essential,
        ReadOnlySpan<Point2d> points1,
        ReadOnlySpan<Point2d> points2,
        Mat<double> cameraMatrix,
        double distanceThresh,
        ReadOnlySpan<byte> mask = default,
        bool triangulate = false)
    {
        Require3x3(essential, nameof(essential));
        RequireSameLength(points1.Length, points2.Length);
        RequireCameraMatrix(cameraMatrix);
        if (!mask.IsEmpty && mask.Length != points1.Length)
        {
            throw new ArgumentException("The mask length must equal the number of points.", nameof(mask));
        }

        int npoints = points1.Length;
        Point2d[] n1 = ArrayPool<Point2d>.Shared.Rent(npoints);
        Point2d[] n2 = ArrayPool<Point2d>.Shared.Rent(npoints);
        try
        {
            TriangulationKernels.Normalize(points1, cameraMatrix, n1);
            TriangulationKernels.Normalize(points2, cameraMatrix, n2);
            int good = TriangulationKernels.RecoverPose(
                essential, n1.AsSpan(0, npoints), n2.AsSpan(0, npoints), distanceThresh, mask, triangulate, out Mat<double> r, out Mat<double> t, out byte[] outMask, out Mat<double>? tri);
            return new RecoverPoseResult(good, r, t, outMask, tri);
        }
        finally
        {
            ArrayPool<Point2d>.Shared.Return(n1);
            ArrayPool<Point2d>.Shared.Return(n2);
        }
    }

    /// <inheritdoc cref="RecoverPose(Mat{double}, ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d}, Mat{double}, ReadOnlySpan{byte})"/>
    public static RecoverPoseResult RecoverPose(
        Mat<double> essential,
        ReadOnlySpan<Point2f> points1,
        ReadOnlySpan<Point2f> points2,
        Mat<double> cameraMatrix,
        ReadOnlySpan<byte> mask = default)
        => RecoverPose(essential, ToDouble(points1), ToDouble(points2), cameraMatrix, DefaultDistanceThreshold, mask, false);

    /// <inheritdoc cref="RecoverPose(Mat{double}, ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d}, Mat{double}, double, ReadOnlySpan{byte}, bool)"/>
    public static RecoverPoseResult RecoverPose(
        Mat<double> essential,
        ReadOnlySpan<Point2f> points1,
        ReadOnlySpan<Point2f> points2,
        Mat<double> cameraMatrix,
        double distanceThresh,
        ReadOnlySpan<byte> mask = default,
        bool triangulate = false)
        => RecoverPose(essential, ToDouble(points1), ToDouble(points2), cameraMatrix, distanceThresh, mask, triangulate);

    /// <summary>
    /// Recovers the relative camera pose for a camera with focal length <paramref name="focal"/> and
    /// principal point <paramref name="pp"/>.
    /// </summary>
    /// <inheritdoc cref="RecoverPose(Mat{double}, ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d}, Mat{double}, ReadOnlySpan{byte})"/>
    public static RecoverPoseResult RecoverPose(
        Mat<double> essential,
        ReadOnlySpan<Point2d> points1,
        ReadOnlySpan<Point2d> points2,
        double focal,
        Point2d pp,
        ReadOnlySpan<byte> mask = default)
        => RecoverPose(essential, points1, points2, CameraMatrix(focal, pp), DefaultDistanceThreshold, mask, false);

    /// <summary>
    /// Reconstructs points by triangulation (linear DLT solved with SVD), as OpenCV
    /// <c>triangulatePoints</c>.
    /// </summary>
    /// <param name="projMatr1">3 × 4 projection matrix of the first camera.</param>
    /// <param name="projMatr2">3 × 4 projection matrix of the second camera.</param>
    /// <param name="projPoints1">Points in the first image.</param>
    /// <param name="projPoints2">Points in the second image.</param>
    /// <returns>4 × N homogeneous points.</returns>
    public static Mat<double> TriangulatePoints(
        Mat<double> projMatr1, Mat<double> projMatr2, ReadOnlySpan<Point2d> projPoints1, ReadOnlySpan<Point2d> projPoints2)
    {
        RequireSameLength(projPoints1.Length, projPoints2.Length);
        if (projPoints1.Length == 0)
        {
            throw new ArgumentException("At least one point is required.", nameof(projPoints1));
        }

        return TriangulationKernels.Triangulate(Projection(projMatr1, nameof(projMatr1)), Projection(projMatr2, nameof(projMatr2)), projPoints1, projPoints2);
    }

    /// <summary>
    /// Reconstructs points by triangulation; the coordinates are widened to <c>double</c> and the
    /// result is rounded to <c>float</c>, as OpenCV for <c>CV_32F</c> input.
    /// </summary>
    /// <inheritdoc cref="TriangulatePoints(Mat{double}, Mat{double}, ReadOnlySpan{Point2d}, ReadOnlySpan{Point2d})"/>
    public static Mat<float> TriangulatePoints(
        Mat<double> projMatr1, Mat<double> projMatr2, ReadOnlySpan<Point2f> projPoints1, ReadOnlySpan<Point2f> projPoints2)
    {
        Mat<double> d = TriangulatePoints(projMatr1, projMatr2, ToDouble(projPoints1), ToDouble(projPoints2));
        var f = new Mat<float>(d.Width, 4);
        for (int r = 0; r < 4; r++)
        {
            ReadOnlySpan<double> src = d.GetReadOnlyRowSpan(r);
            Span<float> dst = f.GetRowSpan(r);
            for (int i = 0; i < src.Length; i++)
            {
                dst[i] = (float)src[i];
            }
        }

        return f;
    }

    /// <summary>
    /// For points in one image, computes the corresponding epilines in the other image, as OpenCV
    /// <c>computeCorrespondEpilines</c>.
    /// </summary>
    /// <param name="points">Points in image <paramref name="whichImage"/>.</param>
    /// <param name="whichImage">1 or 2: the image that contains the points. For 2, <c>Fᵀ</c> is used.</param>
    /// <param name="fundamental">3 × 3 fundamental matrix.</param>
    /// <returns>Lines (a, b, c) with <c>a² + b² = 1</c>, or the raw coefficients when a = b = 0.</returns>
    public static Point3f[] ComputeCorrespondEpilines(ReadOnlySpan<Point2f> points, int whichImage, Mat<double> fundamental)
    {
        Span<double> f = stackalloc double[9];
        EpilineMatrix(fundamental, whichImage, f);
        var lines = new Point3f[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            Line(f, points[i].X, points[i].Y, out double a, out double b, out double c);
            lines[i] = new Point3f((float)a, (float)b, (float)c);
        }

        return lines;
    }

    /// <inheritdoc cref="ComputeCorrespondEpilines(ReadOnlySpan{Point2f}, int, Mat{double})"/>
    public static Point3d[] ComputeCorrespondEpilines(ReadOnlySpan<Point2d> points, int whichImage, Mat<double> fundamental)
    {
        Span<double> f = stackalloc double[9];
        EpilineMatrix(fundamental, whichImage, f);
        var lines = new Point3d[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            Line(f, points[i].X, points[i].Y, out double a, out double b, out double c);
            lines[i] = new Point3d(a, b, c);
        }

        return lines;
    }

    /// <summary>Appends a coordinate 1 to every point, as OpenCV <c>convertPointsToHomogeneous</c>.</summary>
    public static Point3f[] ConvertPointsToHomogeneous(ReadOnlySpan<Point2f> points)
    {
        var dst = new Point3f[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            dst[i] = new Point3f(points[i].X, points[i].Y, 1f);
        }

        return dst;
    }

    /// <inheritdoc cref="ConvertPointsToHomogeneous(ReadOnlySpan{Point2f})"/>
    public static Point3d[] ConvertPointsToHomogeneous(ReadOnlySpan<Point2d> points)
    {
        var dst = new Point3d[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            dst[i] = new Point3d(points[i].X, points[i].Y, 1.0);
        }

        return dst;
    }

    /// <summary>
    /// Divides x and y by z, as OpenCV <c>convertPointsFromHomogeneous</c>. A point with
    /// |z| ≤ FLT_EPSILON is not divided (scale 1).
    /// </summary>
    public static Point2f[] ConvertPointsFromHomogeneous(ReadOnlySpan<Point3f> points)
    {
        const float FltEpsilon = 1.1920929e-07f;
        var dst = new Point2f[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            float z = points[i].Z;
            float scale = MathF.Abs(z) > FltEpsilon ? 1f / z : 1f;
            dst[i] = new Point2f(points[i].X * scale, points[i].Y * scale);
        }

        return dst;
    }

    /// <inheritdoc cref="ConvertPointsFromHomogeneous(ReadOnlySpan{Point3f})"/>
    public static Point2d[] ConvertPointsFromHomogeneous(ReadOnlySpan<Point3d> points)
    {
        var dst = new Point2d[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            double scale = PointSetRegistrator.ScaleFor(points[i].Z);
            dst[i] = new Point2d(points[i].X * scale, points[i].Y * scale);
        }

        return dst;
    }

    private static void EpilineMatrix(Mat<double> fundamental, int whichImage, Span<double> f)
    {
        Require3x3(fundamental, nameof(fundamental));
        if (whichImage is not (1 or 2))
        {
            throw new ArgumentOutOfRangeException(nameof(whichImage), "The image index must be 1 or 2.");
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                f[(i * 3) + j] = whichImage == 2 ? fundamental.At(j, i) : fundamental.At(i, j);
            }
        }
    }

    private static void Line(ReadOnlySpan<double> f, double x, double y, out double a, out double b, out double c)
    {
        a = (f[0] * x) + (f[1] * y) + f[2];
        b = (f[3] * x) + (f[4] * y) + f[5];
        c = (f[6] * x) + (f[7] * y) + f[8];
        double nu = (a * a) + (b * b);
        nu = nu != 0 ? 1.0 / Math.Sqrt(nu) : 1.0;
        a *= nu;
        b *= nu;
        c *= nu;
    }

    private static Mat<double> CameraMatrix(double focal, Point2d pp)
        => new(new double[] { focal, 0, pp.X, 0, focal, pp.Y, 0, 0, 1 }, 3, 3);

    private static double[] Projection(Mat<double> p, string name)
    {
        ArgumentNullException.ThrowIfNull(p, name);
        if (p.Width != 4 || p.Height != 3 || p.Channels != 1)
        {
            throw new ArgumentException("The projection matrix must be 3 × 4.", name);
        }

        double[] a = new double[12];
        for (int i = 0; i < 3; i++)
        {
            p.GetReadOnlyRowSpan(i)[..4].CopyTo(a.AsSpan(i * 4, 4));
        }

        return a;
    }

    private static void Require3x3(Mat<double> m, string name)
    {
        ArgumentNullException.ThrowIfNull(m, name);
        if (m.Width != 3 || m.Height != 3 || m.Channels != 1)
        {
            throw new ArgumentException("The matrix must be 3 × 3.", name);
        }
    }

    private static void RequireCameraMatrix(Mat<double> cameraMatrix) => Require3x3(cameraMatrix, nameof(cameraMatrix));

    private static void RequireSameLength(int a, int b)
    {
        if (a != b)
        {
            throw new ArgumentException("The point sets must have the same length.");
        }
    }

    private static Point2f[] ToFloatRented(ReadOnlySpan<Point2d> p)
    {
        Point2f[] r = ArrayPool<Point2f>.Shared.Rent(p.Length);
        for (int i = 0; i < p.Length; i++)
        {
            r[i] = new Point2f((float)p[i].X, (float)p[i].Y);
        }

        return r;
    }

    private static Point2d[] ToDouble(ReadOnlySpan<Point2f> p)
    {
        var r = new Point2d[p.Length];
        for (int i = 0; i < p.Length; i++)
        {
            r[i] = new Point2d(p[i].X, p[i].Y);
        }

        return r;
    }
}
