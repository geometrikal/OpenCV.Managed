// This implementation is ported from the OpenCV pinhole camera model functions.
//
// Original sources:
//   modules/geometry/src/calibration_base.cpp (cv::Rodrigues, cv::projectPoints)
//   modules/geometry/src/distortion_model.hpp (computeTiltProjectionMatrix)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Pinhole camera model: rotation representations and point projection, as the OpenCV functions
/// <c>Rodrigues</c> and <c>projectPoints</c>.
/// </summary>
/// <remarks>
/// The camera matrix is a 3 × 3 <c>double</c> matrix <c>[fx 0 cx; 0 fy cy; 0 0 1]</c>. The
/// distortion coefficients are <c>(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]])</c>:
/// 0, 4, 5, 8, 12 or 14 values. No function modifies its inputs. All computations use
/// <c>double</c> in the OpenCV operation order.
/// </remarks>
public static class CameraModel
{
    /// <summary>
    /// Converts a rotation vector to a rotation matrix or a rotation matrix to a rotation vector,
    /// as OpenCV <c>Rodrigues</c>.
    /// </summary>
    /// <remarks>
    /// A vector (3 × 1 or 1 × 3) gives the 3 × 3 matrix <c>R = cos θ·I + (1 − cos θ)·r·rᵀ + sin θ·[r]ₓ</c>.
    /// A 3 × 3 matrix gives a 3 × 1 vector; the matrix is first replaced by the nearest rotation
    /// <c>U·Vᵀ</c> of its SVD. A matrix with an element outside [−100, 100) or NaN gives a zero
    /// vector (and a zero jacobian).
    /// </remarks>
    /// <typeparam name="T"><c>float</c> or <c>double</c>.</typeparam>
    /// <param name="source">Rotation vector (3 × 1 or 1 × 3) or rotation matrix (3 × 3), one channel.</param>
    /// <returns>A new 3 × 3 matrix or 3 × 1 vector of type <typeparamref name="T"/>.</returns>
    public static Mat<T> Rodrigues<T>(Mat<T> source)
        where T : unmanaged, IFloatingPointIeee754<T>
        => RodriguesCore(source, null);

    /// <inheritdoc cref="Rodrigues{T}(Mat{T})"/>
    /// <param name="source">Rotation vector (3 × 1 or 1 × 3) or rotation matrix (3 × 3), one channel.</param>
    /// <param name="jacobian">
    /// Partial derivatives of the output elements with respect to the input elements: 3 × 9 for a
    /// vector input, 9 × 3 for a matrix input.
    /// </param>
    public static Mat<T> Rodrigues<T>(Mat<T> source, out Mat<T> jacobian)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var j = new double[27];
        Mat<T> dst = RodriguesCore(source, j);
        bool v2m = source.Width == 1 || source.Height == 1;
        jacobian = v2m ? new Mat<T>(9, 3) : new Mat<T>(3, 9);
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 9; c++)
            {
                T value = T.CreateTruncating(j[r * 9 + c]);
                if (v2m)
                {
                    jacobian.At(r, c) = value;
                }
                else
                {
                    jacobian.At(c, r) = value;
                }
            }
        }

        return dst;
    }

    /// <summary>Converts a rotation vector to a rotation matrix, as OpenCV <c>Rodrigues</c>.</summary>
    /// <param name="rotationVector">Rotation vector: axis times angle in radians.</param>
    /// <returns>A new 3 × 3 rotation matrix.</returns>
    public static Mat<double> Rodrigues(Point3d rotationVector)
        => new(CameraRodrigues.VectorToMatrix(rotationVector.X, rotationVector.Y, rotationVector.Z, null), 3, 3);

    /// <summary>Converts a rotation matrix to a rotation vector, as OpenCV <c>Rodrigues</c>.</summary>
    /// <param name="rotationMatrix">Rotation matrix (3 × 3).</param>
    /// <returns>The rotation vector.</returns>
    public static Point3d RodriguesVector(Mat<double> rotationMatrix)
    {
        (double x, double y, double z) = CameraRodrigues.MatrixToVector(CameraDistortion.Read3x3(rotationMatrix, nameof(rotationMatrix)), null);
        return new Point3d(x, y, z);
    }

    /// <summary>
    /// Projects 3D points to the image plane, as OpenCV <c>projectPoints</c>.
    /// </summary>
    /// <remarks>
    /// For each point: <c>(x, y, z) = R·X + t</c>; <c>x' = x/z</c>, <c>y' = y/z</c> (division by 1
    /// when z is 0); the distortion model gives <c>(x'', y'')</c>; the tilt model gives
    /// <c>(x''', y''')</c>; the result is <c>(fx·x''' + cx, fy·y''' + cy)</c>. With
    /// <paramref name="aspectRatio"/> above <c>FLT_EPSILON</c>, <c>fx = fy·aspectRatio</c>.
    /// </remarks>
    /// <param name="objectPoints">Points in the world coordinate frame.</param>
    /// <param name="rotationVector">Rotation vector (Rodrigues) from world to camera.</param>
    /// <param name="translationVector">Translation vector from world to camera.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="aspectRatio">Fixed aspect ratio <c>fx/fy</c>, or 0.</param>
    /// <returns>A new array with the image points.</returns>
    public static Point2d[] ProjectPoints(
        ReadOnlySpan<Point3d> objectPoints,
        Point3d rotationVector,
        Point3d translationVector,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        double aspectRatio = 0)
    {
        var dst = new Point2d[objectPoints.Length];
        ProjectCore(objectPoints, RotationFromVector(rotationVector), translationVector, cameraMatrix, distCoeffs, aspectRatio, dst, null);
        return dst;
    }

    /// <summary>Projects 3D points to the image plane and computes the jacobian, as OpenCV <c>projectPoints</c>.</summary>
    /// <inheritdoc cref="ProjectPoints(ReadOnlySpan{Point3d}, Point3d, Point3d, Mat{double}, ReadOnlySpan{double}, double)"/>
    /// <param name="objectPoints">Points in the world coordinate frame.</param>
    /// <param name="rotationVector">Rotation vector (Rodrigues) from world to camera.</param>
    /// <param name="translationVector">Translation vector from world to camera.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="jacobian">
    /// 2N × (10 + n) matrix, where n is the coefficient count (5 when <paramref name="distCoeffs"/>
    /// is empty). Columns: ∂p/∂r (3), ∂p/∂t (3), ∂p/∂(fx, fy) (2), ∂p/∂(cx, cy) (2), ∂p/∂k (n).
    /// Rows 2i and 2i + 1 are the x and y of point i.
    /// </param>
    /// <param name="aspectRatio">Fixed aspect ratio <c>fx/fy</c>, or 0.</param>
    public static Point2d[] ProjectPoints(
        ReadOnlySpan<Point3d> objectPoints,
        Point3d rotationVector,
        Point3d translationVector,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        out Mat<double> jacobian,
        double aspectRatio = 0)
    {
        var dst = new Point2d[objectPoints.Length];
        jacobian = ProjectWithJacobian(objectPoints, RotationFromVector(rotationVector), translationVector, cameraMatrix, distCoeffs, aspectRatio, dst);
        return dst;
    }

    /// <summary>Projects 3D points with a rotation matrix, as OpenCV <c>projectPoints</c> with a 3 × 3 <c>rvec</c>.</summary>
    /// <inheritdoc cref="ProjectPoints(ReadOnlySpan{Point3d}, Point3d, Point3d, Mat{double}, ReadOnlySpan{double}, double)"/>
    /// <param name="objectPoints">Points in the world coordinate frame.</param>
    /// <param name="rotationMatrix">Rotation matrix (3 × 3) from world to camera. The projection uses it unchanged.</param>
    /// <param name="translationVector">Translation vector from world to camera.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="aspectRatio">Fixed aspect ratio <c>fx/fy</c>, or 0.</param>
    public static Point2d[] ProjectPoints(
        ReadOnlySpan<Point3d> objectPoints,
        Mat<double> rotationMatrix,
        Point3d translationVector,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        double aspectRatio = 0)
    {
        var dst = new Point2d[objectPoints.Length];
        ProjectCore(objectPoints, RotationFromMatrix(rotationMatrix), translationVector, cameraMatrix, distCoeffs, aspectRatio, dst, null);
        return dst;
    }

    /// <summary>Projects single-precision 3D points; the computation uses <c>double</c> and rounds the result to <c>float</c>.</summary>
    /// <inheritdoc cref="ProjectPoints(ReadOnlySpan{Point3d}, Point3d, Point3d, Mat{double}, ReadOnlySpan{double}, double)"/>
    public static Point2f[] ProjectPoints(
        ReadOnlySpan<Point3f> objectPoints,
        Point3d rotationVector,
        Point3d translationVector,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        double aspectRatio = 0)
    {
        Point3d[] pts = Widen(objectPoints);
        Point2d[] d = ProjectPoints(pts, rotationVector, translationVector, cameraMatrix, distCoeffs, aspectRatio);
        return Narrow(d);
    }

    /// <summary>Projects single-precision 3D points and computes the jacobian.</summary>
    /// <inheritdoc cref="ProjectPoints(ReadOnlySpan{Point3d}, Point3d, Point3d, Mat{double}, ReadOnlySpan{double}, out Mat{double}, double)"/>
    public static Point2f[] ProjectPoints(
        ReadOnlySpan<Point3f> objectPoints,
        Point3d rotationVector,
        Point3d translationVector,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        out Mat<double> jacobian,
        double aspectRatio = 0)
    {
        Point3d[] pts = Widen(objectPoints);
        Point2d[] d = ProjectPoints(pts, rotationVector, translationVector, cameraMatrix, distCoeffs, out jacobian, aspectRatio);
        return Narrow(d);
    }

    /// <summary>
    /// Finds the object pose from 3D-2D point correspondences, as OpenCV <c>solvePnP</c> with
    /// <c>SOLVEPNP_ITERATIVE</c> and no extrinsic guess.
    /// </summary>
    /// <remarks>
    /// The image points are undistorted to normalized coordinates. For a planar object (smallest
    /// to middle singular value ratio of the centred object points below 0.001) the initial pose
    /// comes from a homography; otherwise from a DLT, which needs at least 6 points. Then 20
    /// Levenberg-Marquardt iterations with geodesic acceleration minimize the reprojection error.
    /// The arithmetic follows the OpenCV 5.0 operation order (MatExpr evaluation, SSE2 norm lanes).
    /// </remarks>
    /// <param name="objectPoints">Object points in the object coordinate frame (at least 4).</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="rotationVector">Rotation vector from the object frame to the camera frame.</param>
    /// <param name="translationVector">Translation vector from the object frame to the camera frame.</param>
    /// <returns>True (OpenCV returns true when a solution exists; the iterative method always gives one).</returns>
    public static bool SolvePnP(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        out Point3d rotationVector,
        out Point3d translationVector)
    {
        var param = new double[6];
        CameraPnP.FindExtrinsic(objectPoints, imagePoints, cameraMatrix, distCoeffs, param, false);
        rotationVector = new Point3d(param[0], param[1], param[2]);
        translationVector = new Point3d(param[3], param[4], param[5]);
        return true;
    }

    /// <summary>
    /// Finds the object pose from 3D-2D point correspondences with the given method, as OpenCV
    /// <c>solvePnP</c> without an extrinsic guess.
    /// </summary>
    /// <remarks>This overload calls <see cref="PnP.SolvePnP"/>. <see cref="SolvePnPMethod.Iterative"/> gives the same result as the overload without a method.</remarks>
    /// <param name="objectPoints">Object points in the object coordinate frame.</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="method">Solver.</param>
    /// <param name="rotationVector">Rotation vector from the object frame to the camera frame.</param>
    /// <param name="translationVector">Translation vector from the object frame to the camera frame.</param>
    /// <returns>True when the solver finds a solution.</returns>
    public static bool SolvePnP(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method,
        out Point3d rotationVector,
        out Point3d translationVector)
        => PnP.SolvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, method, out rotationVector, out translationVector);

    /// <summary>
    /// Refines an object pose from 3D-2D point correspondences, as OpenCV <c>solvePnP</c> with
    /// <c>SOLVEPNP_ITERATIVE</c> and <c>useExtrinsicGuess = true</c>.
    /// </summary>
    /// <remarks>
    /// The Levenberg-Marquardt refinement starts from the given pose. Three points are accepted.
    /// </remarks>
    /// <param name="objectPoints">Object points in the object coordinate frame (at least 3).</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients (0, 4, 5, 8, 12 or 14 values).</param>
    /// <param name="rotationVector">Initial rotation vector on input; refined rotation vector on output.</param>
    /// <param name="translationVector">Initial translation vector on input; refined translation vector on output.</param>
    /// <returns>True.</returns>
    public static bool SolvePnPRefine(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        ref Point3d rotationVector,
        ref Point3d translationVector)
    {
        double[] param = [rotationVector.X, rotationVector.Y, rotationVector.Z, translationVector.X, translationVector.Y, translationVector.Z];
        CameraPnP.FindExtrinsic(objectPoints, imagePoints, cameraMatrix, distCoeffs, param, true);
        rotationVector = new Point3d(param[0], param[1], param[2]);
        translationVector = new Point3d(param[3], param[4], param[5]);
        return true;
    }

    private static (double[] R, double[] DRdr) RotationFromVector(Point3d rvec)
    {
        var dRdr = new double[27];
        double[] r = CameraRodrigues.VectorToMatrix(rvec.X, rvec.Y, rvec.Z, dRdr);
        return (r, dRdr);
    }

    private static (double[] R, double[] DRdr) RotationFromMatrix(Mat<double> rotationMatrix)
    {
        double[] r = CameraDistortion.Read3x3(rotationMatrix, nameof(rotationMatrix));
        (double x, double y, double z) = CameraRodrigues.MatrixToVector(r, null);
        var dRdr = new double[27];
        CameraRodrigues.VectorToMatrix(x, y, z, dRdr);
        return (r, dRdr);
    }

    private static Mat<double> ProjectWithJacobian(
        ReadOnlySpan<Point3d> objectPoints,
        (double[] R, double[] DRdr) rotation,
        Point3d tvec,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        double aspectRatio,
        Span<Point2d> dst)
    {
        ReadOnlySpan<double> dist = distCoeffs.IsEmpty ? stackalloc double[5] : distCoeffs;
        var jacobian = new Mat<double>(10 + dist.Length, Math.Max(1, 2 * objectPoints.Length));
        ProjectCore(objectPoints, rotation, tvec, cameraMatrix, dist, aspectRatio, dst, objectPoints.Length > 0 ? jacobian : null);
        return jacobian;
    }

    /// <summary><see cref="ProjectPoints(ReadOnlySpan{Point3d}, Point3d, Point3d, Mat{double}, ReadOnlySpan{double}, double)"/> into <paramref name="dst"/>.</summary>
    internal static void ProjectPointsInto(
        ReadOnlySpan<Point3d> objectPoints,
        Point3d rotationVector,
        Point3d translationVector,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Span<Point2d> dst)
        => ProjectCore(objectPoints, RotationFromVector(rotationVector), translationVector, cameraMatrix, distCoeffs, 0, dst, null);

    private static void ProjectCore(
        ReadOnlySpan<Point3d> objectPoints,
        (double[] R, double[] DRdr) rotation,
        Point3d tvec,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        double aspectRatio,
        Span<Point2d> dst,
        Mat<double>? jacobian)
    {
        double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        CameraDistortion.Expand(distCoeffs, nameof(distCoeffs));
        double[] t = [tvec.X, tvec.Y, tvec.Z];
        CameraProjection.Project(objectPoints, rotation.R, rotation.DRdr, t, a, distCoeffs, aspectRatio, dst, jacobian);
    }

    private static Mat<T> RodriguesCore<T>(Mat<T> source, double[]? jacobian)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(source);
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException("Rodrigues supports float and double matrices.");
        }

        bool vector = source.Channels == 1 && ((source.Width == 1 && source.Height == 3) || (source.Width == 3 && source.Height == 1));
        bool matrix = source.Channels == 1 && source.Width == 3 && source.Height == 3;
        if (!vector && !matrix)
        {
            throw new ArgumentException("The source must be a 1 × 3 or 3 × 1 rotation vector or a 3 × 3 rotation matrix.", nameof(source));
        }

        if (vector)
        {
            double rx, ry, rz;
            if (source.Height == 1)
            {
                ReadOnlySpan<T> row = source.GetReadOnlyRowSpan(0);
                (rx, ry, rz) = (double.CreateTruncating(row[0]), double.CreateTruncating(row[1]), double.CreateTruncating(row[2]));
            }
            else
            {
                (rx, ry, rz) = (double.CreateTruncating(source.At(0, 0)), double.CreateTruncating(source.At(1, 0)), double.CreateTruncating(source.At(2, 0)));
            }

            double[] r = CameraRodrigues.VectorToMatrix(rx, ry, rz, jacobian);
            var dst = new Mat<T>(3, 3);
            Span<T> d = dst.GetSpan();
            for (int i = 0; i < 9; i++)
            {
                d[i] = T.CreateTruncating(r[i]);
            }

            return dst;
        }

        var m = new double[9];
        for (int y = 0; y < 3; y++)
        {
            ReadOnlySpan<T> row = source.GetReadOnlyRowSpan(y);
            for (int x = 0; x < 3; x++)
            {
                m[y * 3 + x] = double.CreateTruncating(row[x]);
            }
        }

        (double vx, double vy, double vz) = CameraRodrigues.MatrixToVector(m, jacobian);
        var v = new Mat<T>(1, 3);
        v.At(0, 0) = T.CreateTruncating(vx);
        v.At(1, 0) = T.CreateTruncating(vy);
        v.At(2, 0) = T.CreateTruncating(vz);
        return v;
    }

    private static Point3d[] Widen(ReadOnlySpan<Point3f> points)
    {
        var d = new Point3d[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            d[i] = new Point3d(points[i].X, points[i].Y, points[i].Z);
        }

        return d;
    }

    private static Point2f[] Narrow(Point2d[] points)
    {
        var f = new Point2f[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            f[i] = new Point2f((float)points[i].X, (float)points[i].Y);
        }

        return f;
    }
}
