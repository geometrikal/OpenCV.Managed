// This implementation is ported from the OpenCV perspective-n-point solvers.
//
// Original sources:
//   modules/geometry/src/solvepnp.cpp (solvePnP, solvePnPGeneric, solveP3P, solvePnPRansac,
//     PnPRansacCallback, solvePnPRefineLM, solvePnPRefineVVS)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib;

using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Perspective-n-point methods, as OpenCV <c>SolvePnPMethod</c> (same values).</summary>
public enum SolvePnPMethod
{
    /// <summary>Homography or DLT initialization, then Levenberg-Marquardt refinement.</summary>
    Iterative = 0,

    /// <summary>EPnP: Efficient Perspective-n-Point.</summary>
    Epnp = 1,

    /// <summary>P3P (Ding et al. 2023). Uses exactly 4 points in <c>solvePnP</c>.</summary>
    P3p = 2,

    /// <summary>Algebraic P3P (Ke and Roumeliotis 2017). Uses exactly 4 points in <c>solvePnP</c>.</summary>
    Ap3p = 3,

    /// <summary>Infinitesimal plane-based pose estimation. The object points must be coplanar.</summary>
    Ippe = 4,

    /// <summary>IPPE for a square marker: 4 points in the order (−s/2, s/2), (s/2, s/2), (s/2, −s/2), (−s/2, −s/2), Z = 0.</summary>
    IppeSquare = 5,

    /// <summary>SQPnP: globally optimal sequential quadratic programming solver.</summary>
    Sqpnp = 6,
}

/// <summary>A camera pose: rotation vector (Rodrigues) and translation vector from the object frame to the camera frame.</summary>
/// <param name="RotationVector">Rotation vector.</param>
/// <param name="TranslationVector">Translation vector.</param>
public readonly record struct PnPPose(Point3d RotationVector, Point3d TranslationVector);

/// <summary>A pose of <see cref="PnP.SolvePnPGeneric"/> with its RMS reprojection error in pixels.</summary>
/// <param name="RotationVector">Rotation vector.</param>
/// <param name="TranslationVector">Translation vector.</param>
/// <param name="ReprojectionError">RMS reprojection error: <c>‖projected − image‖₂ / √(2N)</c>.</param>
public readonly record struct PnPSolution(Point3d RotationVector, Point3d TranslationVector, double ReprojectionError);

/// <summary>The result of <see cref="PnP.SolvePnPRansac(ReadOnlySpan{Point3f}, ReadOnlySpan{Point2f}, Mat{double}, ReadOnlySpan{double}, SolvePnPMethod, int, float, double, PnPPose?)"/>.</summary>
/// <param name="Success">False when RANSAC or the final solver finds no pose.</param>
/// <param name="RotationVector">Rotation vector. Zero when RANSAC finds no model.</param>
/// <param name="TranslationVector">Translation vector. Zero when RANSAC finds no model.</param>
/// <param name="Inliers">Ascending indices of the inlier points. Empty when <paramref name="Success"/> is false.</param>
public sealed record PnPRansacResult(bool Success, Point3d RotationVector, Point3d TranslationVector, int[] Inliers);

/// <summary>
/// Pose estimation from 3D-2D point correspondences, as the OpenCV functions <c>solvePnP</c>,
/// <c>solvePnPGeneric</c>, <c>solveP3P</c>, <c>solvePnPRansac</c>, <c>solvePnPRefineLM</c> and
/// <c>solvePnPRefineVVS</c>.
/// </summary>
/// <remarks>
/// The camera matrix is a 3 × 3 <c>double</c> matrix. The distortion coefficients are 0, 4, 5, 8,
/// 12 or 14 values, as in <see cref="CameraModel"/>. The image points are first undistorted to
/// normalized coordinates (except for <see cref="SolvePnPMethod.Iterative"/>, which undistorts
/// internally). No function modifies its inputs.
/// </remarks>
public static class PnP
{
    /// <summary>Default criteria of <see cref="SolvePnPRefineLM"/> and <see cref="SolvePnPRefineVVS"/>: 20 iterations, <c>FLT_EPSILON</c>.</summary>
    public static readonly TermCriteria DefaultRefineCriteria = new(TermCriteriaType.Eps | TermCriteriaType.MaxIter, 20, 1.1920928955078125e-07);

    /// <summary>Finds the object pose with the given method, as OpenCV <c>solvePnP</c> without an extrinsic guess.</summary>
    /// <remarks>The result is the first solution of <see cref="SolvePnPGeneric"/>.</remarks>
    /// <param name="objectPoints">Object points (at least 4; exactly 4 for P3P, AP3P and IPPE_SQUARE; at least 3 for SQPnP).</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients.</param>
    /// <param name="method">Solver.</param>
    /// <param name="rotationVector">Rotation vector, or zero when there is no solution.</param>
    /// <param name="translationVector">Translation vector, or zero when there is no solution.</param>
    /// <returns>True when the solver finds a solution.</returns>
    public static bool SolvePnP(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method,
        out Point3d rotationVector,
        out Point3d translationVector)
    {
        List<PnPPose> poses = Generic(objectPoints, imagePoints, cameraMatrix, distCoeffs, method, false, default, false);
        if (poses.Count == 0)
        {
            rotationVector = default;
            translationVector = default;
            return false;
        }

        rotationVector = poses[0].RotationVector;
        translationVector = poses[0].TranslationVector;
        return true;
    }

    /// <summary>
    /// Finds all object poses of the given method and their RMS reprojection errors, as OpenCV
    /// <c>solvePnPGeneric</c>.
    /// </summary>
    /// <remarks>
    /// P3P and AP3P give up to 4 solutions sorted by reprojection error; IPPE and IPPE_SQUARE give 2
    /// solutions (the better first); SQPnP gives 1 or more; the other methods give 1. An extrinsic
    /// guess is used only by <see cref="SolvePnPMethod.Iterative"/>.
    /// </remarks>
    /// <param name="objectPoints">Object points.</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients.</param>
    /// <param name="method">Solver.</param>
    /// <param name="extrinsicGuess">Initial pose for <see cref="SolvePnPMethod.Iterative"/>, or null.</param>
    /// <returns>The solutions. Empty when the solver finds none.</returns>
    public static PnPSolution[] SolvePnPGeneric(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method = SolvePnPMethod.Iterative,
        PnPPose? extrinsicGuess = null)
    {
        List<PnPPose> poses = Generic(objectPoints, imagePoints, cameraMatrix, distCoeffs, method, extrinsicGuess.HasValue, extrinsicGuess ?? default, false);
        var result = new PnPSolution[poses.Count];
        int n = objectPoints.Length;
        var diff = new double[2 * n];
        for (int i = 0; i < poses.Count; i++)
        {
            Point2d[] proj = CameraModel.ProjectPoints(objectPoints, poses[i].RotationVector, poses[i].TranslationVector, cameraMatrix, distCoeffs);
            for (int k = 0; k < n; k++)
            {
                diff[2 * k] = proj[k].X;
                diff[2 * k + 1] = proj[k].Y;
            }

            var img = new double[2 * n];
            for (int k = 0; k < n; k++)
            {
                img[2 * k] = imagePoints[k].X;
                img[2 * k + 1] = imagePoints[k].Y;
            }

            double norm = Math.Sqrt(NormDiffL2Sqr(diff, img));
            result[i] = new PnPSolution(poses[i].RotationVector, poses[i].TranslationVector, norm / Math.Sqrt(2 * n));
        }

        return result;
    }

    /// <summary>
    /// Finds the poses from 3 or 4 point correspondences, as OpenCV <c>solveP3P</c>.
    /// </summary>
    /// <remarks>
    /// The solutions are sorted by the sum of squared reprojection errors over all given points
    /// (stable insertion sort). With 4 points, the solvers use the 4th point to select solutions.
    /// </remarks>
    /// <param name="objectPoints">3 or 4 object points.</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients.</param>
    /// <param name="method"><see cref="SolvePnPMethod.P3p"/> or <see cref="SolvePnPMethod.Ap3p"/>.</param>
    /// <returns>Up to 4 poses.</returns>
    public static PnPPose[] SolveP3P(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method)
    {
        if (objectPoints.Length != imagePoints.Length)
        {
            throw new ArgumentException("The object and image point counts must be equal.", nameof(imagePoints));
        }

        if (objectPoints.Length is not (3 or 4))
        {
            throw new ArgumentException("solveP3P needs 3 or 4 points.", nameof(objectPoints));
        }

        if (method is not (SolvePnPMethod.P3p or SolvePnPMethod.Ap3p))
        {
            throw new ArgumentException("The method must be P3P or AP3P.", nameof(method));
        }

        return P3PCore(objectPoints, imagePoints, cameraMatrix, distCoeffs, method, false).ToArray();
    }

    /// <summary>
    /// Finds the object pose with RANSAC, as OpenCV <c>solvePnPRansac</c> with a non-USAC method.
    /// </summary>
    /// <remarks>
    /// The minimal solver is EPnP with 5 points, or P3P/AP3P with 4 points (P3P also when there are
    /// exactly 4 points). RANSAC uses the OpenCV generator <c>RNG((uint64)-1)</c>, so the inlier set
    /// equals the OpenCV inlier set. The final pose comes from <paramref name="method"/> on the
    /// inliers (EPnP for P3P and AP3P); <see cref="SolvePnPMethod.Iterative"/> starts from the RANSAC
    /// model. The computation uses <c>float</c> points, as OpenCV.
    /// </remarks>
    /// <param name="objectPoints">Object points (at least 4).</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients.</param>
    /// <param name="method">Final solver.</param>
    /// <param name="iterationsCount">Maximum number of RANSAC iterations.</param>
    /// <param name="reprojectionError">Maximum reprojection error of an inlier, in pixels.</param>
    /// <param name="confidence">Confidence in (0, 1).</param>
    /// <param name="extrinsicGuess">Initial pose of the final <see cref="SolvePnPMethod.Iterative"/> refinement, or null.</param>
    /// <returns>The pose and the inlier indices.</returns>
    public static PnPRansacResult SolvePnPRansac(
        ReadOnlySpan<Point3f> objectPoints,
        ReadOnlySpan<Point2f> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method = SolvePnPMethod.Iterative,
        int iterationsCount = 100,
        float reprojectionError = 8.0f,
        double confidence = 0.99,
        PnPPose? extrinsicGuess = null)
        => PnpRansac.Run(objectPoints, imagePoints, cameraMatrix, distCoeffs.ToArray(), method, iterationsCount, reprojectionError, confidence, extrinsicGuess);

    /// <summary>Finds the object pose with RANSAC from double-precision points; the points are rounded to <c>float</c>, as OpenCV.</summary>
    /// <inheritdoc cref="SolvePnPRansac(ReadOnlySpan{Point3f}, ReadOnlySpan{Point2f}, Mat{double}, ReadOnlySpan{double}, SolvePnPMethod, int, float, double, PnPPose?)"/>
    public static PnPRansacResult SolvePnPRansac(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method = SolvePnPMethod.Iterative,
        int iterationsCount = 100,
        float reprojectionError = 8.0f,
        double confidence = 0.99,
        PnPPose? extrinsicGuess = null)
    {
        var o = new Point3f[objectPoints.Length];
        for (int i = 0; i < o.Length; i++)
        {
            o[i] = new Point3f((float)objectPoints[i].X, (float)objectPoints[i].Y, (float)objectPoints[i].Z);
        }

        var p = new Point2f[imagePoints.Length];
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = new Point2f((float)imagePoints[i].X, (float)imagePoints[i].Y);
        }

        return SolvePnPRansac(o, p, cameraMatrix, distCoeffs, method, iterationsCount, reprojectionError, confidence, extrinsicGuess);
    }

    /// <summary>
    /// Refines a pose with Levenberg-Marquardt minimization of the reprojection error, as OpenCV
    /// <c>solvePnPRefineLM</c>.
    /// </summary>
    /// <remarks>
    /// The OpenCV 5.0 <c>LevMarq</c> solver runs with geodesic acceleration, <c>MaxCount</c>
    /// iterations, step norm tolerance <c>Epsilon</c> and small energy tolerance <c>Epsilon²</c>.
    /// </remarks>
    /// <param name="objectPoints">Object points (at least 3).</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients.</param>
    /// <param name="rotationVector">Initial rotation vector on input; refined rotation vector on output.</param>
    /// <param name="translationVector">Initial translation vector on input; refined translation vector on output.</param>
    /// <param name="criteria">Termination criteria, or null for <see cref="DefaultRefineCriteria"/>.</param>
    public static void SolvePnPRefineLM(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        ref Point3d rotationVector,
        ref Point3d translationVector,
        TermCriteria? criteria = null)
    {
        CheckRefineInputs(objectPoints, imagePoints);
        TermCriteria c = criteria ?? DefaultRefineCriteria;
        double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        double[] dist = distCoeffs.ToArray();
        CameraDistortion.Expand(dist, nameof(distCoeffs));
        double[] param = [rotationVector.X, rotationVector.Y, rotationVector.Z, translationVector.X, translationVector.Y, translationVector.Z];
        CameraLevMarqCallback callback = CameraPnP.ReprojectionCallback(objectPoints.ToArray(), imagePoints.ToArray(), a, dist);
        new CameraLevMarq(param, 2 * objectPoints.Length, callback, c.MaxCount, true, c.Epsilon, c.Epsilon * c.Epsilon).Optimize();
        rotationVector = new Point3d(param[0], param[1], param[2]);
        translationVector = new Point3d(param[3], param[4], param[5]);
    }

    /// <summary>
    /// Refines a pose with virtual visual servoing, as OpenCV <c>solvePnPRefineVVS</c>.
    /// </summary>
    /// <remarks>
    /// Each iteration computes the interaction matrix L and the normalized residuals e, applies
    /// the twist <c>−λ·L⁺·e</c> through the SE(3) exponential map, and stops after <c>MaxCount</c>
    /// iterations or when the change of <c>eᵀe</c> is below <c>Epsilon</c>.
    /// </remarks>
    /// <param name="objectPoints">Object points (at least 3).</param>
    /// <param name="imagePoints">Corresponding image points in pixels.</param>
    /// <param name="cameraMatrix">Camera matrix (3 × 3).</param>
    /// <param name="distCoeffs">Distortion coefficients.</param>
    /// <param name="rotationVector">Initial rotation vector on input; refined rotation vector on output.</param>
    /// <param name="translationVector">Initial translation vector on input; refined translation vector on output.</param>
    /// <param name="criteria">Termination criteria, or null for <see cref="DefaultRefineCriteria"/>.</param>
    /// <param name="vvsLambda">Gain λ of the control law.</param>
    public static void SolvePnPRefineVVS(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        ref Point3d rotationVector,
        ref Point3d translationVector,
        TermCriteria? criteria = null,
        double vvsLambda = 1)
    {
        CheckRefineInputs(objectPoints, imagePoints);
        TermCriteria c = criteria ?? DefaultRefineCriteria;
        (rotationVector, translationVector) = PnpRefineVvs.Run(objectPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector, c, vvsLambda);
    }

    /// <summary>OpenCV <c>solvePnPGeneric</c> without the reprojection errors.</summary>
    /// <param name="floatPoints">True when OpenCV receives <c>float</c> points: the undistorted points are rounded to <c>float</c>.</param>
    internal static List<PnPPose> Generic(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method,
        bool useExtrinsicGuess,
        PnPPose guess,
        bool floatPoints)
    {
        ArgumentNullException.ThrowIfNull(cameraMatrix);
        int n = objectPoints.Length;
        if (n != imagePoints.Length)
        {
            throw new ArgumentException("The object and image point counts must be equal.", nameof(imagePoints));
        }

        if (method != SolvePnPMethod.Iterative)
        {
            useExtrinsicGuess = false;
        }

        if (!(n >= 4 || (n == 3 && method == SolvePnPMethod.Iterative && useExtrinsicGuess) || (n >= 3 && method == SolvePnPMethod.Sqpnp)))
        {
            throw new ArgumentException("Too few points for the method.", nameof(objectPoints));
        }

        var poses = new List<PnPPose>();
        switch (method)
        {
            case SolvePnPMethod.Iterative:
            {
                double[] param = [guess.RotationVector.X, guess.RotationVector.Y, guess.RotationVector.Z, guess.TranslationVector.X, guess.TranslationVector.Y, guess.TranslationVector.Z];
                CameraPnP.FindExtrinsic(objectPoints, imagePoints, cameraMatrix, distCoeffs, param, useExtrinsicGuess);
                poses.Add(new PnPPose(new Point3d(param[0], param[1], param[2]), new Point3d(param[3], param[4], param[5])));
                break;
            }

            case SolvePnPMethod.Epnp:
            {
                Point2d[] und = Undistort(imagePoints, cameraMatrix, distCoeffs, floatPoints);
                double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
                var r = new double[9];
                var t = new double[3];
                Epnp.ComputePose(a[0], a[4], a[2], a[5], objectPoints, und, r, t);
                (double x, double y, double z) = CameraRodrigues.MatrixToVector(r, null);
                poses.Add(new PnPPose(new Point3d(x, y, z), new Point3d(t[0], t[1], t[2])));
                break;
            }

            case SolvePnPMethod.P3p:
            case SolvePnPMethod.Ap3p:
                if (n > 4)
                {
                    throw new ArgumentException("P3P and AP3P need 3 or 4 points.", nameof(objectPoints));
                }

                poses.AddRange(P3PCore(objectPoints, imagePoints, cameraMatrix, distCoeffs, method, floatPoints));
                break;

            case SolvePnPMethod.Ippe:
            case SolvePnPMethod.IppeSquare:
            {
                if (method == SolvePnPMethod.IppeSquare && n != 4)
                {
                    throw new ArgumentException("IPPE_SQUARE needs 4 points.", nameof(objectPoints));
                }

                Point2d[] und = Undistort(imagePoints, cameraMatrix, distCoeffs, floatPoints);
                bool ok = method == SolvePnPMethod.Ippe
                    ? Ippe.SolveGeneric(objectPoints, und, out Point3d r1, out Point3d t1, out float e1, out Point3d r2, out Point3d t2, out float e2)
                    : Ippe.SolveSquare(objectPoints, und, out r1, out t1, out e1, out r2, out t2, out e2);
                if (ok)
                {
                    if (e1 < e2)
                    {
                        poses.Add(new PnPPose(r1, t1));
                        poses.Add(new PnPPose(r2, t2));
                    }
                    else
                    {
                        poses.Add(new PnPPose(r2, t2));
                        poses.Add(new PnPPose(r1, t1));
                    }
                }

                break;
            }

            case SolvePnPMethod.Sqpnp:
            {
                Point2d[] und = Undistort(imagePoints, cameraMatrix, distCoeffs, floatPoints);
                var rv = new List<Point3d>();
                var tv = new List<Point3d>();
                Sqpnp.Solve(objectPoints, und, rv, tv);
                for (int i = 0; i < rv.Count; i++)
                {
                    poses.Add(new PnPPose(rv[i], tv[i]));
                }

                break;
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(method), "Unknown PnP method.");
        }

        return poses;
    }

    private static List<PnPPose> P3PCore(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        SolvePnPMethod method,
        bool floatPoints)
    {
        Point2d[] und = Undistort(imagePoints, cameraMatrix, distCoeffs, floatPoints);
        var rs = new List<double[]>();
        var ts = new List<double[]>();
        int solutions;
        if (method == SolvePnPMethod.P3p)
        {
            solutions = P3p.Estimate(objectPoints, und, rs, ts);
        }
        else
        {
            double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
            solutions = Ap3p.Solve(a[0], a[4], a[2], a[5], objectPoints, und, rs, ts);
        }

        var poses = new List<PnPPose>(solutions);
        var errors = new double[solutions];
        int n = objectPoints.Length;
        var err = new Mat<double>(1, 2 * n);
        for (int i = 0; i < solutions; i++)
        {
            (double x, double y, double z) = CameraRodrigues.MatrixToVector(rs[i], null);
            var pose = new PnPPose(new Point3d(x, y, z), new Point3d(ts[i][0], ts[i][1], ts[i][2]));
            poses.Add(pose);
            Point2d[] proj = CameraModel.ProjectPoints(objectPoints, pose.RotationVector, pose.TranslationVector, cameraMatrix, distCoeffs);
            for (int k = 0; k < n; k++)
            {
                err.At(2 * k, 0) = imagePoints[k].X - proj[k].X;
                err.At(2 * k + 1, 0) = imagePoints[k].Y - proj[k].Y;
            }

            errors[i] = LinearAlgebra.Gemm(err, err, 1, null, 0, GemmFlags.ATranspose).At(0, 0);
        }

        for (int i = 1; i < solutions; i++)
        {
            for (int j = i; j > 0 && errors[j - 1] > errors[j]; j--)
            {
                (errors[j], errors[j - 1]) = (errors[j - 1], errors[j]);
                (poses[j], poses[j - 1]) = (poses[j - 1], poses[j]);
            }
        }

        return poses;
    }

    private static Point2d[] Undistort(ReadOnlySpan<Point2d> imagePoints, Mat<double> cameraMatrix, ReadOnlySpan<double> distCoeffs, bool floatPoints)
    {
        if (!floatPoints)
        {
            return Undistortion.UndistortPoints(imagePoints, cameraMatrix, distCoeffs);
        }

        var f = new Point2f[imagePoints.Length];
        for (int i = 0; i < f.Length; i++)
        {
            f[i] = new Point2f((float)imagePoints[i].X, (float)imagePoints[i].Y);
        }

        Point2f[] u = Undistortion.UndistortPoints(f, cameraMatrix, distCoeffs);
        var d = new Point2d[u.Length];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = new Point2d(u[i].X, u[i].Y);
        }

        return d;
    }

    /// <summary>
    /// OpenCV <c>NormDiffL2_SIMD&lt;double, double&gt;</c> with 2-lane SSE2 vectors: four accumulators
    /// over blocks of 8 elements, then a scalar tail.
    /// </summary>
    private static double NormDiffL2Sqr(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        Span<double> r = stackalloc double[8];
        int n = a.Length;
        int j = 0;
        for (; j <= n - 8; j += 8)
        {
            for (int k = 0; k < 8; k++)
            {
                double v = Math.Abs(a[j + k] - b[j + k]);
                r[k] = r[k] + (v * v);
            }
        }

        double l0 = ((r[0] + r[2]) + r[4]) + r[6];
        double l1 = ((r[1] + r[3]) + r[5]) + r[7];
        double s = 0.0 + (l0 + l1);
        for (; j < n; j++)
        {
            double v = a[j] - b[j];
            s += v * v;
        }

        return 0.0 + s;
    }

    private static void CheckRefineInputs(ReadOnlySpan<Point3d> objectPoints, ReadOnlySpan<Point2d> imagePoints)
    {
        if (objectPoints.Length < 3 || objectPoints.Length != imagePoints.Length)
        {
            throw new ArgumentException("At least 3 points are necessary and the object and image point counts must be equal.", nameof(objectPoints));
        }
    }
}
