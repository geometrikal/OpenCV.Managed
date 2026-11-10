// This implementation is ported from the OpenCV IPPE (Infinitesimal Plane-based Pose Estimation) solver.
//
// Original sources:
//   modules/geometry/src/ippe.cpp (IPPE::PoseSolver::solveGeneric, IPPE::PoseSolver::solveSquare,
//     HomographyHO::homographyHO, HomographyHO::normalizeDataIsotropic)
//   modules/geometry/src/solvepnp.cpp (solvePnPGeneric with SOLVEPNP_IPPE and SOLVEPNP_IPPE_SQUARE)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// <c>IPPE::PoseSolver</c>: the two pose solutions of a planar object from a homography, sorted by
/// reprojection error. The image points are normalized (undistorted, camera matrix removed).
/// </summary>
/// <remarks>
/// A method returns false where OpenCV raises an exception. <c>solvePnPGeneric</c> catches the
/// exception and returns no solution.
/// </remarks>
internal static class Ippe
{
    private const double IppeSmall = 1e-3;
    private const double FltEpsilon = 1.1920928955078125e-7;

    /// <summary>OpenCV <c>IPPE::PoseSolver::solveGeneric</c>.</summary>
    /// <param name="objectPoints">Four or more coplanar object points.</param>
    /// <param name="normalizedPoints">Normalized image points, one for each object point.</param>
    /// <param name="rvec1">Rotation vector of the pose with the smaller reprojection error.</param>
    /// <param name="tvec1">Translation vector of the pose with the smaller reprojection error.</param>
    /// <param name="reprojErr1">RMS reprojection error of the first pose.</param>
    /// <param name="rvec2">Rotation vector of the second pose.</param>
    /// <param name="tvec2">Translation vector of the second pose.</param>
    /// <param name="reprojErr2">RMS reprojection error of the second pose.</param>
    /// <returns>False when OpenCV raises an exception (solvePnPGeneric catches it and returns no solution).</returns>
    public static bool SolveGeneric(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> normalizedPoints,
        out Point3d rvec1,
        out Point3d tvec1,
        out float reprojErr1,
        out Point3d rvec2,
        out Point3d tvec2,
        out float reprojErr2)
    {
        rvec1 = tvec1 = rvec2 = tvec2 = default;
        reprojErr1 = reprojErr2 = 0;
        int n = normalizedPoints.Length;
        if (n < 4 || objectPoints.Length != n)
        {
            return false;
        }

        if (!MakeCanonicalObjectPoints(objectPoints, out Point2d[] canonicalObjPoints, out Mat<double> modelToCanonical))
        {
            return false;
        }

        Span<double> h = stackalloc double[9];
        HomographyHO(canonicalObjPoints, normalizedPoints, h);

        var maCanon = new Mat<double>(4, 4);
        var mbCanon = new Mat<double>(4, 4);
        if (!SolveCanonicalForm(canonicalObjPoints, normalizedPoints, h, maCanon, mbCanon))
        {
            return false;
        }

        Mat<double> ma = Mul(maCanon, modelToCanonical);
        Mat<double> mb = Mul(mbCanon, modelToCanonical);

        SortAndOutput(objectPoints, normalizedPoints, ma, mb, out rvec1, out tvec1, out reprojErr1, out rvec2, out tvec2, out reprojErr2);
        return true;
    }

    /// <summary>OpenCV <c>IPPE::PoseSolver::solveSquare</c>.</summary>
    /// <param name="objectPoints">
    /// The four corners of a square in the order of <c>SOLVEPNP_IPPE_SQUARE</c>, on the plane z = 0.
    /// </param>
    /// <param name="normalizedPoints">Normalized image points, one for each object point.</param>
    /// <param name="rvec1">Rotation vector of the pose with the smaller reprojection error.</param>
    /// <param name="tvec1">Translation vector of the pose with the smaller reprojection error.</param>
    /// <param name="reprojErr1">RMS reprojection error of the first pose.</param>
    /// <param name="rvec2">Rotation vector of the second pose.</param>
    /// <param name="tvec2">Translation vector of the second pose.</param>
    /// <param name="reprojErr2">RMS reprojection error of the second pose.</param>
    /// <returns>False when OpenCV raises an exception, or when the point count is not 4.</returns>
    public static bool SolveSquare(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> normalizedPoints,
        out Point3d rvec1,
        out Point3d tvec1,
        out float reprojErr1,
        out Point3d rvec2,
        out Point3d tvec2,
        out float reprojErr2)
    {
        rvec1 = tvec1 = rvec2 = tvec2 = default;
        reprojErr1 = reprojErr2 = 0;
        if (objectPoints.Length != 4 || normalizedPoints.Length != 4)
        {
            return false;
        }

        var objectPoints2D = new Point2d[4];
        for (int i = 0; i < 4; i++)
        {
            objectPoints2D[i] = new Point2d(objectPoints[i].X, objectPoints[i].Y);
        }

        double squareLength = Math.Sqrt(
            (objectPoints[1].X - objectPoints[0].X) * (objectPoints[1].X - objectPoints[0].X) +
            (objectPoints[1].Y - objectPoints[0].Y) * (objectPoints[1].Y - objectPoints[0].Y));

        Span<double> h = stackalloc double[9];
        if (!HomographyFromSquarePoints(normalizedPoints, squareLength / 2.0, h))
        {
            return false;
        }

        var ma = new Mat<double>(4, 4);
        var mb = new Mat<double>(4, 4);
        if (!SolveCanonicalForm(objectPoints2D, normalizedPoints, h, ma, mb))
        {
            return false;
        }

        SortAndOutput(objectPoints, normalizedPoints, ma, mb, out rvec1, out tvec1, out reprojErr1, out rvec2, out tvec2, out reprojErr2);
        return true;
    }

    private static Mat<double> Mul(Mat<double> a, Mat<double> b, double alpha = 1.0, GemmFlags flags = GemmFlags.None)
        => LinearAlgebra.Gemm(a, b, alpha, null, 0, flags);

    /// <summary>OpenCV <c>sortPosesByReprojError</c>, then <c>rot2vec</c> and the translation column of each pose.</summary>
    private static void SortAndOutput(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> ma,
        Mat<double> mb,
        out Point3d rvec1,
        out Point3d tvec1,
        out float err1,
        out Point3d rvec2,
        out Point3d tvec2,
        out float err2)
    {
        var projected = new Point2d[objectPoints.Length];
        float erra = EvalReprojError(objectPoints, imagePoints, ma, projected);
        float errb = EvalReprojError(objectPoints, imagePoints, mb, projected);
        Mat<double> m1, m2;
        if (erra < errb)
        {
            err1 = erra;
            m1 = ma;
            err2 = errb;
            m2 = mb;
        }
        else
        {
            err1 = errb;
            m1 = mb;
            err2 = erra;
            m2 = ma;
        }

        rvec1 = Rot2Vec(m1);
        rvec2 = Rot2Vec(m2);
        tvec1 = new Point3d(m1.At(0, 3), m1.At(1, 3), m1.At(2, 3));
        tvec2 = new Point3d(m2.At(0, 3), m2.At(1, 3), m2.At(2, 3));
    }

    /// <summary>OpenCV <c>evalReprojError</c>: <c>projectPoints</c> with an identity camera matrix and no distortion.</summary>
    private static float EvalReprojError(ReadOnlySpan<Point3d> objectPoints, ReadOnlySpan<Point2d> imagePoints, Mat<double> m, Point2d[] projected)
    {
        Point3d r = Rot2Vec(m);
        var dRdr = new double[27];
        double[] rot = CameraRodrigues.VectorToMatrix(r.X, r.Y, r.Z, dRdr);
        double[] t = [m.At(0, 3), m.At(1, 3), m.At(2, 3)];
        CameraProjection.Project(objectPoints, rot, dRdr, t, CameraDistortion.Identity(), default, 0, projected, null);

        float err = 0;
        int n = objectPoints.Length;
        for (int i = 0; i < n; i++)
        {
            float dx = (float)(projected[i].X - imagePoints[i].X);
            float dy = (float)(projected[i].Y - imagePoints[i].Y);
            err += dx * dx + dy * dy;
        }

        return MathF.Sqrt(err / (2.0f * n));
    }

    /// <summary>OpenCV <c>rot2vec</c> of the upper-left 3 × 3 block of a 4 × 4 pose.</summary>
    private static Point3d Rot2Vec(Mat<double> m)
    {
        double trace = m.At(0, 0) + m.At(1, 1) + m.At(2, 2);
        double wNorm = Math.Acos((trace - 1.0) / 2.0);
        double d = 1 / (2 * Math.Sin(wNorm)) * wNorm;
        if (wNorm < FltEpsilon)
        {
            return default;
        }

        double c0 = m.At(2, 1) - m.At(1, 2);
        double c1 = m.At(0, 2) - m.At(2, 0);
        double c2 = m.At(1, 0) - m.At(0, 1);
        return new Point3d(d * c0, d * c1, d * c2);
    }

    /// <summary>OpenCV <c>solveCanonicalForm</c>. <paramref name="ma"/> and <paramref name="mb"/> are zero 4 × 4 matrices on entry.</summary>
    private static bool SolveCanonicalForm(
        ReadOnlySpan<Point2d> canonicalObjPoints,
        ReadOnlySpan<Point2d> normalizedInputPoints,
        ReadOnlySpan<double> h,
        Mat<double> ma,
        Mat<double> mb)
    {
        ma.At(3, 3) = 1;
        mb.At(3, 3) = 1;

        double j00 = h[0] - h[6] * h[2];
        double j01 = h[1] - h[7] * h[2];
        double j10 = h[3] - h[6] * h[5];
        double j11 = h[4] - h[7] * h[5];

        double v0 = h[2];
        double v1 = h[5];

        if (!ComputeRotations(j00, j01, j10, j11, v0, v1, ma, mb))
        {
            return false;
        }

        ComputeTranslation(canonicalObjPoints, normalizedInputPoints, ma);
        ComputeTranslation(canonicalObjPoints, normalizedInputPoints, mb);
        return true;
    }

    /// <summary>OpenCV <c>computeTranslation</c>: reads the rotation block of <paramref name="m"/> and writes its translation column.</summary>
    private static void ComputeTranslation(ReadOnlySpan<Point2d> objectPoints, ReadOnlySpan<Point2d> imgPoints, Mat<double> m)
    {
        int n = imgPoints.Length;

        double ata00 = n;
        double ata02 = 0;
        double ata11 = n;
        double ata12 = 0;
        double ata20 = 0;
        double ata21 = 0;
        double ata22 = 0;

        double atb0 = 0;
        double atb1 = 0;
        double atb2 = 0;

        double r00 = m.At(0, 0), r01 = m.At(0, 1);
        double r10 = m.At(1, 0), r11 = m.At(1, 1);
        double r20 = m.At(2, 0), r21 = m.At(2, 1);

        for (int i = 0; i < n; i++)
        {
            Point2d objPt = objectPoints[i];
            double rx = r00 * objPt.X + r01 * objPt.Y;
            double ry = r10 * objPt.X + r11 * objPt.Y;
            double rz = r20 * objPt.X + r21 * objPt.Y;

            double a2 = -imgPoints[i].X;
            double b2 = -imgPoints[i].Y;

            ata02 = ata02 + a2;
            ata12 = ata12 + b2;
            ata20 = ata20 + a2;
            ata21 = ata21 + b2;
            ata22 = ata22 + a2 * a2 + b2 * b2;

            double bx = -a2 * rz - rx;
            double by = -b2 * rz - ry;

            atb0 = atb0 + bx;
            atb1 = atb1 + by;
            atb2 = atb2 + a2 * bx + b2 * by;
        }

        double detAInv = 1.0 / (ata00 * ata11 * ata22 - ata00 * ata12 * ata21 - ata02 * ata11 * ata20);

        double s00 = ata11 * ata22 - ata12 * ata21;
        double s01 = ata02 * ata21;
        double s02 = -ata02 * ata11;
        double s10 = ata12 * ata20;
        double s11 = ata00 * ata22 - ata02 * ata20;
        double s12 = -ata00 * ata12;
        double s20 = -ata11 * ata20;
        double s21 = -ata00 * ata21;
        double s22 = ata00 * ata11;

        m.At(0, 3) = detAInv * (s00 * atb0 + s01 * atb1 + s02 * atb2);
        m.At(1, 3) = detAInv * (s10 * atb0 + s11 * atb1 + s12 * atb2);
        m.At(2, 3) = detAInv * (s20 * atb0 + s21 * atb1 + s22 * atb2);
    }

    /// <summary>OpenCV <c>computeRotations</c>: writes the two rotation solutions into the rotation blocks of <paramref name="m1"/> and <paramref name="m2"/>.</summary>
    private static bool ComputeRotations(double j00, double j01, double j10, double j11, double p, double q, Mat<double> m1, Mat<double> m2)
    {
        Span<double> rv = stackalloc double[9];
        RotateVec2ZAxis(p, q, 1, rv);

        // Rv = Rv.t()
        double rv00 = rv[0];
        double rv01 = rv[3];
        double rv02 = rv[6];

        double rv10 = rv[1];
        double rv11 = rv[4];
        double rv12 = rv[7];

        double rv20 = rv[2];
        double rv21 = rv[5];
        double rv22 = rv[8];

        double b00 = rv00 - p * rv20;
        double b01 = rv01 - p * rv21;
        double b10 = rv10 - q * rv20;
        double b11 = rv11 - q * rv21;

        double dtinv = 1.0 / ((b00 * b11 - b01 * b10));

        double binv00 = dtinv * b11;
        double binv01 = -dtinv * b01;
        double binv10 = -dtinv * b10;
        double binv11 = dtinv * b00;

        double a00 = binv00 * j00 + binv01 * j10;
        double a01 = binv00 * j01 + binv01 * j11;
        double a10 = binv10 * j00 + binv11 * j10;
        double a11 = binv10 * j01 + binv11 * j11;

        double ata00 = a00 * a00 + a01 * a01;
        double ata01 = a00 * a10 + a01 * a11;
        double ata11 = a10 * a10 + a11 * a11;

        double gamma2 = 0.5 * (ata00 + ata11 + Math.Sqrt((ata00 - ata11) * (ata00 - ata11) + 4.0 * ata01 * ata01));
        if (gamma2 < 0)
        {
            return false;
        }

        double gamma = Math.Sqrt(gamma2);

        if (Math.Abs(gamma) < FltEpsilon)
        {
            return false;
        }

        double rtilde00 = a00 / gamma;
        double rtilde01 = a01 / gamma;
        double rtilde10 = a10 / gamma;
        double rtilde11 = a11 / gamma;

        double rtilde00_2 = rtilde00 * rtilde00;
        double rtilde01_2 = rtilde01 * rtilde01;
        double rtilde10_2 = rtilde10 * rtilde10;
        double rtilde11_2 = rtilde11 * rtilde11;

        double b0 = Math.Sqrt(-rtilde00_2 - rtilde10_2 + 1);
        double b1 = Math.Sqrt(-rtilde01_2 - rtilde11_2 + 1);
        double sp = -rtilde00 * rtilde01 - rtilde10 * rtilde11;

        if (sp < 0)
        {
            b1 = -b1;
        }

        m1.At(0, 0) = (rtilde00) * rv00 + (rtilde10) * rv01 + (b0) * rv02;
        m1.At(0, 1) = (rtilde01) * rv00 + (rtilde11) * rv01 + (b1) * rv02;
        m1.At(0, 2) = (b1 * rtilde10 - b0 * rtilde11) * rv00 + (b0 * rtilde01 - b1 * rtilde00) * rv01 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv02;
        m1.At(1, 0) = (rtilde00) * rv10 + (rtilde10) * rv11 + (b0) * rv12;
        m1.At(1, 1) = (rtilde01) * rv10 + (rtilde11) * rv11 + (b1) * rv12;
        m1.At(1, 2) = (b1 * rtilde10 - b0 * rtilde11) * rv10 + (b0 * rtilde01 - b1 * rtilde00) * rv11 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv12;
        m1.At(2, 0) = (rtilde00) * rv20 + (rtilde10) * rv21 + (b0) * rv22;
        m1.At(2, 1) = (rtilde01) * rv20 + (rtilde11) * rv21 + (b1) * rv22;
        m1.At(2, 2) = (b1 * rtilde10 - b0 * rtilde11) * rv20 + (b0 * rtilde01 - b1 * rtilde00) * rv21 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv22;

        m2.At(0, 0) = (rtilde00) * rv00 + (rtilde10) * rv01 + (-b0) * rv02;
        m2.At(0, 1) = (rtilde01) * rv00 + (rtilde11) * rv01 + (-b1) * rv02;
        m2.At(0, 2) = (b0 * rtilde11 - b1 * rtilde10) * rv00 + (b1 * rtilde00 - b0 * rtilde01) * rv01 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv02;
        m2.At(1, 0) = (rtilde00) * rv10 + (rtilde10) * rv11 + (-b0) * rv12;
        m2.At(1, 1) = (rtilde01) * rv10 + (rtilde11) * rv11 + (-b1) * rv12;
        m2.At(1, 2) = (b0 * rtilde11 - b1 * rtilde10) * rv10 + (b1 * rtilde00 - b0 * rtilde01) * rv11 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv12;
        m2.At(2, 0) = (rtilde00) * rv20 + (rtilde10) * rv21 + (-b0) * rv22;
        m2.At(2, 1) = (rtilde01) * rv20 + (rtilde11) * rv21 + (-b1) * rv22;
        m2.At(2, 2) = (b0 * rtilde11 - b1 * rtilde10) * rv20 + (b1 * rtilde00 - b0 * rtilde01) * rv21 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv22;
        return true;
    }

    /// <summary>OpenCV <c>rotateVec2ZAxis</c>: <paramref name="ra"/> (3 × 3, row-major) rotates the vector a to the z axis.</summary>
    private static void RotateVec2ZAxis(double ax, double ay, double az, Span<double> ra)
    {
        double nrm = Math.Sqrt(ax * ax + ay * ay + az * az);
        ax = ax / nrm;
        ay = ay / nrm;
        az = az / nrm;

        double c = az;

        if (Math.Abs(1.0 + c) < FltEpsilon)
        {
            ra.Clear();
            ra[0] = 1.0;
            ra[4] = 1.0;
            ra[8] = -1.0;
        }
        else
        {
            double d = 1.0 / (1.0 + c);
            double ax2 = ax * ax;
            double ay2 = ay * ay;
            double axay = ax * ay;

            ra[0] = -ax2 * d + 1.0;
            ra[1] = -axay * d;
            ra[2] = -ax;

            ra[3] = -axay * d;
            ra[4] = -ay2 * d + 1.0;
            ra[5] = -ay;

            ra[6] = ax;
            ra[7] = ay;
            ra[8] = 1.0 - (ax2 + ay2) * d;
        }
    }

    /// <summary>OpenCV <c>homographyFromSquarePoints</c>: <paramref name="h"/> is 3 × 3, row-major.</summary>
    private static bool HomographyFromSquarePoints(ReadOnlySpan<Point2d> pts, double halfLength, Span<double> h)
    {
        double p1x = -pts[0].X;
        double p1y = -pts[0].Y;

        double p2x = -pts[1].X;
        double p2y = -pts[1].Y;

        double p3x = -pts[2].X;
        double p3y = -pts[2].Y;

        double p4x = -pts[3].X;
        double p4y = -pts[3].Y;

        double det = (halfLength * (p1x * p2y - p2x * p1y - p1x * p4y + p2x * p3y - p3x * p2y + p4x * p1y + p3x * p4y - p4x * p3y));
        if (Math.Abs(det) < 1e-9)
        {
            return false;
        }

        double detsInv = -1 / det;

        h[0] = detsInv * (p1x * p3x * p2y - p2x * p3x * p1y - p1x * p4x * p2y + p2x * p4x * p1y - p1x * p3x * p4y + p1x * p4x * p3y + p2x * p3x * p4y - p2x * p4x * p3y);
        h[1] = detsInv * (p1x * p2x * p3y - p1x * p3x * p2y - p1x * p2x * p4y + p2x * p4x * p1y + p1x * p3x * p4y - p3x * p4x * p1y - p2x * p4x * p3y + p3x * p4x * p2y);
        h[2] = detsInv * halfLength * (p1x * p2x * p3y - p2x * p3x * p1y - p1x * p2x * p4y + p1x * p4x * p2y - p1x * p4x * p3y + p3x * p4x * p1y + p2x * p3x * p4y - p3x * p4x * p2y);
        h[3] = detsInv * (p1x * p2y * p3y - p2x * p1y * p3y - p1x * p2y * p4y + p2x * p1y * p4y - p3x * p1y * p4y + p4x * p1y * p3y + p3x * p2y * p4y - p4x * p2y * p3y);
        h[4] = detsInv * (p2x * p1y * p3y - p3x * p1y * p2y - p1x * p2y * p4y + p4x * p1y * p2y + p1x * p3y * p4y - p4x * p1y * p3y - p2x * p3y * p4y + p3x * p2y * p4y);
        h[5] = detsInv * halfLength * (p1x * p2y * p3y - p3x * p1y * p2y - p2x * p1y * p4y + p4x * p1y * p2y - p1x * p3y * p4y + p3x * p1y * p4y + p2x * p3y * p4y - p4x * p2y * p3y);
        h[6] = -detsInv * (p1x * p3y - p3x * p1y - p1x * p4y - p2x * p3y + p3x * p2y + p4x * p1y + p2x * p4y - p4x * p2y);
        h[7] = detsInv * (p1x * p2y - p2x * p1y - p1x * p3y + p3x * p1y + p2x * p4y - p4x * p2y - p3x * p4y + p4x * p3y);
        h[8] = 1.0;
        return true;
    }

    /// <summary>
    /// OpenCV <c>makeCanonicalObjectPoints</c>: moves the object points to zero mean and rotates
    /// them to the plane z = 0. <paramref name="modelToCanonical"/> is the 4 × 4 transform.
    /// </summary>
    private static bool MakeCanonicalObjectPoints(ReadOnlySpan<Point3d> objectPoints, out Point2d[] canonicalObjPoints, out Mat<double> modelToCanonical)
    {
        int n = objectPoints.Length;
        canonicalObjPoints = new Point2d[n];
        modelToCanonical = new Mat<double>(4, 4);

        var uZero = new Mat<double>(n, 3);

        double xBar = 0;
        double yBar = 0;
        double zBar = 0;
        bool isOnZPlane = true;
        for (int i = 0; i < n; i++)
        {
            double x = objectPoints[i].X;
            double y = objectPoints[i].Y;
            double z = objectPoints[i].Z;

            if (Math.Abs(z) > IppeSmall)
            {
                isOnZPlane = false;
            }

            xBar += x;
            yBar += y;
            zBar += z;

            uZero.At(0, i) = x;
            uZero.At(1, i) = y;
            uZero.At(2, i) = z;
        }

        xBar = xBar / n;
        yBar = yBar / n;
        zBar = zBar / n;

        for (int i = 0; i < n; i++)
        {
            uZero.At(0, i) -= xBar;
            uZero.At(1, i) -= yBar;
            uZero.At(2, i) -= zBar;
        }

        Span<double> mCenter = stackalloc double[16];
        mCenter[0] = 1;
        mCenter[5] = 1;
        mCenter[10] = 1;
        mCenter[15] = 1;
        mCenter[3] = -xBar;
        mCenter[7] = -yBar;
        mCenter[11] = -zBar;

        if (isOnZPlane)
        {
            mCenter.CopyTo(modelToCanonical.GetSpan());
            for (int i = 0; i < n; i++)
            {
                canonicalObjPoints[i] = new Point2d(uZero.At(0, i), uZero.At(1, i));
            }

            return true;
        }

        var r = new double[9];
        if (!ComputeObjectSpaceR3Pts(objectPoints, r) && !ComputeObjectSpaceRSvd(uZero, r))
        {
            return false;
        }

        Mat<double> uZeroAligned = Mul(new Mat<double>(r, 3, 3), uZero);

        for (int i = 0; i < n; i++)
        {
            canonicalObjPoints[i] = new Point2d(uZeroAligned.At(0, i), uZeroAligned.At(1, i));
            if (Math.Abs(uZeroAligned.At(2, i)) > IppeSmall)
            {
                return false;
            }
        }

        Span<double> mRot = stackalloc double[16];
        mRot[15] = 1;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                mRot[i * 4 + j] = r[i * 3 + j];
            }
        }

        // Matx44d product: each sum starts from 0 and adds the products in index order.
        Span<double> mb = modelToCanonical.GetSpan();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                double s = 0;
                for (int k = 0; k < 4; k++)
                {
                    s += mRot[i * 4 + k] * mCenter[k * 4 + j];
                }

                mb[i * 4 + j] = s;
            }
        }

        return true;
    }

    /// <summary>OpenCV <c>computeObjextSpaceR3Pts</c>: the rotation from the normal of the first three points.</summary>
    private static bool ComputeObjectSpaceR3Pts(ReadOnlySpan<Point3d> objectPoints, double[] r)
    {
        double p1x = objectPoints[0].X, p1y = objectPoints[0].Y, p1z = objectPoints[0].Z;
        double p2x = objectPoints[1].X, p2y = objectPoints[1].Y, p2z = objectPoints[1].Z;
        double p3x = objectPoints[2].X, p3y = objectPoints[2].Y, p3z = objectPoints[2].Z;

        double nx = (p1y - p2y) * (p1z - p3z) - (p1y - p3y) * (p1z - p2z);
        double ny = (p1x - p3x) * (p1z - p2z) - (p1x - p2x) * (p1z - p3z);
        double nz = (p1x - p2x) * (p1y - p3y) - (p1x - p3x) * (p1y - p2y);

        double nrm = Math.Sqrt(nx * nx + ny * ny + nz * nz);
        if (nrm > IppeSmall)
        {
            nx = nx / nrm;
            ny = ny / nrm;
            nz = nz / nrm;
            RotateVec2ZAxis(nx, ny, nz, r);
            return true;
        }

        return false;
    }

    /// <summary>OpenCV <c>computeObjextSpaceRSvD</c>: the rotation from the SVD of the zero-mean points (3 × n).</summary>
    private static bool ComputeObjectSpaceRSvd(Mat<double> objectPointsZeroMean, double[] r)
    {
        var svd = new Svd<double>(Mul(objectPointsZeroMean, objectPointsZeroMean, 1.0, GemmFlags.BTranspose));
        double s3 = svd.W.At(2, 0);
        double s2 = svd.W.At(1, 0);

        if (!(s3 / s2 < IppeSmall))
        {
            return false;
        }

        Mat<double> u = svd.U;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                r[i * 3 + j] = u.At(j, i);
            }
        }

        if (LinearAlgebra.Determinant(new Mat<double>(r, 3, 3)) < 0)
        {
            r[6] = -r[6];
            r[7] = -r[7];
            r[8] = -r[8];
        }

        return true;
    }

    /// <summary>
    /// OpenCV <c>HomographyHO::normalizeDataIsotropic</c>. <paramref name="dataN"/> is 2 × n;
    /// <paramref name="t"/> and <paramref name="ti"/> are 3 × 3, row-major, and zero on entry.
    /// </summary>
    private static void NormalizeDataIsotropic(ReadOnlySpan<Point2d> data, Mat<double> dataN, Span<double> t, Span<double> ti)
    {
        int numPoints = data.Length;

        double xm = 0;
        double ym = 0;
        for (int i = 0; i < numPoints; i++)
        {
            xm = xm + data[i].X;
            ym = ym + data[i].Y;
        }

        xm = xm / numPoints;
        ym = ym / numPoints;

        double kappa = 0;
        for (int i = 0; i < numPoints; i++)
        {
            double xh = data[i].X - xm;
            double yh = data[i].Y - ym;
            dataN.At(0, i) = xh;
            dataN.At(1, i) = yh;
            kappa = kappa + xh * xh + yh * yh;
        }

        double beta = Math.Sqrt(2 * numPoints / kappa);

        // DataN * beta is evaluated by convertTo as x * alpha + 0.
        Span<double> dn = dataN.GetSpan();
        for (int i = 0; i < dn.Length; i++)
        {
            dn[i] = dn[i] * beta + 0.0;
        }

        t[0] = 1.0 / beta;
        t[4] = 1.0 / beta;
        t[2] = xm;
        t[5] = ym;
        t[8] = 1;

        ti[0] = beta;
        ti[4] = beta;
        ti[2] = -beta * xm;
        ti[5] = -beta * ym;
        ti[8] = 1;
    }

    /// <summary>OpenCV <c>HomographyHO::homographyHO</c>: <paramref name="h"/> is 3 × 3, row-major.</summary>
    private static void HomographyHO(ReadOnlySpan<Point2d> srcPoints, ReadOnlySpan<Point2d> targPoints, Span<double> h)
    {
        int n = srcPoints.Length;
        var dataA = new Mat<double>(n, 2);
        var dataB = new Mat<double>(n, 2);
        var ta = new double[9];
        var tai = new double[9];
        var tb = new double[9];
        var tbi = new double[9];
        NormalizeDataIsotropic(srcPoints, dataA, ta, tai);
        NormalizeDataIsotropic(targPoints, dataB, tb, tbi);

        var c1 = new double[n];
        var c2 = new double[n];
        var c3 = new double[n];
        var c4 = new double[n];

        double mC1 = 0, mC2 = 0, mC3 = 0, mC4 = 0;

        for (int i = 0; i < n; i++)
        {
            c1[i] = -dataB.At(0, i) * dataA.At(0, i);
            c2[i] = -dataB.At(0, i) * dataA.At(1, i);
            c3[i] = -dataB.At(1, i) * dataA.At(0, i);
            c4[i] = -dataB.At(1, i) * dataA.At(1, i);

            mC1 += c1[i];
            mC2 += c2[i];
            mC3 += c3[i];
            mC4 += c4[i];
        }

        mC1 /= n;
        mC2 /= n;
        mC3 /= n;
        mC4 /= n;

        var mx = new Mat<double>(3, n);
        var my = new Mat<double>(3, n);

        for (int i = 0; i < n; i++)
        {
            mx.At(i, 0) = c1[i] - mC1;
            mx.At(i, 1) = c2[i] - mC2;
            mx.At(i, 2) = -dataB.At(0, i);

            my.At(i, 0) = c3[i] - mC3;
            my.At(i, 1) = c4[i] - mC4;
            my.At(i, 2) = -dataB.At(1, i);
        }

        var dataAT = new Mat<double>(2, n);
        for (int i = 0; i < n; i++)
        {
            dataAT.At(i, 0) = dataA.At(0, i);
            dataAT.At(i, 1) = dataA.At(1, i);
        }

        Mat<double> dataADataAT = Mul(dataA, dataAT);
        double dt = dataADataAT.At(0, 0) * dataADataAT.At(1, 1) - dataADataAT.At(0, 1) * dataADataAT.At(1, 0);

        var dataADataATi = new Mat<double>(2, 2);
        dataADataATi.At(0, 0) = dataADataAT.At(1, 1) / dt;
        dataADataATi.At(0, 1) = -dataADataAT.At(0, 1) / dt;
        dataADataATi.At(1, 0) = -dataADataAT.At(1, 0) / dt;
        dataADataATi.At(1, 1) = dataADataAT.At(0, 0) / dt;

        Mat<double> pp = Mul(dataADataATi, dataA);

        Mat<double> bx = Mul(pp, mx);
        Mat<double> by = Mul(pp, my);

        Mat<double> ex = Mul(dataAT, bx);
        Mat<double> ey = Mul(dataAT, by);

        var d = new Mat<double>(3, 2 * n);

        for (int i = 0; i < n; i++)
        {
            d.At(i, 0) = mx.At(i, 0) - ex.At(i, 0);
            d.At(i, 1) = mx.At(i, 1) - ex.At(i, 1);
            d.At(i, 2) = mx.At(i, 2) - ex.At(i, 2);

            d.At(i + n, 0) = my.At(i, 0) - ey.At(i, 0);
            d.At(i + n, 1) = my.At(i, 1) - ey.At(i, 1);
            d.At(i + n, 2) = my.At(i, 2) - ey.At(i, 2);
        }

        var dT = new Mat<double>(2 * n, 3);
        for (int i = 0; i < 2 * n; i++)
        {
            dT.At(0, i) = d.At(i, 0);
            dT.At(1, i) = d.At(i, 1);
            dT.At(2, i) = d.At(i, 2);
        }

        Mat<double> dDT = Mul(dT, d);

        LinearAlgebra.Eigen(dDT, out _, out Mat<double> u);

        var h789 = new Mat<double>(1, 3);
        h789.At(0, 0) = u.At(2, 0);
        h789.At(1, 0) = u.At(2, 1);
        h789.At(2, 0) = u.At(2, 2);

        // -Bx * h789 is one gemm with alpha = -1.
        Mat<double> h12 = Mul(bx, h789, -1.0);
        Mat<double> h45 = Mul(by, h789, -1.0);

        double h3 = -(mC1 * h789.At(0, 0) + mC2 * h789.At(1, 0));
        double h6 = -(mC3 * h789.At(0, 0) + mC4 * h789.At(1, 0));

        var hMat = new Mat<double>(3, 3);
        hMat.At(0, 0) = h12.At(0, 0);
        hMat.At(0, 1) = h12.At(1, 0);
        hMat.At(0, 2) = h3;

        hMat.At(1, 0) = h45.At(0, 0);
        hMat.At(1, 1) = h45.At(1, 0);
        hMat.At(1, 2) = h6;

        hMat.At(2, 0) = h789.At(0, 0);
        hMat.At(2, 1) = h789.At(1, 0);
        hMat.At(2, 2) = h789.At(2, 0);

        Mat<double> hFull = Mul(Mul(new Mat<double>(tb, 3, 3), hMat), new Mat<double>(tai, 3, 3));
        Span<double> hs = hFull.GetSpan();
        double h22Inv = 1 / hs[8];
        for (int i = 0; i < 9; i++)
        {
            h[i] = hs[i] * h22Inv;
        }
    }
}
