// This implementation is ported from the OpenCV virtual visual servoing pose refinement.
//
// Original source:
//   modules/geometry/src/solvepnp.cpp (solvePnPRefine with SOLVEPNP_REFINE_VVS,
//     computeInteractionMatrixAndResiduals, exponentialMapToSE3Inv)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>OpenCV <c>solvePnPRefineVVS</c>. The matrix products use <c>gemm</c> as the OpenCV <c>MatExpr</c> evaluation.</summary>
internal static class PnpRefineVvs
{
    public static (Point3d Rvec, Point3d Tvec) Run(
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> imagePoints,
        Mat<double> cameraMatrix,
        ReadOnlySpan<double> distCoeffs,
        Point3d rvec,
        Point3d tvec0,
        TermCriteria criteria,
        double vvsLambda)
    {
        int n = objectPoints.Length;
        Point2d[] norm = Undistortion.UndistortPoints(imagePoints, cameraMatrix, distCoeffs);
        var sd = new Mat<double>(1, 2 * n);
        for (int i = 0; i < n; i++)
        {
            sd.At(2 * i, 0) = norm[i].X;
            sd.At(2 * i + 1, 0) = norm[i].Y;
        }

        var l = new Mat<double>(6, 2 * n);
        var s = new Mat<double>(1, 2 * n);
        var err = new Mat<double>(1, 2 * n);
        double residuals = 0;
        Mat<double> r = new(CameraRodrigues.VectorToMatrix(rvec.X, rvec.Y, rvec.Z, null), 3, 3);
        var tvec = new Mat<double>(1, 3);
        tvec.At(0, 0) = tvec0.X;
        tvec.At(1, 0) = tvec0.Y;
        tvec.At(2, 0) = tvec0.Z;
        var pt = new Mat<double>(3, 1);

        for (int iter = 0; iter < criteria.MaxCount; iter++)
        {
            for (int i = 0; i < n; i++)
            {
                pt.At(0, 0) = objectPoints[i].X;
                pt.At(0, 1) = objectPoints[i].Y;
                pt.At(0, 2) = objectPoints[i].Z;
                Mat<double> pc = LinearAlgebra.Gemm(r, pt, 1, tvec, 1, GemmFlags.BTranspose);
                double zi = pc.At(2, 0);
                double xi = pc.At(0, 0) / zi;
                double yi = pc.At(1, 0) / zi;
                s.At(2 * i, 0) = xi;
                s.At(2 * i + 1, 0) = yi;

                Span<double> l0 = l.GetRowSpan(2 * i);
                l0[0] = -1 / zi;
                l0[1] = 0;
                l0[2] = xi / zi;
                l0[3] = xi * yi;
                l0[4] = -(1 + xi * xi);
                l0[5] = yi;

                Span<double> l1 = l.GetRowSpan(2 * i + 1);
                l1[0] = 0;
                l1[1] = -1 / zi;
                l1[2] = yi / zi;
                l1[3] = 1 + yi * yi;
                l1[4] = -xi * yi;
                l1[5] = -xi;
            }

            for (int k = 0; k < 2 * n; k++)
            {
                err.At(k, 0) = s.At(k, 0) - sd.At(k, 0);
            }

            LinearAlgebra.Invert(l, out Mat<double> lp, DecompTypes.Svd);
            Mat<double> dq = LinearAlgebra.Gemm(lp, err, -vvsLambda, null, 0);

            (Mat<double> r1, Mat<double> t1) = ExponentialMapToSe3Inv(dq);
            r = LinearAlgebra.Gemm(r1, r, 1, null, 0);
            tvec = LinearAlgebra.Gemm(r1, tvec, 1, t1, 1);

            double residuals1 = residuals;
            residuals = LinearAlgebra.Gemm(err, err, 1, null, 0, GemmFlags.ATranspose).At(0, 0);
            if (Math.Abs(residuals - residuals1) < criteria.Epsilon)
            {
                break;
            }
        }

        (double x, double y, double z) = CameraRodrigues.MatrixToVector(r.ToArray(), null);
        return (new Point3d(x, y, z), new Point3d(tvec.At(0, 0), tvec.At(1, 0), tvec.At(2, 0)));
    }

    private static (Mat<double> R1, Mat<double> T1) ExponentialMapToSe3Inv(Mat<double> twist)
    {
        double vx = twist.At(0, 0);
        double vy = twist.At(1, 0);
        double vz = twist.At(2, 0);
        double wx = twist.At(3, 0);
        double wy = twist.At(4, 0);
        double wz = twist.At(5, 0);

        double[] rm = CameraRodrigues.VectorToMatrix(wx, wy, wz, null);

        double theta = Math.Sqrt(wx * wx + wy * wy + wz * wz);
        double sinc = Math.Abs(theta) < 1e-8 ? 1 : Math.Sin(theta) / theta;
        double mcosc = Math.Abs(theta) < 1e-8 ? 0.5 : (1 - Math.Cos(theta)) / (theta * theta);
        double msinc = Math.Abs(theta) < 1e-8 ? (1 / 6.0) : (1 - sinc) / (theta * theta);

        var dt = new Mat<double>(1, 3);
        dt.At(0, 0) = vx * (sinc + wx * wx * msinc) + vy * (wx * wy * msinc - wz * mcosc) + vz * (wx * wz * msinc + wy * mcosc);
        dt.At(1, 0) = vx * (wx * wy * msinc + wz * mcosc) + vy * (sinc + wy * wy * msinc) + vz * (wy * wz * msinc - wx * mcosc);
        dt.At(2, 0) = vx * (wx * wz * msinc - wy * mcosc) + vy * (wy * wz * msinc + wx * mcosc) + vz * (sinc + wz * wz * msinc);

        var r1 = new Mat<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                r1.At(i, j) = rm[j * 3 + i];
            }
        }

        Mat<double> t1 = LinearAlgebra.Gemm(r1, dt, -1, null, 0);
        return (r1, t1);
    }
}
