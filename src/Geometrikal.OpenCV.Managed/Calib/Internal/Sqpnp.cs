// This implementation is ported from the OpenCV SQPnP pose solver.
//
// Original sources:
//   modules/geometry/src/sqpnp.cpp (sqpnp::PoseSolver)
//   modules/geometry/src/sqpnp.hpp
//   modules/geometry/src/solvepnp.cpp (solvePnPGeneric with SOLVEPNP_SQPNP)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).
// SQPnP is based on the BSD 3-Clause implementation of George Terzakis (Copyright (c) 2020).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// SQPnP: globally optimal PnP by sequential quadratic programming on the rotation matrices
/// (G. Terzakis, M. Lourakis, ECCV 2020).
/// </summary>
/// <remarks>
/// The port follows the OpenCV build without the Eigen library: the null space of Ω comes from
/// <c>cv::SVD</c> (Jacobi method). Matrices are row-major <c>double</c> arrays. All expressions keep
/// the C++ evaluation order.
/// </remarks>
internal static class Sqpnp
{
    /// <summary>OpenCV <c>sqpnp::PoseSolver::solve</c> with the default parameters.</summary>
    /// <param name="objectPoints">Three or more 3D points in object coordinates.</param>
    /// <param name="normalizedPoints">Undistorted normalized image points, one for each object point.</param>
    /// <param name="rvecs">Receives the rotation vectors of the solutions.</param>
    /// <param name="tvecs">Receives the translation vectors of the solutions.</param>
    /// <exception cref="ArgumentException">The point counts are not equal, or there are fewer than 3 points.</exception>
    /// <exception cref="InvalidOperationException">The image points or Ω are degenerate (an OpenCV assertion fails).</exception>
    public static void Solve(ReadOnlySpan<Point3d> objectPoints, ReadOnlySpan<Point2d> normalizedPoints, List<Point3d> rvecs, List<Point3d> tvecs)
    {
        ArgumentNullException.ThrowIfNull(rvecs);
        ArgumentNullException.ThrowIfNull(tvecs);
        if (objectPoints.Length != normalizedPoints.Length)
        {
            throw new ArgumentException("The object and image point counts must be equal.", nameof(normalizedPoints));
        }

        if (objectPoints.Length < 3)
        {
            throw new ArgumentException("SQPnP needs at least 3 points.", nameof(objectPoints));
        }

        var solver = new PoseSolver();
        solver.ComputeOmega(objectPoints, normalizedPoints);
        solver.SolveInternal(objectPoints);

        for (int i = 0; i < solver.NumSolutions; i++)
        {
            Solution solution = solver.Solutions[i];
            (double x, double y, double z) = CameraRodrigues.MatrixToVector(solution.RHat, null);
            rvecs.Add(new Point3d(x, y, z));
            tvecs.Add(new Point3d(solution.T[0], solution.T[1], solution.T[2]));
        }
    }

    /// <summary>
    /// <c>Matx::operator*</c>: row-major (m × l)·(l × n), each sum starts at 0 and adds the
    /// products in index order.
    /// </summary>
    private static void MatMul(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> dst, int m, int l, int n)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int k = 0; k < l; k++)
                {
                    s += a[(i * l) + k] * b[(k * n) + j];
                }

                dst[(i * n) + j] = s;
            }
        }
    }

    /// <summary><c>Matx::dot</c> and <c>Matx::ddot</c>.</summary>
    private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++)
        {
            s += a[i] * b[i];
        }

        return s;
    }

    /// <summary><c>cv::norm(Matx, NORM_L2SQR)</c>.</summary>
    private static double NormL2Sqr(ReadOnlySpan<double> a)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double v = a[i];
            s += v * v;
        }

        return s;
    }

    /// <summary>One candidate pose: the rotation matrix (row-major), the translation and the squared error.</summary>
    private sealed class Solution
    {
        public readonly double[] RHat = new double[9];
        public readonly double[] T = new double[3];
        public double SqError;
    }

    /// <summary>State of <c>sqpnp::PoseSolver</c> for one call of <see cref="Solve"/>.</summary>
    private sealed class PoseSolver
    {
        private const double RankTolerance = 1e-7;
        private const double SqpSquaredTolerance = 1e-10;
        private const double SqpDetThreshold = 1.001;
        private const double OrthogonalitySquaredErrorThreshold = 1e-8;
        private const double EqualVectorsSquaredDiff = 1e-10;
        private const double EqualSquaredErrorsDiff = 1e-6;
        private const double PointVarianceThreshold = 1e-5;
        private const int SqpMaxIteration = 15;

        // std::numeric_limits<double>::min().
        private const double DblMinNormal = 2.2250738585072014E-308;

        private static readonly double Sqrt3 = Math.Sqrt(3.0);

        private readonly double[] _omega = new double[81];
        private readonly double[] _s = new double[9];
        private readonly double[] _u = new double[81];
        private readonly double[] _p = new double[27];
        private readonly double[] _pointMean = new double[3];
        private int _numNullVectors = -1;

        // Scratch matrices of SolveSqpSystem and ComputeRowAndNullspace.
        private readonly double[] _h = new double[54];
        private readonly double[] _n = new double[27];
        private readonly double[] _k = new double[36];
        private readonly double[] _pn = new double[81];
        private readonly double[] _ntOmega = new double[27];

        public Solution[] Solutions { get; } = new Solution[18];

        public int NumSolutions { get; private set; }

        private ref double Omega(int r, int c) => ref _omega[(r * 9) + c];

        public void ComputeOmega(ReadOnlySpan<Point3d> objectPoints, ReadOnlySpan<Point2d> imagePoints)
        {
            Span<double> qaSum = stackalloc double[27];
            qaSum.Clear();

            double sumImgX = 0, sumImgY = 0;
            double sumObjX = 0, sumObjY = 0, sumObjZ = 0;
            double sqNormSum = 0;

            int n = objectPoints.Length;

            for (int i = 0; i < n; i++)
            {
                Point2d imgPt = imagePoints[i];
                Point3d objPt = objectPoints[i];

                sumImgX += imgPt.X;
                sumImgY += imgPt.Y;
                sumObjX += objPt.X;
                sumObjY += objPt.Y;
                sumObjZ += objPt.Z;

                double x = imgPt.X, y = imgPt.Y;
                double bigX = objPt.X, bigY = objPt.Y, bigZ = objPt.Z;
                double sqNorm = x * x + y * y;
                sqNormSum += sqNorm;

                double x2 = bigX * bigX,
                    xy = bigX * bigY,
                    xz = bigX * bigZ,
                    y2 = bigY * bigY,
                    yz = bigY * bigZ,
                    z2 = bigZ * bigZ;

                Omega(0, 0) += x2;
                Omega(0, 1) += xy;
                Omega(0, 2) += xz;
                Omega(1, 1) += y2;
                Omega(1, 2) += yz;
                Omega(2, 2) += z2;

                Omega(0, 6) -= x * x2; Omega(0, 7) -= x * xy; Omega(0, 8) -= x * xz;
                Omega(1, 7) -= x * y2; Omega(1, 8) -= x * yz;
                Omega(2, 8) -= x * z2;

                Omega(3, 6) -= y * x2; Omega(3, 7) -= y * xy; Omega(3, 8) -= y * xz;
                Omega(4, 7) -= y * y2; Omega(4, 8) -= y * yz;
                Omega(5, 8) -= y * z2;

                Omega(6, 6) += sqNorm * x2; Omega(6, 7) += sqNorm * xy; Omega(6, 8) += sqNorm * xz;
                Omega(7, 7) += sqNorm * y2; Omega(7, 8) += sqNorm * yz;
                Omega(8, 8) += sqNorm * z2;

                qaSum[0] += bigX; qaSum[1] += bigY; qaSum[2] += bigZ;

                qaSum[6] -= x * bigX; qaSum[7] -= x * bigY; qaSum[8] -= x * bigZ;
                qaSum[9 + 6] -= y * bigX; qaSum[9 + 7] -= y * bigY; qaSum[9 + 8] -= y * bigZ;

                qaSum[18 + 6] += sqNorm * bigX; qaSum[18 + 7] += sqNorm * bigY; qaSum[18 + 8] += sqNorm * bigZ;
            }

            qaSum[9 + 3] = qaSum[0]; qaSum[9 + 4] = qaSum[1]; qaSum[9 + 5] = qaSum[2];
            qaSum[18 + 0] = qaSum[6]; qaSum[18 + 1] = qaSum[7]; qaSum[18 + 2] = qaSum[8];
            qaSum[18 + 3] = qaSum[9 + 6]; qaSum[18 + 4] = qaSum[9 + 7]; qaSum[18 + 5] = qaSum[9 + 8];

            Omega(1, 6) = Omega(0, 7); Omega(2, 6) = Omega(0, 8); Omega(2, 7) = Omega(1, 8);
            Omega(4, 6) = Omega(3, 7); Omega(5, 6) = Omega(3, 8); Omega(5, 7) = Omega(4, 8);
            Omega(7, 6) = Omega(6, 7); Omega(8, 6) = Omega(6, 8); Omega(8, 7) = Omega(7, 8);

            Omega(3, 3) = Omega(0, 0); Omega(3, 4) = Omega(0, 1); Omega(3, 5) = Omega(0, 2);
            Omega(4, 4) = Omega(1, 1); Omega(4, 5) = Omega(1, 2);
            Omega(5, 5) = Omega(2, 2);

            for (int r = 1; r < 9; r++)
            {
                for (int c = 0; c < r; c++)
                {
                    if (r >= 7 && c >= 6)
                    {
                        continue;
                    }

                    Omega(r, c) = Omega(c, r);
                }
            }

            Span<double> q = stackalloc double[9];
            q[0] = n; q[1] = 0; q[2] = -sumImgX;
            q[3] = 0; q[4] = n; q[5] = -sumImgY;
            q[6] = -sumImgX; q[7] = -sumImgY; q[8] = sqNormSum;

            double invN = 1.0 / n;
            double detQ = n * (n * sqNormSum - sumImgY * sumImgY - sumImgX * sumImgX);
            double pointCoordinateVariance = detQ * invN * invN * invN;

            if (!(pointCoordinateVariance >= PointVarianceThreshold))
            {
                throw new InvalidOperationException("SQPnP: the variance of the image points is too small.");
            }

            Span<double> qInv = stackalloc double[9];
            qInv.Clear();
            if (!InvertSpd3x3(q, qInv))
            {
                AnalyticalInverse3x3Symm(q, qInv, 1e-8);
            }

            Span<double> negQInv = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                negQInv[i] = qInv[i] * -1;
            }

            MatMul(negQInv, qaSum, _p, 3, 3, 9);

            Span<double> qaSumT = stackalloc double[27];
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 9; c++)
                {
                    qaSumT[(c * 3) + r] = qaSum[(r * 9) + c];
                }
            }

            Span<double> prod = stackalloc double[81];
            MatMul(qaSumT, _p, prod, 9, 3, 9);
            for (int i = 0; i < 81; i++)
            {
                _omega[i] += prod[i];
            }

            var svd = new Svd<double>(new Mat<double>((double[])_omega.Clone(), 9, 9), SvdFlags.FullUV);
            svd.W.GetReadOnlySpan().CopyTo(_s);
            for (int r = 0; r < 9; r++)
            {
                ReadOnlySpan<double> vtRow = svd.Vt.GetReadOnlyRowSpan(r);
                for (int c = 0; c < 9; c++)
                {
                    _u[(c * 9) + r] = vtRow[c];
                }
            }

            if (!(_s[0] >= 1e-7))
            {
                throw new InvalidOperationException("SQPnP: the largest singular value of Omega is too small.");
            }

            while (_s[7 - _numNullVectors] < RankTolerance)
            {
                _numNullVectors++;
            }

            if (!(++_numNullVectors <= 6))
            {
                throw new InvalidOperationException("SQPnP: Omega has more than 6 null vectors.");
            }

            _pointMean[0] = sumObjX / n;
            _pointMean[1] = sumObjY / n;
            _pointMean[2] = sumObjZ / n;
        }

        public void SolveInternal(ReadOnlySpan<Point3d> objectPoints)
        {
            double minSqErr = double.MaxValue;
            int numEigenPoints = _numNullVectors > 0 ? _numNullVectors : 1;
            Span<double> e = stackalloc double[9];
            Span<double> negE = stackalloc double[9];
            Span<double> r = stackalloc double[9];

            for (int i = 9 - numEigenPoints; i < 9; i++)
            {
                for (int k = 0; k < 9; k++)
                {
                    e[k] = _u[(k * 9) + i] * Sqrt3;
                }

                double orthogonalitySqErr = OrthogonalityError(e);

                if (orthogonalitySqErr < OrthogonalitySquaredErrorThreshold)
                {
                    var solution = new Solution();
                    double det = Det3x3(e);
                    for (int k = 0; k < 9; k++)
                    {
                        solution.RHat[k] = e[k] * det;
                    }

                    MatMul(_p, solution.RHat, solution.T, 3, 9, 1);
                    CheckSolution(solution, objectPoints, ref minSqErr);
                }
                else
                {
                    SolveFromStart(e, negE, r, objectPoints, ref minSqErr);
                }
            }

            int index, c = 1;
            while ((index = 9 - numEigenPoints - c) > 0 && minSqErr > 3 * _s[index])
            {
                for (int k = 0; k < 9; k++)
                {
                    e[k] = _u[(k * 9) + index];
                }

                SolveFromStart(e, negE, r, objectPoints, ref minSqErr);
                c++;
            }
        }

        /// <summary>Runs SQP from the nearest rotations of <c>e</c> and <c>-e</c>, and checks both solutions.</summary>
        private void SolveFromStart(ReadOnlySpan<double> e, Span<double> negE, Span<double> r, ReadOnlySpan<Point3d> objectPoints, ref double minSqErr)
        {
            NearestRotationMatrixFoam(e, r);
            Solution first = RunSqp(r);
            MatMul(_p, first.RHat, first.T, 3, 9, 1);
            CheckSolution(first, objectPoints, ref minSqErr);

            for (int k = 0; k < 9; k++)
            {
                negE[k] = e[k] * -1;
            }

            NearestRotationMatrixFoam(negE, r);
            Solution second = RunSqp(r);
            MatMul(_p, second.RHat, second.T, 3, 9, 1);
            CheckSolution(second, objectPoints, ref minSqErr);
        }

        private Solution RunSqp(ReadOnlySpan<double> r0)
        {
            Span<double> r = stackalloc double[9];
            r0.CopyTo(r);

            double deltaSquaredNorm = double.MaxValue;
            Span<double> delta = stackalloc double[9];

            int step = 0;
            while (deltaSquaredNorm > SqpSquaredTolerance && step++ < SqpMaxIteration)
            {
                SolveSqpSystem(r, delta);
                for (int k = 0; k < 9; k++)
                {
                    r[k] += delta[k];
                }

                deltaSquaredNorm = NormL2Sqr(delta);
            }

            var solution = new Solution();

            double detR = Det3x3(r);
            if (detR < 0)
            {
                for (int k = 0; k < 9; k++)
                {
                    r[k] *= -1;
                }

                detR = -detR;
            }

            if (detR > SqpDetThreshold)
            {
                NearestRotationMatrixFoam(r, solution.RHat);
            }
            else
            {
                r.CopyTo(solution.RHat);
            }

            return solution;
        }

        private void SolveSqpSystem(ReadOnlySpan<double> r, Span<double> delta)
        {
            double sqnormR1 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2],
                sqnormR2 = r[3] * r[3] + r[4] * r[4] + r[5] * r[5],
                sqnormR3 = r[6] * r[6] + r[7] * r[7] + r[8] * r[8];
            double dotR1R2 = r[0] * r[3] + r[1] * r[4] + r[2] * r[5],
                dotR1R3 = r[0] * r[6] + r[1] * r[7] + r[2] * r[8],
                dotR2R3 = r[3] * r[6] + r[4] * r[7] + r[5] * r[8];

            double[] nMat = _n;
            double[] h = _h;
            double[] jh = _k;

            ComputeRowAndNullspace(r, h, nMat, jh, 0.1);

            Span<double> g = stackalloc double[6];
            g[0] = 1 - sqnormR1; g[1] = 1 - sqnormR2; g[2] = 1 - sqnormR3; g[3] = -dotR1R2; g[4] = -dotR2R3; g[5] = -dotR1R3;

            Span<double> x = stackalloc double[6];
            x[0] = g[0] / jh[0];
            x[1] = g[1] / jh[7];
            x[2] = g[2] / jh[14];
            x[3] = (g[3] - jh[18] * x[0] - jh[19] * x[1]) / jh[21];
            x[4] = (g[4] - jh[25] * x[1] - jh[26] * x[2] - jh[27] * x[3]) / jh[28];
            x[5] = (g[5] - jh[30] * x[0] - jh[32] * x[2] - jh[33] * x[3] - jh[34] * x[4]) / jh[35];

            MatMul(h, x, delta, 9, 6, 1);

            Span<double> nT = stackalloc double[27];
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    nT[(j * 9) + i] = nMat[(i * 3) + j];
                }
            }

            double[] ntOmega = _ntOmega;
            MatMul(nT, _omega, ntOmega, 3, 9, 9);
            Span<double> w = stackalloc double[9];
            Span<double> wInv = stackalloc double[9];
            wInv.Clear();
            MatMul(ntOmega, nMat, w, 3, 9, 3);

            AnalyticalInverse3x3Symm(w, wInv, 1e-8);

            Span<double> negWInv = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                negWInv[i] = wInv[i] * -1;
            }

            Span<double> a = stackalloc double[27];
            MatMul(negWInv, ntOmega, a, 3, 3, 9);
            Span<double> deltaPlusR = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                deltaPlusR[i] = delta[i] + r[i];
            }

            Span<double> y = stackalloc double[3];
            MatMul(a, deltaPlusR, y, 3, 9, 1);

            Span<double> ny = stackalloc double[9];
            MatMul(nMat, y, ny, 9, 3, 1);
            for (int i = 0; i < 9; i++)
            {
                delta[i] += ny[i];
            }
        }

        /// <summary>
        /// Inverse of a symmetric positive definite 3 × 3 matrix by the sqrt-free Cholesky
        /// factorization <c>A = L·D·Lᵀ</c>. Reads only the lower triangle of <paramref name="a"/>.
        /// </summary>
        /// <returns>False when a pivot is not larger than 1e-10; <paramref name="a1"/> is then not changed.</returns>
        private static bool InvertSpd3x3(ReadOnlySpan<double> a, Span<double> a1)
        {
            Span<double> l = stackalloc double[9];
            Span<double> d = stackalloc double[3];
            Span<double> v = stackalloc double[2];
            Span<double> x = stackalloc double[3];

            v[0] = d[0] = a[0];
            if (v[0] <= 1E-10)
            {
                return false;
            }

            v[1] = 1.0 / v[0];
            l[3] = a[3] * v[1];
            l[6] = a[6] * v[1];

            v[0] = l[3] * d[0];
            v[1] = d[1] = a[4] - l[3] * v[0];
            if (v[1] <= 1E-10)
            {
                return false;
            }

            l[7] = (a[7] - l[6] * v[0]) / v[1];

            v[0] = l[6] * d[0];
            v[1] = l[7] * d[1];
            d[2] = a[8] - l[6] * v[0] - l[7] * v[1];
            if (d[2] <= 1E-10)
            {
                return false;
            }

            d[0] = 1.0 / d[0];
            d[1] = 1.0 / d[1];
            d[2] = 1.0 / d[2];

            x[1] = -l[3];
            x[2] = -l[6] + l[7] * l[3];

            a1[2] = x[2] = x[2] * d[2];
            a1[1] = x[1] = x[1] * d[1] - l[7] * x[2];
            a1[0] = d[0] - l[3] * x[1] - l[6] * x[2];

            x[2] = -l[7];

            a1[5] = x[2] = x[2] * d[2];
            a1[4] = x[1] = d[1] - l[7] * x[2];
            a1[3] = -l[3] * x[1] - l[6] * x[2];

            a1[8] = x[2] = d[2];
            a1[7] = x[1] = -l[7] * x[2];
            a1[6] = -l[3] * x[1] - l[6] * x[2];

            return true;
        }

        /// <summary>
        /// Inverse of a symmetric 3 × 3 matrix from its lower triangle. When |det| is below
        /// <paramref name="threshold"/>, <paramref name="qInv"/> receives the SVD pseudo-inverse of the full matrix.
        /// </summary>
        private static bool AnalyticalInverse3x3Symm(ReadOnlySpan<double> q, Span<double> qInv, double threshold)
        {
            double a = q[0],
                b = q[3], d = q[4],
                c = q[6], e = q[7], f = q[8];

            double t2, t4, t7, t9, t12;
            t2 = e * e;
            t4 = a * d;
            t7 = b * b;
            t9 = b * c;
            t12 = c * c;
            double det = -t4 * f + a * t2 + t7 * f - 2.0 * t9 * e + t12 * d;

            if (Math.Abs(det) < threshold)
            {
                LinearAlgebra.Invert(new Mat<double>(q.ToArray(), 3, 3), out Mat<double> inverse, DecompTypes.Svd);
                inverse.GetReadOnlySpan().CopyTo(qInv);
                return false;
            }

            double t15, t20, t24, t30;
            t15 = 1.0 / det;
            t20 = (-b * f + c * e) * t15;
            t24 = (b * e - c * d) * t15;
            t30 = (a * e - t9) * t15;
            qInv[0] = (-d * f + t2) * t15;
            qInv[1] = qInv[3] = -t20;
            qInv[2] = qInv[6] = -t24;
            qInv[4] = -(a * f - t12) * t15;
            qInv[5] = qInv[7] = t30;
            qInv[8] = -(t4 - t7) * t15;

            return true;
        }

        /// <summary>
        /// Computes the 6D row space <paramref name="h"/> (9 × 6), the 3D null space <paramref name="nMat"/>
        /// (9 × 3) and the lower triangle of the constraint Jacobian <paramref name="k"/> (6 × 6) at <paramref name="r"/>.
        /// </summary>
        private void ComputeRowAndNullspace(ReadOnlySpan<double> r, double[] h, double[] nMat, double[] k, double normThreshold)
        {
            Array.Clear(h);

            double normR1 = Math.Sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            double invNormR1 = normR1 > 1e-5 ? 1.0 / normR1 : 0.0;
            h[H(0, 0)] = r[0] * invNormR1;
            h[H(1, 0)] = r[1] * invNormR1;
            h[H(2, 0)] = r[2] * invNormR1;
            k[K(0, 0)] = 2 * normR1;

            double normR2 = Math.Sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5]);
            double invNormR2 = 1.0 / normR2;
            h[H(3, 1)] = r[3] * invNormR2;
            h[H(4, 1)] = r[4] * invNormR2;
            h[H(5, 1)] = r[5] * invNormR2;
            k[K(1, 0)] = 0;
            k[K(1, 1)] = 2 * normR2;

            double normR3 = Math.Sqrt(r[6] * r[6] + r[7] * r[7] + r[8] * r[8]);
            double invNormR3 = 1.0 / normR3;
            h[H(6, 2)] = r[6] * invNormR3;
            h[H(7, 2)] = r[7] * invNormR3;
            h[H(8, 2)] = r[8] * invNormR3;
            k[K(2, 0)] = k[K(2, 1)] = 0;
            k[K(2, 2)] = 2 * normR3;

            // q4
            double dotJ4Q1 = r[3] * h[H(0, 0)] + r[4] * h[H(1, 0)] + r[5] * h[H(2, 0)],
                dotJ4Q2 = r[0] * h[H(3, 1)] + r[1] * h[H(4, 1)] + r[2] * h[H(5, 1)];

            h[H(0, 3)] = r[3] - dotJ4Q1 * h[H(0, 0)];
            h[H(1, 3)] = r[4] - dotJ4Q1 * h[H(1, 0)];
            h[H(2, 3)] = r[5] - dotJ4Q1 * h[H(2, 0)];
            h[H(3, 3)] = r[0] - dotJ4Q2 * h[H(3, 1)];
            h[H(4, 3)] = r[1] - dotJ4Q2 * h[H(4, 1)];
            h[H(5, 3)] = r[2] - dotJ4Q2 * h[H(5, 1)];
            double invNormJ4 = 1.0 / Math.Sqrt(h[H(0, 3)] * h[H(0, 3)] + h[H(1, 3)] * h[H(1, 3)] + h[H(2, 3)] * h[H(2, 3)] +
                h[H(3, 3)] * h[H(3, 3)] + h[H(4, 3)] * h[H(4, 3)] + h[H(5, 3)] * h[H(5, 3)]);

            h[H(0, 3)] *= invNormJ4;
            h[H(1, 3)] *= invNormJ4;
            h[H(2, 3)] *= invNormJ4;
            h[H(3, 3)] *= invNormJ4;
            h[H(4, 3)] *= invNormJ4;
            h[H(5, 3)] *= invNormJ4;

            k[K(3, 0)] = r[3] * h[H(0, 0)] + r[4] * h[H(1, 0)] + r[5] * h[H(2, 0)];
            k[K(3, 1)] = r[0] * h[H(3, 1)] + r[1] * h[H(4, 1)] + r[2] * h[H(5, 1)];
            k[K(3, 2)] = 0;
            k[K(3, 3)] = r[3] * h[H(0, 3)] + r[4] * h[H(1, 3)] + r[5] * h[H(2, 3)] + r[0] * h[H(3, 3)] + r[1] * h[H(4, 3)] + r[2] * h[H(5, 3)];

            // q5
            double dotJ5Q2 = r[6] * h[H(3, 1)] + r[7] * h[H(4, 1)] + r[8] * h[H(5, 1)];
            double dotJ5Q3 = r[3] * h[H(6, 2)] + r[4] * h[H(7, 2)] + r[5] * h[H(8, 2)];
            double dotJ5Q4 = r[6] * h[H(3, 3)] + r[7] * h[H(4, 3)] + r[8] * h[H(5, 3)];

            h[H(0, 4)] = -dotJ5Q4 * h[H(0, 3)];
            h[H(1, 4)] = -dotJ5Q4 * h[H(1, 3)];
            h[H(2, 4)] = -dotJ5Q4 * h[H(2, 3)];
            h[H(3, 4)] = r[6] - dotJ5Q2 * h[H(3, 1)] - dotJ5Q4 * h[H(3, 3)];
            h[H(4, 4)] = r[7] - dotJ5Q2 * h[H(4, 1)] - dotJ5Q4 * h[H(4, 3)];
            h[H(5, 4)] = r[8] - dotJ5Q2 * h[H(5, 1)] - dotJ5Q4 * h[H(5, 3)];
            h[H(6, 4)] = r[3] - dotJ5Q3 * h[H(6, 2)]; h[H(7, 4)] = r[4] - dotJ5Q3 * h[H(7, 2)]; h[H(8, 4)] = r[5] - dotJ5Q3 * h[H(8, 2)];

            NormalizeColumn(h, 4);

            k[K(4, 0)] = 0;
            k[K(4, 1)] = r[6] * h[H(3, 1)] + r[7] * h[H(4, 1)] + r[8] * h[H(5, 1)];
            k[K(4, 2)] = r[3] * h[H(6, 2)] + r[4] * h[H(7, 2)] + r[5] * h[H(8, 2)];
            k[K(4, 3)] = r[6] * h[H(3, 3)] + r[7] * h[H(4, 3)] + r[8] * h[H(5, 3)];
            k[K(4, 4)] = r[6] * h[H(3, 4)] + r[7] * h[H(4, 4)] + r[8] * h[H(5, 4)] + r[3] * h[H(6, 4)] + r[4] * h[H(7, 4)] + r[5] * h[H(8, 4)];

            // q6
            double dotJ6Q1 = r[6] * h[H(0, 0)] + r[7] * h[H(1, 0)] + r[8] * h[H(2, 0)];
            double dotJ6Q3 = r[0] * h[H(6, 2)] + r[1] * h[H(7, 2)] + r[2] * h[H(8, 2)];
            double dotJ6Q4 = r[6] * h[H(0, 3)] + r[7] * h[H(1, 3)] + r[8] * h[H(2, 3)];
            double dotJ6Q5 = r[0] * h[H(6, 4)] + r[1] * h[H(7, 4)] + r[2] * h[H(8, 4)] + r[6] * h[H(0, 4)] + r[7] * h[H(1, 4)] + r[8] * h[H(2, 4)];

            h[H(0, 5)] = r[6] - dotJ6Q1 * h[H(0, 0)] - dotJ6Q4 * h[H(0, 3)] - dotJ6Q5 * h[H(0, 4)];
            h[H(1, 5)] = r[7] - dotJ6Q1 * h[H(1, 0)] - dotJ6Q4 * h[H(1, 3)] - dotJ6Q5 * h[H(1, 4)];
            h[H(2, 5)] = r[8] - dotJ6Q1 * h[H(2, 0)] - dotJ6Q4 * h[H(2, 3)] - dotJ6Q5 * h[H(2, 4)];

            h[H(3, 5)] = -dotJ6Q5 * h[H(3, 4)] - dotJ6Q4 * h[H(3, 3)];
            h[H(4, 5)] = -dotJ6Q5 * h[H(4, 4)] - dotJ6Q4 * h[H(4, 3)];
            h[H(5, 5)] = -dotJ6Q5 * h[H(5, 4)] - dotJ6Q4 * h[H(5, 3)];

            h[H(6, 5)] = r[0] - dotJ6Q3 * h[H(6, 2)] - dotJ6Q5 * h[H(6, 4)];
            h[H(7, 5)] = r[1] - dotJ6Q3 * h[H(7, 2)] - dotJ6Q5 * h[H(7, 4)];
            h[H(8, 5)] = r[2] - dotJ6Q3 * h[H(8, 2)] - dotJ6Q5 * h[H(8, 4)];

            NormalizeColumn(h, 5);

            k[K(5, 0)] = r[6] * h[H(0, 0)] + r[7] * h[H(1, 0)] + r[8] * h[H(2, 0)];
            k[K(5, 1)] = 0; k[K(5, 2)] = r[0] * h[H(6, 2)] + r[1] * h[H(7, 2)] + r[2] * h[H(8, 2)];
            k[K(5, 3)] = r[6] * h[H(0, 3)] + r[7] * h[H(1, 3)] + r[8] * h[H(2, 3)];
            k[K(5, 4)] = r[6] * h[H(0, 4)] + r[7] * h[H(1, 4)] + r[8] * h[H(2, 4)] + r[0] * h[H(6, 4)] + r[1] * h[H(7, 4)] + r[2] * h[H(8, 4)];
            k[K(5, 5)] = r[6] * h[H(0, 5)] + r[7] * h[H(1, 5)] + r[8] * h[H(2, 5)] + r[0] * h[H(6, 5)] + r[1] * h[H(7, 5)] + r[2] * h[H(8, 5)];

            // Pn = I - H·Hᵀ, the projector onto the null space of the Jacobian.
            double[] pn = _pn;
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    double s = 0;
                    for (int m = 0; m < 6; m++)
                    {
                        s += h[H(i, m)] * h[H(j, m)];
                    }

                    pn[(i * 9) + j] = (i == j ? 1.0 : 0.0) - s;
                }
            }

            int index1 = 0,
                index2 = 0,
                index3 = 0;
            double maxNorm1 = DblMinNormal;
            double minDot12 = double.MaxValue;
            double minDot1323 = double.MaxValue;

            Span<double> colNorms = stackalloc double[9];
            Span<double> col = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                GetColumn(pn, i, col);
                colNorms[i] = Math.Sqrt(NormL2Sqr(col));
                if (colNorms[i] >= normThreshold)
                {
                    if (maxNorm1 < colNorms[i])
                    {
                        maxNorm1 = colNorms[i];
                        index1 = i;
                    }
                }
            }

            Span<double> v1 = stackalloc double[9];
            GetColumn(pn, index1, v1);
            double invMaxNorm1 = 1.0 / maxNorm1;
            Span<double> n0 = stackalloc double[9];
            for (int i = 0; i < 9; ++i)
            {
                n0[i] = nMat[(i * 3) + 0] = v1[i] * invMaxNorm1;
            }

            colNorms[index1] = -1.0;

            for (int i = 0; i < 9; i++)
            {
                if (colNorms[i] >= normThreshold)
                {
                    GetColumn(pn, i, col);
                    double cosV1XCol = Math.Abs(Dot(col, v1) / colNorms[i]);

                    if (cosV1XCol <= minDot12)
                    {
                        index2 = i;
                        minDot12 = cosV1XCol;
                    }
                }
            }

            Span<double> v2 = stackalloc double[9];
            GetColumn(pn, index2, v2);
            Span<double> u2 = stackalloc double[9];
            double dotV2N0 = Dot(v2, n0);
            for (int i = 0; i < 9; i++)
            {
                u2[i] = v2[i] - n0[i] * dotV2N0;
            }

            double invNormU2 = 1.0 / Math.Sqrt(NormL2Sqr(u2));
            Span<double> n1 = stackalloc double[9];
            for (int i = 0; i < 9; ++i)
            {
                n1[i] = nMat[(i * 3) + 1] = u2[i] * invNormU2;
            }

            colNorms[index2] = -1.0;

            for (int i = 0; i < 9; i++)
            {
                if (colNorms[i] >= normThreshold)
                {
                    GetColumn(pn, i, col);
                    double invNorm = 1.0 / colNorms[i];
                    double cosV1XCol = Math.Abs(Dot(col, v1) * invNorm);
                    double cosV2XCol = Math.Abs(Dot(col, v2) * invNorm);

                    if (cosV1XCol + cosV2XCol <= minDot1323)
                    {
                        index3 = i;
                        minDot1323 = cosV1XCol + cosV2XCol;
                    }
                }
            }

            Span<double> v3 = stackalloc double[9];
            GetColumn(pn, index3, v3);
            double dotV3N1 = Dot(v3, n1);
            double dotV3N0 = Dot(v3, n0);
            Span<double> u3 = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                u3[i] = v3[i] - n1[i] * dotV3N1 - n0[i] * dotV3N0;
            }

            double invNormU3 = 1.0 / Math.Sqrt(NormL2Sqr(u3));
            for (int i = 0; i < 9; ++i)
            {
                nMat[(i * 3) + 2] = u3[i] * invNormU3;
            }
        }

        private static int H(int r, int c) => (r * 6) + c;

        private static int K(int r, int c) => (r * 6) + c;

        private static void GetColumn(double[] m9x9, int c, Span<double> dst)
        {
            for (int i = 0; i < 9; i++)
            {
                dst[i] = m9x9[(i * 9) + c];
            }
        }

        /// <summary>Scales column <paramref name="c"/> of the 9 × 6 matrix <paramref name="h"/> by the inverse of its L2 norm.</summary>
        private static void NormalizeColumn(double[] h, int c)
        {
            Span<double> q = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                q[i] = h[H(i, c)];
            }

            double invNorm = 1.0 / Math.Sqrt(NormL2Sqr(q));
            for (int i = 0; i < 9; ++i)
            {
                h[H(i, c)] = q[i] * invNorm;
            }
        }

        /// <summary>Nearest rotation matrix to <paramref name="e"/> by SVD: <c>U·diag(1, 1, det(U)·det(Vᵀ))·Vᵀ</c>.</summary>
        private static void NearestRotationMatrixSvd(ReadOnlySpan<double> e, Span<double> r)
        {
            var svd = new Svd<double>(new Mat<double>(e.ToArray(), 3, 3), SvdFlags.FullUV);
            double detuv = LinearAlgebra.Determinant(svd.U) * LinearAlgebra.Determinant(svd.Vt);
            var diag = new Mat<double>(3, 3);
            diag.At(0, 0) = 1;
            diag.At(1, 1) = 1;
            diag.At(2, 2) = detuv;
            Mat<double> ud = LinearAlgebra.Gemm(svd.U, diag, 1, null, 0);
            Mat<double> r33 = LinearAlgebra.Gemm(ud, svd.Vt, 1, null, 0);
            for (int i = 0; i < 3; i++)
            {
                ReadOnlySpan<double> row = r33.GetReadOnlyRowSpan(i);
                for (int j = 0; j < 3; j++)
                {
                    r[(i * 3) + j] = row[j];
                }
            }
        }

        /// <summary>
        /// Nearest rotation matrix to <paramref name="e"/> by the FOAM method of Markley and Lourakis.
        /// Uses <see cref="NearestRotationMatrixSvd"/> when |det(e)| is below 1e-4.
        /// </summary>
        private static void NearestRotationMatrixFoam(ReadOnlySpan<double> e, Span<double> r)
        {
            int i;
            double l, lprev, detE, eSq, adjESq;
            Span<double> adjE = stackalloc double[9];

            detE = (e[0] * e[4] * e[8] - e[0] * e[5] * e[7] - e[1] * e[3] * e[8]) + (e[2] * e[3] * e[7] + e[1] * e[6] * e[5] - e[2] * e[6] * e[4]);
            if (Math.Abs(detE) < 1E-04)
            {
                NearestRotationMatrixSvd(e, r);
                return;
            }

            adjE[0] = e[4] * e[8] - e[5] * e[7]; adjE[1] = e[2] * e[7] - e[1] * e[8]; adjE[2] = e[1] * e[5] - e[2] * e[4];
            adjE[3] = e[5] * e[6] - e[3] * e[8]; adjE[4] = e[0] * e[8] - e[2] * e[6]; adjE[5] = e[2] * e[3] - e[0] * e[5];
            adjE[6] = e[3] * e[7] - e[4] * e[6]; adjE[7] = e[1] * e[6] - e[0] * e[7]; adjE[8] = e[0] * e[4] - e[1] * e[3];

            eSq = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) + (e[3] * e[3] + e[4] * e[4] + e[5] * e[5]) + (e[6] * e[6] + e[7] * e[7] + e[8] * e[8]);
            adjESq = (adjE[0] * adjE[0] + adjE[1] * adjE[1] + adjE[2] * adjE[2]) + (adjE[3] * adjE[3] + adjE[4] * adjE[4] + adjE[5] * adjE[5]) + (adjE[6] * adjE[6] + adjE[7] * adjE[7] + adjE[8] * adjE[8]);

            // Newton-Raphson for the largest root of the FOAM characteristic polynomial.
            l = 0.5 * (eSq + 3.0);
            if (detE < 0.0)
            {
                l = -l;
            }

            for (i = 15, lprev = 0.0; Math.Abs(l - lprev) > 1E-12 * Math.Abs(lprev) && i > 0; --i)
            {
                double tmpL, p, pp;

                tmpL = l * l - eSq;
                p = tmpL * tmpL - 8.0 * l * detE - 4.0 * adjESq;
                pp = 8.0 * (0.5 * tmpL * l - detE);

                lprev = l;
                l -= p / pp;
            }

            Span<double> tmp = stackalloc double[9];
            Span<double> eEt = stackalloc double[9];
            double denom;
            double a = l * l + eSq;

            eEt[0] = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            eEt[1] = e[0] * e[3] + e[1] * e[4] + e[2] * e[5];
            eEt[2] = e[0] * e[6] + e[1] * e[7] + e[2] * e[8];

            eEt[3] = eEt[1];
            eEt[4] = e[3] * e[3] + e[4] * e[4] + e[5] * e[5];
            eEt[5] = e[3] * e[6] + e[4] * e[7] + e[5] * e[8];

            eEt[6] = eEt[2];
            eEt[7] = eEt[5];
            eEt[8] = e[6] * e[6] + e[7] * e[7] + e[8] * e[8];

            tmp[0] = eEt[0] * e[0] + eEt[1] * e[3] + eEt[2] * e[6];
            tmp[1] = eEt[0] * e[1] + eEt[1] * e[4] + eEt[2] * e[7];
            tmp[2] = eEt[0] * e[2] + eEt[1] * e[5] + eEt[2] * e[8];

            tmp[3] = eEt[3] * e[0] + eEt[4] * e[3] + eEt[5] * e[6];
            tmp[4] = eEt[3] * e[1] + eEt[4] * e[4] + eEt[5] * e[7];
            tmp[5] = eEt[3] * e[2] + eEt[4] * e[5] + eEt[5] * e[8];

            tmp[6] = eEt[6] * e[0] + eEt[7] * e[3] + eEt[8] * e[6];
            tmp[7] = eEt[6] * e[1] + eEt[7] * e[4] + eEt[8] * e[7];
            tmp[8] = eEt[6] * e[2] + eEt[7] * e[5] + eEt[8] * e[8];

            denom = l * (l * l - eSq) - 2.0 * detE;
            denom = 1.0 / denom;
            r[0] = (a * e[0] + 2.0 * (l * adjE[0] - tmp[0])) * denom;
            r[1] = (a * e[1] + 2.0 * (l * adjE[3] - tmp[1])) * denom;
            r[2] = (a * e[2] + 2.0 * (l * adjE[6] - tmp[2])) * denom;

            r[3] = (a * e[3] + 2.0 * (l * adjE[1] - tmp[3])) * denom;
            r[4] = (a * e[4] + 2.0 * (l * adjE[4] - tmp[4])) * denom;
            r[5] = (a * e[5] + 2.0 * (l * adjE[7] - tmp[5])) * denom;

            r[6] = (a * e[6] + 2.0 * (l * adjE[2] - tmp[6])) * denom;
            r[7] = (a * e[7] + 2.0 * (l * adjE[5] - tmp[7])) * denom;
            r[8] = (a * e[8] + 2.0 * (l * adjE[8] - tmp[8])) * denom;
        }

        private static double Det3x3(ReadOnlySpan<double> e)
        {
            return (e[0] * e[4] * e[8] + e[1] * e[5] * e[6] + e[2] * e[3] * e[7])
                 - (e[6] * e[4] * e[2] + e[7] * e[5] * e[0] + e[8] * e[3] * e[1]);
        }

        private bool PositiveDepth(Solution solution)
        {
            double[] r = solution.RHat;
            double[] t = solution.T;
            double[] mean = _pointMean;
            return r[6] * mean[0] + r[7] * mean[1] + r[8] * mean[2] + t[2] > 0;
        }

        private static bool PositiveMajorityDepths(Solution solution, ReadOnlySpan<Point3d> objectPoints)
        {
            double[] r = solution.RHat;
            double[] t = solution.T;
            int npos = 0, nneg = 0;

            for (int i = 0; i < objectPoints.Length; i++)
            {
                Point3d objPt = objectPoints[i];
                if (r[6] * objPt.X + r[7] * objPt.Y + r[8] * objPt.Z + t[2] > 0)
                {
                    ++npos;
                }
                else
                {
                    ++nneg;
                }
            }

            return npos >= nneg;
        }

        private void CheckSolution(Solution solution, ReadOnlySpan<Point3d> objectPoints, ref double minError)
        {
            bool cheirok = PositiveDepth(solution) || PositiveMajorityDepths(solution, objectPoints);
            if (!cheirok)
            {
                return;
            }

            Span<double> omegaR = stackalloc double[9];
            MatMul(_omega, solution.RHat, omegaR, 9, 9, 1);
            solution.SqError = Dot(omegaR, solution.RHat);
            if (Math.Abs(minError - solution.SqError) > EqualSquaredErrorsDiff)
            {
                if (minError > solution.SqError)
                {
                    minError = solution.SqError;
                    Solutions[0] = solution;
                    NumSolutions = 1;
                }
            }
            else
            {
                bool found = false;
                Span<double> diff = stackalloc double[9];
                for (int i = 0; i < NumSolutions; i++)
                {
                    double[] stored = Solutions[i].RHat;
                    for (int k = 0; k < 9; k++)
                    {
                        diff[k] = stored[k] - solution.RHat[k];
                    }

                    if (NormL2Sqr(diff) < EqualVectorsSquaredDiff)
                    {
                        if (Solutions[i].SqError > solution.SqError)
                        {
                            Solutions[i] = solution;
                        }

                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    Solutions[NumSolutions++] = solution;
                }

                if (minError > solution.SqError)
                {
                    minError = solution.SqError;
                }
            }
        }

        private static double OrthogonalityError(ReadOnlySpan<double> e)
        {
            double sqNormE1 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            double sqNormE2 = e[3] * e[3] + e[4] * e[4] + e[5] * e[5];
            double sqNormE3 = e[6] * e[6] + e[7] * e[7] + e[8] * e[8];
            double dotE1E2 = e[0] * e[3] + e[1] * e[4] + e[2] * e[5];
            double dotE1E3 = e[0] * e[6] + e[1] * e[7] + e[2] * e[8];
            double dotE2E3 = e[3] * e[6] + e[4] * e[7] + e[5] * e[8];

            return ((sqNormE1 - 1) * (sqNormE1 - 1) + (sqNormE2 - 1) * (sqNormE2 - 1)) + ((sqNormE3 - 1) * (sqNormE3 - 1) +
                2 * (dotE1E2 * dotE1E2 + dotE1E3 * dotE1E3 + dotE2E3 * dotE2E3));
        }
    }
}
