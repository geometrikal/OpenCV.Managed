// This implementation is ported from the OpenCV AP3P solver (Ke and Roumeliotis, CVPR 2017).
//
// Original sources:
//   modules/geometry/src/ap3p.cpp (ap3p::ap3p, ap3p::solve, ap3p::computePoses, polishQuarticRoots)
//   modules/geometry/src/ap3p.h (ap3p::init_camera_parameters, ap3p::extract_points)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// AP3P pose solver (<c>cv::ap3p</c>, <c>SOLVEPNP_AP3P</c>).
/// </summary>
/// <remarks>
/// <c>solveP3P</c> gives normalized image points and the camera matrix to the solver.
/// The solver projects the points with the camera matrix and then normalizes them again with the inverse
/// parameters. This port keeps that round trip because it changes the rounding of the point coordinates.
/// </remarks>
internal static class Ap3p
{
    /// <summary>
    /// Calculates the camera poses for 3 or 4 point correspondences
    /// (<c>ap3p(cameraMatrix).solve(Rs, tvecs, opoints, ipoints)</c>).
    /// </summary>
    /// <param name="fx">The camera matrix element (0, 0).</param>
    /// <param name="fy">The camera matrix element (1, 1).</param>
    /// <param name="cx">The camera matrix element (0, 2).</param>
    /// <param name="cy">The camera matrix element (1, 2).</param>
    /// <param name="objectPoints">The object points. The count must be 3 or 4.</param>
    /// <param name="normalizedPoints">The normalized (undistorted, no camera matrix) image points. The count must be equal to the object point count.</param>
    /// <param name="rotations">The output rotation matrices (row-major, 9 elements). The solver appends one matrix for each solution.</param>
    /// <param name="translations">The output translation vectors (3 elements). The solver appends one vector for each solution.</param>
    /// <returns>The number of solutions: 0 to 4.</returns>
    /// <remarks>
    /// For 4 points, the solver sorts the solutions by the squared reprojection error of the 4th point, smallest first.
    /// </remarks>
    public static int Solve(
        double fx,
        double fy,
        double cx,
        double cy,
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> normalizedPoints,
        List<double[]> rotations,
        List<double[]> translations)
    {
        double invFx = 1.0 / fx;
        double invFy = 1.0 / fy;
        double cxFx = cx / fx;
        double cyFy = cy / fy;

        int npoints = objectPoints.Length;

        // extract_points: points[i * 5 + k] = (u, v, X, Y, Z); unused entries are zero.
        Span<double> points = stackalloc double[20];
        points.Clear();
        for (int i = 0; i < npoints; i++)
        {
            points[i * 5] = normalizedPoints[i].X * fx + cx;
            points[i * 5 + 1] = normalizedPoints[i].Y * fy + cy;
            points[i * 5 + 2] = objectPoints[i].X;
            points[i * 5 + 3] = objectPoints[i].Y;
            points[i * 5 + 4] = objectPoints[i].Z;
        }

        bool p4p = npoints == 4;

        // featureVectors[3][4] and worldPoints[3][4], row-major.
        Span<double> fv = stackalloc double[12];
        Span<double> wp = stackalloc double[12];
        for (int i = 0; i < 4; i++)
        {
            double mu = points[i * 5];
            double mv = points[i * 5 + 1];
            mu = invFx * mu - cxFx;
            mv = invFy * mv - cyFy;
            double mk = 1;
            if (i < 3)
            {
                double norm = Math.Sqrt(mu * mu + mv * mv + 1);
                mk = 1.0 / norm;
                mu *= mk;
                mv *= mk;
            }

            fv[i] = mu;
            fv[4 + i] = mv;
            fv[8 + i] = mk;
            wp[i] = points[i * 5 + 2];
            wp[4 + i] = points[i * 5 + 3];
            wp[8 + i] = points[i * 5 + 4];
        }

        Span<double> solR = stackalloc double[36];
        Span<double> solT = stackalloc double[12];
        int solutions = ComputePoses(fv, wp, solR, solT, p4p);

        for (int i = 0; i < solutions; i++)
        {
            rotations.Add(solR.Slice(i * 9, 9).ToArray());
            translations.Add(solT.Slice(i * 3, 3).ToArray());
        }

        return solutions;
    }

    private static int ComputePoses(
        ReadOnlySpan<double> featureVectors,
        ReadOnlySpan<double> worldPoints,
        Span<double> solutionsR,
        Span<double> solutionsT,
        bool p4p)
    {
        Span<double> w1 = stackalloc double[3] { worldPoints[0], worldPoints[4], worldPoints[8] };
        Span<double> w2 = stackalloc double[3] { worldPoints[1], worldPoints[5], worldPoints[9] };
        Span<double> w3 = stackalloc double[3] { worldPoints[2], worldPoints[6], worldPoints[10] };

        Span<double> u0 = stackalloc double[3];
        VectSub(w1, w2, u0);

        double nu0 = VectNorm(u0);
        Span<double> k1 = stackalloc double[3];
        VectDivide(u0, nu0, k1);

        Span<double> b1 = stackalloc double[3] { featureVectors[0], featureVectors[4], featureVectors[8] };
        Span<double> b2 = stackalloc double[3] { featureVectors[1], featureVectors[5], featureVectors[9] };
        Span<double> b3 = stackalloc double[3] { featureVectors[2], featureVectors[6], featureVectors[10] };

        Span<double> k3 = stackalloc double[3];
        VectCross(b1, b2, k3);
        double nk3 = VectNorm(k3);
        VectDivide(k3, nk3, k3);

        Span<double> tz = stackalloc double[3];
        VectCross(b1, k3, tz);

        Span<double> v1 = stackalloc double[3];
        VectCross(b1, b3, v1);
        Span<double> v2 = stackalloc double[3];
        VectCross(b2, b3, v2);

        Span<double> u1 = stackalloc double[3];
        VectSub(w1, w3, u1);

        double u1k1 = VectDot(u1, k1);
        double k3b3 = VectDot(k3, b3);

        double f11 = k3b3;
        double f13 = VectDot(k3, v1);
        double f15 = -u1k1 * f11;

        Span<double> nl = stackalloc double[3];
        VectCross(u1, k1, nl);
        double delta = VectNorm(nl);
        VectDivide(nl, delta, nl);
        f11 *= delta;
        f13 *= delta;

        double u2k1 = u1k1 - nu0;
        double f21 = VectDot(tz, v2);
        double f22 = nk3 * k3b3;
        double f23 = VectDot(k3, v2);
        double f24 = u2k1 * f22;
        double f25 = -u2k1 * f21;
        f21 *= delta;
        f22 *= delta;
        f23 *= delta;
        double g1 = f13 * f22;
        double g2 = f13 * f25 - f15 * f23;
        double g3 = f11 * f23 - f13 * f21;
        double g4 = -f13 * f24;
        double g5 = f11 * f22;
        double g6 = f11 * f25 - f15 * f21;
        double g7 = -f15 * f24;

        Span<double> coeffs = stackalloc double[5]
        {
            g5 * g5 + g1 * g1 + g3 * g3,
            2 * (g5 * g6 + g1 * g2 + g3 * g4),
            g6 * g6 + 2 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3,
            2 * (g6 * g7 - g1 * g2 - g3 * g4),
            g7 * g7 - g2 * g2 - g4 * g4,
        };
        Span<double> s = stackalloc double[4];
        int nbRoots = PnpPolynomials.SolveDeg4(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], s);
        PolishQuarticRoots(coeffs, s, nbRoots);

        Span<double> temp = stackalloc double[3];
        VectCross(k1, nl, temp);

        Span<double> ck1nl = stackalloc double[9]
        {
            k1[0], nl[0], temp[0],
            k1[1], nl[1], temp[1],
            k1[2], nl[2], temp[2],
        };

        Span<double> cb1k3tzT = stackalloc double[9]
        {
            b1[0], b1[1], b1[2],
            k3[0], k3[1], k3[2],
            tz[0], tz[1], tz[2],
        };

        Span<double> b3p = stackalloc double[3];
        VectScale(delta / k3b3, b3, b3p);

        double x3 = worldPoints[3];
        double y3 = worldPoints[7];
        double z3 = worldPoints[11];
        double mu3 = featureVectors[3];
        double mv3 = featureVectors[7];
        Span<double> reprojErrors = stackalloc double[4];

        Span<double> c13 = stackalloc double[9];
        Span<double> tempMatrix = stackalloc double[9];
        Span<double> r = stackalloc double[9];
        Span<double> rp3 = stackalloc double[3];
        Span<double> pxstheta1p = stackalloc double[3];

        int nbSolutions = 0;
        for (int i = 0; i < nbRoots; ++i)
        {
            double ctheta1p = s[i];
            if (Math.Abs(ctheta1p) > 1)
            {
                continue;
            }

            double stheta1p = Math.Sqrt(1 - ctheta1p * ctheta1p);
            stheta1p = (k3b3 > 0) ? stheta1p : -stheta1p;
            double ctheta3 = g1 * ctheta1p + g2;
            double stheta3 = g3 * ctheta1p + g4;
            double ntheta3 = stheta1p / ((g5 * ctheta1p + g6) * ctheta1p + g7);
            ctheta3 *= ntheta3;
            stheta3 *= ntheta3;

            c13[0] = ctheta3;
            c13[1] = 0;
            c13[2] = -stheta3;
            c13[3] = stheta1p * stheta3;
            c13[4] = ctheta1p;
            c13[5] = stheta1p * ctheta3;
            c13[6] = ctheta1p * stheta3;
            c13[7] = -stheta1p;
            c13[8] = ctheta1p * ctheta3;

            MatMult(ck1nl, c13, tempMatrix);
            MatMult(tempMatrix, cb1k3tzT, r);

            // R' * p3
            rp3[0] = w3[0] * r[0] + w3[1] * r[3] + w3[2] * r[6];
            rp3[1] = w3[0] * r[1] + w3[1] * r[4] + w3[2] * r[7];
            rp3[2] = w3[0] * r[2] + w3[1] * r[5] + w3[2] * r[8];

            VectScale(stheta1p, b3p, pxstheta1p);

            Span<double> sT = solutionsT.Slice(nbSolutions * 3, 3);
            VectSub(pxstheta1p, rp3, sT);

            Span<double> sR = solutionsR.Slice(nbSolutions * 9, 9);
            sR[0] = r[0];
            sR[3] = r[1];
            sR[6] = r[2];
            sR[1] = r[3];
            sR[4] = r[4];
            sR[7] = r[5];
            sR[2] = r[6];
            sR[5] = r[7];
            sR[8] = r[8];

            if (p4p)
            {
                double x3p = sR[0] * x3 + sR[1] * y3 + sR[2] * z3 + sT[0];
                double y3p = sR[3] * x3 + sR[4] * y3 + sR[5] * z3 + sT[1];
                double z3p = sR[6] * x3 + sR[7] * y3 + sR[8] * z3 + sT[2];
                double mu3p = x3p / z3p;
                double mv3p = y3p / z3p;
                reprojErrors[nbSolutions] = (mu3p - mu3) * (mu3p - mu3) + (mv3p - mv3) * (mv3p - mv3);
            }

            nbSolutions++;
        }

        if (p4p)
        {
            for (int i = 1; i < nbSolutions; i++)
            {
                for (int j = i; j > 0 && reprojErrors[j - 1] > reprojErrors[j]; j--)
                {
                    (reprojErrors[j], reprojErrors[j - 1]) = (reprojErrors[j - 1], reprojErrors[j]);
                    for (int k = 0; k < 9; k++)
                    {
                        (solutionsR[j * 9 + k], solutionsR[(j - 1) * 9 + k]) = (solutionsR[(j - 1) * 9 + k], solutionsR[j * 9 + k]);
                    }

                    for (int k = 0; k < 3; k++)
                    {
                        (solutionsT[j * 3 + k], solutionsT[(j - 1) * 3 + k]) = (solutionsT[(j - 1) * 3 + k], solutionsT[j * 3 + k]);
                    }
                }
            }
        }

        return nbSolutions;
    }

    private static void PolishQuarticRoots(ReadOnlySpan<double> coeffs, Span<double> roots, int nbRoots)
    {
        const int Iterations = 2;
        for (int i = 0; i < Iterations; ++i)
        {
            for (int j = 0; j < nbRoots; ++j)
            {
                double error =
                    (((coeffs[0] * roots[j] + coeffs[1]) * roots[j] + coeffs[2]) * roots[j] + coeffs[3]) * roots[j] +
                    coeffs[4];
                double derivative =
                    ((4 * coeffs[0] * roots[j] + 3 * coeffs[1]) * roots[j] + 2 * coeffs[2]) * roots[j] + coeffs[3];
                roots[j] -= error / derivative;
            }
        }
    }

    private static void VectCross(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        double r0 = a[1] * b[2] - a[2] * b[1];
        double r1 = -(a[0] * b[2] - a[2] * b[0]);
        double r2 = a[0] * b[1] - a[1] * b[0];
        result[0] = r0;
        result[1] = r1;
        result[2] = r2;
    }

    private static double VectDot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    private static double VectNorm(ReadOnlySpan<double> a)
    {
        return Math.Sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }

    private static void VectScale(double s, ReadOnlySpan<double> a, Span<double> result)
    {
        result[0] = a[0] * s;
        result[1] = a[1] * s;
        result[2] = a[2] * s;
    }

    private static void VectSub(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        result[0] = a[0] - b[0];
        result[1] = a[1] - b[1];
        result[2] = a[2] - b[2];
    }

    private static void VectDivide(ReadOnlySpan<double> a, double d, Span<double> result)
    {
        result[0] = a[0] / d;
        result[1] = a[1] / d;
        result[2] = a[2] / d;
    }

    private static void MatMult(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> result)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                result[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            }
        }
    }
}
