// This implementation is ported from the OpenCV closed-form polynomial solvers used by the P3P solvers.
//
// Original sources:
//   modules/geometry/src/polynom_solver.cpp (solve_deg2, solve_deg3, solve_deg4)
//   modules/geometry/src/polynom_solver.h
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

/// <summary>
/// Closed-form real-root solvers for polynomials of degree 2, 3 and 4 (<c>polynom_solver.cpp</c>).
/// </summary>
/// <remarks>
/// Each solver writes the real roots to the start of <c>roots</c> and returns the number of real roots.
/// A solver does not change the elements of <c>roots</c> that it does not write.
/// </remarks>
internal static class PnpPolynomials
{
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>
    /// Solves <c>a*x^2 + b*x + c = 0</c> (<c>solve_deg2</c>).
    /// </summary>
    /// <param name="a">The coefficient of <c>x^2</c>.</param>
    /// <param name="b">The coefficient of <c>x</c>.</param>
    /// <param name="c">The constant coefficient.</param>
    /// <param name="roots">The output roots. The length must be 2 or more.</param>
    /// <returns>The number of real roots: 0, 1 or 2. For 1 root, the solver writes the root to both elements.</returns>
    public static int SolveDeg2(double a, double b, double c, Span<double> roots)
    {
        double delta = b * b - 4 * a * c;

        if (delta < 0)
        {
            return 0;
        }

        double inv2a = 0.5 / a;

        if (delta == 0)
        {
            roots[0] = -b * inv2a;
            roots[1] = roots[0];
            return 1;
        }

        double sqrtDelta = Math.Sqrt(delta);
        roots[0] = (-b + sqrtDelta) * inv2a;
        roots[1] = (-b - sqrtDelta) * inv2a;
        return 2;
    }

    /// <summary>
    /// Solves <c>a*x^3 + b*x^2 + c*x + d = 0</c> (<c>solve_deg3</c>).
    /// </summary>
    /// <param name="a">The coefficient of <c>x^3</c>.</param>
    /// <param name="b">The coefficient of <c>x^2</c>.</param>
    /// <param name="c">The coefficient of <c>x</c>.</param>
    /// <param name="d">The constant coefficient.</param>
    /// <param name="roots">The output roots. The length must be 3 or more.</param>
    /// <returns>The number of real roots: 0, 1, 2 or 3.</returns>
    public static int SolveDeg3(double a, double b, double c, double d, Span<double> roots)
    {
        if (a == 0)
        {
            if (b == 0)
            {
                if (c == 0)
                {
                    return 0;
                }

                roots[0] = -d / c;
                return 1;
            }

            roots[2] = 0;
            return SolveDeg2(b, c, d, roots);
        }

        double invA = 1.0 / a;
        double bA = invA * b, bA2 = bA * bA;
        double cA = invA * c;
        double dA = invA * d;

        double q = (3 * cA - bA2) / 9;
        double r = (9 * bA * cA - 27 * dA - 2 * bA * bA2) / 54;
        double q3 = q * q * q;
        double dd = q3 + r * r;
        double bA3 = (1.0 / 3.0) * bA;

        if (q == 0)
        {
            if (r == 0)
            {
                roots[0] = roots[1] = roots[2] = -bA3;
                return 3;
            }

            double cubeRoot = Math.Cbrt(2 * r);
            roots[0] = cubeRoot - bA3;
            return 1;
        }

        if (dd <= 0)
        {
            double theta = Math.Acos(r / Math.Sqrt(-q3));
            double sqrtQ = Math.Sqrt(-q);
            roots[0] = 2 * sqrtQ * Math.Cos(theta / 3.0) - bA3;
            roots[1] = 2 * sqrtQ * Math.Cos((theta + 2 * Math.PI) / 3.0) - bA3;
            roots[2] = 2 * sqrtQ * Math.Cos((theta + 4 * Math.PI) / 3.0) - bA3;
            return 3;
        }

        double ad = 0.0;
        double bd = 0.0;
        double rAbs = Math.Abs(r);
        if (rAbs > DblEpsilon)
        {
            ad = Math.Cbrt(rAbs + Math.Sqrt(dd));
            ad = (r >= 0) ? ad : -ad;
            bd = -q / ad;
        }

        roots[0] = ad + bd - bA3;
        return 1;
    }

    /// <summary>
    /// Solves <c>a*x^4 + b*x^3 + c*x^2 + d*x + e = 0</c> (<c>solve_deg4</c>).
    /// </summary>
    /// <param name="a">The coefficient of <c>x^4</c>.</param>
    /// <param name="b">The coefficient of <c>x^3</c>.</param>
    /// <param name="c">The coefficient of <c>x^2</c>.</param>
    /// <param name="d">The coefficient of <c>x</c>.</param>
    /// <param name="e">The constant coefficient.</param>
    /// <param name="roots">The output roots. The length must be 4 or more.</param>
    /// <returns>The number of real roots: 0 to 4.</returns>
    public static int SolveDeg4(double a, double b, double c, double d, double e, Span<double> roots)
    {
        if (a == 0)
        {
            roots[3] = 0;
            return SolveDeg3(b, c, d, e, roots);
        }

        double invA = 1.0 / a;
        b *= invA;
        c *= invA;
        d *= invA;
        e *= invA;
        double b2 = b * b, bc = b * c, b3 = b2 * b;

        Span<double> r = stackalloc double[3];
        int n = SolveDeg3(1, -c, d * b - 4 * e, 4 * c * e - d * d - b2 * e, r);
        if (n == 0)
        {
            return 0;
        }

        double r2 = 0.25 * b2 - c + r[0];
        if (r2 < 0)
        {
            return 0;
        }

        double rr = Math.Sqrt(r2);
        double invR = 1.0 / rr;

        int nbRealRoots = 0;

        double d2, e2;
        if (rr < 10E-12)
        {
            double temp = r[0] * r[0] - 4 * e;
            if (temp < 0)
            {
                d2 = e2 = -1;
            }
            else
            {
                double sqrtTemp = Math.Sqrt(temp);
                d2 = 0.75 * b2 - 2 * c + 2 * sqrtTemp;
                e2 = d2 - 4 * sqrtTemp;
            }
        }
        else
        {
            double u = 0.75 * b2 - 2 * c - r2,
                v = 0.25 * invR * (4 * bc - 8 * d - b3);
            d2 = u + v;
            e2 = u - v;
        }

        double b4 = 0.25 * b, rHalf = 0.5 * rr;
        if (d2 >= 0)
        {
            double dRoot = Math.Sqrt(d2);
            nbRealRoots = 2;
            double dHalf = 0.5 * dRoot;
            roots[0] = rHalf + dHalf - b4;
            roots[1] = roots[0] - dRoot;
        }

        if (e2 >= 0)
        {
            double eRoot = Math.Sqrt(e2);
            double eHalf = 0.5 * eRoot;
            if (nbRealRoots == 0)
            {
                roots[0] = -rHalf + eHalf - b4;
                roots[1] = roots[0] - eRoot;
                nbRealRoots = 2;
            }
            else
            {
                roots[2] = -rHalf + eHalf - b4;
                roots[3] = roots[2] - eRoot;
                nbRealRoots = 4;
            }
        }

        return nbRealRoots;
    }
}
