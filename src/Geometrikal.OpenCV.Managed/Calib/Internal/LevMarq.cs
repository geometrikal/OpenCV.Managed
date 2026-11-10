// This implementation is ported from the OpenCV 5.0 Levenberg-Marquardt solver.
//
// Original sources:
//   modules/geometry/src/levmarq.cpp (LevMarq::Settings, LevMarqBase::optimize,
//   LevMarqDenseLinearBackend)
//   modules/core/src/norm.simd.hpp (NormL2_SIMD<double, double>)
//   modules/core/src/sum.simd.hpp (sum_ for double)
//   modules/core/src/matmul.simd.hpp (dotProd_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>A least-squares problem for <see cref="LevMarq"/>: an OpenCV "long" callback.</summary>
internal interface ILevMarqProblem
{
    /// <summary>Gets the number of parameters.</summary>
    int ParamCount { get; }

    /// <summary>Gets the number of residuals.</summary>
    int ErrorCount { get; }

    /// <summary>
    /// Computes the residuals at <paramref name="x"/> and, when <paramref name="jac"/> is not
    /// empty, the row-major Jacobian. The caller clears <paramref name="jac"/> first.
    /// </summary>
    void Compute(ReadOnlySpan<double> x, Span<double> err, Span<double> jac);
}

/// <summary>
/// OpenCV 5.0 <c>LevMarq</c> with the dense linear backend, the default settings, geodesic
/// acceleration and <c>DECOMP_SVD</c>, as the 2-D estimators use it. The element operations
/// follow the OpenCV <c>MatExpr</c> evaluation order.
/// </summary>
internal static class LevMarq
{
    private const double HGeo = 1e-4;
    private const double GeoScale = 0.5;
    private const double StepNormTolerance = 1e-6;
    private const double RelEnergyDeltaTolerance = 1e-6;
    private const double MinGradientTolerance = 1e-6;
    private const double SmallEnergyTolerance = 0;
    private const double InitialLambda = 0.0001;
    private const double InitialLmUpFactor = 2.0;
    private const double InitialLmDownFactor = 3.0;
    private const double MinDiag = 1e-6;
    private const double MaxDiag = 1e32;
    private const double MaxLambda = 1e32;
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>
    /// Runs <c>LevMarq(param, callback, Settings().setMaxIterations(n).setGeodesic(true)).optimize()</c>.
    /// </summary>
    /// <param name="param">Parameters. The accepted steps are written in place.</param>
    /// <param name="problem">Residual and Jacobian callback.</param>
    /// <param name="maxIterations">Maximum number of iterations.</param>
    /// <param name="stepNormTolerance"><c>Settings::stepNormTolerance</c>.</param>
    /// <param name="smallEnergyTolerance"><c>Settings::smallEnergyTolerance</c>.</param>
    /// <returns>True when a convergence criterion was met.</returns>
    public static bool Optimize<TProblem>(
        Span<double> param,
        in TProblem problem,
        int maxIterations,
        double stepNormTolerance = StepNormTolerance,
        double smallEnergyTolerance = SmallEnergyTolerance)
        where TProblem : struct, ILevMarqProblem
    {
        int p = problem.ParamCount;
        int nerr = problem.ErrorCount;
        using PooledMat<double> jPooled = PooledMat<double>.Rent(p, nerr, 1);
        using PooledMat<double> bPooled = PooledMat<double>.Rent(1, nerr, 1);
        using PooledMat<double> bvPooled = PooledMat<double>.Rent(1, nerr, 1);
        Mat<double> jLong = jPooled.Mat;
        Mat<double> bLong = bPooled.Mat;
        Mat<double> bvMat = bvPooled.Mat;
        Span<double> bv = bvMat.GetSpan();
        bv.Clear();
        double[] probeX = param.ToArray();
        double[] geoX = new double[p];

        if (!CalcFunc(problem, probeX, jLong, bLong, true, out Mat<double> jtj, out Mat<double> jtb))
        {
            return false;
        }

        double energy = NormL2Sqr(bLong.GetReadOnlySpan());
        if (energy < 0)
        {
            return false;
        }

        double oldEnergy = energy;
        bool smallGradient = false, smallStep = false, smallEnergyDelta = false, smallEnergy = false;
        double lmUpFactor = InitialLmUpFactor;
        double lambda = InitialLambda;
        uint iter = 0;
        bool done = false;

        double[] diag = new double[p];
        double[] lmDiag = new double[p];
        double[] tmp = new double[p];
        var rhs = new Mat<double>(1, p);
        while (!done)
        {
            ReadOnlySpan<double> jtbS = jtb.GetReadOnlySpan();
            double gradientMax = NormInf(jtbS);
            for (int i = 0; i < p; i++)
            {
                diag[i] = jtj.At(i, i);
            }

            bool enoughLm = false;
            while (!enoughLm && !done)
            {
                // lmDiag = clamp(diag * lambda); jtj.diag = lmDiag + diag.
                bool copyScale = Math.Abs(lambda - 1.0) < DblEpsilon;
                for (int i = 0; i < p; i++)
                {
                    double d = copyScale ? diag[i] : (diag[i] * lambda) + 0.0;
                    d = d < MinDiag ? MinDiag : d;
                    d = MaxDiag < d ? MaxDiag : d;
                    lmDiag[i] = d;
                    jtj.At(i, i) = d + diag[i];
                }

                Mat<double> x = SolveNeg(jtj, jtbS, rhs);

                // Solve with SVD always succeeds.
                Span<double> xs = x.GetSpan();
                for (int i = 0; i < p; i++)
                {
                    tmp[i] = xs[i] * (jtbS[i] - (lmDiag[i] * xs[i]));
                }

                double jacCostChange = -0.5 * Sum(tmp);
                double xNorm = NormL2Sqr(xs);

                // Geodesic acceleration.
                for (int i = 0; i < p; i++)
                {
                    geoX[i] = param[i] + ((xs[i] * HGeo) + 0.0);
                }

                problem.Compute(geoX, bv, default);
                Mat<double> jtbv = MulAtB(jLong, bvMat);
                ReadOnlySpan<double> jtbvS = jtbv.GetReadOnlySpan();
                const double Alpha = 1.0 / (HGeo * HGeo);
                for (int i = 0; i < p; i++)
                {
                    double m1 = (jtbS[i] * (HGeo - 1.0)) + jtbvS[i];
                    double m2 = (lmDiag[i] * HGeo) * xs[i];
                    tmp[i] = (m1 * Alpha) + ((m2 * Alpha) + 0.0);
                }

                Mat<double> xgeo = SolveNeg(jtj, tmp, rhs);
                ReadOnlySpan<double> xg = xgeo.GetReadOnlySpan();
                double truncerr = Math.Sqrt(Dot(xg, xg) / Dot(xs, xs));
                if (truncerr < 1.0)
                {
                    for (int i = 0; i < p; i++)
                    {
                        xs[i] += (xg[i] * GeoScale) + 0.0;
                    }
                }

                for (int i = 0; i < p; i++)
                {
                    probeX[i] = param[i] + xs[i];
                }

                problem.Compute(probeX, bLong.GetSpan(), default);
                energy = NormL2Sqr(bLong.GetReadOnlySpan());
                if (energy < 0 || double.IsNaN(energy))
                {
                    return false;
                }

                double costChange = oldEnergy - energy;
                double stepQuality = costChange / jacCostChange;

                if (costChange < 0)
                {
                    lambda *= lmUpFactor;
                    lmUpFactor *= 2.0;
                }
                else
                {
                    enoughLm = true;
                    double down = 1.0 - Math.Pow((2.0 * stepQuality) - 1.0, 3);
                    double third = 1.0 / InitialLmDownFactor;
                    lambda *= third < down ? down : third;
                    lmUpFactor = InitialLmUpFactor;

                    smallGradient = gradientMax < MinGradientTolerance;
                    smallStep = xNorm < stepNormTolerance;
                    smallEnergyDelta = costChange / energy < RelEnergyDeltaTolerance;
                    smallEnergy = energy < smallEnergyTolerance;

                    probeX.CopyTo(param);
                    oldEnergy = energy;
                }

                iter++;
                done = iter >= (uint)maxIterations || lambda >= MaxLambda
                    || smallGradient || smallStep || smallEnergyDelta || smallEnergy;
            }

            if (!done && !CalcFunc(problem, probeX, jLong, bLong, true, out jtj, out jtb))
            {
                return false;
            }
        }

        return smallGradient || smallStep || smallEnergyDelta || smallEnergy;
    }

    /// <summary>
    /// <c>calcFunc</c> with the Jacobian: residuals and Jacobian at <paramref name="x"/>,
    /// <c>JᵀJ</c> by <c>mulTransposed</c> and <c>Jᵀb</c> by <c>gemm</c>.
    /// </summary>
    private static bool CalcFunc<TProblem>(
        in TProblem problem,
        double[] x,
        Mat<double> jLong,
        Mat<double> bLong,
        bool calcJacobian,
        out Mat<double> jtj,
        out Mat<double> jtb)
        where TProblem : struct, ILevMarqProblem
    {
        jLong.Clear();
        bLong.Clear();
        problem.Compute(x, bLong.GetSpan(), calcJacobian ? jLong.GetSpan() : default);
        jtj = MulTransposedR(jLong.GetReadOnlySpan(), jLong.Height, jLong.Width);
        jtb = MulAtB(jLong, bLong);
        return true;
    }

    /// <summary>
    /// <c>mulTransposed(J, JᵀJ, true)</c> for fewer than 100 columns (<c>MulTransposedR</c>):
    /// every element is a sequential <c>double</c> sum over the rows. One pass over the rows
    /// computes all elements in the same per-element order.
    /// </summary>
    private static Mat<double> MulTransposedR(ReadOnlySpan<double> j, int rows, int p)
    {
        var result = new Mat<double>(p, p);
        Span<double> d = result.GetSpan();
        ref double rj = ref MemoryMarshal.GetReference(j);
        ref double rd = ref MemoryMarshal.GetReference(d);
        if (p == 6 && Vector256.IsHardwareAccelerated)
        {
            MulTransposed6(ref rj, rows, ref rd);
        }
        else if (p == 4 && Vector256.IsHardwareAccelerated)
        {
            MulTransposed4(ref rj, rows, ref rd);
        }
        else if (p == 9 && Vector256.IsHardwareAccelerated)
        {
            MulTransposed9Third(ref rj, rows, 0, ref rd);
            MulTransposed9Third(ref rj, rows, 3, ref rd);
            MulTransposed9Third(ref rj, rows, 6, ref rd);
        }
        else
        {
            MulTransposedRows(ref rj, rows, p, ref rd);
        }

        for (int a = 0; a < p; a++)
        {
            for (int b = 0; b < a; b++)
            {
                d[(a * p) + b] = d[(b * p) + a];
            }
        }

        return result;
    }

    /// <summary>Six-column <see cref="MulTransposedR"/> rows with register accumulators.</summary>
    private static void MulTransposed6(ref double rj, int rows, ref double rd)
    {
        Vector256<double> l0 = default, l1 = default, l2 = default, l3 = default, l4 = default, l5 = default;
        Vector128<double> h0 = default, h1 = default, h2 = default, h3 = default, h4 = default, h5 = default;
        for (int k = 0; k < rows; k++)
        {
            ref double row = ref Unsafe.Add(ref rj, k * 6);
            Vector256<double> lo = Vector256.LoadUnsafe(ref row);
            Vector128<double> hi = Vector128.LoadUnsafe(ref row, 4u);
            Vector256<double> s = Vector256.Create(row);
            l0 += s * lo;
            h0 += s.GetLower() * hi;
            s = Vector256.Create(Unsafe.Add(ref row, 1));
            l1 += s * lo;
            h1 += s.GetLower() * hi;
            s = Vector256.Create(Unsafe.Add(ref row, 2));
            l2 += s * lo;
            h2 += s.GetLower() * hi;
            s = Vector256.Create(Unsafe.Add(ref row, 3));
            l3 += s * lo;
            h3 += s.GetLower() * hi;
            s = Vector256.Create(Unsafe.Add(ref row, 4));
            l4 += s * lo;
            h4 += s.GetLower() * hi;
            s = Vector256.Create(Unsafe.Add(ref row, 5));
            l5 += s * lo;
            h5 += s.GetLower() * hi;
        }

        l0.StoreUnsafe(ref rd, 0u);
        h0.StoreUnsafe(ref rd, 4u);
        l1.StoreUnsafe(ref rd, 6u);
        h1.StoreUnsafe(ref rd, 10u);
        l2.StoreUnsafe(ref rd, 12u);
        h2.StoreUnsafe(ref rd, 16u);
        l3.StoreUnsafe(ref rd, 18u);
        h3.StoreUnsafe(ref rd, 22u);
        l4.StoreUnsafe(ref rd, 24u);
        h4.StoreUnsafe(ref rd, 28u);
        l5.StoreUnsafe(ref rd, 30u);
        h5.StoreUnsafe(ref rd, 34u);
    }

    /// <summary>Four-column <see cref="MulTransposedR"/> rows with register accumulators.</summary>
    private static void MulTransposed4(ref double rj, int rows, ref double rd)
    {
        Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default;
        for (int k = 0; k < rows; k++)
        {
            ref double row = ref Unsafe.Add(ref rj, k * 4);
            Vector256<double> v = Vector256.LoadUnsafe(ref row);
            a0 += Vector256.Create(row) * v;
            a1 += Vector256.Create(Unsafe.Add(ref row, 1)) * v;
            a2 += Vector256.Create(Unsafe.Add(ref row, 2)) * v;
            a3 += Vector256.Create(Unsafe.Add(ref row, 3)) * v;
        }

        a0.StoreUnsafe(ref rd, 0u);
        a1.StoreUnsafe(ref rd, 4u);
        a2.StoreUnsafe(ref rd, 8u);
        a3.StoreUnsafe(ref rd, 12u);
    }

    /// <summary>Rows <paramref name="a0"/> to <c>a0 + 2</c> of a nine-column <see cref="MulTransposedR"/>.</summary>
    private static void MulTransposed9Third(ref double rj, int rows, int a0, ref double rd)
    {
        Vector256<double> l0 = default, l1 = default, l2 = default;
        Vector256<double> m0 = default, m1 = default, m2 = default;
        double e0 = 0, e1 = 0, e2 = 0;
        for (int k = 0; k < rows; k++)
        {
            ref double row = ref Unsafe.Add(ref rj, k * 9);
            Vector256<double> lo = Vector256.LoadUnsafe(ref row);
            Vector256<double> mid = Vector256.LoadUnsafe(ref row, 4u);
            double last = Unsafe.Add(ref row, 8);
            double r0 = Unsafe.Add(ref row, a0), r1 = Unsafe.Add(ref row, a0 + 1), r2 = Unsafe.Add(ref row, a0 + 2);
            Vector256<double> s = Vector256.Create(r0);
            l0 += s * lo;
            m0 += s * mid;
            e0 += r0 * last;
            s = Vector256.Create(r1);
            l1 += s * lo;
            m1 += s * mid;
            e1 += r1 * last;
            s = Vector256.Create(r2);
            l2 += s * lo;
            m2 += s * mid;
            e2 += r2 * last;
        }

        ref double d = ref Unsafe.Add(ref rd, a0 * 9);
        l0.StoreUnsafe(ref d, 0u);
        m0.StoreUnsafe(ref d, 4u);
        Unsafe.Add(ref d, 8) = e0;
        l1.StoreUnsafe(ref d, 9u);
        m1.StoreUnsafe(ref d, 13u);
        Unsafe.Add(ref d, 17) = e1;
        l2.StoreUnsafe(ref d, 18u);
        m2.StoreUnsafe(ref d, 22u);
        Unsafe.Add(ref d, 26) = e2;
    }

    private static void MulTransposedRows(ref double rj, int rows, int p, ref double rd)
    {
        int vc = Vector<double>.Count;
        int vEnd = Vector.IsHardwareAccelerated ? p - (p % vc) : 0;
        for (int k = 0; k < rows; k++)
        {
            ref double row = ref Unsafe.Add(ref rj, k * p);
            for (int a = 0; a < p; a++)
            {
                double ra = Unsafe.Add(ref row, a);
                var rav = new Vector<double>(ra);
                ref double dr = ref Unsafe.Add(ref rd, a * p);

                // The caller overwrites the lower triangle; computing it keeps the loop vectorised.
                int b = 0;
                for (; b < vEnd; b += vc)
                {
                    (Vector.LoadUnsafe(ref dr, (nuint)b) + (rav * Vector.LoadUnsafe(ref row, (nuint)b))).StoreUnsafe(ref dr, (nuint)b);
                }

                for (; b < p; b++)
                {
                    Unsafe.Add(ref dr, b) += ra * Unsafe.Add(ref row, b);
                }
            }
        }
    }

    /// <summary>
    /// <c>gemm(J, b, 1, noArray(), 0, dst, GEMM_1_T)</c> for a column <paramref name="b"/>. From
    /// <see cref="GemmKernels.BlasRowThreshold"/> rows, the gemm sums every element sequentially
    /// over the rows from zero; this loop keeps that order.
    /// </summary>
    private static Mat<double> MulAtB(Mat<double> j, Mat<double> b)
    {
        int rows = j.Height, p = j.Width;
        if (rows < GemmKernels.BlasRowThreshold || !j.IsContinuous || !b.IsContinuous)
        {
            return LinearAlgebra.Gemm(j, b, 1, null, 0, GemmFlags.ATranspose);
        }

        var result = new Mat<double>(1, p);
        Span<double> acc = result.GetSpan();
        ref double rj = ref MemoryMarshal.GetReference(j.GetReadOnlySpan());
        ref double rb = ref MemoryMarshal.GetReference(b.GetReadOnlySpan());
        ref double ra = ref MemoryMarshal.GetReference(acc);
        int vc = Vector<double>.Count;
        int vEnd = Vector.IsHardwareAccelerated ? p - (p % vc) : 0;
        for (int k = 0; k < rows; k++)
        {
            double bk = Unsafe.Add(ref rb, k);
            ref double row = ref Unsafe.Add(ref rj, k * p);
            var bv = new Vector<double>(bk);
            int i = 0;
            for (; i < vEnd; i += vc)
            {
                (Vector.LoadUnsafe(ref ra, (nuint)i) + (Vector.LoadUnsafe(ref row, (nuint)i) * bv)).StoreUnsafe(ref ra, (nuint)i);
            }

            for (; i < p; i++)
            {
                Unsafe.Add(ref ra, i) += Unsafe.Add(ref row, i) * bk;
            }
        }

        return result;
    }

    /// <summary><c>cv::solve(jtj, -right, x, DECOMP_SVD)</c>; <c>-right</c> is <c>0 - right</c>.</summary>
    private static Mat<double> SolveNeg(Mat<double> jtj, ReadOnlySpan<double> right, Mat<double> rhs)
    {
        Span<double> r = rhs.GetSpan();
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = 0.0 - right[i];
        }

        LinearAlgebra.Solve(jtj, rhs, out Mat<double> x, DecompTypes.Svd);
        return x;
    }

    /// <summary><c>norm(src, NORM_L2SQR)</c> for <c>double</c>: four 2-lane accumulators, then a sequential tail.</summary>
    internal static double NormL2Sqr(ReadOnlySpan<double> src)
    {
        int n = src.Length, j = 0;
        double r00a = 0, r00b = 0, r01a = 0, r01b = 0, r10a = 0, r10b = 0, r11a = 0, r11b = 0;
        for (; j <= n - 8; j += 8)
        {
            r00a += src[j] * src[j];
            r00b += src[j + 1] * src[j + 1];
            r01a += src[j + 2] * src[j + 2];
            r01b += src[j + 3] * src[j + 3];
            r10a += src[j + 4] * src[j + 4];
            r10b += src[j + 5] * src[j + 5];
            r11a += src[j + 6] * src[j + 6];
            r11b += src[j + 7] * src[j + 7];
        }

        double la = ((r00a + r01a) + r10a) + r11a;
        double lb = ((r00b + r01b) + r10b) + r11b;
        double s = 0.0 + (la + lb);
        for (; j < n; j++)
        {
            s += src[j] * src[j];
        }

        return s;
    }

    /// <summary><c>cv::sum</c> for one <c>double</c> channel: groups of four, then a sequential tail.</summary>
    private static double Sum(ReadOnlySpan<double> src)
    {
        double s = 0;
        int i = 0;
        for (; i <= src.Length - 4; i += 4)
        {
            s += src[i] + src[i + 1] + src[i + 2] + src[i + 3];
        }

        for (; i < src.Length; i++)
        {
            s += src[i];
        }

        return s;
    }

    /// <summary><c>Mat::dot</c> for <c>double</c> (<c>dotProd_</c>).</summary>
    private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double r = 0;
        int i = 0;
        for (; i <= a.Length - 4; i += 4)
        {
            r += (a[i] * b[i]) + (a[i + 1] * b[i + 1]) + (a[i + 2] * b[i + 2]) + (a[i + 3] * b[i + 3]);
        }

        for (; i < a.Length; i++)
        {
            r += a[i] * b[i];
        }

        return r;
    }

    /// <summary><c>norm(src, NORM_INF)</c>.</summary>
    private static double NormInf(ReadOnlySpan<double> src)
    {
        double m = 0;
        for (int i = 0; i < src.Length; i++)
        {
            double v = Math.Abs(src[i]);
            m = m < v ? v : m;
        }

        return m;
    }
}
