// This implementation is ported from the OpenCV Levenberg-Marquardt solver.
//
// Original source:
//   modules/geometry/src/levmarq.cpp (LevMarqBase::optimize, LevMarqDenseLinearBackend)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/levmarq.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Computes the residuals <paramref name="err"/> (n × 1) at <paramref name="x"/> and, when
/// <paramref name="jacobian"/> is not null, the jacobian (n × params). Returns false on failure.
/// </summary>
internal delegate bool CameraLevMarqCallback(double[] x, Mat<double> err, Mat<double>? jacobian);

/// <summary>
/// Dense Levenberg-Marquardt solver with the OpenCV 5.0 default settings, a "long" callback,
/// no mask and the SVD linear solver (OpenCV <c>LevMarq</c> with <c>LevMarqDenseLinearBackend</c>).
/// </summary>
internal sealed class CameraLevMarq
{
    private const double MinDiag = 1e-6;
    private const double MaxDiag = 1e32;
    private const double MaxLambda = 1e32;
    private const double HGeo = 1e-4;
    private const double GeoScale = 0.5;
    private const double RelEnergyDeltaTolerance = 1e-6;
    private const double MinGradientTolerance = 1e-6;
    private const double InitialLambda = 0.0001;
    private const double InitialLmUpFactor = 2.0;
    private const double InitialLmDownFactor = 3.0;

    private readonly int _nVars;
    private readonly CameraLevMarqCallback _callback;
    private readonly int _maxIterations;
    private readonly bool _geodesic;
    private readonly double _stepNormTolerance;
    private readonly double _smallEnergyTolerance;
    private readonly Mat<double> _b;
    private readonly Mat<double> _j;
    private double[] _currentX;
    private double[] _probeX;
    private Mat<double> _jtj = null!;
    private Mat<double> _jtb = null!;

    public CameraLevMarq(double[] x, int nErrs, CameraLevMarqCallback callback, int maxIterations, bool geodesic, double stepNormTolerance = 1e-6, double smallEnergyTolerance = 0)
    {
        _stepNormTolerance = stepNormTolerance;
        _smallEnergyTolerance = smallEnergyTolerance;
        _nVars = x.Length;
        _currentX = x;
        _probeX = (double[])x.Clone();
        _callback = callback;
        _maxIterations = maxIterations;
        _geodesic = geodesic;
        _b = new Mat<double>(1, nErrs);
        _j = new Mat<double>(_nVars, nErrs);
    }

    /// <summary>Runs the optimization; the parameters passed to the constructor are updated in place.</summary>
    /// <returns>True when a convergence criterion was met.</returns>
    public bool Optimize()
    {
        _probeX = (double[])_currentX.Clone();
        if (!CalcFunc(out double energy, true, true) || energy < 0)
        {
            return false;
        }

        double oldEnergy = energy;
        bool smallGradient = false, smallStep = false, smallEnergyDelta = false, smallEnergy = false;
        double lmUpFactor = InitialLmUpFactor;
        double lambda = InitialLambda;
        int iter = 0;
        bool done = false;
        int n = _nVars;
        var lmDiag = new double[n];

        while (!done)
        {
            double[] jtb = _jtb.ToArray();
            double gradientMax = Reductions.Norm(_jtb, NormType.Inf);
            var diag = new double[n];
            for (int i = 0; i < n; i++)
            {
                diag[i] = _jtj.At(i, i);
            }

            bool enoughLm = false;
            while (!enoughLm && !done)
            {
                for (int i = 0; i < n; i++)
                {
                    double d = diag[i] * lambda;
                    d = Math.Min(Math.Max(d, MinDiag), MaxDiag);
                    lmDiag[i] = d;
                    _jtj.At(i, i) = d + diag[i];
                }

                bool solved = SolveSystem(jtb, out double[] x);
                double costChange = 0, jacCostChange = 0, stepQuality = 0, xNorm = 0;
                if (solved)
                {
                    Mat<double> xm = Column(x), lm = Column(lmDiag);
                    Mat<double> d = Arithmetic.Subtract(Column(jtb), Arithmetic.Multiply(lm, xm));
                    jacCostChange = -0.5 * Reductions.Sum(Arithmetic.Multiply(xm, d))[0];
                    xNorm = Reductions.Norm(xm, NormType.L2Sqr);

                    if (_geodesic)
                    {
                        ApplyGeodesic(jtb, lmDiag, x);
                    }

                    for (int i = 0; i < n; i++)
                    {
                        _probeX[i] = _currentX[i] + x[i];
                    }

                    bool success = CalcFunc(out energy, true, false);
                    if (!success || energy < 0 || double.IsNaN(energy))
                    {
                        return false;
                    }

                    costChange = oldEnergy - energy;
                    stepQuality = costChange / jacCostChange;
                }

                if (!solved || costChange < 0)
                {
                    lambda *= lmUpFactor;
                    lmUpFactor *= 2.0;
                }
                else
                {
                    enoughLm = true;
                    lambda *= Math.Max(1.0 / InitialLmDownFactor, 1.0 - Math.Pow(2.0 * stepQuality - 1.0, 3));
                    lmUpFactor = InitialLmUpFactor;
                    smallGradient = gradientMax < MinGradientTolerance;
                    smallStep = xNorm < _stepNormTolerance;
                    smallEnergyDelta = costChange / energy < RelEnergyDeltaTolerance;
                    smallEnergy = energy < _smallEnergyTolerance;
                    Array.Copy(_probeX, _currentX, n);
                    oldEnergy = energy;
                }

                iter++;
                done = iter >= _maxIterations || lambda >= MaxLambda || smallGradient || smallStep || smallEnergyDelta || smallEnergy;
            }

            if (!done && !CalcFunc(out _, false, true))
            {
                return false;
            }
        }

        return smallGradient || smallStep || smallEnergyDelta || smallEnergy;
    }

    private bool CalcFunc(out double energy, bool calcEnergy, bool calcJacobian)
    {
        energy = 0;
        _b.Clear();
        if (calcJacobian)
        {
            _j.Clear();
        }

        if (!_callback(_probeX, _b, calcJacobian ? _j : null))
        {
            return false;
        }

        if (calcJacobian)
        {
            _jtj = LinearAlgebra.MulTransposed(_j, true);
            _jtb = LinearAlgebra.Gemm(_j, _b, 1, null, 0, GemmFlags.ATranspose);
        }

        if (calcEnergy)
        {
            energy = Reductions.Norm(_b, NormType.L2Sqr);
        }

        return true;
    }

    private bool SolveSystem(double[] right, out double[] x)
    {
        var rhs = new Mat<double>(1, right.Length);
        for (int i = 0; i < right.Length; i++)
        {
            rhs.At(i, 0) = -right[i];
        }

        bool ok = LinearAlgebra.Solve(_jtj, rhs, out Mat<double> sol, DecompTypes.Svd);
        x = sol.ToArray();
        return ok;
    }

    private void ApplyGeodesic(double[] jtb, double[] lmDiag, double[] x)
    {
        int n = _nVars;
        var geoX = new double[n];
        for (int i = 0; i < n; i++)
        {
            geoX[i] = _currentX[i] + x[i] * HGeo;
        }

        var bv = new Mat<double>(1, _b.Height);
        if (!_callback(geoX, bv, null))
        {
            return;
        }

        // MatExpr (jtbv + jtb·(h − 1) + lmDiag.mul(x, h)) / h² evaluates as two addWeighted calls.
        Mat<double> jtbv = LinearAlgebra.Gemm(_j, bv, 1, null, 0, GemmFlags.ATranspose);
        Mat<double> m1 = Arithmetic.AddWeighted(jtbv, 1, Column(jtb), HGeo - 1.0, 0);
        Mat<double> m2 = Arithmetic.Multiply(Column(lmDiag), Column(x), HGeo);
        double s = 1.0 / (HGeo * HGeo);
        double[] jtrvv = Arithmetic.AddWeighted(m1, s, m2, s, 0).ToArray();

        if (!SolveSystem(jtrvv, out double[] xgeo))
        {
            return;
        }

        double truncerr = Math.Sqrt(Dot(xgeo, xgeo) / Dot(x, x));
        if (truncerr < 1.0)
        {
            for (int i = 0; i < n; i++)
            {
                x[i] += xgeo[i] * GeoScale;
            }
        }
    }

    private static Mat<double> Column(double[] v) => new((double[])v.Clone(), 1, v.Length);

    /// <summary>OpenCV <c>dotProd_</c>: blocks of four, then single terms.</summary>
    private static double Dot(double[] a, double[] b)
    {
        int i = 0;
        double result = 0;
        for (; i <= a.Length - 4; i += 4)
        {
            result += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        }

        for (; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }

        return result;
    }
}
