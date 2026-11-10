// This implementation is ported from the OpenCV EPnP pose solver.
//
// Original sources:
//   modules/geometry/src/epnp.h (epnp)
//   modules/geometry/src/epnp.cpp (epnp::compute_pose)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// EPnP pose estimation (<c>cv::epnp</c>): four control points, three beta approximations with
/// Gauss-Newton refinement, and selection of the pose with the smallest reprojection error.
/// </summary>
internal sealed class Epnp
{
    private readonly double _uc;
    private readonly double _vc;
    private readonly double _fu;
    private readonly double _fv;
    private readonly int _n;
    private readonly double[] _pws;
    private readonly double[] _us;
    private readonly double[] _alphas;
    private readonly double[] _pcs;
    private readonly double[] _cws = new double[12];
    private readonly double[] _ccs = new double[12];
    private readonly double[] _a1 = new double[6];
    private readonly double[] _a2 = new double[6];

    private Epnp(double fx, double fy, double cx, double cy, ReadOnlySpan<Point3d> objectPoints, ReadOnlySpan<Point2d> normalizedPoints)
    {
        _uc = cx;
        _vc = cy;
        _fu = fx;
        _fv = fy;
        _n = objectPoints.Length;
        _pws = new double[3 * _n];
        _us = new double[2 * _n];
        _alphas = new double[4 * _n];
        _pcs = new double[3 * _n];
        for (int i = 0; i < _n; i++)
        {
            _pws[3 * i] = objectPoints[i].X;
            _pws[(3 * i) + 1] = objectPoints[i].Y;
            _pws[(3 * i) + 2] = objectPoints[i].Z;
            _us[2 * i] = (normalizedPoints[i].X * _fu) + _uc;
            _us[(2 * i) + 1] = (normalizedPoints[i].Y * _fv) + _vc;
        }
    }

    /// <summary>OpenCV <c>epnp::compute_pose</c>.</summary>
    /// <param name="fx">Focal length in x (camera matrix element (0, 0)).</param>
    /// <param name="fy">Focal length in y (camera matrix element (1, 1)).</param>
    /// <param name="cx">Principal point x (camera matrix element (0, 2)).</param>
    /// <param name="cy">Principal point y (camera matrix element (1, 2)).</param>
    /// <param name="objectPoints">Object points. The count must be equal to the count of <paramref name="normalizedPoints"/>.</param>
    /// <param name="normalizedPoints">Undistorted, normalized image points.</param>
    /// <param name="rotation">Receives the row-major 3 × 3 rotation.</param>
    /// <param name="translation">Receives the translation.</param>
    public static void ComputePose(
        double fx,
        double fy,
        double cx,
        double cy,
        ReadOnlySpan<Point3d> objectPoints,
        ReadOnlySpan<Point2d> normalizedPoints,
        Span<double> rotation,
        Span<double> translation)
    {
        if (objectPoints.Length != normalizedPoints.Length)
        {
            throw new ArgumentException("The object and image point counts must be equal.", nameof(normalizedPoints));
        }

        if (rotation.Length < 9 || translation.Length < 3)
        {
            throw new ArgumentException("The rotation must hold 9 values and the translation 3 values.");
        }

        new Epnp(fx, fy, cx, cy, objectPoints, normalizedPoints).ComputePose(rotation, translation);
    }

    private static double Dot(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2)
        => (v1[0] * v2[0]) + (v1[1] * v2[1]) + (v1[2] * v2[2]);

    private static double Dist2(ReadOnlySpan<double> p1, ReadOnlySpan<double> p2)
        => ((p1[0] - p2[0]) * (p1[0] - p2[0])) +
           ((p1[1] - p2[1]) * (p1[1] - p2[1])) +
           ((p1[2] - p2[2]) * (p1[2] - p2[2]));

    private static void ComputeL6x10(double[] ut, double[] l6x10)
    {
        Span<double> dv = stackalloc double[4 * 6 * 3];
        for (int i = 0; i < 4; i++)
        {
            int v = 12 * (11 - i);
            int a = 0, b = 1;
            for (int j = 0; j < 6; j++)
            {
                int o = (((i * 6) + j) * 3);
                dv[o] = ut[v + (3 * a)] - ut[v + (3 * b)];
                dv[o + 1] = ut[v + (3 * a) + 1] - ut[v + (3 * b) + 1];
                dv[o + 2] = ut[v + (3 * a) + 2] - ut[v + (3 * b) + 2];

                b++;
                if (b > 3)
                {
                    a++;
                    b = a + 1;
                }
            }
        }

        for (int i = 0; i < 6; i++)
        {
            ReadOnlySpan<double> d0 = dv.Slice(((0 * 6) + i) * 3, 3);
            ReadOnlySpan<double> d1 = dv.Slice(((1 * 6) + i) * 3, 3);
            ReadOnlySpan<double> d2 = dv.Slice(((2 * 6) + i) * 3, 3);
            ReadOnlySpan<double> d3 = dv.Slice(((3 * 6) + i) * 3, 3);
            int r = 10 * i;
            l6x10[r] = Dot(d0, d0);
            l6x10[r + 1] = 2.0 * Dot(d0, d1);
            l6x10[r + 2] = Dot(d1, d1);
            l6x10[r + 3] = 2.0 * Dot(d0, d2);
            l6x10[r + 4] = 2.0 * Dot(d1, d2);
            l6x10[r + 5] = Dot(d2, d2);
            l6x10[r + 6] = 2.0 * Dot(d0, d3);
            l6x10[r + 7] = 2.0 * Dot(d1, d3);
            l6x10[r + 8] = 2.0 * Dot(d2, d3);
            l6x10[r + 9] = Dot(d3, d3);
        }
    }

    /// <summary>Solves the 6 × <paramref name="cols"/> subsystem of <c>L_6x10</c> with <c>cv::solve(DECOMP_SVD)</c>.</summary>
    private static double[] SolveSubset(double[] l6x10, double[] rho, ReadOnlySpan<int> cols)
    {
        var l = new Mat<double>(cols.Length, 6);
        for (int i = 0; i < 6; i++)
        {
            Span<double> row = l.GetRowSpan(i);
            for (int j = 0; j < cols.Length; j++)
            {
                row[j] = l6x10[(10 * i) + cols[j]];
            }
        }

        LinearAlgebra.Solve(l, new Mat<double>(rho, 1, 6), out Mat<double> x, DecompTypes.Svd);
        return x.GetReadOnlySpan().ToArray();
    }

    private static void FindBetasApprox1(double[] l6x10, double[] rho, Span<double> betas)
    {
        double[] b4 = SolveSubset(l6x10, rho, [0, 1, 3, 6]);
        if (b4[0] < 0)
        {
            betas[0] = Math.Sqrt(-b4[0]);
            betas[1] = -b4[1] / betas[0];
            betas[2] = -b4[2] / betas[0];
            betas[3] = -b4[3] / betas[0];
        }
        else
        {
            betas[0] = Math.Sqrt(b4[0]);
            betas[1] = b4[1] / betas[0];
            betas[2] = b4[2] / betas[0];
            betas[3] = b4[3] / betas[0];
        }
    }

    private static void FindBetasApprox2(double[] l6x10, double[] rho, Span<double> betas)
    {
        double[] b3 = SolveSubset(l6x10, rho, [0, 1, 2]);
        if (b3[0] < 0)
        {
            betas[0] = Math.Sqrt(-b3[0]);
            betas[1] = (b3[2] < 0) ? Math.Sqrt(-b3[2]) : 0.0;
        }
        else
        {
            betas[0] = Math.Sqrt(b3[0]);
            betas[1] = (b3[2] > 0) ? Math.Sqrt(b3[2]) : 0.0;
        }

        if (b3[1] < 0)
        {
            betas[0] = -betas[0];
        }

        betas[2] = 0.0;
        betas[3] = 0.0;
    }

    private static void FindBetasApprox3(double[] l6x10, double[] rho, Span<double> betas)
    {
        double[] b5 = SolveSubset(l6x10, rho, [0, 1, 2, 3, 4]);
        if (b5[0] < 0)
        {
            betas[0] = Math.Sqrt(-b5[0]);
            betas[1] = (b5[2] < 0) ? Math.Sqrt(-b5[2]) : 0.0;
        }
        else
        {
            betas[0] = Math.Sqrt(b5[0]);
            betas[1] = (b5[2] > 0) ? Math.Sqrt(b5[2]) : 0.0;
        }

        if (b5[1] < 0)
        {
            betas[0] = -betas[0];
        }

        betas[2] = b5[3] / betas[0];
        betas[3] = 0.0;
    }

    private static void ComputeAAndBGaussNewton(double[] l6x10, double[] rho, ReadOnlySpan<double> betas, Span<double> a, Span<double> b)
    {
        for (int i = 0; i < 6; i++)
        {
            ReadOnlySpan<double> rowL = l6x10.AsSpan(10 * i, 10);
            Span<double> rowA = a.Slice(4 * i, 4);

            rowA[0] = (2 * rowL[0] * betas[0]) + (rowL[1] * betas[1]) + (rowL[3] * betas[2]) + (rowL[6] * betas[3]);
            rowA[1] = (rowL[1] * betas[0]) + (2 * rowL[2] * betas[1]) + (rowL[4] * betas[2]) + (rowL[7] * betas[3]);
            rowA[2] = (rowL[3] * betas[0]) + (rowL[4] * betas[1]) + (2 * rowL[5] * betas[2]) + (rowL[8] * betas[3]);
            rowA[3] = (rowL[6] * betas[0]) + (rowL[7] * betas[1]) + (rowL[8] * betas[2]) + (2 * rowL[9] * betas[3]);

            b[i] = rho[i] -
                ((rowL[0] * betas[0] * betas[0]) +
                 (rowL[1] * betas[0] * betas[1]) +
                 (rowL[2] * betas[1] * betas[1]) +
                 (rowL[3] * betas[0] * betas[2]) +
                 (rowL[4] * betas[1] * betas[2]) +
                 (rowL[5] * betas[2] * betas[2]) +
                 (rowL[6] * betas[0] * betas[3]) +
                 (rowL[7] * betas[1] * betas[3]) +
                 (rowL[8] * betas[2] * betas[3]) +
                 (rowL[9] * betas[3] * betas[3]));
        }
    }

    private void ComputePose(Span<double> rotation, Span<double> translation)
    {
        ChooseControlPoints();
        ComputeBarycentricCoordinates();

        Mat<double> mtm;
        using (PooledMat<double> m = PooledMat<double>.Rent(12, 2 * _n, 1))
        {
            m.Mat.Clear();
            for (int i = 0; i < _n; i++)
            {
                FillM(m.Mat, 2 * i, _alphas.AsSpan(4 * i, 4), _us[2 * i], _us[(2 * i) + 1]);
            }

            mtm = LinearAlgebra.MulTransposed(m.Mat, true);
        }

        LinearAlgebra.SVDecomp(mtm, out _, out Mat<double> u, out _, SvdFlags.ModifyA);
        double[] ut = new double[12 * 12];
        for (int r = 0; r < 12; r++)
        {
            for (int c = 0; c < 12; c++)
            {
                ut[(r * 12) + c] = u.At(c, r);
            }
        }

        double[] l6x10 = new double[6 * 10];
        double[] rho = new double[6];
        ComputeL6x10(ut, l6x10);
        ComputeRho(rho);

        Span<double> betas = stackalloc double[4 * 4];
        Span<double> repErrors = stackalloc double[4];
        Span<double> rs = stackalloc double[4 * 9];
        Span<double> ts = stackalloc double[4 * 3];
        betas.Clear();
        repErrors.Clear();
        rs.Clear();
        ts.Clear();

        FindBetasApprox1(l6x10, rho, betas.Slice(4, 4));
        GaussNewton(l6x10, rho, betas.Slice(4, 4));
        repErrors[1] = ComputeRAndT(ut, betas.Slice(4, 4), rs.Slice(9, 9), ts.Slice(3, 3));

        FindBetasApprox2(l6x10, rho, betas.Slice(8, 4));
        GaussNewton(l6x10, rho, betas.Slice(8, 4));
        repErrors[2] = ComputeRAndT(ut, betas.Slice(8, 4), rs.Slice(18, 9), ts.Slice(6, 3));

        FindBetasApprox3(l6x10, rho, betas.Slice(12, 4));
        GaussNewton(l6x10, rho, betas.Slice(12, 4));
        repErrors[3] = ComputeRAndT(ut, betas.Slice(12, 4), rs.Slice(27, 9), ts.Slice(9, 3));

        int best = 1;
        if (repErrors[2] < repErrors[1])
        {
            best = 2;
        }

        if (repErrors[3] < repErrors[best])
        {
            best = 3;
        }

        ts.Slice(3 * best, 3).CopyTo(translation);
        rs.Slice(9 * best, 9).CopyTo(rotation);
    }

    private void ChooseControlPoints()
    {
        int n = _n;
        double[] cws = _cws;
        cws[0] = cws[1] = cws[2] = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cws[j] += _pws[(3 * i) + j];
            }
        }

        for (int j = 0; j < 3; j++)
        {
            cws[j] /= n;
        }

        var pw0 = new Mat<double>(3, n);
        for (int i = 0; i < n; i++)
        {
            Span<double> row = pw0.GetRowSpan(i);
            for (int j = 0; j < 3; j++)
            {
                row[j] = _pws[(3 * i) + j] - cws[j];
            }
        }

        Mat<double> pw0tpw0 = LinearAlgebra.MulTransposed(pw0, true);
        LinearAlgebra.SVDecomp(pw0tpw0, out Mat<double> dc, out Mat<double> u, out _, SvdFlags.ModifyA);

        for (int i = 1; i < 4; i++)
        {
            double k = Math.Sqrt(dc.At(i - 1, 0) / n);
            for (int j = 0; j < 3; j++)
            {
                cws[(3 * i) + j] = cws[j] + (k * u.At(j, i - 1));
            }
        }
    }

    private void ComputeBarycentricCoordinates()
    {
        double[] cws = _cws;
        var cc = new Mat<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 1; j < 4; j++)
            {
                cc.At(i, j - 1) = cws[(3 * j) + i] - cws[i];
            }
        }

        LinearAlgebra.Invert(cc, out Mat<double> ccInv, DecompTypes.Svd);
        Span<double> ci = stackalloc double[9];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                ci[(3 * r) + c] = ccInv.At(r, c);
            }
        }

        for (int i = 0; i < _n; i++)
        {
            ReadOnlySpan<double> pi = _pws.AsSpan(3 * i, 3);
            Span<double> a = _alphas.AsSpan(4 * i, 4);

            for (int j = 0; j < 3; j++)
            {
                a[1 + j] =
                    (ci[3 * j] * (pi[0] - cws[0])) +
                    (ci[(3 * j) + 1] * (pi[1] - cws[1])) +
                    (ci[(3 * j) + 2] * (pi[2] - cws[2]));
            }

            a[0] = 1.0 - a[1] - a[2] - a[3];
        }
    }

    private void FillM(Mat<double> m, int row, ReadOnlySpan<double> alphas, double u, double v)
    {
        Span<double> m1 = m.GetRowSpan(row);
        Span<double> m2 = m.GetRowSpan(row + 1);

        for (int i = 0; i < 4; i++)
        {
            m1[3 * i] = alphas[i] * _fu;
            m1[(3 * i) + 1] = 0.0;
            m1[(3 * i) + 2] = alphas[i] * (_uc - u);

            m2[3 * i] = 0.0;
            m2[(3 * i) + 1] = alphas[i] * _fv;
            m2[(3 * i) + 2] = alphas[i] * (_vc - v);
        }
    }

    private void ComputeCcs(ReadOnlySpan<double> betas, double[] ut)
    {
        Array.Clear(_ccs);
        for (int i = 0; i < 4; i++)
        {
            int v = 12 * (11 - i);
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    _ccs[(3 * j) + k] += betas[i] * ut[v + (3 * j) + k];
                }
            }
        }
    }

    private void ComputePcs()
    {
        double[] ccs = _ccs;
        for (int i = 0; i < _n; i++)
        {
            ReadOnlySpan<double> a = _alphas.AsSpan(4 * i, 4);
            Span<double> pc = _pcs.AsSpan(3 * i, 3);

            for (int j = 0; j < 3; j++)
            {
                pc[j] = (a[0] * ccs[j]) + (a[1] * ccs[3 + j]) + (a[2] * ccs[6 + j]) + (a[3] * ccs[9 + j]);
            }
        }
    }

    private void ComputeRho(double[] rho)
    {
        ReadOnlySpan<double> c0 = _cws.AsSpan(0, 3);
        ReadOnlySpan<double> c1 = _cws.AsSpan(3, 3);
        ReadOnlySpan<double> c2 = _cws.AsSpan(6, 3);
        ReadOnlySpan<double> c3 = _cws.AsSpan(9, 3);
        rho[0] = Dist2(c0, c1);
        rho[1] = Dist2(c0, c2);
        rho[2] = Dist2(c0, c3);
        rho[3] = Dist2(c1, c2);
        rho[4] = Dist2(c1, c3);
        rho[5] = Dist2(c2, c3);
    }

    private void GaussNewton(double[] l6x10, double[] rho, Span<double> betas)
    {
        const int IterationsNumber = 5;

        Span<double> a = stackalloc double[6 * 4];
        Span<double> b = stackalloc double[6];

        // QrSolve keeps the previous solution when A is singular.
        Span<double> x = stackalloc double[4];
        a.Clear();
        b.Clear();
        x.Clear();

        for (int k = 0; k < IterationsNumber; k++)
        {
            ComputeAAndBGaussNewton(l6x10, rho, betas, a, b);
            QrSolve(a, 6, 4, b, x);
            for (int i = 0; i < 4; i++)
            {
                betas[i] += x[i];
            }
        }
    }

    /// <summary><c>epnp::qr_solve</c>: Householder least-squares solution of <c>A·X = b</c>. A and b are modified.</summary>
    private void QrSolve(Span<double> a, int nr, int nc, Span<double> b, Span<double> x)
    {
        double[] a1 = _a1;
        double[] a2 = _a2;
        int akk = 0;
        for (int k = 0; k < nc; k++)
        {
            int aik1 = akk;
            double eta = Math.Abs(a[aik1]);
            for (int i = k + 1; i < nr; i++)
            {
                double elt = Math.Abs(a[aik1]);
                if (eta < elt)
                {
                    eta = elt;
                }

                aik1 += nc;
            }

            if (eta == 0)
            {
                a1[k] = a2[k] = 0.0;
                return;
            }

            int aik2 = akk;
            double sum2 = 0.0, invEta = 1.0 / eta;
            for (int i = k; i < nr; i++)
            {
                a[aik2] *= invEta;
                sum2 += a[aik2] * a[aik2];
                aik2 += nc;
            }

            double sigma = Math.Sqrt(sum2);
            if (a[akk] < 0)
            {
                sigma = -sigma;
            }

            a[akk] += sigma;
            a1[k] = sigma * a[akk];
            a2[k] = -eta * sigma;
            for (int j = k + 1; j < nc; j++)
            {
                int aik = akk;
                double sum = 0;
                for (int i = k; i < nr; i++)
                {
                    sum += a[aik] * a[aik + j - k];
                    aik += nc;
                }

                double tau = sum / a1[k];
                aik = akk;
                for (int i = k; i < nr; i++)
                {
                    a[aik + j - k] -= tau * a[aik];
                    aik += nc;
                }
            }

            akk += nc + 1;
        }

        // b <- Qt b
        int ajj = 0;
        for (int j = 0; j < nc; j++)
        {
            int aij = ajj;
            double tau = 0;
            for (int i = j; i < nr; i++)
            {
                tau += a[aij] * b[i];
                aij += nc;
            }

            tau /= a1[j];
            aij = ajj;
            for (int i = j; i < nr; i++)
            {
                b[i] -= tau * a[aij];
                aij += nc;
            }

            ajj += nc + 1;
        }

        // X = R^-1 b
        x[nc - 1] = b[nc - 1] / a2[nc - 1];
        for (int i = nc - 2; i >= 0; i--)
        {
            int aij = (i * nc) + (i + 1);
            double sum = 0;
            for (int j = i + 1; j < nc; j++)
            {
                sum += a[aij] * x[j];
                aij++;
            }

            x[i] = (b[i] - sum) / a2[i];
        }
    }

    private double ComputeRAndT(double[] ut, ReadOnlySpan<double> betas, Span<double> r, Span<double> t)
    {
        ComputeCcs(betas, ut);
        ComputePcs();
        SolveForSign();
        EstimateRAndT(r, t);
        return ReprojectionError(r, t);
    }

    private void SolveForSign()
    {
        if (_pcs[2] < 0.0)
        {
            for (int i = 0; i < 12; i++)
            {
                _ccs[i] = -_ccs[i];
            }

            for (int i = 0; i < 3 * _n; i++)
            {
                _pcs[i] = -_pcs[i];
            }
        }
    }

    private void EstimateRAndT(Span<double> r, Span<double> t)
    {
        int n = _n;
        Span<double> pc0 = stackalloc double[3];
        Span<double> pw0 = stackalloc double[3];
        pc0.Clear();
        pw0.Clear();

        for (int i = 0; i < n; i++)
        {
            ReadOnlySpan<double> pc = _pcs.AsSpan(3 * i, 3);
            ReadOnlySpan<double> pw = _pws.AsSpan(3 * i, 3);
            for (int j = 0; j < 3; j++)
            {
                pc0[j] += pc[j];
                pw0[j] += pw[j];
            }
        }

        for (int j = 0; j < 3; j++)
        {
            pc0[j] /= n;
            pw0[j] /= n;
        }

        var abt = new Mat<double>(3, 3);
        Span<double> ab = abt.GetSpan();
        for (int i = 0; i < n; i++)
        {
            ReadOnlySpan<double> pc = _pcs.AsSpan(3 * i, 3);
            ReadOnlySpan<double> pw = _pws.AsSpan(3 * i, 3);
            for (int j = 0; j < 3; j++)
            {
                ab[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
                ab[(3 * j) + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
                ab[(3 * j) + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
            }
        }

        LinearAlgebra.SVDecomp(abt, out _, out Mat<double> abtU, out Mat<double> abtVt, SvdFlags.ModifyA);
        Mat<double> mR = LinearAlgebra.Gemm(abtU, abtVt, 1, null, 0);
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                r[(3 * y) + x] = mR.At(y, x);
            }
        }

        double det = LinearAlgebra.Determinant(mR);
        if (det < 0)
        {
            r[6] = -r[6];
            r[7] = -r[7];
            r[8] = -r[8];
        }

        t[0] = pc0[0] - Dot(r.Slice(0, 3), pw0);
        t[1] = pc0[1] - Dot(r.Slice(3, 3), pw0);
        t[2] = pc0[2] - Dot(r.Slice(6, 3), pw0);
    }

    private double ReprojectionError(ReadOnlySpan<double> r, ReadOnlySpan<double> t)
    {
        double sum2 = 0.0;
        ReadOnlySpan<double> r0 = r.Slice(0, 3);
        ReadOnlySpan<double> r1 = r.Slice(3, 3);
        ReadOnlySpan<double> r2 = r.Slice(6, 3);

        for (int i = 0; i < _n; i++)
        {
            ReadOnlySpan<double> pw = _pws.AsSpan(3 * i, 3);
            double xc = Dot(r0, pw) + t[0];
            double yc = Dot(r1, pw) + t[1];
            double invZc = 1.0 / (Dot(r2, pw) + t[2]);
            double ue = _uc + (_fu * xc * invZc);
            double ve = _vc + (_fv * yc * invZc);
            double u = _us[2 * i], v = _us[(2 * i) + 1];

            sum2 += Math.Sqrt(((u - ue) * (u - ue)) + ((v - ve) * (v - ve)));
        }

        return sum2 / _n;
    }
}
