// This implementation is ported from the OpenCV two-view triangulation and pose recovery.
//
// Original sources:
//   modules/geometry/src/triangulate.cpp (triangulatePoints)
//   modules/geometry/src/five-point.cpp (recoverPose, decomposeEssentialMat)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using static Geometrikal.OpenCV.Managed.Core.Internal.LapackScalar;

/// <summary>Triangulation, essential matrix decomposition and the chirality test.</summary>
internal static class TriangulationKernels
{
    /// <summary>
    /// <c>triangulatePoints</c> for one pair: the DLT system solved with <c>hal::SVD64f</c>.
    /// </summary>
    /// <param name="p1">First 3 × 4 projection matrix, row-major.</param>
    /// <param name="p2">Second 3 × 4 projection matrix, row-major.</param>
    /// <param name="result">Receives the homogeneous point (4 values).</param>
    public static void TriangulateOne(
        ReadOnlySpan<double> p1, ReadOnlySpan<double> p2, double x1, double y1, double x2, double y2, Span<double> result)
    {
        Span<double> a = stackalloc double[16];
        for (int k = 0; k < 4; k++)
        {
            a[(k * 4) + 0] = (x1 * p1[8 + k]) - p1[k];
            a[(k * 4) + 1] = (y1 * p1[8 + k]) - p1[4 + k];
            a[(k * 4) + 2] = (x2 * p2[8 + k]) - p2[k];
            a[(k * 4) + 3] = (y2 * p2[8 + k]) - p2[4 + k];
        }

        JacobiSvd4LastVector(a, result);
    }

    /// <summary>
    /// <see cref="LapackDecomp.JacobiSvd{T}"/> for a <c>double</c> 4 × 4 matrix with
    /// <c>m = n = n1 = 4</c>; writes only the last row of <c>Vᵀ</c>.
    /// </summary>
    /// <remarks>The left singular vectors and the singular values are not computed after the sort because no caller reads them.</remarks>
    private static void JacobiSvd4LastVector(Span<double> at, Span<double> result)
    {
        const double eps = DblEpsilon * 10;
        ref double a0 = ref MemoryMarshal.GetReference(at[..16]);
        Span<double> vtBuf = stackalloc double[16];
        vtBuf.Clear();
        ref double v0 = ref MemoryMarshal.GetReference(vtBuf);
        Span<double> wBuf = stackalloc double[4];
        ref double w0 = ref MemoryMarshal.GetReference(wBuf);

        for (int i = 0; i < 4; i++)
        {
            double sd = 0;
            for (int k = 0; k < 4; k++)
            {
                double t = Unsafe.Add(ref a0, (i * 4) + k);
                sd += t * t;
            }

            Unsafe.Add(ref w0, i) = sd;
            Unsafe.Add(ref v0, i * 5) = 1.0;
        }

        for (int iter = 0; iter < 30; iter++)
        {
            bool changed = false;
            for (int i = 0; i < 3; i++)
            {
                for (int j = i + 1; j < 4; j++)
                {
                    ref double ai = ref Unsafe.Add(ref a0, i * 4);
                    ref double aj = ref Unsafe.Add(ref a0, j * 4);
                    double a = Unsafe.Add(ref w0, i), p = 0, b = Unsafe.Add(ref w0, j);
                    p += ai * aj;
                    p += Unsafe.Add(ref ai, 1) * Unsafe.Add(ref aj, 1);
                    p += Unsafe.Add(ref ai, 2) * Unsafe.Add(ref aj, 2);
                    p += Unsafe.Add(ref ai, 3) * Unsafe.Add(ref aj, 3);

                    if (Math.Abs(p) <= eps * Math.Sqrt(a * b))
                    {
                        continue;
                    }

                    p *= 2;
                    double beta = a - b, gamma = LapackScalar.Hypot(p, beta);
                    double c, s;
                    if (beta < 0)
                    {
                        double delta = (gamma - beta) * 0.5;
                        s = Math.Sqrt(delta / gamma);
                        c = p / (gamma * s * 2);
                    }
                    else
                    {
                        c = Math.Sqrt((gamma + beta) / (gamma * 2));
                        s = p / (gamma * c * 2);
                    }

                    a = b = 0;
                    for (int k = 0; k < 4; k++)
                    {
                        ref double aik = ref Unsafe.Add(ref ai, k);
                        ref double ajk = ref Unsafe.Add(ref aj, k);
                        double t0 = (c * aik) + (s * ajk);
                        double t1 = (-s * aik) + (c * ajk);
                        aik = t0;
                        ajk = t1;
                        a += t0 * t0;
                        b += t1 * t1;
                    }

                    Unsafe.Add(ref w0, i) = a;
                    Unsafe.Add(ref w0, j) = b;
                    changed = true;

                    ref double vi = ref Unsafe.Add(ref v0, i * 4);
                    ref double vj = ref Unsafe.Add(ref v0, j * 4);
                    for (int k = 0; k < 4; k++)
                    {
                        ref double vik = ref Unsafe.Add(ref vi, k);
                        ref double vjk = ref Unsafe.Add(ref vj, k);
                        double t0 = (c * vik) + (s * vjk);
                        double t1 = (-s * vik) + (c * vjk);
                        vik = t0;
                        vjk = t1;
                    }
                }
            }

            if (!changed)
            {
                break;
            }
        }

        for (int i = 0; i < 4; i++)
        {
            double sd = 0;
            for (int k = 0; k < 4; k++)
            {
                double t = Unsafe.Add(ref a0, (i * 4) + k);
                sd += t * t;
            }

            Unsafe.Add(ref w0, i) = Math.Sqrt(sd);
        }

        Span<int> order = [0, 1, 2, 3];
        for (int i = 0; i < 3; i++)
        {
            int j = i;
            for (int k = i + 1; k < 4; k++)
            {
                if (wBuf[j] < wBuf[k])
                {
                    j = k;
                }
            }

            if (i != j)
            {
                (wBuf[i], wBuf[j]) = (wBuf[j], wBuf[i]);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }

        vtBuf.Slice(order[3] * 4, 4).CopyTo(result);
    }

    /// <summary>
    /// <c>triangulatePoints</c> for <c>double</c> coordinates; writes a 4 × N matrix.
    /// </summary>
    public static Mat<double> Triangulate(
        ReadOnlySpan<double> p1, ReadOnlySpan<double> p2, ReadOnlySpan<Point2d> pts1, ReadOnlySpan<Point2d> pts2)
    {
        var q = new Mat<double>(pts1.Length, 4);
        TriangulateInto(p1, p2, pts1, pts2, q);
        return q;
    }

    /// <summary><see cref="Triangulate"/> into the N × 4 matrix <paramref name="q"/>; every element is written.</summary>
    private static void TriangulateInto(
        ReadOnlySpan<double> p1, ReadOnlySpan<double> p2, ReadOnlySpan<Point2d> pts1, ReadOnlySpan<Point2d> pts2, Mat<double> q)
    {
        int n = pts1.Length;
        Span<double> r = stackalloc double[4];
        Span<double> q0 = q.GetRowSpan(0), q1 = q.GetRowSpan(1), q2 = q.GetRowSpan(2), q3 = q.GetRowSpan(3);
        for (int i = 0; i < n; i++)
        {
            TriangulateOne(p1, p2, pts1[i].X, pts1[i].Y, pts2[i].X, pts2[i].Y, r);
            q0[i] = r[0];
            q1[i] = r[1];
            q2[i] = r[2];
            q3[i] = r[3];
        }
    }

    /// <summary><c>decomposeEssentialMat</c> for a <c>double</c> 3 × 3 matrix.</summary>
    public static void DecomposeEssential(Mat<double> e, out Mat<double> r1, out Mat<double> r2, out Mat<double> t)
    {
        var svd = new Svd<double>(e);
        Mat<double> u = svd.U.Clone();
        Mat<double> vt = svd.Vt.Clone();
        if (LinearAlgebra.Determinant(u) < 0)
        {
            NegateInPlace(u);
        }

        if (LinearAlgebra.Determinant(vt) < 0)
        {
            NegateInPlace(vt);
        }

        var w = new Mat<double>(new double[] { 0, 1, 0, -1, 0, 0, 0, 0, 1 }, 3, 3);
        r1 = LinearAlgebra.Gemm(LinearAlgebra.Gemm(u, w, 1.0, null, 0.0), vt, 1.0, null, 0.0);
        r2 = LinearAlgebra.Gemm(LinearAlgebra.Gemm(u, w, 1.0, null, 0.0, GemmFlags.BTranspose), vt, 1.0, null, 0.0);
        t = new Mat<double>(1, 3);
        for (int i = 0; i < 3; i++)
        {
            t.At(i, 0) = u.At(i, 2) + 0.0;
        }
    }

    /// <summary>
    /// The <c>recoverPose</c> chirality test on normalized coordinates.
    /// </summary>
    /// <param name="inputMask">Optional mask; it is combined with every candidate mask by bitwise AND.</param>
    /// <param name="mask">Receives the mask of the selected candidate (255 or the input mask value for a point in front of both cameras).</param>
    /// <param name="triangulated">Receives the 4 × N homogeneous points of the selected candidate when requested.</param>
    /// <returns>The number of points that pass the test for the selected candidate.</returns>
    public static int RecoverPose(
        Mat<double> e,
        ReadOnlySpan<Point2d> pts1,
        ReadOnlySpan<Point2d> pts2,
        double distanceThresh,
        ReadOnlySpan<byte> inputMask,
        bool wantTriangulated,
        out Mat<double> r,
        out Mat<double> t,
        out byte[] mask,
        out Mat<double>? triangulated)
    {
        int n = pts1.Length;
        DecomposeEssential(e, out Mat<double> r1, out Mat<double> r2, out Mat<double> tt);
        Span<double> tNeg = stackalloc double[3];
        for (int i = 0; i < 3; i++)
        {
            tNeg[i] = 0.0 - tt.At(i, 0);
        }

        double[] p0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
        var masks = new byte[4][];
        var counts = new int[4];
        var tris = new Mat<double>?[4];
        for (int c = 0; c < 4; c++)
        {
            Mat<double> rc = (c & 1) == 0 ? r1 : r2;
            var p = new Mat<double>(4, 3);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    p.At(i, j) = rc.At(i, j) + 0.0;
                }

                p.At(i, 3) = c < 2 ? tt.At(i, 0) + 0.0 : tNeg[i];
            }

            double[] pArr = p.ToArray();
            using PooledMat<double> qPooled = PooledMat<double>.Rent(n, 4, 1);
            Mat<double> q = qPooled.Mat;
            TriangulateInto(p0, pArr, pts1, pts2, q);
            if (wantTriangulated)
            {
                tris[c] = q.Clone();
            }

            byte[] m = new byte[n];
            Span<double> q0 = q.GetRowSpan(0), q1 = q.GetRowSpan(1), q2 = q.GetRowSpan(2), q3 = q.GetRowSpan(3);
            for (int i = 0; i < n; i++)
            {
                m[i] = q2[i] * q3[i] > 0 ? (byte)255 : (byte)0;
            }

            for (int i = 0; i < n; i++)
            {
                q0[i] /= q3[i];
            }

            for (int i = 0; i < n; i++)
            {
                q1[i] /= q3[i];
            }

            for (int i = 0; i < n; i++)
            {
                q2[i] /= q3[i];
            }

            for (int i = 0; i < n; i++)
            {
                q3[i] /= q3[i];
            }

            for (int i = 0; i < n; i++)
            {
                m[i] &= q2[i] < distanceThresh ? (byte)255 : (byte)0;
            }

            using PooledMat<double> pqPooled = PooledMat<double>.Rent(n, 3, 1);
            Mat<double> pq = pqPooled.Mat;
            pq.Clear();
            GemmKernels.Run(p, q, 1.0, null, 0.0, pq, 0, 4);
            ReadOnlySpan<double> z = pq.GetReadOnlyRowSpan(2);
            for (int i = 0; i < n; i++)
            {
                m[i] &= z[i] > 0 ? (byte)255 : (byte)0;
                m[i] &= z[i] < distanceThresh ? (byte)255 : (byte)0;
            }

            if (!inputMask.IsEmpty)
            {
                for (int i = 0; i < n; i++)
                {
                    m[i] &= inputMask[i];
                }
            }

            int good = 0;
            for (int i = 0; i < n; i++)
            {
                good += m[i] != 0 ? 1 : 0;
            }

            masks[c] = m;
            counts[c] = good;
        }

        int g1 = counts[0], g2 = counts[1], g3 = counts[2], g4 = counts[3];
        int sel = g1 >= g2 && g1 >= g3 && g1 >= g4 ? 0
            : g2 >= g1 && g2 >= g3 && g2 >= g4 ? 1
            : g3 >= g1 && g3 >= g2 && g3 >= g4 ? 2
            : 3;
        r = (sel & 1) == 0 ? r1 : r2;
        if (sel < 2)
        {
            t = tt;
        }
        else
        {
            t = new Mat<double>(1, 3);
            for (int i = 0; i < 3; i++)
            {
                t.At(i, 0) = tNeg[i];
            }
        }

        mask = masks[sel];
        triangulated = tris[sel];
        return counts[sel];
    }

    /// <summary>The normalization of <c>findEssentialMat</c> and <c>recoverPose</c>: <c>(x - c)/f</c> in the MatExpr order.</summary>
    public static void Normalize(ReadOnlySpan<Point2d> pts, Mat<double> cameraMatrix, Span<Point2d> dst)
    {
        double fx = cameraMatrix.At(0, 0), fy = cameraMatrix.At(1, 1);
        double cx = cameraMatrix.At(0, 2), cy = cameraMatrix.At(1, 2);
        double ax = 1.0 / fx, ay = 1.0 / fy;
        double bx = -cx * ax, by = -cy * ay;
        for (int i = 0; i < pts.Length; i++)
        {
            Point2d p = pts[i];
            dst[i] = new Point2d((p.X * ax) + bx, (p.Y * ay) + by);
        }
    }

    private static void NegateInPlace(Mat<double> m)
    {
        for (int i = 0; i < m.Height; i++)
        {
            Span<double> row = m.GetRowSpan(i);
            for (int j = 0; j < row.Length; j++)
            {
                row[j] = (row[j] * -1.0) + 0.0;
            }
        }
    }
}
