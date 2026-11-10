namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Analytic and structural tests for <see cref="LinearAlgebra"/>, <see cref="Svd{T}"/> and the
/// point types: ROIs, invalid input, source immutability, thread-count independence and the
/// adjudicated cv2 difference.
/// </summary>
[Collection(nameof(CvParallelCollection))]
public sealed class LinearAlgebraBasicTests
{
    [Fact]
    public void Gemm_Identity_ReturnsOperand()
    {
        Mat<double> a = Random(5, 5, 1);
        var id = new Mat<double>(5, 5);
        LinearAlgebra.SetIdentity(id);
        Assert.Equal(a.ToArray(), LinearAlgebra.Gemm(a, id, 1, null, 0).ToArray());
        Assert.Equal(a.ToArray(), LinearAlgebra.Gemm(id, a, 1, null, 0).ToArray());
    }

    [Theory]
    [InlineData(4, 3, 5, GemmFlags.None)]
    [InlineData(9, 9, 9, GemmFlags.ATranspose | GemmFlags.BTranspose)]
    [InlineData(120, 70, 45, GemmFlags.None)]
    [InlineData(120, 70, 45, GemmFlags.ATranspose)]
    [InlineData(120, 70, 45, GemmFlags.BTranspose)]
    [InlineData(130, 33, 71, GemmFlags.ATranspose | GemmFlags.BTranspose | GemmFlags.CTranspose)]
    public void Gemm_RoiInputs_MatchPackedInputs(int rows, int len, int cols, GemmFlags flags)
    {
        bool aT = (flags & GemmFlags.ATranspose) != 0, bT = (flags & GemmFlags.BTranspose) != 0, cT = (flags & GemmFlags.CTranspose) != 0;
        Mat<float> a = RandomF(aT ? rows : len, aT ? len : rows, 2);
        Mat<float> b = RandomF(bT ? len : cols, bT ? cols : len, 3);
        Mat<float> c = RandomF(cT ? rows : cols, cT ? cols : rows, 4);
        Mat<float> expected = LinearAlgebra.Gemm(a, b, 0.75, c, -0.5, flags);

        Mat<float> aRoi = Embed(a), bRoi = Embed(b), cRoi = Embed(c);
        Mat<float> actual = LinearAlgebra.Gemm(aRoi, bRoi, 0.75, cRoi, -0.5, flags);
        Assert.Equal(expected.ToArray(), actual.ToArray());

        double tol = 1e-4 * len;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double s = 0;
                for (int k = 0; k < len; k++)
                {
                    s += (aT ? a.At(k, i) : a.At(i, k)) * (double)(bT ? b.At(j, k) : b.At(k, j));
                }

                double r = (0.75 * s) - (0.5 * (cT ? c.At(j, i) : c.At(i, j)));
                Assert.True(Math.Abs(r - expected.At(i, j)) < tol, $"({i},{j}) {expected.At(i, j)} vs {r}");
            }
        }
    }

    [Fact]
    public void Gemm_Large_DoesNotDependOnThreadCount()
    {
        Mat<float> a = RandomF(300, 260, 5);
        Mat<float> b = RandomF(270, 300, 6);
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            float[] one = LinearAlgebra.Gemm(a, b, 1, null, 0).ToArray();
            CvParallel.MaxDegreeOfParallelism = 8;
            float[] eight = LinearAlgebra.Gemm(a, b, 1, null, 0).ToArray();
            Assert.Equal(one, eight);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void Gemm_InvalidInput_Throws()
    {
        Mat<double> a = Random(3, 4, 1);
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Gemm(a, a, 1, null, 0));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Gemm(a, Random(4, 3, 2), 1, Random(2, 2, 3), 1));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Gemm(new Mat<double>(3, 3, 2), a, 1, null, 0));
        Assert.Throws<NotSupportedException>(() => LinearAlgebra.Gemm(new Mat<Half>(2, 2), new Mat<Half>(2, 2), 1, null, 0));
    }

    [Fact]
    public void Operations_DoNotModifySources()
    {
        Mat<double> a = Random(6, 6, 7);
        Mat<double> b = Random(1, 6, 8);
        double[] a0 = a.ToArray(), b0 = b.ToArray();
        LinearAlgebra.Solve(a, b, out _, DecompTypes.Svd);
        LinearAlgebra.Solve(a, b, out _, DecompTypes.Qr);
        LinearAlgebra.Invert(a, out _);
        LinearAlgebra.Determinant(a);
        LinearAlgebra.Eigen(a, out _, out _);
        _ = new Svd<double>(a, SvdFlags.ModifyA);
        Assert.Equal(a0, a.ToArray());
        Assert.Equal(b0, b.ToArray());
    }

    [Theory]
    [InlineData(DecompTypes.Lu)]
    [InlineData(DecompTypes.Cholesky)]
    [InlineData(DecompTypes.Qr)]
    [InlineData(DecompTypes.Svd)]
    [InlineData(DecompTypes.Eig)]
    public void Solve_RoiInputs_MatchPackedInputs(DecompTypes method)
    {
        Mat<double> m = Random(7, 7, 9);
        Mat<double> a = method is DecompTypes.Cholesky or DecompTypes.Eig ? LinearAlgebra.MulTransposed(m, true) : m;
        Mat<double> b = Random(2, 7, 10);
        Assert.True(LinearAlgebra.Solve(a, b, out Mat<double> x, method));
        Assert.True(LinearAlgebra.Solve(Embed(a), Embed(b), out Mat<double> xr, method));
        Assert.Equal(x.ToArray(), xr.ToArray());
        Mat<double> r = LinearAlgebra.Gemm(a, x, 1, b, -1);
        foreach (double v in r.ToArray())
        {
            Assert.True(Math.Abs(v) < 1e-9);
        }
    }

    [Fact]
    public void Solve_Qr_LargeSquare_MatchesLu()
    {
        Mat<double> a = Random(50, 50, 11);
        Mat<double> b = Random(1, 50, 12);
        Assert.True(LinearAlgebra.Solve(a, b, out Mat<double> xq, DecompTypes.Qr));
        Assert.True(LinearAlgebra.Solve(a, b, out Mat<double> xl, DecompTypes.Lu));
        for (int i = 0; i < 50; i++)
        {
            Assert.True(Math.Abs(xq.At(i, 0) - xl.At(i, 0)) < 1e-8 * Math.Max(1, Math.Abs(xl.At(i, 0))));
        }
    }

    [Fact]
    public void Solve_Singular_ReturnsFalseAndZero()
    {
        var a = new Mat<double>(new double[] { 1, 2, 3, 4, 2, 4, 6, 8, 1, 0, 1, 2, 5, 3, 5, 1 }, 4, 4);
        Assert.False(LinearAlgebra.Solve(a, Random(1, 4, 13), out Mat<double> x));
        Assert.All(x.ToArray(), v => Assert.Equal(0.0, v));
        Assert.Equal(0.0, LinearAlgebra.Invert(a, out Mat<double> inv));
        Assert.All(inv.ToArray(), v => Assert.Equal(0.0, v));
    }

    [Fact]
    public void Solve_InvalidInput_Throws()
    {
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Solve(Random(3, 5, 1), Random(1, 5, 2), out _));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Solve(Random(5, 3, 1), Random(1, 3, 2), out _, DecompTypes.Svd));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.Solve(Random(4, 4, 1), Random(1, 3, 2), out _));
        Assert.Throws<NotSupportedException>(() => LinearAlgebra.Solve(Random(3, 5, 1), Random(1, 5, 2), out _, DecompTypes.Eig));
    }

    [Fact]
    public void Invert_TimesSource_IsIdentity()
    {
        foreach (int n in new[] { 1, 2, 3, 4, 9 })
        {
            Mat<double> a = Random(n, n, 20 + n);
            Assert.Equal(1.0, LinearAlgebra.Invert(a, out Mat<double> inv));
            Mat<double> p = LinearAlgebra.Gemm(a, inv, 1, null, 0);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Assert.True(Math.Abs(p.At(i, j) - (i == j ? 1 : 0)) < 1e-9);
                }
            }
        }
    }

    [Fact]
    public void Determinant_KnownValues()
    {
        var id = new Mat<float>(6, 6);
        LinearAlgebra.SetIdentity(id, 2);
        Assert.Equal(64.0, LinearAlgebra.Determinant(id));
        Assert.Equal(-2.0, LinearAlgebra.Determinant(new Mat<double>(new double[] { 1, 2, 3, 4 }, 2, 2)));
    }

    [Fact]
    public void Svd_ReconstructsSourceAndIsOrthogonal()
    {
        foreach ((int rows, int cols) in new[] { (6, 4), (4, 6), (9, 9), (30, 20) })
        {
            Mat<double> a = Random(cols, rows, rows * 31 + cols);
            var svd = new Svd<double>(a, SvdFlags.FullUV);
            int nm = Math.Min(rows, cols);
            Assert.Equal(rows, svd.U.Height);
            Assert.Equal(rows, svd.U.Width);
            Assert.Equal(cols, svd.Vt.Width);
            Assert.Equal(cols, svd.Vt.Height);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double s = 0;
                    for (int k = 0; k < nm; k++)
                    {
                        s += svd.U.At(i, k) * svd.W.At(k, 0) * svd.Vt.At(k, j);
                    }

                    Assert.True(Math.Abs(s - a.At(i, j)) < 1e-12);
                }
            }

            Mat<double> utu = LinearAlgebra.MulTransposed(svd.U, true);
            for (int i = 0; i < rows; i++)
            {
                Assert.True(Math.Abs(utu.At(i, i) - 1) < 1e-12);
            }
        }
    }

    [Fact]
    public void Svd_NoUV_HasNoVectors()
    {
        var svd = new Svd<float>(RandomF(3, 5, 3), SvdFlags.NoUV);
        Assert.False(svd.HasVectors);
        Assert.Equal(3, svd.W.Height);
        Assert.Throws<InvalidOperationException>(() => svd.BackSubst(null));
        Assert.Equal(svd.W.ToArray(), Svd<float>.ComputeValues(RandomF(3, 5, 3)).ToArray());
    }

    [Fact]
    public void Svd_SolveZ_FindsNullVector()
    {
        // Rows orthogonal to (1, 2, 3, 4) / |.|.
        var a = new Mat<double>(new double[] { 2, -1, 0, 0, 3, 0, -1, 0, 4, 0, 0, -1, 0, 3, -2, 0 }, 4, 4);
        Mat<double> z = Svd<double>.SolveZ(a);
        double norm = Math.Sqrt(30);
        double sign = Math.Sign(z.At(0, 0));
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs((sign * z.At(i, 0)) - ((i + 1) / norm)) < 1e-12);
        }
    }

    [Fact]
    public void Eigen_DiagonalMatrix_SortsDescending()
    {
        var a = new Mat<double>(3, 3);
        a.At(0, 0) = 1;
        a.At(1, 1) = 5;
        a.At(2, 2) = 3;
        LinearAlgebra.Eigen(a, out Mat<double> w, out Mat<double> v);
        Assert.Equal(new[] { 5.0, 3, 1 }, w.ToArray());
        Assert.Equal(1.0, v.At(0, 1));
        Assert.Equal(1.0, v.At(1, 2));
        Assert.Equal(1.0, v.At(2, 0));
    }

    [Fact]
    public void SolveCubic_And_SolvePoly_KnownRoots()
    {
        double[] r = new double[3];
        Assert.Equal(3, LinearAlgebra.SolveCubic(new double[] { 1, -6, 11, -6 }, r));
        Array.Sort(r);
        Assert.Equal(new[] { 1.0, 2, 3 }, r, new Tol(1e-12));
        Assert.Equal(-1, LinearAlgebra.SolveCubic(new double[] { 0, 0, 0, 0 }, r));

        var roots = new Complex[2];
        LinearAlgebra.SolvePoly(new double[] { 1, 0, 1 }, roots);
        Assert.True(Math.Abs(Math.Abs(roots[0].Imaginary) - 1) < 1e-12);
        Assert.True(Math.Abs(roots[0].Real) < 1e-12);
    }

    [Fact]
    public void PerspectiveTransform_IdentityAndZeroW()
    {
        var h = new Mat<double>(3, 3);
        LinearAlgebra.SetIdentity(h);
        var pts = new[] { new Point2f(1.5f, -2), new Point2f(0, 0) };
        Assert.Equal(pts, LinearAlgebra.PerspectiveTransform(pts, h));
        h.At(2, 2) = 0;
        Assert.Equal(new[] { new Point2f(0, 0), new Point2f(0, 0) }, LinearAlgebra.PerspectiveTransform(pts, h));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.PerspectiveTransform(pts, new Mat<double>(3, 2)));
        Assert.Throws<ArgumentException>(() => LinearAlgebra.PerspectiveTransform(pts, new Point2f[1], h));
    }

    [Fact]
    public void PerspectiveTransform_ManyPoints_MatchesScalarFormula()
    {
        var h = new Mat<double>(3, 3);
        double[] m = [1.1, -0.2, 3.5, 0.3, 0.9, -7.25, 0.001, -0.002, 1];
        for (int i = 0; i < 9; i++)
        {
            h.At(i / 3, i % 3) = m[i];
        }

        var rng = new Random(5);
        var pts = new Point2f[1003];
        for (int i = 0; i < pts.Length; i++)
        {
            pts[i] = new Point2f((float)((rng.NextDouble() * 2000) - 1000), (float)((rng.NextDouble() * 2000) - 1000));
        }

        // w = 0 exactly: 0.001·1000 − 0.002·1000 + 1.
        pts[5] = new Point2f(1000, 1000);
        pts[10] = new Point2f(float.NaN, 1);
        Point2f[] actual = LinearAlgebra.PerspectiveTransform(pts, h);
        for (int i = 0; i < pts.Length; i++)
        {
            double x = pts[i].X, y = pts[i].Y;
            double w = (x * m[6]) + (y * m[7]) + m[8];
            Point2f expected = default;
            if (Math.Abs(w) > 1.1920928955078125e-7)
            {
                w = 1.0 / w;
                expected = new Point2f((float)(((x * m[0]) + (y * m[1]) + m[2]) * w), (float)(((x * m[3]) + (y * m[4]) + m[5]) * w));
            }

            Assert.Equal(BitConverter.SingleToInt32Bits(expected.X), BitConverter.SingleToInt32Bits(actual[i].X));
            Assert.Equal(BitConverter.SingleToInt32Bits(expected.Y), BitConverter.SingleToInt32Bits(actual[i].Y));
        }
    }

    [Fact]
    public void Transform_RoiSource_MatchesPackedSource()
    {
        var src = new Mat<float>(5, 4, 3);
        var rng = new Random(3);
        for (int i = 0; i < 4; i++)
        {
            foreach (ref float v in src.GetRowSpan(i))
            {
                v = (float)rng.NextDouble();
            }
        }

        var parent = new Mat<float>(9, 7, 3);
        Mat<float> roi = parent.Roi(2, 1, 5, 4);
        src.CopyTo(roi);
        var m = new Mat<double>(new double[] { 0.5, 0.25, 0.125, 1, 0.1, 0.7, 0.3, -2, 0.3, 0.2, 0.9, 0.5 }, 4, 3);
        Mat<float> packed = LinearAlgebra.Transform(src, m);
        Mat<float> fromRoi = LinearAlgebra.Transform(roi, m);
        for (int y = 0; y < 4; y++)
        {
            // A packed source is one 20-point run; an ROI is 5-point rows. Only the last point of each run uses the scalar order.
            for (int x = 0; x < 4; x++)
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    Assert.True(Math.Abs(packed.At(y, x, ch) - fromRoi.At(y, x, ch)) < 1e-5);
                }
            }
        }

        Assert.Throws<NotSupportedException>(() => LinearAlgebra.Transform(new Mat<float>(3, 3), new Mat<double>(2, 1)));
    }

    [Fact]
    public void SetIdentity_Trace_CompleteSymm()
    {
        var m = new Mat<double>(4, 3);
        m.Fill(7);
        LinearAlgebra.SetIdentity(m, 2.5);
        Assert.Equal(new double[] { 2.5, 0, 0, 0, 0, 2.5, 0, 0, 0, 0, 2.5, 0 }, m.ToArray());
        Assert.Equal(7.5, LinearAlgebra.Trace(m));

        var s = new Mat<int>(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3);
        LinearAlgebra.CompleteSymm(s);
        Assert.Equal(new[] { 1, 2, 3, 2, 5, 6, 3, 6, 9 }, s.ToArray());
        var l = new Mat<int>(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 3, 3);
        LinearAlgebra.CompleteSymm(l, lowerToUpper: true);
        Assert.Equal(new[] { 1, 4, 7, 4, 5, 8, 7, 8, 9 }, l.ToArray());
    }

    [Fact]
    public void MulTransposed_Large_UsesGemmAndIsSymmetric()
    {
        Mat<double> a = Random(110, 105, 40);
        Mat<double> ata = LinearAlgebra.MulTransposed(a, true);
        Mat<double> g = LinearAlgebra.Gemm(a, a, 1, null, 0, GemmFlags.ATranspose);
        Assert.Equal(g.ToArray(), ata.ToArray());
    }

    [Fact]
    public void PointTypes_AreValueRecords()
    {
        Assert.Equal(new Point2d(1, 2), new Point2d(1, 2));
        Assert.Equal(new Point3f(1, 2, 3), new Point3f(1, 2, 3));
        Assert.NotEqual(new Point3d(1, 2, 3), new Point3d(1, 2, 4));
    }

    private static Mat<double> Random(int width, int height, int seed)
    {
        var rng = new Random(seed);
        var m = new Mat<double>(width, height);
        for (int y = 0; y < height; y++)
        {
            foreach (ref double v in m.GetRowSpan(y))
            {
                v = (rng.NextDouble() * 2) - 1;
            }
        }

        return m;
    }

    private static Mat<float> RandomF(int width, int height, int seed)
    {
        var rng = new Random(seed);
        var m = new Mat<float>(width, height);
        for (int y = 0; y < height; y++)
        {
            foreach (ref float v in m.GetRowSpan(y))
            {
                v = (float)((rng.NextDouble() * 2) - 1);
            }
        }

        return m;
    }

    private static Mat<T> Embed<T>(Mat<T> m)
        where T : unmanaged
    {
        var parent = new Mat<T>(m.Width + 5, m.Height + 3);
        Mat<T> roi = parent.Roi(3, 2, m.Width, m.Height);
        m.CopyTo(roi);
        return roi;
    }

    private sealed class Tol(double eps) : System.Collections.Generic.IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= eps;

        public int GetHashCode(double obj) => 0;
    }
}
