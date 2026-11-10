// This implementation is ported from the OpenCV fundamental matrix estimators.
//
// Original source:
//   modules/geometry/src/fundam.cpp (run7Point, run8Point, FMEstimatorCallback)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// OpenCV <c>FMEstimatorCallback</c>: the 7-point algorithm for a sample of 7 pairs and the
/// normalized 8-point algorithm for more pairs.
/// </summary>
internal readonly struct FundamentalKernel : IMultiModelKernel<Point2f>
{
    private const double FltEpsilon = 1.1920928955078125e-07;
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <inheritdoc/>
    public int ModelPoints => 7;

    /// <inheritdoc/>
    public int MaxModels => 3;

    /// <inheritdoc/>
    public bool CheckSubset(ReadOnlySpan<Point2f> ms1, ReadOnlySpan<Point2f> ms2, int count)
        => !PointSetRegistrator.HaveCollinearPoints(ms1, count) && !PointSetRegistrator.HaveCollinearPoints(ms2, count);

    /// <inheritdoc/>
    public int RunKernel(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, Span<double> models)
        => m1.Length == 7 ? Run7Point(m1, m2, models) : Run8Point(m1, m2, models);

    /// <inheritdoc/>
    public void ComputeError(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, ReadOnlySpan<double> f, Span<float> err)
    {
        double f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4], f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8];
        int i = 0;
        if (Avx.IsSupported)
        {
            i = ComputeErrorAvx(m1, m2, f, err);
        }

        for (; i < m1.Length; i++)
        {
            double x1 = m1[i].X, y1 = m1[i].Y, x2 = m2[i].X, y2 = m2[i].Y;
            double a = (f0 * x1) + (f1 * y1) + f2;
            double b = (f3 * x1) + (f4 * y1) + f5;
            double c = (f6 * x1) + (f7 * y1) + f8;

            double s2 = 1.0 / ((a * a) + (b * b));
            double d2 = (x2 * a) + (y2 * b) + c;

            a = (f0 * x2) + (f3 * y2) + f6;
            b = (f1 * x2) + (f4 * y2) + f7;
            c = (f2 * x2) + (f5 * y2) + f8;

            double s1 = 1.0 / ((a * a) + (b * b));
            double d1 = (x1 * a) + (y1 * b) + c;

            double e1 = d1 * d1 * s1, e2 = d2 * d2 * s2;
            err[i] = (float)(e1 < e2 ? e2 : e1);
        }
    }

    /// <summary>
    /// <see cref="ComputeError"/> for 4 pairs per step with the scalar operation order.
    /// </summary>
    /// <returns>The number of pairs processed.</returns>
    private static int ComputeErrorAvx(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, ReadOnlySpan<double> f, Span<float> err)
    {
        Vector256<double> f0 = Vector256.Create(f[0]), f1 = Vector256.Create(f[1]), f2 = Vector256.Create(f[2]);
        Vector256<double> f3 = Vector256.Create(f[3]), f4 = Vector256.Create(f[4]), f5 = Vector256.Create(f[5]);
        Vector256<double> f6 = Vector256.Create(f[6]), f7 = Vector256.Create(f[7]), f8 = Vector256.Create(f[8]);
        Vector256<double> one = Vector256.Create(1.0);
        ref float p1 = ref Unsafe.As<Point2f, float>(ref MemoryMarshal.GetReference(m1));
        ref float p2 = ref Unsafe.As<Point2f, float>(ref MemoryMarshal.GetReference(m2));
        ref float pe = ref MemoryMarshal.GetReference(err);
        int n = m1.Length;
        int i = 0;
        for (; i <= n - 4; i += 4)
        {
            Vector256<float> v1 = Vector256.LoadUnsafe(ref p1, (nuint)(2 * i));
            Vector256<float> v2 = Vector256.LoadUnsafe(ref p2, (nuint)(2 * i));
            Vector256<double> lo1 = Avx.ConvertToVector256Double(v1.GetLower());
            Vector256<double> hi1 = Avx.ConvertToVector256Double(v1.GetUpper());
            Vector256<double> lo2 = Avx.ConvertToVector256Double(v2.GetLower());
            Vector256<double> hi2 = Avx.ConvertToVector256Double(v2.GetUpper());

            // Lane order is pairs 0, 2, 1, 3.
            Vector256<double> x1 = Avx.UnpackLow(lo1, hi1), y1 = Avx.UnpackHigh(lo1, hi1);
            Vector256<double> x2 = Avx.UnpackLow(lo2, hi2), y2 = Avx.UnpackHigh(lo2, hi2);

            Vector256<double> a = (f0 * x1) + (f1 * y1) + f2;
            Vector256<double> b = (f3 * x1) + (f4 * y1) + f5;
            Vector256<double> c = (f6 * x1) + (f7 * y1) + f8;
            Vector256<double> s2 = one / ((a * a) + (b * b));
            Vector256<double> d2 = (x2 * a) + (y2 * b) + c;

            a = (f0 * x2) + (f3 * y2) + f6;
            b = (f1 * x2) + (f4 * y2) + f7;
            c = (f2 * x2) + (f5 * y2) + f8;
            Vector256<double> s1 = one / ((a * a) + (b * b));
            Vector256<double> d1 = (x1 * a) + (y1 * b) + c;

            Vector256<double> e1 = d1 * d1 * s1, e2 = d2 * d2 * s2;
            Vector256<double> e = Vector256.ConditionalSelect(Avx.CompareLessThan(e1, e2), e2, e1);
            Vector128<float> ef = Avx.ConvertToVector128Single(e);
            Sse.Shuffle(ef, ef, 0b11_01_10_00).StoreUnsafe(ref pe, (nuint)i);
        }

        return i;
    }

    /// <summary>OpenCV <c>run7Point</c>.</summary>
    /// <returns>The number of models (1 to 3), or a value ≤ 0 on failure.</returns>
    public static int Run7Point(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, Span<double> fmatrix)
    {
        const int count = 7;
        if (!Normalization(m1, m2, count, out double m1cx, out double m1cy, out double m2cx, out double m2cy, out double scale1, out double scale2))
        {
            return 0;
        }

        var a = new Mat<double>(9, 7);
        for (int i = 0; i < 7; i++)
        {
            double x0 = (m1[i].X - m1cx) * scale1;
            double y0 = (m1[i].Y - m1cy) * scale1;
            double x1 = (m2[i].X - m2cx) * scale2;
            double y1 = (m2[i].Y - m2cy) * scale2;
            Span<double> r = a.GetRowSpan(i);
            r[0] = x1 * x0;
            r[1] = x1 * y0;
            r[2] = x1;
            r[3] = y1 * x0;
            r[4] = y1 * y0;
            r[5] = y1;
            r[6] = x0;
            r[7] = y0;
            r[8] = 1;
        }

        var svd = new Svd<double>(a, SvdFlags.FullUV);
        Span<double> f1 = stackalloc double[9];
        Span<double> f2 = stackalloc double[9];
        svd.Vt.GetReadOnlyRowSpan(7).CopyTo(f1);
        svd.Vt.GetReadOnlyRowSpan(8).CopyTo(f2);

        for (int i = 0; i < 9; i++)
        {
            f1[i] -= f2[i];
        }

        Span<double> c = stackalloc double[4];
        double t0 = (f2[4] * f2[8]) - (f2[5] * f2[7]);
        double t1 = (f2[3] * f2[8]) - (f2[5] * f2[6]);
        double t2 = (f2[3] * f2[7]) - (f2[4] * f2[6]);

        c[3] = (f2[0] * t0) - (f2[1] * t1) + (f2[2] * t2);

        c[2] = (f1[0] * t0) - (f1[1] * t1) + (f1[2] * t2) -
            (f1[3] * ((f2[1] * f2[8]) - (f2[2] * f2[7]))) +
            (f1[4] * ((f2[0] * f2[8]) - (f2[2] * f2[6]))) -
            (f1[5] * ((f2[0] * f2[7]) - (f2[1] * f2[6]))) +
            (f1[6] * ((f2[1] * f2[5]) - (f2[2] * f2[4]))) -
            (f1[7] * ((f2[0] * f2[5]) - (f2[2] * f2[3]))) +
            (f1[8] * ((f2[0] * f2[4]) - (f2[1] * f2[3])));

        t0 = (f1[4] * f1[8]) - (f1[5] * f1[7]);
        t1 = (f1[3] * f1[8]) - (f1[5] * f1[6]);
        t2 = (f1[3] * f1[7]) - (f1[4] * f1[6]);

        c[1] = (f2[0] * t0) - (f2[1] * t1) + (f2[2] * t2) -
            (f2[3] * ((f1[1] * f1[8]) - (f1[2] * f1[7]))) +
            (f2[4] * ((f1[0] * f1[8]) - (f1[2] * f1[6]))) -
            (f2[5] * ((f1[0] * f1[7]) - (f1[1] * f1[6]))) +
            (f2[6] * ((f1[1] * f1[5]) - (f1[2] * f1[4]))) -
            (f2[7] * ((f1[0] * f1[5]) - (f1[2] * f1[3]))) +
            (f2[8] * ((f1[0] * f1[4]) - (f1[1] * f1[3])));

        c[0] = (f1[0] * t0) - (f1[1] * t1) + (f1[2] * t2);

        Span<double> roots = stackalloc double[3];
        roots.Clear();
        int n = LinearAlgebra.SolveCubic(c, roots);
        if (n < 1 || n > 3)
        {
            return n;
        }

        Mat<double> t2t = MatrixT(scale2, m2cx, m2cy, transpose: true);
        Mat<double> tm1 = MatrixT(scale1, m1cx, m1cy, transpose: false);
        var fm = new Mat<double>(3, 3);
        for (int k = 0; k < n; k++)
        {
            Span<double> fmk = fm.GetSpan();
            double lambda = roots[k], mu = 1.0;
            double s = (f1[8] * roots[k]) + f2[8];
            if (Math.Abs(s) > DblEpsilon)
            {
                mu = 1.0 / s;
                lambda *= mu;
                fmk[8] = 1.0;
            }
            else
            {
                fmk[8] = 0.0;
            }

            for (int i = 0; i < 8; i++)
            {
                fmk[i] = (f1[i] * lambda) + (f2[i] * mu);
            }

            Mat<double> res = LinearAlgebra.Gemm(LinearAlgebra.Gemm(t2t, fm, 1.0, null, 0.0), tm1, 1.0, null, 0.0);
            Span<double> dst = fmatrix.Slice(k * 9, 9);
            res.GetReadOnlySpan().CopyTo(dst);
            if (Math.Abs(dst[8]) > FltEpsilon)
            {
                PointSetRegistrator.ScaleInPlace(dst, 1.0 / dst[8]);
            }
        }

        return n;
    }

    /// <summary>OpenCV <c>run8Point</c> over all pairs.</summary>
    /// <returns>1 on success, 0 on failure.</returns>
    public static int Run8Point(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, Span<double> fmatrix)
    {
        int count = m1.Length;
        if (!Normalization(m1, m2, count, out double m1cx, out double m1cy, out double m2cx, out double m2cy, out double scale1, out double scale2))
        {
            return 0;
        }

        var a = new Mat<double>(9, 9);
        Span<double> av = a.GetSpan();
        Span<double> r = stackalloc double[9];
        for (int i = 0; i < count; i++)
        {
            double x1 = (m1[i].X - m1cx) * scale1;
            double y1 = (m1[i].Y - m1cy) * scale1;
            double x2 = (m2[i].X - m2cx) * scale2;
            double y2 = (m2[i].Y - m2cy) * scale2;
            r[0] = x2 * x1;
            r[1] = x2 * y1;
            r[2] = x2;
            r[3] = y2 * x1;
            r[4] = y2 * y1;
            r[5] = y2;
            r[6] = x1;
            r[7] = y1;
            r[8] = 1;
            for (int p = 0; p < 9; p++)
            {
                for (int q = 0; q < 9; q++)
                {
                    av[(p * 9) + q] += 0.0 + (r[p] * r[q]);
                }
            }
        }

        LinearAlgebra.Eigen(a, out Mat<double> w, out Mat<double> v);
        ReadOnlySpan<double> wv = w.GetReadOnlySpan();
        int k = 0;
        for (; k < 9; k++)
        {
            if (Math.Abs(wv[k]) < DblEpsilon)
            {
                break;
            }
        }

        if (k < 8)
        {
            return 0;
        }

        var f0 = new Mat<double>(3, 3);
        v.GetReadOnlyRowSpan(8).CopyTo(f0.GetSpan());

        var svd = new Svd<double>(f0);
        Span<double> d = stackalloc double[9];
        d.Clear();
        d[0] = svd.W.At(0, 0);
        d[4] = svd.W.At(1, 0);
        d[8] = 0.0;

        Span<double> u = stackalloc double[9];
        Span<double> vt = stackalloc double[9];
        Copy(svd.U, u);
        Copy(svd.Vt, vt);

        Span<double> tmp = stackalloc double[9];
        Span<double> fr = stackalloc double[9];
        MatxMul(u, d, tmp);
        MatxMul(tmp, vt, fr);

        Span<double> t2t = stackalloc double[9];
        Span<double> t1 = stackalloc double[9];
        Copy(MatrixT(scale2, m2cx, m2cy, transpose: true), t2t);
        Copy(MatrixT(scale1, m1cx, m1cy, transpose: false), t1);
        MatxMul(t2t, fr, tmp);
        MatxMul(tmp, t1, fr);

        if (Math.Abs(fr[8]) > FltEpsilon)
        {
            double s = 1.0 / fr[8];
            for (int i = 0; i < 9; i++)
            {
                fr[i] *= s;
            }
        }

        fr.CopyTo(fmatrix);
        return 1;
    }

    private static bool Normalization(
        ReadOnlySpan<Point2f> m1,
        ReadOnlySpan<Point2f> m2,
        int count,
        out double m1cx,
        out double m1cy,
        out double m2cx,
        out double m2cy,
        out double scale1,
        out double scale2)
    {
        m1cx = m1cy = m2cx = m2cy = 0;
        for (int i = 0; i < count; i++)
        {
            m1cx += m1[i].X;
            m1cy += m1[i].Y;
            m2cx += m2[i].X;
            m2cy += m2[i].Y;
        }

        double t = 1.0 / count;
        m1cx *= t;
        m1cy *= t;
        m2cx *= t;
        m2cy *= t;

        scale1 = scale2 = 0;
        for (int i = 0; i < count; i++)
        {
            double dx1 = m1[i].X - m1cx, dy1 = m1[i].Y - m1cy;
            double dx2 = m2[i].X - m2cx, dy2 = m2[i].Y - m2cy;
            scale1 += Math.Sqrt((dx1 * dx1) + (dy1 * dy1));
            scale2 += Math.Sqrt((dx2 * dx2) + (dy2 * dy2));
        }

        scale1 *= t;
        scale2 *= t;
        if (scale1 < FltEpsilon || scale2 < FltEpsilon)
        {
            return false;
        }

        scale1 = Math.Sqrt(2.0) / scale1;
        scale2 = Math.Sqrt(2.0) / scale2;
        return true;
    }

    /// <summary>The normalizing transform <c>T</c> (or <c>Tᵀ</c>) of a point set.</summary>
    private static Mat<double> MatrixT(double scale, double cx, double cy, bool transpose)
    {
        double[] t = [scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1];
        if (transpose)
        {
            t = [t[0], t[3], t[6], t[1], t[4], t[7], t[2], t[5], t[8]];
        }

        return new Mat<double>(t, 3, 3);
    }

    private static void Copy(Mat<double> m, Span<double> dst)
    {
        for (int i = 0; i < 3; i++)
        {
            m.GetReadOnlyRowSpan(i)[..3].CopyTo(dst.Slice(i * 3, 3));
        }
    }

    /// <summary><c>Matx_MatMulOp</c> for 3 × 3 matrices: <c>s = 0; s += a·b</c> in k order.</summary>
    private static void MatxMul(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double s = 0;
                for (int k = 0; k < 3; k++)
                {
                    s += a[(i * 3) + k] * b[(k * 3) + j];
                }

                c[(i * 3) + j] = s;
            }
        }
    }
}
