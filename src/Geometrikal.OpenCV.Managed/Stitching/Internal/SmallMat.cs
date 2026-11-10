// Small matrix helpers that follow the OpenCV MatExpr evaluation of the stitching code.
//
// Original sources:
//   modules/core/src/matrix_expressions.cpp (MatOp::matmul, MatOp_Invert)
//   modules/core/src/convert_scale.simd.hpp (cvtScale32f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV evaluates <c>A * B.inv() * C</c> as sequential <c>gemm</c> calls with each
/// <c>inv()</c> computed first by <c>invert(DECOMP_LU)</c>.
/// </summary>
internal static class SmallMat
{
    public static Mat<T> Mul<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
        => LinearAlgebra.Gemm(a, b, 1, null, 0);

    public static Mat<T> Inv<T>(Mat<T> a)
        where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        LinearAlgebra.Invert(a, out Mat<T> inv);
        return inv;
    }

    /// <summary><see cref="Mul{T}"/> of two 3 × 3 matrices into <paramref name="dst"/>.</summary>
    public static void MulInto(Mat<double> a, Mat<double> b, Mat<double> dst)
    {
        dst.Clear();
        GemmKernels.Run(a, b, 1, null, 0, dst, 0, a.Width);
    }

    /// <summary>
    /// <see cref="Inv{T}"/> of a packed 3 × 3 matrix into <paramref name="dst"/>: the explicit
    /// 3 × 3 formula of <c>invert(DECOMP_LU)</c>, or zeros for a zero determinant.
    /// </summary>
    public static void Inv3Into(Mat<double> a, Mat<double> dst)
    {
        Span<double> o = dst.GetSpan();
        if (!LinearAlgebra.Invert3x3(a.GetReadOnlySpan(), o))
        {
            o.Clear();
        }
    }

    public static Mat<double> Eye3() => CameraParams.Identity();

    /// <summary>Sets <paramref name="m"/> to the identity.</summary>
    public static void SetEye3(Mat<double> m)
    {
        m.Clear();
        m.At(0, 0) = m.At(1, 1) = m.At(2, 2) = 1;
    }

    /// <summary><see cref="K"/> into <paramref name="k"/>.</summary>
    public static void SetK(Mat<double> k, double fx, double fy, double ppx, double ppy)
    {
        SetEye3(k);
        k.At(0, 0) = fx;
        k.At(1, 1) = fy;
        k.At(0, 2) = ppx;
        k.At(1, 2) = ppy;
    }

    public static Mat<double> K(double fx, double fy, double ppx, double ppy)
    {
        Mat<double> k = Eye3();
        k.At(0, 0) = fx;
        k.At(1, 1) = fy;
        k.At(0, 2) = ppx;
        k.At(1, 2) = ppy;
        return k;
    }

    /// <summary><c>convertTo(CV_32F)</c> of a <c>double</c> matrix.</summary>
    public static Mat<float> ToFloat(Mat<double> m)
    {
        var r = new Mat<float>(m.Width, m.Height);
        for (int y = 0; y < m.Height; y++)
        {
            for (int x = 0; x < m.Width; x++)
            {
                r.At(y, x) = (float)m.At(y, x);
            }
        }

        return r;
    }

    public static Mat<double> ToDouble(Mat<float> m)
    {
        var r = new Mat<double>(m.Width, m.Height);
        for (int y = 0; y < m.Height; y++)
        {
            for (int x = 0; x < m.Width; x++)
            {
                r.At(y, x) = m.At(y, x);
            }
        }

        return r;
    }

    /// <summary>Rounds every element of a <c>double</c> matrix to <c>float</c> (OpenCV <c>CV_32F</c> storage).</summary>
    public static Mat<double> RoundToFloat(Mat<double> m) => ToDouble(ToFloat(m));
}
