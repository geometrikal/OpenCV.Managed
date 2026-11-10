// This implementation is ported from the OpenCV spectrum multiplication.
//
// Original source:
//   modules/core/src/dxt.cpp (cv::mulSpectrums, mulSpectrums_Impl, cv::divSpectrums)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/dxt.cpp
//
// OpenCV is licensed under the Apache License 2.0.
// See the OpenCV license and this repository's licensing documentation.

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// <c>mulSpectrums</c>: complex products in double, rounded to T; real CCS elements multiplied in T.
/// </summary>
internal static class SpectrumKernels
{
    public static void Mul<T>(Mat<T> a, Mat<T> b, Mat<T> c, bool rowsFlag, bool conjB)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int cn = a.Channels;
        int rows = a.Height, cols = a.Width;
        bool is1d = rowsFlag || rows == 1 || (cols == 1 && a.IsContinuous && b.IsContinuous && c.IsContinuous);
        if (is1d && !rowsFlag)
        {
            cols = cols + rows - 1;
            rows = 1;
        }

        bool cn1 = cn == 1;
        int j0 = cn1 ? 1 : 0;
        int j1 = (cols * cn) - ((cols & 1) == 0 && cn1 ? 1 : 0);

        if (!is1d && cn1)
        {
            MulColumn(a, b, c, 0, conjB);
            if ((cols & 1) == 0)
            {
                MulColumn(a, b, c, cols - 1, conjB);
            }
        }

        int rowLen = cols * cn;
        bool edges = is1d && cn1;
        if (rows == 1)
        {
            ReadOnlySpan<T> ra = rowsFlag || a.Height == 1 ? a.GetReadOnlyRowSpan(0) : a.GetReadOnlySpan();
            ReadOnlySpan<T> rb = rowsFlag || b.Height == 1 ? b.GetReadOnlyRowSpan(0) : b.GetReadOnlySpan();
            Span<T> rc = rowsFlag || c.Height == 1 ? c.GetRowSpan(0) : c.GetSpan();
            MulRow(ra[..rowLen], rb[..rowLen], rc[..rowLen], j0, j1, edges, conjB);
            return;
        }

        ParallelRows.For(rows, rowLen / 4, (r0, r1) => MulRows(a, b, c, r0, r1, j0, j1, edges, conjB));
    }

    private static void MulRows<T>(Mat<T> a, Mat<T> b, Mat<T> c, int r0, int r1, int j0, int j1, bool edges, bool conjB)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        for (int y = r0; y < r1; y++)
        {
            MulRow(a.GetReadOnlyRowSpan(y), b.GetReadOnlyRowSpan(y), c.GetRowSpan(y), j0, j1, edges, conjB);
        }
    }

    private static void MulRow<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, int j0, int j1, bool edges, bool conjB)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        if (edges)
        {
            c[0] = a[0] * b[0];
        }

        int j = j0;
        if (typeof(T) == typeof(float))
        {
            j = MulPairsVector(MemoryMarshal.Cast<T, float>(a), MemoryMarshal.Cast<T, float>(b), MemoryMarshal.Cast<T, float>(c), j0, j1, conjB);
        }
        else if (typeof(T) == typeof(double))
        {
            j = MulPairsVector(MemoryMarshal.Cast<T, double>(a), MemoryMarshal.Cast<T, double>(b), MemoryMarshal.Cast<T, double>(c), j0, j1, conjB);
        }

        for (; j < j1; j += 2)
        {
            (c[j], c[j + 1]) = MulPair(a[j], a[j + 1], b[j], b[j + 1], conjB);
        }

        if (edges && (c.Length & 1) == 0)
        {
            c[j1] = a[j1] * b[j1];
        }
    }

    private static void MulColumn<T>(Mat<T> a, Mat<T> b, Mat<T> c, int x, bool conjB)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int rows = a.Height;
        c.At(0, x) = a.At(0, x) * b.At(0, x);
        for (int j = 1; j <= rows - 2; j += 2)
        {
            (c.At(j, x), c.At(j + 1, x)) = MulPair(a.At(j, x), a.At(j + 1, x), b.At(j, x), b.At(j + 1, x), conjB);
        }

        if ((rows & 1) == 0)
        {
            c.At(rows - 1, x) = a.At(rows - 1, x) * b.At(rows - 1, x);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (T Re, T Im) MulPair<T>(T are, T aim, T bre, T bim, bool conjB)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        double ar = double.CreateTruncating(are), ai = double.CreateTruncating(aim);
        double br = double.CreateTruncating(bre), bi = double.CreateTruncating(bim);
        if (conjB)
        {
            bi = -bi;
        }

        double cr = (ar * br) - (ai * bi);
        double ci = (ar * bi) + (ai * br);
        return (T.CreateTruncating(cr), T.CreateTruncating(ci));
    }

    /// <summary>
    /// Complex pairs from <paramref name="j0"/> with AVX: <c>addsub(ar·b, ai·swap(b))</c> gives
    /// <c>(ar·br − ai·bi, ar·bi + ai·br)</c> with the scalar rounding of each operation.
    /// </summary>
    private static int MulPairsVector(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int j0, int j1, bool conjB)
    {
        int j = j0;
        if (!Avx.IsSupported)
        {
            return j;
        }

        Vector256<double> sign = conjB ? Vector256.Create(0.0, -0.0, 0.0, -0.0) : Vector256<double>.Zero;
        ref float pa = ref MemoryMarshal.GetReference(a);
        ref float pb = ref MemoryMarshal.GetReference(b);
        ref float pc = ref MemoryMarshal.GetReference(c);
        for (; j <= j1 - 4; j += 4)
        {
            Vector256<double> va = Avx.ConvertToVector256Double(Vector128.LoadUnsafe(ref pa, (nuint)j));
            Vector256<double> vb = Avx.ConvertToVector256Double(Vector128.LoadUnsafe(ref pb, (nuint)j)) ^ sign;
            Vector256<double> r = MulComplex(va, vb);
            Avx.ConvertToVector128Single(r).StoreUnsafe(ref pc, (nuint)j);
        }

        return j;
    }

    private static int MulPairsVector(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c, int j0, int j1, bool conjB)
    {
        int j = j0;
        if (!Avx.IsSupported)
        {
            return j;
        }

        Vector256<double> sign = conjB ? Vector256.Create(0.0, -0.0, 0.0, -0.0) : Vector256<double>.Zero;
        ref double pa = ref MemoryMarshal.GetReference(a);
        ref double pb = ref MemoryMarshal.GetReference(b);
        ref double pc = ref MemoryMarshal.GetReference(c);
        for (; j <= j1 - 4; j += 4)
        {
            Vector256<double> va = Vector256.LoadUnsafe(ref pa, (nuint)j);
            Vector256<double> vb = Vector256.LoadUnsafe(ref pb, (nuint)j) ^ sign;
            MulComplex(va, vb).StoreUnsafe(ref pc, (nuint)j);
        }

        return j;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> MulComplex(Vector256<double> a, Vector256<double> b)
    {
        Vector256<double> ar = Avx.Permute(a, 0b0000);
        Vector256<double> ai = Avx.Permute(a, 0b1111);
        Vector256<double> bs = Avx.Permute(b, 0b0101);
        return Avx.AddSubtract(ar * b, ai * bs);
    }

    // cv::divSpectrums: OpenCV computes a·conj(b) without conjB and a·b with conjB, over |b|² + eps.
    public static void Div<T>(Mat<T> a, Mat<T> b, Mat<T> c, bool rowsFlag, bool conjB, bool legacy455 = false)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        T eps = typeof(T) == typeof(float) ? T.CreateTruncating(1.1920928955078125e-7) : T.CreateTruncating(2.220446049250313e-16);
        int cn = a.Channels;
        int rows = a.Height, cols = a.Width;
        bool is1d = rowsFlag || rows == 1 || (cols == 1 && a.IsContinuous && b.IsContinuous && c.IsContinuous);
        if (is1d && !rowsFlag)
        {
            cols = cols + rows - 1;
            rows = 1;
        }

        bool cn1 = cn == 1;
        int j0 = cn1 ? 1 : 0;
        int j1 = (cols * cn) - ((cols & 1) == 0 && cn1 ? 1 : 0);
        if (!is1d && cn1)
        {
            DivColumn(a, b, c, 0, conjB, eps);
            if ((cols & 1) == 0)
            {
                DivColumn(a, b, c, cols - 1, conjB, eps);
            }
        }

        int rowLen = cols * cn;
        bool edges = is1d && cn1;
        if (rows == 1)
        {
            ReadOnlySpan<T> ra = rowsFlag || a.Height == 1 ? a.GetReadOnlyRowSpan(0) : a.GetReadOnlySpan();
            ReadOnlySpan<T> rb = rowsFlag || b.Height == 1 ? b.GetReadOnlyRowSpan(0) : b.GetReadOnlySpan();
            Span<T> rc = rowsFlag || c.Height == 1 ? c.GetRowSpan(0) : c.GetSpan();
            DivRow(ra[..rowLen], rb[..rowLen], rc[..rowLen], j0, j1, edges, conjB, eps, legacy455);
            return;
        }

        for (int y = 0; y < rows; y++)
        {
            DivRow(a.GetReadOnlyRowSpan(y), b.GetReadOnlyRowSpan(y), c.GetRowSpan(y), j0, j1, edges, conjB, eps, legacy455);
        }
    }

    private static void DivRow<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, int j0, int j1, bool edges, bool conjB, T eps, bool legacy455)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        if (edges)
        {
            c[0] = a[0] / (b[0] + eps);
            if ((c.Length & 1) == 0)
            {
                c[j1] = a[j1] / (b[j1] + eps);
            }
        }

        for (int j = j0; j < j1; j += 2)
        {
            (c[j], c[j + 1]) = legacy455
                ? DivPairInT(a[j], a[j + 1], b[j], b[j + 1], conjB, eps)
                : DivPair(a[j], a[j + 1], b[j], b[j + 1], conjB, eps);
        }
    }

    // cv2 4.5.5 row pairs: products and sums in T, then the quotient in double.
    private static (T Re, T Im) DivPairInT<T>(T ar, T ai, T br, T bi, bool conjB, T eps)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        double denom = double.CreateTruncating((br * br) + (bi * bi) + eps);
        double re = double.CreateTruncating(conjB ? (ar * br) - (ai * bi) : (ar * br) + (ai * bi));
        double im = double.CreateTruncating(conjB ? (ai * br) + (ar * bi) : (ai * br) - (ar * bi));
        return (T.CreateTruncating(re / denom), T.CreateTruncating(im / denom));
    }

    private static void DivColumn<T>(Mat<T> a, Mat<T> b, Mat<T> c, int x, bool conjB, T eps)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int rows = a.Height;
        c.At(0, x) = a.At(0, x) / (b.At(0, x) + eps);
        if ((rows & 1) == 0)
        {
            c.At(rows - 1, x) = a.At(rows - 1, x) / (b.At(rows - 1, x) + eps);
        }

        for (int j = 1; j <= rows - 2; j += 2)
        {
            (c.At(j, x), c.At(j + 1, x)) = DivPair(a.At(j, x), a.At(j + 1, x), b.At(j, x), b.At(j + 1, x), conjB, eps);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (T Re, T Im) DivPair<T>(T are, T aim, T bre, T bim, bool conjB, T eps)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        double ar = double.CreateTruncating(are), ai = double.CreateTruncating(aim);
        double br = double.CreateTruncating(bre), bi = double.CreateTruncating(bim);
        double denom = (br * br) + (bi * bi) + double.CreateTruncating(eps);
        double re = conjB ? (ar * br) - (ai * bi) : (ar * br) + (ai * bi);
        double im = conjB ? (ai * br) + (ar * bi) : (ai * br) - (ar * bi);
        return (T.CreateTruncating(re / denom), T.CreateTruncating(im / denom));
    }
}
