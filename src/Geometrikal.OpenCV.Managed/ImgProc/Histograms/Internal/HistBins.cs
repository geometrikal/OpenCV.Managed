// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (calcHist_, calcBackProj_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Bin offsets of <c>ushort</c> and <c>float</c> pixels for one row. An offset is the element
/// index of the bin in the histogram; a negative offset marks a pixel outside the histogram.
/// </summary>
/// <remarks>
/// Uniform bins use <c>idx = cvFloor(v * a + b)</c> in double. <c>calcHist</c> drops values outside
/// <c>[lo, hi)</c> and clamps the index; NaN passes the range test and goes to bin 0.
/// <c>calcBackProject</c> does not test the range and accepts only <c>0 &lt;= idx &lt; size</c>.
/// The vector path computes each element with the same double operations as the scalar path.
/// </remarks>
internal static class HistBins
{
    /// <summary>2^52: adding it to an integral double in [0, 2^31) puts the integer in the low 32 bits.</summary>
    private const double Magic = 4503599627370496.0;

    /// <summary>Fills <paramref name="idx"/> with the bin offsets of row <paramref name="y"/>.</summary>
    public static void Row<T>(HistLayout<T> layout, int y, Span<int> idx, bool backProject)
        where T : unmanaged
    {
        int w = layout.Width;
        for (int d = 0; d < layout.Dims; d++)
        {
            ReadOnlySpan<T> row = layout.Images[d].GetReadOnlyRowSpan(y);
            ref T src = ref Unsafe.Add(ref MemoryMarshal.GetReference(row), layout.Offsets[d]);
            int cn = layout.Steps[d];
            bool first = d == 0;
            int sz = layout.Sizes[d];
            int hstep = layout.HSteps[d];
            ref int dst = ref MemoryMarshal.GetReference(idx);
            if (!layout.Uniform)
            {
                NonUniform(ref src, cn, w, layout.Ranges![d], hstep, first, ref dst);
                continue;
            }

            double a = layout.A![d];
            double b = layout.B![d];
            double lo = layout.Lo![d];
            double hi = layout.Hi![d];
            int x = 0;
            if (cn == 1 && Vector.IsHardwareAccelerated)
            {
                x = backProject
                    ? UniformVector<T, BackProjectRule>(ref src, w, a, b, lo, hi, sz, hstep, first, ref dst)
                    : UniformVector<T, CalcHistRule>(ref src, w, a, b, lo, hi, sz, hstep, first, ref dst);
            }

            for (; x < w; x++)
            {
                double v = ToDouble(Unsafe.Add(ref src, x * cn));
                int bin = ScalarConverter.CvFloor((v * a) + b);
                bool valid;
                if (backProject)
                {
                    valid = (uint)bin < (uint)sz;
                }
                else
                {
                    valid = !(v < lo || v >= hi);
                    bin = Math.Clamp(bin, 0, sz - 1);
                }

                ref int o = ref Unsafe.Add(ref dst, x);
                o = valid ? (first ? 0 : o) + (bin * hstep) : int.MinValue;
            }
        }
    }

    /// <summary>
    /// Non-uniform bins: <c>idx = -1; while (v &gt;= R[idx + 1] &amp;&amp; ++idx &lt; size);</c> in
    /// float. Boundaries increase strictly, so a binary search gives the same bin; NaN gives -1.
    /// </summary>
    private static void NonUniform<T>(ref T src, int cn, int w, float[] r, int hstep, bool first, ref int dst)
        where T : unmanaged
    {
        int sz = r.Length - 1;
        for (int x = 0; x < w; x++)
        {
            float v = ToFloat(Unsafe.Add(ref src, x * cn));
            int lowIdx = 0;
            int high = r.Length;
            while (lowIdx < high)
            {
                int mid = (lowIdx + high) >>> 1;
                if (v >= r[mid])
                {
                    lowIdx = mid + 1;
                }
                else
                {
                    high = mid;
                }
            }

            int bin = lowIdx - 1;
            ref int o = ref Unsafe.Add(ref dst, x);
            o = (uint)bin < (uint)sz ? (first ? 0 : o) + (bin * hstep) : int.MinValue;
        }
    }

    private interface IBinRule
    {
        static abstract Vector<long> Bin(
            Vector<double> v, Vector<double> a, Vector<double> b, Vector<double> lo, Vector<double> hi,
            Vector<double> maxBin, Vector<double> size, out Vector<long> valid);
    }

    /// <summary>calcHist: range test on the value, then clamp; NaN passes and clamps to 0.</summary>
    private readonly struct CalcHistRule : IBinRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<long> Bin(
            Vector<double> v, Vector<double> a, Vector<double> b, Vector<double> lo, Vector<double> hi,
            Vector<double> maxBin, Vector<double> size, out Vector<long> valid)
        {
            Vector<double> f = Vector.Floor((v * a) + b);
            valid = ~(Vector.LessThan(v, lo) | Vector.GreaterThanOrEqual(v, hi));
            Vector<double> c = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(f, Vector<double>.Zero), f, Vector<double>.Zero);
            c = Vector.ConditionalSelect(Vector.LessThanOrEqual(c, maxBin), c, maxBin);
            return Vector.AsVectorInt64(c + new Vector<double>(Magic));
        }
    }

    /// <summary>calcBackProject: no range test and no clamp; valid iff <c>0 &lt;= idx &lt; size</c>.</summary>
    private readonly struct BackProjectRule : IBinRule
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<long> Bin(
            Vector<double> v, Vector<double> a, Vector<double> b, Vector<double> lo, Vector<double> hi,
            Vector<double> maxBin, Vector<double> size, out Vector<long> valid)
        {
            Vector<double> f = Vector.Floor((v * a) + b);
            valid = Vector.GreaterThanOrEqual(f, Vector<double>.Zero) & Vector.LessThan(f, size);
            Vector<double> c = Vector.ConditionalSelect(valid, f, Vector<double>.Zero);
            return Vector.AsVectorInt64(c + new Vector<double>(Magic));
        }
    }

    private static int UniformVector<T, TRule>(
        ref T src, int w, double a, double b, double lo, double hi, int sz, int hstep, bool first, ref int dst)
        where T : unmanaged
        where TRule : struct, IBinRule
    {
        var va = new Vector<double>(a);
        var vb = new Vector<double>(b);
        var vlo = new Vector<double>(lo);
        var vhi = new Vector<double>(hi);
        var vmax = new Vector<double>(sz - 1);
        var vsz = new Vector<double>(sz);
        var vstep = new Vector<int>(hstep);
        int ic = Vector<int>.Count;
        int x = 0;
        if (typeof(T) == typeof(float))
        {
            ref float s = ref Unsafe.As<T, float>(ref src);
            for (; x <= w - ic; x += ic)
            {
                Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)x), out Vector<double> d0, out Vector<double> d1);
                Store<TRule>(d0, d1, va, vb, vlo, vhi, vmax, vsz, vstep, first, ref Unsafe.Add(ref dst, x));
            }
        }
        else
        {
            ref ushort s = ref Unsafe.As<T, ushort>(ref src);
            int uc = Vector<ushort>.Count;
            for (; x <= w - uc; x += uc)
            {
                Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)x), out Vector<uint> u0, out Vector<uint> u1);
                Vector.Widen(Vector.ConvertToSingle(Vector.AsVectorInt32(u0)), out Vector<double> d0, out Vector<double> d1);
                Vector.Widen(Vector.ConvertToSingle(Vector.AsVectorInt32(u1)), out Vector<double> d2, out Vector<double> d3);
                Store<TRule>(d0, d1, va, vb, vlo, vhi, vmax, vsz, vstep, first, ref Unsafe.Add(ref dst, x));
                Store<TRule>(d2, d3, va, vb, vlo, vhi, vmax, vsz, vstep, first, ref Unsafe.Add(ref dst, x + ic));
            }
        }

        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store<TRule>(
        Vector<double> d0, Vector<double> d1, Vector<double> va, Vector<double> vb, Vector<double> vlo,
        Vector<double> vhi, Vector<double> vmax, Vector<double> vsz, Vector<int> vstep, bool first, ref int dst)
        where TRule : struct, IBinRule
    {
        Vector<long> i0 = TRule.Bin(d0, va, vb, vlo, vhi, vmax, vsz, out Vector<long> m0);
        Vector<long> i1 = TRule.Bin(d1, va, vb, vlo, vhi, vmax, vsz, out Vector<long> m1);
        Vector<int> off = Vector.Narrow(i0, i1) * vstep;
        Vector<int> valid = Vector.Narrow(m0, m1);
        if (!first)
        {
            off += Vector.LoadUnsafe(ref dst);
        }

        Vector.ConditionalSelect(valid, off, new Vector<int>(int.MinValue)).StoreUnsafe(ref dst);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ToDouble<T>(T v)
        where T : unmanaged
        => typeof(T) == typeof(float) ? Unsafe.As<T, float>(ref v) : Unsafe.As<T, ushort>(ref v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ToFloat<T>(T v)
        where T : unmanaged
        => typeof(T) == typeof(float) ? Unsafe.As<T, float>(ref v)
            : typeof(T) == typeof(ushort) ? Unsafe.As<T, ushort>(ref v)
            : Unsafe.As<T, byte>(ref v);
}
