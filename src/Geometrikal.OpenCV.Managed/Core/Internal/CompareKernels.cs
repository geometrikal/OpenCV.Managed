// This implementation is ported from the OpenCV comparison and range-check functions.
//
// Original sources:
//   modules/core/src/arithm.cpp (compare, inRange, inRange_, inRangeReduce)
//   modules/core/src/arithm.simd.hpp (cmp kernels)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

internal static class CompareMath
{
    /// <summary>Compares two values converted exactly to double; NaN compares false except for Ne.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool Apply(double a, double b, CmpType op) => op switch
    {
        CmpType.Eq => a == b,
        CmpType.Gt => a > b,
        CmpType.Ge => a >= b,
        CmpType.Lt => a < b,
        CmpType.Le => a <= b,
        _ => a != b,
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Apply<T>(Vector<T> a, Vector<T> b, CmpType op)
        where T : unmanaged
        => op switch
        {
            CmpType.Eq => Vector.Equals(a, b),
            CmpType.Gt => Vector.GreaterThan(a, b),
            CmpType.Ge => Vector.GreaterThanOrEqual(a, b),
            CmpType.Lt => Vector.LessThan(a, b),
            CmpType.Le => Vector.LessThanOrEqual(a, b),
            _ => ~Vector.Equals(a, b),
        };
}

/// <summary><c>compare</c>: 255 where the comparison holds, else 0.</summary>
internal readonly struct CompareOp<T>(CmpType op) : IBinaryRowOp<T, byte>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<byte> dst)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            if (typeof(T) == typeof(byte))
            {
                var ba = MemoryMarshal.Cast<T, byte>(a);
                var bb = MemoryMarshal.Cast<T, byte>(b);
                for (; i <= ba.Length - Vector<byte>.Count; i += Vector<byte>.Count)
                {
                    CompareMath.Apply(new Vector<byte>(ba[i..]), new Vector<byte>(bb[i..]), op).CopyTo(dst[i..]);
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var fa = MemoryMarshal.Cast<T, float>(a);
                var fb = MemoryMarshal.Cast<T, float>(b);
                int n = Vector<float>.Count;
                for (; i <= fa.Length - (4 * n); i += 4 * n)
                {
                    Vector<int> m0 = Vector.AsVectorInt32(CompareMath.Apply(new Vector<float>(fa[i..]), new Vector<float>(fb[i..]), op));
                    Vector<int> m1 = Vector.AsVectorInt32(CompareMath.Apply(new Vector<float>(fa[(i + n)..]), new Vector<float>(fb[(i + n)..]), op));
                    Vector<int> m2 = Vector.AsVectorInt32(CompareMath.Apply(new Vector<float>(fa[(i + (2 * n))..]), new Vector<float>(fb[(i + (2 * n))..]), op));
                    Vector<int> m3 = Vector.AsVectorInt32(CompareMath.Apply(new Vector<float>(fa[(i + (3 * n))..]), new Vector<float>(fb[(i + (3 * n))..]), op));
                    Vector<short> lo = Vector.Narrow(m0, m1);
                    Vector<short> hi = Vector.Narrow(m2, m3);
                    Vector.AsVectorByte(Vector.Narrow(lo, hi)).CopyTo(dst[i..]);
                }
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = CompareMath.Apply(Saturate.ToDouble(a[i]), Saturate.ToDouble(b[i]), op) ? (byte)255 : (byte)0;
        }
    }
}

/// <summary>
/// <c>compare</c> with a fixed value. <see cref="Constant"/> is set when the comparison has the
/// same result for every element.
/// </summary>
internal readonly struct CompareScalarOp<T>(double value, CmpType op, byte? constant) : IUnaryRowOp<T, byte>
    where T : unmanaged
{
    public byte? Constant { get; } = constant;

    public void Apply(ReadOnlySpan<T> a, Span<byte> dst)
    {
        if (Constant is byte c)
        {
            dst.Fill(c);
            return;
        }

        for (int i = 0; i < a.Length; i++)
        {
            dst[i] = CompareMath.Apply(Saturate.ToDouble(a[i]), value, op) ? (byte)255 : (byte)0;
        }
    }

    /// <summary>
    /// Builds the operation for <c>compare(src, s[0], op)</c>. Integer sources follow OpenCV:
    /// out-of-range values give a constant result; fractional values use <c>ceil</c> for
    /// <c>Lt</c>/<c>Ge</c> and <c>floor</c> for <c>Le</c>/<c>Gt</c>; <c>Eq</c> is never true and
    /// <c>Ne</c> always true.
    /// </summary>
    public static CompareScalarOp<T> Create(double f, CmpType op)
    {
        if (typeof(T) == typeof(float))
        {
            return new CompareScalarOp<T>((float)f, op, null);
        }

        if (typeof(T) == typeof(double))
        {
            return new CompareScalarOp<T>(f, op, null);
        }

        double min = Saturate.MinValue<T>();
        double max = Saturate.MaxValue<T>();
        if (f < min)
        {
            return Fixed(op is CmpType.Gt or CmpType.Ge or CmpType.Ne);
        }

        if (f > max)
        {
            return Fixed(op is CmpType.Lt or CmpType.Le or CmpType.Ne);
        }

        if (Math.Round(f) != f)
        {
            switch (op)
            {
                case CmpType.Eq:
                    return Fixed(false);
                case CmpType.Ne:
                    return Fixed(true);
                case CmpType.Lt:
                case CmpType.Ge:
                    f = Math.Ceiling(f);
                    break;
                default:
                    f = Math.Floor(f);
                    break;
            }
        }

        return new CompareScalarOp<T>(f, op, null);

        static CompareScalarOp<T> Fixed(bool result) => new(0, CmpType.Eq, result ? (byte)255 : (byte)0);
    }
}

/// <summary><c>inRange</c>: 255 where every channel lies in <c>[lower, upper]</c>.</summary>
internal static class InRangeKernel
{
    /// <summary>
    /// Byte rows: per-element range masks with <see cref="Vector{T}"/>, then an AND over the
    /// channels of each pixel. <paramref name="tmp"/> holds one row of element masks.
    /// </summary>
    public static void RowBytes(ReadOnlySpan<byte> a, ReadOnlySpan<byte> lo, ReadOnlySpan<byte> hi, Span<byte> tmp, Span<byte> dst, int cn)
    {
        if (cn is 3 or 4 && ColorVectors.IsSupported)
        {
            RowBytesPlanar(a, lo, hi, dst, cn);
            return;
        }

        Span<byte> m = cn == 1 ? dst : tmp;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= a.Length - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                var v = new Vector<byte>(a[i..]);
                (Vector.GreaterThanOrEqual(v, new Vector<byte>(lo[i..])) & Vector.LessThanOrEqual(v, new Vector<byte>(hi[i..])))
                    .CopyTo(m[i..]);
            }
        }

        for (; i < a.Length; i++)
        {
            m[i] = a[i] >= lo[i] && a[i] <= hi[i] ? (byte)255 : (byte)0;
        }

        if (cn == 1)
        {
            return;
        }

        for (int x = 0, j = 0; x < dst.Length; x++, j += cn)
        {
            byte r = m[j];
            for (int c = 1; c < cn; c++)
            {
                r &= m[j + c];
            }

            dst[x] = r;
        }
    }

    /// <summary>
    /// 3- and 4-channel byte rows: the source and bound rows are deinterleaved into planes, and
    /// 16 pixels are compared and combined at a time.
    /// </summary>
    private static void RowBytesPlanar(ReadOnlySpan<byte> a, ReadOnlySpan<byte> lo, ReadOnlySpan<byte> hi, Span<byte> dst, int cn)
    {
        ref byte ra = ref MemoryMarshal.GetReference(a);
        ref byte rl = ref MemoryMarshal.GetReference(lo);
        ref byte rh = ref MemoryMarshal.GetReference(hi);
        ref byte rd = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= dst.Length - 16; x += 16)
        {
            int o = x * cn;
            ColorVectors.Deinterleave(ref Unsafe.Add(ref ra, o), cn, out var v0, out var v1, out var v2, out var v3);
            ColorVectors.Deinterleave(ref Unsafe.Add(ref rl, o), cn, out var l0, out var l1, out var l2, out var l3);
            ColorVectors.Deinterleave(ref Unsafe.Add(ref rh, o), cn, out var h0, out var h1, out var h2, out var h3);
            Vector128<byte> m = In(v0, l0, h0) & In(v1, l1, h1) & In(v2, l2, h2);
            if (cn == 4)
            {
                m &= In(v3, l3, h3);
            }

            m.StoreUnsafe(ref rd, (nuint)x);
        }

        for (; x < dst.Length; x++)
        {
            bool inside = true;
            for (int c = 0, i = x * cn; c < cn; c++, i++)
            {
                inside &= a[i] >= lo[i] && a[i] <= hi[i];
            }

            dst[x] = inside ? (byte)255 : (byte)0;
        }

        static Vector128<byte> In(Vector128<byte> v, Vector128<byte> l, Vector128<byte> h)
            => Vector128.GreaterThanOrEqual(v, l) & Vector128.LessThanOrEqual(v, h);
    }

    public static void Row<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> lo, ReadOnlySpan<T> hi, Span<byte> dst, int cn)
        where T : unmanaged
    {
        for (int x = 0, i = 0; x < dst.Length; x++)
        {
            bool inside = true;
            for (int c = 0; c < cn; c++, i++)
            {
                double v = Saturate.ToDouble(a[i]);
                inside &= Saturate.ToDouble(lo[i]) <= v && v <= Saturate.ToDouble(hi[i]);
            }

            dst[x] = inside ? (byte)255 : (byte)0;
        }
    }

    public static void Row<T>(ReadOnlySpan<T> a, double[] lo, double[] hi, Span<byte> dst)
        where T : unmanaged
    {
        int cn = lo.Length;
        for (int x = 0, i = 0; x < dst.Length; x++)
        {
            bool inside = true;
            for (int c = 0; c < cn; c++, i++)
            {
                double v = Saturate.ToDouble(a[i]);
                inside &= lo[c] <= v && v <= hi[c];
            }

            dst[x] = inside ? (byte)255 : (byte)0;
        }
    }

    /// <summary>
    /// Converts scalar bounds to the element type as OpenCV <c>inRange</c>: integer types round
    /// half to even; an empty or out-of-range 8/16-bit interval selects nothing.
    /// </summary>
    public static void ConvertBounds<T>(Scalar lower, Scalar upper, int cn, out double[] lo, out double[] hi)
        where T : unmanaged
    {
        lo = new double[cn];
        hi = new double[cn];
        for (int c = 0; c < cn; c++)
        {
            if (Saturate.IsFloating<T>())
            {
                lo[c] = Saturate.ToDouble(Saturate.FromDouble<T>(lower[c]));
                hi[c] = Saturate.ToDouble(Saturate.FromDouble<T>(upper[c]));
                continue;
            }

            int il = ScalarConverter.CvRound(lower[c]);
            int ih = ScalarConverter.CvRound(upper[c]);
            if (typeof(T) != typeof(int))
            {
                double min = Saturate.MinValue<T>();
                double max = Saturate.MaxValue<T>();
                if (il > ih || il > max || ih < min)
                {
                    lo[c] = min + 1;
                    hi[c] = min;
                    continue;
                }

                lo[c] = Math.Clamp(il, min, max);
                hi[c] = Math.Clamp(ih, min, max);
                continue;
            }

            lo[c] = il;
            hi[c] = ih;
        }
    }
}
