// This implementation is ported from the OpenCV element-wise arithmetic kernels.
//
// Original sources:
//   modules/core/src/arithm.simd.hpp (add, sub, absdiff, min, max, mul, div, recip,
//     addWeighted, bitwise operations)
//   modules/core/include/opencv2/core/saturate.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>Scalar arithmetic on element types with OpenCV's per-depth overflow rules.</summary>
internal static class ElementMath
{
    /// <summary>Saturating for 8- and 16-bit types, wrapping for <see cref="int"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Add<T>(T a, T b)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            float r = Unsafe.As<T, float>(ref a) + Unsafe.As<T, float>(ref b);
            return Unsafe.As<float, T>(ref r);
        }

        if (typeof(T) == typeof(double))
        {
            double r = Unsafe.As<T, double>(ref a) + Unsafe.As<T, double>(ref b);
            return Unsafe.As<double, T>(ref r);
        }

        return Saturate.FromInt<T>(unchecked(Saturate.ToInt(a) + Saturate.ToInt(b)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Subtract<T>(T a, T b)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            float r = Unsafe.As<T, float>(ref a) - Unsafe.As<T, float>(ref b);
            return Unsafe.As<float, T>(ref r);
        }

        if (typeof(T) == typeof(double))
        {
            double r = Unsafe.As<T, double>(ref a) - Unsafe.As<T, double>(ref b);
            return Unsafe.As<double, T>(ref r);
        }

        return Saturate.FromInt<T>(unchecked(Saturate.ToInt(a) - Saturate.ToInt(b)));
    }

    /// <summary>
    /// <c>|a - b|</c>: exact for unsigned types, saturated for signed 8/16-bit types, and
    /// <c>abs</c> of the wrapped difference for <see cref="int"/> (<c>int.MinValue</c> stays).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T AbsDiff<T>(T a, T b)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            float r = MathF.Abs(Unsafe.As<T, float>(ref a) - Unsafe.As<T, float>(ref b));
            return Unsafe.As<float, T>(ref r);
        }

        if (typeof(T) == typeof(double))
        {
            double r = Math.Abs(Unsafe.As<T, double>(ref a) - Unsafe.As<T, double>(ref b));
            return Unsafe.As<double, T>(ref r);
        }

        if (typeof(T) == typeof(int))
        {
            int d = unchecked(Unsafe.As<T, int>(ref a) - Unsafe.As<T, int>(ref b));
            int r = d < 0 ? unchecked(-d) : d;
            return Unsafe.As<int, T>(ref r);
        }

        return Saturate.FromInt<T>(Math.Abs(Saturate.ToInt(a) - Saturate.ToInt(b)));
    }
}

internal readonly struct AddOp<T> : IBinaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (MinMaxOps.IsVectorized<T>())
        {
            ref T ra = ref MemoryMarshal.GetReference(a);
            ref T rb = ref MemoryMarshal.GetReference(b);
            ref T rd = ref MemoryMarshal.GetReference(dst);
            for (; i <= a.Length - Vector<T>.Count; i += Vector<T>.Count)
            {
                Vector<T> va = Vector.LoadUnsafe(ref ra, (nuint)i);
                Vector<T> vb = Vector.LoadUnsafe(ref rb, (nuint)i);
                Vector<T> r = Saturate.IsSmallInteger<T>() ? Vector.AddSaturate(va, vb) : va + vb;
                r.StoreUnsafe(ref rd, (nuint)i);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = ElementMath.Add(a[i], b[i]);
        }
    }
}

internal readonly struct SubtractOp<T> : IBinaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (MinMaxOps.IsVectorized<T>())
        {
            ref T ra = ref MemoryMarshal.GetReference(a);
            ref T rb = ref MemoryMarshal.GetReference(b);
            ref T rd = ref MemoryMarshal.GetReference(dst);
            for (; i <= a.Length - Vector<T>.Count; i += Vector<T>.Count)
            {
                Vector<T> va = Vector.LoadUnsafe(ref ra, (nuint)i);
                Vector<T> vb = Vector.LoadUnsafe(ref rb, (nuint)i);
                Vector<T> r = Saturate.IsSmallInteger<T>() ? Vector.SubtractSaturate(va, vb) : va - vb;
                r.StoreUnsafe(ref rd, (nuint)i);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = ElementMath.Subtract(a[i], b[i]);
        }
    }
}

/// <summary><c>b - a</c>, for a scalar first operand.</summary>
internal readonly struct ReverseSubtractOp<T> : IBinaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
        => default(SubtractOp<T>).Apply(b, a, dst);
}

internal readonly struct AbsDiffOp<T> : IBinaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (MinMaxOps.IsVectorized<T>())
        {
            ref T ra = ref MemoryMarshal.GetReference(a);
            ref T rb = ref MemoryMarshal.GetReference(b);
            ref T rd = ref MemoryMarshal.GetReference(dst);
            for (; i <= a.Length - Vector<T>.Count; i += Vector<T>.Count)
            {
                Vector<T> va = Vector.LoadUnsafe(ref ra, (nuint)i);
                Vector<T> vb = Vector.LoadUnsafe(ref rb, (nuint)i);
                Vector<T> r = Saturate.IsSmallInteger<T>()
                    ? Vector.SubtractSaturate(Vector.Max(va, vb), Vector.Min(va, vb))
                    : Vector.Abs(va - vb);
                r.StoreUnsafe(ref rd, (nuint)i);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = ElementMath.AbsDiff(a[i], b[i]);
        }
    }
}

internal readonly struct MinMaxRowOp<T, TOp> : IBinaryRowOp<T, T>
    where T : unmanaged
    where TOp : struct, IMinMaxOp
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (MinMaxOps.IsVectorized<T>())
        {
            ref T ra = ref MemoryMarshal.GetReference(a);
            ref T rb = ref MemoryMarshal.GetReference(b);
            ref T rd = ref MemoryMarshal.GetReference(dst);
            for (; i <= a.Length - Vector<T>.Count; i += Vector<T>.Count)
            {
                TOp.Apply(Vector.LoadUnsafe(ref ra, (nuint)i), Vector.LoadUnsafe(ref rb, (nuint)i))
                    .StoreUnsafe(ref rd, (nuint)i);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = TOp.Apply(a[i], b[i]);
        }
    }
}

/// <summary>
/// <c>multiply</c> with scale. 8/16-bit and <see cref="float"/>: <c>(a * (float)scale) * b</c>
/// in float. <see cref="int"/>: <c>(a * b) * scale</c> in double. <see cref="double"/>:
/// <c>(a * scale) * b</c>.
/// </summary>
internal readonly struct MultiplyOp<T>(double scale) : IBinaryRowOp<T, T>
    where T : unmanaged
{
    private readonly double _scale = scale;
    private readonly float _fscale = (float)scale;

    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (typeof(T) == typeof(byte) && Vector.IsHardwareAccelerated)
        {
            i = ScaledByteKernels.Multiply(
                MemoryMarshal.Cast<T, byte>(a), MemoryMarshal.Cast<T, byte>(b), MemoryMarshal.Cast<T, byte>(dst), _fscale);
        }
        else if (typeof(T) == typeof(float) && Vector.IsHardwareAccelerated)
        {
            var fa = MemoryMarshal.Cast<T, float>(a);
            var fb = MemoryMarshal.Cast<T, float>(b);
            var fd = MemoryMarshal.Cast<T, float>(dst);
            var vs = new Vector<float>(_fscale);
            for (; i <= fa.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(fa[i..]) * vs * new Vector<float>(fb[i..])).CopyTo(fd[i..]);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = Scalar(a[i], b[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T Scalar(T a, T b)
    {
        if (typeof(T) == typeof(double))
        {
            return Saturate.FromDouble<T>(Saturate.ToDouble(a) * _scale * Saturate.ToDouble(b));
        }

        if (typeof(T) == typeof(int))
        {
            return Saturate.FromDouble<T>(Saturate.ToDouble(a) * Saturate.ToDouble(b) * _scale);
        }

        return Saturate.FromFloat<T>(Saturate.ToFloat(a) * _fscale * Saturate.ToFloat(b));
    }
}

/// <summary>
/// <c>divide</c>: <c>(a * scale) / b</c>. Integer types give 0 where <c>b == 0</c>; 8/16-bit
/// types compute in float, <see cref="int"/> in double.
/// </summary>
internal readonly struct DivideOp<T>(double scale) : IBinaryRowOp<T, T>
    where T : unmanaged
{
    private readonly double _scale = scale;
    private readonly float _fscale = (float)scale;

    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (typeof(T) == typeof(float) && Vector.IsHardwareAccelerated)
        {
            var fa = MemoryMarshal.Cast<T, float>(a);
            var fb = MemoryMarshal.Cast<T, float>(b);
            var fd = MemoryMarshal.Cast<T, float>(dst);
            var vs = new Vector<float>(_fscale);
            for (; i <= fa.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(fa[i..]) * vs / new Vector<float>(fb[i..])).CopyTo(fd[i..]);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = Scalar(a[i], b[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T Scalar(T a, T b)
    {
        if (typeof(T) == typeof(double))
        {
            return Saturate.FromDouble<T>(Saturate.ToDouble(a) * _scale / Saturate.ToDouble(b));
        }

        if (typeof(T) == typeof(float))
        {
            return Saturate.FromFloat<T>(Saturate.ToFloat(a) * _fscale / Saturate.ToFloat(b));
        }

        int bi = Saturate.ToInt(b);
        if (bi == 0)
        {
            return default;
        }

        return typeof(T) == typeof(int)
            ? Saturate.FromDouble<T>(Saturate.ToDouble(a) * _scale / bi)
            : Saturate.FromFloat<T>(Saturate.ToFloat(a) * _fscale / bi);
    }
}

/// <summary><c>divide(scale, b)</c>: <c>scale / b</c>; integer types give 0 where <c>b == 0</c>.</summary>
internal readonly struct ReciprocalOp<T>(double scale) : IUnaryRowOp<T, T>
    where T : unmanaged
{
    private readonly double _scale = scale;
    private readonly float _fscale = (float)scale;

    public void Apply(ReadOnlySpan<T> b, Span<T> dst)
    {
        for (int i = 0; i < b.Length; i++)
        {
            dst[i] = Scalar(b[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T Scalar(T b)
    {
        if (typeof(T) == typeof(double))
        {
            return Saturate.FromDouble<T>(_scale / Saturate.ToDouble(b));
        }

        if (typeof(T) == typeof(float))
        {
            return Saturate.FromFloat<T>(_fscale / Saturate.ToFloat(b));
        }

        int bi = Saturate.ToInt(b);
        if (bi == 0)
        {
            return default;
        }

        return typeof(T) == typeof(int)
            ? Saturate.FromDouble<T>(_scale / bi)
            : Saturate.FromFloat<T>(_fscale / bi);
    }
}

/// <summary>
/// <c>addWeighted</c>. 8/16-bit and <see cref="float"/>: <c>a*w1 + (b*w2 + delta)</c> with
/// float weights. <see cref="int"/>: <c>(a*w1 + b*w2) + delta</c> in double.
/// <see cref="double"/>: <c>a*w1 + (b*w2 + delta)</c>.
/// </summary>
internal readonly struct AddWeightedOp<T>(double alpha, double beta, double gamma) : IBinaryRowOp<T, T>
    where T : unmanaged
{
    private readonly double _w1 = alpha;
    private readonly double _w2 = beta;
    private readonly double _delta = gamma;
    private readonly float _fw1 = (float)alpha;
    private readonly float _fw2 = (float)beta;
    private readonly float _fdelta = (float)gamma;

    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        int i = 0;
        if (typeof(T) == typeof(byte) && Vector.IsHardwareAccelerated)
        {
            i = ScaledByteKernels.AddWeighted(
                MemoryMarshal.Cast<T, byte>(a), MemoryMarshal.Cast<T, byte>(b), MemoryMarshal.Cast<T, byte>(dst),
                _fw1, _fw2, _fdelta);
        }
        else if (typeof(T) == typeof(float) && Vector.IsHardwareAccelerated)
        {
            var fa = MemoryMarshal.Cast<T, float>(a);
            var fb = MemoryMarshal.Cast<T, float>(b);
            var fd = MemoryMarshal.Cast<T, float>(dst);
            var w1 = new Vector<float>(_fw1);
            var w2 = new Vector<float>(_fw2);
            var d = new Vector<float>(_fdelta);
            for (; i <= fa.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                ((new Vector<float>(fa[i..]) * w1) + ((new Vector<float>(fb[i..]) * w2) + d)).CopyTo(fd[i..]);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = Scalar(a[i], b[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private T Scalar(T a, T b)
    {
        if (typeof(T) == typeof(double))
        {
            return Saturate.FromDouble<T>((Saturate.ToDouble(a) * _w1) + ((Saturate.ToDouble(b) * _w2) + _delta));
        }

        if (typeof(T) == typeof(int))
        {
            return Saturate.FromDouble<T>((Saturate.ToDouble(a) * _w1) + (Saturate.ToDouble(b) * _w2) + _delta);
        }

        return Saturate.FromFloat<T>((Saturate.ToFloat(a) * _fw1) + ((Saturate.ToFloat(b) * _fw2) + _fdelta));
    }
}

/// <summary><c>scaleAdd</c> for float types: <c>a * alpha + b</c> (float alpha for <see cref="float"/>).</summary>
internal readonly struct ScaleAddOp<T>(double alpha) : IBinaryRowOp<T, T>
    where T : unmanaged
{
    private readonly double _alpha = alpha;
    private readonly float _falpha = (float)alpha;

    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        if (typeof(T) == typeof(float))
        {
            var fa = MemoryMarshal.Cast<T, float>(a);
            var fb = MemoryMarshal.Cast<T, float>(b);
            var fd = MemoryMarshal.Cast<T, float>(dst);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                var va = new Vector<float>(_falpha);
                for (; i <= fa.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    ((new Vector<float>(fa[i..]) * va) + new Vector<float>(fb[i..])).CopyTo(fd[i..]);
                }
            }

            for (; i < fa.Length; i++)
            {
                fd[i] = (fa[i] * _falpha) + fb[i];
            }
        }
        else
        {
            var da = MemoryMarshal.Cast<T, double>(a);
            var db = MemoryMarshal.Cast<T, double>(b);
            var dd = MemoryMarshal.Cast<T, double>(dst);
            for (int i = 0; i < da.Length; i++)
            {
                dd[i] = (da[i] * _alpha) + db[i];
            }
        }
    }
}

/// <summary>Vector kernels for 8-bit operations computed in float.</summary>
internal static class ScaledByteKernels
{
    /// <summary>Processes whole vector blocks; returns the number of elements done.</summary>
    public static int Multiply(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> dst, float scale)
    {
        int i = 0;
        var vs = new Vector<float>(scale);
        bool bounded = MathF.Abs(scale) * (255f * 255f) < 2e9f;
        for (; i <= a.Length - Vector<byte>.Count; i += Vector<byte>.Count)
        {
            FilterVectors.WidenToSingle(new Vector<byte>(a[i..]), out var a0, out var a1, out var a2, out var a3);
            FilterVectors.WidenToSingle(new Vector<byte>(b[i..]), out var b0, out var b1, out var b2, out var b3);
            FilterVectors.SaturateToByte(
                Round(a0 * vs * b0, bounded),
                Round(a1 * vs * b1, bounded),
                Round(a2 * vs * b2, bounded),
                Round(a3 * vs * b3, bounded)).CopyTo(dst[i..]);
        }

        return i;
    }

    public static int AddWeighted(
        ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> dst, float w1, float w2, float delta)
    {
        int i = 0;
        var v1 = new Vector<float>(w1);
        var v2 = new Vector<float>(w2);
        var vd = new Vector<float>(delta);
        bool bounded = ((MathF.Abs(w1) + MathF.Abs(w2)) * 255f) + MathF.Abs(delta) < 2e9f;
        for (; i <= a.Length - Vector<byte>.Count; i += Vector<byte>.Count)
        {
            FilterVectors.WidenToSingle(new Vector<byte>(a[i..]), out var a0, out var a1, out var a2, out var a3);
            FilterVectors.WidenToSingle(new Vector<byte>(b[i..]), out var b0, out var b1, out var b2, out var b3);
            FilterVectors.SaturateToByte(
                Round((a0 * v1) + ((b0 * v2) + vd), bounded),
                Round((a1 * v1) + ((b1 * v2) + vd), bounded),
                Round((a2 * v1) + ((b2 * v2) + vd), bounded),
                Round((a3 * v1) + ((b3 * v2) + vd), bounded)).CopyTo(dst[i..]);
        }

        return i;
    }

    /// <summary>
    /// <c>cvRound</c>; when <paramref name="bounded"/> is set the values are finite and inside
    /// the <see cref="int"/> range, so the range check is skipped.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Round(Vector<float> v, bool bounded)
        => bounded ? Vector.ConvertToInt32(Vector.Round(v)) : FilterVectors.CvRound(v);
}

/// <summary>Bitwise operation on the raw bytes of each element.</summary>
internal interface IBitwiseOp
{
    static abstract Vector<byte> Apply(Vector<byte> a, Vector<byte> b);

    static abstract byte Apply(byte a, byte b);
}

internal readonly struct AndBits : IBitwiseOp
{
    public static Vector<byte> Apply(Vector<byte> a, Vector<byte> b) => a & b;

    public static byte Apply(byte a, byte b) => (byte)(a & b);
}

internal readonly struct OrBits : IBitwiseOp
{
    public static Vector<byte> Apply(Vector<byte> a, Vector<byte> b) => a | b;

    public static byte Apply(byte a, byte b) => (byte)(a | b);
}

internal readonly struct XorBits : IBitwiseOp
{
    public static Vector<byte> Apply(Vector<byte> a, Vector<byte> b) => a ^ b;

    public static byte Apply(byte a, byte b) => (byte)(a ^ b);
}

internal readonly struct BitwiseOp<T, TBits> : IBinaryRowOp<T, T>
    where T : unmanaged
    where TBits : struct, IBitwiseOp
{
    public void Apply(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
    {
        var ba = MemoryMarshal.AsBytes(a);
        var bb = MemoryMarshal.AsBytes(b);
        var bd = MemoryMarshal.AsBytes(dst);
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= ba.Length - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                TBits.Apply(new Vector<byte>(ba[i..]), new Vector<byte>(bb[i..])).CopyTo(bd[i..]);
            }
        }

        for (; i < ba.Length; i++)
        {
            bd[i] = TBits.Apply(ba[i], bb[i]);
        }
    }
}

internal readonly struct BitwiseNotOp<T> : IUnaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, Span<T> dst)
    {
        var ba = MemoryMarshal.AsBytes(a);
        var bd = MemoryMarshal.AsBytes(dst);
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= ba.Length - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                (~new Vector<byte>(ba[i..])).CopyTo(bd[i..]);
            }
        }

        for (; i < ba.Length; i++)
        {
            bd[i] = (byte)~ba[i];
        }
    }
}
