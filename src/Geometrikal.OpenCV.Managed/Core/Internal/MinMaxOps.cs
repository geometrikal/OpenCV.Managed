namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// Scalar and <see cref="Vector{T}"/> minimum/maximum with C++ <c>std::min</c> /
/// <c>std::max</c> comparison semantics. For floating types a NaN in the second
/// operand is ignored and a NaN in the first operand is kept, as in OpenCV's scalar code.
/// </summary>
internal interface IMinMaxOp
{
    static abstract T Apply<T>(T a, T b)
        where T : unmanaged;

    static abstract Vector<T> Apply<T>(Vector<T> a, Vector<T> b)
        where T : unmanaged;
}

/// <summary><c>std::min(a, b)</c>: <c>b &lt; a ? b : a</c>.</summary>
internal readonly struct MinOp : IMinMaxOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Apply<T>(T a, T b)
        where T : unmanaged
        => MinMaxOps.LessThan(b, a) ? b : a;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Apply<T>(Vector<T> a, Vector<T> b)
        where T : unmanaged
        => MinMaxOps.IsFloating<T>()
            ? Vector.ConditionalSelect(Vector.LessThan(b, a), b, a)
            : Vector.Min(a, b);
}

/// <summary><c>std::max(a, b)</c>: <c>a &lt; b ? b : a</c>.</summary>
internal readonly struct MaxOp : IMinMaxOp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Apply<T>(T a, T b)
        where T : unmanaged
        => MinMaxOps.LessThan(a, b) ? b : a;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Apply<T>(Vector<T> a, Vector<T> b)
        where T : unmanaged
        => MinMaxOps.IsFloating<T>()
            ? Vector.ConditionalSelect(Vector.LessThan(a, b), b, a)
            : Vector.Max(a, b);
}

internal static class MinMaxOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsFloating<T>()
        where T : unmanaged
        => typeof(T) == typeof(float) || typeof(T) == typeof(double);

    /// <summary>
    /// True when <see cref="Vector{T}"/> is hardware accelerated for <typeparamref name="T"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsVectorized<T>()
        where T : unmanaged
        => Vector.IsHardwareAccelerated && Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool LessThan<T>(T a, T b)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return Unsafe.As<T, byte>(ref a) < Unsafe.As<T, byte>(ref b);
        }

        if (typeof(T) == typeof(sbyte))
        {
            return Unsafe.As<T, sbyte>(ref a) < Unsafe.As<T, sbyte>(ref b);
        }

        if (typeof(T) == typeof(ushort))
        {
            return Unsafe.As<T, ushort>(ref a) < Unsafe.As<T, ushort>(ref b);
        }

        if (typeof(T) == typeof(short))
        {
            return Unsafe.As<T, short>(ref a) < Unsafe.As<T, short>(ref b);
        }

        if (typeof(T) == typeof(int))
        {
            return Unsafe.As<T, int>(ref a) < Unsafe.As<T, int>(ref b);
        }

        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<T, float>(ref a) < Unsafe.As<T, float>(ref b);
        }

        if (typeof(T) == typeof(double))
        {
            return Unsafe.As<T, double>(ref a) < Unsafe.As<T, double>(ref b);
        }

        throw new NotSupportedException($"Unsupported element type '{typeof(T)}'.");
    }
}
