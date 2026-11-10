namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// Conversions and constants for the generic <c>float</c>/<c>double</c> linear algebra kernels.
/// </summary>
internal static class LapackScalar
{
    public const double FltEpsilon = 1.1920928955078125e-7;
    public const double FltMin = 1.1754943508222875e-38;
    public const double DblEpsilon = 2.220446049250313e-16;
    public const double DblMin = 2.2250738585072014e-308;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double D<T>(T v)
        where T : unmanaged
        => typeof(T) == typeof(float) ? Unsafe.As<T, float>(ref v) : Unsafe.As<T, double>(ref v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T F<T>(double v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            float f = (float)v;
            return Unsafe.As<float, T>(ref f);
        }

        return Unsafe.As<double, T>(ref v);
    }

    public static bool IsFloat<T>() => typeof(T) == typeof(float);

    /// <summary><c>std::numeric_limits&lt;T&gt;::epsilon()</c>.</summary>
    public static T Epsilon<T>()
        where T : unmanaged
        => F<T>(IsFloat<T>() ? FltEpsilon : DblEpsilon);

    public static void ThrowIfUnsupported<T>()
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"Element type {typeof(T).Name} is not supported. Use float or double.");
        }
    }

    /// <summary>OpenCV <c>RNG::next</c> (multiply-with-carry).</summary>
    public static uint NextRng(ref ulong state)
    {
        state = unchecked(((ulong)(uint)state * 4164903690UL) + (uint)(state >> 32));
        return (uint)state;
    }

    /// <summary>The <c>hypot</c> template of lapack.cpp.</summary>
    public static T Hypot<T>(T a, T b)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        a = T.Abs(a);
        b = T.Abs(b);
        if (a > b)
        {
            b /= a;
            return a * T.Sqrt(T.One + (b * b));
        }

        if (b > T.Zero)
        {
            a /= b;
            return b * T.Sqrt(T.One + (a * a));
        }

        return T.Zero;
    }
}
