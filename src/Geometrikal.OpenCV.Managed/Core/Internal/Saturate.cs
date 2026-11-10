// This implementation mirrors OpenCV's saturate_cast semantics.
//
// Original source:
//   modules/core/include/opencv2/core/saturate.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;

/// <summary>
/// Element conversions with OpenCV <c>saturate_cast</c> semantics for all supported depths.
/// Float-to-integer conversions use <see cref="ScalarConverter.CvRound"/>, so NaN and
/// out-of-range values become <see cref="int.MinValue"/> before clamping.
/// </summary>
internal static class Saturate
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsFloating<T>()
        where T : unmanaged
        => typeof(T) == typeof(float) || typeof(T) == typeof(double);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsSmallInteger<T>()
        where T : unmanaged
        => typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte) ||
           typeof(T) == typeof(ushort) || typeof(T) == typeof(short);

    public static void EnsureSupported<T>()
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(sbyte) &&
            typeof(T) != typeof(ushort) && typeof(T) != typeof(short) &&
            typeof(T) != typeof(int) && typeof(T) != typeof(float) &&
            typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"The matrix element type '{typeof(T)}' is not supported.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T FromInt<T>(int v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            byte r = (byte)Math.Clamp(v, 0, 255);
            return Unsafe.As<byte, T>(ref r);
        }

        if (typeof(T) == typeof(sbyte))
        {
            sbyte r = (sbyte)Math.Clamp(v, -128, 127);
            return Unsafe.As<sbyte, T>(ref r);
        }

        if (typeof(T) == typeof(ushort))
        {
            ushort r = (ushort)Math.Clamp(v, 0, 65535);
            return Unsafe.As<ushort, T>(ref r);
        }

        if (typeof(T) == typeof(short))
        {
            short r = (short)Math.Clamp(v, short.MinValue, short.MaxValue);
            return Unsafe.As<short, T>(ref r);
        }

        if (typeof(T) == typeof(int))
        {
            return Unsafe.As<int, T>(ref v);
        }

        if (typeof(T) == typeof(float))
        {
            float r = v;
            return Unsafe.As<float, T>(ref r);
        }

        if (typeof(T) == typeof(double))
        {
            double r = v;
            return Unsafe.As<double, T>(ref r);
        }

        throw new NotSupportedException();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T FromFloat<T>(float v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<float, T>(ref v);
        }

        if (typeof(T) == typeof(double))
        {
            double r = v;
            return Unsafe.As<double, T>(ref r);
        }

        return FromInt<T>(ScalarConverter.CvRound(v));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T FromDouble<T>(double v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(double))
        {
            return Unsafe.As<double, T>(ref v);
        }

        if (typeof(T) == typeof(float))
        {
            float r = (float)v;
            return Unsafe.As<float, T>(ref r);
        }

        return FromInt<T>(ScalarConverter.CvRound(v));
    }

    /// <summary>Exact conversion of an integer-typed element; truncates floating elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ToInt<T>(T v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return Unsafe.As<T, byte>(ref v);
        }

        if (typeof(T) == typeof(sbyte))
        {
            return Unsafe.As<T, sbyte>(ref v);
        }

        if (typeof(T) == typeof(ushort))
        {
            return Unsafe.As<T, ushort>(ref v);
        }

        if (typeof(T) == typeof(short))
        {
            return Unsafe.As<T, short>(ref v);
        }

        if (typeof(T) == typeof(int))
        {
            return Unsafe.As<T, int>(ref v);
        }

        if (typeof(T) == typeof(float))
        {
            return (int)Unsafe.As<T, float>(ref v);
        }

        if (typeof(T) == typeof(double))
        {
            return (int)Unsafe.As<T, double>(ref v);
        }

        throw new NotSupportedException();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat<T>(T v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<T, float>(ref v);
        }

        if (typeof(T) == typeof(double))
        {
            return (float)Unsafe.As<T, double>(ref v);
        }

        return ToInt(v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ToDouble<T>(T v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<T, float>(ref v);
        }

        if (typeof(T) == typeof(double))
        {
            return Unsafe.As<T, double>(ref v);
        }

        return ToInt(v);
    }

    /// <summary>Converts between element types with <c>saturate_cast&lt;TDst&gt;(TSrc)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TDst Convert<TSrc, TDst>(TSrc v)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        if (typeof(TSrc) == typeof(TDst))
        {
            return Unsafe.As<TSrc, TDst>(ref v);
        }

        if (typeof(TSrc) == typeof(float))
        {
            return FromFloat<TDst>(Unsafe.As<TSrc, float>(ref v));
        }

        if (typeof(TSrc) == typeof(double))
        {
            return FromDouble<TDst>(Unsafe.As<TSrc, double>(ref v));
        }

        return FromInt<TDst>(ToInt(v));
    }

    /// <summary>Minimum representable value of an integer element type, as a double.</summary>
    public static double MinValue<T>()
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte)) return byte.MinValue;
        if (typeof(T) == typeof(sbyte)) return sbyte.MinValue;
        if (typeof(T) == typeof(ushort)) return ushort.MinValue;
        if (typeof(T) == typeof(short)) return short.MinValue;
        if (typeof(T) == typeof(int)) return int.MinValue;
        if (typeof(T) == typeof(float)) return float.MinValue;
        return double.MinValue;
    }

    /// <summary>Maximum representable value of an element type, as a double.</summary>
    public static double MaxValue<T>()
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte)) return byte.MaxValue;
        if (typeof(T) == typeof(sbyte)) return sbyte.MaxValue;
        if (typeof(T) == typeof(ushort)) return ushort.MaxValue;
        if (typeof(T) == typeof(short)) return short.MaxValue;
        if (typeof(T) == typeof(int)) return int.MaxValue;
        if (typeof(T) == typeof(float)) return float.MaxValue;
        return double.MaxValue;
    }
}
