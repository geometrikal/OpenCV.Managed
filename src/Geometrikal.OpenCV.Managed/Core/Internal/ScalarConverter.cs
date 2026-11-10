// This implementation mirrors OpenCV's scalarToRawData / saturate_cast /
// cvRound semantics.
//
// Original sources:
//   modules/core/src/copy.cpp (scalarToRawData)
//   modules/core/include/opencv2/core/saturate.hpp
//   modules/core/include/opencv2/core/fast_math.hpp (cvRound, cvFloor, cvCeil)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;

/// <summary>
/// Converts <see cref="Scalar"/> components to matrix element types with
/// OpenCV's saturating semantics. Shared by the drawing primitives (colours)
/// and border construction (constant border values).
/// </summary>
internal static class ScalarConverter
{
    /// <summary>
    /// OpenCV's <c>cvRound(double)</c> as it behaves on x86-64
    /// (<c>_mm_cvtsd_si32</c>): round-half-to-even for values inside the int32
    /// range, and the 0x80000000 "integer indefinite" sentinel
    /// (<see cref="int.MinValue"/>) for NaN or out-of-range values. The sentinel
    /// then saturates to the element type's MINIMUM, so e.g. a Scalar component
    /// of 3e9 produces 0 on byte images — matching OpenCV, not .NET's saturating
    /// float-to-int conversion (which would go to the maximum).
    /// </summary>
    public static int CvRound(double v)
    {
        double rounded = Math.Round(v, MidpointRounding.ToEven);
        return rounded >= int.MinValue && rounded <= int.MaxValue
            ? (int)rounded
            : int.MinValue; // NaN comparisons are false, so NaN lands here too.
    }

    /// <summary>
    /// OpenCV's <c>cvFloor(double)</c> as it behaves on x86-64:
    /// <c>i = (int)v; return i - (i &gt; v)</c> with a <c>cvttsd2si</c> truncation and wrapping
    /// arithmetic. NaN and +inf give <see cref="int.MinValue"/>; -inf and values below the
    /// <c>int</c> range give <see cref="int.MaxValue"/>.
    /// </summary>
    public static int CvFloor(double v)
    {
        int i = TruncX86(v);
        return unchecked(i - (i > v ? 1 : 0));
    }

    /// <summary>
    /// OpenCV's <c>cvCeil(double)</c> as it behaves on x86-64:
    /// <c>i = (int)v; return i + (i &lt; v)</c> with a <c>cvttsd2si</c> truncation and wrapping
    /// arithmetic. NaN and -inf give <see cref="int.MinValue"/>; +inf gives
    /// <c>int.MinValue + 1</c>.
    /// </summary>
    public static int CvCeil(double v)
    {
        int i = TruncX86(v);
        return unchecked(i + (i < v ? 1 : 0));
    }

    /// <summary><c>cvttsd2si</c>: truncation, with <see cref="int.MinValue"/> for NaN and out-of-range values.</summary>
    private static int TruncX86(double v)
        => v > -2147483649.0 && v < 2147483648.0 ? (int)v : int.MinValue;

    /// <summary>
    /// Converts one <see cref="Scalar"/> component per channel into a raw
    /// per-channel value array, mirroring OpenCV's <c>scalarToRawData</c>.
    /// </summary>
    /// <exception cref="NotSupportedException">Thrown for unsupported element types.</exception>
    public static T[] ToRawColor<T>(Scalar color, int channels)
        where T : unmanaged
    {
        var result = new T[channels];
        for (int c = 0; c < channels; c++)
        {
            result[c] = SaturateCast<T>(color[c]);
        }

        return result;
    }

    /// <summary>
    /// OpenCV <c>saturate_cast&lt;T&gt;(double)</c> for the element types the
    /// library supports.
    /// </summary>
    public static T SaturateCast<T>(double v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            byte value = (byte)Math.Clamp(CvRound(v), 0, 255);
            return Unsafe.As<byte, T>(ref value);
        }

        if (typeof(T) == typeof(sbyte))
        {
            sbyte value = (sbyte)Math.Clamp(CvRound(v), -128, 127);
            return Unsafe.As<sbyte, T>(ref value);
        }

        if (typeof(T) == typeof(ushort))
        {
            ushort value = (ushort)Math.Clamp(CvRound(v), 0, 65535);
            return Unsafe.As<ushort, T>(ref value);
        }

        if (typeof(T) == typeof(short))
        {
            short value = (short)Math.Clamp(CvRound(v), short.MinValue, short.MaxValue);
            return Unsafe.As<short, T>(ref value);
        }

        if (typeof(T) == typeof(int))
        {
            int value = CvRound(v);
            return Unsafe.As<int, T>(ref value);
        }

        if (typeof(T) == typeof(float))
        {
            float value = (float)v;
            return Unsafe.As<float, T>(ref value);
        }

        if (typeof(T) == typeof(double))
        {
            double value = v;
            return Unsafe.As<double, T>(ref value);
        }

        throw new NotSupportedException(
            $"The matrix element type '{typeof(T)}' is not supported.");
    }
}
