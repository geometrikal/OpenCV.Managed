// This implementation mirrors how OpenCV's arithm_op applies a scalar operand
// (work-type selection and convertAndUnrollScalar).
//
// Original sources:
//   modules/core/src/arithm.cpp (arithm_op, actualScalarDepth)
//   modules/core/src/copy.cpp (convertAndUnrollScalar)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

internal enum ScalarArithm
{
    Add,
    Subtract,
    ReverseSubtract,
    AbsDiff,
}

/// <summary>
/// add, subtract or absdiff with an <see cref="int"/> work value per element: the 8/16-bit
/// source widens to <see cref="int"/>, the operation wraps in 32 bits, and the result saturates.
/// </summary>
internal readonly struct IntWorkScalarOp<T>(int[] row, ScalarArithm op) : IUnaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, Span<T> dst)
    {
        for (int i = 0; i < a.Length; i++)
        {
            int x = Saturate.ToInt(a[i]);
            int s = row[i];
            int r = op switch
            {
                ScalarArithm.Add => unchecked(x + s),
                ScalarArithm.Subtract => unchecked(x - s),
                ScalarArithm.ReverseSubtract => unchecked(s - x),
                _ => AbsInt32(unchecked(x - s)),
            };
            dst[i] = Saturate.FromInt<T>(r);
        }
    }

    private static int AbsInt32(int d) => d < 0 ? unchecked(-d) : d;
}

internal enum ScalarScaled
{
    Multiply,
    Divide,
    ReverseDivide,
}

/// <summary>multiply or divide with a scalar, computed in double for every depth.</summary>
internal readonly struct DoubleWorkScalarOp<T>(double[] row, double scale, ScalarScaled op) : IUnaryRowOp<T, T>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, Span<T> dst)
    {
        for (int i = 0; i < a.Length; i++)
        {
            double x = Saturate.ToDouble(a[i]);
            double s = row[i];
            double r = op switch
            {
                ScalarScaled.Multiply => x * scale * s,
                ScalarScaled.Divide => x * scale / s,
                _ => s * scale / x,
            };
            dst[i] = Saturate.FromDouble<T>(r);
        }
    }
}

internal static class ScalarOperand
{
    /// <summary>
    /// Converts the first <paramref name="cn"/> scalar components to the element type with
    /// <c>saturate_cast</c>, as <c>convertAndUnrollScalar</c>.
    /// </summary>
    public static T[] ToElements<T>(Scalar s, int cn)
        where T : unmanaged
    {
        if (cn > 4)
        {
            throw new ArgumentException("A scalar operand supports at most 4 channels.");
        }

        return ScalarConverter.ToRawColor<T>(s, cn);
    }

    /// <summary>
    /// Integer work values (<c>cvRound</c>) for add/subtract/absdiff on 8/16-bit and
    /// <see cref="int"/> sources, or null when the elements fit the source type and the native
    /// kernel gives the same result.
    /// </summary>
    public static int[]? IntWorkValues<T>(Scalar s, int cn, out T[] native)
        where T : unmanaged
    {
        native = ToElements<T>(s, cn);
        if (Saturate.IsFloating<T>() || typeof(T) == typeof(int))
        {
            return null;
        }

        var work = new int[cn];
        bool fits = true;
        for (int c = 0; c < cn; c++)
        {
            work[c] = ScalarConverter.CvRound(s[c]);
            fits &= work[c] >= Saturate.MinValue<T>() && work[c] <= Saturate.MaxValue<T>();
        }

        return fits ? null : work;
    }
}
