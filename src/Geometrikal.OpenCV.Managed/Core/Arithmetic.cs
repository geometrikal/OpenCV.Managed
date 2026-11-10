// This implementation is ported from the OpenCV per-element arithmetic functions.
//
// Original sources:
//   modules/core/src/arithm.cpp (add, subtract, absdiff, multiply, divide, addWeighted,
//     min, max, arithm_op, binary_op)
//   modules/core/src/arithm.simd.hpp
//   modules/core/src/matmul.dispatch.cpp (scaleAdd)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV's per-element arithmetic: <c>add</c>, <c>subtract</c>,
/// <c>absdiff</c>, <c>multiply</c>, <c>divide</c>, <c>addWeighted</c>, <c>scaleAdd</c>,
/// <c>min</c> and <c>max</c>.
/// </summary>
/// <remarks>
/// <para><b>Element types.</b> <c>byte</c>, <c>sbyte</c>, <c>ushort</c>, <c>short</c>,
/// <c>int</c>, <c>float</c> and <c>double</c>, with 1 or more channels. Both operands and the
/// destination have the same element type.</para>
/// <para><b>Overflow.</b> 8- and 16-bit results saturate. <c>int</c> addition, subtraction and
/// absolute difference wrap, as in OpenCV. Results computed in floating point convert with
/// <c>saturate_cast</c>: round half to even; NaN and values outside the <c>int</c> range become
/// the minimum of the element type.</para>
/// <para><b>Scalars.</b> Component <c>c</c> of a <see cref="Scalar"/> applies to channel
/// <c>c</c> (at most 4 channels). For add, subtract and absdiff on integer types the scalar is
/// rounded to an integer first (<c>add(x, 2.5)</c> adds 2). Multiply and divide with a scalar
/// compute in double.</para>
/// <para><b>Results.</b> Overloads without a destination return a new matrix. Overloads with a
/// destination write into it; it must have the shape of the first operand and may be the same
/// matrix as an operand. Where a mask is zero, the destination keeps its values. A mask is a
/// single-channel <c>byte</c> matrix of the same size.</para>
/// </remarks>
public static class Arithmetic
{
    /// <summary>Per-element sum, as OpenCV <c>add(a, b)</c>.</summary>
    public static Mat<T> Add<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Create(a, d => Add(a, b, d));

    /// <summary>Per-element sum into <paramref name="destination"/>, as OpenCV <c>add(a, b, dst, mask)</c>.</summary>
    public static void Add<T>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, mask);
        Elementwise.Binary(a, b, null, destination, mask, default(AddOp<T>));
    }

    /// <summary>Adds a scalar to every element, as OpenCV <c>add(a, Scalar)</c>.</summary>
    public static Mat<T> Add<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Create(a, d => Add(a, b, d));

    /// <summary>Adds a scalar to every element of <paramref name="a"/> into <paramref name="destination"/>.</summary>
    public static void Add<T>(Mat<T> a, Scalar b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => ScalarArithmetic(a, b, destination, mask, ScalarArithm.Add);

    /// <summary>Per-element difference <c>a - b</c>, as OpenCV <c>subtract(a, b)</c>.</summary>
    public static Mat<T> Subtract<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Create(a, d => Subtract(a, b, d));

    /// <summary>Per-element difference <c>a - b</c> into <paramref name="destination"/>.</summary>
    public static void Subtract<T>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, mask);
        Elementwise.Binary(a, b, null, destination, mask, default(SubtractOp<T>));
    }

    /// <summary><c>a - b</c> for a scalar <paramref name="b"/>.</summary>
    public static Mat<T> Subtract<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Create(a, d => Subtract(a, b, d));

    /// <summary><c>a - b</c> for a scalar <paramref name="b"/>, into <paramref name="destination"/>.</summary>
    public static void Subtract<T>(Mat<T> a, Scalar b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => ScalarArithmetic(a, b, destination, mask, ScalarArithm.Subtract);

    /// <summary><c>a - b</c> for a scalar <paramref name="a"/>, as OpenCV <c>subtract(Scalar, b)</c>.</summary>
    public static Mat<T> Subtract<T>(Scalar a, Mat<T> b)
        where T : unmanaged
        => Create(b, d => Subtract(a, b, d));

    /// <summary><c>a - b</c> for a scalar <paramref name="a"/>, into <paramref name="destination"/>.</summary>
    public static void Subtract<T>(Scalar a, Mat<T> b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => ScalarArithmetic(b, a, destination, mask, ScalarArithm.ReverseSubtract);

    /// <summary>Per-element absolute difference, as OpenCV <c>absdiff(a, b)</c>.</summary>
    /// <remarks>For <c>int</c>, the result is the absolute value of the wrapped difference.</remarks>
    public static Mat<T> AbsDiff<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Create(a, d => AbsDiff(a, b, d));

    /// <summary>Per-element absolute difference into <paramref name="destination"/>.</summary>
    public static void AbsDiff<T>(Mat<T> a, Mat<T> b, Mat<T> destination)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, default(AbsDiffOp<T>));
    }

    /// <summary>Absolute difference between every element and a scalar.</summary>
    public static Mat<T> AbsDiff<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Create(a, d => AbsDiff(a, b, d));

    /// <summary>Absolute difference between every element and a scalar, into <paramref name="destination"/>.</summary>
    public static void AbsDiff<T>(Mat<T> a, Scalar b, Mat<T> destination)
        where T : unmanaged
        => ScalarArithmetic(a, b, destination, null, ScalarArithm.AbsDiff);

    /// <summary>
    /// Per-element scaled product <c>a * b * scale</c>, as OpenCV <c>multiply</c>. 8/16-bit and
    /// <c>float</c> compute <c>(a * (float)scale) * b</c> in float; <c>int</c> computes in double.
    /// </summary>
    public static Mat<T> Multiply<T>(Mat<T> a, Mat<T> b, double scale = 1)
        where T : unmanaged
        => Create(a, d => Multiply(a, b, d, scale));

    /// <summary>Per-element scaled product into <paramref name="destination"/>.</summary>
    public static void Multiply<T>(Mat<T> a, Mat<T> b, Mat<T> destination, double scale = 1)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, new MultiplyOp<T>(scale));
    }

    /// <summary>Multiplies every element by a scalar and <paramref name="scale"/>, in double.</summary>
    public static Mat<T> Multiply<T>(Mat<T> a, Scalar b, double scale = 1)
        where T : unmanaged
        => Create(a, d => Multiply(a, b, d, scale));

    /// <summary>Multiplies every element by a scalar and <paramref name="scale"/>, into <paramref name="destination"/>.</summary>
    public static void Multiply<T>(Mat<T> a, Scalar b, Mat<T> destination, double scale = 1)
        where T : unmanaged
        => ScalarScaledOp(a, b, destination, scale, ScalarScaled.Multiply);

    /// <summary>
    /// Per-element quotient <c>a * scale / b</c>, as OpenCV <c>divide</c>. Integer types give 0
    /// where <c>b</c> is 0; floating types follow IEEE rules.
    /// </summary>
    public static Mat<T> Divide<T>(Mat<T> a, Mat<T> b, double scale = 1)
        where T : unmanaged
        => Create(a, d => Divide(a, b, d, scale));

    /// <summary>Per-element quotient into <paramref name="destination"/>.</summary>
    public static void Divide<T>(Mat<T> a, Mat<T> b, Mat<T> destination, double scale = 1)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, new DivideOp<T>(scale));
    }

    /// <summary>
    /// Divides every element by a scalar, in double. Division by a zero component is not
    /// special-cased: for integer types the infinite result converts to the type minimum.
    /// </summary>
    public static Mat<T> Divide<T>(Mat<T> a, Scalar b, double scale = 1)
        where T : unmanaged
        => Create(a, d => Divide(a, b, d, scale));

    /// <summary>Divides every element by a scalar, into <paramref name="destination"/>.</summary>
    public static void Divide<T>(Mat<T> a, Scalar b, Mat<T> destination, double scale = 1)
        where T : unmanaged
        => ScalarScaledOp(a, b, destination, scale, ScalarScaled.Divide);

    /// <summary>
    /// Per-element reciprocal <c>scale / b</c>, as OpenCV <c>divide(scale, b)</c>. Integer types
    /// give 0 where <c>b</c> is 0.
    /// </summary>
    public static Mat<T> Divide<T>(double scale, Mat<T> b)
        where T : unmanaged
        => Create(b, d => Divide(scale, b, d));

    /// <summary>Per-element reciprocal <c>scale / b</c> into <paramref name="destination"/>.</summary>
    public static void Divide<T>(double scale, Mat<T> b, Mat<T> destination)
        where T : unmanaged
    {
        ValidateUnary(b, destination, null);
        Elementwise.Unary(b, destination, null, new ReciprocalOp<T>(scale));
    }

    /// <summary>
    /// Divides a scalar by every element, <c>a * scale / b</c>, in double, as OpenCV
    /// <c>divide(Scalar, b)</c>.
    /// </summary>
    public static Mat<T> Divide<T>(Scalar a, Mat<T> b, double scale = 1)
        where T : unmanaged
        => Create(b, d => Divide(a, b, d, scale));

    /// <summary>Divides a scalar by every element, into <paramref name="destination"/>.</summary>
    public static void Divide<T>(Scalar a, Mat<T> b, Mat<T> destination, double scale = 1)
        where T : unmanaged
        => ScalarScaledOp(b, a, destination, scale, ScalarScaled.ReverseDivide);

    /// <summary>
    /// Weighted sum <c>a * alpha + b * beta + gamma</c>, as OpenCV <c>addWeighted</c>.
    /// 8/16-bit and <c>float</c> compute <c>a*alpha + (b*beta + gamma)</c> with float weights;
    /// <c>int</c> and <c>double</c> compute in double.
    /// </summary>
    public static Mat<T> AddWeighted<T>(Mat<T> a, double alpha, Mat<T> b, double beta, double gamma)
        where T : unmanaged
        => Create(a, d => AddWeighted(a, alpha, b, beta, gamma, d));

    /// <summary>Weighted sum into <paramref name="destination"/>.</summary>
    public static void AddWeighted<T>(Mat<T> a, double alpha, Mat<T> b, double beta, double gamma, Mat<T> destination)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, new AddWeightedOp<T>(alpha, beta, gamma));
    }

    /// <summary>
    /// <c>a * alpha + b</c>, as OpenCV <c>scaleAdd</c>. Integer types use
    /// <see cref="AddWeighted{T}(Mat{T}, double, Mat{T}, double, double)"/> with
    /// <c>beta = 1</c> and <c>gamma = 0</c>.
    /// </summary>
    public static Mat<T> ScaleAdd<T>(Mat<T> a, double alpha, Mat<T> b)
        where T : unmanaged
        => Create(a, d => ScaleAdd(a, alpha, b, d));

    /// <summary><c>a * alpha + b</c> into <paramref name="destination"/>.</summary>
    public static void ScaleAdd<T>(Mat<T> a, double alpha, Mat<T> b, Mat<T> destination)
        where T : unmanaged
    {
        if (!Saturate.IsFloating<T>())
        {
            AddWeighted(a, alpha, b, 1, 0, destination);
            return;
        }

        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, new ScaleAddOp<T>(alpha));
    }

    /// <summary>
    /// Per-element minimum, as OpenCV <c>min</c>: <c>b &lt; a ? b : a</c>. A NaN in
    /// <paramref name="a"/> is kept.
    /// </summary>
    public static Mat<T> Min<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Create(a, d => Min(a, b, d));

    /// <summary>Per-element minimum into <paramref name="destination"/>.</summary>
    public static void Min<T>(Mat<T> a, Mat<T> b, Mat<T> destination)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, default(MinMaxRowOp<T, MinOp>));
    }

    /// <summary>
    /// Per-element minimum with a scalar. The scalar converts to the element type with
    /// <c>saturate_cast</c> first.
    /// </summary>
    public static Mat<T> Min<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Create(a, d => Min(a, b, d));

    /// <summary>Per-element minimum with a scalar, into <paramref name="destination"/>.</summary>
    public static void Min<T>(Mat<T> a, Scalar b, Mat<T> destination)
        where T : unmanaged
    {
        ValidateUnary(a, destination, null);
        T[] row = Elementwise.UnrollScalar(ScalarOperand.ToElements<T>(b, a.Channels), a.RowLength);
        Elementwise.Binary(a, null, row, destination, null, default(MinMaxRowOp<T, MinOp>));
    }

    /// <summary>
    /// Per-element maximum, as OpenCV <c>max</c>: <c>a &lt; b ? b : a</c>. A NaN in
    /// <paramref name="a"/> is kept.
    /// </summary>
    public static Mat<T> Max<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Create(a, d => Max(a, b, d));

    /// <summary>Per-element maximum into <paramref name="destination"/>.</summary>
    public static void Max<T>(Mat<T> a, Mat<T> b, Mat<T> destination)
        where T : unmanaged
    {
        ValidateBinary(a, b, destination, null);
        Elementwise.Binary(a, b, null, destination, null, default(MinMaxRowOp<T, MaxOp>));
    }

    /// <summary>
    /// Per-element maximum with a scalar. The scalar converts to the element type with
    /// <c>saturate_cast</c> first.
    /// </summary>
    public static Mat<T> Max<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Create(a, d => Max(a, b, d));

    /// <summary>Per-element maximum with a scalar, into <paramref name="destination"/>.</summary>
    public static void Max<T>(Mat<T> a, Scalar b, Mat<T> destination)
        where T : unmanaged
    {
        ValidateUnary(a, destination, null);
        T[] row = Elementwise.UnrollScalar(ScalarOperand.ToElements<T>(b, a.Channels), a.RowLength);
        Elementwise.Binary(a, null, row, destination, null, default(MinMaxRowOp<T, MaxOp>));
    }

    internal static Mat<T> Create<T>(Mat<T> shape, Action<Mat<T>> fill)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(shape);
        var result = new Mat<T>(shape.Width, shape.Height, shape.Channels, clear: false);
        fill(result);
        return result;
    }

    internal static void ValidateUnary<T>(Mat<T> a, Mat<T> destination, Mat<byte>? mask)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(a);
        Saturate.EnsureSupported<T>();
        Elementwise.ValidateSameShape(a, destination, nameof(destination));
        Elementwise.ValidateMask(a, mask);
    }

    internal static void ValidateBinary<T>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask)
        where T : unmanaged
    {
        ValidateUnary(a, destination, mask);
        Elementwise.ValidateSameShape(a, b, nameof(b));
    }

    private static void ScalarArithmetic<T>(Mat<T> a, Scalar s, Mat<T> destination, Mat<byte>? mask, ScalarArithm op)
        where T : unmanaged
    {
        ValidateUnary(a, destination, mask);
        int[]? work = ScalarOperand.IntWorkValues(s, a.Channels, out T[] native);
        if (work is not null)
        {
            int[] row = Elementwise.UnrollScalar(work, a.RowLength);
            Elementwise.Unary(a, destination, mask, new IntWorkScalarOp<T>(row, op));
            return;
        }

        T[] nativeRow = Elementwise.UnrollScalar(native, a.RowLength);
        switch (op)
        {
            case ScalarArithm.Add:
                Elementwise.Binary(a, null, nativeRow, destination, mask, default(AddOp<T>));
                break;
            case ScalarArithm.Subtract:
                Elementwise.Binary(a, null, nativeRow, destination, mask, default(SubtractOp<T>));
                break;
            case ScalarArithm.ReverseSubtract:
                Elementwise.Binary(a, null, nativeRow, destination, mask, default(ReverseSubtractOp<T>));
                break;
            default:
                Elementwise.Binary(a, null, nativeRow, destination, mask, default(AbsDiffOp<T>));
                break;
        }
    }

    private static void ScalarScaledOp<T>(Mat<T> a, Scalar s, Mat<T> destination, double scale, ScalarScaled op)
        where T : unmanaged
    {
        ValidateUnary(a, destination, null);
        if (a.Channels > 4)
        {
            throw new ArgumentException("A scalar operand supports at most 4 channels.", nameof(a));
        }

        var perChannel = new double[a.Channels];
        for (int c = 0; c < perChannel.Length; c++)
        {
            perChannel[c] = s[c];
        }

        double[] row = Elementwise.UnrollScalar(perChannel, a.RowLength);
        Elementwise.Unary(a, destination, null, new DoubleWorkScalarOp<T>(row, scale, op));
    }
}
