// This implementation is ported from the OpenCV bitwise functions.
//
// Original sources:
//   modules/core/src/arithm.cpp (bitwise_and, bitwise_or, bitwise_xor, bitwise_not, binary_op)
//   modules/core/src/arithm.simd.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>bitwise_and</c>, <c>bitwise_or</c>, <c>bitwise_xor</c> and
/// <c>bitwise_not</c>.
/// </summary>
/// <remarks>
/// <para>The operations act on the raw bytes of each element, for every element type
/// (floating types use their IEEE bit patterns).</para>
/// <para>A <see cref="Scalar"/> operand converts to the element type with
/// <c>saturate_cast</c> per channel before the bitwise operation.</para>
/// <para>Overloads without a destination return a new matrix. Overloads with a destination
/// write into it; it may be the same matrix as an operand. Where a mask is zero, the
/// destination keeps its values.</para>
/// </remarks>
public static class Bitwise
{
    /// <summary>Per-element bitwise AND, as OpenCV <c>bitwise_and</c>.</summary>
    public static Mat<T> And<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Arithmetic.Create(a, d => And(a, b, d));

    /// <summary>Per-element bitwise AND into <paramref name="destination"/>.</summary>
    public static void And<T>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => Run<T, AndBits>(a, b, destination, mask);

    /// <summary>Bitwise AND of every element with a scalar.</summary>
    public static Mat<T> And<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Arithmetic.Create(a, d => And(a, b, d));

    /// <summary>Bitwise AND of every element with a scalar, into <paramref name="destination"/>.</summary>
    public static void And<T>(Mat<T> a, Scalar b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => Run<T, AndBits>(a, b, destination, mask);

    /// <summary>Per-element bitwise OR, as OpenCV <c>bitwise_or</c>.</summary>
    public static Mat<T> Or<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Arithmetic.Create(a, d => Or(a, b, d));

    /// <summary>Per-element bitwise OR into <paramref name="destination"/>.</summary>
    public static void Or<T>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => Run<T, OrBits>(a, b, destination, mask);

    /// <summary>Bitwise OR of every element with a scalar.</summary>
    public static Mat<T> Or<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Arithmetic.Create(a, d => Or(a, b, d));

    /// <summary>Bitwise OR of every element with a scalar, into <paramref name="destination"/>.</summary>
    public static void Or<T>(Mat<T> a, Scalar b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => Run<T, OrBits>(a, b, destination, mask);

    /// <summary>Per-element bitwise XOR, as OpenCV <c>bitwise_xor</c>.</summary>
    public static Mat<T> Xor<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => Arithmetic.Create(a, d => Xor(a, b, d));

    /// <summary>Per-element bitwise XOR into <paramref name="destination"/>.</summary>
    public static void Xor<T>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => Run<T, XorBits>(a, b, destination, mask);

    /// <summary>Bitwise XOR of every element with a scalar.</summary>
    public static Mat<T> Xor<T>(Mat<T> a, Scalar b)
        where T : unmanaged
        => Arithmetic.Create(a, d => Xor(a, b, d));

    /// <summary>Bitwise XOR of every element with a scalar, into <paramref name="destination"/>.</summary>
    public static void Xor<T>(Mat<T> a, Scalar b, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
        => Run<T, XorBits>(a, b, destination, mask);

    /// <summary>Per-element bitwise inversion, as OpenCV <c>bitwise_not</c>.</summary>
    public static Mat<T> Not<T>(Mat<T> a)
        where T : unmanaged
        => Arithmetic.Create(a, d => Not(a, d));

    /// <summary>Per-element bitwise inversion into <paramref name="destination"/>.</summary>
    public static void Not<T>(Mat<T> a, Mat<T> destination, Mat<byte>? mask = null)
        where T : unmanaged
    {
        Arithmetic.ValidateUnary(a, destination, mask);
        Elementwise.Unary(a, destination, mask, default(BitwiseNotOp<T>));
    }

    private static void Run<T, TBits>(Mat<T> a, Mat<T> b, Mat<T> destination, Mat<byte>? mask)
        where T : unmanaged
        where TBits : struct, IBitwiseOp
    {
        Arithmetic.ValidateBinary(a, b, destination, mask);
        Elementwise.Binary(a, b, null, destination, mask, default(BitwiseOp<T, TBits>));
    }

    private static void Run<T, TBits>(Mat<T> a, Scalar b, Mat<T> destination, Mat<byte>? mask)
        where T : unmanaged
        where TBits : struct, IBitwiseOp
    {
        Arithmetic.ValidateUnary(a, destination, mask);
        T[] row = Elementwise.UnrollScalar(ScalarOperand.ToElements<T>(b, a.Channels), a.RowLength);
        Elementwise.Binary(a, null, row, destination, mask, default(BitwiseOp<T, TBits>));
    }
}
