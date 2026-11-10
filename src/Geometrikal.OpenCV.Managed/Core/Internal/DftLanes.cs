namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// Element arithmetic of the DFT kernels. <typeparamref name="TV"/> is either one value of
/// <typeparamref name="T"/> or a <see cref="Vector{T}"/> whose lanes hold independent
/// transforms. Every lane sees the scalar IEEE operation sequence; no operation is fused.
/// </summary>
internal interface IDftLanes<TV, T>
    where TV : unmanaged
    where T : unmanaged
{
    static abstract int Count { get; }

    static abstract TV Add(TV a, TV b);

    static abstract TV Sub(TV a, TV b);

    static abstract TV Mul(TV a, TV b);

    /// <summary>Sign flip, as C++ unary minus (also for zeros and NaN).</summary>
    static abstract TV Neg(TV a);

    static abstract TV Splat(T a);
}

internal readonly struct ScalarLanes<T> : IDftLanes<T, T>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
{
    public static int Count => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Add(T a, T b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Sub(T a, T b) => a - b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Mul(T a, T b) => a * b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Neg(T a) => -a;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Splat(T a) => a;
}

internal readonly struct VectorLanes<T> : IDftLanes<Vector<T>, T>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
{
    public static int Count => Vector<T>.Count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Add(Vector<T> a, Vector<T> b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Sub(Vector<T> a, Vector<T> b) => a - b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Mul(Vector<T> a, Vector<T> b) => a * b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Neg(Vector<T> a) => a ^ new Vector<T>(T.NegativeZero);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Splat(T a) => new(a);
}
