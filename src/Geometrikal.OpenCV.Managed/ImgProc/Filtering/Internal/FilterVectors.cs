namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;

/// <summary>
/// <see cref="Vector{T}"/> helpers for the filter engines. A block of
/// <c>Vector&lt;byte&gt;.Count</c> source bytes widens to four <see cref="Vector{T}"/> of
/// <see cref="int"/> or <see cref="float"/>, in element order.
/// </summary>
internal static class FilterVectors
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void WidenToInt32(
        Vector<byte> v, out Vector<int> w0, out Vector<int> w1, out Vector<int> w2, out Vector<int> w3)
    {
        Vector.Widen(v, out Vector<ushort> lo, out Vector<ushort> hi);
        Vector.Widen(lo, out Vector<uint> u0, out Vector<uint> u1);
        Vector.Widen(hi, out Vector<uint> u2, out Vector<uint> u3);
        w0 = Vector.AsVectorInt32(u0);
        w1 = Vector.AsVectorInt32(u1);
        w2 = Vector.AsVectorInt32(u2);
        w3 = Vector.AsVectorInt32(u3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void WidenToSingle(
        Vector<byte> v, out Vector<float> f0, out Vector<float> f1, out Vector<float> f2, out Vector<float> f3)
    {
        WidenToInt32(v, out Vector<int> w0, out Vector<int> w1, out Vector<int> w2, out Vector<int> w3);
        f0 = Vector.ConvertToSingle(w0);
        f1 = Vector.ConvertToSingle(w1);
        f2 = Vector.ConvertToSingle(w2);
        f3 = Vector.ConvertToSingle(w3);
    }

    /// <summary>
    /// <see cref="Core.Internal.ScalarConverter.CvRound"/> per lane: round half to even;
    /// NaN and values outside the <see cref="int"/> range map to <see cref="int.MinValue"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<int> CvRound(Vector<float> v)
    {
        Vector<float> r = Vector.Round(v);
        Vector<int> inRange = Vector.AsVectorInt32(
            Vector.GreaterThanOrEqual(r, new Vector<float>(-2147483648f))
            & Vector.LessThan(r, new Vector<float>(2147483648f)));
        return Vector.ConditionalSelect(inRange, Vector.ConvertToInt32(r), new Vector<int>(int.MinValue));
    }

    /// <summary>Saturates four int vectors to bytes, in element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<byte> SaturateToByte(Vector<int> a, Vector<int> b, Vector<int> c, Vector<int> d)
    {
        var zero = Vector<int>.Zero;
        var max = new Vector<int>(255);
        Vector<short> lo = Vector.Narrow(Vector.Min(Vector.Max(a, zero), max), Vector.Min(Vector.Max(b, zero), max));
        Vector<short> hi = Vector.Narrow(Vector.Min(Vector.Max(c, zero), max), Vector.Min(Vector.Max(d, zero), max));
        return Vector.Narrow(Vector.AsVectorUInt16(lo), Vector.AsVectorUInt16(hi));
    }

    /// <summary>Saturates two int vectors to shorts, in element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<short> SaturateToInt16(Vector<int> a, Vector<int> b)
    {
        var min = new Vector<int>(short.MinValue);
        var max = new Vector<int>(short.MaxValue);
        return Vector.Narrow(Vector.Min(Vector.Max(a, min), max), Vector.Min(Vector.Max(b, min), max));
    }
}
