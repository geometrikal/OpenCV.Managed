// This implementation is ported from the OpenCV type-conversion kernels.
//
// Original sources:
//   modules/core/src/convert.simd.hpp (cvt_)
//   modules/core/src/convert_scale.simd.hpp (cvt_32f, cvt_64f, cvtabs_32f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary><c>saturate_cast&lt;TDst&gt;(x)</c> per element.</summary>
internal readonly struct ConvertOp<TSrc, TDst> : IUnaryRowOp<TSrc, TDst>
    where TSrc : unmanaged
    where TDst : unmanaged
{
    public void Apply(ReadOnlySpan<TSrc> a, Span<TDst> dst)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            if (typeof(TSrc) == typeof(byte) && typeof(TDst) == typeof(float))
            {
                i = ConvertVectors.ByteToFloat(MemoryMarshal.Cast<TSrc, byte>(a), MemoryMarshal.Cast<TDst, float>(dst), 1f, 0f, scaled: false);
            }
            else if (typeof(TSrc) == typeof(float) && typeof(TDst) == typeof(byte))
            {
                i = ConvertVectors.FloatToByte(MemoryMarshal.Cast<TSrc, float>(a), MemoryMarshal.Cast<TDst, byte>(dst), 1f, 0f, scaled: false);
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = Saturate.Convert<TSrc, TDst>(a[i]);
        }
    }
}

/// <summary>Scaled conversion with float work: <c>(float)x * alpha + beta</c>.</summary>
internal readonly struct ConvertScaleFloatOp<TSrc, TDst>(float alpha, float beta) : IUnaryRowOp<TSrc, TDst>
    where TSrc : unmanaged
    where TDst : unmanaged
{
    public void Apply(ReadOnlySpan<TSrc> a, Span<TDst> dst)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            if (typeof(TSrc) == typeof(byte) && typeof(TDst) == typeof(float))
            {
                i = ConvertVectors.ByteToFloat(MemoryMarshal.Cast<TSrc, byte>(a), MemoryMarshal.Cast<TDst, float>(dst), alpha, beta, scaled: true);
            }
            else if (typeof(TSrc) == typeof(float) && typeof(TDst) == typeof(byte))
            {
                i = ConvertVectors.FloatToByte(MemoryMarshal.Cast<TSrc, float>(a), MemoryMarshal.Cast<TDst, byte>(dst), alpha, beta, scaled: true);
            }
            else if (typeof(TSrc) == typeof(float) && typeof(TDst) == typeof(float))
            {
                var fa = MemoryMarshal.Cast<TSrc, float>(a);
                var fd = MemoryMarshal.Cast<TDst, float>(dst);
                var va = new Vector<float>(alpha);
                var vb = new Vector<float>(beta);
                for (; i <= fa.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    ((new Vector<float>(fa[i..]) * va) + vb).CopyTo(fd[i..]);
                }
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = Saturate.FromFloat<TDst>((Saturate.ToFloat(a[i]) * alpha) + beta);
        }
    }
}

/// <summary>Scaled conversion with double work: <c>(double)x * alpha + beta</c>.</summary>
internal readonly struct ConvertScaleDoubleOp<TSrc, TDst>(double alpha, double beta) : IUnaryRowOp<TSrc, TDst>
    where TSrc : unmanaged
    where TDst : unmanaged
{
    public void Apply(ReadOnlySpan<TSrc> a, Span<TDst> dst)
    {
        for (int i = 0; i < a.Length; i++)
        {
            dst[i] = Saturate.FromDouble<TDst>((Saturate.ToDouble(a[i]) * alpha) + beta);
        }
    }
}

/// <summary><c>saturate_cast&lt;byte&gt;(|(float)x * alpha + beta|)</c>.</summary>
internal readonly struct ConvertScaleAbsOp<T>(float alpha, float beta) : IUnaryRowOp<T, byte>
    where T : unmanaged
{
    public void Apply(ReadOnlySpan<T> a, Span<byte> dst)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var va = new Vector<float>(alpha);
            var vb = new Vector<float>(beta);
            if (typeof(T) == typeof(byte))
            {
                var ba = MemoryMarshal.Cast<T, byte>(a);
                for (; i <= ba.Length - Vector<byte>.Count; i += Vector<byte>.Count)
                {
                    FilterVectors.WidenToSingle(new Vector<byte>(ba[i..]), out var f0, out var f1, out var f2, out var f3);
                    FilterVectors.SaturateToByte(
                        FilterVectors.CvRound(Vector.Abs((f0 * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((f1 * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((f2 * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((f3 * va) + vb))).CopyTo(dst[i..]);
                }
            }
            else if (typeof(T) == typeof(short))
            {
                var sa = MemoryMarshal.Cast<T, short>(a);
                for (; i <= sa.Length - (2 * Vector<short>.Count); i += 2 * Vector<short>.Count)
                {
                    Vector.Widen(new Vector<short>(sa[i..]), out Vector<int> w0, out Vector<int> w1);
                    Vector.Widen(new Vector<short>(sa[(i + Vector<short>.Count)..]), out Vector<int> w2, out Vector<int> w3);
                    FilterVectors.SaturateToByte(
                        FilterVectors.CvRound(Vector.Abs((Vector.ConvertToSingle(w0) * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((Vector.ConvertToSingle(w1) * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((Vector.ConvertToSingle(w2) * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((Vector.ConvertToSingle(w3) * va) + vb))).CopyTo(dst[i..]);
                }
            }
            else if (typeof(T) == typeof(float))
            {
                var fa = MemoryMarshal.Cast<T, float>(a);
                int n = Vector<float>.Count;
                for (; i <= fa.Length - (4 * n); i += 4 * n)
                {
                    FilterVectors.SaturateToByte(
                        FilterVectors.CvRound(Vector.Abs((new Vector<float>(fa[i..]) * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((new Vector<float>(fa[(i + n)..]) * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((new Vector<float>(fa[(i + (2 * n))..]) * va) + vb)),
                        FilterVectors.CvRound(Vector.Abs((new Vector<float>(fa[(i + (3 * n))..]) * va) + vb))).CopyTo(dst[i..]);
                }
            }
        }

        for (; i < a.Length; i++)
        {
            dst[i] = Saturate.FromFloat<byte>(MathF.Abs((Saturate.ToFloat(a[i]) * alpha) + beta));
        }
    }
}

internal static class ConvertVectors
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ByteToFloat(ReadOnlySpan<byte> a, Span<float> dst, float alpha, float beta, bool scaled)
    {
        int i = 0;
        int n = Vector<float>.Count;
        var va = new Vector<float>(alpha);
        var vb = new Vector<float>(beta);
        for (; i <= a.Length - Vector<byte>.Count; i += Vector<byte>.Count)
        {
            FilterVectors.WidenToSingle(new Vector<byte>(a[i..]), out var f0, out var f1, out var f2, out var f3);
            if (scaled)
            {
                f0 = (f0 * va) + vb;
                f1 = (f1 * va) + vb;
                f2 = (f2 * va) + vb;
                f3 = (f3 * va) + vb;
            }

            f0.CopyTo(dst[i..]);
            f1.CopyTo(dst[(i + n)..]);
            f2.CopyTo(dst[(i + (2 * n))..]);
            f3.CopyTo(dst[(i + (3 * n))..]);
        }

        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FloatToByte(ReadOnlySpan<float> a, Span<byte> dst, float alpha, float beta, bool scaled)
    {
        int i = 0;
        int n = Vector<float>.Count;
        var va = new Vector<float>(alpha);
        var vb = new Vector<float>(beta);
        for (; i <= a.Length - (4 * n); i += 4 * n)
        {
            var f0 = new Vector<float>(a[i..]);
            var f1 = new Vector<float>(a[(i + n)..]);
            var f2 = new Vector<float>(a[(i + (2 * n))..]);
            var f3 = new Vector<float>(a[(i + (3 * n))..]);
            if (scaled)
            {
                f0 = (f0 * va) + vb;
                f1 = (f1 * va) + vb;
                f2 = (f2 * va) + vb;
                f3 = (f3 * va) + vb;
            }

            FilterVectors.SaturateToByte(
                FilterVectors.CvRound(f0), FilterVectors.CvRound(f1),
                FilterVectors.CvRound(f2), FilterVectors.CvRound(f3)).CopyTo(dst[i..]);
        }

        return i;
    }
}
