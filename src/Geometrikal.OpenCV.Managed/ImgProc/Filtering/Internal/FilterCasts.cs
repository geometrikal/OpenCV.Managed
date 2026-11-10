// This implementation mirrors OpenCV's saturate_cast semantics for filter
// results (Cast<float, T> and Cast<double, T> in modules/imgproc/src/filter.simd.hpp).
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <c>saturate_cast&lt;TDst&gt;</c> applied to rows of float or double filter sums, and exact
/// widening of source rows to the buffer type, shared by the separable and 2-D convolution engines.
/// </summary>
internal static class FilterCasts
{
    /// <summary>Stores a row of float sums into the destination row with OpenCV's cast semantics.</summary>
    public static void StoreRow<TDst>(Span<TDst> destination, ReadOnlySpan<float> sums)
        where TDst : unmanaged
    {
        if (typeof(TDst) == typeof(float))
        {
            var df = MemoryMarshal.Cast<TDst, float>(destination);
            sums[..df.Length].CopyTo(df);
        }
        else if (typeof(TDst) == typeof(byte))
        {
            var db = MemoryMarshal.Cast<TDst, byte>(destination);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                ref float s = ref MemoryMarshal.GetReference(sums);
                ref byte d = ref MemoryMarshal.GetReference(db);
                int vf = Vector<float>.Count;
                for (; i <= db.Length - Vector<byte>.Count; i += Vector<byte>.Count)
                {
                    FilterVectors.SaturateToByte(
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)i)),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)(i + vf))),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)(i + (2 * vf)))),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)(i + (3 * vf)))))
                        .StoreUnsafe(ref d, (nuint)i);
                }
            }

            for (; i < db.Length; i++)
            {
                db[i] = (byte)Math.Clamp(ScalarConverter.CvRound(sums[i]), 0, 255);
            }
        }
        else if (typeof(TDst) == typeof(short))
        {
            var ds = MemoryMarshal.Cast<TDst, short>(destination);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                ref float s = ref MemoryMarshal.GetReference(sums);
                ref short d = ref MemoryMarshal.GetReference(ds);
                int vf = Vector<float>.Count;
                for (; i <= ds.Length - Vector<short>.Count; i += Vector<short>.Count)
                {
                    FilterVectors.SaturateToInt16(
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)i)),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)(i + vf))))
                        .StoreUnsafe(ref d, (nuint)i);
                }
            }

            for (; i < ds.Length; i++)
            {
                ds[i] = (short)Math.Clamp(ScalarConverter.CvRound(sums[i]), short.MinValue, short.MaxValue);
            }
        }
        else if (typeof(TDst) == typeof(ushort))
        {
            var du = MemoryMarshal.Cast<TDst, ushort>(destination);
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                ref float s = ref MemoryMarshal.GetReference(sums);
                ref ushort d = ref MemoryMarshal.GetReference(du);
                int vf = Vector<float>.Count;
                var zero = Vector<int>.Zero;
                var max = new Vector<int>(ushort.MaxValue);
                for (; i <= du.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
                {
                    Vector<int> a = Vector.Min(Vector.Max(FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)i)), zero), max);
                    Vector<int> b = Vector.Min(Vector.Max(FilterVectors.CvRound(Vector.LoadUnsafe(ref s, (nuint)(i + vf))), zero), max);
                    Vector.Narrow(Vector.AsVectorUInt32(a), Vector.AsVectorUInt32(b)).StoreUnsafe(ref d, (nuint)i);
                }
            }

            for (; i < du.Length; i++)
            {
                du[i] = (ushort)Math.Clamp(ScalarConverter.CvRound(sums[i]), 0, ushort.MaxValue);
            }
        }
        else
        {
            throw new NotSupportedException($"Unsupported filter destination type '{typeof(TDst)}'.");
        }
    }

    /// <summary>Stores a row of double sums into the destination row with OpenCV's cast semantics.</summary>
    public static void StoreRow<TDst>(Span<TDst> destination, ReadOnlySpan<double> sums)
        where TDst : unmanaged
    {
        if (typeof(TDst) == typeof(double))
        {
            var dd = MemoryMarshal.Cast<TDst, double>(destination);
            sums[..dd.Length].CopyTo(dd);
            return;
        }

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Saturate.FromDouble<TDst>(sums[i]);
        }
    }

    /// <summary>Converts a row of 8-bit, 16-bit or float elements to float; the conversion is exact.</summary>
    public static void WidenRow<TSrc>(ReadOnlySpan<TSrc> source, Span<float> destination)
        where TSrc : unmanaged
    {
        int n = source.Length;
        int i = 0;
        ref float d = ref MemoryMarshal.GetReference(destination);
        if (typeof(TSrc) == typeof(float))
        {
            MemoryMarshal.Cast<TSrc, float>(source).CopyTo(destination);
            return;
        }

        if (typeof(TSrc) == typeof(byte))
        {
            ref byte s = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<TSrc, byte>(source));
            if (Vector.IsHardwareAccelerated)
            {
                int vf = Vector<float>.Count;
                for (; i <= n - Vector<byte>.Count; i += Vector<byte>.Count)
                {
                    FilterVectors.WidenToSingle(
                        Vector.LoadUnsafe(ref s, (nuint)i),
                        out Vector<float> f0, out Vector<float> f1, out Vector<float> f2, out Vector<float> f3);
                    f0.StoreUnsafe(ref d, (nuint)i);
                    f1.StoreUnsafe(ref d, (nuint)(i + vf));
                    f2.StoreUnsafe(ref d, (nuint)(i + (2 * vf)));
                    f3.StoreUnsafe(ref d, (nuint)(i + (3 * vf)));
                }
            }

            for (; i < n; i++)
            {
                Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
            }
        }
        else if (typeof(TSrc) == typeof(ushort))
        {
            ref ushort s = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<TSrc, ushort>(source));
            if (Vector.IsHardwareAccelerated)
            {
                int vf = Vector<float>.Count;
                for (; i <= n - Vector<ushort>.Count; i += Vector<ushort>.Count)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<uint> lo, out Vector<uint> hi);
                    Vector.ConvertToSingle(Vector.AsVectorInt32(lo)).StoreUnsafe(ref d, (nuint)i);
                    Vector.ConvertToSingle(Vector.AsVectorInt32(hi)).StoreUnsafe(ref d, (nuint)(i + vf));
                }
            }

            for (; i < n; i++)
            {
                Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
            }
        }
        else if (typeof(TSrc) == typeof(short))
        {
            ref short s = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<TSrc, short>(source));
            if (Vector.IsHardwareAccelerated)
            {
                int vf = Vector<float>.Count;
                for (; i <= n - Vector<short>.Count; i += Vector<short>.Count)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<int> lo, out Vector<int> hi);
                    Vector.ConvertToSingle(lo).StoreUnsafe(ref d, (nuint)i);
                    Vector.ConvertToSingle(hi).StoreUnsafe(ref d, (nuint)(i + vf));
                }
            }

            for (; i < n; i++)
            {
                Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
            }
        }
        else
        {
            throw new NotSupportedException($"Unsupported filter source type '{typeof(TSrc)}'.");
        }
    }

    /// <summary>Converts a row of filter source elements to double; the conversion is exact.</summary>
    public static void WidenRow<TSrc>(ReadOnlySpan<TSrc> source, Span<double> destination)
        where TSrc : unmanaged
    {
        if (typeof(TSrc) == typeof(double))
        {
            MemoryMarshal.Cast<TSrc, double>(source).CopyTo(destination);
            return;
        }

        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = Saturate.ToDouble(source[i]);
        }
    }
}
