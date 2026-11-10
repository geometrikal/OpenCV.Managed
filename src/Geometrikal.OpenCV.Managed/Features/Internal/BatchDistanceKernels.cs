// This implementation is ported from the OpenCV descriptor distance functions.
//
// Original sources:
//   modules/core/src/batch_distance.cpp (batchDistL1_, batchDistL2Sqr_, batchDistL2_, batchDistHamming*)
//   modules/core/src/norm.dispatch.cpp (hal::normL2Sqr_, hal::normHamming with cellSize, popCountTable2)
//   modules/core/src/stat.simd.hpp (hal::normHamming)
//   modules/core/include/opencv2/core/base.hpp (normL1, normL2Sqr templates)
//   modules/core/include/opencv2/core/hal/intrin_sse.hpp (v_reduce_sum for v_float32x4)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>Distance function of <c>batchDistance</c> for one element type, norm and output type.</summary>
internal enum DistanceKind
{
    /// <summary><c>batchDistHamming</c>: byte input, int output.</summary>
    Hamming,

    /// <summary><c>batchDistHamming2</c>: byte input, int output.</summary>
    Hamming2,

    /// <summary><c>batchDistL1_8u32s</c>: byte input, int output.</summary>
    L1Byte,

    /// <summary><c>batchDistL2Sqr_8u32f</c>: byte input, float output.</summary>
    L2SqrByte,

    /// <summary><c>batchDistL2_8u32f</c>: byte input, float output.</summary>
    L2Byte,

    /// <summary><c>batchDistL1_32f</c>: float input, float output.</summary>
    L1Float,

    /// <summary><c>batchDistL2Sqr_32f</c>: float input, float output.</summary>
    L2SqrFloat,

    /// <summary><c>batchDistL2_32f</c>: float input, float output.</summary>
    L2Float,
}

/// <summary>
/// Distances from one query descriptor to every train descriptor. Results are stored as raw
/// 32-bit values: <c>int</c> for the Hamming and byte L1 kinds, float bits for the others.
/// </summary>
internal static class BatchDistanceKernels
{
    /// <summary>Largest descriptor length for which the byte L2 sum is exact in float.</summary>
    private const int ExactByteL2Length = 258;

    public static bool HasIntOutput(DistanceKind kind) =>
        kind is DistanceKind.Hamming or DistanceKind.Hamming2 or DistanceKind.L1Byte;

    /// <summary>Raw value that marks a masked-out pair: <c>INT_MAX</c> or the bits of <c>FLT_MAX</c>.</summary>
    public static int MaskedValue(DistanceKind kind) =>
        HasIntOutput(kind) ? int.MaxValue : BitConverter.SingleToInt32Bits(float.MaxValue);

    /// <summary>
    /// Writes the distances from <paramref name="query"/> to the <paramref name="count"/> train
    /// rows that start at <paramref name="train"/> with element step <paramref name="stride"/>.
    /// </summary>
    public static void ByteRow(
        DistanceKind kind, ReadOnlySpan<byte> query, ref byte train, int stride, int count, Span<int> dist)
    {
        switch (kind)
        {
            case DistanceKind.Hamming:
                HammingRow(query, ref train, stride, count, dist, false);
                break;
            case DistanceKind.Hamming2:
                HammingRow(query, ref train, stride, count, dist, true);
                break;
            case DistanceKind.L1Byte:
                for (int r = 0; r < count; r++)
                {
                    dist[r] = L1Byte(query, ref Unsafe.Add(ref train, (nint)r * stride));
                }

                break;
            case DistanceKind.L2SqrByte:
            case DistanceKind.L2Byte:
                bool root = kind == DistanceKind.L2Byte;
                for (int r = 0; r < count; r++)
                {
                    float d = L2SqrByte(query, ref Unsafe.Add(ref train, (nint)r * stride));
                    dist[r] = BitConverter.SingleToInt32Bits(root ? MathF.Sqrt(d) : d);
                }

                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }

    /// <summary>Float counterpart of <see cref="ByteRow"/>.</summary>
    public static void FloatRow(
        DistanceKind kind, ReadOnlySpan<float> query, ref float train, int stride, int count, Span<int> dist)
    {
        switch (kind)
        {
            case DistanceKind.L1Float:
                L1FloatRow(query, ref train, stride, count, dist);
                break;
            case DistanceKind.L2SqrFloat:
            case DistanceKind.L2Float:
                bool root = kind == DistanceKind.L2Float;
                ref float q = ref MemoryMarshal.GetReference(query);
                int r = 0;
                if (Vector256.IsHardwareAccelerated && query.Length >= 16)
                {
                    for (; r <= count - 2; r += 2)
                    {
                        ref float t0 = ref Unsafe.Add(ref train, (nint)r * stride);
                        (float d0, float d1) = L2SqrFloatX2(ref q, ref t0, ref Unsafe.Add(ref t0, stride), query.Length);
                        dist[r] = BitConverter.SingleToInt32Bits(root ? MathF.Sqrt(d0) : d0);
                        dist[r + 1] = BitConverter.SingleToInt32Bits(root ? MathF.Sqrt(d1) : d1);
                    }
                }

                for (; r < count; r++)
                {
                    float d = L2SqrFloat(ref q, ref Unsafe.Add(ref train, (nint)r * stride), query.Length);
                    dist[r] = BitConverter.SingleToInt32Bits(root ? MathF.Sqrt(d) : d);
                }

                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }

    private static void HammingRow(
        ReadOnlySpan<byte> query, ref byte train, int stride, int count, Span<int> dist, bool cells2)
    {
        int len = query.Length;
        ref byte q = ref MemoryMarshal.GetReference(query);
        if (len == 32 && !cells2)
        {
            ulong q0 = Unsafe.ReadUnaligned<ulong>(ref q);
            ulong q1 = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref q, 8));
            ulong q2 = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref q, 16));
            ulong q3 = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref q, 24));
            for (int r = 0; r < count; r++)
            {
                ref byte t = ref Unsafe.Add(ref train, (nint)r * stride);
                dist[r] = BitOperations.PopCount(q0 ^ Unsafe.ReadUnaligned<ulong>(ref t))
                    + BitOperations.PopCount(q1 ^ Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref t, 8)))
                    + BitOperations.PopCount(q2 ^ Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref t, 16)))
                    + BitOperations.PopCount(q3 ^ Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref t, 24)));
            }

            return;
        }

        for (int r = 0; r < count; r++)
        {
            ref byte t = ref Unsafe.Add(ref train, (nint)r * stride);
            int sum = 0;
            int j = 0;
            for (; j <= len - 8; j += 8)
            {
                ulong x = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref q, j))
                    ^ Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref t, j));
                sum += BitOperations.PopCount(cells2 ? (x | (x >> 1)) & 0x5555555555555555UL : x);
            }

            for (; j < len; j++)
            {
                uint x = (uint)(Unsafe.Add(ref q, j) ^ Unsafe.Add(ref t, j));
                sum += BitOperations.PopCount(cells2 ? (x | (x >> 1)) & 0x55u : x);
            }

            dist[r] = sum;
        }
    }

    private static int L1Byte(ReadOnlySpan<byte> query, ref byte t)
    {
        int len = query.Length;
        ref byte q = ref MemoryMarshal.GetReference(query);
        int sum = 0;
        int j = 0;
        if (Sse2.IsSupported)
        {
            Vector128<long> acc = Vector128<long>.Zero;
            for (; j <= len - 16; j += 16)
            {
                Vector128<byte> a = Vector128.LoadUnsafe(ref q, (nuint)j);
                Vector128<byte> b = Vector128.LoadUnsafe(ref t, (nuint)j);
                acc += Sse2.SumAbsoluteDifferences(a, b).AsInt64();
            }

            sum = (int)(acc.GetElement(0) + acc.GetElement(1));
        }

        for (; j < len; j++)
        {
            sum += Math.Abs(Unsafe.Add(ref q, j) - Unsafe.Add(ref t, j));
        }

        return sum;
    }

    private static float L2SqrByte(ReadOnlySpan<byte> query, ref byte t)
    {
        int len = query.Length;
        ref byte q = ref MemoryMarshal.GetReference(query);
        if (len <= ExactByteL2Length)
        {
            // Every partial sum is an integer below 2^24, so the float sum is exact.
            int sum = 0;
            for (int j = 0; j < len; j++)
            {
                int d = Unsafe.Add(ref q, j) - Unsafe.Add(ref t, j);
                sum += d * d;
            }

            return sum;
        }

        float s = 0f;
        for (int j = 0; j < len; j++)
        {
            float d = (float)Unsafe.Add(ref q, j) - (float)Unsafe.Add(ref t, j);
            s += d * d;
        }

        return s;
    }

    /// <summary>
    /// OpenCV <c>normL1&lt;float, float&gt;</c>: one sequential sum per train row. Four train rows run
    /// together for instruction-level parallelism; each row keeps the OpenCV order.
    /// </summary>
    private static void L1FloatRow(ReadOnlySpan<float> query, ref float train, int stride, int count, Span<int> dist)
    {
        int len = query.Length;
        ref float q = ref MemoryMarshal.GetReference(query);
        int r = 0;
        for (; r <= count - 4; r += 4)
        {
            ref float t0 = ref Unsafe.Add(ref train, (nint)r * stride);
            ref float t1 = ref Unsafe.Add(ref t0, stride);
            ref float t2 = ref Unsafe.Add(ref t1, stride);
            ref float t3 = ref Unsafe.Add(ref t2, stride);
            float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
            for (int j = 0; j < len; j++)
            {
                float a = Unsafe.Add(ref q, j);
                s0 += MathF.Abs(a - Unsafe.Add(ref t0, j));
                s1 += MathF.Abs(a - Unsafe.Add(ref t1, j));
                s2 += MathF.Abs(a - Unsafe.Add(ref t2, j));
                s3 += MathF.Abs(a - Unsafe.Add(ref t3, j));
            }

            dist[r] = BitConverter.SingleToInt32Bits(s0);
            dist[r + 1] = BitConverter.SingleToInt32Bits(s1);
            dist[r + 2] = BitConverter.SingleToInt32Bits(s2);
            dist[r + 3] = BitConverter.SingleToInt32Bits(s3);
        }

        for (; r < count; r++)
        {
            ref float t = ref Unsafe.Add(ref train, (nint)r * stride);
            float s = 0f;
            for (int j = 0; j < len; j++)
            {
                s += MathF.Abs(Unsafe.Add(ref q, j) - Unsafe.Add(ref t, j));
            }

            dist[r] = BitConverter.SingleToInt32Bits(s);
        }
    }

    /// <summary>
    /// <see cref="L2SqrFloat"/> for two train rows at once; each row keeps the OpenCV order. The
    /// two accumulator chains run in parallel.
    /// </summary>
    private static (float, float) L2SqrFloatX2(ref float a, ref float b0, ref float b1, int n)
    {
        Vector256<float> p0 = Vector256<float>.Zero, p1 = Vector256<float>.Zero;
        Vector256<float> r0 = Vector256<float>.Zero, r1 = Vector256<float>.Zero;
        int j = 0;
        for (; j <= n - 16; j += 16)
        {
            Vector256<float> a0 = Vector256.LoadUnsafe(ref a, (nuint)j);
            Vector256<float> a1 = Vector256.LoadUnsafe(ref a, (nuint)(j + 8));
            Vector256<float> t0 = a0 - Vector256.LoadUnsafe(ref b0, (nuint)j);
            Vector256<float> t1 = a1 - Vector256.LoadUnsafe(ref b0, (nuint)(j + 8));
            Vector256<float> u0 = a0 - Vector256.LoadUnsafe(ref b1, (nuint)j);
            Vector256<float> u1 = a1 - Vector256.LoadUnsafe(ref b1, (nuint)(j + 8));
            p0 += t0 * t0;
            p1 += t1 * t1;
            r0 += u0 * u0;
            r1 += u1 * u1;
        }

        Vector128<float> s = ((p0.GetLower() + p0.GetUpper()) + p1.GetLower()) + p1.GetUpper();
        Vector128<float> v = ((r0.GetLower() + r0.GetUpper()) + r1.GetLower()) + r1.GetUpper();
        float d0 = (s.GetElement(0) + s.GetElement(2)) + (s.GetElement(1) + s.GetElement(3));
        float d1 = (v.GetElement(0) + v.GetElement(2)) + (v.GetElement(1) + v.GetElement(3));
        for (; j < n; j++)
        {
            float t = Unsafe.Add(ref a, j) - Unsafe.Add(ref b0, j);
            d0 += t * t;
            float u = Unsafe.Add(ref a, j) - Unsafe.Add(ref b1, j);
            d1 += u * u;
        }

        return (d0, d1);
    }

    /// <summary>
    /// OpenCV <c>hal::normL2Sqr_</c> on the SSE3 baseline: four 4-lane accumulators over blocks of
    /// 16 elements (element <c>j</c> goes to partial sum <c>j mod 16</c>), the lane sum
    /// <c>((d0 + d1) + d2) + d3</c>, the reduction <c>(s0 + s2) + (s1 + s3)</c>, then a
    /// sequential tail. Multiply and add round separately.
    /// </summary>
    public static float L2SqrFloat(ref float a, ref float b, int n)
    {
        int j = 0;
        float d = 0f;
        if (n >= 16)
        {
            Vector128<float> s;
            if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> acc0 = Vector256<float>.Zero, acc1 = Vector256<float>.Zero;
                for (; j <= n - 16; j += 16)
                {
                    Vector256<float> t0 = Vector256.LoadUnsafe(ref a, (nuint)j) - Vector256.LoadUnsafe(ref b, (nuint)j);
                    Vector256<float> t1 = Vector256.LoadUnsafe(ref a, (nuint)(j + 8)) - Vector256.LoadUnsafe(ref b, (nuint)(j + 8));
                    acc0 += t0 * t0;
                    acc1 += t1 * t1;
                }

                s = ((acc0.GetLower() + acc0.GetUpper()) + acc1.GetLower()) + acc1.GetUpper();
            }
            else
            {
                Vector128<float> d0 = Vector128<float>.Zero, d1 = Vector128<float>.Zero;
                Vector128<float> d2 = Vector128<float>.Zero, d3 = Vector128<float>.Zero;
                for (; j <= n - 16; j += 16)
                {
                    Vector128<float> t0 = Vector128.LoadUnsafe(ref a, (nuint)j) - Vector128.LoadUnsafe(ref b, (nuint)j);
                    Vector128<float> t1 = Vector128.LoadUnsafe(ref a, (nuint)(j + 4)) - Vector128.LoadUnsafe(ref b, (nuint)(j + 4));
                    Vector128<float> t2 = Vector128.LoadUnsafe(ref a, (nuint)(j + 8)) - Vector128.LoadUnsafe(ref b, (nuint)(j + 8));
                    Vector128<float> t3 = Vector128.LoadUnsafe(ref a, (nuint)(j + 12)) - Vector128.LoadUnsafe(ref b, (nuint)(j + 12));
                    d0 += t0 * t0;
                    d1 += t1 * t1;
                    d2 += t2 * t2;
                    d3 += t3 * t3;
                }

                s = ((d0 + d1) + d2) + d3;
            }

            d = (s.GetElement(0) + s.GetElement(2)) + (s.GetElement(1) + s.GetElement(3));
        }

        for (; j < n; j++)
        {
            float t = Unsafe.Add(ref a, j) - Unsafe.Add(ref b, j);
            d += t * t;
        }

        return d;
    }
}
