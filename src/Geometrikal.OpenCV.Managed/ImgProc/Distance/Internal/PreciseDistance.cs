// This implementation is ported from the OpenCV precise Euclidean distance transform.
//
// Original source:
//   modules/imgproc/src/distransform.cpp (trueDistTrans, DTColumnInvoker, DTRowInvoker)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/distransform.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <c>trueDistTrans</c> of OpenCV 5.0 (Felzenszwalb-Huttenlocher): an integer column pass, then
/// a float32 lower envelope of parabolas per row. Squared distances of 65536 and more map to
/// <c>(float)uint.MaxValue</c>.
/// </summary>
internal static class PreciseDistance
{
    private const int MaxDist = 1 << 16;
    private const float Inf = 1e15f;
    private const float UIntMax = 4294967296f;

    public static void Transform(Mat<byte> source, Mat<float> destination)
    {
        int m = source.Height, n = source.Width;
        var src = new Plane<byte>(source.DangerousGetArray(), source.DangerousGetOffset(), source.Stride);
        var dst = new Plane<float>(destination.DangerousGetArray(), destination.DangerousGetOffset(), destination.Stride);

        int vb = Vector<byte>.Count;
        if (Vector.IsHardwareAccelerated && n >= vb)
        {
            int blocks = (n + vb - 1) / vb;
            ParallelRows.For(blocks, (long)m * 8, (b0, b1) => ColumnBlocks(src, dst, m, n, b0, b1));
        }
        else
        {
            ColumnsScalar(src, dst, m, 0, n);
        }

        float[] inv = ArrayPool<float>.Shared.Rent(n);
        uint[] sqr = ArrayPool<uint>.Shared.Rent(n);
        float[] sqrF = ArrayPool<float>.Shared.Rent(n);
        try
        {
            inv[0] = 0;
            sqr[0] = 0;
            sqrF[0] = 0;
            for (int i = 1; i < n; i++)
            {
                inv[i] = (float)(0.5 / i);
                sqr[i] = i >= MaxDist ? uint.MaxValue : (uint)i * (uint)i;
                sqrF[i] = sqr[i];
            }

            var tables = new RowTables(inv, sqr, sqrF);
            ParallelRows.For(m, (long)n * 16, (r0, r1) => Rows(dst, n, tables, r0, r1));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(inv);
            ArrayPool<uint>.Shared.Return(sqr);
            ArrayPool<float>.Shared.Return(sqrF);
        }
    }

    /// <summary>
    /// Column pass over blocks of <c>Vector&lt;byte&gt;.Count</c> adjacent columns, one column per lane.
    /// The last block overlaps the previous one; columns are independent, so the overlap rewrites equal values.
    /// </summary>
    private static void ColumnBlocks(Plane<byte> src, Plane<float> dst, int m, int n, int b0, int b1)
    {
        int vb = Vector<byte>.Count, vi = Vector<int>.Count;
        int[] buffer = ArrayPool<int>.Shared.Rent(m * vb);
        try
        {
            ref int d = ref MemoryMarshal.GetArrayDataReference(buffer);
            var one = Vector<int>.One;
            var limit = new Vector<int>(Math.Min(m, MaxDist));
            var inf = new Vector<float>(UIntMax);
            for (int b = b0; b < b1; b++)
            {
                int c0 = Math.Min(b * vb, n - vb);
                var start = new Vector<int>(m - 1);
                Vector<int> a0 = start, a1 = start, a2 = start, a3 = start;
                for (int j = m - 1; j >= 0; j--)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref src.Row(j), (nuint)c0), out Vector<ushort> lo, out Vector<ushort> hi);
                    Vector.Widen(lo, out Vector<uint> s0, out Vector<uint> s1);
                    Vector.Widen(hi, out Vector<uint> s2, out Vector<uint> s3);
                    a0 = Vector.AndNot(a0 + one, Vector.AsVectorInt32(Vector.Equals(s0, Vector<uint>.Zero)));
                    a1 = Vector.AndNot(a1 + one, Vector.AsVectorInt32(Vector.Equals(s1, Vector<uint>.Zero)));
                    a2 = Vector.AndNot(a2 + one, Vector.AsVectorInt32(Vector.Equals(s2, Vector<uint>.Zero)));
                    a3 = Vector.AndNot(a3 + one, Vector.AsVectorInt32(Vector.Equals(s3, Vector<uint>.Zero)));
                    ref int dj = ref Unsafe.Add(ref d, j * vb);
                    a0.StoreUnsafe(ref dj);
                    a1.StoreUnsafe(ref dj, (nuint)vi);
                    a2.StoreUnsafe(ref dj, (nuint)(2 * vi));
                    a3.StoreUnsafe(ref dj, (nuint)(3 * vi));
                }

                a0 = a1 = a2 = a3 = start;
                for (int j = 0; j < m; j++)
                {
                    ref int dj = ref Unsafe.Add(ref d, j * vb);
                    ref float o = ref Unsafe.Add(ref dst.Row(j), c0);
                    a0 = Vector.Min(a0 + one, Vector.LoadUnsafe(ref dj));
                    a1 = Vector.Min(a1 + one, Vector.LoadUnsafe(ref dj, (nuint)vi));
                    a2 = Vector.Min(a2 + one, Vector.LoadUnsafe(ref dj, (nuint)(2 * vi)));
                    a3 = Vector.Min(a3 + one, Vector.LoadUnsafe(ref dj, (nuint)(3 * vi)));
                    Square(a0, limit, inf).StoreUnsafe(ref o);
                    Square(a1, limit, inf).StoreUnsafe(ref o, (nuint)vi);
                    Square(a2, limit, inf).StoreUnsafe(ref o, (nuint)(2 * vi));
                    Square(a3, limit, inf).StoreUnsafe(ref o, (nuint)(3 * vi));
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// <c>(float)sqr_tab[dist]</c>. For <c>dist &lt; 65536</c> the float product of the exact
    /// operands rounds once, as the <c>uint</c> to <c>float</c> conversion of <c>dist * dist</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Square(Vector<int> dist, Vector<int> limit, Vector<float> inf)
    {
        Vector<float> f = Vector.ConvertToSingle(dist);
        return Vector.ConditionalSelect(Vector.AsVectorSingle(Vector.GreaterThanOrEqual(dist, limit)), inf, f * f);
    }

    private static void ColumnsScalar(Plane<byte> src, Plane<float> dst, int m, int c0, int c1)
    {
        int[] buffer = ArrayPool<int>.Shared.Rent(m);
        try
        {
            int limit = Math.Min(m, MaxDist);
            for (int c = c0; c < c1; c++)
            {
                int dist = m - 1;
                for (int j = m - 1; j >= 0; j--)
                {
                    dist = Unsafe.Add(ref src.Row(j), c) == 0 ? 0 : dist + 1;
                    buffer[j] = dist;
                }

                dist = m - 1;
                for (int j = 0; j < m; j++)
                {
                    dist = Math.Min(dist + 1, buffer[j]);
                    Unsafe.Add(ref dst.Row(j), c) = dist >= limit ? UIntMax : (float)((uint)dist * (uint)dist);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    private static void Rows(Plane<float> dst, int n, RowTables tables, int r0, int r1)
    {
        float[] fbuf = ArrayPool<float>.Shared.Rent(n);
        float[] zbuf = ArrayPool<float>.Shared.Rent(n + 1);
        int[] vbuf = ArrayPool<int>.Shared.Rent(n);
        try
        {
            ref float f = ref MemoryMarshal.GetArrayDataReference(fbuf);
            ref float z = ref MemoryMarshal.GetArrayDataReference(zbuf);
            ref int v = ref MemoryMarshal.GetArrayDataReference(vbuf);
            ref float inv = ref MemoryMarshal.GetArrayDataReference(tables.Inv);
            ref uint sqr = ref MemoryMarshal.GetArrayDataReference(tables.Sqr);
            ref float sqrF = ref MemoryMarshal.GetArrayDataReference(tables.SqrF);
            int tableEnd = Math.Min(MaxDist, n);
            for (int r = r0; r < r1; r++)
            {
                ref float d = ref dst.Row(r);
                MemoryMarshal.CreateReadOnlySpan(ref d, n).CopyTo(fbuf);
                v = 0;
                z = -Inf;
                Unsafe.Add(ref z, 1) = Inf;
                int k = 0, q = 1;
                for (; q < tableEnd; q++)
                {
                    float fq = Unsafe.Add(ref f, q);
                    uint sq = Unsafe.Add(ref sqr, q);
                    for (; ; k--)
                    {
                        int p = Unsafe.Add(ref v, k);
                        float s = (fq - Unsafe.Add(ref f, p) + (float)(sq - Unsafe.Add(ref sqr, p))) * Unsafe.Add(ref inv, q - p);
                        if (s > Unsafe.Add(ref z, k))
                        {
                            k++;
                            Unsafe.Add(ref v, k) = q;
                            Unsafe.Add(ref z, k) = s;
                            Unsafe.Add(ref z, k + 1) = Inf;
                            break;
                        }
                    }
                }

                for (; q < n; q++)
                {
                    float fq = Unsafe.Add(ref f, q);
                    for (; ; k--)
                    {
                        int p = Unsafe.Add(ref v, k);
                        float s = (fq - Unsafe.Add(ref f, p) + ((float)(q + p) * (q - p))) * Unsafe.Add(ref inv, q - p);
                        if (s > Unsafe.Add(ref z, k))
                        {
                            k++;
                            Unsafe.Add(ref v, k) = q;
                            Unsafe.Add(ref z, k) = s;
                            Unsafe.Add(ref z, k + 1) = Inf;
                            break;
                        }
                    }
                }

                k = 0;
                for (q = 0; q < n; q++)
                {
                    while (Unsafe.Add(ref z, k + 1) < q)
                    {
                        k++;
                    }

                    int p = Unsafe.Add(ref v, k);
                    Unsafe.Add(ref d, q) = MathF.Sqrt(Unsafe.Add(ref sqrF, Math.Abs(q - p)) + Unsafe.Add(ref f, p));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(fbuf);
            ArrayPool<float>.Shared.Return(zbuf);
            ArrayPool<int>.Shared.Return(vbuf);
        }
    }

    private readonly record struct RowTables(float[] Inv, uint[] Sqr, float[] SqrF);

    private readonly record struct Plane<T>(T[] Data, int Offset, int Stride)
        where T : unmanaged
    {
        public ref T Row(int y) => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + ((nint)y * Stride));
    }
}
