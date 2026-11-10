// This implementation computes OpenCV's Filter2D template for 8-bit sources with integer
// kernels in 16-bit lanes.
//
// Original sources:
//   modules/imgproc/src/filter.simd.hpp (Filter2D)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Direct correlation of a byte image with an integer kernel. With integer coefficients, an
/// integer delta and <c>Σ|k|·255 + |delta| ≤ 32767</c>, every partial sum of OpenCV's float
/// accumulation is an exact integer inside the int16 range. The int16 sums are therefore equal to
/// the float sums, and the final cast is exact.
/// </summary>
internal static class SmallIntFilter2D
{
    /// <summary>
    /// True when <paramref name="coeffs"/> and <paramref name="delta"/> admit the 16-bit path for a
    /// byte source and a byte or short destination.
    /// </summary>
    public static bool Supports<TSrc, TDst>(List<double> coeffs, double delta)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        if (typeof(TSrc) != typeof(byte) || (typeof(TDst) != typeof(byte) && typeof(TDst) != typeof(short)))
        {
            return false;
        }

        if (delta != Math.Round(delta))
        {
            return false;
        }

        double bound = Math.Abs(delta);
        foreach (double c in coeffs)
        {
            if (c != Math.Round(c))
            {
                return false;
            }

            bound += Math.Abs(c) * 255;
        }

        return bound <= short.MaxValue;
    }

    public static Mat<TDst> Apply<TDst>(
        Mat<byte> source, int kw, int kh, int anchorX, int anchorY, BorderType borderType, List<Point> coords, List<double> coeffs, double delta)
        where TDst : unmanaged
    {
        int cn = source.Channels;
        int lineLength = RowStream.LineLength(source.Width, kw, cn);
        int taps = coords.Count <= 5 ? 5 : coords.Count;
        var k = new short[taps];
        var rows = new int[taps];
        var cols = new int[taps];
        for (int t = 0; t < coords.Count; t++)
        {
            k[t] = (short)coeffs[t];
            rows[t] = coords[t].Y;
            cols[t] = coords[t].X * cn;
        }

        var dst = new Mat<TDst>(source.Width, source.Height, cn, clear: false);
        RowStream.Run<byte, short, TDst, Kernel<TDst>>(
            source, dst, kw, kh, anchorX, anchorY, borderType, lineLength,
            new Kernel<TDst>(k, rows, cols, (short)delta),
            (long)source.RowLength * (taps + 2) / Vector<short>.Count);
        return dst;
    }

    private readonly struct Kernel<TDst>(short[] k, int[] rows, int[] cols, short delta) : IRowStreamKernel<byte, short, TDst>
        where TDst : unmanaged
    {
        public void Row(ReadOnlySpan<byte> line, Span<short> row, Span<short> wide)
        {
            ref byte s = ref MemoryMarshal.GetReference(line);
            ref short d = ref MemoryMarshal.GetReference(row);
            int n = row.Length;
            int i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<byte>.Count)
            {
                int vb = Vector<byte>.Count;
                for (; i < n; i += vb)
                {
                    // The final block overlaps the previous one; it rewrites equal values.
                    i = Math.Min(i, n - vb);
                    Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<ushort> lo, out Vector<ushort> hi);
                    Vector.AsVectorInt16(lo).StoreUnsafe(ref d, (nuint)i);
                    Vector.AsVectorInt16(hi).StoreUnsafe(ref d, (nuint)(i + Vector<short>.Count));
                }

                return;
            }

            for (; i < n; i++)
            {
                Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
            }
        }

        public void Column(ref short buffer, int rowStride, Span<TDst> dst, Span<short> scratch)
        {
            Span<short> sums = typeof(TDst) == typeof(short) ? MemoryMarshal.Cast<TDst, short>(dst) : scratch[..dst.Length];
            ref short d = ref MemoryMarshal.GetReference(sums);
            int n = sums.Length;
            int i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<short>.Count)
            {
                if (k.Length == 5)
                {
                    Taps5(ref buffer, rowStride, ref d, n);
                }
                else
                {
                    TapsAny(ref buffer, rowStride, ref d, n);
                }

                i = n;
            }

            for (; i < n; i++)
            {
                int sum = delta;
                for (int t = 0; t < k.Length; t++)
                {
                    sum += k[t] * Unsafe.Add(ref buffer, (rows[t] * rowStride) + cols[t] + i);
                }

                Unsafe.Add(ref d, i) = (short)sum;
            }

            if (typeof(TDst) == typeof(byte))
            {
                StoreBytes(sums, MemoryMarshal.Cast<TDst, byte>(dst));
            }
        }

        private void Taps5(ref short buffer, int stride, ref short d, int n)
        {
            int vs = Vector<short>.Count;
            Vector<short> c0 = new(k[0]), c1 = new(k[1]), c2 = new(k[2]), c3 = new(k[3]), c4 = new(k[4]), start = new(delta);
            ref short r0 = ref Unsafe.Add(ref buffer, (rows[0] * stride) + cols[0]);
            ref short r1 = ref Unsafe.Add(ref buffer, (rows[1] * stride) + cols[1]);
            ref short r2 = ref Unsafe.Add(ref buffer, (rows[2] * stride) + cols[2]);
            ref short r3 = ref Unsafe.Add(ref buffer, (rows[3] * stride) + cols[3]);
            ref short r4 = ref Unsafe.Add(ref buffer, (rows[4] * stride) + cols[4]);
            for (int i = 0; i < n; i += vs)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vs);
                (start + (Vector.LoadUnsafe(ref r0, (nuint)i) * c0) + (Vector.LoadUnsafe(ref r1, (nuint)i) * c1)
                    + (Vector.LoadUnsafe(ref r2, (nuint)i) * c2) + (Vector.LoadUnsafe(ref r3, (nuint)i) * c3)
                    + (Vector.LoadUnsafe(ref r4, (nuint)i) * c4)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        private void TapsAny(ref short buffer, int stride, ref short d, int n)
        {
            int vs = Vector<short>.Count;
            for (int i = 0; i < n; i += vs)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vs);
                var a = new Vector<short>(delta);
                for (int t = 0; t < k.Length; t++)
                {
                    a += Vector.LoadUnsafe(ref buffer, (nuint)((rows[t] * stride) + cols[t] + i)) * new Vector<short>(k[t]);
                }

                a.StoreUnsafe(ref d, (nuint)i);
            }
        }

        private static void StoreBytes(ReadOnlySpan<short> sums, Span<byte> dst)
        {
            ref short s = ref MemoryMarshal.GetReference(sums);
            ref byte d = ref MemoryMarshal.GetReference(dst);
            int n = dst.Length;
            int i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<byte>.Count)
            {
                int vs = Vector<short>.Count;
                var zero = Vector<short>.Zero;
                var max = new Vector<short>(255);
                for (; i < n; i += Vector<byte>.Count)
                {
                    // The final block overlaps the previous one; it rewrites equal values.
                    i = Math.Min(i, n - Vector<byte>.Count);
                    Vector<short> a = Vector.Min(Vector.Max(Vector.LoadUnsafe(ref s, (nuint)i), zero), max);
                    Vector<short> b = Vector.Min(Vector.Max(Vector.LoadUnsafe(ref s, (nuint)(i + vs)), zero), max);
                    Vector.Narrow(Vector.AsVectorUInt16(a), Vector.AsVectorUInt16(b)).StoreUnsafe(ref d, (nuint)i);
                }

                return;
            }

            for (; i < n; i++)
            {
                Unsafe.Add(ref d, i) = (byte)Math.Clamp((int)Unsafe.Add(ref s, i), 0, 255);
            }
        }
    }
}
