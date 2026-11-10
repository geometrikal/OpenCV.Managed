// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (calcBackProj_8u, calcBackProj_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Back projection. OpenCV computes <c>saturate_cast&lt;T&gt;(hist[bin] * scale)</c> per pixel;
/// the value depends only on the bin, so it is computed once per bin. Rows have no running state.
/// </summary>
internal static class BackProjectEngine
{
    public static void Run<T>(HistLayout<T> layout, float[] hist, float scale, Mat<T> dst)
        where T : unmanaged
    {
        int total = layout.Total;

        // Index total holds 0 for pixels outside the histogram.
        var values = new T[total + 1];
        for (int k = 0; k < total; k++)
        {
            values[k] = Saturate.FromFloat<T>(hist[k] * scale);
        }

        if (typeof(T) == typeof(byte))
        {
            var l8 = (HistLayout<byte>)(object)layout;
            var d8 = (Mat<byte>)(object)dst;
            var v8 = (byte[])(object)values;
            int[] tab = l8.BuildTable8u();
            if (l8.Dims == 1)
            {
                var lut = new byte[256];
                for (int v = 0; v < 256; v++)
                {
                    lut[v] = tab[v] < HistLayout<byte>.OutOfRange ? v8[tab[v]] : (byte)0;
                }

                ParallelRows.For(
                    layout.Height,
                    Elementwise.WorkPerRow<byte>(layout.Width),
                    (y0, y1) => Lut1D(l8, lut, d8, y0, y1));
                return;
            }

            ParallelRows.For(
                layout.Height,
                (long)layout.Width * layout.Dims,
                (y0, y1) => Table8u(l8, tab, v8, d8, y0, y1));
            return;
        }

        ParallelRows.For(
            layout.Height,
            (long)layout.Width * layout.Dims * 2,
            (y0, y1) => Generic(layout, values, dst, y0, y1));
    }

    private static void Lut1D(HistLayout<byte> layout, byte[] lut, Mat<byte> dst, int y0, int y1)
    {
        Mat<byte> src = layout.Images[0];
        int off = layout.Offsets[0];
        int cn = layout.Steps[0];
        int w = layout.Width;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<byte> row = src.GetReadOnlyRowSpan(y);
            Span<byte> d = dst.GetRowSpan(y);
            if (cn == 1)
            {
                ByteLut.Apply(row, d, lut);
                continue;
            }

            ref byte s = ref Unsafe.Add(ref MemoryMarshal.GetReference(row), off);
            ref byte dr = ref MemoryMarshal.GetReference(d);
            for (int x = 0; x < w; x++)
            {
                Unsafe.Add(ref dr, x) = lut[Unsafe.Add(ref s, x * cn)];
            }
        }
    }

    private static void Table8u(HistLayout<byte> layout, int[] tab, byte[] values, Mat<byte> dst, int y0, int y1)
    {
        int dims = layout.Dims;
        int total = layout.Total;
        int w = layout.Width;
        ref int t = ref MemoryMarshal.GetArrayDataReference(tab);
        ref byte vr = ref MemoryMarshal.GetArrayDataReference(values);
        int[] startBuffer = new int[dims];
        for (int y = y0; y < y1; y++)
        {
            ref byte dr = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
            if (dims <= 3)
            {
                var row = new CalcHistEngine.TableRow(
                    ref RowRef(layout, 0, y),
                    ref RowRef(layout, 1, y),
                    ref dims == 3 ? ref RowRef(layout, 2, y) : ref RowRef(layout, 0, y),
                    layout.Steps[0],
                    layout.Steps[1],
                    dims == 3 ? layout.Steps[2] : 0,
                    ref t,
                    ref Unsafe.Add(ref t, 256),
                    ref dims == 3 ? ref Unsafe.Add(ref t, 512) : ref MemoryMarshal.GetArrayDataReference(ZeroTable));
                TableRow(row, w, total, ref vr, ref dr);
                continue;
            }

            for (int d = 0; d < dims; d++)
            {
                Mat<byte> im = layout.Images[d];
                startBuffer[d] = im.DangerousGetOffset() + (y * im.Stride) + layout.Offsets[d];
            }

            for (int x = 0; x < w; x++)
            {
                int idx = 0;
                for (int d = 0; d < dims; d++)
                {
                    int o = Unsafe.Add(ref t, (d * 256) + layout.Images[d].DangerousGetArray()[startBuffer[d] + (x * layout.Steps[d])]);
                    if (o >= HistLayout<byte>.OutOfRange)
                    {
                        idx = total;
                        break;
                    }

                    idx += o;
                }

                Unsafe.Add(ref dr, x) = Unsafe.Add(ref vr, idx);
            }
        }
    }

    private static readonly int[] ZeroTable = new int[256];

    private static void TableRow(in CalcHistEngine.TableRow row, int w, int total, ref byte values, ref byte dst)
        => TableRow(ref row.P0, ref row.P1, ref row.P2, row.C0, row.C1, row.C2, ref row.T0, ref row.T1, ref row.T2, w, total, ref values, ref dst);

    private static void TableRow(
        ref byte p0, ref byte p1, ref byte p2, nint c0, nint c1, nint c2, ref int t0, ref int t1, ref int t2,
        int w, int total, ref byte values, ref byte dst)
    {
        nint i0 = 0;
        nint i1 = 0;
        nint i2 = 0;
        for (nint x = 0; x < w; x++)
        {
            int a = Unsafe.Add(ref t0, Unsafe.Add(ref p0, i0)) + Unsafe.Add(ref t1, Unsafe.Add(ref p1, i1)) + Unsafe.Add(ref t2, Unsafe.Add(ref p2, i2));
            i0 += c0;
            i1 += c1;
            i2 += c2;
            Unsafe.Add(ref dst, x) = Unsafe.Add(ref values, BinSelect.Min(a, total));
        }
    }

    private static void Generic<T>(HistLayout<T> layout, T[] values, Mat<T> dst, int y0, int y1)
        where T : unmanaged
    {
        int w = layout.Width;
        int total = layout.Total;
        int[] idxRow = ArrayPool<int>.Shared.Rent(w);
        try
        {
            Span<int> idx = idxRow.AsSpan(0, w);
            ref int ir = ref MemoryMarshal.GetReference(idx);
            ref T vr = ref MemoryMarshal.GetArrayDataReference(values);
            for (int y = y0; y < y1; y++)
            {
                HistBins.Row(layout, y, idx, backProject: true);
                ref T dr = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
                for (int x = 0; x < w; x++)
                {
                    int i = Unsafe.Add(ref ir, x);
                    Unsafe.Add(ref dr, x) = Unsafe.Add(ref vr, BinSelect.Clip(i, total));
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(idxRow);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref byte RowRef(HistLayout<byte> layout, int d, int y)
        => ref Unsafe.Add(ref MemoryMarshal.GetReference(layout.Images[d].GetReadOnlyRowSpan(y)), layout.Offsets[d]);
}
