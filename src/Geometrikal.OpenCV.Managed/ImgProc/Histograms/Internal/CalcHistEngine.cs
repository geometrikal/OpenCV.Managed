// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (calcHist_8u, calcHist_, EqualizeHistCalcHist_Invoker)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Integer bin counting. Row stripes count into local histograms; the stripes then add their
/// counts to the shared histogram under a lock. Integer addition is exact in any order, so the
/// result does not depend on the thread count.
/// </summary>
internal static class CalcHistEngine
{
    /// <summary>Stripes are used only when the pixel count is at least this multiple of the bin count.</summary>
    private const int PixelsPerBinForParallel = 32;

    /// <summary>
    /// Adds the counts of the selected pixels to <paramref name="hist"/> (one <c>int</c> per bin,
    /// <paramref name="layout"/> order).
    /// </summary>
    public static void Count<T>(HistLayout<T> layout, Mat<byte>? mask, int[] hist)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            var l8 = (HistLayout<byte>)(object)layout;
            int[] tab = l8.BuildTable8u();
            if (l8.Dims == 1)
            {
                var counts = new int[256];
                Histogram256(l8.Images[0], l8.Offsets[0], l8.Steps[0], mask, counts);
                for (int v = 0; v < 256; v++)
                {
                    if (tab[v] < HistLayout<byte>.OutOfRange)
                    {
                        hist[tab[v]] += counts[v];
                    }
                }

                return;
            }

            object sync = new();
            Run(layout, (y0, y1) => Count8uStripe(l8, tab, mask, hist, sync, y0, y1));
            return;
        }

        object gate = new();
        Run(layout, (y0, y1) => CountStripe(layout, mask, hist, gate, y0, y1));
    }

    /// <summary>
    /// Counts the byte values of one channel (all 256 values, before any binning), as the
    /// first pass of <c>calcHist_8u</c> 1-D and <c>equalizeHist</c>.
    /// </summary>
    public static void Histogram256(Mat<byte> image, int offset, int cn, Mat<byte>? mask, int[] counts)
    {
        object sync = new();
        ParallelRows.For(
            image.Height,
            image.Width,
            (y0, y1) => Histogram256Stripe(image, offset, cn, mask, counts, sync, y0, y1));
    }

    private static void Run<T>(HistLayout<T> layout, Action<int, int> body)
        where T : unmanaged
    {
        long pixels = (long)layout.Width * layout.Height;
        if ((long)layout.Total * PixelsPerBinForParallel > pixels)
        {
            body(0, layout.Height);
            return;
        }

        ParallelRows.For(layout.Height, (long)layout.Width * layout.Dims, body);
    }

    private static void Histogram256Stripe(
        Mat<byte> image, int offset, int cn, Mat<byte>? mask, int[] counts, object sync, int y0, int y1)
    {
        // Four interleaved sub-histograms of 257 bins; bin 256 receives masked-out pixels.
        Span<int> h = stackalloc int[4 * 257];
        h.Clear();
        ref int h0 = ref MemoryMarshal.GetReference(h);
        int w = image.Width;
        if (mask is null && cn == 1 && image.IsContinuous && y1 > y0)
        {
            // A continuous image is one row of width * rows pixels, as OpenCV histPrepareImages.
            CountRow(ref MemoryMarshal.GetReference(image.GetReadOnlyRowSpan(y0)), checked((y1 - y0) * w), ref h0);
            y1 = y0;
        }

        for (int y = y0; y < y1; y++)
        {
            ref byte p = ref Unsafe.Add(ref MemoryMarshal.GetReference(image.GetReadOnlyRowSpan(y)), offset);
            if (mask is not null)
            {
                CountMaskedRow(ref p, ref MemoryMarshal.GetReference(mask.GetReadOnlyRowSpan(y)), w, cn, ref h0);
            }
            else if (cn == 1)
            {
                CountRow(ref p, w, ref h0);
            }
            else
            {
                CountStridedRow(ref p, w, cn, ref h0);
            }
        }

        lock (sync)
        {
            for (int v = 0; v < 256; v++)
            {
                counts[v] += h[v] + h[257 + v] + h[514 + v] + h[771 + v];
            }
        }
    }

    private static void CountRow(ref byte p, int w, ref int h0)
    {
        ref int h1 = ref Unsafe.Add(ref h0, 257);
        ref int h2 = ref Unsafe.Add(ref h0, 514);
        ref int h3 = ref Unsafe.Add(ref h0, 771);
        nint x = 0;
        for (; x <= w - 8; x += 8)
        {
            Unsafe.Add(ref h0, Unsafe.Add(ref p, x))++;
            Unsafe.Add(ref h1, Unsafe.Add(ref p, x + 1))++;
            Unsafe.Add(ref h2, Unsafe.Add(ref p, x + 2))++;
            Unsafe.Add(ref h3, Unsafe.Add(ref p, x + 3))++;
            Unsafe.Add(ref h0, Unsafe.Add(ref p, x + 4))++;
            Unsafe.Add(ref h1, Unsafe.Add(ref p, x + 5))++;
            Unsafe.Add(ref h2, Unsafe.Add(ref p, x + 6))++;
            Unsafe.Add(ref h3, Unsafe.Add(ref p, x + 7))++;
        }

        for (; x < w; x++)
        {
            Unsafe.Add(ref h0, Unsafe.Add(ref p, x))++;
        }
    }

    private static void CountStridedRow(ref byte p, int w, int cn, ref int h0)
    {
        ref int h1 = ref Unsafe.Add(ref h0, 257);
        ref int h2 = ref Unsafe.Add(ref h0, 514);
        ref int h3 = ref Unsafe.Add(ref h0, 771);
        nint i = 0;
        nint step = cn;
        int x = 0;
        for (; x <= w - 4; x += 4, i += 4 * step)
        {
            Unsafe.Add(ref h0, Unsafe.Add(ref p, i))++;
            Unsafe.Add(ref h1, Unsafe.Add(ref p, i + step))++;
            Unsafe.Add(ref h2, Unsafe.Add(ref p, i + (2 * step)))++;
            Unsafe.Add(ref h3, Unsafe.Add(ref p, i + (3 * step)))++;
        }

        for (; x < w; x++, i += step)
        {
            Unsafe.Add(ref h0, Unsafe.Add(ref p, i))++;
        }
    }

    private static void CountMaskedRow(ref byte p, ref byte m, int w, int cn, ref int h0)
    {
        ref int h1 = ref Unsafe.Add(ref h0, 257);
        nint i = 0;
        nint step = cn;
        int x = 0;
        for (; x <= w - 2; x += 2, i += 2 * step)
        {
            Unsafe.Add(ref h0, MaskedBin(Unsafe.Add(ref p, i), Unsafe.Add(ref m, x)))++;
            Unsafe.Add(ref h1, MaskedBin(Unsafe.Add(ref p, i + step), Unsafe.Add(ref m, x + 1)))++;
        }

        for (; x < w; x++, i += step)
        {
            Unsafe.Add(ref h0, MaskedBin(Unsafe.Add(ref p, i), Unsafe.Add(ref m, x)))++;
        }
    }

    /// <summary>The byte value, or bin 256 when the mask byte is zero.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int MaskedBin(byte v, byte m)
    {
        int unselected = BinSelect.Unselected(m);
        return (v & ~unselected) | (256 & unselected);
    }

    private static void Count8uStripe(
        HistLayout<byte> layout, int[] tab, Mat<byte>? mask, int[] hist, object sync, int y0, int y1)
    {
        int total = layout.Total;
        int dims = layout.Dims;

        // Up to four sub-histograms, each with an overflow bin at index total. The scalar row
        // kernel uses the first two; the VBMI kernel uses all four.
        int subs = dims <= 3 && total <= 4096 ? 4 : 1;
        int stride = total + 1;
        int[] local = ArrayPool<int>.Shared.Rent(stride * subs);
        try
        {
            Span<int> h = local.AsSpan(0, stride * subs);
            h.Clear();
            ref int hr = ref MemoryMarshal.GetReference(h);
            ref int t = ref MemoryMarshal.GetArrayDataReference(tab);
            int w = layout.Width;
            Span<int> start = stackalloc int[DenseHistogram.MaxDims];
            nint subStride = subs == 4 ? stride : 0;
            bool vbmi = TableBinsVbmi.Supported(layout);
            byte[]? split = vbmi ? TableBinsVbmi.SplitTables(tab, dims) : null;
            Span<ushort> buffer = stackalloc ushort[64];
            int vectorWidth = vbmi ? TableBinsVbmi.CountRows(layout, split!, mask, y0, y1, ref hr, subStride, buffer) : 0;
            for (int y = y0; y < y1; y++)
            {
                ref byte m = ref mask is null ? ref Unsafe.NullRef<byte>() : ref MemoryMarshal.GetReference(mask.GetReadOnlyRowSpan(y));
                if (dims <= 3)
                {
                    int x0 = vectorWidth;
                    if (x0 == w)
                    {
                        continue;
                    }

                    var row = new TableRow(
                        ref Unsafe.Add(ref RowRef(layout, 0, y), x0 * layout.Steps[0]),
                        ref Unsafe.Add(ref RowRef(layout, 1, y), x0 * layout.Steps[1]),
                        ref dims == 3 ? ref Unsafe.Add(ref RowRef(layout, 2, y), x0 * layout.Steps[2]) : ref RowRef(layout, 0, y),
                        layout.Steps[0],
                        layout.Steps[1],
                        dims == 3 ? layout.Steps[2] : 0,
                        ref t,
                        ref Unsafe.Add(ref t, 256),
                        ref dims == 3 ? ref Unsafe.Add(ref t, 512) : ref MemoryMarshal.GetArrayDataReference(ZeroTable));
                    if (mask is null)
                    {
                        CountTableRow(row, w - x0, total, subStride, ref hr);
                    }
                    else
                    {
                        CountTableRowMasked(row, ref Unsafe.Add(ref m, x0), w - x0, total, ref hr);
                    }

                    continue;
                }

                for (int d = 0; d < dims; d++)
                {
                    Mat<byte> im = layout.Images[d];
                    start[d] = im.DangerousGetOffset() + (y * im.Stride) + layout.Offsets[d];
                }

                for (int x = 0; x < w; x++)
                {
                    if (mask is not null && Unsafe.Add(ref m, x) == 0)
                    {
                        continue;
                    }

                    int idx = 0;
                    int d = 0;
                    for (; d < dims; d++)
                    {
                        byte v = layout.Images[d].DangerousGetArray()[start[d] + (x * layout.Steps[d])];
                        int o = Unsafe.Add(ref t, (d * 256) + v);
                        if (o >= HistLayout<byte>.OutOfRange)
                        {
                            break;
                        }

                        idx += o;
                    }

                    if (d == dims)
                    {
                        Unsafe.Add(ref hr, idx)++;
                    }
                }
            }

            for (int s = 1; s < subs; s++)
            {
                Span<int> sub = h.Slice(s * stride, total);
                for (int k = 0; k < total; k++)
                {
                    h[k] += sub[k];
                }
            }

            Merge(h[..total], hist, sync);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(local);
        }
    }

    /// <summary>
    /// Row references of up to three byte planes and their 8U tables. With two dimensions the
    /// third plane reads a table of zeros.
    /// </summary>
    internal readonly ref struct TableRow(
        ref byte p0, ref byte p1, ref byte p2, int c0, int c1, int c2, ref int t0, ref int t1, ref int t2)
    {
        public readonly ref byte P0 = ref p0;
        public readonly ref byte P1 = ref p1;
        public readonly ref byte P2 = ref p2;
        public readonly int C0 = c0;
        public readonly int C1 = c1;
        public readonly int C2 = c2;
        public readonly ref int T0 = ref t0;
        public readonly ref int T1 = ref t1;
        public readonly ref int T2 = ref t2;
    }

    private static void CountTableRow(in TableRow row, int w, int total, nint subStride, ref int h0)
        => CountTableRow(ref row.P0, ref row.P1, ref row.P2, row.C0, row.C1, row.C2, ref row.T0, ref row.T1, ref row.T2, w, total, subStride, ref h0);

    /// <summary>
    /// Bin offsets <c>t0[p0] + t1[p1] + t2[p2]</c>; an out-of-range plane makes the sum at least 2^28,
    /// which <see cref="BinSelect.Min"/> maps to the overflow bin.
    /// </summary>
    private static void CountTableRow(
        ref byte p0, ref byte p1, ref byte p2, nint c0, nint c1, nint c2, ref int t0, ref int t1, ref int t2,
        int w, int total, nint subStride, ref int h0)
    {
        nint i0 = 0;
        nint i1 = 0;
        nint i2 = 0;
        int x = 0;
        for (; x <= w - 2; x += 2)
        {
            int a = Unsafe.Add(ref t0, Unsafe.Add(ref p0, i0)) + Unsafe.Add(ref t1, Unsafe.Add(ref p1, i1)) + Unsafe.Add(ref t2, Unsafe.Add(ref p2, i2));
            int b = Unsafe.Add(ref t0, Unsafe.Add(ref p0, i0 + c0)) + Unsafe.Add(ref t1, Unsafe.Add(ref p1, i1 + c1)) + Unsafe.Add(ref t2, Unsafe.Add(ref p2, i2 + c2));
            i0 += 2 * c0;
            i1 += 2 * c1;
            i2 += 2 * c2;
            Unsafe.Add(ref h0, BinSelect.Min(a, total))++;
            Unsafe.Add(ref h0, subStride + BinSelect.Min(b, total))++;
        }

        if (x < w)
        {
            int a = Unsafe.Add(ref t0, Unsafe.Add(ref p0, i0)) + Unsafe.Add(ref t1, Unsafe.Add(ref p1, i1)) + Unsafe.Add(ref t2, Unsafe.Add(ref p2, i2));
            Unsafe.Add(ref h0, BinSelect.Min(a, total))++;
        }
    }

    private static void CountTableRowMasked(in TableRow row, ref byte m, int w, int total, ref int h0)
    {
        ref byte p0 = ref row.P0;
        ref byte p1 = ref row.P1;
        ref byte p2 = ref row.P2;
        ref int t0 = ref row.T0;
        ref int t1 = ref row.T1;
        ref int t2 = ref row.T2;
        nint c0 = row.C0;
        nint c1 = row.C1;
        nint c2 = row.C2;
        for (nint x = 0; x < w; x++)
        {
            int a = Unsafe.Add(ref t0, Unsafe.Add(ref p0, x * c0)) + Unsafe.Add(ref t1, Unsafe.Add(ref p1, x * c1)) + Unsafe.Add(ref t2, Unsafe.Add(ref p2, x * c2));
            Unsafe.Add(ref h0, BinSelect.Min(a | (BinSelect.Unselected(Unsafe.Add(ref m, x)) & HistLayout<byte>.OutOfRange), total))++;
        }
    }

    /// <summary>Byte values of a third dimension when there are two dimensions: every entry adds 0.</summary>
    private static readonly int[] ZeroTable = new int[256];

    private static void CountStripe<T>(HistLayout<T> layout, Mat<byte>? mask, int[] hist, object sync, int y0, int y1)
        where T : unmanaged
    {
        int total = layout.Total;
        int w = layout.Width;

        // Four sub-histograms break the store-to-load dependency of repeated bins.
        int subs = total <= 4096 ? 4 : 1;
        int stride = total + 1;
        int[] local = ArrayPool<int>.Shared.Rent(stride * subs);
        int[] idxRow = ArrayPool<int>.Shared.Rent(w);
        try
        {
            Span<int> h = local.AsSpan(0, stride * subs);
            h.Clear();
            Span<int> idx = idxRow.AsSpan(0, w);
            ref int h0 = ref MemoryMarshal.GetReference(h);
            ref int h1 = ref Unsafe.Add(ref h0, subs == 4 ? stride : 0);
            ref int h2 = ref Unsafe.Add(ref h0, subs == 4 ? 2 * stride : 0);
            ref int h3 = ref Unsafe.Add(ref h0, subs == 4 ? 3 * stride : 0);
            ref int ir = ref MemoryMarshal.GetReference(idx);
            for (int y = y0; y < y1; y++)
            {
                HistBins.Row(layout, y, idx, backProject: false);
                if (mask is not null)
                {
                    ref byte m = ref MemoryMarshal.GetReference(mask.GetReadOnlyRowSpan(y));
                    for (int x = 0; x < w; x++)
                    {
                        Unsafe.Add(ref ir, x) |= BinSelect.Unselected(Unsafe.Add(ref m, x));
                    }
                }

                int i = 0;
                for (; i <= w - 4; i += 4)
                {
                    Unsafe.Add(ref h0, BinSelect.Clip(Unsafe.Add(ref ir, i), total))++;
                    Unsafe.Add(ref h1, BinSelect.Clip(Unsafe.Add(ref ir, i + 1), total))++;
                    Unsafe.Add(ref h2, BinSelect.Clip(Unsafe.Add(ref ir, i + 2), total))++;
                    Unsafe.Add(ref h3, BinSelect.Clip(Unsafe.Add(ref ir, i + 3), total))++;
                }

                for (; i < w; i++)
                {
                    Unsafe.Add(ref h0, BinSelect.Clip(Unsafe.Add(ref ir, i), total))++;
                }
            }

            for (int s = 1; s < subs; s++)
            {
                Span<int> sub = h.Slice(s * stride, total);
                for (int k = 0; k < total; k++)
                {
                    h[k] += sub[k];
                }
            }

            Merge(h[..total], hist, sync);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(local);
            ArrayPool<int>.Shared.Return(idxRow);
        }
    }

    private static void Merge(ReadOnlySpan<int> local, int[] hist, object sync)
    {
        lock (sync)
        {
            for (int k = 0; k < local.Length; k++)
            {
                hist[k] += local[k];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref byte RowRef(HistLayout<byte> layout, int d, int y)
        => ref Unsafe.Add(ref MemoryMarshal.GetReference(layout.Images[d].GetReadOnlyRowSpan(y)), layout.Offsets[d]);

}
