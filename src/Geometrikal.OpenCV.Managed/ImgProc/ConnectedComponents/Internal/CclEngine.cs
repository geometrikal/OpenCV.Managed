// This implementation is ported from the OpenCV connected components labeling.
//
// Original source:
//   modules/imgproc/src/connectedcomponents.cpp
//     (connectedComponents_sub1, Labeling*Parallel, CCStatsOp)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/connectedcomponents.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Runs the two labeling scans, sequentially or in row stripes. Stripes start at even rows and
/// use disjoint provisional label ranges that increase with the stripe index, so the final
/// labels do not depend on the stripe count.
/// </summary>
internal static class CclEngine
{
    private const long MinPixelsPerStripe = 1 << 15;

    private enum Scan
    {
        Pixel4,
        Pixel8,
        Block8,
    }

    /// <summary>
    /// Labels <paramref name="img"/> into <paramref name="labels"/> (<c>int</c> or <c>ushort</c>)
    /// and returns the label count. With <paramref name="withStats"/>, also returns the
    /// statistics and centroid matrices.
    /// </summary>
    public static int Run<TLabel>(
        CclImage img,
        Mat<TLabel> labels,
        int connectivity,
        ConnectedComponentsAlgorithm algorithm,
        bool withStats,
        out Mat<int>? stats,
        out Mat<double>? centroids)
        where TLabel : unmanaged
    {
        int w = img.Width;
        int h = img.Height;
        bool narrow = typeof(TLabel) == typeof(ushort);
        Scan scan = connectivity == 4 ? Scan.Pixel4
            : algorithm is ConnectedComponentsAlgorithm.Wu or ConnectedComponentsAlgorithm.Sauf ? Scan.Pixel8
            : Scan.Block8;
        int bw = (w + 1) >> 1;
        int bh = (h + 1) >> 1;
        long maxLabels = scan == Scan.Pixel4 ? (((long)w * h) + 1) / 2 : (long)bw * bh;
        int stripes = StripeCount(w, h);
        int[] bounds = StripeBounds(h, stripes);
        bool parallelScan = stripes > 1;

        int[] parent = ArrayPool<int>.Shared.Rent(checked((int)(maxLabels + 1)));
        int[]? blocks = scan == Scan.Block8 ? ArrayPool<int>.Shared.Rent(bw * bh) : null;
        int[]? pixels = scan != Scan.Block8 && narrow ? ArrayPool<int>.Shared.Rent(w * h) : null;
        try
        {
            CclStore store = scan == Scan.Block8 ? default
                : narrow ? new CclStore(pixels!, 0, w)
                : new CclStore((int[])(object)labels.DangerousGetArray(), labels.DangerousGetOffset(), labels.Stride);
            parent[0] = 0;

            int[] first = new int[stripes];
            int[] end = new int[stripes];
            int count;
            if (!parallelScan)
            {
                end[0] = ScanStripe(scan, img, store, blocks, parent, 0, h, 1, narrow ? ushort.MaxValue : int.MaxValue);
                count = CclScans.Flatten(parent, 1, end[0], 1);
            }
            else
            {
                for (int s = 0; s < stripes; s++)
                {
                    first[s] = FirstLabel(scan, bounds[s], w);
                }

                Parallel.For(
                    0,
                    stripes,
                    new ParallelOptions { MaxDegreeOfParallelism = stripes },
                    s => end[s] = ScanStripe(scan, img, store, blocks, parent, bounds[s], bounds[s + 1], first[s], int.MaxValue));

                // The sequential scan creates the stripe labels minus the stripe-top labels that connect upwards.
                long sequential = end[0] - first[0];
                for (int s = 1; s < stripes; s++)
                {
                    sequential += end[s] - first[s];
                    sequential -= scan == Scan.Block8
                        ? CclScans.MergeBlock(img, blocks!, parent, bounds[s])
                        : CclScans.MergePixel(img, store, parent, bounds[s], scan == Scan.Pixel8);
                }

                if (narrow && sequential >= ushort.MaxValue)
                {
                    CclScans.ThrowOverflow();
                }

                count = 1;
                for (int s = 0; s < stripes; s++)
                {
                    count = CclScans.Flatten(parent, first[s], end[s], count);
                }
            }

            CcStat[]? global = null;
            try
            {
                if (withStats)
                {
                    global = ArrayPool<CcStat>.Shared.Rent(count);
                    global.AsSpan(0, count).Fill(CcStat.Empty);
                }

                var pass = new SecondPass<TLabel>(scan, img, store, blocks, parent, labels, global, narrow);
                if (stripes == 1)
                {
                    pass.Stripe(0, h, global, 0, ref global is null ? ref Unsafe.NullRef<CcStat>() : ref global[0]);
                }
                else
                {
                    RunStripes(pass, bounds, parallelScan ? first : null, end, count);
                }

                if (withStats)
                {
                    Finish(global!, count, out stats, out centroids);
                }
                else
                {
                    stats = null;
                    centroids = null;
                }
            }
            finally
            {
                if (global is not null)
                {
                    ArrayPool<CcStat>.Shared.Return(global);
                }
            }

            return count;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(parent);
            if (blocks is not null)
            {
                ArrayPool<int>.Shared.Return(blocks);
            }

            if (pixels is not null)
            {
                ArrayPool<int>.Shared.Return(pixels);
            }
        }
    }

    private static int ScanStripe(
        Scan scan, CclImage img, CclStore store, int[]? blocks, int[] parent, int y0, int y1, int label, int limit)
        => scan == Scan.Block8
            ? CclScans.Block8(img, blocks!, parent, y0, y1, label, limit)
            : CclScans.Pixel(img, store, parent, y0, y1, label, limit, scan == Scan.Pixel8);

    /// <summary>First provisional label of the stripe that starts at even row <paramref name="y"/>.</summary>
    private static int FirstLabel(Scan scan, int y, int w)
        => scan == Scan.Pixel4
            ? (int)(((long)y * w / 2) + 1)
            : (int)(((long)(y / 2) * ((w + 1) / 2)) + 1);

    private static int StripeCount(int w, int h)
    {
        int dop = CvParallel.MaxDegreeOfParallelism;
        if (dop <= 1)
        {
            return 1;
        }

        long byWork = (long)w * h / MinPixelsPerStripe;
        return (int)Math.Max(1, Math.Min(Math.Min(dop, (h + 1) / 2), byWork));
    }

    private static int[] StripeBounds(int h, int stripes)
    {
        int pairs = (h + 1) / 2;
        int[] bounds = new int[stripes + 1];
        for (int s = 0; s < stripes; s++)
        {
            bounds[s] = 2 * (int)((long)pairs * s / stripes);
        }

        bounds[stripes] = h;
        return bounds;
    }

    /// <summary>
    /// Runs the second pass in parallel stripes. Stripe 0 accumulates into the global statistics;
    /// every other stripe accumulates into a local array over its label range, which is then
    /// merged. After a parallel first scan, the labels of stripe <c>s</c> are the final labels of
    /// its provisional range <c>[first[s], end[s])</c>.
    /// </summary>
    private static void RunStripes<TLabel>(SecondPass<TLabel> pass, int[] bounds, int[]? first, int[] end, int count)
        where TLabel : unmanaged
    {
        int stripes = bounds.Length - 1;
        if (pass.Global is null)
        {
            Parallel.For(
                0,
                stripes,
                new ParallelOptions { MaxDegreeOfParallelism = stripes },
                s => pass.Stripe(bounds[s], bounds[s + 1], null, 0, ref Unsafe.NullRef<CcStat>()));
            return;
        }

        var locals = new CcStat[stripes][];
        var lows = new int[stripes];
        var highs = new int[stripes];
        var backgrounds = new CcStat[stripes];
        for (int s = 1; s < stripes; s++)
        {
            int lo = 1;
            int hi = count - 1;
            if (first is not null)
            {
                (lo, hi) = Range(pass.Parent, first[s], end[s]);
            }

            lows[s] = lo;
            highs[s] = hi;
            int length = Math.Max(hi - lo + 1, 1);
            locals[s] = ArrayPool<CcStat>.Shared.Rent(length);
            locals[s].AsSpan(0, length).Fill(CcStat.Empty);
            backgrounds[s] = CcStat.Empty;
        }

        try
        {
            CcStat[] global = pass.Global;
            Parallel.For(
                0,
                stripes,
                new ParallelOptions { MaxDegreeOfParallelism = stripes },
                s =>
                {
                    if (s == 0)
                    {
                        pass.Stripe(bounds[0], bounds[1], global, 0, ref global[0]);
                    }
                    else
                    {
                        pass.Stripe(bounds[s], bounds[s + 1], locals[s], lows[s], ref backgrounds[s]);
                    }
                });

            for (int s = 1; s < stripes; s++)
            {
                global[0].Merge(backgrounds[s]);
                int lo = lows[s];
                Span<CcStat> local = locals[s].AsSpan(0, Math.Max(highs[s] - lo + 1, 0));
                for (int k = 0; k < local.Length; k++)
                {
                    if (local[k].Area > 0)
                    {
                        global[lo + k].Merge(local[k]);
                    }
                }
            }
        }
        finally
        {
            for (int s = 1; s < stripes; s++)
            {
                ArrayPool<CcStat>.Shared.Return(locals[s]);
            }
        }
    }

    /// <summary>Minimum and maximum final label of the provisional labels <c>[start, end)</c>.</summary>
    private static (int Lo, int Hi) Range(int[] parent, int start, int end)
    {
        int lo = int.MaxValue;
        int hi = 0;
        for (int i = start; i < end; i++)
        {
            int v = parent[i];
            lo = Math.Min(lo, v);
            hi = Math.Max(hi, v);
        }

        return hi == 0 ? (1, 0) : (lo, hi);
    }

    private static void Finish(CcStat[] global, int count, out Mat<int>? stats, out Mat<double>? centroids)
    {
        double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
        var st = new Mat<int>(5, count, clear: false);
        var ce = new Mat<double>(2, count, clear: false);
        Span<int> sv = st.GetSpan();
        Span<double> cv = ce.GetSpan();
        for (int l = 0; l < count; l++)
        {
            ref CcStat s = ref global[l];
            Span<int> row = sv.Slice(l * 5, 5);
            if (s.Area > 0)
            {
                double area = (uint)s.Area;
                row[0] = s.Left;
                row[1] = s.Top;
                row[2] = s.Right - s.Left + 1;
                row[3] = s.Bottom - s.Top + 1;
                row[4] = s.Area;
                cv[2 * l] = s.SumX / area;
                cv[(2 * l) + 1] = s.SumY / area;
            }
            else
            {
                row[0] = -1;
                row[1] = int.MaxValue;
                row[2] = 0;
                row[3] = 0;
                row[4] = 0;
                cv[2 * l] = nan;
                cv[(2 * l) + 1] = nan;
            }
        }

        stats = st;
        centroids = ce;
    }

    /// <summary>Second-pass state shared by the stripes.</summary>
    private sealed class SecondPass<TLabel>(
        Scan scan,
        CclImage img,
        CclStore store,
        int[]? blocks,
        int[] parent,
        Mat<TLabel> labels,
        CcStat[]? global,
        bool narrow)
        where TLabel : unmanaged
    {
        public int[] Parent => parent;

        public CcStat[]? Global => global;

        /// <summary>Relabels rows <c>[y0, y1)</c>; <paramref name="y0"/> is even.</summary>
        public void Stripe(int y0, int y1, CcStat[]? stats, int lo, ref CcStat background)
        {
            int w = img.Width;
            int h = img.Height;
            ref int p = ref MemoryMarshal.GetArrayDataReference(parent);
            ref CcStat st = ref stats is null ? ref Unsafe.NullRef<CcStat>() : ref MemoryMarshal.GetArrayDataReference(stats);
            ref ushort outBase = ref narrow
                ? ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference((ushort[])(object)labels.DangerousGetArray()), labels.DangerousGetOffset())
                : ref Unsafe.NullRef<ushort>();
            int outStride = labels.Stride;

            if (scan != Scan.Block8)
            {
                for (int y = y0; y < y1; y++)
                {
                    ref int row = ref store.Row(y);
                    CclRelabel.PixelRow(ref row, ref p, w);
                    if (stats is not null)
                    {
                        CclRelabel.AccumulateRow(ref row, w, y, ref st, lo, ref background);
                    }

                    if (narrow)
                    {
                        CclRelabel.Narrow(ref row, ref Unsafe.Add(ref outBase, y * outStride), w);
                    }
                }

                return;
            }

            int bw = (w + 1) >> 1;
            int[]? scratch = narrow ? ArrayPool<int>.Shared.Rent(w) : null;
            try
            {
                ref int intBase = ref narrow
                    ? ref MemoryMarshal.GetArrayDataReference(scratch!)
                    : ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference((int[])(object)labels.DangerousGetArray()), labels.DangerousGetOffset());
                int intStride = narrow ? 0 : outStride;
                ref int bl = ref MemoryMarshal.GetArrayDataReference(blocks!);
                for (int y = y0; y < y1; y += 2)
                {
                    ref int brow = ref Unsafe.Add(ref bl, (y >> 1) * bw);
                    CclRelabel.BlockLabels(ref brow, ref p, bw);
                    int yEnd = Math.Min(y + 2, h);
                    for (int r = y; r < yEnd; r++)
                    {
                        ref int row = ref Unsafe.Add(ref intBase, r * intStride);
                        CclRelabel.ExpandBlockRow(ref brow, ref img.Row(r), ref row, w);
                        if (stats is not null)
                        {
                            CclRelabel.AccumulateRow(ref row, w, r, ref st, lo, ref background);
                        }

                        if (narrow)
                        {
                            CclRelabel.Narrow(ref row, ref Unsafe.Add(ref outBase, r * outStride), w);
                        }
                    }
                }
            }
            finally
            {
                if (scratch is not null)
                {
                    ArrayPool<int>.Shared.Return(scratch);
                }
            }
        }
    }
}
