// This implementation is ported from the OpenCV discrete Fourier and cosine transforms.
//
// Original source:
//   modules/core/src/dxt.cpp (OcvDftImpl, OcvDftBasicImpl, complementComplex, OcvDctImpl)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/dxt.cpp
//
// OpenCV is licensed under the Apache License 2.0.
// See the OpenCV license and this repository's licensing documentation.

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// The <c>OcvDftImpl</c> and <c>OcvDctImpl</c> drivers. Rows and columns are transformed in
/// batches: each lane of a <see cref="Vector{T}"/> holds one row or column, and batches run in
/// parallel. Each output element has the same operation sequence for any batch or stripe split.
/// </summary>
internal static class DftEngine<T>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
{
    private enum Kind
    {
        Complex,
        RealForward,
        CcsInverse,
    }

    private enum Dims
    {
        OneDim,
        OneDimColWise,
        TwoDims,
    }

    /// <summary>Runs <c>cv::dft</c> from <paramref name="src"/> into the distinct <paramref name="dst"/>.</summary>
    public static void Dft(Mat<T> src, Mat<T> dst, bool inv, bool rowTransform, bool scaled, int nonzeroRows)
    {
        int width = src.Width, height = src.Height;
        int srcCn = src.Channels, dstCn = dst.Channels;
        bool real = srcCn != 2 || dstCn != 2;
        bool fwdRealToComplex = !inv && srcCn == 1 && dstCn == 2;

        Dims dims = DetermineDims(height, width, rowTransform, src.IsContinuous && dst.IsContinuous);
        int[] stages = dims == Dims.TwoDims
            ? (inv && real ? [1, 0] : [0, 1])
            : [dims == Dims.OneDimColWise ? 1 : 0];

        var calls = new DftCall<T>[stages.Length];
        for (int s = 0; s < stages.Length; s++)
        {
            int stage = stages[s];
            bool last = s + 1 == stages.Length;
            int len, count;
            if (stage == 0)
            {
                (len, count) = width == 1 && !rowTransform ? (height, width) : (width, height);
            }
            else
            {
                (len, count) = (height, width);
            }

            DftPlan<T> plan = DftPlan<T>.Get(len, stage == 0 && inv && real);
            double scale = 1;
            if (last && scaled)
            {
                long rowCount = stage == 0 && rowTransform ? 1 : count;
                scale = 1.0 / (len * rowCount);
            }

            calls[s] = new DftCall<T>(plan, inv, T.CreateTruncating(scale));
        }

        Plane srcPlane = new(src);
        Plane dstPlane = new(dst);
        int stageSrcCn = srcCn;
        for (int s = 0; s < stages.Length; s++)
        {
            if (s == 1)
            {
                srcPlane = dstPlane;
                stageSrcCn = dstCn;
            }

            bool last = s + 1 == stages.Length;
            bool isComplex = stageSrcCn != dstCn;
            if (stages[s] == 0)
            {
                RowDft(calls[s], srcPlane, dstPlane, width, height, rowTransform, inv, real, isComplex, nonzeroRows);
                if (last && fwdRealToComplex)
                {
                    int len = width == 1 && !rowTransform ? height : width;
                    int count = width == 1 && !rowTransform ? 1 : height;
                    int nz = nonzeroRows <= 0 || nonzeroRows > count ? count : nonzeroRows;
                    ComplementComplex(dstPlane, len, nz, 1);
                }
            }
            else
            {
                ColDft(calls[s], srcPlane, dstPlane, width, height, inv, real, stageSrcCn, dstCn);
                if (last && fwdRealToComplex)
                {
                    ComplementComplex(dstPlane, width, height, 2);
                }
            }
        }
    }

    /// <summary>Runs <c>cv::dct</c> from <paramref name="src"/> into the distinct <paramref name="dst"/>.</summary>
    public static void Dct(Mat<T> src, Mat<T> dst, bool inv, bool rowTransform)
    {
        int width = src.Width, height = src.Height;
        int startStage, endStage;
        if (rowTransform || height == 1 || (width == 1 && src.IsContinuous && dst.IsContinuous))
        {
            startStage = endStage = 0;
        }
        else
        {
            startStage = width == 1 ? 1 : 0;
            endStage = 1;
        }

        for (int stage = startStage; stage <= endStage; stage++)
        {
            int len = stage == 0 ? (width == 1 && !rowTransform ? height : width) : height;
            if (len > 1 && (len & 1) != 0)
            {
                throw new NotSupportedException("Odd-size DCTs are not implemented.");
            }
        }

        Plane srcPlane = new(src);
        Plane dstPlane = new(dst);
        for (int stage = startStage; stage <= endStage; stage++)
        {
            if (stage == 0)
            {
                bool vector = width == 1 && !rowTransform;
                int len = vector ? height : width;
                int count = vector ? 1 : height;
                var job = new LineJob(srcPlane, dstPlane, len, count, inv) { Dct = DctPlan<T>.Get(len, inv) };
                RunBatches(job, 1, static (j, b0, b1) => DctRows<Vector<T>, VectorLanes<T>>(j, b0, b1), static (j, b0, b1) => DctRows<T, ScalarLanes<T>>(j, b0, b1));
            }
            else
            {
                var job = new LineJob(srcPlane, dstPlane, height, width, inv) { Dct = DctPlan<T>.Get(height, inv) };
                RunBatches(job, ColumnGroup(1), static (j, b0, b1) => DctCols<Vector<T>, VectorLanes<T>>(j, b0, b1), static (j, b0, b1) => DctCols<T, ScalarLanes<T>>(j, b0, b1));
            }

            srcPlane = dstPlane;
        }
    }

    private static Dims DetermineDims(int rows, int cols, bool rowTransform, bool continuous)
    {
        if (rowTransform)
        {
            return Dims.OneDim;
        }

        if (cols == 1 && rows > 1)
        {
            return continuous ? Dims.OneDim : Dims.OneDimColWise;
        }

        return rows == 1 ? Dims.OneDim : Dims.TwoDims;
    }

    /// <summary>
    /// Batches per work unit of a column pass, so that one visit of a row reads about 256
    /// contiguous bytes. <paramref name="valuesPerLane"/> is 2 for complex columns.
    /// </summary>
    private static int ColumnGroup(int valuesPerLane)
        => Math.Max(1, 256 / (Vector<T>.Count * valuesPerLane * Unsafe.SizeOf<T>()));

    /// <summary>
    /// Splits <c>job.Count</c> lines into lane batches and runs groups of
    /// <paramref name="group"/> batches through <see cref="ParallelRows"/>. A single line uses
    /// the scalar kernels. The bodies receive batch ranges.
    /// </summary>
    private static void RunBatches(LineJob job, int group, Action<LineJob, int, int> vectorBody, Action<LineJob, int, int> scalarBody)
    {
        int count = job.Count;
        if (count == 1 || !Vector.IsHardwareAccelerated)
        {
            ParallelRows.For(count, job.WorkPerBatch, (b0, b1) => scalarBody(job, b0, b1));
            return;
        }

        int lanes = Vector<T>.Count;
        int batches = (count + lanes - 1) / lanes;
        int units = (batches + group - 1) / group;
        ParallelRows.For(units, job.WorkPerBatch * group, (u0, u1) => vectorBody(job, u0 * group, Math.Min(u1 * group, batches)));
    }

    private static void RowDft(in DftCall<T> call, Plane src, Plane dst, int width, int height, bool rowTransform, bool inv, bool real, bool isComplex, int nonzeroRows)
    {
        bool vector = width == 1 && !rowTransform;
        int len = vector ? height : width;
        int count = vector ? 1 : height;
        Kind kind = real ? (inv ? Kind.CcsInverse : Kind.RealForward) : Kind.Complex;

        int dstFullLen = kind == Kind.Complex ? 2 * len : len;
        if (!inv && isComplex)
        {
            dstFullLen += (len & 1) != 0 ? 1 : 2;
        }

        int nz = nonzeroRows <= 0 || nonzeroRows > count ? count : nonzeroRows;
        var job = new LineJob(src, dst, len, nz, inv)
        {
            Call = call,
            Kind = kind,
            ComplexOutput = !inv && isComplex,
            ComplexInput = inv && isComplex,
            InLen = kind == Kind.Complex ? 2 * len : len,
            OutLen = Math.Max(2 * len, len + 2),
            DptrOffset = kind == Kind.RealForward && !isComplex && (len & 1) != 0 && len > 1 ? 1 : 0,
            DstFullLen = dstFullLen,
        };

        RunBatches(job, 1, static (j, b0, b1) => RowDftBatches<Vector<T>, VectorLanes<T>>(j, b0, b1), static (j, b0, b1) => RowDftBatches<T, ScalarLanes<T>>(j, b0, b1));

        for (int i = nz; i < count; i++)
        {
            dst.Row(i, dstFullLen).Clear();
        }
    }

    private static void ColDft(in DftCall<T> call, Plane src, Plane dst, int width, int height, bool inv, bool real, int srcCn, int dstCn)
    {
        int count = width;
        int a = 0, b = count, specials = 0;
        if (real)
        {
            a = 1;
            b = (count + 1) / 2;
            specials = (count & 1) == 0 ? 2 : 1;
        }

        var job = new LineJob(src, dst, height, b - a, inv)
        {
            Call = call,
            SrcCn = srcCn,
            DstCn = dstCn,
            Width = width,
            B = b,
            SrcBase = real ? (!inv ? dstCn : (srcCn == 1 ? 1 : 2)) : 0,
            DstBase = real ? (dstCn == 1 ? 1 : 2) : 0,
        };

        if (specials > 0)
        {
            RunBatches(job.WithSpecial(specials), 1, static (j, b0, b1) => ColDftBatches<Vector<T>, VectorLanes<T>>(j, b0, b1), static (j, b0, b1) => ColDftBatches<T, ScalarLanes<T>>(j, b0, b1));
        }

        if (b > a)
        {
            RunBatches(job, ColumnGroup(2), static (j, b0, b1) => ColDftBatches<Vector<T>, VectorLanes<T>>(j, b0, b1), static (j, b0, b1) => ColDftBatches<T, ScalarLanes<T>>(j, b0, b1));
        }
    }

    private static void RowDftBatches<TV, TL>(LineJob job, int b0, int b1)
        where TV : unmanaged
        where TL : struct, IDftLanes<TV, T>
    {
        int lanes = TL.Count;
        int len = job.Len;
        int inLen = job.InLen, outLen = job.OutLen;
        int dptr = job.DptrOffset, full = job.DstFullLen;
        bool complexInput = job.ComplexInput;
        DftCall<T> call = job.Call;
        Plane src = job.Src, dst = job.Dst;

        TV[] inArr = ArrayPool<TV>.Shared.Rent(inLen);
        TV[] outArr = ArrayPool<TV>.Shared.Rent(outLen);
        TV[] scratchArr = ArrayPool<TV>.Shared.Rent(DftKernels<T, TV, TL>.ScratchLength(call));
        try
        {
            Span<TV> inBuf = inArr.AsSpan(0, inLen);
            Span<TV> outBuf = outArr.AsSpan(0, outLen);
            Span<T> inT = MemoryMarshal.Cast<TV, T>(inBuf);
            Span<T> outT = MemoryMarshal.Cast<TV, T>(outBuf);

            for (int batch = b0; batch < b1; batch++)
            {
                int r0 = batch * lanes;
                int used = Math.Min(lanes, job.Count - r0);
                if (used < lanes)
                {
                    inT.Clear();
                }

                int start = src.Offset + (r0 * src.Stride);
                if (lanes == 1)
                {
                    ReadOnlySpan<T> row = src.Array.AsSpan(start, complexInput ? len + 1 : inLen);
                    if (complexInput)
                    {
                        inT[0] = row[0];
                        row.Slice(2, len - 1).CopyTo(inT[1..]);
                    }
                    else
                    {
                        row.CopyTo(inT);
                    }
                }
                else if (complexInput)
                {
                    LaneTranspose.RowsToLanes(src.Array, start, src.Stride, used, 1, inT, 0);
                    LaneTranspose.RowsToLanes(src.Array, start + 2, src.Stride, used, len - 1, inT, 1);
                }
                else
                {
                    LaneTranspose.RowsToLanes(src.Array, start, src.Stride, used, inLen, inT, 0);
                }

                switch (job.Kind)
                {
                    case Kind.RealForward:
                        DftKernels<T, TV, TL>.RealDft(call, inBuf, outBuf, job.ComplexOutput, scratchArr);
                        break;
                    case Kind.CcsInverse:
                        DftKernels<T, TV, TL>.CcsIdft(call, inBuf, outBuf, scratchArr);
                        break;
                    default:
                        DftKernels<T, TV, TL>.Dft(call, inBuf, outBuf, scratchArr);
                        break;
                }

                int dstStart = dst.Offset + (r0 * dst.Stride);
                if (lanes == 1)
                {
                    outT.Slice(dptr, full).CopyTo(dst.Array.AsSpan(dstStart, full));
                }
                else
                {
                    LaneTranspose.LanesToRows<T>(outT, dptr, used, full, dst.Array, dstStart, dst.Stride);
                }
            }
        }
        finally
        {
            ArrayPool<TV>.Shared.Return(scratchArr);
            ArrayPool<TV>.Shared.Return(outArr);
            ArrayPool<TV>.Shared.Return(inArr);
        }
    }

    /// <summary>Column transforms of the batches <c>[b0, b1)</c>, in groups that share one pass over the rows.</summary>
    private static void ColDftBatches<TV, TL>(LineJob job, int b0, int b1)
        where TV : unmanaged
        where TL : struct, IDftLanes<TV, T>
    {
        int lanes = TL.Count;
        int height = job.Len;
        int n2 = 2 * height;
        int group = job.Special || lanes == 1 ? 1 : ColumnGroup(2);
        DftCall<T> call = job.Call;

        TV[] buf = ArrayPool<TV>.Shared.Rent(n2 * group);
        TV[] outArr = ArrayPool<TV>.Shared.Rent(n2);
        TV[] scratchArr = ArrayPool<TV>.Shared.Rent(DftKernels<T, TV, TL>.ScratchLength(call));
        try
        {
            for (int g0 = b0; g0 < b1; g0 += group)
            {
                int g1 = Math.Min(g0 + group, b1);
                if (job.Special)
                {
                    LoadSpecialColumns<TV, TL>(job, buf.AsSpan(0, n2), g0);
                }
                else
                {
                    LoadColumns(job, buf, g0, g1);
                }

                for (int g = g0; g < g1; g++)
                {
                    Span<TV> lane = buf.AsSpan((g - g0) * n2, n2);
                    DftKernels<T, TV, TL>.Dft(call, lane, outArr.AsSpan(0, n2), scratchArr);
                    outArr.AsSpan(0, n2).CopyTo(lane);
                }

                if (job.Special)
                {
                    StoreSpecialColumns<TV>(job, buf.AsSpan(0, n2), g0);
                }
                else
                {
                    StoreColumns(job, buf, g0, g1);
                }
            }
        }
        finally
        {
            ArrayPool<TV>.Shared.Return(scratchArr);
            ArrayPool<TV>.Shared.Return(outArr);
            ArrayPool<TV>.Shared.Return(buf);
        }
    }

    /// <summary>Loads complex columns <c>SrcBase + 2·j</c> of the batches <c>[g0, g1)</c>; batch g uses elements <c>(g - g0)·2H</c> onwards.</summary>
    private static void LoadColumns<TV>(LineJob job, TV[] buf, int g0, int g1)
        where TV : unmanaged
    {
        int lanes = typeof(TV) == typeof(T) ? 1 : Vector<T>.Count;
        int height = job.Len, n2 = 2 * height;
        Plane src = job.Src;
        Span<T> bufT = MemoryMarshal.Cast<TV, T>(buf.AsSpan(0, n2 * (g1 - g0)));
        int span = 2 * Math.Min(lanes * (g1 - g0), job.Count - (g0 * lanes));
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<T> row = src.Row(y, job.SrcBase + (2 * g0 * lanes) + span);
            for (int g = g0; g < g1; g++)
            {
                int j0 = g * lanes;
                int used = Math.Min(lanes, job.Count - j0);
                Span<T> lane = bufT.Slice((g - g0) * n2 * lanes, n2 * lanes);
                if (lanes == 1)
                {
                    lane[2 * y] = row[job.SrcBase + (2 * j0)];
                    lane[(2 * y) + 1] = row[job.SrcBase + (2 * j0) + 1];
                    continue;
                }

                if (used < lanes)
                {
                    lane.Slice(2 * y * lanes, 2 * lanes).Clear();
                }

                LaneTranspose.Deinterleave(row.Slice(job.SrcBase + (2 * j0), 2 * used), used, lane, 2 * y);
            }
        }
    }

    private static void StoreColumns<TV>(LineJob job, TV[] buf, int g0, int g1)
        where TV : unmanaged
    {
        int lanes = typeof(TV) == typeof(T) ? 1 : Vector<T>.Count;
        int height = job.Len, n2 = 2 * height;
        Plane dst = job.Dst;
        ReadOnlySpan<T> bufT = MemoryMarshal.Cast<TV, T>(buf.AsSpan(0, n2 * (g1 - g0)));
        int span = 2 * Math.Min(lanes * (g1 - g0), job.Count - (g0 * lanes));
        for (int y = 0; y < height; y++)
        {
            Span<T> row = dst.Row(y, job.DstBase + (2 * g0 * lanes) + span);
            for (int g = g0; g < g1; g++)
            {
                int j0 = g * lanes;
                int used = Math.Min(lanes, job.Count - j0);
                ReadOnlySpan<T> lane = bufT.Slice((g - g0) * n2 * lanes, n2 * lanes);
                if (lanes == 1)
                {
                    row[job.DstBase + (2 * j0)] = lane[2 * y];
                    row[job.DstBase + (2 * j0) + 1] = lane[(2 * y) + 1];
                    continue;
                }

                LaneTranspose.Interleave(lane, 2 * y, used, row.Slice(job.DstBase + (2 * j0), 2 * used));
            }
        }
    }

    /// <summary>
    /// Loads column 0 (lane 0) and, for an even real width, the last column (lane 1), as
    /// <c>colDft</c> before the complex transform.
    /// </summary>
    private static void LoadSpecialColumns<TV, TL>(LineJob job, Span<TV> buf, int batch)
        where TV : unmanaged
        where TL : struct, IDftLanes<TV, T>
    {
        int lanes = TL.Count;
        int height = job.Len, width = job.Width;
        Plane src = job.Src;
        Span<T> inT = MemoryMarshal.Cast<TV, T>(buf);
        inT.Clear();
        int j0 = batch * lanes;
        int used = Math.Min(lanes, job.Count - j0);
        for (int l = 0; l < used; l++)
        {
            bool second = j0 + l == 1;
            if (!job.Inv)
            {
                int col = second ? job.DstCn + width - 2 : 0;
                for (int y = 0; y < height; y++)
                {
                    inT[(2 * y * lanes) + l] = src.Row(y, col + 1)[col];
                }
            }
            else if (job.SrcCn == 1)
            {
                int col = second ? width - 1 : 0;
                for (int y = 0; y < height; y++)
                {
                    inT[(y * lanes) + l] = src.Row(y, col + 1)[col];
                }
            }
            else
            {
                int col = second ? 2 * job.B : 0;
                for (int y = 0; y < height; y++)
                {
                    ReadOnlySpan<T> row = src.Row(y, col + 2);
                    inT[(2 * y * lanes) + l] = row[col];
                    inT[(((2 * y) + 1) * lanes) + l] = row[col + 1];
                }
            }
        }

        if (job.Inv && job.SrcCn == 1)
        {
            DftKernels<T, TV, TL>.ExpandCcs(buf, height);
        }
    }

    private static void StoreSpecialColumns<TV>(LineJob job, Span<TV> buf, int batch)
        where TV : unmanaged
    {
        int lanes = typeof(TV) == typeof(T) ? 1 : Vector<T>.Count;
        int height = job.Len, width = job.Width;
        Plane dst = job.Dst;
        Span<T> outT = MemoryMarshal.Cast<TV, T>(buf);
        int j0 = batch * lanes;
        int used = Math.Min(lanes, job.Count - j0);
        for (int l = 0; l < used; l++)
        {
            bool second = j0 + l == 1;
            if (job.DstCn == 1)
            {
                int col = second ? width - 1 : 0;
                if (!job.Inv)
                {
                    outT[lanes + l] = outT[l];
                    for (int y = 0; y < height; y++)
                    {
                        dst.Row(y, col + 1)[col] = outT[((1 + y) * lanes) + l];
                    }
                }
                else
                {
                    for (int y = 0; y < height; y++)
                    {
                        dst.Row(y, col + 1)[col] = outT[(2 * y * lanes) + l];
                    }
                }
            }
            else
            {
                int col = second ? 2 * job.B : 0;
                for (int y = 0; y < height; y++)
                {
                    Span<T> row = dst.Row(y, col + 2);
                    row[col] = outT[(2 * y * lanes) + l];
                    row[col + 1] = outT[(((2 * y) + 1) * lanes) + l];
                }
            }
        }
    }

    private static void DctRows<TV, TL>(LineJob job, int b0, int b1)
        where TV : unmanaged
        where TL : struct, IDftLanes<TV, T>
    {
        int lanes = TL.Count;
        int len = job.Len;
        DctPlan<T> plan = job.Dct!;
        var call = new DftCall<T>(plan.Dft, false, T.One);
        Plane src = job.Src, dst = job.Dst;

        TV[] xArr = ArrayPool<TV>.Shared.Rent(len);
        TV[] yArr = ArrayPool<TV>.Shared.Rent(len);
        TV[] bufArr = ArrayPool<TV>.Shared.Rent((2 * len) + 2);
        TV[] scratchArr = ArrayPool<TV>.Shared.Rent(DftKernels<T, TV, TL>.ScratchLength(call));
        try
        {
            Span<TV> x = xArr.AsSpan(0, len);
            Span<TV> y = yArr.AsSpan(0, len);
            Span<T> xT = MemoryMarshal.Cast<TV, T>(x);
            Span<T> yT = MemoryMarshal.Cast<TV, T>(y);

            for (int batch = b0; batch < b1; batch++)
            {
                int r0 = batch * lanes;
                int used = Math.Min(lanes, job.Count - r0);
                if (used < lanes)
                {
                    xT.Clear();
                }

                int start = src.Offset + (r0 * src.Stride);
                if (lanes == 1)
                {
                    src.Array.AsSpan(start, len).CopyTo(xT);
                }
                else
                {
                    LaneTranspose.RowsToLanes(src.Array, start, src.Stride, used, len, xT, 0);
                }

                Transform<TV, TL>(job.Inv, call, plan.Wave, x, y, bufArr, scratchArr);

                int dstStart = dst.Offset + (r0 * dst.Stride);
                if (lanes == 1)
                {
                    yT.CopyTo(dst.Array.AsSpan(dstStart, len));
                }
                else
                {
                    LaneTranspose.LanesToRows<T>(yT, 0, used, len, dst.Array, dstStart, dst.Stride);
                }
            }
        }
        finally
        {
            ArrayPool<TV>.Shared.Return(scratchArr);
            ArrayPool<TV>.Shared.Return(bufArr);
            ArrayPool<TV>.Shared.Return(yArr);
            ArrayPool<TV>.Shared.Return(xArr);
        }
    }

    /// <summary>Column DCTs of the batches <c>[b0, b1)</c>; lane l of batch g is column <c>g·L + l</c>.</summary>
    private static void DctCols<TV, TL>(LineJob job, int b0, int b1)
        where TV : unmanaged
        where TL : struct, IDftLanes<TV, T>
    {
        int lanes = TL.Count;
        int len = job.Len;
        int group = lanes == 1 ? 1 : ColumnGroup(1);
        DctPlan<T> plan = job.Dct!;
        var call = new DftCall<T>(plan.Dft, false, T.One);
        Plane src = job.Src, dst = job.Dst;

        TV[] xArr = ArrayPool<TV>.Shared.Rent(len * group);
        TV[] yArr = ArrayPool<TV>.Shared.Rent(len * group);
        TV[] bufArr = ArrayPool<TV>.Shared.Rent((2 * len) + 2);
        TV[] scratchArr = ArrayPool<TV>.Shared.Rent(DftKernels<T, TV, TL>.ScratchLength(call));
        try
        {
            Span<T> xT = MemoryMarshal.Cast<TV, T>(xArr.AsSpan(0, len * group));
            Span<T> yT = MemoryMarshal.Cast<TV, T>(yArr.AsSpan(0, len * group));

            for (int g0 = b0; g0 < b1; g0 += group)
            {
                int g1 = Math.Min(g0 + group, b1);
                int c0 = g0 * lanes;
                int cols = Math.Min((g1 - g0) * lanes, job.Count - c0);
                if (cols < (g1 - g0) * lanes)
                {
                    xT.Clear();
                }

                for (int e = 0; e < len; e++)
                {
                    Span<T> row = src.Row(e, c0 + cols)[c0..];
                    for (int g = 0, c = 0; c < cols; g++, c += lanes)
                    {
                        int n = cols - c;
                        if (n >= lanes)
                        {
                            xArr[(g * len) + e] = Unsafe.ReadUnaligned<TV>(ref Unsafe.As<T, byte>(ref row[c]));
                        }
                        else
                        {
                            row.Slice(c, n).CopyTo(xT.Slice(((g * len) + e) * lanes, n));
                        }
                    }
                }

                for (int g = 0; g < g1 - g0; g++)
                {
                    Transform<TV, TL>(job.Inv, call, plan.Wave, xArr.AsSpan(g * len, len), yArr.AsSpan(g * len, len), bufArr, scratchArr);
                }

                for (int e = 0; e < len; e++)
                {
                    Span<T> row = dst.Row(e, c0 + cols)[c0..];
                    for (int g = 0, c = 0; c < cols; g++, c += lanes)
                    {
                        int n = cols - c;
                        if (n >= lanes)
                        {
                            Unsafe.WriteUnaligned(ref Unsafe.As<T, byte>(ref row[c]), yArr[(g * len) + e]);
                        }
                        else
                        {
                            yT.Slice(((g * len) + e) * lanes, n).CopyTo(row.Slice(c, n));
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<TV>.Shared.Return(scratchArr);
            ArrayPool<TV>.Shared.Return(bufArr);
            ArrayPool<TV>.Shared.Return(yArr);
            ArrayPool<TV>.Shared.Return(xArr);
        }
    }

    private static void Transform<TV, TL>(bool inv, in DftCall<T> call, T[] wave, ReadOnlySpan<TV> x, Span<TV> y, Span<TV> buf, Span<TV> scratch)
        where TV : unmanaged
        where TL : struct, IDftLanes<TV, T>
    {
        if (inv)
        {
            DftKernels<T, TV, TL>.Idct(call, wave, x, y, buf, scratch);
        }
        else
        {
            DftKernels<T, TV, TL>.Dct(call, wave, x, y, buf, scratch);
        }
    }

    /// <summary><c>complementComplex</c>: fills the upper half of complex rows by conjugate symmetry.</summary>
    private static void ComplementComplex(Plane p, int n, int len, int dims)
    {
        if ((long)n * len < (1 << 18))
        {
            ComplementRows(p, n, len, dims, 0, len);
            return;
        }

        ParallelRows.For(len, n / 4, (r0, r1) => ComplementRows(p, n, len, dims, r0, r1));
    }

    private static void ComplementRows(Plane p, int n, int len, int dims, int r0, int r1)
    {
        for (int i = r0; i < r1; i++)
        {
            Span<T> row = p.Row(i, 2 * n);
            int qi = dims == 1 || i == 0 || i * 2 == len ? i : len - i;
            ReadOnlySpan<T> q = p.Row(qi, 2 * n);
            for (int j = 1; j < (n + 1) / 2; j++)
            {
                row[(n - j) * 2] = q[j * 2];
                row[((n - j) * 2) + 1] = -q[(j * 2) + 1];
            }
        }
    }

    /// <summary>Array, offset and stride of a matrix; row <c>y</c> starts at <c>Offset + y·Stride</c>.</summary>
    private readonly struct Plane
    {
        public Plane(Mat<T> m)
        {
            Array = m.DangerousGetArray();
            Offset = m.DangerousGetOffset();
            Stride = m.Stride;
        }

        public T[] Array { get; }

        public int Offset { get; }

        public int Stride { get; }

        public Span<T> Row(int y, int length) => Array.AsSpan(Offset + (y * Stride), length);
    }

    /// <summary>Parameters of the row or column transforms of one stage.</summary>
    private sealed class LineJob
    {
        public LineJob(Plane src, Plane dst, int len, int count, bool inv)
        {
            Src = src;
            Dst = dst;
            Len = len;
            Count = count;
            Inv = inv;
        }

        public Plane Src { get; }

        public Plane Dst { get; }

        /// <summary>Transform length.</summary>
        public int Len { get; }

        /// <summary>Number of line transforms.</summary>
        public int Count { get; }

        public bool Inv { get; }

        public DctPlan<T>? Dct { get; init; }

        public DftCall<T> Call { get; init; }

        public Kind Kind { get; init; }

        public bool ComplexOutput { get; init; }

        public bool ComplexInput { get; init; }

        public int InLen { get; init; }

        public int OutLen { get; init; }

        public int DptrOffset { get; init; }

        public int DstFullLen { get; init; }

        public int SrcCn { get; init; }

        public int DstCn { get; init; }

        public int Width { get; init; }

        public int B { get; init; }

        public int SrcBase { get; init; }

        public int DstBase { get; init; }

        /// <summary>Column 0 and the last column of a real column pass.</summary>
        public bool Special { get; private init; }

        /// <summary>
        /// Work estimate of one batch for <see cref="ParallelRows"/>; one unit is about 1.5 ns. A
        /// batch costs about the same with one lane or a full vector.
        /// </summary>
        public long WorkPerBatch => (long)Len * BitOperations.Log2((uint)Math.Max(Len, 2)) / 2;

        public LineJob WithSpecial(int count) => new(Src, Dst, Len, count, Inv)
        {
            Call = Call,
            SrcCn = SrcCn,
            DstCn = DstCn,
            Width = Width,
            B = B,
            Special = true,
        };
    }
}
