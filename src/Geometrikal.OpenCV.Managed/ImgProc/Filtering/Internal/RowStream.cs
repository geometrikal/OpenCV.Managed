// This implementation follows the row ring buffer of the OpenCV filter engine.
//
// Original sources:
//   modules/imgproc/src/filter.dispatch.cpp (FilterEngine::proceed, ringBuf)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// The row stage and the column stage of a streaming filter. The row stage converts one
/// border-extended source line to one buffer row. The column stage computes one output row from
/// consecutive buffer rows.
/// </summary>
internal interface IRowStreamKernel<TSrc, TBuf, TDst>
    where TSrc : unmanaged
    where TBuf : unmanaged
    where TDst : unmanaged
{
    /// <summary>
    /// <paramref name="line"/> and <paramref name="wide"/> hold <c>(width + kw − 1)·cn</c>
    /// elements; <paramref name="row"/> holds one buffer row. <paramref name="wide"/> is scratch.
    /// </summary>
    void Row(ReadOnlySpan<TSrc> line, Span<TBuf> row, Span<TBuf> wide);

    /// <summary>
    /// Buffer row <c>k</c> of the kernel window starts at <c>rows[k · rowStride]</c>.
    /// <paramref name="scratch"/> holds one buffer row.
    /// </summary>
    void Column(ref TBuf rows, int rowStride, Span<TDst> dst, Span<TBuf> scratch);
}

/// <summary>
/// Streams the source through a small row buffer: the border-extended source rows of a block of
/// output rows go through the row stage, then the column stage writes the block. Consecutive
/// blocks share the <c>kh − 1</c> overlapping buffer rows. Every output row gets the same row
/// stage results as with a fully padded image, so the result does not depend on the block size
/// or the thread count.
/// </summary>
internal static class RowStream
{
    /// <summary>Target size of the row buffer of one stripe, in elements.</summary>
    private const int BufferElements = 1 << 15;

    /// <summary><see cref="ParallelRows"/> work units per vector operation (measured on 256×256 images).</summary>
    private const int VectorOpWork = 8;

    /// <summary>
    /// Runs the filter. <paramref name="bufferRowLength"/> is the element count of one buffer row;
    /// <paramref name="vectorOpsPerRow"/> estimates the vector operations of one output row.
    /// </summary>
    public static void Run<TSrc, TBuf, TDst, TKernel>(
        Mat<TSrc> src, Mat<TDst> dst, int kw, int kh, int ax, int ay, BorderType border, int bufferRowLength, TKernel kernel, long vectorOpsPerRow)
        where TSrc : unmanaged
        where TBuf : unmanaged
        where TDst : unmanaged
        where TKernel : struct, IRowStreamKernel<TSrc, TBuf, TDst>
    {
        if (typeof(TSrc) == typeof(TDst) && ReferenceEquals(src.DangerousGetArray(), dst.DangerousGetArray()))
        {
            src = src.Clone();
        }

        long workPerRow = vectorOpsPerRow * VectorOpWork;
        if (!ParallelRows.IsSerial(dst.Height, workPerRow))
        {
            PageTouch.Rows(dst);
        }

        var job = new Job<TSrc>(src, kw, kh, ax, ay, border, bufferRowLength);
        ParallelRows.For(dst.Height, workPerRow, (y0, y1) => Stripe<TSrc, TBuf, TDst, TKernel>(job, dst, kernel, y0, y1));
    }

    /// <summary>Element count of one border-extended source line.</summary>
    public static int LineLength(int width, int kw, int cn) => (width + kw - 1) * cn;

    private sealed record Job<TSrc>(Mat<TSrc> Src, int Kw, int Kh, int Ax, int Ay, BorderType Border, int BufferRowLength)
        where TSrc : unmanaged;

    private static void Stripe<TSrc, TBuf, TDst, TKernel>(Job<TSrc> job, Mat<TDst> dst, TKernel kernel, int y0, int y1)
        where TSrc : unmanaged
        where TBuf : unmanaged
        where TDst : unmanaged
        where TKernel : struct, IRowStreamKernel<TSrc, TBuf, TDst>
    {
        Mat<TSrc> src = job.Src;
        int cn = src.Channels;
        int w = src.Width;
        int h = src.Height;
        int kh = job.Kh;
        int ax = job.Ax;
        int right = job.Kw - 1 - ax;
        int lineLength = LineLength(w, job.Kw, cn);
        int rowLength = job.BufferRowLength;
        int block = Math.Clamp(BufferElements / Math.Max(rowLength, 1), 4, 64);
        block = Math.Min(block, y1 - y0);
        int bufferRows = block + kh - 1;

        TSrc[] lineArray = ArrayPool<TSrc>.Shared.Rent(lineLength);
        TBuf[] buffer = ArrayPool<TBuf>.Shared.Rent(bufferRows * rowLength);
        TBuf[] scratch = ArrayPool<TBuf>.Shared.Rent(rowLength + lineLength);
        int[] colMap = ArrayPool<int>.Shared.Rent(ax + right + 1);
        try
        {
            for (int x = 0; x < ax; x++)
            {
                colMap[x] = Borders.BorderInterpolate(x - ax, w, job.Border);
            }

            for (int x = 0; x < right; x++)
            {
                colMap[ax + x] = Borders.BorderInterpolate(w + x, w, job.Border);
            }

            Span<TSrc> line = lineArray.AsSpan(0, lineLength);
            Span<TBuf> scratchRow = scratch.AsSpan(0, rowLength);
            Span<TBuf> wide = scratch.AsSpan(rowLength, lineLength);
            int filled = 0;
            int lastSource = int.MinValue;
            for (int y = y0; y < y1; y += block)
            {
                int n = Math.Min(block, y1 - y);
                int needed = n + kh - 1;
                if (filled > 0)
                {
                    // The previous block started `block` rows earlier; keep its last kh − 1 rows.
                    int keep = filled - block;
                    Array.Copy(buffer, block * rowLength, buffer, 0, keep * rowLength);
                    filled = keep;
                }

                for (int p = y + filled; p < y + needed; p++)
                {
                    int sy = Borders.BorderInterpolate(p - job.Ay, h, job.Border);
                    if (sy != lastSource || sy < 0)
                    {
                        BuildLine(src, sy, line, colMap, ax, right, cn);
                        lastSource = sy;
                    }

                    kernel.Row(line, buffer.AsSpan((p - y) * rowLength, rowLength), wide);
                }

                filled = needed;
                for (int r = 0; r < n; r++)
                {
                    kernel.Column(ref buffer[r * rowLength], rowLength, dst.GetRowSpan(y + r), scratchRow);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(colMap);
            ArrayPool<TBuf>.Shared.Return(scratch);
            ArrayPool<TBuf>.Shared.Return(buffer);
            ArrayPool<TSrc>.Shared.Return(lineArray);
        }
    }

    /// <summary>Source row <paramref name="sy"/> with <c>left</c> and <c>right</c> border pixels, or zeros for a constant border row.</summary>
    private static void BuildLine<TSrc>(Mat<TSrc> src, int sy, Span<TSrc> line, int[] colMap, int left, int right, int cn)
        where TSrc : unmanaged
    {
        if (sy < 0)
        {
            line.Clear();
            return;
        }

        ReadOnlySpan<TSrc> s = src.GetReadOnlyRowSpan(sy);
        s.CopyTo(line[(left * cn)..]);
        int rightStart = (left * cn) + s.Length;
        for (int x = 0; x < left; x++)
        {
            CopyPixel(s, colMap[x], line.Slice(x * cn, cn));
        }

        for (int x = 0; x < right; x++)
        {
            CopyPixel(s, colMap[left + x], line.Slice(rightStart + (x * cn), cn));
        }
    }

    private static void CopyPixel<TSrc>(ReadOnlySpan<TSrc> s, int sx, Span<TSrc> pixel)
        where TSrc : unmanaged
    {
        if (sx < 0)
        {
            pixel.Clear();
        }
        else
        {
            s.Slice(sx * pixel.Length, pixel.Length).CopyTo(pixel);
        }
    }
}
