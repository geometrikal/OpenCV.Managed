// This implementation is ported from the OpenCV polygon approximation.
//
// Original source:
//   modules/geometry/src/approx.cpp (approxPolyDP_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;

/// <summary>
/// OpenCV <c>approxPolyDP_</c>: Ramer-Douglas-Peucker simplification with an explicit slice stack,
/// the OpenCV start-point search for closed curves and the final collinear-point clean-up.
/// </summary>
internal static class DescrApproxPoly
{
    /// <summary>
    /// Writes the approximated curve to <paramref name="dst"/> and returns its point count.
    /// <paramref name="dst"/> must hold at least <c>src.Length</c> points.
    /// </summary>
    public static int Run<TP, TOps>(ReadOnlySpan<TP> src, Span<TP> dst, bool isClosed0, double eps)
        where TP : unmanaged
        where TOps : struct, IDescrPointOps<TP>
    {
        int count = src.Length;
        if (count == 0)
        {
            return 0;
        }

        int initIters = 3;
        int sliceStart = 0, sliceEnd = 0, rightStart = 0, rightEnd = 0;
        TP startPt = default, endPt, pt = default;
        int pos = 0, newCount = 0;
        bool isClosed = isClosed0;
        bool leEps = false;

        // The stack holds (start, end) pairs.
        int[] stack = ArrayPool<int>.Shared.Rent(2 * Math.Max(count, 4));
        int top = 0;
        try
        {
            eps *= eps;

            if (!isClosed)
            {
                rightStart = count;
                endPt = src[0];
                startPt = src[count - 1];
                if (TOps.Differ(startPt, endPt))
                {
                    sliceStart = 0;
                    sliceEnd = count - 1;
                    Push(ref stack, ref top, sliceStart, sliceEnd);
                }
                else
                {
                    isClosed = true;
                    initIters = 1;
                }
            }

            if (isClosed)
            {
                // Approximate the two farthest points of the curve.
                rightStart = 0;
                for (int i = 0; i < initIters; i++)
                {
                    double maxDist = 0;
                    pos = (pos + rightStart) % count;
                    startPt = src[pos];
                    if (++pos >= count)
                    {
                        pos = 0;
                    }

                    for (int j = 1; j < count; j++)
                    {
                        pt = src[pos];
                        if (++pos >= count)
                        {
                            pos = 0;
                        }

                        double dx = TOps.Dx(pt, startPt);
                        double dy = TOps.Dy(pt, startPt);
                        double dist = (dx * dx) + (dy * dy);
                        if (dist > maxDist)
                        {
                            maxDist = dist;
                            rightStart = j;
                        }
                    }

                    leEps = maxDist <= eps;
                }

                if (!leEps)
                {
                    rightEnd = sliceStart = pos % count;
                    sliceEnd = rightStart = (rightStart + sliceStart) % count;
                    Push(ref stack, ref top, rightStart, rightEnd);
                    Push(ref stack, ref top, sliceStart, sliceEnd);
                }
                else
                {
                    dst[newCount++] = startPt;
                }
            }

            while (top > 0)
            {
                top--;
                sliceStart = stack[2 * top];
                sliceEnd = stack[(2 * top) + 1];
                endPt = src[sliceEnd];
                pos = sliceStart;
                startPt = src[pos];
                if (++pos >= count)
                {
                    pos = 0;
                }

                if (pos != sliceEnd)
                {
                    double maxDistMulLen = 0;
                    double dx = TOps.Dx(endPt, startPt);
                    double dy = TOps.Dy(endPt, startPt);
                    double segmentLen2 = (dx * dx) + (dy * dy);

                    while (pos != sliceEnd)
                    {
                        pt = src[pos];
                        if (++pos >= count)
                        {
                            pos = 0;
                        }

                        double px = TOps.Dx(pt, startPt), py = TOps.Dy(pt, startPt);
                        double projection = (px * dx) + (py * dy);
                        double distMulLen;
                        if (projection < 0)
                        {
                            distMulLen = TOps.SquaredDistance(pt, startPt) * segmentLen2;
                        }
                        else if (projection > segmentLen2)
                        {
                            distMulLen = TOps.SquaredDistance(pt, endPt) * segmentLen2;
                        }
                        else
                        {
                            double dist = (py * dx) - (px * dy);
                            distMulLen = dist * dist;
                        }

                        if (distMulLen > maxDistMulLen)
                        {
                            maxDistMulLen = distMulLen;
                            rightStart = (pos + count - 1) % count;
                        }
                    }

                    leEps = maxDistMulLen <= eps * segmentLen2;
                }
                else
                {
                    leEps = true;
                    startPt = src[sliceStart];
                }

                if (leEps)
                {
                    dst[newCount++] = startPt;
                }
                else
                {
                    rightEnd = sliceEnd;
                    sliceEnd = rightStart;
                    Push(ref stack, ref top, rightStart, rightEnd);
                    Push(ref stack, ref top, sliceStart, sliceEnd);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(stack);
        }

        if (!isClosed)
        {
            dst[newCount++] = src[count - 1];
        }

        return CleanUp<TP, TOps>(dst, newCount, isClosed0, eps);
    }

    /// <summary>
    /// Removes points on almost straight lines (the last stage of <c>approxPolyDP_</c>).
    /// <paramref name="eps"/> is the squared tolerance.
    /// </summary>
    private static int CleanUp<TP, TOps>(Span<TP> dst, int count, bool isClosed, double eps)
        where TP : unmanaged
        where TOps : struct, IDescrPointOps<TP>
    {
        int newCount = count;
        int open = isClosed ? 0 : 1;
        int pos = isClosed ? count - 1 : 0;
        TP startPt = dst[pos];
        if (++pos >= count)
        {
            pos = 0;
        }

        int wpos = pos;
        TP pt = dst[pos];
        if (++pos >= count)
        {
            pos = 0;
        }

        for (int i = open; i < count - open && newCount > 2; i++)
        {
            TP endPt = dst[pos];
            if (++pos >= count)
            {
                pos = 0;
            }

            double dx = TOps.Dx(endPt, startPt);
            double dy = TOps.Dy(endPt, startPt);
            double dist = Math.Abs((TOps.Dx(pt, startPt) * dy) - (TOps.Dy(pt, startPt) * dx));
            double successiveInnerProduct = TOps.InnerProduct(startPt, pt, endPt);

            if (dist * dist <= 0.5 * eps * ((dx * dx) + (dy * dy)) && dx != 0 && dy != 0 && successiveInnerProduct >= 0)
            {
                newCount--;
                dst[wpos] = startPt = endPt;
                if (++wpos >= count)
                {
                    wpos = 0;
                }

                pt = dst[pos];
                if (++pos >= count)
                {
                    pos = 0;
                }

                i++;
                continue;
            }

            dst[wpos] = startPt = pt;
            if (++wpos >= count)
            {
                wpos = 0;
            }

            pt = endPt;
        }

        if (!isClosed)
        {
            dst[wpos] = pt;
        }

        return newCount;
    }

    private static void Push(ref int[] stack, ref int top, int start, int end)
    {
        if (2 * (top + 1) > stack.Length)
        {
            int[] grown = ArrayPool<int>.Shared.Rent(stack.Length * 2);
            stack.AsSpan(0, 2 * top).CopyTo(grown);
            ArrayPool<int>.Shared.Return(stack);
            stack = grown;
        }

        stack[2 * top] = start;
        stack[(2 * top) + 1] = end;
        top++;
    }
}
