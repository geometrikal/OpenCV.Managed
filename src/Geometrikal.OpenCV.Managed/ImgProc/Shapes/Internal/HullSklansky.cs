// This implementation is ported from OpenCV's convex hull functions.
//
// Original source:
//   modules/geometry/src/convhull.cpp (Sklansky_, convexHull, convexityDefects, isContourConvex)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/convhull.cpp
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
/// Sklansky convex hull, convexity test and convexity defects over interleaved (x, y)
/// coordinates of type <see cref="int"/> or <see cref="float"/>.
/// </summary>
internal static class HullSklansky
{
    /// <summary>
    /// OpenCV <c>convexHull</c>. Writes the original indices of the hull vertices to
    /// <paramref name="hull"/> (length at least the point count) and returns their count.
    /// </summary>
    public static int ConvexHull<T>(ReadOnlySpan<T> xy, bool clockwise, bool returnPoints, Span<int> hull)
        where T : unmanaged, INumber<T>
    {
        int total = xy.Length / 2;
        if (total == 0)
        {
            return 0;
        }

        int[] rented = ArrayPool<int>.Shared.Rent((total * 2) + 4);
        try
        {
            Span<int> order = rented.AsSpan(0, total);
            Span<int> stack = rented.AsSpan(total, total + 4);
            SortPoints(xy, order);
            return Build(xy, order, stack, clockwise, returnPoints, hull);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    private static int Build<T>(ReadOnlySpan<T> xy, ReadOnlySpan<int> ptr, Span<int> stack, bool clockwise, bool returnPoints, Span<int> hullbuf)
        where T : unmanaged, INumber<T>
    {
        int total = ptr.Length;
        int nout = 0;
        int minyInd = 0, maxyInd = 0;
        for (int i = 1; i < total; i++)
        {
            T y = Y(xy, ptr[i]);
            if (Y(xy, ptr[minyInd]) > y)
            {
                minyInd = i;
            }

            if (Y(xy, ptr[maxyInd]) < y)
            {
                maxyInd = i;
            }
        }

        if (SameBits(xy, ptr[0], ptr[total - 1]))
        {
            hullbuf[nout++] = 0;
            return nout;
        }

        int tlOff = 0;
        int tlCount = Sklansky(xy, ptr, 0, maxyInd, stack, -1, 1);
        int trOff = tlCount;
        int trCount = Sklansky(xy, ptr, total - 1, maxyInd, stack[trOff..], -1, -1);
        if (!clockwise)
        {
            (tlOff, trOff) = (trOff, tlOff);
            (tlCount, trCount) = (trCount, tlCount);
        }

        for (int i = 0; i < tlCount - 1; i++)
        {
            hullbuf[nout++] = stack[tlOff + i];
        }

        for (int i = trCount - 1; i > 0; i--)
        {
            hullbuf[nout++] = stack[trOff + i];
        }

        int stopIdx = trCount > 2 ? stack[trOff + 1] : tlCount > 2 ? stack[tlOff + tlCount - 2] : -1;

        int blOff = 0;
        int blCount = Sklansky(xy, ptr, 0, minyInd, stack, 1, -1);
        int brOff = blCount;
        int brCount = Sklansky(xy, ptr, total - 1, minyInd, stack[brOff..], 1, 1);
        if (clockwise)
        {
            (blOff, brOff) = (brOff, blOff);
            (blCount, brCount) = (brCount, blCount);
        }

        if (stopIdx >= 0)
        {
            int checkIdx = blCount > 2 ? stack[blOff + 1] : blCount + brCount > 2 ? stack[brOff + 2 - blCount] : -1;
            if (checkIdx == stopIdx || (checkIdx >= 0 && SameBits(xy, ptr[checkIdx], ptr[stopIdx])))
            {
                // All points are collinear: the lower chain mirrors the upper chain.
                blCount = Math.Min(blCount, 2);
                brCount = Math.Min(brCount, 2);
            }
        }

        for (int i = 0; i < blCount - 1; i++)
        {
            hullbuf[nout++] = stack[blOff + i];
        }

        for (int i = brCount - 1; i > 0; i--)
        {
            hullbuf[nout++] = stack[brOff + i];
        }

        if (!returnPoints)
        {
            KeepMonotonousIndices(xy, ptr, hullbuf[..nout]);
        }

        for (int i = 0; i < nout; i++)
        {
            hullbuf[i] = ptr[hullbuf[i]];
        }

        CyclicShiftToMonotonous(hullbuf[..nout], stack);
        return nout;
    }

    /// <summary>Sorts point indices by x, then y, then index (the pointer order of OpenCV).</summary>
    private static void SortPoints<T>(ReadOnlySpan<T> xy, Span<int> order)
        where T : unmanaged, INumber<T>
    {
        int total = order.Length;
        ulong[] rentedKeys = ArrayPool<ulong>.Shared.Rent(total);
        try
        {
            Span<ulong> keys = rentedKeys.AsSpan(0, total);
            for (int i = 0; i < total; i++)
            {
                keys[i] = ((ulong)SortKey(xy[2 * i]) << 32) | SortKey(xy[(2 * i) + 1]);
                order[i] = i;
            }

            keys.Sort(order);

            // The sort is not stable: order equal points by index, as the pointer comparison does.
            int start = 0;
            for (int i = 1; i <= total; i++)
            {
                if (i == total || keys[i] != keys[start])
                {
                    if (i - start > 1)
                    {
                        order[start..i].Sort();
                    }

                    start = i;
                }
            }
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(rentedKeys);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint SortKey<T>(T v)
        where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(int))
        {
            return (uint)Unsafe.As<T, int>(ref v) ^ 0x80000000u;
        }

        // -0 and +0 compare equal in OpenCV, so both map to the key of +0.
        float f = Unsafe.As<T, float>(ref v) + 0.0f;
        uint bits = BitConverter.SingleToUInt32Bits(f);
        return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
    }

    private static int Sklansky<T>(ReadOnlySpan<T> xy, ReadOnlySpan<int> ptr, int start, int end, Span<int> stack, int nsign, int sign2)
        where T : unmanaged, INumber<T>
    {
        int incr = end > start ? 1 : -1;
        int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
        int stacksize = 3;

        if (start == end || SamePoint(xy, ptr[start], ptr[end]))
        {
            stack[0] = start;
            return 1;
        }

        stack[0] = pprev;
        stack[1] = pcur;
        stack[2] = pnext;

        end += incr;

        while (pnext != end)
        {
            T cury = Y(xy, ptr[pcur]);
            T nexty = Y(xy, ptr[pnext]);
            T by = nexty - cury;

            if (Sign(by) != nsign)
            {
                T ax = X(xy, ptr[pcur]) - X(xy, ptr[pprev]);
                T ay = cury - Y(xy, ptr[pprev]);
                T bx = X(xy, ptr[pnext]) - X(xy, ptr[pcur]);
                int convexitySign;
                bool aNonZero;
                if (typeof(T) == typeof(int))
                {
                    long a0 = int.CreateTruncating(ax), a1 = int.CreateTruncating(ay);
                    long b0 = int.CreateTruncating(bx), b1 = int.CreateTruncating(by);
                    convexitySign = Math.Sign((a1 * b0) - (a0 * b1));
                    aNonZero = a0 != 0 || a1 != 0;
                }
                else
                {
                    float a0 = float.CreateTruncating(ax), a1 = float.CreateTruncating(ay);
                    float b0 = float.CreateTruncating(bx), b1 = float.CreateTruncating(by);
                    Normalize(ref a0, ref a1);
                    Normalize(ref b0, ref b1);
                    double convexity = ((double)a1 * b0) - ((double)a0 * b1);
                    convexitySign = (convexity > 0 ? 1 : 0) - (convexity < 0 ? 1 : 0);
                    aNonZero = a0 != 0 || a1 != 0;
                }

                if (convexitySign == sign2 && aNonZero)
                {
                    pprev = pcur;
                    pcur = pnext;
                    pnext += incr;
                    stack[stacksize] = pnext;
                    stacksize++;
                }
                else if (pprev == start)
                {
                    pcur = pnext;
                    stack[1] = pcur;
                    pnext += incr;
                    stack[2] = pnext;
                }
                else
                {
                    stack[stacksize - 2] = pnext;
                    pcur = pprev;
                    pprev = stack[stacksize - 4];
                    stacksize--;
                }
            }
            else
            {
                pnext += incr;
                stack[stacksize - 1] = pnext;
            }
        }

        return --stacksize;
    }

    /// <summary><c>cv::normalize(Vec2f)</c>: scales by <c>1 / norm</c> computed in double.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Normalize(ref float x, ref float y)
    {
        double nv = Math.Sqrt(((double)x * x) + ((double)y * y));
        double s = nv != 0 ? 1.0 / nv : 0.0;
        x = (float)(x * s);
        y = (float)(y * s);
    }

    /// <summary>Moves duplicate-point indices so that the index sequence stays monotonous where possible.</summary>
    private static void KeepMonotonousIndices<T>(ReadOnlySpan<T> xy, ReadOnlySpan<int> ptr, Span<int> hullbuf)
        where T : unmanaged, INumber<T>
    {
        int nout = hullbuf.Length;
        int total = ptr.Length;
        for (int i = 0; i < nout; i++)
        {
            int prev = ptr[hullbuf[(i == 0 ? nout : i) - 1]];
            int next = ptr[hullbuf[(i + 1) % nout]];
            int cur = ptr[hullbuf[i]];
            if ((prev < cur && cur < next) || (prev > cur && cur > next))
            {
                continue;
            }

            for (int j = hullbuf[i] + 1; j < total; j++)
            {
                cur = ptr[j];
                if (!SameBits(xy, ptr[hullbuf[i]], cur))
                {
                    break;
                }

                if ((prev < cur && cur < next) || (prev > cur && cur > next))
                {
                    hullbuf[i] = j;
                    break;
                }
            }
        }
    }

    /// <summary>Rotates the index sequence so that it is ascending or descending, when a rotation can do that.</summary>
    private static void CyclicShiftToMonotonous(Span<int> hullbuf, Span<int> stack)
    {
        int nout = hullbuf.Length;
        if (nout < 3)
        {
            return;
        }

        int minIdx = 0, maxIdx = 0, lt = 0;
        for (int i = 1; i < nout; i++)
        {
            int idx = hullbuf[i];
            lt += hullbuf[i - 1] < idx ? 1 : 0;
            if (lt > 1 && lt <= i - 2)
            {
                break;
            }

            if (idx < hullbuf[minIdx])
            {
                minIdx = i;
            }

            if (idx > hullbuf[maxIdx])
            {
                maxIdx = i;
            }
        }

        int mmdist = Math.Abs(maxIdx - minIdx);
        if ((mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt >= nout - 2))
        {
            bool ascending = (maxIdx + 1) % nout == minIdx;
            int i0 = ascending ? minIdx : maxIdx, j = i0;
            if (i0 > 0)
            {
                int i;
                for (i = 0; i < nout; i++)
                {
                    int currIdx = stack[i] = hullbuf[j];
                    int nextJ = j + 1 < nout ? j + 1 : 0;
                    int nextIdx = hullbuf[nextJ];
                    if (i < nout - 1 && ascending != (currIdx < nextIdx))
                    {
                        break;
                    }

                    j = nextJ;
                }

                if (i == nout)
                {
                    stack[..nout].CopyTo(hullbuf);
                }
            }
        }
    }

    /// <summary>OpenCV <c>isContourConvex</c> for a non-empty contour.</summary>
    public static bool IsContourConvex<T>(ReadOnlySpan<T> xy)
        where T : unmanaged, INumber<T>
    {
        int n = xy.Length / 2;
        int prevIndex = ((n - 2) + n) % n;
        T curX = xy[2 * (n - 1)], curY = xy[(2 * (n - 1)) + 1];
        T dx0 = curX - xy[2 * prevIndex];
        T dy0 = curY - xy[(2 * prevIndex) + 1];
        int orientation = 0;

        for (int i = 0; i < n; i++)
        {
            T prevX = curX, prevY = curY;
            curX = xy[2 * i];
            curY = xy[(2 * i) + 1];

            T dx = curX - prevX;
            T dy = curY - prevY;
            T dxdy0 = dx * dy0;
            T dydx0 = dy * dx0;

            orientation |= dydx0 > dxdy0 ? 1 : (dydx0 < dxdy0 ? 2 : 3);
            if (orientation == 3)
            {
                return false;
            }

            dx0 = dx;
            dy0 = dy;
        }

        return true;
    }

    /// <summary>OpenCV <c>convexityDefects</c>. Returns an empty array when OpenCV releases the output.</summary>
    public static ConvexityDefect[] ConvexityDefects(ReadOnlySpan<Point> ptr, ReadOnlySpan<int> hptr)
    {
        int npoints = ptr.Length;
        if (npoints <= 3)
        {
            return [];
        }

        int hpoints = hptr.Length;
        if (hpoints == 0)
        {
            throw new ArgumentException("The hull must not be empty.", "hull");
        }

        if (hpoints < 3)
        {
            return [];
        }

        bool revOrientation = ((hptr[1] > hptr[0] ? 1 : 0) + (hptr[2] > hptr[1] ? 1 : 0) + (hptr[0] > hptr[2] ? 1 : 0)) != 2;
        int hcurr = hptr[revOrientation ? 0 : hpoints - 1];
        CheckIndex(hcurr, npoints);

        var defects = new List<ConvexityDefect>();
        int increasingIdx = -1;
        for (int i = 0; i < hpoints; i++)
        {
            int hnext = hptr[revOrientation ? hpoints - i - 1 : i];
            CheckIndex(hnext, npoints);

            Point pt0 = ptr[hcurr], pt1 = ptr[hnext];
            int inc = hcurr < hnext ? 1 : 0;
            if (increasingIdx < 0)
            {
                increasingIdx = inc ^ 1;
            }
            else if (increasingIdx != inc)
            {
                throw new ArgumentException(
                    "The convex hull indices are not monotonous, which can be in the case when the input contour contains self-intersections.",
                    "hull");
            }

            double dx0 = pt1.X - pt0.X;
            double dy0 = pt1.Y - pt0.Y;
            double scale = dx0 == 0 && dy0 == 0 ? 0.0 : 1.0 / Math.Sqrt((dx0 * dx0) + (dy0 * dy0));

            int deepest = -1;
            double depth = 0;
            bool isDefect = false;
            int j = hcurr;
            while (true)
            {
                j++;
                if (j >= npoints)
                {
                    j = 0;
                }

                if (j == hnext)
                {
                    break;
                }

                double dx = ptr[j].X - pt0.X;
                double dy = ptr[j].Y - pt0.Y;
                double dist = Math.Abs((-dy0 * dx) + (dx0 * dy)) * scale;
                if (dist > depth)
                {
                    depth = dist;
                    deepest = j;
                    isDefect = true;
                }
            }

            if (isDefect)
            {
                defects.Add(new ConvexityDefect(hcurr, hnext, deepest, ScalarConverter.CvRound(depth * 256)));
            }

            hcurr = hnext;
        }

        return defects.ToArray();
    }

    private static void CheckIndex(int index, int npoints)
    {
        if (index < 0 || index >= npoints)
        {
            throw new ArgumentOutOfRangeException("hull", "A hull index is outside the contour.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T X<T>(ReadOnlySpan<T> xy, int i)
        where T : unmanaged => xy[2 * i];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Y<T>(ReadOnlySpan<T> xy, int i)
        where T : unmanaged => xy[(2 * i) + 1];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool SamePoint<T>(ReadOnlySpan<T> xy, int a, int b)
        where T : unmanaged, INumber<T> => xy[2 * a] == xy[2 * b] && xy[(2 * a) + 1] == xy[(2 * b) + 1];

    /// <summary>
    /// Point equality of <c>convexHull</c> outside the Sklansky scan: OpenCV compares float points
    /// through <c>Point*</c>, so -0 and +0 differ there.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool SameBits<T>(ReadOnlySpan<T> xy, int a, int b)
        where T : unmanaged, INumber<T>
    {
        ReadOnlySpan<int> bits = MemoryMarshal.Cast<T, int>(xy);
        return bits[2 * a] == bits[2 * b] && bits[(2 * a) + 1] == bits[(2 * b) + 1];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Sign<T>(T v)
        where T : unmanaged, INumber<T> => (v > T.Zero ? 1 : 0) - (v < T.Zero ? 1 : 0);

    /// <summary>Reinterprets points as interleaved coordinates.</summary>
    public static ReadOnlySpan<int> Coords(ReadOnlySpan<Point> points) => MemoryMarshal.Cast<Point, int>(points);

    /// <summary>Reinterprets points as interleaved coordinates.</summary>
    public static ReadOnlySpan<float> Coords(ReadOnlySpan<Point2f> points) => MemoryMarshal.Cast<Point2f, float>(points);
}
