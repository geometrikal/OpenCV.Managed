// This implementation is ported from the OpenCV contour implementation.
//
// Original sources:
//   modules/imgproc/src/contours_approx.cpp        (in this repo: src/OpenCV/modules/imgproc/src/contours_approx.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Teh–Chin (TC89) chain-code approximation, ported faithfully from OpenCV's
/// <c>approximateChainTC89</c> in <c>contours_approx.cpp</c>.
///
/// <para>Converts a Freeman chain code (8-connected border directions, values
/// 0..7) starting at a given origin into a polygonal contour. Four methods are
/// supported, matching OpenCV's <c>CHAIN_APPROX_*</c> integer values:
/// <c>1 = NONE</c>, <c>2 = SIMPLE</c>, <c>3 = TC89_L1</c>,
/// <c>4 = TC89_KCOS</c>. In practice this port is only invoked for the two TC89
/// methods, but all paths are ported for completeness.</para>
///
/// <para>The float-bit comparisons in the original C++ (which reinterpret a
/// <c>float</c> through a union as <c>int</c>) are reproduced exactly using
/// <see cref="BitConverter.SingleToInt32Bits(float)"/>; floats are never
/// compared directly because the sign bit of values such as <c>-0.0f</c> is
/// load-bearing.</para>
/// </summary>
internal static class ChainApproximatorTc89
{
    // OpenCV CHAIN_APPROX_* integer values.
    private const int MethodNone = 1;
    private const int MethodSimple = 2;
    private const int MethodTc89L1 = 3;
    private const int MethodTc89KCos = 4;

    // abs_diff[16]: C++ initialises 15 entries, so the trailing element is 0.
    // Indexed by (cur - prev + 7) with cur, prev in 0..7 (range 0..14); index 15
    // is unreachable in practice but retained to match the original table size.
    private static readonly int[] AbsDiff =
        [1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0];

    private static readonly Point[] ChainCodeDeltas =
        [new(1, 0), new(1, -1), new(0, -1), new(-1, -1), new(-1, 0), new(-1, 1), new(0, 1), new(1, 1)];

    /// <summary>A restored chain point and its running approximation state.</summary>
    private struct ApproxItem
    {
        /// <summary>Point in image coordinates.</summary>
        public Point Pt;

        /// <summary>Support region size (C++ <c>size_t k</c>).</summary>
        public int K;

        /// <summary>1-curvature measure (C++ <c>int s</c>).</summary>
        public int S;

        /// <summary>Whether the point has been eliminated from the result.</summary>
        public bool Removed;
    }

    /// <summary>
    /// Approximates a Freeman chain code into a polygon.
    /// </summary>
    /// <param name="chain">Freeman chain codes (values 0..7).</param>
    /// <param name="origin">Starting point of the chain, in image coordinates.</param>
    /// <param name="method">
    /// OpenCV approximation method as an integer (1 = None, 2 = Simple,
    /// 3 = TC89_L1, 4 = TC89_KCOS).
    /// </param>
    /// <param name="output">Destination point list; cleared and refilled.</param>
    public static void Approximate(List<sbyte> chain, Point origin, int method, List<Point> output)
    {
        if (chain.Count == 0)
        {
            output.Clear();
            output.Add(origin);
            return;
        }

        bool isApprox = method == MethodTc89L1 || method == MethodTc89KCos;

        List<ApproxItem> ares = Pass0(chain, origin, isApprox, method == MethodNone);

        if (isApprox)
        {
            // Pass 1.
            // Determine the support region for every remaining point.
            for (int i = 0; i < ares.Count; ++i)
            {
                if (ares[i].Removed)
                    continue;

                int k = CalcSupport(ares, i);

                Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
                items[i].K = k;

                if (method == MethodTc89KCos)
                {
                    // CalcCosine reads the K just written above.
                    items[i].S = CalcCosine(ares, i);
                }
            }

            // Pass 2.
            // Non-maxima suppression.
            for (int i = 0; i < ares.Count; ++i)
            {
                if (CalcNmsCleanup(ares, i))
                {
                    Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
                    items[i].S = 0; // "clear"
                    items[i].Removed = true;
                }
            }

            // Pass 3.
            // Remove non-dominant points that have a 1-length support region.
            for (int i = 0; i < ares.Count; ++i)
            {
                if (ares[i].Removed)
                    continue;

                if (ares[i].K == 1 && CalcDominance(ares, i))
                {
                    Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
                    items[i].S = 0;
                    items[i].Removed = true;
                }
            }

            if (method == MethodTc89L1)
            {
                // Pass 4.
                // Clean up remaining couples of points (L1 only).
                bool skip = false;
                int newStartIdx = 0;
                int len = ares.Count;
                if (ares[0].S != 0 && ares[len - 1].S != 0)
                {
                    if (!CalcNewStart(ares, ref newStartIdx))
                    {
                        skip = true;
                    }
                }
                if (!skip)
                {
                    PassCleanup(ares, newStartIdx);
                }
            }
        }

        GatherPoints(ares, output);
    }

    // Pass 0.
    // Restores all the digital curve points from the chain code and removes the
    // points (from the resultant polygon) that have zero 1-curvature.
    private static List<ApproxItem> Pass0(List<sbyte> chain, Point pt, bool isApprox, bool isFull)
    {
        int len = chain.Count;
        var res = new List<ApproxItem>(len / 2);
        for (int i = 0; i < len; ++i)
        {
            int prev = (i == 0) ? chain[len - 1] : chain[i - 1];
            int cur = chain[i];
            int s = AbsDiff[cur - prev + 7];
            if ((!isApprox && (isFull || s != 0)) || isApprox)
            {
                var item = new ApproxItem { Pt = pt, K = 0, S = s, Removed = false };
                if (s == 0)
                {
                    item.Removed = true;
                }
                res.Add(item);
            }
            pt += ChainCodeDeltas[cur];
        }
        return res;
    }

    private static void GatherPoints(List<ApproxItem> ares, List<Point> output)
    {
        output.Clear();
        for (int i = 0; i < ares.Count; ++i)
        {
            if (!ares[i].Removed)
            {
                output.Add(ares[i].Pt);
            }
        }
    }

    private static int CalcSupport(List<ApproxItem> ares, int i)
    {
        int len = ares.Count;

        // Determine support region.
        int dNum = 0;
        int l = 0;
        int k = 1;
        for (; ; k++)
        {
            // C++ has CV_Assert(k <= len); the closed-curve geometry guarantees
            // an earlier break in every real case.

            // Calc indices.
            int i1 = (i >= k) ? (i - k) : (len - k + i);
            int i2 = (i + k < len) ? (i + k) : (i + k - len);

            int dx = ares[i2].Pt.X - ares[i1].Pt.X;
            int dy = ares[i2].Pt.Y - ares[i1].Pt.Y;

            // Distance between p_(i - k) and p_(i + k).
            int lk = dx * dx + dy * dy;

            // Distance between p_i and the line (p_(i-k), p_(i+k)).
            int dkNum =
                (ares[i].Pt.X - ares[i1].Pt.X) * dy - (ares[i].Pt.Y - ares[i1].Pt.Y) * dx;

            // Reproduce the C++ union { int i; float f; } bit comparison.
            float df = (float)((double)dNum * lk - (double)dkNum * l);
            int di = BitConverter.SingleToInt32Bits(df);

            if (k > 1 && (l >= lk || ((dNum > 0 && di <= 0) || (dNum < 0 && di >= 0))))
            {
                break;
            }

            dNum = dkNum;
            l = lk;
        }
        return k - 1;
    }

    private static int CalcCosine(List<ApproxItem> ares, int i)
    {
        int k = ares[i].K;
        int len = ares.Count;

        // Calc k-cosine curvature.
        int s = 0;
        for (int j = k; j > 0; j--)
        {
            int i1 = (i >= j) ? (i - j) : (len - j + i);
            int i2 = (i + j < len) ? (i + j) : (i + j - len);

            int dx1 = ares[i1].Pt.X - ares[i].Pt.X;
            int dy1 = ares[i1].Pt.Y - ares[i].Pt.Y;
            int dx2 = ares[i2].Pt.X - ares[i].Pt.X;
            int dy2 = ares[i2].Pt.Y - ares[i].Pt.Y;

            if ((dx1 | dy1) == 0 || (dx2 | dy2) == 0)
            {
                break;
            }

            // C++: double temp_num = dx1 * dx2 + dy1 * dy2; -- the products and
            // sum are int arithmetic, then widened to double.
            double tempNum = dx1 * dx2 + dy1 * dy2;
            tempNum = (float)(tempNum / Math.Sqrt(((double)dx1 * dx1 + (double)dy1 * dy1) *
                                                  ((double)dx2 * dx2 + (double)dy2 * dy2)));

            float skf = (float)(tempNum + 1.1);
            int ski = BitConverter.SingleToInt32Bits(skf);

            // C++: CV_Assert(0 <= sk.f && sk.f <= 2.2);

            if (j < k && ski <= s)
            {
                break;
            }

            s = ski;
        }
        return s;
    }

    private static bool CalcNmsCleanup(List<ApproxItem> ares, int i)
    {
        int k2 = ares[i].K >> 1;
        int s = ares[i].S;
        int len = ares.Count;
        int j;
        for (j = 1; j <= k2; j++)
        {
            int i1 = (i >= j) ? (i - j) : (len - j + i);
            int i2 = (i + j < len) ? (i + j) : (i + j - len);
            if (ares[i1].S > s || ares[i2].S > s)
            {
                break;
            }
        }
        return j <= k2;
    }

    private static bool CalcDominance(List<ApproxItem> ares, int i)
    {
        int len = ares.Count;

        // C++ has CV_Assert(len > 0).
        int i1 = (i >= 1) ? (i - 1) : (len - 1 + i);
        int i2 = (i + 1 < len) ? (i + 1) : (i + 1 - len);
        return ares[i].S <= ares[i1].S || ares[i].S <= ares[i2].S;
    }

    private static int GetNextIdx(List<ApproxItem> ares, int start)
    {
        int len = ares.Count;
        int res = start + 1;
        for (; res < len; ++res)
        {
            if (!ares[res].Removed)
            {
                break;
            }
        }
        return res;
    }

    private static void ClearUntil(List<ApproxItem> ares, int start, int finish)
    {
        int len = ares.Count;
        Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
        for (int i = start + 1; i < finish && i < len; ++i)
        {
            items[i].Removed = true;
        }
    }

    private static bool CalcNewStart(List<ApproxItem> ares, ref int res)
    {
        int len = ares.Count;

        // C++ has CV_Assert(len > 0).
        int i1;

        // Remove all previous items from the beginning.
        {
            Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
            for (i1 = 1; i1 < len && ares[i1].S != 0; i1++)
            {
                items[i1 - 1].S = 0;
            }
        }
        if (i1 == len)
        {
            // All points survived - skip to the end.
            return false;
        }
        i1--;

        int i2;

        // Remove all following items from the end.
        for (i2 = len - 2; i2 > 0 && ares[i2].S != 0; i2--)
        {
            ClearUntil(ares, i2, len);
            Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
            items[i2 + 1].S = 0;
        }
        i2++;

        // Only two points left.
        if (i1 == 0 && i2 == len - 1)
        {
            // Find the first non-removed element from the start.
            i1 = GetNextIdx(ares, 0);
            // Append the first item to the end (as a live copy).
            ApproxItem appended = ares[0];
            appended.Removed = false;
            ares.Add(appended);
        }
        res = i1;
        return true;
    }

    private static void PassCleanup(List<ApproxItem> ares, int startIdx)
    {
        int count = 1;

        int len = ares.Count;
        int first = startIdx;
        int prev = startIdx;
        for (int i = startIdx; i < len; ++i)
        {
            if (ares[i].Removed)
            {
                continue;
            }
            int nextIdx = GetNextIdx(ares, i);
            if (nextIdx == len || nextIdx - i != 1)
            {
                if (count >= 2)
                {
                    if (count == 2)
                    {
                        int s1 = ares[prev].S;
                        int s2 = ares[i].S;

                        Span<ApproxItem> items = CollectionsMarshal.AsSpan(ares);
                        if (s1 > s2 || (s1 == s2 && ares[prev].K <= ares[i].K))
                        {
                            // Remove second.
                            items[i].Removed = true;
                        }
                        else
                        {
                            // Remove first.
                            items[prev].Removed = true;
                        }
                    }
                    else
                    {
                        first = GetNextIdx(ares, first);
                        ClearUntil(ares, first, i);
                    }
                }
                first = i;
                count = 1;
            }
            else
            {
                ++count;
            }
            prev = i;
        }
    }
}
