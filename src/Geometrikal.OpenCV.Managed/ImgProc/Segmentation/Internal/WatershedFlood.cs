// This implementation is ported from the OpenCV marker-based watershed.
//
// Original source:
//   modules/imgproc/src/segmentation.cpp (cv::watershed)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/segmentation.cpp
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
/// Priority flooding of <c>cv::watershed</c>: 256 FIFO queues keyed by the colour difference.
/// </summary>
/// <remarks>
/// The flood runs on packed rows, so pixel index <c>p</c> addresses marker <c>p</c> and image
/// bytes <c>3p</c>. ROI inputs are copied to pooled packed buffers; the markers are copied back.
/// Every pixel enters a queue at most once, so the queues are linked through one pooled array
/// indexed by pixel: <c>link[p + 1]</c> is the node after pixel <c>p</c> (0 ends a queue), and
/// the queue heads and tails follow the pixel links in the same array.
/// </remarks>
internal static class WatershedFlood
{
    private const int InQueue = -2;
    private const int Wshed = -1;
    private const int QueueCount = 256;

    public static void Run(Mat<byte> image, Mat<int> markers)
    {
        int width = markers.Width, height = markers.Height;
        markers.GetRowSpan(0).Fill(Wshed);
        markers.GetRowSpan(height - 1).Fill(Wshed);
        if (height < 3 || width < 3)
        {
            for (int i = 1; i < height - 1; i++)
            {
                Span<int> row = markers.GetRowSpan(i);
                row[0] = Wshed;
                row[width - 1] = Wshed;
            }

            return;
        }

        PooledMat<int> markerCopy = default;
        PooledMat<byte> imageCopy = default;
        int count = width * height;
        int[] links = ArrayPool<int>.Shared.Rent(count + 1 + (2 * QueueCount));
        try
        {
            Mat<int> packedMarkers = markers;
            if (!markers.IsContinuous)
            {
                markerCopy = PooledMat<int>.Rent(width, height, 1);
                packedMarkers = markerCopy.Mat;
                markers.CopyTo(packedMarkers);
            }

            Mat<byte> packedImage = image;
            if (!image.IsContinuous)
            {
                imageCopy = PooledMat<byte>.Rent(width, height, 3);
                packedImage = imageCopy.Mat;
                image.CopyTo(packedImage);
            }

            ref byte img = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(packedImage.DangerousGetArray()), packedImage.DangerousGetOffset());
            ref int mask = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(packedMarkers.DangerousGetArray()), packedMarkers.DangerousGetOffset());
            ref int link = ref MemoryMarshal.GetArrayDataReference(links);
            ref int ends = ref Unsafe.Add(ref link, count + 1);
            MemoryMarshal.CreateSpan(ref ends, 2 * QueueCount).Clear();
            if (Seed(ref img, ref mask, ref link, ref ends, width, height))
            {
                Flood(ref img, ref mask, ref link, ref ends, width);
            }

            if (!markers.IsContinuous)
            {
                packedMarkers.CopyTo(markers);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(links);
            if (markerCopy.Mat is not null)
            {
                markerCopy.Dispose();
            }

            if (imageCopy.Mat is not null)
            {
                imageCopy.Dispose();
            }
        }
    }

    /// <summary>
    /// Marks the left and right frame columns, clears negative interior values and queues every
    /// unknown pixel next to a seed. Returns <see langword="false"/> when no pixel is queued.
    /// </summary>
    private static bool Seed(ref byte img, ref int mask, ref int link, ref int ends, int width, int height)
    {
        nint istep = 3 * width;
        bool queued = false;
        bool vector = Vector.IsHardwareAccelerated;
        int vc = Vector<int>.Count;
        for (int i = 1; i < height - 1; i++)
        {
            int row = i * width;
            Unsafe.Add(ref mask, row) = Wshed;
            Unsafe.Add(ref mask, row + width - 1) = Wshed;
            for (int j = 1; j < width - 1; j++)
            {
                if (vector && j + vc <= width - 1 && Quiet(ref Unsafe.Add(ref mask, row + j), width))
                {
                    j += vc - 1;
                    continue;
                }

                int pix = row + j;
                ref int m = ref Unsafe.Add(ref mask, pix);
                if (m < 0)
                {
                    m = 0;
                }

                if (m != 0)
                {
                    continue;
                }

                int left = Unsafe.Add(ref m, -1), right = Unsafe.Add(ref m, 1);
                int top = Unsafe.Add(ref m, -width), bottom = Unsafe.Add(ref m, width);
                if (left > 0 || right > 0 || top > 0 || bottom > 0)
                {
                    ref byte p = ref Unsafe.Add(ref img, 3 * (nint)pix);
                    int idx = 256;
                    if (left > 0)
                    {
                        idx = Diff(ref p, -3);
                    }

                    if (right > 0)
                    {
                        idx = Math.Min(idx, Diff(ref p, 3));
                    }

                    if (top > 0)
                    {
                        idx = Math.Min(idx, Diff(ref p, -istep));
                    }

                    if (bottom > 0)
                    {
                        idx = Math.Min(idx, Diff(ref p, istep));
                    }

                    Push(ref link, ref ends, idx, pix + 1);
                    m = InQueue;
                    queued = true;
                }
            }
        }

        return queued;
    }

    /// <summary>
    /// <see langword="true"/> when none of the <c>Vector&lt;int&gt;.Count</c> markers at <paramref name="m"/>
    /// is negative or is an unknown pixel with a positive 4-neighbour. The seed phase writes only 0
    /// and IN_QUEUE, never a positive value, so this test on the current markers equals the
    /// sequential test.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool Quiet(ref int m, int width)
    {
        Vector<int> cur = Vector.LoadUnsafe(ref m);
        Vector<int> neighbour = Vector.Max(
            Vector.Max(Vector.LoadUnsafe(ref Unsafe.Add(ref m, -1)), Vector.LoadUnsafe(ref Unsafe.Add(ref m, 1))),
            Vector.Max(Vector.LoadUnsafe(ref Unsafe.Add(ref m, -width)), Vector.LoadUnsafe(ref Unsafe.Add(ref m, width))));
        Vector<int> active = Vector.BitwiseOr(
            Vector.LessThan(cur, Vector<int>.Zero),
            Vector.BitwiseAnd(Vector.Equals(cur, Vector<int>.Zero), Vector.GreaterThan(neighbour, Vector<int>.Zero)));
        return active == Vector<int>.Zero;
    }

    /// <summary>
    /// Pops the lowest non-empty queue until all queues are empty. Few live values keep the loop in
    /// registers: the neighbours are read again after the label is stored, as in OpenCV.
    /// </summary>
    private static void Flood(ref byte img, ref int mask, ref int link, ref int ends, nint width)
    {
        nint active = 0;
        while (Unsafe.Add(ref ends, 2 * active) == 0)
        {
            active++;
        }

        for (; ;)
        {
            ref int head = ref Unsafe.Add(ref ends, 2 * active);
            nint node = head;
            int next = Unsafe.Add(ref link, node);
            head = next;
            if (next == 0)
            {
                Unsafe.Add(ref ends, (2 * active) + 1) = 0;
            }

            ref int m = ref Unsafe.Add(ref mask, node - 1);
            int lab = Merge(0, Unsafe.Add(ref m, -1));
            lab = Merge(lab, Unsafe.Add(ref m, 1));
            lab = Merge(lab, Unsafe.Add(ref m, -width));
            lab = Merge(lab, Unsafe.Add(ref m, width));
            m = lab;
            if (lab != Wshed)
            {
                ref byte p = ref Unsafe.Add(ref img, 3 * (node - 1));
                active = Spread(ref link, ref ends, ref m, ref p, -1, -3, node, active);
                active = Spread(ref link, ref ends, ref m, ref p, 1, 3, node, active);
                active = Spread(ref link, ref ends, ref m, ref p, -width, -3 * width, node, active);
                active = Spread(ref link, ref ends, ref m, ref p, width, 3 * width, node, active);
            }

            if (Unsafe.Add(ref ends, 2 * active) == 0)
            {
                do
                {
                    if (++active == QueueCount)
                    {
                        return;
                    }
                }
                while (Unsafe.Add(ref ends, 2 * active) == 0);
            }
        }
    }

    /// <summary>Queues the neighbour at the given offsets when it is unknown; returns the new active level.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static nint Spread(ref int link, ref int ends, ref int m, ref byte p, nint mdelta, nint idelta, nint node, nint active)
    {
        ref int n = ref Unsafe.Add(ref m, mdelta);
        if (n != 0)
        {
            return active;
        }

        nint t = Diff(ref p, idelta);
        Push(ref link, ref ends, t, (int)(node + mdelta));
        n = InQueue;
        return Min(t, active);
    }

    /// <summary>Appends <paramref name="node"/> to queue <paramref name="idx"/>. An empty queue has tail 0; <c>link[0]</c> is a scratch slot.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Push(ref int link, ref int ends, nint idx, int node)
    {
        Unsafe.Add(ref link, node) = 0;
        ref int tail = ref Unsafe.Add(ref ends, (2 * idx) + 1);
        int last = tail;
        Unsafe.Add(ref link, last) = node;
        if (last == 0)
        {
            Unsafe.Add(ref ends, 2 * idx) = node;
        }

        tail = node;
    }

    /// <summary>The label rule of the flooding step: the first positive label, or WSHED when two positive labels differ.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Merge(int lab, int t)
    {
        if (t > 0)
        {
            if (lab == 0)
            {
                return t;
            }

            if (t != lab)
            {
                return Wshed;
            }
        }

        return lab;
    }

    /// <summary>Maximum absolute channel difference between the pixel at <paramref name="a"/> and the pixel <paramref name="offset"/> bytes away.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Diff(ref byte a, nint offset)
    {
        ref byte b = ref Unsafe.Add(ref a, offset);
        int d = Abs(a - b);
        d = Max(d, Abs(Unsafe.Add(ref a, 1) - Unsafe.Add(ref b, 1)));
        return Max(d, Abs(Unsafe.Add(ref a, 2) - Unsafe.Add(ref b, 2)));
    }

    /// <summary>Branch-free maximum of two values in <c>[0, 255]</c>; the JIT emits <see cref="Math.Max(int, int)"/> here as an unpredictable branch.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Max(int a, int b)
    {
        int t = a - b;
        return a - (t & (t >> 31));
    }

    /// <summary>Branch-free minimum of two queue levels.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static nint Min(nint a, nint b)
    {
        nint t = a - b;
        return b + (t & (t >> 63));
    }

    /// <summary>Branch-free absolute value; <see cref="Math.Abs(int)"/> tests for <c>int.MinValue</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Abs(int x)
    {
        int s = x >> 31;
        return (x ^ s) - s;
    }
}
