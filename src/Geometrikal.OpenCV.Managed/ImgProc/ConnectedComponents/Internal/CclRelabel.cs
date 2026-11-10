// This implementation is ported from the OpenCV connected components labeling.
//
// Original source:
//   modules/imgproc/src/connectedcomponents.cpp (second scans, CCStatsOp)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/connectedcomponents.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Accumulated statistics of one label. <see cref="Right"/> and <see cref="Bottom"/> are the
/// maximum coordinates; the sums are exact.
/// </summary>
internal struct CcStat
{
    public int Left;
    public int Right;
    public int Top;
    public int Bottom;
    public int Area;
    public ulong SumX;
    public ulong SumY;

    public static CcStat Empty => new() { Left = int.MaxValue, Top = int.MaxValue, Right = int.MinValue, Bottom = int.MinValue };

    public void Merge(in CcStat other)
    {
        Left = Math.Min(Left, other.Left);
        Right = Math.Max(Right, other.Right);
        Top = Math.Min(Top, other.Top);
        Bottom = Math.Max(Bottom, other.Bottom);
        Area += other.Area;
        SumX += other.SumX;
        SumY += other.SumY;
    }
}

/// <summary>
/// Second scan of the labeling: replaces provisional labels with final labels and accumulates
/// statistics per horizontal run of equal labels.
/// </summary>
internal static class CclRelabel
{
    /// <summary>Replaces the provisional pixel labels of one row with final labels, in place.</summary>
    public static void PixelRow(ref int row, ref int p, int w)
    {
        int c = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            for (; c <= w - vc; c += vc)
            {
                Vector<int> v = Vector.LoadUnsafe(ref row, (nuint)c);
                int first = v.ToScalar();
                if (v == new Vector<int>(first))
                {
                    new Vector<int>(Unsafe.Add(ref p, first)).StoreUnsafe(ref row, (nuint)c);
                    continue;
                }

                for (int k = c; k < c + vc; k++)
                {
                    Unsafe.Add(ref row, k) = Unsafe.Add(ref p, Unsafe.Add(ref row, k));
                }
            }
        }

        for (; c < w; c++)
        {
            Unsafe.Add(ref row, c) = Unsafe.Add(ref p, Unsafe.Add(ref row, c));
        }
    }

    /// <summary>Replaces the provisional labels of one block row with final labels, in place.</summary>
    public static void BlockLabels(ref int blocks, ref int p, int bw) => PixelRow(ref blocks, ref p, bw);

    /// <summary>
    /// Writes the final labels of one image row from its final block labels: the block label
    /// for a foreground pixel, 0 for a background pixel.
    /// </summary>
    public static void ExpandBlockRow(ref int blocks, ref byte src, ref int dst, int w)
    {
        int c = 0;
        if (Avx2.IsSupported)
        {
            Vector256<int> pairs = Vector256.Create(0, 0, 1, 1, 2, 2, 3, 3);
            for (; c <= w - 8; c += 8)
            {
                Vector256<int> labels = Avx2.PermuteVar8x32(
                    Vector128.LoadUnsafe(ref blocks, (nuint)(c >> 1)).ToVector256Unsafe(), pairs);
                Vector256<int> pixels = Avx2.ConvertToVector256Int32(
                    Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(ref Unsafe.Add(ref src, c))).AsByte());
                Vector256.AndNot(labels, Vector256.Equals(pixels, Vector256<int>.Zero)).StoreUnsafe(ref dst, (nuint)c);
            }
        }

        for (; c < w; c++)
        {
            Unsafe.Add(ref dst, c) = Unsafe.Add(ref src, c) != 0 ? Unsafe.Add(ref blocks, c >> 1) : 0;
        }
    }

    /// <summary>Copies final labels to a <c>ushort</c> row; every label is below 65535.</summary>
    public static void Narrow(ref int src, ref ushort dst, int w)
    {
        int c = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            for (; c <= w - (2 * vc); c += 2 * vc)
            {
                Vector<uint> lo = Vector.AsVectorUInt32(Vector.LoadUnsafe(ref src, (nuint)c));
                Vector<uint> hi = Vector.AsVectorUInt32(Vector.LoadUnsafe(ref src, (nuint)(c + vc)));
                Vector.Narrow(lo, hi).StoreUnsafe(ref dst, (nuint)c);
            }
        }

        for (; c < w; c++)
        {
            Unsafe.Add(ref dst, c) = (ushort)Unsafe.Add(ref src, c);
        }
    }

    /// <summary>
    /// Adds the runs of one final label row to the statistics. A label <c>l &gt; 0</c> updates
    /// <c>Unsafe.Add(ref stats, l - lo)</c>; label 0 updates <paramref name="background"/>.
    /// </summary>
    public static void AccumulateRow(ref int row, int w, int y, ref CcStat stats, int lo, ref CcStat background)
    {
        int start = 0;
        int label = row;
        int c = 1;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; c <= w - 8; c += 8)
            {
                Vector256<int> a = Vector256.LoadUnsafe(ref row, (nuint)c);
                Vector256<int> b = Vector256.LoadUnsafe(ref row, (nuint)(c - 1));
                uint changes = ~Vector256.Equals(a, b).ExtractMostSignificantBits() & 0xFF;
                while (changes != 0)
                {
                    int k = c + BitOperations.TrailingZeroCount(changes);
                    AddRun(label, start, k - 1, y, ref stats, lo, ref background);
                    label = Unsafe.Add(ref row, k);
                    start = k;
                    changes &= changes - 1;
                }
            }
        }

        for (; c < w; c++)
        {
            int v = Unsafe.Add(ref row, c);
            if (v != label)
            {
                AddRun(label, start, c - 1, y, ref stats, lo, ref background);
                label = v;
                start = c;
            }
        }

        AddRun(label, start, w - 1, y, ref stats, lo, ref background);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddRun(int label, int c0, int c1, int y, ref CcStat stats, int lo, ref CcStat background)
    {
        ref CcStat s = ref label == 0 ? ref background : ref Unsafe.Add(ref stats, label - lo);
        if (c0 < s.Left)
        {
            s.Left = c0;
        }

        if (c1 > s.Right)
        {
            s.Right = c1;
        }

        if (y < s.Top)
        {
            s.Top = y;
        }

        if (y > s.Bottom)
        {
            s.Bottom = y;
        }

        int length = c1 - c0 + 1;
        s.Area += length;
        s.SumX += (ulong)(((long)c0 + c1) * length / 2);
        s.SumY += (ulong)((long)y * length);
    }
}
