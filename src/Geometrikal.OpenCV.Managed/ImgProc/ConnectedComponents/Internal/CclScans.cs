// This implementation is ported from the OpenCV connected components labeling.
//
// Original source:
//   modules/imgproc/src/connectedcomponents.cpp
//     (findRoot, setRoot, set_union, flattenL, flattenLParallel, LabelingWu,
//      LabelingWuParallel, LabelingBolelli, LabelingBolelliParallel, LabelingGrana,
//      LabelingBolelli4C, mergeLabels*)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/connectedcomponents.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>A single-channel byte image; a nonzero element is foreground.</summary>
internal readonly record struct CclImage(byte[] Data, int Offset, int Stride, int Width, int Height)
{
    public ref byte Row(int y) => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + (y * Stride));
}

/// <summary>An <c>int</c> label plane with a row stride in elements.</summary>
internal readonly record struct CclStore(int[] Data, int Offset, int Stride)
{
    public ref int Row(int y) => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + (y * Stride));
}

/// <summary>
/// First scan of the labeling: provisional labels, union-find with minimum roots, stripe merge
/// and flattening. Provisional labels are created in scan order, so the final label of a
/// component is the rank of its first pixel (pixel scans) or first 2x2 block (block scan).
/// </summary>
internal static class CclScans
{
    /// <summary>
    /// SAUF pixel scan of rows <c>[y0, y1)</c>; row <paramref name="y0"/> is treated as the
    /// image top. Returns the next unused provisional label.
    /// </summary>
    public static int Pixel(CclImage img, CclStore store, int[] parent, int y0, int y1, int label, int limit, bool eight)
    {
        int w = img.Width;
        int len = w + 3;
        byte[] rows = ArrayPool<byte>.Shared.Rent(2 * len);
        try
        {
            rows.AsSpan(0, len).Clear();
            ref byte buf = ref MemoryMarshal.GetArrayDataReference(rows);
            ref int p = ref MemoryMarshal.GetArrayDataReference(parent);
            int up = 0;
            int cur = len;
            for (int r = y0; r < y1; r++)
            {
                LoadRow(ref img.Row(r), ref Unsafe.Add(ref buf, cur), w);
                ref int lab = ref store.Row(r);
                ref int plab = ref r > y0 ? ref Unsafe.Subtract(ref lab, store.Stride) : ref lab;
                label = eight
                    ? Pixel8Row(ref Unsafe.Add(ref buf, up), ref Unsafe.Add(ref buf, cur), ref lab, ref plab, ref p, w, label, limit)
                    : Pixel4Row(ref Unsafe.Add(ref buf, up), ref Unsafe.Add(ref buf, cur), ref lab, ref plab, ref p, w, label, limit);
                (up, cur) = (cur, up);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rows);
        }

        return label;
    }

    /// <summary>
    /// 2x2 block scan (8-connectivity) of the row pairs in <c>[y0, y1)</c>; <paramref name="y0"/>
    /// is even and is treated as the image top. The provisional label of block
    /// <c>(y / 2, x / 2)</c> is stored in <paramref name="blocks"/>, 0 for an empty block.
    /// Returns the next unused provisional label.
    /// </summary>
    public static int Block8(CclImage img, int[] blocks, int[] parent, int y0, int y1, int label, int limit)
    {
        int w = img.Width;
        int h = img.Height;
        int bw = (w + 1) >> 1;
        int len = w + 3;
        byte[] rows = ArrayPool<byte>.Shared.Rent(3 * len);
        try
        {
            rows.AsSpan(0, len).Clear();
            ref byte buf = ref MemoryMarshal.GetArrayDataReference(rows);
            ref int p = ref MemoryMarshal.GetArrayDataReference(parent);
            ref int bl = ref MemoryMarshal.GetArrayDataReference(blocks);
            int above = 0;
            int top = len;
            int bottom = 2 * len;
            for (int r = y0; r < y1; r += 2)
            {
                LoadRow(ref img.Row(r), ref Unsafe.Add(ref buf, top), w);
                if (r + 1 < h)
                {
                    LoadRow(ref img.Row(r + 1), ref Unsafe.Add(ref buf, bottom), w);
                }
                else
                {
                    rows.AsSpan(bottom, len).Clear();
                }

                ref int cur = ref Unsafe.Add(ref bl, (r >> 1) * bw);
                ref int prev = ref r > y0 ? ref Unsafe.Subtract(ref cur, bw) : ref cur;
                label = Block8Row(
                    ref Unsafe.Add(ref buf, above), ref Unsafe.Add(ref buf, top), ref Unsafe.Add(ref buf, bottom),
                    ref cur, ref prev, ref p, bw, label, limit);
                (above, bottom) = (bottom, above);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rows);
        }

        return label;
    }

    /// <summary>
    /// Unites the pixel labels of row <paramref name="y"/> with row <c>y - 1</c>. Returns the
    /// number of provisional labels that the stripe scan created on row <paramref name="y"/> and
    /// that a sequential scan does not create, because the pixel connects to row <c>y - 1</c>.
    /// </summary>
    public static int MergePixel(CclImage img, CclStore store, int[] parent, int y, bool eight)
    {
        int w = img.Width;
        ref byte cur = ref img.Row(y);
        ref byte up = ref img.Row(y - 1);
        ref int lab = ref store.Row(y);
        ref int plab = ref store.Row(y - 1);
        ref int p = ref MemoryMarshal.GetArrayDataReference(parent);
        int extra = 0;
        for (int c = 0; c < w; c++)
        {
            if (Unsafe.Add(ref cur, c) == 0)
            {
                continue;
            }

            int x = Unsafe.Add(ref lab, c);
            bool connected = false;
            if (Unsafe.Add(ref up, c) != 0)
            {
                Union(ref p, x, Unsafe.Add(ref plab, c));
                connected = true;
            }
            else if (eight)
            {
                if (c > 0 && Unsafe.Add(ref up, c - 1) != 0)
                {
                    x = Union(ref p, x, Unsafe.Add(ref plab, c - 1));
                    connected = true;
                }

                if (c + 1 < w && Unsafe.Add(ref up, c + 1) != 0)
                {
                    Union(ref p, x, Unsafe.Add(ref plab, c + 1));
                    connected = true;
                }
            }

            if (connected && (c == 0 || Unsafe.Add(ref cur, c - 1) == 0))
            {
                extra++;
            }
        }

        return extra;
    }

    /// <summary>
    /// Unites the block labels of the row pair at even row <paramref name="y"/> with the pair
    /// above. Returns the number of provisional labels that the stripe scan created in this pair
    /// and that a sequential scan does not create, because the block connects to the pair above.
    /// </summary>
    public static int MergeBlock(CclImage img, int[] blocks, int[] parent, int y)
    {
        int w = img.Width;
        int bw = (w + 1) >> 1;
        bool below = y + 1 < img.Height;
        ref byte a = ref img.Row(y - 1);
        ref byte b = ref img.Row(y);
        ref byte d = ref below ? ref img.Row(y + 1) : ref b;
        ref int cur = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(blocks), (y >> 1) * bw);
        ref int prev = ref Unsafe.Subtract(ref cur, bw);
        ref int p = ref MemoryMarshal.GetArrayDataReference(parent);
        int extra = 0;
        for (int bc = 0; bc < bw; bc++)
        {
            int x = Unsafe.Add(ref cur, bc);
            if (x == 0)
            {
                continue;
            }

            int c = 2 * bc;
            bool o = Unsafe.Add(ref b, c) != 0;
            bool pp = c + 1 < w && Unsafe.Add(ref b, c + 1) != 0;
            bool i = Unsafe.Add(ref a, c) != 0;
            bool j = c + 1 < w && Unsafe.Add(ref a, c + 1) != 0;
            bool connected = false;
            if ((i || j) && (o || pp))
            {
                x = Union(ref p, x, Unsafe.Add(ref prev, bc));
                connected = true;
            }

            if (o && c > 0 && Unsafe.Add(ref a, c - 1) != 0)
            {
                x = Union(ref p, x, Unsafe.Add(ref prev, bc - 1));
                connected = true;
            }

            if (pp && c + 2 < w && Unsafe.Add(ref a, c + 2) != 0)
            {
                Union(ref p, x, Unsafe.Add(ref prev, bc + 1));
                connected = true;
            }

            if (connected)
            {
                bool s = below && Unsafe.Add(ref d, c) != 0;
                bool left = c > 0 && (Unsafe.Add(ref b, c - 1) != 0 || (below && Unsafe.Add(ref d, c - 1) != 0));
                if (!(left && (o || s)))
                {
                    extra++;
                }
            }
        }

        return extra;
    }

    /// <summary>
    /// Replaces <c>parent[i]</c> for <c>i</c> in <c>[start, end)</c> with the final label, as
    /// OpenCV <c>flattenLParallel</c>. Returns the next unused final label.
    /// </summary>
    public static int Flatten(int[] parent, int start, int end, int next)
    {
        ref int p = ref MemoryMarshal.GetArrayDataReference(parent);
        for (int i = start; i < end; i++)
        {
            int pi = Unsafe.Add(ref p, i);
            Unsafe.Add(ref p, i) = pi < i ? Unsafe.Add(ref p, pi) : next++;
        }

        return next;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FindRoot(ref int p, int i)
    {
        int root = i;
        int next;
        while ((next = Unsafe.Add(ref p, root)) < root)
        {
            root = next;
        }

        return root;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SetRoot(ref int p, int i, int root)
    {
        int next;
        while ((next = Unsafe.Add(ref p, i)) < i)
        {
            Unsafe.Add(ref p, i) = root;
            i = next;
        }

        Unsafe.Add(ref p, i) = root;
    }

    /// <summary>Unites the sets of <paramref name="i"/> and <paramref name="j"/>; the root is the smaller root.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Union(ref int p, int i, int j)
    {
        if (i == j)
        {
            return i;
        }

        int root = FindRoot(ref p, i);
        int rj = FindRoot(ref p, j);
        if (root > rj)
        {
            root = rj;
        }

        SetRoot(ref p, j, root);
        SetRoot(ref p, i, root);
        return root;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int NewLabel(ref int p, ref int label, int limit)
    {
        int x = label;
        if (x == limit)
        {
            ThrowOverflow();
        }

        Unsafe.Add(ref p, x) = x;
        label = x + 1;
        return x;
    }

    /// <summary>
    /// Writes a row as 0/1 bytes to <c>dst[1..w]</c>, with zero at <c>dst[0]</c>, <c>dst[w + 1]</c>
    /// and <c>dst[w + 2]</c>.
    /// </summary>
    private static void LoadRow(ref byte src, ref byte dst, int w)
    {
        dst = 0;
        ref byte d = ref Unsafe.Add(ref dst, 1);
        int c = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<byte>.Count;
            for (; c <= w - vc; c += vc)
            {
                Vector.Min(Vector.LoadUnsafe(ref src, (nuint)c), Vector<byte>.One).StoreUnsafe(ref d, (nuint)c);
            }
        }

        for (; c < w; c++)
        {
            Unsafe.Add(ref d, c) = Unsafe.Add(ref src, c) != 0 ? (byte)1 : (byte)0;
        }

        Unsafe.Add(ref dst, w + 1) = 0;
        Unsafe.Add(ref dst, w + 2) = 0;
    }

    /// <summary>Number of leading zero bytes in <c>v</c>, at most 32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int LeadingZeroBytes(Vector256<byte> v)
    {
        uint nonzero = ~Vector256.Equals(v, Vector256<byte>.Zero).ExtractMostSignificantBits();
        return nonzero == 0 ? 32 : BitOperations.TrailingZeroCount(nonzero);
    }

    /// <summary>
    /// Writes label 0 for the background run that starts at pixel <paramref name="c"/> and
    /// returns its length (at least 1). With 32 pixels left in the row, the 32 labels from
    /// <paramref name="c"/> are cleared; labels after the run are written again later.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SkipBackground(ref byte x, int i, ref int lab, int c, int w)
    {
        if (!Vector256.IsHardwareAccelerated || c + 32 > w)
        {
            Unsafe.Add(ref lab, c) = 0;
            return 1;
        }

        ref int dst = ref Unsafe.Add(ref lab, c);
        Vector256<int>.Zero.StoreUnsafe(ref dst);
        Vector256<int>.Zero.StoreUnsafe(ref dst, 8);
        Vector256<int>.Zero.StoreUnsafe(ref dst, 16);
        Vector256<int>.Zero.StoreUnsafe(ref dst, 24);
        return LeadingZeroBytes(Vector256.LoadUnsafe(ref x, (nuint)i));
    }

    private static int Pixel8Row(ref byte u, ref byte x, ref int lab, ref int plab, ref int p, int w, int label, int limit)
    {
        // Label of pixel c - 1; valid when that pixel is foreground.
        int left = 0;
        for (int c = 0; c < w; c++)
        {
            int i = c + 1;
            if (Unsafe.Add(ref x, i) == 0)
            {
                c += SkipBackground(ref x, i, ref lab, c, w) - 1;
                continue;
            }

            int value;
            if (Unsafe.Add(ref u, i) != 0)
            {
                // Eight pixels and the eight above them are foreground: every pixel copies q.
                if (Vector256.IsHardwareAccelerated && c + 8 <= w &&
                    Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref x, i)) == 0x0101010101010101ul &&
                    Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref u, i)) == 0x0101010101010101ul)
                {
                    Vector256.LoadUnsafe(ref plab, (nuint)c).StoreUnsafe(ref lab, (nuint)c);
                    c += 7;
                    left = Unsafe.Add(ref plab, c);
                    continue;
                }

                value = Unsafe.Add(ref plab, c);
            }
            else if (Unsafe.Add(ref u, i + 1) != 0)
            {
                if (Unsafe.Add(ref u, i - 1) != 0)
                {
                    value = Union(ref p, Unsafe.Add(ref plab, c - 1), Unsafe.Add(ref plab, c + 1));
                }
                else if (Unsafe.Add(ref x, i - 1) != 0)
                {
                    value = Union(ref p, left, Unsafe.Add(ref plab, c + 1));
                }
                else
                {
                    value = Unsafe.Add(ref plab, c + 1);
                }
            }
            else if (Unsafe.Add(ref u, i - 1) != 0)
            {
                value = Unsafe.Add(ref plab, c - 1);
            }
            else if (Unsafe.Add(ref x, i - 1) != 0)
            {
                value = left;
            }
            else
            {
                value = NewLabel(ref p, ref label, limit);
            }

            Unsafe.Add(ref lab, c) = value;
            left = value;
        }

        return label;
    }

    private static int Pixel4Row(ref byte u, ref byte x, ref int lab, ref int plab, ref int p, int w, int label, int limit)
    {
        // Label of pixel c - 1; valid when that pixel is foreground.
        int left = 0;
        for (int c = 0; c < w; c++)
        {
            int i = c + 1;
            if (Unsafe.Add(ref x, i) == 0)
            {
                c += SkipBackground(ref x, i, ref lab, c, w) - 1;
                continue;
            }

            int value;
            if (Unsafe.Add(ref u, i) != 0)
            {
                int q = Unsafe.Add(ref plab, c);
                if (Unsafe.Add(ref x, i - 1) == 0)
                {
                    value = q;
                }
                else if (Unsafe.Add(ref u, i - 1) != 0)
                {
                    // s joined the pixel above it, which is in one set with q.
                    value = left;
                    if (Vector256.IsHardwareAccelerated && c + 8 <= w &&
                        Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref x, i)) == 0x0101010101010101ul &&
                        Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref u, i)) == 0x0101010101010101ul)
                    {
                        Vector256.Create(left).StoreUnsafe(ref lab, (nuint)c);
                        c += 7;
                        continue;
                    }
                }
                else
                {
                    value = Union(ref p, left, q);
                }
            }
            else if (Unsafe.Add(ref x, i - 1) != 0)
            {
                value = left;
            }
            else
            {
                value = NewLabel(ref p, ref label, limit);
            }

            Unsafe.Add(ref lab, c) = value;
            left = value;
        }

        return label;
    }

    /// <summary>
    /// One row pair of the block scan. Pixel names follow <c>LabelingBolelli</c>: the row above is
    /// <c>g h | i j | k l</c>, the pair is <c>m n | o p</c> over <c>q r | s t</c>, and the current
    /// block X is <c>{o, p, s, t}</c>. A union is skipped when an earlier step already connected
    /// the two blocks.
    /// </summary>
    private static int Block8Row(
        ref byte a, ref byte b0, ref byte b1, ref int cur, ref int prev, ref int p, int bw, int label, int limit)
    {
        for (int bc = 0; bc < bw; bc++)
        {
            int i = (2 * bc) + 1;
            int o = Unsafe.Add(ref b0, i);
            int pp = Unsafe.Add(ref b0, i + 1);
            int s = Unsafe.Add(ref b1, i);
            int t = Unsafe.Add(ref b1, i + 1);
            if ((o | pp | s | t) == 0)
            {
                if (Vector256.IsHardwareAccelerated && bc + 16 <= bw)
                {
                    // Labels after the empty run are written again later.
                    ref int dst = ref Unsafe.Add(ref cur, bc);
                    Vector256<int>.Zero.StoreUnsafe(ref dst);
                    Vector256<int>.Zero.StoreUnsafe(ref dst, 8);
                    bc += (LeadingZeroBytes(Vector256.LoadUnsafe(ref b0, (nuint)i) | Vector256.LoadUnsafe(ref b1, (nuint)i)) >> 1) - 1;
                }
                else
                {
                    Unsafe.Add(ref cur, bc) = 0;
                }

                continue;
            }

            // h, i, j, k and n, o, p all foreground: X joins Q, which already holds P, R and S.
            if (Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref a, i - 1)) == 0x01010101u &&
                (Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref b0, i - 1)) & 0x00FFFFFFu) == 0x00010101u)
            {
                Unsafe.Add(ref cur, bc) = Unsafe.Add(ref prev, bc);
                continue;
            }

            int hh = Unsafe.Add(ref a, i - 1);
            int ii = Unsafe.Add(ref a, i);
            int jj = Unsafe.Add(ref a, i + 1);
            int kk = Unsafe.Add(ref a, i + 2);
            int n = Unsafe.Add(ref b0, i - 1);
            int rr = Unsafe.Add(ref b1, i - 1);
            bool toS = (n | rr) != 0 && (o | s) != 0;
            int x;
            if ((ii | jj) != 0 && (o | pp) != 0)
            {
                x = Unsafe.Add(ref prev, bc);
                if (hh != 0 && o != 0 && ii == 0)
                {
                    x = Union(ref p, Unsafe.Add(ref prev, bc - 1), x);
                }

                if (kk != 0 && pp != 0 && jj == 0)
                {
                    x = Union(ref p, x, Unsafe.Add(ref prev, bc + 1));
                }

                if (toS && (n == 0 || ii == 0))
                {
                    x = Union(ref p, x, Unsafe.Add(ref cur, bc - 1));
                }
            }
            else
            {
                x = 0;
                bool toP = hh != 0 && o != 0;
                if (toP)
                {
                    x = Unsafe.Add(ref prev, bc - 1);
                }

                if (kk != 0 && pp != 0)
                {
                    x = x == 0 ? Unsafe.Add(ref prev, bc + 1) : Union(ref p, x, Unsafe.Add(ref prev, bc + 1));
                }

                if (toS)
                {
                    if (x == 0)
                    {
                        x = Unsafe.Add(ref cur, bc - 1);
                    }
                    else if (!(toP && n != 0))
                    {
                        x = Union(ref p, x, Unsafe.Add(ref cur, bc - 1));
                    }
                }

                if (x == 0)
                {
                    x = NewLabel(ref p, ref label, limit);
                }
            }

            Unsafe.Add(ref cur, bc) = x;
        }

        return label;
    }

    /// <summary>Throws the OpenCV 5.0 <c>ushort</c> label overflow error.</summary>
    public static void ThrowOverflow()
        => throw new OverflowException(
            "Total number of labels overflowed the label type. Use int labels instead of ushort labels.");
}
