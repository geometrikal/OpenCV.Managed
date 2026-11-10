namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Byte rows of <c>copyMakeBorder</c> with AVX-512 VBMI: each border is one 64-byte permutation
/// of the first or last 64 source bytes, blended with the constant colour. Every permuted byte that
/// is not a border byte is the interior byte at that position, so the stores and the interior copy
/// can overlap.
/// </summary>
internal readonly struct BorderPermutation
{
    private const int Width = 64;

    private readonly Vector512<byte> _leftIndex;
    private readonly Vector512<byte> _leftConstant;
    private readonly Vector512<byte> _leftMask;
    private readonly Vector512<byte> _rightIndex;
    private readonly Vector512<byte> _rightConstant;
    private readonly Vector512<byte> _rightMask;
    private readonly byte[] _constantRow;
    private readonly int _leftCount;

    private BorderPermutation(
        Vector512<byte> leftIndex,
        Vector512<byte> leftConstant,
        Vector512<byte> leftMask,
        Vector512<byte> rightIndex,
        Vector512<byte> rightConstant,
        Vector512<byte> rightMask,
        byte[] constantRow,
        int leftCount)
    {
        _leftCount = leftCount;
        _leftIndex = leftIndex;
        _leftConstant = leftConstant;
        _leftMask = leftMask;
        _rightIndex = rightIndex;
        _rightConstant = rightConstant;
        _rightMask = rightMask;
        _constantRow = constantRow;
    }

    /// <summary>
    /// Builds the permutations. <paramref name="map"/> holds the source element of every left
    /// (first <paramref name="leftCount"/>) and right border element; <c>−1 − c</c> selects
    /// constant channel <c>c</c>. Fails when VBMI is missing or a border does not fit 64 bytes.
    /// </summary>
    public static bool TryCreate(int[] map, int leftCount, int rightCount, int srcLength, ReadOnlySpan<byte> constant, out BorderPermutation permutation)
    {
        permutation = default;
        int dstLength = leftCount + srcLength + rightCount;
        if (!Avx512Vbmi.IsSupported || srcLength < Width || leftCount > Width || rightCount > Width)
        {
            return false;
        }

        Span<byte> li = stackalloc byte[Width], lc = stackalloc byte[Width], lm = stackalloc byte[Width];
        Span<byte> ri = stackalloc byte[Width], rc = stackalloc byte[Width], rm = stackalloc byte[Width];
        int rightBase = srcLength - Width;
        for (int q = 0; q < Width; q++)
        {
            // Left store at destination 0, window at source 0.
            int e = q < leftCount ? map[q] : q - leftCount;
            if (e < 0)
            {
                lc[q] = constant[-1 - e];
                lm[q] = 0xFF;
            }
            else if (e < Width)
            {
                li[q] = (byte)e;
            }
            else
            {
                return false;
            }

            // Right store at destination dstLength − 64, window at source srcLength − 64.
            int dq = dstLength - Width + q;
            int r = dq < leftCount + srcLength ? dq - leftCount : map[leftCount + (dq - leftCount - srcLength)];
            if (r < 0)
            {
                rc[q] = constant[-1 - r];
                rm[q] = 0xFF;
            }
            else if (r - rightBase is >= 0 and < Width)
            {
                ri[q] = (byte)(r - rightBase);
            }
            else
            {
                return false;
            }
        }

        var row = new byte[dstLength];
        for (int i = 0; i < dstLength; i++)
        {
            row[i] = constant[i % constant.Length];
        }

        permutation = new BorderPermutation(
            Vector512.Create<byte>(li), Vector512.Create<byte>(lc), Vector512.Create<byte>(lm),
            Vector512.Create<byte>(ri), Vector512.Create<byte>(rc), Vector512.Create<byte>(rm),
            row,
            leftCount);
        return true;
    }

    /// <summary>Writes every destination row; constant border rows come from the constant row.</summary>
    public void Apply(Mat<byte> source, Mat<byte> dst, int top, BorderType borderType)
    {
        int srcH = source.Height;
        for (int y = 0; y < dst.Height; y++)
        {
            Span<byte> d = dst.GetRowSpan(y);
            int sy = Borders.BorderInterpolate(y - top, srcH, borderType);
            if (sy < 0)
            {
                _constantRow.CopyTo(d);
                continue;
            }

            ReadOnlySpan<byte> s = source.GetReadOnlyRowSpan(sy);
            ref byte sr = ref MemoryMarshal.GetReference(s);
            ref byte dr = ref MemoryMarshal.GetReference(d);
            Vector512<byte> head = Vector512.LoadUnsafe(ref sr);
            Vector512<byte> tail = Vector512.LoadUnsafe(ref sr, (nuint)(s.Length - Width));
            Vector512.ConditionalSelect(_leftMask, _leftConstant, Avx512Vbmi.PermuteVar64x8(head, _leftIndex)).StoreUnsafe(ref dr);
            s.CopyTo(d[_leftCount..]);
            Vector512.ConditionalSelect(_rightMask, _rightConstant, Avx512Vbmi.PermuteVar64x8(tail, _rightIndex))
                .StoreUnsafe(ref dr, (nuint)(d.Length - Width));
        }
    }
}
