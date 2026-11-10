// This implementation is ported from the OpenCV separable resize.
//
// Original source:
//   modules/imgproc/src/resize.cpp (HResizeLinear, HResizeCubic)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Horizontal linear and bicubic resize of <c>float</c> rows with AVX2 permutes. A group of 8
/// destination elements reads tap <c>j</c> from two 8-element windows (one per half of 4
/// elements) that start <c>j·cn</c> after the half's lowest tap. Each element is the left-to-right
/// sum <c>S[t0]·a0 + S[t1]·a1 (+ S[t2]·a2 + S[t3]·a3)</c>, as the scalar loop.
/// </summary>
internal sealed class ResizeLinearGather : IDisposable
{
    private ResizeLinearGather(int groups, int k)
    {
        Groups = groups;
        K = k;
        Bases = ArrayPool<int>.Shared.Rent(2 * groups);
        Perms = ArrayPool<Vector256<int>>.Shared.Rent(groups);
        Weights = ArrayPool<Vector256<float>>.Shared.Rent(groups * k);
    }

    public int Groups { get; }

    public int K { get; }

    /// <summary>Window start of each half for tap 0; -1 marks a group for the scalar loop.</summary>
    public int[] Bases { get; }

    public Vector256<int>[] Perms { get; }

    public Vector256<float>[] Weights { get; }

    /// <summary>
    /// Plans the groups of <c>[0, xmax·cn)</c> (linear) or <c>[xmin·cn, xmax·cn)</c> (bicubic);
    /// returns null when AVX2 is not available or no group fits.
    /// </summary>
    public static ResizeLinearGather? Create(ResizeCoefficients c, int srcWidth, int cn)
    {
        int k = c.Ksize;
        if (!Avx2.IsSupported || k > 4)
        {
            return null;
        }

        int xmaxE = c.XMax * cn;
        int xminE = k == 2 ? 0 : c.XMin * cn;
        int off = k == 2 ? 0 : -cn;
        int swidth = srcWidth * cn;
        int groups = xmaxE / 8;
        if (groups == 0)
        {
            return null;
        }

        var plan = new ResizeLinearGather(groups, k);
        Span<int> perm = stackalloc int[8];
        Span<float> w = stackalloc float[8];
        int usable = 0;
        for (int g = 0; g < groups; g++)
        {
            int e0 = g * 8;
            bool ok = e0 >= xminE;
            for (int half = 0; half < 2 && ok; half++)
            {
                int b = int.MaxValue, last = int.MinValue;
                for (int l = 0; l < 4; l++)
                {
                    b = Math.Min(b, c.XOffsets[e0 + (half * 4) + l] + off);
                    last = Math.Max(last, c.XOffsets[e0 + (half * 4) + l] + off);
                }

                if (b < 0 || last - b > 7 || b + ((k - 1) * cn) + 8 > swidth)
                {
                    ok = false;
                    break;
                }

                plan.Bases[(2 * g) + half] = b;
                for (int l = 0; l < 4; l++)
                {
                    perm[(half * 4) + l] = c.XOffsets[e0 + (half * 4) + l] + off - b;
                }
            }

            if (!ok)
            {
                plan.Bases[2 * g] = -1;
                continue;
            }

            usable++;
            plan.Perms[g] = Vector256.Create((ReadOnlySpan<int>)perm);
            for (int j = 0; j < k; j++)
            {
                for (int l = 0; l < 8; l++)
                {
                    w[l] = c.Alpha[((e0 + l) * k) + j];
                }

                plan.Weights[(g * k) + j] = Vector256.Create((ReadOnlySpan<float>)w);
            }
        }

        if (usable == 0)
        {
            plan.Dispose();
            return null;
        }

        return plan;
    }

    public void Dispose()
    {
        ArrayPool<int>.Shared.Return(Bases);
        ArrayPool<Vector256<int>>.Shared.Return(Perms);
        ArrayPool<Vector256<float>>.Shared.Return(Weights);
    }

    /// <summary>Writes group <paramref name="g"/>; returns false for a group the caller computes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGroup(ref float src, ref float dst, int g, int cn)
    {
        ref int bases = ref MemoryMarshal.GetArrayDataReference(Bases);
        int ba = Unsafe.Add(ref bases, 2 * g);
        if (ba < 0)
        {
            return false;
        }

        int bb = Unsafe.Add(ref bases, (2 * g) + 1);
        Vector256<int> perm = Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Perms), g);
        ref Vector256<float> w = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Weights), g * K);
        Vector256<float> acc = Tap(ref src, ba, bb, perm) * w;
        acc += Tap(ref src, ba + cn, bb + cn, perm) * Unsafe.Add(ref w, 1);
        if (K == 4)
        {
            acc += Tap(ref src, ba + (2 * cn), bb + (2 * cn), perm) * Unsafe.Add(ref w, 2);
            acc += Tap(ref src, ba + (3 * cn), bb + (3 * cn), perm) * Unsafe.Add(ref w, 3);
        }

        acc.StoreUnsafe(ref dst, (nuint)(g * 8));
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Tap(ref float src, int ba, int bb, Vector256<int> perm)
        => Avx.Blend(
            Avx2.PermuteVar8x32(Vector256.LoadUnsafe(ref src, (nuint)ba), perm),
            Avx2.PermuteVar8x32(Vector256.LoadUnsafe(ref src, (nuint)bb), perm),
            0b1111_0000);
}
