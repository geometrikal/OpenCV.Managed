namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Vector split and merge of 3- and 4-channel rows into single-channel planes. Each method
/// processes whole 16-byte blocks from the start of the row and returns the number of pixels
/// done.
/// </summary>
internal static class ChannelShuffle
{
    public static int Split<T>(ReadOnlySpan<T> src, Span<T> d0, Span<T> d1, Span<T> d2, Span<T> d3, int cn)
        where T : unmanaged
    {
        int e = Unsafe.SizeOf<T>();
        if (!ColorVectors.IsSupported || e is not (1 or 2 or 4))
        {
            return 0;
        }

        int n = 16 / e;
        ref T s = ref MemoryMarshal.GetReference(src);
        ref byte p0 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(d0));
        ref byte p1 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(d1));
        ref byte p2 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(d2));
        ref byte p3 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(d3));
        int x = 0;
        for (; x <= d0.Length - n; x += n)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * cn), cn, out var v0, out var v1, out var v2, out var v3);
            nuint o = (nuint)(x * e);
            v0.StoreUnsafe(ref p0, o);
            v1.StoreUnsafe(ref p1, o);
            v2.StoreUnsafe(ref p2, o);
            if (cn == 4)
            {
                v3.StoreUnsafe(ref p3, o);
            }
        }

        return x;
    }

    public static int Merge<T>(ReadOnlySpan<T> s0, ReadOnlySpan<T> s1, ReadOnlySpan<T> s2, ReadOnlySpan<T> s3, Span<T> dst, int cn)
        where T : unmanaged
    {
        int e = Unsafe.SizeOf<T>();
        if (!ColorVectors.IsSupported || e is not (1 or 2 or 4))
        {
            return 0;
        }

        int n = 16 / e;
        ref byte p0 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(s0));
        ref byte p1 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(s1));
        ref byte p2 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(s2));
        ref byte p3 = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(s3));
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= s0.Length - n; x += n)
        {
            nuint o = (nuint)(x * e);
            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * cn), cn,
                Vector128.LoadUnsafe(ref p0, o), Vector128.LoadUnsafe(ref p1, o), Vector128.LoadUnsafe(ref p2, o),
                cn == 4 ? Vector128.LoadUnsafe(ref p3, o) : default);
        }

        return x;
    }
}
