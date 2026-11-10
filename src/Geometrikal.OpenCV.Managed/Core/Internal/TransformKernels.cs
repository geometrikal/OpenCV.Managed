// This implementation is ported from the OpenCV matrix transforms.
//
// Original source:
//   modules/core/src/matrix_transform.cpp (flip, transpose, rotate)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Byte-level kernels for flip and transpose. A pixel is a block of <c>ps</c> bytes; the kernels
/// move whole pixels and do no arithmetic.
/// </summary>
internal static class TransformKernels
{
    /// <summary>A byte view of a matrix: first element and row step in bytes.</summary>
    internal readonly struct Plane<T>
        where T : unmanaged
    {
        public readonly T[] Array;
        public readonly int Offset;
        public readonly int Step;

        public Plane(Mat<T> m)
        {
            Array = m.DangerousGetArray();
            Offset = m.DangerousGetOffset();
            Step = m.Stride * Unsafe.SizeOf<T>();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref byte Row(int y)
            => ref Unsafe.Add(ref Unsafe.As<T, byte>(ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Array), Offset)), y * Step);
    }

    private static bool UnpackSupported => Sse2.IsSupported || AdvSimd.Arm64.IsSupported;

    // ---------------------------------------------------------------- flip

    /// <summary>Reverses the pixel order of rows <c>[y0, y1)</c>; destination row y takes source row <c>srcRow(y)</c>.</summary>
    public static void FlipRows<T>(Plane<T> src, Plane<T> dst, int width, int height, int ps, bool horizontal, bool vertical, int y0, int y1)
        where T : unmanaged
    {
        int rowBytes = width * ps;
        for (int y = y0; y < y1; y++)
        {
            int sy = vertical ? height - 1 - y : y;
            ref byte s = ref src.Row(sy);
            ref byte d = ref dst.Row(y);
            if (!horizontal)
            {
                Unsafe.CopyBlockUnaligned(ref d, ref s, (uint)rowBytes);
                continue;
            }

            ReverseRow(ref s, ref d, width, ps);
        }
    }

    private static void ReverseRow(ref byte s, ref byte d, int width, int ps)
    {
        switch (ps)
        {
            case 1:
                Unsafe.CopyBlockUnaligned(ref d, ref s, (uint)width);
                MemoryMarshal.CreateSpan(ref d, width).Reverse();
                return;
            case 2:
                Unsafe.CopyBlockUnaligned(ref d, ref s, (uint)(width * 2));
                MemoryMarshal.CreateSpan(ref Unsafe.As<byte, ushort>(ref d), width).Reverse();
                return;
            case 4:
                Unsafe.CopyBlockUnaligned(ref d, ref s, (uint)(width * 4));
                MemoryMarshal.CreateSpan(ref Unsafe.As<byte, uint>(ref d), width).Reverse();
                return;
            case 8:
                Unsafe.CopyBlockUnaligned(ref d, ref s, (uint)(width * 8));
                MemoryMarshal.CreateSpan(ref Unsafe.As<byte, ulong>(ref d), width).Reverse();
                return;
            case 3:
                Reverse3(ref s, ref d, width);
                return;
            default:
                for (int x = 0; x < width; x++)
                {
                    Unsafe.CopyBlockUnaligned(ref Unsafe.Add(ref d, x * ps), ref Unsafe.Add(ref s, (width - 1 - x) * ps), (uint)ps);
                }

                return;
        }
    }

    private static void Reverse3(ref byte s, ref byte d, int width)
    {
        int x = 0;
        if (Ssse3.IsSupported)
        {
            // Source bytes 13-3k .. 15-3k hold pixel k of five reversed pixels; byte 15 of the store is rewritten by the next step.
            Vector128<byte> mask = Vector128.Create((byte)13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0x80);
            for (; x + 6 <= width; x += 5)
            {
                Vector128<byte> v = Vector128.LoadUnsafe(ref s, (nuint)((3 * (width - x)) - 16));
                Ssse3.Shuffle(v, mask).StoreUnsafe(ref d, (nuint)(3 * x));
            }
        }

        for (; x < width; x++)
        {
            ref byte sp = ref Unsafe.Add(ref s, 3 * (width - 1 - x));
            ref byte dp = ref Unsafe.Add(ref d, 3 * x);
            dp = sp;
            Unsafe.Add(ref dp, 1) = Unsafe.Add(ref sp, 1);
            Unsafe.Add(ref dp, 2) = Unsafe.Add(ref sp, 2);
        }
    }

    // ---------------------------------------------------------------- transpose

    /// <summary>
    /// Writes destination rows <c>[i0, i1)</c> of <c>dst(i, j) = src(rj, ci)</c>, where
    /// <c>rj = reverseRows ? srcHeight-1-j : j</c> and <c>ci = reverseCols ? srcWidth-1-i : i</c>.
    /// The destination has <c>srcWidth</c> rows and <c>srcHeight</c> columns.
    /// </summary>
    public static void Transpose<T>(
        Plane<T> src, Plane<T> dst, int srcWidth, int srcHeight, int ps, bool reverseRows, bool reverseCols, int i0, int i1)
        where T : unmanaged
    {
        int n = ps <= 8 && (ps & (ps - 1)) == 0 && UnpackSupported ? 16 / ps : 0;
        int i = i0;
        if (n > 0)
        {
            nint sstep = reverseRows ? -src.Step : src.Step;
            nint dstep = reverseCols ? -dst.Step : dst.Step;
            for (; i + n <= i1; i += n)
            {
                int colStart = reverseCols ? srcWidth - n - i : i;
                ref byte d0 = ref dst.Row(reverseCols ? i + n - 1 : i);
                int j = 0;
                for (; j + n <= srcHeight; j += n)
                {
                    ref byte s = ref Unsafe.Add(ref src.Row(reverseRows ? srcHeight - 1 - j : j), colStart * ps);
                    ref byte d = ref Unsafe.Add(ref d0, j * ps);
                    switch (ps)
                    {
                        case 1: Block1(ref s, sstep, ref d, dstep); break;
                        case 2: Block2(ref s, sstep, ref d, dstep); break;
                        case 4: Block4(ref s, sstep, ref d, dstep); break;
                        default: Block8(ref s, sstep, ref d, dstep); break;
                    }
                }

                TransposeScalar(src, dst, srcWidth, srcHeight, ps, reverseRows, reverseCols, i, i + n, j, srcHeight);
            }
        }

        // Blocks of 16 pixels keep the source rows of one block in cache.
        for (int ib = i; ib < i1; ib += 16)
        {
            int ie = Math.Min(ib + 16, i1);
            for (int jb = 0; jb < srcHeight; jb += 16)
            {
                TransposeScalar(src, dst, srcWidth, srcHeight, ps, reverseRows, reverseCols, ib, ie, jb, Math.Min(jb + 16, srcHeight));
            }
        }
    }

    private static void TransposeScalar<T>(
        Plane<T> src, Plane<T> dst, int srcWidth, int srcHeight, int ps, bool reverseRows, bool reverseCols, int i0, int i1, int j0, int j1)
        where T : unmanaged
    {
        for (int i = i0; i < i1; i++)
        {
            int ci = reverseCols ? srcWidth - 1 - i : i;
            ref byte d = ref dst.Row(i);
            int sOff = ci * ps;
            for (int j = j0; j < j1; j++)
            {
                int rj = reverseRows ? srcHeight - 1 - j : j;
                ref byte sp = ref Unsafe.Add(ref src.Row(rj), sOff);
                ref byte dp = ref Unsafe.Add(ref d, j * ps);
                switch (ps)
                {
                    case 1: dp = sp; break;
                    case 2: Unsafe.WriteUnaligned(ref dp, Unsafe.ReadUnaligned<ushort>(ref sp)); break;
                    case 4: Unsafe.WriteUnaligned(ref dp, Unsafe.ReadUnaligned<uint>(ref sp)); break;
                    case 8: Unsafe.WriteUnaligned(ref dp, Unsafe.ReadUnaligned<ulong>(ref sp)); break;
                    default: Unsafe.CopyBlockUnaligned(ref dp, ref sp, (uint)ps); break;
                }
            }
        }
    }

    // Each block is an n x n pixel transpose of 16-byte rows: log2(n) rounds of interleaves of rows k and k + n/2.
    private static void Block1(ref byte s, nint sstep, ref byte d, nint dstep)
    {
        Vector128<byte> r0 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 0 * sstep));
        Vector128<byte> r1 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 1 * sstep));
        Vector128<byte> r2 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 2 * sstep));
        Vector128<byte> r3 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 3 * sstep));
        Vector128<byte> r4 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 4 * sstep));
        Vector128<byte> r5 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 5 * sstep));
        Vector128<byte> r6 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 6 * sstep));
        Vector128<byte> r7 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 7 * sstep));
        Vector128<byte> r8 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 8 * sstep));
        Vector128<byte> r9 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 9 * sstep));
        Vector128<byte> r10 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 10 * sstep));
        Vector128<byte> r11 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 11 * sstep));
        Vector128<byte> r12 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 12 * sstep));
        Vector128<byte> r13 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 13 * sstep));
        Vector128<byte> r14 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 14 * sstep));
        Vector128<byte> r15 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 15 * sstep));
        Vector128<byte> a0 = Zip8Lo(r0, r8);
        Vector128<byte> a1 = Zip8Hi(r0, r8);
        Vector128<byte> a2 = Zip8Lo(r1, r9);
        Vector128<byte> a3 = Zip8Hi(r1, r9);
        Vector128<byte> a4 = Zip8Lo(r2, r10);
        Vector128<byte> a5 = Zip8Hi(r2, r10);
        Vector128<byte> a6 = Zip8Lo(r3, r11);
        Vector128<byte> a7 = Zip8Hi(r3, r11);
        Vector128<byte> a8 = Zip8Lo(r4, r12);
        Vector128<byte> a9 = Zip8Hi(r4, r12);
        Vector128<byte> a10 = Zip8Lo(r5, r13);
        Vector128<byte> a11 = Zip8Hi(r5, r13);
        Vector128<byte> a12 = Zip8Lo(r6, r14);
        Vector128<byte> a13 = Zip8Hi(r6, r14);
        Vector128<byte> a14 = Zip8Lo(r7, r15);
        Vector128<byte> a15 = Zip8Hi(r7, r15);
        Vector128<byte> b0 = Zip8Lo(a0, a8);
        Vector128<byte> b1 = Zip8Hi(a0, a8);
        Vector128<byte> b2 = Zip8Lo(a1, a9);
        Vector128<byte> b3 = Zip8Hi(a1, a9);
        Vector128<byte> b4 = Zip8Lo(a2, a10);
        Vector128<byte> b5 = Zip8Hi(a2, a10);
        Vector128<byte> b6 = Zip8Lo(a3, a11);
        Vector128<byte> b7 = Zip8Hi(a3, a11);
        Vector128<byte> b8 = Zip8Lo(a4, a12);
        Vector128<byte> b9 = Zip8Hi(a4, a12);
        Vector128<byte> b10 = Zip8Lo(a5, a13);
        Vector128<byte> b11 = Zip8Hi(a5, a13);
        Vector128<byte> b12 = Zip8Lo(a6, a14);
        Vector128<byte> b13 = Zip8Hi(a6, a14);
        Vector128<byte> b14 = Zip8Lo(a7, a15);
        Vector128<byte> b15 = Zip8Hi(a7, a15);
        Vector128<byte> c0 = Zip8Lo(b0, b8);
        Vector128<byte> c1 = Zip8Hi(b0, b8);
        Vector128<byte> c2 = Zip8Lo(b1, b9);
        Vector128<byte> c3 = Zip8Hi(b1, b9);
        Vector128<byte> c4 = Zip8Lo(b2, b10);
        Vector128<byte> c5 = Zip8Hi(b2, b10);
        Vector128<byte> c6 = Zip8Lo(b3, b11);
        Vector128<byte> c7 = Zip8Hi(b3, b11);
        Vector128<byte> c8 = Zip8Lo(b4, b12);
        Vector128<byte> c9 = Zip8Hi(b4, b12);
        Vector128<byte> c10 = Zip8Lo(b5, b13);
        Vector128<byte> c11 = Zip8Hi(b5, b13);
        Vector128<byte> c12 = Zip8Lo(b6, b14);
        Vector128<byte> c13 = Zip8Hi(b6, b14);
        Vector128<byte> c14 = Zip8Lo(b7, b15);
        Vector128<byte> c15 = Zip8Hi(b7, b15);
        Vector128<byte> d0 = Zip8Lo(c0, c8);
        Vector128<byte> d1 = Zip8Hi(c0, c8);
        Vector128<byte> d2 = Zip8Lo(c1, c9);
        Vector128<byte> d3 = Zip8Hi(c1, c9);
        Vector128<byte> d4 = Zip8Lo(c2, c10);
        Vector128<byte> d5 = Zip8Hi(c2, c10);
        Vector128<byte> d6 = Zip8Lo(c3, c11);
        Vector128<byte> d7 = Zip8Hi(c3, c11);
        Vector128<byte> d8 = Zip8Lo(c4, c12);
        Vector128<byte> d9 = Zip8Hi(c4, c12);
        Vector128<byte> d10 = Zip8Lo(c5, c13);
        Vector128<byte> d11 = Zip8Hi(c5, c13);
        Vector128<byte> d12 = Zip8Lo(c6, c14);
        Vector128<byte> d13 = Zip8Hi(c6, c14);
        Vector128<byte> d14 = Zip8Lo(c7, c15);
        Vector128<byte> d15 = Zip8Hi(c7, c15);
        d0.StoreUnsafe(ref Unsafe.Add(ref d, 0 * dstep));
        d1.StoreUnsafe(ref Unsafe.Add(ref d, 1 * dstep));
        d2.StoreUnsafe(ref Unsafe.Add(ref d, 2 * dstep));
        d3.StoreUnsafe(ref Unsafe.Add(ref d, 3 * dstep));
        d4.StoreUnsafe(ref Unsafe.Add(ref d, 4 * dstep));
        d5.StoreUnsafe(ref Unsafe.Add(ref d, 5 * dstep));
        d6.StoreUnsafe(ref Unsafe.Add(ref d, 6 * dstep));
        d7.StoreUnsafe(ref Unsafe.Add(ref d, 7 * dstep));
        d8.StoreUnsafe(ref Unsafe.Add(ref d, 8 * dstep));
        d9.StoreUnsafe(ref Unsafe.Add(ref d, 9 * dstep));
        d10.StoreUnsafe(ref Unsafe.Add(ref d, 10 * dstep));
        d11.StoreUnsafe(ref Unsafe.Add(ref d, 11 * dstep));
        d12.StoreUnsafe(ref Unsafe.Add(ref d, 12 * dstep));
        d13.StoreUnsafe(ref Unsafe.Add(ref d, 13 * dstep));
        d14.StoreUnsafe(ref Unsafe.Add(ref d, 14 * dstep));
        d15.StoreUnsafe(ref Unsafe.Add(ref d, 15 * dstep));
    }

    private static void Block2(ref byte s, nint sstep, ref byte d, nint dstep)
    {
        Vector128<byte> r0 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 0 * sstep));
        Vector128<byte> r1 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 1 * sstep));
        Vector128<byte> r2 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 2 * sstep));
        Vector128<byte> r3 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 3 * sstep));
        Vector128<byte> r4 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 4 * sstep));
        Vector128<byte> r5 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 5 * sstep));
        Vector128<byte> r6 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 6 * sstep));
        Vector128<byte> r7 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 7 * sstep));
        Vector128<byte> a0 = Zip16Lo(r0, r4);
        Vector128<byte> a1 = Zip16Hi(r0, r4);
        Vector128<byte> a2 = Zip16Lo(r1, r5);
        Vector128<byte> a3 = Zip16Hi(r1, r5);
        Vector128<byte> a4 = Zip16Lo(r2, r6);
        Vector128<byte> a5 = Zip16Hi(r2, r6);
        Vector128<byte> a6 = Zip16Lo(r3, r7);
        Vector128<byte> a7 = Zip16Hi(r3, r7);
        Vector128<byte> b0 = Zip16Lo(a0, a4);
        Vector128<byte> b1 = Zip16Hi(a0, a4);
        Vector128<byte> b2 = Zip16Lo(a1, a5);
        Vector128<byte> b3 = Zip16Hi(a1, a5);
        Vector128<byte> b4 = Zip16Lo(a2, a6);
        Vector128<byte> b5 = Zip16Hi(a2, a6);
        Vector128<byte> b6 = Zip16Lo(a3, a7);
        Vector128<byte> b7 = Zip16Hi(a3, a7);
        Vector128<byte> c0 = Zip16Lo(b0, b4);
        Vector128<byte> c1 = Zip16Hi(b0, b4);
        Vector128<byte> c2 = Zip16Lo(b1, b5);
        Vector128<byte> c3 = Zip16Hi(b1, b5);
        Vector128<byte> c4 = Zip16Lo(b2, b6);
        Vector128<byte> c5 = Zip16Hi(b2, b6);
        Vector128<byte> c6 = Zip16Lo(b3, b7);
        Vector128<byte> c7 = Zip16Hi(b3, b7);
        c0.StoreUnsafe(ref Unsafe.Add(ref d, 0 * dstep));
        c1.StoreUnsafe(ref Unsafe.Add(ref d, 1 * dstep));
        c2.StoreUnsafe(ref Unsafe.Add(ref d, 2 * dstep));
        c3.StoreUnsafe(ref Unsafe.Add(ref d, 3 * dstep));
        c4.StoreUnsafe(ref Unsafe.Add(ref d, 4 * dstep));
        c5.StoreUnsafe(ref Unsafe.Add(ref d, 5 * dstep));
        c6.StoreUnsafe(ref Unsafe.Add(ref d, 6 * dstep));
        c7.StoreUnsafe(ref Unsafe.Add(ref d, 7 * dstep));
    }

    private static void Block4(ref byte s, nint sstep, ref byte d, nint dstep)
    {
        Vector128<byte> r0 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 0 * sstep));
        Vector128<byte> r1 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 1 * sstep));
        Vector128<byte> r2 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 2 * sstep));
        Vector128<byte> r3 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 3 * sstep));
        Vector128<byte> a0 = Zip32Lo(r0, r2);
        Vector128<byte> a1 = Zip32Hi(r0, r2);
        Vector128<byte> a2 = Zip32Lo(r1, r3);
        Vector128<byte> a3 = Zip32Hi(r1, r3);
        Vector128<byte> b0 = Zip32Lo(a0, a2);
        Vector128<byte> b1 = Zip32Hi(a0, a2);
        Vector128<byte> b2 = Zip32Lo(a1, a3);
        Vector128<byte> b3 = Zip32Hi(a1, a3);
        b0.StoreUnsafe(ref Unsafe.Add(ref d, 0 * dstep));
        b1.StoreUnsafe(ref Unsafe.Add(ref d, 1 * dstep));
        b2.StoreUnsafe(ref Unsafe.Add(ref d, 2 * dstep));
        b3.StoreUnsafe(ref Unsafe.Add(ref d, 3 * dstep));
    }

    private static void Block8(ref byte s, nint sstep, ref byte d, nint dstep)
    {
        Vector128<byte> r0 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 0 * sstep));
        Vector128<byte> r1 = Vector128.LoadUnsafe(ref Unsafe.Add(ref s, 1 * sstep));
        Vector128<byte> a0 = Zip64Lo(r0, r1);
        Vector128<byte> a1 = Zip64Hi(r0, r1);
        a0.StoreUnsafe(ref Unsafe.Add(ref d, 0 * dstep));
        a1.StoreUnsafe(ref Unsafe.Add(ref d, 1 * dstep));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip8Lo(Vector128<byte> a, Vector128<byte> b) => Sse2.IsSupported ? Sse2.UnpackLow(a, b) : AdvSimd.Arm64.ZipLow(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip8Hi(Vector128<byte> a, Vector128<byte> b) => Sse2.IsSupported ? Sse2.UnpackHigh(a, b) : AdvSimd.Arm64.ZipHigh(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip16Lo(Vector128<byte> a, Vector128<byte> b)
        => Sse2.IsSupported ? Sse2.UnpackLow(a.AsUInt16(), b.AsUInt16()).AsByte() : AdvSimd.Arm64.ZipLow(a.AsUInt16(), b.AsUInt16()).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip16Hi(Vector128<byte> a, Vector128<byte> b)
        => Sse2.IsSupported ? Sse2.UnpackHigh(a.AsUInt16(), b.AsUInt16()).AsByte() : AdvSimd.Arm64.ZipHigh(a.AsUInt16(), b.AsUInt16()).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip32Lo(Vector128<byte> a, Vector128<byte> b)
        => Sse2.IsSupported ? Sse2.UnpackLow(a.AsUInt32(), b.AsUInt32()).AsByte() : AdvSimd.Arm64.ZipLow(a.AsUInt32(), b.AsUInt32()).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip32Hi(Vector128<byte> a, Vector128<byte> b)
        => Sse2.IsSupported ? Sse2.UnpackHigh(a.AsUInt32(), b.AsUInt32()).AsByte() : AdvSimd.Arm64.ZipHigh(a.AsUInt32(), b.AsUInt32()).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip64Lo(Vector128<byte> a, Vector128<byte> b)
        => Sse2.IsSupported ? Sse2.UnpackLow(a.AsUInt64(), b.AsUInt64()).AsByte() : AdvSimd.Arm64.ZipLow(a.AsUInt64(), b.AsUInt64()).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Zip64Hi(Vector128<byte> a, Vector128<byte> b)
        => Sse2.IsSupported ? Sse2.UnpackHigh(a.AsUInt64(), b.AsUInt64()).AsByte() : AdvSimd.Arm64.ZipHigh(a.AsUInt64(), b.AsUInt64()).AsByte();

    /// <summary>Copies a matrix that shares its array with another matrix into a pooled packed matrix.</summary>
    public static PooledMat<T> Detach<T>(Mat<T> source)
        where T : unmanaged
    {
        PooledMat<T> copy = PooledMat<T>.Rent(source.Width, source.Height, source.Channels);
        source.CopyTo(copy.Mat);
        return copy;
    }

    public static bool SharesArray<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
        => ReferenceEquals(a.DangerousGetArray(), b.DangerousGetArray());
}
