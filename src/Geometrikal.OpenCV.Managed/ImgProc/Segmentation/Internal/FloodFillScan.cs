// This implementation is ported from the OpenCV flood fill.
//
// Original source:
//   modules/imgproc/src/floodfill.cpp (FFillSegment, ICV_PUSH, ICV_POP, floodFill_CnIR,
//   floodFillGrad_CnIR)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/floodfill.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// OpenCV <c>FFillSegment</c>: one filled run of row <see cref="Y"/> from <see cref="L"/> to
/// <see cref="R"/>, the run <see cref="PrevL"/>..<see cref="PrevR"/> of the parent row, and the
/// direction <see cref="Dir"/> from the parent row to this row.
/// </summary>
/// <remarks>The fields are 16-bit as in OpenCV, so image dimensions are limited to 65535.</remarks>
[StructLayout(LayoutKind.Sequential)]
internal struct FloodSegment
{
    public ushort Y;
    public ushort L;
    public ushort R;
    public ushort PrevL;
    public ushort PrevR;
    public short Dir;
}

/// <summary>The last-in first-out segment stack of the scan-line fill, in a pooled array.</summary>
internal ref struct FloodSegmentStack
{
    private FloodSegment[] _buffer;
    private int _count;

    public FloodSegmentStack(int capacity)
    {
        _buffer = ArrayPool<FloodSegment>.Shared.Rent(capacity);
        _count = 0;
    }

    public readonly bool IsEmpty => _count == 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Push(int y, int l, int r, int prevL, int prevR, int dir)
    {
        if (_count == _buffer.Length)
        {
            Grow();
        }

        ref FloodSegment s = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_buffer), _count++);
        s.Y = (ushort)y;
        s.L = (ushort)l;
        s.R = (ushort)r;
        s.PrevL = (ushort)prevL;
        s.PrevR = (ushort)prevR;
        s.Dir = (short)dir;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Pop(out int y, out int l, out int r, out int prevL, out int prevR, out int dir)
    {
        ref FloodSegment s = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_buffer), --_count);
        y = s.Y;
        l = s.L;
        r = s.R;
        prevL = s.PrevL;
        prevR = s.PrevR;
        dir = s.Dir;
    }

    public void Dispose()
    {
        ArrayPool<FloodSegment>.Shared.Return(_buffer);
        _buffer = [];
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void Grow()
    {
        FloodSegment[] next = ArrayPool<FloodSegment>.Shared.Rent(_buffer.Length * 2);
        _buffer.AsSpan(0, _count).CopyTo(next);
        ArrayPool<FloodSegment>.Shared.Return(_buffer);
        _buffer = next;
    }
}

/// <summary>
/// The scan-line fills of <c>cv::floodFill</c>. The segment push order and the pixel tests are
/// those of OpenCV, so the visited pixels, the area and the bounding rectangle are identical.
/// </summary>
/// <remarks>
/// <para>Image references address pixel (0, 0); pixel <c>x</c> of a row starts at element
/// <c>x * Channels</c>. The mask reference addresses mask element (1, 1), so the one-pixel mask
/// border is at offsets -1 and <c>width</c> and at rows -1 and <c>height</c>.</para>
/// <para>The mask border is non-zero, so a scan never passes it. A neighbour row outside the
/// image is skipped: OpenCV scans the mask border row there and finds no candidate.</para>
/// </remarks>
internal static class FloodFillScan
{
    private const int Up = 1;
    private const int FixedRangeFlag = 1 << 16;
    private const int MaskOnlyFlag = 1 << 17;

    /// <summary>
    /// OpenCV <c>floodFillGrad_CnIR</c>: grows the region in the mask with <paramref name="diff"/>
    /// and writes <paramref name="newValue"/> to each run after its neighbours are scanned.
    /// </summary>
    public static FloodFillResult Gradient<T, TDiff>(
        ref T image,
        int stride,
        int width,
        int height,
        ref byte mask,
        int maskStride,
        Point seed,
        ReadOnlySpan<T> newValue,
        byte newMaskValue,
        TDiff diff,
        int flags)
        where T : unmanaged
        where TDiff : struct, IFloodDiff<T>
    {
        int cn = TDiff.Channels;
        int e = (flags & 255) == 8 ? 1 : 0;
        bool eight = e != 0;
        bool fixedRange = (flags & FixedRangeFlag) != 0;
        bool fillImage = (flags & MaskOnlyFlag) == 0;

        ref byte m = ref Unsafe.Add(ref mask, (nint)seed.Y * maskStride);
        ref T img = ref Unsafe.Add(ref image, (nint)seed.Y * stride);
        int l = seed.X;
        int r = seed.X;
        if (Unsafe.Add(ref m, l) != 0)
        {
            return default;
        }

        Unsafe.Add(ref m, l) = newMaskValue;
        Span<T> val0Buffer = stackalloc T[3];
        MemoryMarshal.CreateReadOnlySpan(ref Unsafe.Add(ref img, l * cn), cn).CopyTo(val0Buffer);
        ref T val0 = ref MemoryMarshal.GetReference(val0Buffer);

        if (fixedRange)
        {
            while (Unsafe.Add(ref m, r + 1) == 0 && diff.Within(ref Unsafe.Add(ref img, (r + 1) * cn), ref val0))
            {
                Unsafe.Add(ref m, ++r) = newMaskValue;
            }

            while (Unsafe.Add(ref m, l - 1) == 0 && diff.Within(ref Unsafe.Add(ref img, (l - 1) * cn), ref val0))
            {
                Unsafe.Add(ref m, --l) = newMaskValue;
            }
        }
        else
        {
            while (Unsafe.Add(ref m, r + 1) == 0 && diff.Within(ref Unsafe.Add(ref img, (r + 1) * cn), ref Unsafe.Add(ref img, r * cn)))
            {
                Unsafe.Add(ref m, ++r) = newMaskValue;
            }

            while (Unsafe.Add(ref m, l - 1) == 0 && diff.Within(ref Unsafe.Add(ref img, (l - 1) * cn), ref Unsafe.Add(ref img, l * cn)))
            {
                Unsafe.Add(ref m, --l) = newMaskValue;
            }
        }

        int xMax = r;
        int xMin = l;
        int yMin = seed.Y;
        int yMax = seed.Y;
        int area = 0;

        var stack = new FloodSegmentStack(Math.Max(width, height) * 2);
        try
        {
            stack.Push(seed.Y, l, r, r + 1, r, Up);
            while (!stack.IsEmpty)
            {
                stack.Pop(out int yc, out l, out r, out int pl, out int pr, out int dir);
                int length = r - l;
                area += length + 1;
                xMax = Math.Max(xMax, r);
                xMin = Math.Min(xMin, l);
                yMax = Math.Max(yMax, yc);
                yMin = Math.Min(yMin, yc);

                ref T img1 = ref Unsafe.Add(ref image, (nint)yc * stride);
                for (int k = 0; k < 3; k++)
                {
                    int d = k == 0 ? -dir : dir;
                    int left = k == 2 ? pr + 1 : l - e;
                    int right = k == 0 ? r + e : k == 1 ? pl - 1 : r + e;
                    int y = yc + d;
                    if ((uint)y >= (uint)height)
                    {
                        continue;
                    }

                    img = ref Unsafe.Add(ref image, (nint)y * stride);
                    m = ref Unsafe.Add(ref mask, (nint)y * maskStride);

                    if (fixedRange)
                    {
                        ScanFixed(ref stack, ref img, ref m, ref val0, cn, diff, newMaskValue, left, right, y, l, r, -d);
                    }
                    else if (!eight)
                    {
                        Scan4(ref stack, ref img, ref img1, ref m, cn, diff, newMaskValue, left, right, y, l, r, -d);
                    }
                    else
                    {
                        Scan8(ref stack, ref img, ref img1, ref m, cn, diff, newMaskValue, left, right, y, l, r, -d);
                    }
                }

                if (fillImage)
                {
                    FillRun(ref img1, l, r, cn, newValue);
                }
            }
        }
        finally
        {
            stack.Dispose();
        }

        return new FloodFillResult(area, new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1));
    }

    /// <summary>
    /// OpenCV <c>floodFill_CnIR</c>: fills the 4- or 8-connected run of pixels equal to the seed
    /// pixel. The caller ensures the seed pixel differs from <paramref name="newValue"/>.
    /// </summary>
    public static FloodFillResult Simple<T, TEq>(
        ref T image,
        int stride,
        int width,
        int height,
        Point seed,
        ReadOnlySpan<T> newValue,
        int flags)
        where T : unmanaged
        where TEq : struct, IFloodDiff<T>
    {
        int cn = TEq.Channels;
        TEq eq = default;
        int e = (flags & 255) == 8 ? 1 : 0;
        ref T img = ref Unsafe.Add(ref image, (nint)seed.Y * stride);
        ref T nv = ref MemoryMarshal.GetReference(newValue);
        int l = seed.X;
        int r = seed.X;

        Span<T> val0Buffer = stackalloc T[3];
        MemoryMarshal.CreateReadOnlySpan(ref Unsafe.Add(ref img, l * cn), cn).CopyTo(val0Buffer);
        ref T val0 = ref MemoryMarshal.GetReference(val0Buffer);
        SetPixel(ref Unsafe.Add(ref img, l * cn), ref nv, cn);

        while (++r < width && eq.Within(ref Unsafe.Add(ref img, r * cn), ref val0))
        {
            SetPixel(ref Unsafe.Add(ref img, r * cn), ref nv, cn);
        }

        while (--l >= 0 && eq.Within(ref Unsafe.Add(ref img, l * cn), ref val0))
        {
            SetPixel(ref Unsafe.Add(ref img, l * cn), ref nv, cn);
        }

        int xMax = --r;
        int xMin = ++l;
        int yMin = seed.Y;
        int yMax = seed.Y;
        int area = 0;

        var stack = new FloodSegmentStack(Math.Max(width, height) * 2);
        try
        {
            stack.Push(seed.Y, l, r, r + 1, r, Up);
            while (!stack.IsEmpty)
            {
                stack.Pop(out int yc, out l, out r, out int pl, out int pr, out int dir);
                area += r - l + 1;
                xMax = Math.Max(xMax, r);
                xMin = Math.Min(xMin, l);
                yMax = Math.Max(yMax, yc);
                yMin = Math.Min(yMin, yc);

                for (int k = 0; k < 3; k++)
                {
                    int d = k == 0 ? -dir : dir;
                    int y = yc + d;
                    if ((uint)y >= (uint)height)
                    {
                        continue;
                    }

                    int left = k == 2 ? pr + 1 : l - e;
                    int right = k == 1 ? pl - 1 : r + e;
                    img = ref Unsafe.Add(ref image, (nint)y * stride);
                    for (int i = left; i <= right; i++)
                    {
                        if ((uint)i < (uint)width && eq.Within(ref Unsafe.Add(ref img, i * cn), ref val0))
                        {
                            int j = i;
                            SetPixel(ref Unsafe.Add(ref img, i * cn), ref nv, cn);
                            while (--j >= 0 && eq.Within(ref Unsafe.Add(ref img, j * cn), ref val0))
                            {
                                SetPixel(ref Unsafe.Add(ref img, j * cn), ref nv, cn);
                            }

                            while (++i < width && eq.Within(ref Unsafe.Add(ref img, i * cn), ref val0))
                            {
                                SetPixel(ref Unsafe.Add(ref img, i * cn), ref nv, cn);
                            }

                            stack.Push(y, j + 1, i - 1, l, r, -d);
                        }
                    }
                }
            }
        }
        finally
        {
            stack.Dispose();
        }

        return new FloodFillResult(area, new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1));
    }

    /// <summary>
    /// Single-channel <see cref="Simple{T, TEq}"/> with vectorized searches for run ends and run
    /// starts. The pushed segments and their order are those of <see cref="Simple{T, TEq}"/>.
    /// </summary>
    public static FloodFillResult SimpleC1<T>(
        ref T image,
        int stride,
        int width,
        int height,
        Point seed,
        T newValue,
        int flags)
        where T : unmanaged, IEquatable<T>
    {
        int e = (flags & 255) == 8 ? 1 : 0;
        ref T img = ref Unsafe.Add(ref image, (nint)seed.Y * stride);
        T val0 = Unsafe.Add(ref img, seed.X);
        int r = FillRight(ref img, seed.X, width, val0, newValue) - 1;
        int l = FillLeft(ref img, seed.X, val0, newValue) + 1;
        int xMax = r;
        int xMin = l;
        int yMin = seed.Y;
        int yMax = seed.Y;
        int area = 0;

        var stack = new FloodSegmentStack(Math.Max(width, height) * 2);
        try
        {
            stack.Push(seed.Y, l, r, r + 1, r, Up);
            while (!stack.IsEmpty)
            {
                stack.Pop(out int yc, out l, out r, out int pl, out int pr, out int dir);
                area += r - l + 1;
                xMax = Math.Max(xMax, r);
                xMin = Math.Min(xMin, l);
                yMax = Math.Max(yMax, yc);
                yMin = Math.Min(yMin, yc);

                for (int k = 0; k < 3; k++)
                {
                    int d = k == 0 ? -dir : dir;
                    int y = yc + d;
                    if ((uint)y >= (uint)height)
                    {
                        continue;
                    }

                    int i = Math.Max(k == 2 ? pr + 1 : l - e, 0);
                    int right = Math.Min(k == 1 ? pl - 1 : r + e, width - 1);
                    img = ref Unsafe.Add(ref image, (nint)y * stride);
                    while (i <= right)
                    {
                        int found = MemoryMarshal.CreateReadOnlySpan(ref Unsafe.Add(ref img, i), right - i + 1).IndexOf(val0);
                        if (found < 0)
                        {
                            break;
                        }

                        i += found;
                        int j = FillLeft(ref img, i, val0, newValue);
                        i = FillRight(ref img, i, width, val0, newValue);
                        stack.Push(y, j + 1, i - 1, l, r, -d);
                        i++;
                    }
                }
            }
        }
        finally
        {
            stack.Dispose();
        }

        return new FloodFillResult(area, new Rect(xMin, yMin, xMax - xMin + 1, yMax - yMin + 1));
    }

    /// <summary>Fills pixel <paramref name="i"/> and the equal pixels to its right; returns the first index after them.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FillRight<T>(ref T row, int i, int width, T val0, T newValue)
        where T : unmanaged, IEquatable<T>
    {
        Unsafe.Add(ref row, i) = newValue;
        int start = i + 1;
        Span<T> rest = MemoryMarshal.CreateSpan(ref Unsafe.Add(ref row, start), width - start);
        int n = rest.IndexOfAnyExcept(val0);
        if (n < 0)
        {
            n = rest.Length;
        }

        rest[..n].Fill(newValue);
        return start + n;
    }

    /// <summary>Fills the equal pixels to the left of pixel <paramref name="i"/>; returns the index before them (-1 at the row start).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FillLeft<T>(ref T row, int i, T val0, T newValue)
        where T : unmanaged, IEquatable<T>
    {
        Span<T> before = MemoryMarshal.CreateSpan(ref row, i);
        int j = before.LastIndexOfAnyExcept(val0);
        before[(j + 1)..].Fill(newValue);
        return j;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScanFixed<T, TDiff>(
        ref FloodSegmentStack stack, ref T img, ref byte m, ref T val0, int cn, TDiff diff, byte nm,
        int left, int right, int y, int l, int r, int pushDir)
        where T : unmanaged
        where TDiff : struct, IFloodDiff<T>
    {
        for (int i = left; i <= right; i++)
        {
            if (Unsafe.Add(ref m, i) == 0 && diff.Within(ref Unsafe.Add(ref img, i * cn), ref val0))
            {
                int j = i;
                Unsafe.Add(ref m, i) = nm;
                while (Unsafe.Add(ref m, --j) == 0 && diff.Within(ref Unsafe.Add(ref img, j * cn), ref val0))
                {
                    Unsafe.Add(ref m, j) = nm;
                }

                while (Unsafe.Add(ref m, ++i) == 0 && diff.Within(ref Unsafe.Add(ref img, i * cn), ref val0))
                {
                    Unsafe.Add(ref m, i) = nm;
                }

                stack.Push(y, j + 1, i - 1, l, r, pushDir);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Scan4<T, TDiff>(
        ref FloodSegmentStack stack, ref T img, ref T img1, ref byte m, int cn, TDiff diff, byte nm,
        int left, int right, int y, int l, int r, int pushDir)
        where T : unmanaged
        where TDiff : struct, IFloodDiff<T>
    {
        for (int i = left; i <= right; i++)
        {
            if (Unsafe.Add(ref m, i) == 0 && diff.Within(ref Unsafe.Add(ref img, i * cn), ref Unsafe.Add(ref img1, i * cn)))
            {
                int j = i;
                Unsafe.Add(ref m, i) = nm;
                while (Unsafe.Add(ref m, --j) == 0 && diff.Within(ref Unsafe.Add(ref img, j * cn), ref Unsafe.Add(ref img, (j + 1) * cn)))
                {
                    Unsafe.Add(ref m, j) = nm;
                }

                while (Unsafe.Add(ref m, ++i) == 0
                    && (diff.Within(ref Unsafe.Add(ref img, i * cn), ref Unsafe.Add(ref img, (i - 1) * cn))
                        || (diff.Within(ref Unsafe.Add(ref img, i * cn), ref Unsafe.Add(ref img1, i * cn)) && i <= r)))
                {
                    Unsafe.Add(ref m, i) = nm;
                }

                stack.Push(y, j + 1, i - 1, l, r, pushDir);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Scan8<T, TDiff>(
        ref FloodSegmentStack stack, ref T img, ref T img1, ref byte m, int cn, TDiff diff, byte nm,
        int left, int right, int y, int l, int r, int pushDir)
        where T : unmanaged
        where TDiff : struct, IFloodDiff<T>
    {
        uint length = (uint)(r - l);
        for (int i = left; i <= right; i++)
        {
            if (Unsafe.Add(ref m, i) == 0 && TouchesParent(ref img, ref img1, cn, diff, i, l, length))
            {
                int j = i;
                Unsafe.Add(ref m, i) = nm;
                while (Unsafe.Add(ref m, --j) == 0 && diff.Within(ref Unsafe.Add(ref img, j * cn), ref Unsafe.Add(ref img, (j + 1) * cn)))
                {
                    Unsafe.Add(ref m, j) = nm;
                }

                while (Unsafe.Add(ref m, ++i) == 0
                    && (diff.Within(ref Unsafe.Add(ref img, i * cn), ref Unsafe.Add(ref img, (i - 1) * cn))
                        || TouchesParent(ref img, ref img1, cn, diff, i, l, length)))
                {
                    Unsafe.Add(ref m, i) = nm;
                }

                stack.Push(y, j + 1, i - 1, l, r, pushDir);
            }
        }
    }

    /// <summary>Tests pixel <paramref name="i"/> against parent pixels i-1, i and i+1 that lie in the parent run l..l+length.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TouchesParent<T, TDiff>(ref T img, ref T img1, int cn, TDiff diff, int i, int l, uint length)
        where T : unmanaged
        where TDiff : struct, IFloodDiff<T>
    {
        ref T val = ref Unsafe.Add(ref img, i * cn);
        int idx = i - l - 1;
        return ((uint)idx <= length && diff.Within(ref val, ref Unsafe.Add(ref img1, (i - 1) * cn)))
            || ((uint)(++idx) <= length && diff.Within(ref val, ref Unsafe.Add(ref img1, i * cn)))
            || ((uint)(++idx) <= length && diff.Within(ref val, ref Unsafe.Add(ref img1, (i + 1) * cn)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SetPixel<T>(ref T dst, ref T value, int cn)
        where T : unmanaged
    {
        dst = value;
        if (cn == 3)
        {
            Unsafe.Add(ref dst, 1) = Unsafe.Add(ref value, 1);
            Unsafe.Add(ref dst, 2) = Unsafe.Add(ref value, 2);
        }
    }

    private static void FillRun<T>(ref T row, int l, int r, int cn, ReadOnlySpan<T> value)
        where T : unmanaged
    {
        if (cn == 1)
        {
            MemoryMarshal.CreateSpan(ref Unsafe.Add(ref row, l), r - l + 1).Fill(value[0]);
            return;
        }

        ref T v = ref MemoryMarshal.GetReference(value);
        for (int i = l; i <= r; i++)
        {
            SetPixel(ref Unsafe.Add(ref row, i * cn), ref v, cn);
        }
    }
}
