// This implementation is ported from the OpenCV nearest-neighbour resize.
//
// Original source:
//   modules/imgproc/src/resize.cpp (resizeNN, resizeNNInvoker, resizeNN_bitexact,
//     resizeNN_bitexact_tab)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Nearest-neighbour resize: per-axis source indices and a pixel gather.</summary>
internal static class ResizeNearest
{
    /// <summary><c>INTER_NEAREST</c>: <c>min(cvFloor(d / invScale), size - 1)</c>.</summary>
    public static void Offsets(int srcSize, int dstSize, double invScale, Span<int> ofs)
    {
        double ifx = 1.0 / invScale;
        for (int d = 0; d < dstSize; d++)
        {
            ofs[d] = Math.Min(ScalarConverter.CvFloor(d * ifx), srcSize - 1);
        }
    }

    /// <summary>
    /// <c>INTER_NEAREST_EXACT</c> of OpenCV 5.0: start at half a source step and accumulate the step
    /// <c>src / dst</c> in double precision; each index is <c>min(floor(f), src - 1)</c>.
    /// </summary>
    public static void ExactOffsets(int srcSize, int dstSize, Span<int> ofs)
    {
        double scale = (double)srcSize / dstSize;
        double f = scale * 0.5;
        for (int d = 0; d < dstSize; d++)
        {
            ofs[d] = Math.Min(ScalarConverter.CvFloor(f), srcSize - 1);
            f += scale;
        }
    }

    public static void Run<T>(Mat<T> src, Mat<T> dst, int[] xofs, int[] yofs)
        where T : unmanaged
    {
        int ps = Unsafe.SizeOf<T>() * src.Channels;
        var s = new TransformKernels.Plane<T>(src);
        var d = new TransformKernels.Plane<T>(dst);
        int width = dst.Width;
        int srcLast = src.Width - 1;
        ParallelRows.For(dst.Height, (long)width * ps / 4, (y0, y1) => Rows(s, d, xofs, yofs, width, srcLast, ps, y0, y1));
    }

    private static void Rows<T>(TransformKernels.Plane<T> src, TransformKernels.Plane<T> dst, int[] xofs, int[] yofs, int width, int srcLast, int ps, int y0, int y1)
        where T : unmanaged
    {
        ref int xo = ref MemoryMarshal.GetArrayDataReference(xofs);
        for (int y = y0; y < y1; y++)
        {
            ref byte s = ref src.Row(yofs[y]);
            ref byte d = ref dst.Row(y);
            switch (ps)
            {
                case 1:
                    for (int x = 0; x < width; x++)
                    {
                        Unsafe.Add(ref d, x) = Unsafe.Add(ref s, Unsafe.Add(ref xo, x));
                    }

                    break;
                case 2:
                    Gather<ushort>(ref s, ref d, ref xo, width);
                    break;
                case 4:
                    Gather<uint>(ref s, ref d, ref xo, width);
                    break;
                case 8:
                    Gather<ulong>(ref s, ref d, ref xo, width);
                    break;
                case 3:
                {
                    // A 4-byte copy rewrites the first byte of the next destination pixel, so the last pixel and
                    // reads of the last source pixel use the exact copy.
                    int x = 0;
                    for (; x < width - 1; x++)
                    {
                        int so = Unsafe.Add(ref xo, x);
                        if (so >= srcLast)
                        {
                            break;
                        }

                        Unsafe.WriteUnaligned(ref Unsafe.Add(ref d, x * 3), Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref s, so * 3)));
                    }

                    for (; x < width; x++)
                    {
                        ref byte sp = ref Unsafe.Add(ref s, Unsafe.Add(ref xo, x) * 3);
                        ref byte dp = ref Unsafe.Add(ref d, x * 3);
                        Unsafe.WriteUnaligned(ref dp, Unsafe.ReadUnaligned<ushort>(ref sp));
                        Unsafe.Add(ref dp, 2) = Unsafe.Add(ref sp, 2);
                    }

                    break;
                }

                default:
                    for (int x = 0; x < width; x++)
                    {
                        Unsafe.CopyBlockUnaligned(ref Unsafe.Add(ref d, x * ps), ref Unsafe.Add(ref s, Unsafe.Add(ref xo, x) * ps), (uint)ps);
                    }

                    break;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Gather<TP>(ref byte s, ref byte d, ref int xo, int width)
        where TP : unmanaged
    {
        ref TP sp = ref Unsafe.As<byte, TP>(ref s);
        ref TP dp = ref Unsafe.As<byte, TP>(ref d);
        for (int x = 0; x < width; x++)
        {
            Unsafe.WriteUnaligned(ref Unsafe.As<TP, byte>(ref Unsafe.Add(ref dp, x)), Unsafe.ReadUnaligned<TP>(ref Unsafe.As<TP, byte>(ref Unsafe.Add(ref sp, Unsafe.Add(ref xo, x)))));
        }
    }

    public static int[] Rent(int length) => ArrayPool<int>.Shared.Rent(length);

    public static void Return(int[] array) => ArrayPool<int>.Shared.Return(array);
}
