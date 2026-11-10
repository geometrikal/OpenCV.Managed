// This implementation is ported from the OpenCV bit-exact 8-bit Gaussian.
//
// Original sources:
//   modules/imgproc/src/smooth.simd.hpp (GaussianBlurFixedPoint,
//     hlineSmooth*/vlineSmooth*, fixedSmoothInvoker)
//   modules/imgproc/src/fixedpoint.inl.hpp (ufixedpoint16 / ufixedpoint32)
//
// The Q8.8 horizontal pass and Q16.16 vertical pass are exact integer
// arithmetic without reachable saturation (kernel raws sum to exactly 256 by
// construction, so row sums are ≤ 65280 and column sums ≤ 16,711,680), and
// the single rounding happens in the final cast: saturate_u8((sum + 0x8000)
// >> 16). This makes OpenCV's many kernel-size specializations and its
// streaming border logic arithmetic-identical to the single generic padded
// two-pass implemented here.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// The bit-exact fixed-point Gaussian blur for 8-bit images
/// (<c>GaussianBlurFixedPoint&lt;uint16_t&gt;</c>).
/// </summary>
internal static class GaussianFixedPoint
{
    /// <summary>
    /// Applies the separable Q8.8 Gaussian. Kernels are raw
    /// <c>ufixedpoint16</c> values from
    /// <see cref="GaussianKernels.GetKernelFixedPointQ8"/>.
    /// </summary>
    public static Mat<byte> Apply(Mat<byte> source, ushort[] kernelX, ushort[] kernelY, BorderType borderType)
    {
        int kw = kernelX.Length;
        int kh = kernelY.Length;
        int rowLength = source.RowLength;
        var dst = new Mat<byte>(source.Width, source.Height, source.Channels, clear: false);
        RowStream.Run<byte, ushort, byte, Kernel>(
            source, dst, kw, kh, kw / 2, kh / 2, borderType, rowLength,
            new Kernel(kernelX, kernelY, source.Channels),
            (long)rowLength * (kw + (2 * kh)) / Vector<ushort>.Count);
        return dst;
    }

    /// <summary>Q8.8 row sums and Q16.16 column sums with one round-half-up cast.</summary>
    private readonly struct Kernel(ushort[] kernelX, ushort[] kernelY, int cn) : IRowStreamKernel<byte, ushort, byte>
    {
        public void Row(ReadOnlySpan<byte> line, Span<ushort> row, Span<ushort> wide)
            => RowPass(line, kernelX, cn, ref MemoryMarshal.GetReference(row), row.Length);

        public void Column(ref ushort rows, int rowStride, Span<byte> dst, Span<ushort> scratch)
            => ColumnPass(ref rows, rowStride, kernelY, dst);
    }

    private static void RowPass(ReadOnlySpan<byte> row, ushort[] kernel, int cn, ref ushort d, int length)
    {
        ref byte s = ref MemoryMarshal.GetReference(row);
        int taps = kernel.Length;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && length >= Vector512<byte>.Count)
        {
            int vc = Vector512<byte>.Count;
            int half = Vector512<ushort>.Count;
            for (; i < length; i += vc)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, length - vc);
                (Vector512<ushort> lo, Vector512<ushort> hi) = Vector512.Widen(Vector512.LoadUnsafe(ref s, (nuint)i));
                var k0 = Vector512.Create(kernel[0]);
                lo *= k0;
                hi *= k0;
                for (int k = 1; k < taps; k++)
                {
                    (Vector512<ushort> wl, Vector512<ushort> wh) = Vector512.Widen(Vector512.LoadUnsafe(ref s, (nuint)(i + (k * cn))));
                    var kk = Vector512.Create(kernel[k]);
                    lo += wl * kk;
                    hi += wh * kk;
                }

                lo.StoreUnsafe(ref d, (nuint)i);
                hi.StoreUnsafe(ref d, (nuint)(i + half));
            }

            return;
        }

        if (MinMaxOps.IsVectorized<byte>())
        {
            int vc = Vector<byte>.Count;
            int half = Vector<ushort>.Count;
            for (; i <= length - vc; i += vc)
            {
                Vector<ushort> lo = Vector<ushort>.Zero;
                Vector<ushort> hi = Vector<ushort>.Zero;
                for (int k = 0; k < taps; k++)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)(i + (k * cn))), out Vector<ushort> wl, out Vector<ushort> wh);
                    var kk = new Vector<ushort>(kernel[k]);
                    lo += wl * kk;
                    hi += wh * kk;
                }

                lo.StoreUnsafe(ref d, (nuint)i);
                hi.StoreUnsafe(ref d, (nuint)(i + half));
            }
        }

        for (; i < length; i++)
        {
            uint sum = 0;
            for (int k = 0; k < taps; k++)
            {
                sum += (uint)kernel[k] * Unsafe.Add(ref s, i + (k * cn));
            }

            Unsafe.Add(ref d, i) = (ushort)sum;
        }
    }

    private static void ColumnPass(ref ushort rows, int rowLength, ushort[] kernel, Span<byte> dstRow)
    {
        ref byte d = ref MemoryMarshal.GetReference(dstRow);
        int taps = kernel.Length;
        int length = dstRow.Length;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && length >= Vector512<byte>.Count)
        {
            int vc = Vector512<byte>.Count;
            int half = Vector512<ushort>.Count;
            var round = Vector512.Create(0x8000u);
            var max = Vector512.Create(255u);
            for (; i < length; i += vc)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, length - vc);
                Vector512<uint> a0 = round, a1 = round, a2 = round, a3 = round;
                for (int k = 0; k < taps; k++)
                {
                    ref ushort r = ref Unsafe.Add(ref rows, k * rowLength);
                    var kk = Vector512.Create((uint)kernel[k]);
                    (Vector512<uint> w0, Vector512<uint> w1) = Vector512.Widen(Vector512.LoadUnsafe(ref r, (nuint)i));
                    (Vector512<uint> w2, Vector512<uint> w3) = Vector512.Widen(Vector512.LoadUnsafe(ref r, (nuint)(i + half)));
                    a0 += w0 * kk;
                    a1 += w1 * kk;
                    a2 += w2 * kk;
                    a3 += w3 * kk;
                }

                Vector512<ushort> n0 = Vector512.Narrow(Vector512.Min(a0 >>> 16, max), Vector512.Min(a1 >>> 16, max));
                Vector512<ushort> n1 = Vector512.Narrow(Vector512.Min(a2 >>> 16, max), Vector512.Min(a3 >>> 16, max));
                Vector512.Narrow(n0, n1).StoreUnsafe(ref d, (nuint)i);
            }

            return;
        }

        if (MinMaxOps.IsVectorized<byte>())
        {
            int vc = Vector<byte>.Count;
            int half = Vector<ushort>.Count;
            var round = new Vector<uint>(0x8000u);
            var max = new Vector<uint>(255u);
            for (; i <= length - vc; i += vc)
            {
                Vector<uint> a0 = round, a1 = round, a2 = round, a3 = round;
                for (int k = 0; k < taps; k++)
                {
                    ref ushort r = ref Unsafe.Add(ref rows, k * rowLength);
                    var kk = new Vector<uint>(kernel[k]);
                    Vector.Widen(Vector.LoadUnsafe(ref r, (nuint)i), out Vector<uint> w0, out Vector<uint> w1);
                    Vector.Widen(Vector.LoadUnsafe(ref r, (nuint)(i + half)), out Vector<uint> w2, out Vector<uint> w3);
                    a0 += w0 * kk;
                    a1 += w1 * kk;
                    a2 += w2 * kk;
                    a3 += w3 * kk;
                }

                Vector<ushort> n0 = Vector.Narrow(Vector.Min(a0 >>> 16, max), Vector.Min(a1 >>> 16, max));
                Vector<ushort> n1 = Vector.Narrow(Vector.Min(a2 >>> 16, max), Vector.Min(a3 >>> 16, max));
                Vector.Narrow(n0, n1).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < length; i++)
        {
            uint sum = 0;
            for (int k = 0; k < taps; k++)
            {
                sum += (uint)kernel[k] * Unsafe.Add(ref rows, (k * rowLength) + i);
            }

            uint rounded = (sum + 0x8000u) >> 16;
            Unsafe.Add(ref d, i) = rounded > 255 ? (byte)255 : (byte)rounded;
        }
    }
}
