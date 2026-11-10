// This implementation is ported from the OpenCV linear blend.
//
// Original source:
//   modules/imgproc/src/blend.cpp (cv::blendLinear, BlendLinearInvoker, blendLinearSimd)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Weighted blending of two images, as OpenCV <c>blendLinear</c>.</summary>
public static class Blending
{
    /// <summary>
    /// Blends two images with per-pixel weights, as OpenCV <c>blendLinear</c>:
    /// <c>dst = (src1·w1 + src2·w2) / (w1 + w2 + 1e-5)</c>.
    /// </summary>
    /// <param name="source1">First image, <c>byte</c> or <c>float</c>, any channel count.</param>
    /// <param name="source2">Second image of the same type and size.</param>
    /// <param name="weights1">Single-channel weights of <paramref name="source1"/>.</param>
    /// <param name="weights2">Single-channel weights of <paramref name="source2"/>.</param>
    /// <returns>A new image. Byte results round half to even and saturate.</returns>
    /// <remarks>The arithmetic is float, in the order of the OpenCV scalar and SSE code (they agree).</remarks>
    public static Mat<T> BlendLinear<T>(Mat<T> source1, Mat<T> source2, Mat<float> weights1, Mat<float> weights2)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source1);
        ArgumentNullException.ThrowIfNull(source2);
        ArgumentNullException.ThrowIfNull(weights1);
        ArgumentNullException.ThrowIfNull(weights2);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException("Only byte and float images are supported.");
        }

        int w = source1.Width, h = source1.Height, cn = source1.Channels;
        if (source2.Width != w || source2.Height != h || source2.Channels != cn)
        {
            throw new ArgumentException("The images must have the same size and channel count.", nameof(source2));
        }

        if (weights1.Width != w || weights1.Height != h || weights1.Channels != 1 || weights2.Width != w || weights2.Height != h || weights2.Channels != 1)
        {
            throw new ArgumentException("The weights must be single-channel with the image size.", nameof(weights1));
        }

        var dst = new Mat<T>(w, h, cn, clear: false);
        ParallelRows.For(h, (long)w * cn, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                var a = source1.GetReadOnlyRowSpan(y);
                var b = source2.GetReadOnlyRowSpan(y);
                var w1 = weights1.GetReadOnlyRowSpan(y);
                var w2 = weights2.GetReadOnlyRowSpan(y);
                var o = dst.GetRowSpan(y);
                for (int x = 0; x < o.Length; x++)
                {
                    float p = w1[x / cn], q = w2[x / cn];
                    float den = p + q + 1e-5f;
                    float num = (Value(a[x]) * p) + (Value(b[x]) * q);
                    o[x] = Store<T>(num / den);
                }
            }
        });
        return dst;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Value<T>(T v)
        where T : unmanaged =>
        typeof(T) == typeof(byte) ? Unsafe.As<T, byte>(ref v) : Unsafe.As<T, float>(ref v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Store<T>(float v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<float, T>(ref v);
        }

        byte b = (byte)Math.Clamp(ScalarConverter.CvRound(v), 0, 255);
        return Unsafe.As<byte, T>(ref b);
    }
}
