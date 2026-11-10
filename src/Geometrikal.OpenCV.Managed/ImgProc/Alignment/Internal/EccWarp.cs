// This implementation is ported from the OpenCV 5.0 float-coordinate warp kernels.
//
// Original sources:
//   modules/imgproc/src/warp_kernels.simd.hpp (warpAffineLinearInvoker_32FC1/32FC3,
//   warpPerspectiveLinearInvoker_32FC1/32FC3, warpAffineNearestInvoker_8UC1,
//   warpPerspectiveNearestInvoker_8UC1)
//   modules/imgproc/src/warp_common.scalar.hpp, warp_common.vector.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// The OpenCV 5.0 warp kernels that <c>findTransformECC</c> calls (<c>WARP_INVERSE_MAP</c>,
/// constant border 0): bilinear for float images with 1 or 3 channels, nearest for byte masks.
/// </summary>
/// <remarks>
/// <para>OpenCV 5.0 computes the source coordinates in single precision without the 1/32 pixel
/// quantisation of the table path that <see cref="Geometry"/> ports. The 128-bit build
/// processes blocks of 8 destination pixels with <c>M0·x + (y·M1 + M2)</c> and the remaining
/// pixels of a row with <c>(x·M0 + y·M1) + M2</c>; the port uses the same split. The vector
/// multiply-add is not fused in this build.</para>
/// <para>The bilinear value is <c>v0 + β·(v1 - v0)</c> with <c>v0 = p00 + α·(p01 - p00)</c>
/// and <c>v1 = p10 + α·(p11 - p10)</c>; taps outside the source are 0.</para>
/// </remarks>
internal static class EccWarp
{
    private const int BlockWidth = 8;

    public static void Linear(Mat<float> src, Mat<float> dst, Mat<float> map, bool perspective)
    {
        float[] m = Coefficients(map, perspective);
        ParallelRows.For(dst.Height, (long)dst.Width * dst.Channels * 8, (y0, y1) => LinearRows(src, dst, m, perspective, y0, y1));
    }

    public static void Nearest(Mat<byte> src, Mat<byte> dst, Mat<float> map, bool perspective)
    {
        float[] m = Coefficients(map, perspective);
        for (int y = 0; y < dst.Height; y++)
        {
            Span<byte> d = dst.GetRowSpan(y);
            for (int x = 0; x < dst.Width; x++)
            {
                (float sx, float sy) = Map(m, perspective, x, y, dst.Width);
                int ix = RoundHalfEven(sx), iy = RoundHalfEven(sy);
                d[x] = (uint)ix < (uint)src.Width && (uint)iy < (uint)src.Height ? src.At(iy, ix) : (byte)0;
            }
        }
    }

    private static float[] Coefficients(Mat<float> map, bool perspective)
    {
        var m = new float[9];
        for (int i = 0; i < (perspective ? 9 : 6); i++)
        {
            m[i] = map.At(i / 3, i % 3);
        }

        return m;
    }

    private static (float X, float Y) Map(float[] m, bool perspective, int x, int y, int width)
    {
        bool block = x < width - (width % BlockWidth);
        float fx = x, fy = y;
        if (block)
        {
            float mx = (fy * m[1]) + m[2];
            float my = (fy * m[4]) + m[5];
            float sx = (m[0] * fx) + mx;
            float sy = (m[3] * fx) + my;
            if (perspective)
            {
                float mw = (fy * m[7]) + m[8];
                float sw = (m[6] * fx) + mw;
                sx /= sw;
                sy /= sw;
            }

            return (sx, sy);
        }

        if (perspective)
        {
            float w = (fx * m[6]) + (fy * m[7]) + m[8];
            return (((fx * m[0]) + (fy * m[1]) + m[2]) / w, ((fx * m[3]) + (fy * m[4]) + m[5]) / w);
        }

        return ((fx * m[0]) + (fy * m[1]) + m[2], (fx * m[3]) + (fy * m[4]) + m[5]);
    }

    private static void LinearRows(Mat<float> src, Mat<float> dst, float[] m, bool perspective, int y0, int y1)
    {
        int cn = src.Channels, sw = src.Width, sh = src.Height;
        float[] buf = src.DangerousGetArray();
        int offset = src.DangerousGetOffset(), stride = src.Stride;
        for (int y = y0; y < y1; y++)
        {
            Span<float> d = dst.GetRowSpan(y);
            for (int x = 0; x < dst.Width; x++)
            {
                (float sx, float sy) = Map(m, perspective, x, y, dst.Width);
                int ix = Floor(sx), iy = Floor(sy);
                if ((uint)(ix + 1) >= (uint)(sw + 1) || (uint)(iy + 1) >= (uint)(sh + 1))
                {
                    d.Slice(x * cn, cn).Clear();
                    continue;
                }

                float a = sx - ix, b = sy - iy;
                bool inner = (uint)ix < (uint)(sw - 1) && (uint)iy < (uint)(sh - 1);
                int i00 = offset + (iy * stride) + (ix * cn);
                for (int k = 0; k < cn; k++)
                {
                    float p00, p01, p10, p11;
                    if (inner)
                    {
                        p00 = buf[i00 + k];
                        p01 = buf[i00 + cn + k];
                        p10 = buf[i00 + stride + k];
                        p11 = buf[i00 + stride + cn + k];
                    }
                    else
                    {
                        p00 = Tap(src, ix, iy, k);
                        p01 = Tap(src, ix + 1, iy, k);
                        p10 = Tap(src, ix, iy + 1, k);
                        p11 = Tap(src, ix + 1, iy + 1, k);
                    }

                    float v0 = p00 + (a * (p01 - p00));
                    float v1 = p10 + (a * (p11 - p10));
                    d[(x * cn) + k] = v0 + (b * (v1 - v0));
                }
            }
        }
    }

    private static float Tap(Mat<float> src, int x, int y, int k)
        => (uint)x < (uint)src.Width && (uint)y < (uint)src.Height ? src.At(y, x, k) : 0f;

    private static int Floor(float v) => float.IsNaN(v) ? int.MinValue : (int)Math.Clamp(MathF.Floor(v), int.MinValue, int.MaxValue);

    private static int RoundHalfEven(float v) => float.IsNaN(v) ? int.MinValue : (int)Math.Clamp(MathF.Round(v, MidpointRounding.ToEven), int.MinValue, int.MaxValue);
}
