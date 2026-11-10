// This implementation is ported from the OpenCV sub-pixel rectangle sampler.
//
// Original sources:
//   modules/imgproc/src/samplers.cpp (adjustRect, getRectSubPix_Cn_, getRectSubPix_8u32f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>getRectSubPix</c> for one channel with a float patch: byte (<c>getRectSubPix_8u32f</c>)
/// and float (<c>getRectSubPix_Cn_</c>) sources. Pixels outside the image replicate the border.
/// </summary>
internal static class RectSubPix
{
    /// <summary>Fills the packed <paramref name="w"/>×<paramref name="h"/> patch centred at <paramref name="center"/>.</summary>
    public static void Extract<T>(Mat<T> src, int w, int h, Point2f center, Span<float> dst)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            Extract8u((Mat<byte>)(object)src, w, h, center, dst);
        }
        else
        {
            Generic((Mat<float>)(object)src, w, h, center, dst);
        }
    }

    private static void Extract8u(Mat<byte> src, int w, int h, Point2f center0, Span<float> dst)
    {
        float cx = center0.X - ((w - 1) * 0.5f);
        float cy = center0.Y - ((h - 1) * 0.5f);
        int ipx = ScalarConverter.CvFloor(cx);
        int ipy = ScalarConverter.CvFloor(cy);

        if (0 <= ipx && ipx + w < src.Width && 0 <= ipy && ipy + h < src.Height && w > 0 && h > 0)
        {
            float a = cx - ipx;
            float b = cy - ipy;
            a = Math.Max(a, 0.0001f);
            float a12 = a * (1f - b);
            float a22 = a * b;
            float b1 = 1f - b;
            float b2 = b;
            double s = (1.0 - a) / a;

            for (int i = 0; i < h; i++)
            {
                ReadOnlySpan<byte> r0 = src.GetReadOnlyRowSpan(ipy + i).Slice(ipx);
                ReadOnlySpan<byte> r1 = src.GetReadOnlyRowSpan(ipy + i + 1).Slice(ipx);
                Span<float> d = dst.Slice(i * w, w);
                float prev = (1 - a) * ((b1 * r0[0]) + (b2 * r1[0]));
                for (int j = 0; j < w; j++)
                {
                    float t = (a12 * r0[j + 1]) + (a22 * r1[j + 1]);
                    d[j] = prev + t;
                    prev = (float)(t * s);
                }
            }

            return;
        }

        Generic(src, w, h, center0, dst);
    }

    private static float Get<T>(Mat<T> src, int y, int x)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return ((Mat<byte>)(object)src).At(y, x);
        }

        return ((Mat<float>)(object)src).At(y, x);
    }

    private static void Generic<T>(Mat<T> src, int w, int h, Point2f center, Span<float> dst)
        where T : unmanaged
    {
        float cx = center.X - ((w - 1) * 0.5f);
        float cy = center.Y - ((h - 1) * 0.5f);
        int ipx = ScalarConverter.CvFloor(cx);
        int ipy = ScalarConverter.CvFloor(cy);

        float a = cx - ipx;
        float b = cy - ipy;
        float a11 = (1f - a) * (1f - b);
        float a12 = a * (1f - b);
        float a21 = (1f - a) * b;
        float a22 = a * b;
        float b1 = 1f - b;
        float b2 = b;

        int sw = src.Width, sh = src.Height;
        if (0 <= ipx && ipx < sw - w && 0 <= ipy && ipy < sh - h)
        {
            for (int i = 0; i < h; i++)
            {
                Span<float> d = dst.Slice(i * w, w);
                for (int j = 0; j < w; j++)
                {
                    int x = ipx + j, y = ipy + i;
                    d[j] = (Get(src, y, x) * a11) + (Get(src, y, x + 1) * a12) + (Get(src, y + 1, x) * a21) + (Get(src, y + 1, x + 1) * a22);
                }
            }

            return;
        }

        // adjustRect: (sx, sy) is the image position of the OpenCV row pointer before "- rect.x".
        int rx, rw, ry, rh;
        int sx = 0, sy = 0;
        if (ipx >= 0)
        {
            sx = ipx;
            rx = 0;
        }
        else
        {
            rx = Math.Min(-ipx, w);
        }

        if (ipx < sw - w)
        {
            rw = w;
        }
        else
        {
            rw = sw - ipx - 1;
            if (rw < 0)
            {
                sx += rw;
                rw = 0;
            }
        }

        if (ipy >= 0)
        {
            sy = ipy;
            ry = 0;
        }
        else
        {
            ry = -ipy;
        }

        if (ipy < sh - h)
        {
            rh = h;
        }
        else
        {
            rh = sh - ipy - 1;
            if (rh < 0)
            {
                sy += rh;
                rh = 0;
            }
        }

        sx -= rx;
        int row = sy;
        for (int i = 0; i < h; i++)
        {
            Span<float> d = dst.Slice(i * w, w);
            int row2 = row + 1;
            if (i < ry || i >= rh)
            {
                row2 = row;
            }

            float s0 = (Get(src, row, sx + rx) * b1) + (Get(src, row2, sx + rx) * b2);
            for (int j = 0; j < rx; j++)
            {
                d[j] = s0;
            }

            s0 = (Get(src, row, sx + rw) * b1) + (Get(src, row2, sx + rw) * b2);
            for (int j = rw; j < w; j++)
            {
                d[j] = s0;
            }

            for (int j = rx; j < rw; j++)
            {
                d[j] = (Get(src, row, sx + j) * a11) + (Get(src, row, sx + j + 1) * a12) + (Get(src, row2, sx + j) * a21) + (Get(src, row2, sx + j + 1) * a22);
            }

            if (i < rh)
            {
                row = row2;
            }
        }
    }
}
