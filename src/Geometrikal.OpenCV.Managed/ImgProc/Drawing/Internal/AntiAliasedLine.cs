// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

using static Geometrikal.OpenCV.Managed.ImgProc.Internal.DrawingConstants;

/// <summary>
/// Port of the OpenCV <c>LineAA</c> rasteriser (drawing.cpp:301-632). The
/// rasteriser draws a 3-pixel-wide Gaussian-filtered line with sub-pixel
/// end-point correction.
///
/// <para>The anti-aliased path applies only to 8-bit images with 1, 3 or 4
/// channels. For other images, <see cref="LineAA{T}"/> draws an 8-connected
/// line between the truncated end points, as OpenCV does.</para>
/// </summary>
internal static class AntiAliasedLine
{
    /// <summary>OpenCV <c>LINE_AA</c>.</summary>
    public const int LineAa = 16;

    /// <summary>Slope correction table, indexed by 5-bit slope (OpenCV <c>SlopeCorrTable</c>).</summary>
    private static ReadOnlySpan<byte> SlopeCorrTable =>
    [
        181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201,
        203, 206, 209, 211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254,
    ];

    /// <summary>Gaussian filter weights (OpenCV <c>FilterTable</c>).</summary>
    private static readonly int[] FilterTable =
    [
        168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254,
        254, 254, 252, 249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168,
        158, 149, 140, 131, 122, 114, 105, 97, 89, 82, 75, 68, 62, 56, 50, 45,
        40, 36, 32, 28, 25, 22, 19, 16, 14, 12, 11, 9, 8, 7, 5, 5,
    ];

    /// <summary>
    /// Draws an anti-aliased line between two <c>XY_SHIFT</c> fixed-point
    /// end points.
    /// </summary>
    public static void LineAA<T>(Mat<T> img, PointL pt1, PointL pt2, ReadOnlySpan<T> color)
        where T : unmanaged
    {
        int nch = img.Channels;
        if (typeof(T) != typeof(byte) || !(nch == 1 || nch == 3 || nch == 4))
        {
            LinePrimitives.Line(
                img,
                new Point((int)(pt1.X >> XY_SHIFT), (int)(pt1.Y >> XY_SHIFT)),
                new Point((int)(pt2.X >> XY_SHIFT), (int)(pt2.Y >> XY_SHIFT)),
                color,
                8);
            return;
        }

        Mat<byte> bimg = Unsafe.As<Mat<byte>>(img);
        ReadOnlySpan<byte> bcolor = System.Runtime.InteropServices.MemoryMarshal.Cast<T, byte>(color);
        Draw(bimg, pt1, pt2, bcolor);
    }

    private static void Draw(Mat<byte> img, PointL p1, PointL p2, ReadOnlySpan<byte> color)
    {
        int nch = img.Channels;
        int width0 = img.Width;
        int height0 = img.Height;

        long pt1x = p1.X, pt1y = p1.Y, pt2x = p2.X, pt2y = p2.Y;
        if (!LinePrimitives.ClipLine((long)width0 << XY_SHIFT, (long)height0 << XY_SHIFT,
                ref pt1x, ref pt1y, ref pt2x, ref pt2y))
        {
            return;
        }

        long dx = pt2x - pt1x;
        long dy = pt2y - pt1y;

        long j = dx < 0 ? -1 : 0;
        long ax = (dx ^ j) - j;
        long i = dy < 0 ? -1 : 0;
        long ay = (dy ^ i) - i;

        long xStep, yStep;
        int ecount;
        int slope;
        bool horizontal = ax > ay;

        if (horizontal)
        {
            dy = (dy ^ j) - j;
            pt1x ^= pt2x & j;
            pt2x ^= pt1x & j;
            pt1x ^= pt2x & j;
            pt1y ^= pt2y & j;
            pt2y ^= pt1y & j;
            pt1y ^= pt2y & j;

            xStep = XY_ONE;
            yStep = (long)((ulong)dy << XY_SHIFT) / (ax | 1);
            pt2x += XY_ONE;
            ecount = (int)((pt2x >> XY_SHIFT) - (pt1x >> XY_SHIFT));
            j = -(pt1x & (XY_ONE - 1));
            pt1y += ((yStep * j) >> XY_SHIFT) + (XY_ONE >> 1);
            slope = (int)((yStep >> (XY_SHIFT - 5)) & 0x3f);
            slope ^= yStep < 0 ? 0x3f : 0;

            i = (pt1x >> (XY_SHIFT - 7)) & 0x78;
            j = (pt2x >> (XY_SHIFT - 7)) & 0x78;
        }
        else
        {
            dx = (dx ^ i) - i;
            pt1x ^= pt2x & i;
            pt2x ^= pt1x & i;
            pt1x ^= pt2x & i;
            pt1y ^= pt2y & i;
            pt2y ^= pt1y & i;
            pt1y ^= pt2y & i;

            xStep = (long)((ulong)dx << XY_SHIFT) / (ay | 1);
            yStep = XY_ONE;
            pt2y += XY_ONE;
            ecount = (int)((pt2y >> XY_SHIFT) - (pt1y >> XY_SHIFT));
            j = -(pt1y & (XY_ONE - 1));
            pt1x += ((xStep * j) >> XY_SHIFT) + (XY_ONE >> 1);
            slope = (int)((xStep >> (XY_SHIFT - 5)) & 0x3f);
            slope ^= xStep < 0 ? 0x3f : 0;

            i = (pt1y >> (XY_SHIFT - 7)) & 0x78;
            j = (pt2y >> (XY_SHIFT - 7)) & 0x78;
        }

        slope = (slope & 0x20) != 0 ? 0x100 : SlopeCorrTable[slope];

        Span<int> epTable = stackalloc int[9];
        {
            int t0 = slope << 7;
            int t1 = ((0x78 - (int)i) | 4) * slope;
            int t2 = ((int)j | 4) * slope;

            epTable[0] = 0;
            epTable[8] = slope;
            epTable[1] = epTable[3] = (int)(((((j - i) & 0x78) | 4) * slope >> 8) & 0x1ff);
            epTable[2] = (t1 >> 8) & 0x1ff;
            epTable[4] = (int)(((((j - i) + 0x80) | 4) * slope >> 8) & 0x1ff);
            epTable[5] = ((t1 + t0) >> 8) & 0x1ff;
            epTable[6] = (t2 >> 8) & 0x1ff;
            epTable[7] = ((t2 + t0) >> 8) & 0x1ff;
        }

        int cb = color[0];
        int cg = nch > 1 ? color[1] : 0;
        int cr = nch > 2 ? color[2] : 0;
        int ca = nch > 3 ? color[3] : 0;
        var walk = new LineWalk(pt1x, pt1y, xStep, yStep, ecount, epTable, cb, cg, cr, ca);
        switch (nch)
        {
            case 1:
                Rasterise<Nch1>(img, horizontal, walk);
                break;
            case 3:
                Rasterise<Nch3>(img, horizontal, walk);
                break;
            default:
                Rasterise<Nch4>(img, horizontal, walk);
                break;
        }
    }

    private static void Rasterise<TNch>(Mat<byte> img, bool horizontal, in LineWalk s)
        where TNch : struct, INch
    {
        int nch = TNch.Count;
        int width0 = img.Width;
        int height0 = img.Height;
        nint stride = img.Stride;
        ref byte data = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(img.DangerousGetArray()), img.DangerousGetOffset());
        ref int filter = ref MemoryMarshal.GetArrayDataReference(FilterTable);
        ReadOnlySpan<int> epTable = s.EpTable;
        int cb = s.Cb, cg = s.Cg, cr = s.Cr, ca = s.Ca;
        int ecount = s.ECount;
        int scount = 0;

        if (horizontal)
        {
            long pt1y = s.Pt1Y, yStep = s.YStep;
            int x = (int)(s.Pt1X >> XY_SHIFT);
            for (; ecount >= 0; x++, pt1y += yStep, scount++, ecount--)
            {
                if ((uint)x >= (uint)width0)
                {
                    continue;
                }

                int y = (int)((pt1y >> XY_SHIFT) - 1);
                int epCorr = epTable[EpIndex(scount, ecount)];
                int dist = (int)((pt1y >> (XY_SHIFT - 5)) & 31);
                nint px = ((nint)y * stride) + (x * nch);

                int a = (epCorr * Unsafe.Add(ref filter, dist + 32) >> 8) & 0xff;
                if ((uint)y < (uint)height0)
                {
                    Blend<TNch>(ref Unsafe.Add(ref data, px), a, cb, cg, cr, ca);
                }

                a = (epCorr * Unsafe.Add(ref filter, dist) >> 8) & 0xff;
                if ((uint)(y + 1) < (uint)height0)
                {
                    Blend<TNch>(ref Unsafe.Add(ref data, px + stride), a, cb, cg, cr, ca);
                }

                a = (epCorr * Unsafe.Add(ref filter, 63 - dist) >> 8) & 0xff;
                if ((uint)(y + 2) < (uint)height0)
                {
                    Blend<TNch>(ref Unsafe.Add(ref data, px + (2 * stride)), a, cb, cg, cr, ca);
                }
            }
        }
        else
        {
            long pt1x = s.Pt1X, xStep = s.XStep;
            int y = (int)(s.Pt1Y >> XY_SHIFT);
            for (; ecount >= 0; y++, pt1x += xStep, scount++, ecount--)
            {
                if ((uint)y >= (uint)height0)
                {
                    continue;
                }

                int x = (int)((pt1x >> XY_SHIFT) - 1);
                int epCorr = epTable[EpIndex(scount, ecount)];
                int dist = (int)((pt1x >> (XY_SHIFT - 5)) & 31);
                nint px = ((nint)y * stride) + (x * nch);

                int a = (epCorr * Unsafe.Add(ref filter, dist + 32) >> 8) & 0xff;
                if ((uint)x < (uint)width0)
                {
                    Blend<TNch>(ref Unsafe.Add(ref data, px), a, cb, cg, cr, ca);
                }

                a = (epCorr * Unsafe.Add(ref filter, dist) >> 8) & 0xff;
                if ((uint)(x + 1) < (uint)width0)
                {
                    Blend<TNch>(ref Unsafe.Add(ref data, px + nch), a, cb, cg, cr, ca);
                }

                a = (epCorr * Unsafe.Add(ref filter, 63 - dist) >> 8) & 0xff;
                if ((uint)(x + 2) < (uint)width0)
                {
                    Blend<TNch>(ref Unsafe.Add(ref data, px + (2 * nch)), a, cb, cg, cr, ca);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int EpIndex(int scount, int ecount) =>
        ((((scount >= 2) ? 1 : 0) + 1) & (scount | 2)) * 3 +
        ((((ecount >= 2) ? 1 : 0) + 1) & (ecount | 2));

    /// <summary>Applies the OpenCV <c>ICV_PUT_POINT</c> double blend to one pixel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Blend<TNch>(ref byte px, int a, int cb, int cg, int cr, int ca)
        where TNch : struct, INch
    {
        px = BlendChannel(px, cb, a);
        if (TNch.Count == 1)
        {
            return;
        }

        Unsafe.Add(ref px, 1) = BlendChannel(Unsafe.Add(ref px, 1), cg, a);
        Unsafe.Add(ref px, 2) = BlendChannel(Unsafe.Add(ref px, 2), cr, a);
        if (TNch.Count == 4)
        {
            Unsafe.Add(ref px, 3) = BlendChannel(Unsafe.Add(ref px, 3), ca, a);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte BlendChannel(int v, int c, int a)
    {
        v += ((c - v) * a + 127) >> 8;
        v += ((c - v) * a + 127) >> 8;
        return (byte)v;
    }

    private interface INch
    {
        static abstract int Count { get; }
    }

    private readonly struct Nch1 : INch
    {
        public static int Count => 1;
    }

    private readonly struct Nch3 : INch
    {
        public static int Count => 3;
    }

    private readonly struct Nch4 : INch
    {
        public static int Count => 4;
    }

    private readonly ref struct LineWalk(long pt1x, long pt1y, long xStep, long yStep, int eCount, ReadOnlySpan<int> epTable, int cb, int cg, int cr, int ca)
    {
        public long Pt1X { get; } = pt1x;

        public long Pt1Y { get; } = pt1y;

        public long XStep { get; } = xStep;

        public long YStep { get; } = yStep;

        public int ECount { get; } = eCount;

        public ReadOnlySpan<int> EpTable { get; } = epTable;

        public int Cb { get; } = cb;

        public int Cg { get; } = cg;

        public int Cr { get; } = cr;

        public int Ca { get; } = ca;
    }
}
