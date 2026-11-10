// This implementation is ported from the OpenCV Bayer demosaicing.
//
// Original source:
//   modules/imgproc/src/demosaicing.cpp (Bayer2Gray_, Bayer2RGB_, Bayer2RGB_EdgeAware_T,
//   cv::demosaicing)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Bayer pattern in OpenCV naming: the colours of the second and third pixels of the second row
/// (<c>COLOR_Bayer&lt;pattern&gt;2BGR</c>).
/// </summary>
public enum BayerPattern
{
    /// <summary><c>BG</c>: the sensor pattern RGGB (first row R G, second row G B).</summary>
    BG = 0,

    /// <summary><c>GB</c>: the sensor pattern GRBG.</summary>
    GB = 1,

    /// <summary><c>RG</c>: the sensor pattern BGGR.</summary>
    RG = 2,

    /// <summary><c>GR</c>: the sensor pattern GBRG.</summary>
    GR = 3,
}

/// <summary>Bayer interpolation method.</summary>
public enum DemosaicMethod
{
    /// <summary>Bilinear interpolation (<c>COLOR_Bayer*2BGR</c>, <c>*2BGRA</c>, <c>*2GRAY</c>).</summary>
    Bilinear,

    /// <summary>Variable number of gradients (<c>COLOR_Bayer*2BGR_VNG</c>). 8-bit only.</summary>
    Vng,

    /// <summary>Edge-aware green interpolation (<c>COLOR_Bayer*2BGR_EA</c>). BGR output only.</summary>
    EdgeAware,
}

/// <summary>Bayer demosaicing, as OpenCV <c>demosaicing</c> and the Bayer codes of <c>cvtColor</c>.</summary>
/// <remarks>
/// Sources are single-channel <c>byte</c> or <c>ushort</c> images. The outer rows and columns copy
/// their inner neighbours, as OpenCV; images with fewer than 3 rows (bilinear: or 3 columns) give
/// zero colour values.
/// </remarks>
public static partial class Demosaicing
{
    private const int G2Y = 9617;
    private const int R2Y = 4899;
    private const int B2Y = 1868;

    /// <summary>Converts a Bayer image to BGR or BGRA.</summary>
    /// <param name="source">Single-channel 8-bit or 16-bit Bayer image.</param>
    /// <param name="pattern">Bayer pattern.</param>
    /// <param name="method">Interpolation method.</param>
    /// <param name="dstChannels">3 (BGR) or 4 (BGRA, bilinear only; alpha is the type maximum).</param>
    /// <returns>A new image of the source size.</returns>
    public static Mat<T> Demosaic<T>(Mat<T> source, BayerPattern pattern, DemosaicMethod method = DemosaicMethod.Bilinear, int dstChannels = 3)
        where T : unmanaged
    {
        ValidateSource(source);
        if (dstChannels is not (3 or 4) || (dstChannels == 4 && method != DemosaicMethod.Bilinear))
        {
            throw new ArgumentOutOfRangeException(nameof(dstChannels), "BGRA output needs bilinear interpolation; other methods give BGR.");
        }

        var dst = new Mat<T>(source.Width, source.Height, dstChannels);
        switch (method)
        {
            case DemosaicMethod.Bilinear:
                BayerToBgr(source, dst, pattern);
                break;
            case DemosaicMethod.EdgeAware:
                BayerToBgrEdgeAware(source, dst, pattern);
                break;
            default:
                if (typeof(T) != typeof(byte))
                {
                    throw new NotSupportedException("VNG demosaicing supports 8-bit images only.");
                }

                BayerToBgrVng(Unsafe.As<Mat<byte>>(source), Unsafe.As<Mat<byte>>(dst), pattern);
                break;
        }

        return dst;
    }

    /// <summary>Converts a Bayer image to gray (<c>COLOR_Bayer*2GRAY</c>) with fixed-point BT.601 weights.</summary>
    /// <param name="source">Single-channel 8-bit or 16-bit Bayer image.</param>
    /// <param name="pattern">Bayer pattern.</param>
    /// <returns>A new single-channel image of the source size.</returns>
    public static Mat<T> BayerToGray<T>(Mat<T> source, BayerPattern pattern)
        where T : unmanaged
    {
        ValidateSource(source);
        var dst = new Mat<T>(source.Width, source.Height);
        int bcoeff = B2Y, rcoeff = R2Y;
        bool startWithGreen = pattern is BayerPattern.GB or BayerPattern.GR;
        if (pattern is not (BayerPattern.BG or BayerPattern.GB))
        {
            (bcoeff, rcoeff) = (rcoeff, bcoeff);
        }

        int w = source.Width - 2, h = source.Height - 2;
        for (int i = 0; i < h; i++)
        {
            bool odd = (i & 1) != 0;
            GrayRow(source.GetReadOnlyRowSpan(i), source.GetReadOnlyRowSpan(i + 1), source.GetReadOnlyRowSpan(i + 2), dst.GetRowSpan(i + 1), w,
                odd ? rcoeff : bcoeff, odd ? bcoeff : rcoeff, startWithGreen ^ odd);
        }

        FillOuterRows(dst);
        return dst;
    }

    private static void ValidateSource<T>(Mat<T> source)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort))
        {
            throw new NotSupportedException("Bayer demosaicing supports 8-bit and 16-bit images only.");
        }

        if (source.Channels != 1)
        {
            throw new ArgumentException("The Bayer image must have one channel.", nameof(source));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Get<T>(ReadOnlySpan<T> row, int i)
        where T : unmanaged =>
        typeof(T) == typeof(byte) ? Unsafe.As<T, byte>(ref Unsafe.AsRef(in row[i])) : Unsafe.As<T, ushort>(ref Unsafe.AsRef(in row[i]));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Cast<T>(int v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            byte b = (byte)v;
            return Unsafe.As<byte, T>(ref b);
        }

        ushort u = (ushort)v;
        return Unsafe.As<ushort, T>(ref u);
    }

    private static T MaxValue<T>()
        where T : unmanaged =>
        typeof(T) == typeof(byte) ? Cast<T>(byte.MaxValue) : Cast<T>(ushort.MaxValue);

    private static void GrayRow<T>(ReadOnlySpan<T> r0, ReadOnlySpan<T> r1, ReadOnlySpan<T> r2, Span<T> dstRow, int w, int bcoeff, int rcoeff, bool startWithGreen)
        where T : unmanaged
    {
        // OpenCV dst0 points at pixel 1 of the row.
        var dst = dstRow[1..];
        if (w <= 0)
        {
            dstRow.Clear();
            return;
        }

        int b = 0, d = 0;
        uint t0, t1, t2;
        if (startWithGreen)
        {
            t0 = (uint)((Get(r0, b + 1) + Get(r2, b + 1)) * rcoeff);
            t1 = (uint)((Get(r1, b) + Get(r1, b + 2)) * bcoeff);
            t2 = (uint)Get(r1, b + 1) * (2 * G2Y);
            dst[d++] = Cast<T>((int)((t0 + t1 + t2 + (1u << 14)) >> 15));
            b++;
        }

        for (; b <= w - 2; b += 2, d += 2)
        {
            t0 = (uint)(Get(r0, b) + Get(r0, b + 2) + Get(r2, b) + Get(r2, b + 2)) * (uint)rcoeff;
            t1 = (uint)(Get(r0, b + 1) + Get(r1, b) + Get(r1, b + 2) + Get(r2, b + 1)) * G2Y;
            t2 = (uint)Get(r1, b + 1) * (uint)(4 * bcoeff);
            dst[d] = Cast<T>((int)((t0 + t1 + t2 + (1u << 15)) >> 16));

            t0 = (uint)((Get(r0, b + 2) + Get(r2, b + 2)) * rcoeff);
            t1 = (uint)((Get(r1, b + 1) + Get(r1, b + 3)) * bcoeff);
            t2 = (uint)Get(r1, b + 2) * (2 * G2Y);
            dst[d + 1] = Cast<T>((int)((t0 + t1 + t2 + (1u << 14)) >> 15));
        }

        if (b < w)
        {
            t0 = (uint)(Get(r0, b) + Get(r0, b + 2) + Get(r2, b) + Get(r2, b + 2)) * (uint)rcoeff;
            t1 = (uint)(Get(r0, b + 1) + Get(r1, b) + Get(r1, b + 2) + Get(r2, b + 1)) * G2Y;
            t2 = (uint)Get(r1, b + 1) * (uint)(4 * bcoeff);
            dst[d] = Cast<T>((int)((t0 + t1 + t2 + (1u << 15)) >> 16));
        }

        dstRow[0] = dstRow[1];
        dstRow[w + 1] = dstRow[w];
    }

    private static void BayerToBgr<T>(Mat<T> source, Mat<T> dst, BayerPattern pattern)
        where T : unmanaged
    {
        int blue = pattern is BayerPattern.BG or BayerPattern.GB ? -1 : 1;
        bool startWithGreen = pattern is BayerPattern.GB or BayerPattern.GR;
        int w = source.Width - 2, h = source.Height - 2;
        int dcn = dst.Channels;
        T alpha = MaxValue<T>();
        for (int i = 0; i < h; i++)
        {
            bool odd = (i & 1) != 0;
            BgrRow(source.GetReadOnlyRowSpan(i), source.GetReadOnlyRowSpan(i + 1), source.GetReadOnlyRowSpan(i + 2), dst.GetRowSpan(i + 1),
                w, dcn, odd ? -blue : blue, startWithGreen ^ odd, alpha);
        }

        FillOuterRows(dst);
    }

    private static void BgrRow<T>(ReadOnlySpan<T> r0, ReadOnlySpan<T> r1, ReadOnlySpan<T> r2, Span<T> dstRow, int w, int dcn, int blue, bool startWithGreen, T alpha)
        where T : unmanaged
    {
        if (w <= 0)
        {
            dstRow.Clear();
            if (dcn == 4)
            {
                for (int p = 3; p < dstRow.Length; p += 4)
                {
                    dstRow[p] = alpha;
                }
            }

            return;
        }

        // OpenCV dst0 points at the green element of pixel 1.
        int origin = dcn + 1;
        int b = 0, d = origin;
        int t0, t1;
        if (startWithGreen)
        {
            t0 = (Get(r0, b + 1) + Get(r2, b + 1) + 1) >> 1;
            t1 = (Get(r1, b) + Get(r1, b + 2) + 1) >> 1;
            dstRow[d - blue] = Cast<T>(t0);
            dstRow[d] = r1[b + 1];
            dstRow[d + blue] = Cast<T>(t1);
            if (dcn == 4)
            {
                dstRow[d + 2] = alpha;
            }

            b++;
            d += dcn;
        }

        for (; b <= w - 2; b += 2, d += 2 * dcn)
        {
            int c0 = (Get(r0, b) + Get(r0, b + 2) + Get(r2, b) + Get(r2, b + 2) + 2) >> 2;
            int c1 = (Get(r0, b + 1) + Get(r1, b) + Get(r1, b + 2) + Get(r2, b + 1) + 2) >> 2;
            int n0 = (Get(r0, b + 2) + Get(r2, b + 2) + 1) >> 1;
            int n1 = (Get(r1, b + 1) + Get(r1, b + 3) + 1) >> 1;
            if (dcn == 3)
            {
                if (blue > 0)
                {
                    dstRow[d - 1] = Cast<T>(c0);
                    dstRow[d] = Cast<T>(c1);
                    dstRow[d + 1] = r1[b + 1];
                    dstRow[d + 2] = Cast<T>(n0);
                    dstRow[d + 3] = r1[b + 2];
                    dstRow[d + 4] = Cast<T>(n1);
                }
                else
                {
                    dstRow[d + 1] = Cast<T>(c0);
                    dstRow[d] = Cast<T>(c1);
                    dstRow[d - 1] = r1[b + 1];
                    dstRow[d + 4] = Cast<T>(n0);
                    dstRow[d + 3] = r1[b + 2];
                    dstRow[d + 2] = Cast<T>(n1);
                }
            }
            else if (blue > 0)
            {
                dstRow[d - 1] = Cast<T>(c0);
                dstRow[d] = Cast<T>(c1);
                dstRow[d + 1] = r1[b + 1];
                dstRow[d + 2] = alpha;
                dstRow[d + 3] = Cast<T>(n0);
                dstRow[d + 4] = r1[b + 2];
                dstRow[d + 5] = Cast<T>(n1);
                dstRow[d + 6] = alpha;
            }
            else
            {
                dstRow[d - 1] = r1[b + 1];
                dstRow[d] = Cast<T>(c1);
                dstRow[d + 1] = Cast<T>(c0);
                dstRow[d + 2] = alpha;
                dstRow[d + 3] = Cast<T>(n1);
                dstRow[d + 4] = r1[b + 2];
                dstRow[d + 5] = Cast<T>(n0);
                dstRow[d + 6] = alpha;
            }
        }

        if (b < w)
        {
            t0 = (Get(r0, b) + Get(r0, b + 2) + Get(r2, b) + Get(r2, b + 2) + 2) >> 2;
            t1 = (Get(r0, b + 1) + Get(r1, b) + Get(r1, b + 2) + Get(r2, b + 1) + 2) >> 2;
            dstRow[d - blue] = Cast<T>(t0);
            dstRow[d] = Cast<T>(t1);
            dstRow[d + blue] = r1[b + 1];
            if (dcn == 4)
            {
                dstRow[d + 2] = alpha;
            }
        }

        // First and last pixels copy their neighbours.
        int last = origin + (w * dcn);
        for (int k = 0; k < dcn; k++)
        {
            dstRow[k] = dstRow[dcn + k];
            dstRow[last - 1 + k] = dstRow[last - 1 - dcn + k];
        }
    }

    private static void BayerToBgrEdgeAware<T>(Mat<T> source, Mat<T> dst, BayerPattern pattern)
        where T : unmanaged
    {
        if (source.Width <= 2 || source.Height <= 2)
        {
            return;
        }

        int w = source.Width - 2, h = source.Height - 2;
        bool startWithGreen = pattern is BayerPattern.GB or BayerPattern.GR;
        int blue = pattern is BayerPattern.GB or BayerPattern.BG ? 1 : 0;
        for (int y = 0; y < h; y++)
        {
            int odd = y & 1;
            EdgeAwareRow(source.GetReadOnlyRowSpan(y), source.GetReadOnlyRowSpan(y + 1), source.GetReadOnlyRowSpan(y + 2), dst.GetRowSpan(y + 1),
                w, blue ^ odd, startWithGreen ^ (odd != 0));
        }

        FillOuterRows(dst);
    }

    private static void EdgeAwareRow<T>(ReadOnlySpan<T> up, ReadOnlySpan<T> mid, ReadOnlySpan<T> down, Span<T> dst, int w, int blue, bool startWithGreen)
        where T : unmanaged
    {
        const int dcn = 3;
        int x = 1, s = 1, d = dcn;
        if (startWithGreen)
        {
            dst[d + (blue << 1)] = Cast<T>((Get(up, s) + Get(down, s) + 1) >> 1);
            dst[d + 1] = mid[s];
            dst[d + 2 - (blue << 1)] = Cast<T>((Get(mid, s - 1) + Get(mid, s + 1) + 1) >> 1);
            d += dcn;
            s++;
            x++;
        }

        for (; x < w; x += 2, s += 2, d += 2 * dcn)
        {
            int diag = (Get(up, s - 1) + Get(up, s + 1) + Get(down, s - 1) + Get(down, s + 1) + 2) >> 2;
            int green = EdgeGreen(up, mid, down, s);
            int horizontal = (Get(mid, s) + Get(mid, s + 2) + 1) >> 1;
            int vertical = (Get(up, s + 1) + Get(down, s + 1) + 1) >> 1;
            if (blue != 0)
            {
                dst[d] = mid[s];
                dst[d + 1] = Cast<T>(green);
                dst[d + 2] = Cast<T>(diag);
                dst[d + 3] = Cast<T>(horizontal);
                dst[d + 4] = mid[s + 1];
                dst[d + 5] = Cast<T>(vertical);
            }
            else
            {
                dst[d] = Cast<T>(diag);
                dst[d + 1] = Cast<T>(green);
                dst[d + 2] = mid[s];
                dst[d + 3] = Cast<T>(vertical);
                dst[d + 4] = mid[s + 1];
                dst[d + 5] = Cast<T>(horizontal);
            }
        }

        if (x <= w)
        {
            dst[d + (blue << 1)] = Cast<T>((Get(up, s - 1) + Get(up, s + 1) + Get(down, s - 1) + Get(down, s + 1) + 2) >> 2);
            dst[d + 1] = Cast<T>(EdgeGreen(up, mid, down, s));
            dst[d + 2 - (blue << 1)] = mid[s];
            d += dcn;
        }

        // OpenCV writes pixel w + 1 from pixel w, and pixel 0 from pixel 1 through the previous row pointer.
        for (int i = 0; i < dcn; i++)
        {
            dst[d + i] = dst[d - dcn + i];
            dst[i] = dst[dcn + i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int EdgeGreen<T>(ReadOnlySpan<T> up, ReadOnlySpan<T> mid, ReadOnlySpan<T> down, int s)
        where T : unmanaged
    {
        int l = Get(mid, s - 1), r = Get(mid, s + 1), u = Get(up, s), dn = Get(down, s);
        return (Math.Abs(l - r) > Math.Abs(dn - u) ? (dn + u + 1) : (l + r + 1)) >> 1;
    }

    private static void FillOuterRows<T>(Mat<T> dst)
        where T : unmanaged
    {
        int h = dst.Height;
        if (h > 2)
        {
            dst.GetReadOnlyRowSpan(1).CopyTo(dst.GetRowSpan(0));
            dst.GetReadOnlyRowSpan(h - 2).CopyTo(dst.GetRowSpan(h - 1));
        }
        else
        {
            dst.GetRowSpan(0).Clear();
            dst.GetRowSpan(h - 1).Clear();
        }
    }
}
