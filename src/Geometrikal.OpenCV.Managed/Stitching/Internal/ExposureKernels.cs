// This implementation is ported from the OpenCV stitching exposure compensators.
//
// Original source:
//   modules/stitching/src/exposure_compensate.cpp (GainCompensator::singleFeed,
//   GainCompensator::buildSimilarityMask, BlocksCompensator::apply)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Pixel loops of the exposure compensators.</summary>
internal static class ExposureKernels
{
    /// <summary>
    /// Counts the overlap pixels where both masks have their level (and the similarity mask is set),
    /// and sums the pixel norms of both images there, as <c>GainCompensator::singleFeed</c>.
    /// Sums run in row-major order in <c>double</c>.
    /// </summary>
    public static int OverlapSums(Mat<byte> img1, Mat<byte> img2, Mat<byte> mask1, byte level1, Mat<byte> mask2, byte level2, Mat<byte>? similarity, out double sum1, out double sum2)
    {
        if (img1.Channels != 3)
        {
            return OverlapSumsGray(img1, img2, mask1, level1, mask2, level2, similarity, out sum1, out sum2);
        }

        // Equal views give equal norm sequences, so the second sum equals the first.
        bool sameImage = img1.DangerousGetArray() == img2.DangerousGetArray()
            && img1.DangerousGetOffset() == img2.DangerousGetOffset() && img1.Stride == img2.Stride;
        int width = img1.Width;
        int count = 0;
        double s1 = 0, s2 = 0;
        uint levels1 = level1 * 0x01010101u, levels2 = level2 * 0x01010101u;
        for (int y = 0; y < img1.Height; y++)
        {
            ref byte r1 = ref MemoryMarshal.GetReference(img1.GetReadOnlyRowSpan(y));
            ref byte r2 = ref MemoryMarshal.GetReference(img2.GetReadOnlyRowSpan(y));
            ref byte m1 = ref MemoryMarshal.GetReference(mask1.GetReadOnlyRowSpan(y));
            ref byte m2 = ref MemoryMarshal.GetReference(mask2.GetReadOnlyRowSpan(y));
            ref byte sim = ref similarity is null ? ref Unsafe.NullRef<byte>() : ref MemoryMarshal.GetReference(similarity.GetReadOnlyRowSpan(y));
            int x = 0;
            for (; x <= width - 4; x += 4)
            {
                bool all = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref m1, x)) == levels1
                    && Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref m2, x)) == levels2
                    && (similarity is null || AllNonZero(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref sim, x))));
                if (!all)
                {
                    for (int i = x; i < x + 4; i++)
                    {
                        AddPixel(ref r1, ref r2, ref m1, level1, ref m2, level2, ref sim, similarity is not null, sameImage, i, ref count, ref s1, ref s2);
                    }

                    continue;
                }

                // Vector256.Sqrt is correctly rounded, as Math.Sqrt; the sums keep the pixel order.
                count += 4;
                Vector256<double> q1 = Vector256.Sqrt(Norms4(ref Unsafe.Add(ref r1, 3 * x)));
                s1 += q1.GetElement(0);
                s1 += q1.GetElement(1);
                s1 += q1.GetElement(2);
                s1 += q1.GetElement(3);
                if (!sameImage)
                {
                    Vector256<double> q2 = Vector256.Sqrt(Norms4(ref Unsafe.Add(ref r2, 3 * x)));
                    s2 += q2.GetElement(0);
                    s2 += q2.GetElement(1);
                    s2 += q2.GetElement(2);
                    s2 += q2.GetElement(3);
                }
            }

            for (; x < width; x++)
            {
                AddPixel(ref r1, ref r2, ref m1, level1, ref m2, level2, ref sim, similarity is not null, sameImage, x, ref count, ref s1, ref s2);
            }
        }

        sum1 = s1;
        sum2 = sameImage ? s1 : s2;
        return count;
    }

    private static bool AllNonZero(uint v) => ((v - 0x01010101u) & ~v & 0x80808080u) == 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddPixel(ref byte r1, ref byte r2, ref byte m1, byte level1, ref byte m2, byte level2, ref byte sim, bool hasSimilarity, bool sameImage, int x, ref int count, ref double s1, ref double s2)
    {
        if (Unsafe.Add(ref m1, x) != level1 || Unsafe.Add(ref m2, x) != level2 || (hasSimilarity && Unsafe.Add(ref sim, x) == 0))
        {
            return;
        }

        count++;
        s1 += Math.Sqrt(SquaredNorm(ref Unsafe.Add(ref r1, 3 * x)));
        if (!sameImage)
        {
            s2 += Math.Sqrt(SquaredNorm(ref Unsafe.Add(ref r2, 3 * x)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Norms4(ref byte p)
        => Vector256.Create(SquaredNorm(ref p), SquaredNorm(ref Unsafe.Add(ref p, 3)), SquaredNorm(ref Unsafe.Add(ref p, 6)), SquaredNorm(ref Unsafe.Add(ref p, 9)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SquaredNorm(ref byte p)
    {
        int a = p, b = Unsafe.Add(ref p, 1), c = Unsafe.Add(ref p, 2);
        return (a * a) + (b * b) + (c * c);
    }

    private static int OverlapSumsGray(Mat<byte> img1, Mat<byte> img2, Mat<byte> mask1, byte level1, Mat<byte> mask2, byte level2, Mat<byte>? similarity, out double sum1, out double sum2)
    {
        int count = 0;
        double s1 = 0, s2 = 0;
        for (int y = 0; y < img1.Height; y++)
        {
            ReadOnlySpan<byte> r1 = img1.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> r2 = img2.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> m1 = mask1.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> m2 = mask2.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> sim = similarity is null ? default : similarity.GetReadOnlyRowSpan(y);
            for (int x = 0; x < m1.Length; x++)
            {
                if (m1[x] != level1 || m2[x] != level2 || (similarity is not null && sim[x] == 0))
                {
                    continue;
                }

                count++;
                s1 += r1[x];
                s2 += r2[x];
            }
        }

        sum1 = s1;
        sum2 = s2;
        return count;
    }

    /// <summary>OpenCV <c>GainCompensator::buildSimilarityMask</c>.</summary>
    public static Mat<byte> SimilarityMask(Mat<byte> src1, Mat<byte> src2, double threshold)
    {
        var sim = new Mat<byte>(src1.Width, src1.Height, 1, clear: false);
        int cn = src1.Channels;
        for (int y = 0; y < src1.Height; y++)
        {
            ReadOnlySpan<byte> a = src1.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> b = src2.GetReadOnlyRowSpan(y);
            Span<byte> s = sim.GetRowSpan(y);
            for (int x = 0; x < s.Length; x++)
            {
                if (cn == 3)
                {
                    int i = 3 * x;
                    const float inv = 1f / 255f;
                    double d0 = ((float)a[i] - b[i]) * inv;
                    double d1 = ((float)a[i + 1] - b[i + 1]) * inv;
                    double d2 = ((float)a[i + 2] - b[i + 2]) * inv;
                    double sq = 0;
                    sq += d0 * d0;
                    sq += d1 * d1;
                    sq += d2 * d2;
                    s[x] = Math.Sqrt(sq) <= threshold ? (byte)255 : (byte)0;
                }
                else
                {
                    float diff = Math.Abs(a[x] - b[x]) / 255f;
                    s[x] = diff <= threshold ? (byte)255 : (byte)0;
                }
            }
        }

        Mat<sbyte> kernel = Morphology.GetStructuringElement(MorphShape.Rect, new Size(3, 3));
        return Morphology.Dilate(Morphology.Erode(sim, kernel), kernel);
    }

    /// <summary>
    /// <c>image = saturate_cast&lt;uchar&gt;((float)image · gain)</c> per element, as OpenCV
    /// <c>multiply</c> of an 8-bit image by a <c>float</c> map with an 8-bit destination.
    /// </summary>
    public static void MultiplyByFloatMap(Mat<byte> image, Mat<float> gains)
    {
        ParallelRows.For(image.Height, image.RowLength, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                Span<byte> r = image.GetRowSpan(y);
                ReadOnlySpan<float> g = gains.GetReadOnlyRowSpan(y);
                for (int i = 0; i < r.Length; i++)
                {
                    float v = MathF.Round(r[i] * g[i], MidpointRounding.ToEven);
                    r[i] = v >= 255f ? (byte)255 : v <= 0f ? (byte)0 : (byte)v;
                }
            }
        });
    }
}
