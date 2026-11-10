// This implementation is ported from the OpenCV float reductions.
//
// Original sources:
//   modules/core/src/sum.simd.hpp (sum_, Sum_SIMD<float, double>)
//   modules/core/src/mean.dispatch.cpp (mean, meanStdDev)
//   modules/core/src/mean.simd.hpp (sumsqr_)
//   modules/core/src/norm.dispatch.cpp, norm.simd.hpp (norm, NormL2_SIMD<float, double>)
//   modules/core/src/matmul.dispatch.cpp, matmul.simd.hpp (Mat::dot, dotProd_32f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>sum</c>, <c>mean</c>, <c>meanStdDev</c> and <c>norm(L2, L2SQR)</c> of float matrices
/// with the OpenCV 5.0 summation order of the 128-bit SIMD build. A continuous matrix is one
/// plane; a non-continuous matrix is one plane per row, as <c>NAryMatIterator</c>.
/// </summary>
/// <remarks>
/// <see cref="Reductions"/> sums floats per row and adds the rows, so its last bits can differ
/// from OpenCV. The alignment functions use these exact forms because their results amplify
/// the difference.
/// </remarks>
internal static class FloatReductions
{
    /// <summary>OpenCV <c>sum</c>: per-channel sums (channels above <c>cn</c> are zero).</summary>
    public static double[] Sum(Mat<float> src)
    {
        int cn = src.Channels;
        var dst = new double[4];
        if (src.IsContinuous)
        {
            SumPlane(src.GetReadOnlySpan()[..(src.Width * src.Height * cn)], cn, dst);
        }
        else
        {
            for (int y = 0; y < src.Height; y++)
            {
                SumPlane(src.GetReadOnlyRowSpan(y), cn, dst);
            }
        }

        return dst;
    }

    /// <summary>OpenCV <c>mean</c> without a mask: <c>sum * (1 / N)</c>.</summary>
    public static Scalar Mean(Mat<float> src)
    {
        double[] s = Sum(src);
        double scale = 1.0 / ((double)src.Width * src.Height);
        return new Scalar(s[0] * scale, s[1] * scale, s[2] * scale, s[3] * scale);
    }

    /// <summary>
    /// OpenCV <c>meanStdDev</c>: sequential double sums of <c>v</c> and <c>v·v</c> in pixel order.
    /// Integer sources give exact sums, as the OpenCV block sums.
    /// </summary>
    public static void MeanStdDev<T>(Mat<T> src, Mat<byte>? mask, out Scalar mean, out Scalar stdDev)
        where T : unmanaged
    {
        int cn = src.Channels;
        Span<double> s = stackalloc double[4];
        Span<double> sq = stackalloc double[4];
        long n = 0;
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<T> row = src.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> m = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            for (int x = 0; x < src.Width; x++)
            {
                if (mask is not null && m[x] == 0)
                {
                    continue;
                }

                n++;
                for (int k = 0; k < cn; k++)
                {
                    double v = Saturate.ToDouble(row[(x * cn) + k]);
                    s[k] += v;
                    sq[k] += v * v;
                }
            }
        }

        double scale = n != 0 ? 1.0 / n : 0.0;
        for (int k = 0; k < cn; k++)
        {
            s[k] *= scale;
            sq[k] = Math.Sqrt(Math.Max((sq[k] * scale) - (s[k] * s[k]), 0.0));
        }

        mean = new Scalar(s[0], s[1], s[2], s[3]);
        stdDev = new Scalar(sq[0], sq[1], sq[2], sq[3]);
    }

    /// <summary>
    /// OpenCV <c>Mat::dot</c> for float matrices of the same size: one plane when both are
    /// continuous, one plane per row otherwise.
    /// </summary>
    public static double Dot(Mat<float> a, Mat<float> b)
    {
        if (a.IsContinuous && b.IsContinuous)
        {
            int len = a.Width * a.Height * a.Channels;
            return DotProd32f(a.GetReadOnlySpan()[..len], b.GetReadOnlySpan()[..len]);
        }

        double r = 0;
        for (int y = 0; y < a.Height; y++)
        {
            r += DotProd32f(a.GetReadOnlyRowSpan(y), b.GetReadOnlyRowSpan(y));
        }

        return r;
    }

    /// <summary>
    /// OpenCV <c>dotProd_32f</c>: blocks of 8192 elements with four 4-lane float accumulators
    /// (multiply, then add), reduced to double per block; a double tail in groups of four.
    /// </summary>
    public static double DotProd32f(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = a.Length;
        double r = 0;
        int i = 0;
        int len0 = len & -4;
        while (i < len0)
        {
            int blockSize = Math.Min(len0 - i, 1 << 13);
            ReadOnlySpan<float> pa = a.Slice(i, blockSize), pb = b.Slice(i, blockSize);
            Vector128<float> s0 = Vector128<float>.Zero, s1 = s0, s2 = s0, s3 = s0;
            int j = 0;
            for (; j <= blockSize - 16; j += 16)
            {
                s0 = (Vector128.Create(pa.Slice(j, 4)) * Vector128.Create(pb.Slice(j, 4))) + s0;
                s1 = (Vector128.Create(pa.Slice(j + 4, 4)) * Vector128.Create(pb.Slice(j + 4, 4))) + s1;
                s2 = (Vector128.Create(pa.Slice(j + 8, 4)) * Vector128.Create(pb.Slice(j + 8, 4))) + s2;
                s3 = (Vector128.Create(pa.Slice(j + 12, 4)) * Vector128.Create(pb.Slice(j + 12, 4))) + s3;
            }

            s0 += (s1 + s2) + s3;
            for (; j <= blockSize - 4; j += 4)
            {
                s0 = (Vector128.Create(pa.Slice(j, 4)) * Vector128.Create(pb.Slice(j, 4))) + s0;
            }

            r += (s0.GetElement(0) + s0.GetElement(2)) + (s0.GetElement(1) + s0.GetElement(3));
            i += blockSize;
        }

        double tail = 0;
        for (; i <= len - 4; i += 4)
        {
            tail += ((double)a[i] * b[i]) + ((double)a[i + 1] * b[i + 1]) + ((double)a[i + 2] * b[i + 2]) + ((double)a[i + 3] * b[i + 3]);
        }

        for (; i < len; i++)
        {
            tail += (double)a[i] * b[i];
        }

        return r + tail;
    }

    /// <summary>OpenCV <c>norm(src, NORM_L2SQR)</c> without a mask.</summary>
    public static double NormL2Sqr(Mat<float> src)
    {
        if (src.IsContinuous)
        {
            return NormL2SqrPlane(src.GetReadOnlySpan()[..(src.Width * src.Height * src.Channels)]);
        }

        double result = 0;
        for (int y = 0; y < src.Height; y++)
        {
            result += NormL2SqrPlane(src.GetReadOnlyRowSpan(y));
        }

        return result;
    }

    /// <summary><c>NormL2_SIMD&lt;float, double&gt;</c> with 2-lane double accumulators.</summary>
    private static double NormL2SqrPlane(ReadOnlySpan<float> p)
    {
        int n = p.Length, j = 0;
        double s = 0;
        double r00a = 0, r00b = 0, r01a = 0, r01b = 0, r10a = 0, r10b = 0, r11a = 0, r11b = 0;
        for (; j <= n - 8; j += 8)
        {
            double v0 = p[j], v1 = p[j + 1], v2 = p[j + 2], v3 = p[j + 3];
            double v4 = p[j + 4], v5 = p[j + 5], v6 = p[j + 6], v7 = p[j + 7];
            r00a += v0 * v0;
            r00b += v1 * v1;
            r01a += v2 * v2;
            r01b += v3 * v3;
            r10a += v4 * v4;
            r10b += v5 * v5;
            r11a += v6 * v6;
            r11b += v7 * v7;
        }

        double la = ((r00a + r01a) + r10a) + r11a;
        double lb = ((r00b + r01b) + r10b) + r11b;
        s += la + lb;
        for (; j < n; j++)
        {
            double v = p[j];
            s += v * v;
        }

        return s;
    }

    /// <summary>OpenCV <c>sum_&lt;float, double, float&gt;</c> for one plane of <c>len</c> pixels.</summary>
    private static void SumPlane(ReadOnlySpan<float> p, int cn, double[] dst)
    {
        int len = p.Length / cn;
        int i0 = 0;
        if (cn <= 3)
        {
            // Sum_SIMD<float, double>: six 2-lane accumulators over blocks of 12 elements.
            int total = len * cn, x = 0;
            double s0a = 0, s0b = 0, s1a = 0, s1b = 0, s2a = 0, s2b = 0;
            for (; x <= total - 12; x += 12)
            {
                s0a += p[x];
                s0b += p[x + 1];
                s1a += p[x + 2];
                s1b += p[x + 3];
                s2a += p[x + 4];
                s2b += p[x + 5];
                s0a += p[x + 6];
                s0b += p[x + 7];
                s1a += p[x + 8];
                s1b += p[x + 9];
                s2a += p[x + 10];
                s2b += p[x + 11];
            }

            if (cn == 1)
            {
                dst[0] += ((s0a + s1a) + s2a) + ((s0b + s1b) + s2b);
            }
            else if (cn == 2)
            {
                dst[0] += ((s0a + s1a) + s2a) + 0.0;
                dst[1] += 0.0 + ((s0b + s1b) + s2b);
            }
            else
            {
                dst[0] += s0a + s1b;
                dst[1] += s2a + s0b;
                dst[2] += s1a + s2b;
            }

            i0 = x / cn;
        }

        int i = i0;
        int baseIndex = i0 * cn;
        int k = cn % 4;
        if (k == 1)
        {
            double s = dst[0];
            int idx = baseIndex;
            for (; i <= len - 4; i += 4, idx += cn * 4)
            {
                s += (double)(p[idx] + p[idx + cn] + p[idx + (cn * 2)] + p[idx + (cn * 3)]);
            }

            for (; i < len; i++, idx += cn)
            {
                s += p[idx];
            }

            dst[0] = s;
        }
        else if (k == 2 || k == 3)
        {
            for (int idx = baseIndex; i < len; i++, idx += cn)
            {
                for (int c = 0; c < k; c++)
                {
                    dst[c] += p[idx + c];
                }
            }
        }

        for (; k < cn; k += 4)
        {
            for (int ii = i0, idx = baseIndex + k; ii < len; ii++, idx += cn)
            {
                dst[k] += p[idx];
                dst[k + 1] += p[idx + 1];
                dst[k + 2] += p[idx + 2];
                dst[k + 3] += p[idx + 3];
            }
        }
    }

    /// <summary>
    /// <c>convertTo(dst, CV_32F, alpha)</c> for float: a copy when <c>alpha</c> is 1 within
    /// <c>DBL_EPSILON</c>, otherwise <c>x·(float)alpha + 0f</c> in float. <paramref name="dst"/>
    /// can be <paramref name="src"/>.
    /// </summary>
    public static void ConvertScale(Mat<float> src, double alpha, Mat<float> dst)
    {
        if (Math.Abs(alpha - 1) < 2.220446049250313E-16)
        {
            if (!ReferenceEquals(src, dst))
            {
                src.CopyTo(dst);
            }

            return;
        }

        float a = (float)alpha;
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<float> s = src.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            for (int i = 0; i < s.Length; i++)
            {
                d[i] = (s[i] * a) + 0f;
            }
        }
    }
}
