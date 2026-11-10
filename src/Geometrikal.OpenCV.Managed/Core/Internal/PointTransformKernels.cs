// This implementation is ported from the OpenCV per-element matrix transforms.
//
// Original sources:
//   modules/core/src/matmul.dispatch.cpp (transform, perspectiveTransform)
//   modules/core/src/matmul.simd.hpp (transform_, transform_32f, transform_64f, diagtransform_,
//                                     perspectiveTransform_)
//   modules/core/include/opencv2/core/hal/intrin_sse.hpp (v_matmuladd, v_reduce_sum4)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static Geometrikal.OpenCV.Managed.Core.Internal.LapackScalar;

/// <summary>
/// Row kernels for <c>transform</c> and <c>perspectiveTransform</c> over interleaved
/// <c>float</c>/<c>double</c> elements. The operation order reproduces the OpenCV SSE4.1
/// (128-bit) dispatch.
/// </summary>
internal static class PointTransformKernels
{
    /// <summary>
    /// <c>perspectiveTransform_</c> for <paramref name="len"/> elements. <paramref name="m"/> is the
    /// (dcn+1) × (scn+1) matrix in row-major order.
    /// </summary>
    public static void Perspective<T>(ReadOnlySpan<T> src, Span<T> dst, ReadOnlySpan<double> m, int len, int scn, int dcn)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        const double eps = FltEpsilon;
        if (scn == 2 && dcn == 2)
        {
            int start = 0;
            if (typeof(T) == typeof(float) && Avx.IsSupported)
            {
                start = Perspective2Avx(MemoryMarshal.Cast<T, float>(src), MemoryMarshal.Cast<T, float>(dst), m, len) * 2;
            }

            for (int i = start; i < len * 2; i += 2)
            {
                double x = D(src[i]), y = D(src[i + 1]);
                double w = (x * m[6]) + (y * m[7]) + m[8];
                if (Math.Abs(w) > eps)
                {
                    w = 1.0 / w;
                    dst[i] = F<T>(((x * m[0]) + (y * m[1]) + m[2]) * w);
                    dst[i + 1] = F<T>(((x * m[3]) + (y * m[4]) + m[5]) * w);
                }
                else
                {
                    dst[i] = dst[i + 1] = T.Zero;
                }
            }
        }
        else if (scn == 3 && dcn == 3)
        {
            for (int i = 0; i < len * 3; i += 3)
            {
                double x = D(src[i]), y = D(src[i + 1]), z = D(src[i + 2]);
                double w = (x * m[12]) + (y * m[13]) + (z * m[14]) + m[15];
                if (Math.Abs(w) > eps)
                {
                    w = 1.0 / w;
                    dst[i] = F<T>(((x * m[0]) + (y * m[1]) + (z * m[2]) + m[3]) * w);
                    dst[i + 1] = F<T>(((x * m[4]) + (y * m[5]) + (z * m[6]) + m[7]) * w);
                    dst[i + 2] = F<T>(((x * m[8]) + (y * m[9]) + (z * m[10]) + m[11]) * w);
                }
                else
                {
                    dst[i] = dst[i + 1] = dst[i + 2] = T.Zero;
                }
            }
        }
        else if (scn == 3 && dcn == 2)
        {
            for (int i = 0; i < len; i++)
            {
                double x = D(src[i * 3]), y = D(src[(i * 3) + 1]), z = D(src[(i * 3) + 2]);
                double w = (x * m[8]) + (y * m[9]) + (z * m[10]) + m[11];
                if (Math.Abs(w) > eps)
                {
                    w = 1.0 / w;
                    dst[i * 2] = F<T>(((x * m[0]) + (y * m[1]) + (z * m[2]) + m[3]) * w);
                    dst[(i * 2) + 1] = F<T>(((x * m[4]) + (y * m[5]) + (z * m[6]) + m[7]) * w);
                }
                else
                {
                    dst[i * 2] = dst[(i * 2) + 1] = T.Zero;
                }
            }
        }
        else
        {
            // OpenCV multiplies by w here, not by 1/w.
            for (int i = 0; i < len; i++)
            {
                ReadOnlySpan<T> s = src.Slice(i * scn, scn);
                Span<T> o = dst.Slice(i * dcn, dcn);
                int wo = dcn * (scn + 1);
                double w = m[wo + scn];
                for (int k = 0; k < scn; k++)
                {
                    w += m[wo + k] * D(s[k]);
                }

                if (Math.Abs(w) > eps)
                {
                    for (int j = 0; j < dcn; j++)
                    {
                        int mo = j * (scn + 1);
                        double acc = m[mo + scn];
                        for (int k = 0; k < scn; k++)
                        {
                            acc += m[mo + k] * D(s[k]);
                        }

                        o[j] = F<T>(acc * w);
                    }
                }
                else
                {
                    o.Clear();
                }
            }
        }
    }

    /// <summary>
    /// Four 2D <c>float</c> points per iteration in <c>double</c> precision, with the scalar
    /// operation order. Returns the number of points written.
    /// </summary>
    private static int Perspective2Avx(ReadOnlySpan<float> src, Span<float> dst, ReadOnlySpan<double> m, int len)
    {
        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        Vector256<double> m0 = Vector256.Create(m[0]), m1 = Vector256.Create(m[1]), m2 = Vector256.Create(m[2]);
        Vector256<double> m3 = Vector256.Create(m[3]), m4 = Vector256.Create(m[4]), m5 = Vector256.Create(m[5]);
        Vector256<double> m6 = Vector256.Create(m[6]), m7 = Vector256.Create(m[7]), m8 = Vector256.Create(m[8]);
        Vector256<double> eps = Vector256.Create(FltEpsilon);
        Vector256<double> one = Vector256.Create(1.0);
        int p = 0;
        for (; p <= len - 4; p += 4)
        {
            Vector256<float> v = Vector256.LoadUnsafe(ref s, (nuint)(p * 2));
            Vector256<double> lo = Avx.ConvertToVector256Double(v.GetLower());
            Vector256<double> hi = Avx.ConvertToVector256Double(v.GetUpper());

            // Lane order of x, y and the results: points 0, 2, 1, 3.
            Vector256<double> x = Avx.UnpackLow(lo, hi);
            Vector256<double> y = Avx.UnpackHigh(lo, hi);
            Vector256<double> w = ((x * m6) + (y * m7)) + m8;
            Vector256<double> mask = Avx.CompareGreaterThan(Vector256.Abs(w), eps);
            w = one / w;
            Vector256<double> rx = (((x * m0) + (y * m1)) + m2) * w;
            Vector256<double> ry = (((x * m3) + (y * m4)) + m5) * w;
            Vector256<double> r01 = Avx.UnpackLow(rx, ry) & Avx.UnpackLow(mask, mask);
            Vector256<double> r23 = Avx.UnpackHigh(rx, ry) & Avx.UnpackHigh(mask, mask);
            Vector256.Create(Avx.ConvertToVector128Single(r01), Avx.ConvertToVector128Single(r23)).StoreUnsafe(ref d, (nuint)(p * 2));
        }

        return p;
    }

    /// <summary>
    /// <c>transform</c> kernels for <paramref name="len"/> elements. <paramref name="m"/> is the
    /// dcn × (scn+1) matrix in <c>T</c> precision. <paramref name="isDiag"/> selects
    /// <c>diagtransform_</c>.
    /// </summary>
    public static void Transform<T>(ReadOnlySpan<T> src, Span<T> dst, ReadOnlySpan<T> m, int len, int scn, int dcn, bool isDiag)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (isDiag)
        {
            int cn = scn;
            for (int x = 0; x < len; x++)
            {
                ReadOnlySpan<T> s = src.Slice(x * cn, cn);
                Span<T> o = dst.Slice(x * cn, cn);
                for (int j = 0; j < cn; j++)
                {
                    int mo = j * (cn + 1);
                    o[j] = (m[mo + j] * s[j]) + m[mo + cn];
                }
            }

            return;
        }

        bool isFloat = IsFloat<T>();
        if (scn == 2 && dcn == 2)
        {
            for (int x = 0; x < len * 2; x += 2)
            {
                T v0 = src[x], v1 = src[x + 1];
                T t0 = (m[0] * v0) + (m[1] * v1) + m[2];
                T t1 = (m[3] * v0) + (m[4] * v1) + m[5];
                dst[x] = t0;
                dst[x + 1] = t1;
            }
        }
        else if (scn == 3 && dcn == 3)
        {
            // Points handled by the 128-bit loop: every point but the last (float), pairs of points (double).
            int vecPoints = isFloat ? Math.Max(0, len - 1) : len - (len & 1);
            for (int p = 0; p < len; p++)
            {
                int x = p * 3;
                T v0 = src[x], v1 = src[x + 1], v2 = src[x + 2];
                T t0, t1, t2;
                if (p < vecPoints && isFloat)
                {
                    t0 = ((v0 * m[0]) + (v1 * m[1])) + ((v2 * m[2]) + m[3]);
                    t1 = ((v0 * m[4]) + (v1 * m[5])) + ((v2 * m[6]) + m[7]);
                    t2 = ((v0 * m[8]) + (v1 * m[9])) + ((v2 * m[10]) + m[11]);
                }
                else if (p < vecPoints)
                {
                    t0 = (v0 * m[0]) + ((v1 * m[1]) + ((v2 * m[2]) + m[3]));
                    t1 = (v0 * m[4]) + ((v1 * m[5]) + ((v2 * m[6]) + m[7]));
                    t2 = (v0 * m[8]) + ((v1 * m[9]) + ((v2 * m[10]) + m[11]));
                }
                else
                {
                    t0 = (m[0] * v0) + (m[1] * v1) + (m[2] * v2) + m[3];
                    t1 = (m[4] * v0) + (m[5] * v1) + (m[6] * v2) + m[7];
                    t2 = (m[8] * v0) + (m[9] * v1) + (m[10] * v2) + m[11];
                }

                dst[x] = t0;
                dst[x + 1] = t1;
                dst[x + 2] = t2;
            }
        }
        else if (scn == 3 && dcn == 1)
        {
            for (int x = 0; x < len; x++)
            {
                int s = x * 3;
                dst[x] = (m[0] * src[s]) + (m[1] * src[s + 1]) + (m[2] * src[s + 2]) + m[3];
            }
        }
        else if (scn == 4 && dcn == 4)
        {
            for (int x = 0; x < len * 4; x += 4)
            {
                T v0 = src[x], v1 = src[x + 1], v2 = src[x + 2], v3 = src[x + 3];
                for (int j = 0; j < 4; j++)
                {
                    int mo = j * 5;
                    dst[x + j] = isFloat
                        ? (((v0 * m[mo]) + (v1 * m[mo + 1])) + ((v2 * m[mo + 2]) + (v3 * m[mo + 3]))) + m[mo + 4]
                        : (m[mo] * v0) + (m[mo + 1] * v1) + (m[mo + 2] * v2) + (m[mo + 3] * v3) + m[mo + 4];
                }
            }
        }
        else
        {
            for (int x = 0; x < len; x++)
            {
                ReadOnlySpan<T> s = src.Slice(x * scn, scn);
                Span<T> o = dst.Slice(x * dcn, dcn);
                for (int j = 0; j < dcn; j++)
                {
                    int mo = j * (scn + 1);
                    T acc = m[mo + scn];
                    for (int k = 0; k < scn; k++)
                    {
                        acc += m[mo + k] * s[k];
                    }

                    o[j] = acc;
                }
            }
        }
    }
}
