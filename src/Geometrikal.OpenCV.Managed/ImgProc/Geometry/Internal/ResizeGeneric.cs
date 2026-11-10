// This implementation is ported from the OpenCV separable resize.
//
// Original source:
//   modules/imgproc/src/resize.cpp (resizeGeneric_Invoker, HResizeLinear, HResizeCubic,
//     HResizeLanczos4, VResizeLinear, VResizeCubic, VResizeLanczos4 and their vector variants)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Horizontal and vertical passes of one element type of the separable resize.</summary>
internal interface IResizeRows<T, TWork>
    where T : unmanaged
    where TWork : unmanaged
{
    /// <summary>Resizes one source row into a row of <c>dstWidth * cn</c> work elements.</summary>
    void Horizontal(ref T src, ref TWork dst);

    /// <summary>Combines the <c>ksize</c> work rows at <paramref name="rows"/> into destination row <paramref name="dy"/>.</summary>
    void Vertical(ref TWork buffer, ReadOnlySpan<int> rows, int dy, ref T dst);
}

/// <summary>
/// Row driver of <c>resizeGeneric_Invoker</c>. Each stripe of destination rows keeps its own
/// ring of horizontally resized source rows; a source row is resized once per stripe. Values do
/// not depend on the stripe split.
/// </summary>
internal static class ResizeGeneric
{
    public static void Run<T, TWork, TRows>(Mat<T> src, Mat<T> dst, TRows rows, int[] yOffsets, int ksize)
        where T : unmanaged
        where TWork : unmanaged
        where TRows : struct, IResizeRows<T, TWork>
    {
        int width = dst.Width * dst.Channels;
        long work = (long)width * ksize / Math.Max(1, Vector<float>.Count / 2);
        ParallelRows.For(dst.Height, work, (y0, y1) => Stripe<T, TWork, TRows>(src, dst, rows, yOffsets, ksize, width, y0, y1));
    }

    private static void Stripe<T, TWork, TRows>(
        Mat<T> src, Mat<T> dst, TRows rows, int[] yOffsets, int ksize, int width, int y0, int y1)
        where T : unmanaged
        where TWork : unmanaged
        where TRows : struct, IResizeRows<T, TWork>
    {
        TWork[] buffer = ArrayPool<TWork>.Shared.Rent(width * ksize);
        try
        {
            ref TWork buf = ref MemoryMarshal.GetArrayDataReference(buffer);
            Span<int> slotRow = stackalloc int[ksize];
            Span<int> need = stackalloc int[ksize];
            Span<int> offsets = stackalloc int[ksize];
            slotRow.Fill(-1);
            int height = src.Height;
            int ksize2 = ksize / 2;
            for (int dy = y0; dy < y1; dy++)
            {
                int sy0 = yOffsets[dy] - ksize2 + 1;
                for (int k = 0; k < ksize; k++)
                {
                    int sy = sy0 + k;
                    need[k] = sy < 0 ? 0 : (sy < height ? sy : height - 1);
                    offsets[k] = -1;
                }

                for (int k = 0; k < ksize; k++)
                {
                    int slot = slotRow.IndexOf(need[k]);
                    if (slot < 0)
                    {
                        slot = 0;
                        while (need.Contains(slotRow[slot]))
                        {
                            slot++;
                        }

                        rows.Horizontal(ref MemoryMarshal.GetReference(src.GetReadOnlyRowSpan(need[k])), ref Unsafe.Add(ref buf, slot * width));
                        slotRow[slot] = need[k];
                    }

                    offsets[k] = slot * width;
                }

                rows.Vertical(ref buf, offsets, dy, ref MemoryMarshal.GetReference(dst.GetRowSpan(dy)));
            }
        }
        finally
        {
            ArrayPool<TWork>.Shared.Return(buffer);
        }
    }
}

/// <summary>Floating-point separable resize for <c>ushort</c>, <c>short</c> and <c>float</c> (work type <c>float</c>).</summary>
internal readonly struct ResizeRowsFloat<T> : IResizeRows<T, float>
    where T : unmanaged
{
    private readonly int[] _xofs;
    private readonly float[] _alpha;
    private readonly float[] _beta;
    private readonly int _ksize;
    private readonly int _cn;
    private readonly int _swidth;
    private readonly int _width;
    private readonly int _xminE;
    private readonly int _xmaxE;
    private readonly int _rightAssocEnd;
    private readonly ResizeLinearGather? _gather;

    public ResizeRowsFloat(ResizeCoefficients c, int srcWidth, int dstWidth, int cn, ResizeLinearGather? gather)
    {
        _gather = gather;
        _xofs = c.XOffsets;
        _alpha = c.Alpha;
        _beta = c.Beta;
        _ksize = c.Ksize;
        _cn = cn;
        _swidth = srcWidth * cn;
        _width = dstWidth * cn;
        _xminE = c.XMin * cn;
        _xmaxE = c.XMax * cn;

        // 128-bit vector bodies sum right to left in chunks of 8 (16-bit output) or 4 (float) lanes.
        // The 16U Lanczos4 body (SSE4.1) sums left to right like the scalar tail.
        bool rightAssoc = _ksize > 2 && !(typeof(T) == typeof(ushort) && _ksize == 8);
        int lanes = typeof(T) == typeof(float) ? 4 : 8;
        _rightAssocEnd = rightAssoc ? _width / lanes * lanes : 0;
    }

    public void Horizontal(ref T src, ref float dst)
    {
        ref int xofs = ref MemoryMarshal.GetArrayDataReference(_xofs);
        ref float alpha = ref MemoryMarshal.GetArrayDataReference(_alpha);
        int cn = _cn;
        if (_ksize == 2)
        {
            int e = 0;
            if (typeof(T) == typeof(float) && _gather is not null)
            {
                ref float sf = ref Unsafe.As<T, float>(ref src);
                for (int g = 0; g < _gather.Groups; g++, e += 8)
                {
                    if (!_gather.TryGroup(ref sf, ref dst, g, cn))
                    {
                        for (int i = e; i < e + 8; i++)
                        {
                            int sx = Unsafe.Add(ref xofs, i);
                            Unsafe.Add(ref dst, i) = (ToF(Unsafe.Add(ref src, sx)) * Unsafe.Add(ref alpha, 2 * i))
                                + (ToF(Unsafe.Add(ref src, sx + cn)) * Unsafe.Add(ref alpha, (2 * i) + 1));
                        }
                    }
                }
            }

            for (; e < _xmaxE; e++)
            {
                int sx = Unsafe.Add(ref xofs, e);
                Unsafe.Add(ref dst, e) = (ToF(Unsafe.Add(ref src, sx)) * Unsafe.Add(ref alpha, 2 * e))
                    + (ToF(Unsafe.Add(ref src, sx + cn)) * Unsafe.Add(ref alpha, (2 * e) + 1));
            }

            for (; e < _width; e++)
            {
                Unsafe.Add(ref dst, e) = ToF(Unsafe.Add(ref src, Unsafe.Add(ref xofs, e)));
            }

            return;
        }

        int k = _ksize;
        int off = k == 4 ? -cn : -3 * cn;
        int limit = _xminE;
        int e2 = 0;
        while (true)
        {
            for (; e2 < limit; e2++)
            {
                int sx = Unsafe.Add(ref xofs, e2) + off;
                ref float a = ref Unsafe.Add(ref alpha, e2 * k);
                float v = 0;
                for (int j = 0; j < k; j++)
                {
                    int sxj = sx + (j * cn);
                    while (sxj < 0)
                    {
                        sxj += cn;
                    }

                    while (sxj >= _swidth)
                    {
                        sxj -= cn;
                    }

                    v += ToF(Unsafe.Add(ref src, sxj)) * Unsafe.Add(ref a, j);
                }

                Unsafe.Add(ref dst, e2) = v;
            }

            if (limit == _width)
            {
                break;
            }

            if (k == 4)
            {
                if (typeof(T) == typeof(float) && _gather is not null)
                {
                    ref float sf = ref Unsafe.As<T, float>(ref src);
                    for (; (e2 & 7) != 0 && e2 < _xmaxE; e2++)
                    {
                        Cubic(ref src, ref dst, ref xofs, ref alpha, cn, e2);
                    }

                    for (int g = e2 / 8; g < _gather.Groups; g++, e2 += 8)
                    {
                        if (!_gather.TryGroup(ref sf, ref dst, g, cn))
                        {
                            for (int i = e2; i < e2 + 8; i++)
                            {
                                Cubic(ref src, ref dst, ref xofs, ref alpha, cn, i);
                            }
                        }
                    }
                }

                for (; e2 < _xmaxE; e2++)
                {
                    ref T s = ref Unsafe.Add(ref src, Unsafe.Add(ref xofs, e2) - cn);
                    ref float a = ref Unsafe.Add(ref alpha, e2 * 4);
                    Unsafe.Add(ref dst, e2) = (ToF(s) * a) + (ToF(Unsafe.Add(ref s, cn)) * Unsafe.Add(ref a, 1))
                        + (ToF(Unsafe.Add(ref s, 2 * cn)) * Unsafe.Add(ref a, 2)) + (ToF(Unsafe.Add(ref s, 3 * cn)) * Unsafe.Add(ref a, 3));
                }
            }
            else
            {
                for (; e2 < _xmaxE; e2++)
                {
                    ref T s = ref Unsafe.Add(ref src, Unsafe.Add(ref xofs, e2) - (3 * cn));
                    ref float a = ref Unsafe.Add(ref alpha, e2 * 8);
                    Unsafe.Add(ref dst, e2) = (ToF(s) * a) + (ToF(Unsafe.Add(ref s, cn)) * Unsafe.Add(ref a, 1))
                        + (ToF(Unsafe.Add(ref s, 2 * cn)) * Unsafe.Add(ref a, 2)) + (ToF(Unsafe.Add(ref s, 3 * cn)) * Unsafe.Add(ref a, 3))
                        + (ToF(Unsafe.Add(ref s, 4 * cn)) * Unsafe.Add(ref a, 4)) + (ToF(Unsafe.Add(ref s, 5 * cn)) * Unsafe.Add(ref a, 5))
                        + (ToF(Unsafe.Add(ref s, 6 * cn)) * Unsafe.Add(ref a, 6)) + (ToF(Unsafe.Add(ref s, 7 * cn)) * Unsafe.Add(ref a, 7));
                }
            }

            limit = _width;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Cubic(ref T src, ref float dst, ref int xofs, ref float alpha, int cn, int e)
    {
        ref T s = ref Unsafe.Add(ref src, Unsafe.Add(ref xofs, e) - cn);
        ref float a = ref Unsafe.Add(ref alpha, e * 4);
        Unsafe.Add(ref dst, e) = (ToF(s) * a) + (ToF(Unsafe.Add(ref s, cn)) * Unsafe.Add(ref a, 1))
            + (ToF(Unsafe.Add(ref s, 2 * cn)) * Unsafe.Add(ref a, 2)) + (ToF(Unsafe.Add(ref s, 3 * cn)) * Unsafe.Add(ref a, 3));
    }

    public void Vertical(ref float buffer, ReadOnlySpan<int> rows, int dy, ref T dst)
    {
        int k = _ksize;
        ReadOnlySpan<float> b = _beta.AsSpan(dy * k, k);
        Span<int> r = stackalloc int[8];
        rows.CopyTo(r);
        if (k == 2)
        {
            VerticalRange(ref buffer, r, b, ref dst, 0, _width, rightAssoc: false);
            return;
        }

        VerticalRange(ref buffer, r, b, ref dst, 0, _rightAssocEnd, rightAssoc: true);
        VerticalRange(ref buffer, r, b, ref dst, _rightAssocEnd, _width, rightAssoc: false);
    }

    private void VerticalRange(ref float buffer, Span<int> r, ReadOnlySpan<float> b, ref T dst, int x0, int x1, bool rightAssoc)
    {
        int k = _ksize;
        int x = x0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<float>.Count;
            Span<Vector<float>> vb = stackalloc Vector<float>[8];
            for (int i = 0; i < k; i++)
            {
                vb[i] = new Vector<float>(b[i]);
            }

            if (typeof(T) == typeof(float) && k <= 4)
            {
                ref float d = ref Unsafe.As<T, float>(ref dst);
                ref float s0 = ref Unsafe.Add(ref buffer, r[0]);
                ref float s1 = ref Unsafe.Add(ref buffer, r[1]);
                Vector<float> b0 = vb[0], b1 = vb[1];
                if (k == 2)
                {
                    for (; x <= x1 - vc; x += vc)
                    {
                        ((Vector.LoadUnsafe(ref s0, (nuint)x) * b0) + (Vector.LoadUnsafe(ref s1, (nuint)x) * b1)).StoreUnsafe(ref d, (nuint)x);
                    }
                }
                else
                {
                    ref float s2 = ref Unsafe.Add(ref buffer, r[2]);
                    ref float s3 = ref Unsafe.Add(ref buffer, r[3]);
                    Vector<float> b2 = vb[2], b3 = vb[3];
                    for (; x <= x1 - vc; x += vc)
                    {
                        Vector<float> p0 = Vector.LoadUnsafe(ref s0, (nuint)x) * b0, p1 = Vector.LoadUnsafe(ref s1, (nuint)x) * b1;
                        Vector<float> p2 = Vector.LoadUnsafe(ref s2, (nuint)x) * b2, p3 = Vector.LoadUnsafe(ref s3, (nuint)x) * b3;
                        (rightAssoc ? p0 + (p1 + (p2 + p3)) : ((p0 + p1) + p2) + p3).StoreUnsafe(ref d, (nuint)x);
                    }
                }
            }
            else if (typeof(T) == typeof(float))
            {
                for (; x <= x1 - vc; x += vc)
                {
                    Sum(ref buffer, r, vb, k, x, rightAssoc).StoreUnsafe(ref Unsafe.As<T, float>(ref dst), (nuint)x);
                }
            }
            else
            {
                for (; x <= x1 - (2 * vc); x += 2 * vc)
                {
                    Vector<int> i0 = FilterVectors.CvRound(Sum(ref buffer, r, vb, k, x, rightAssoc));
                    Vector<int> i1 = FilterVectors.CvRound(Sum(ref buffer, r, vb, k, x + vc, rightAssoc));
                    if (typeof(T) == typeof(short))
                    {
                        FilterVectors.SaturateToInt16(i0, i1).StoreUnsafe(ref Unsafe.As<T, short>(ref dst), (nuint)x);
                    }
                    else
                    {
                        var max = new Vector<int>(ushort.MaxValue);
                        Vector<uint> u0 = Vector.AsVectorUInt32(Vector.Min(Vector.Max(i0, Vector<int>.Zero), max));
                        Vector<uint> u1 = Vector.AsVectorUInt32(Vector.Min(Vector.Max(i1, Vector<int>.Zero), max));
                        Vector.Narrow(u0, u1).StoreUnsafe(ref Unsafe.As<T, ushort>(ref dst), (nuint)x);
                    }
                }
            }
        }

        for (; x < x1; x++)
        {
            float acc;
            if (rightAssoc)
            {
                acc = Unsafe.Add(ref buffer, r[k - 1] + x) * b[k - 1];
                for (int i = k - 2; i >= 0; i--)
                {
                    acc = (Unsafe.Add(ref buffer, r[i] + x) * b[i]) + acc;
                }
            }
            else
            {
                acc = Unsafe.Add(ref buffer, r[0] + x) * b[0];
                for (int i = 1; i < k; i++)
                {
                    acc += Unsafe.Add(ref buffer, r[i] + x) * b[i];
                }
            }

            Unsafe.Add(ref dst, x) = FromF(acc);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Sum(ref float buffer, Span<int> r, Span<Vector<float>> vb, int k, int x, bool rightAssoc)
    {
        Vector<float> acc;
        if (rightAssoc)
        {
            acc = Vector.LoadUnsafe(ref buffer, (nuint)(r[k - 1] + x)) * vb[k - 1];
            for (int i = k - 2; i >= 0; i--)
            {
                acc = (Vector.LoadUnsafe(ref buffer, (nuint)(r[i] + x)) * vb[i]) + acc;
            }
        }
        else
        {
            acc = Vector.LoadUnsafe(ref buffer, (nuint)(r[0] + x)) * vb[0];
            for (int i = 1; i < k; i++)
            {
                acc += Vector.LoadUnsafe(ref buffer, (nuint)(r[i] + x)) * vb[i];
            }
        }

        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ToF(T v)
    {
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<T, float>(ref v);
        }

        if (typeof(T) == typeof(ushort))
        {
            return Unsafe.As<T, ushort>(ref v);
        }

        return Unsafe.As<T, short>(ref v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T FromF(float v)
    {
        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<float, T>(ref v);
        }

        return Saturate.FromInt<T>(ScalarConverter.CvRound(v));
    }
}

/// <summary>Separable resize for <c>double</c> (work type <c>double</c>, single-precision weights).</summary>
internal readonly struct ResizeRowsDouble : IResizeRows<double, double>
{
    private readonly int[] _xofs;
    private readonly float[] _alpha;
    private readonly float[] _beta;
    private readonly int _ksize;
    private readonly int _cn;
    private readonly int _swidth;
    private readonly int _width;
    private readonly int _xminE;
    private readonly int _xmaxE;

    public ResizeRowsDouble(ResizeCoefficients c, int srcWidth, int dstWidth, int cn)
    {
        _xofs = c.XOffsets;
        _alpha = c.Alpha;
        _beta = c.Beta;
        _ksize = c.Ksize;
        _cn = cn;
        _swidth = srcWidth * cn;
        _width = dstWidth * cn;
        _xminE = c.XMin * cn;
        _xmaxE = c.XMax * cn;
    }

    public void Horizontal(ref double src, ref double dst)
    {
        ref int xofs = ref MemoryMarshal.GetArrayDataReference(_xofs);
        ref float alpha = ref MemoryMarshal.GetArrayDataReference(_alpha);
        int cn = _cn;
        int k = _ksize;
        if (k == 2)
        {
            int e = 0;
            for (; e < _xmaxE; e++)
            {
                int sx = Unsafe.Add(ref xofs, e);
                Unsafe.Add(ref dst, e) = (Unsafe.Add(ref src, sx) * Unsafe.Add(ref alpha, 2 * e))
                    + (Unsafe.Add(ref src, sx + cn) * Unsafe.Add(ref alpha, (2 * e) + 1));
            }

            for (; e < _width; e++)
            {
                Unsafe.Add(ref dst, e) = Unsafe.Add(ref src, Unsafe.Add(ref xofs, e));
            }

            return;
        }

        int off = k == 4 ? -cn : -3 * cn;
        for (int e = 0; e < _width; e++)
        {
            int sx = Unsafe.Add(ref xofs, e) + off;
            ref float a = ref Unsafe.Add(ref alpha, e * k);
            if (e < _xminE || e >= _xmaxE)
            {
                double v = 0;
                for (int j = 0; j < k; j++)
                {
                    int sxj = sx + (j * cn);
                    while (sxj < 0)
                    {
                        sxj += cn;
                    }

                    while (sxj >= _swidth)
                    {
                        sxj -= cn;
                    }

                    v += Unsafe.Add(ref src, sxj) * Unsafe.Add(ref a, j);
                }

                Unsafe.Add(ref dst, e) = v;
            }
            else
            {
                ref double s = ref Unsafe.Add(ref src, sx);
                double v = s * a;
                for (int j = 1; j < k; j++)
                {
                    v += Unsafe.Add(ref s, j * cn) * Unsafe.Add(ref a, j);
                }

                Unsafe.Add(ref dst, e) = v;
            }
        }
    }

    public void Vertical(ref double buffer, ReadOnlySpan<int> rows, int dy, ref double dst)
    {
        int k = _ksize;
        ReadOnlySpan<float> b = _beta.AsSpan(dy * k, k);
        Span<int> r = stackalloc int[8];
        rows.CopyTo(r);
        int width = _width;
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<double>.Count;
            Span<Vector<double>> vb = stackalloc Vector<double>[8];
            for (int i = 0; i < k; i++)
            {
                vb[i] = new Vector<double>(b[i]);
            }

            for (; x <= width - vc; x += vc)
            {
                Vector<double> acc = Vector.LoadUnsafe(ref buffer, (nuint)(r[0] + x)) * vb[0];
                for (int i = 1; i < k; i++)
                {
                    acc += Vector.LoadUnsafe(ref buffer, (nuint)(r[i] + x)) * vb[i];
                }

                acc.StoreUnsafe(ref dst, (nuint)x);
            }
        }

        for (; x < width; x++)
        {
            double acc = Unsafe.Add(ref buffer, r[0] + x) * b[0];
            for (int i = 1; i < k; i++)
            {
                acc += Unsafe.Add(ref buffer, r[i] + x) * b[i];
            }

            Unsafe.Add(ref dst, x) = acc;
        }
    }
}
