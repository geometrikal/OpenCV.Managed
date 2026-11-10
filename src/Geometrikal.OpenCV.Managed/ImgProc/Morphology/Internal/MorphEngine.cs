// This implementation is ported from the OpenCV morphology implementation.
//
// Original sources:
//   modules/imgproc/src/morph.dispatch.cpp (orchestration, default border values)
//   modules/imgproc/src/morph.simd.hpp     (MinOp/MaxOp, row/column/2D filter semantics)
//
// OpenCV's streaming FilterEngine is deliberately replaced by a padded working
// image: erosion/dilation are pure neighbourhood minima/maxima, so the padded
// formulation produces bit-identical output without the row-buffer machinery.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Core erode/dilate machinery shared by <see cref="Morphology"/>: a single
/// min/max pass over a border-padded working image, with a separable two-pass
/// fast path for all-ones (rectangular) kernels and an offset-list pass for
/// arbitrary shapes.
/// </summary>
internal static class MorphEngine
{
    /// <summary>
    /// Applies one erosion or dilation of <paramref name="source"/> by the
    /// structuring element into <paramref name="dst"/>. The kernel must be
    /// non-empty and the anchor already normalized (inside the kernel).
    /// <paramref name="dst"/> can be <paramref name="source"/>: the source is
    /// copied into a padded buffer first.
    /// </summary>
    public static void ApplyInto<T>(
        Mat<T> source,
        Mat<T> dst,
        Mat<sbyte> kernel,
        Point anchor,
        bool erode,
        BorderType borderType,
        Scalar? borderValue)
        where T : unmanaged
    {
        int kw = kernel.Width;
        int kh = kernel.Height;

        // Effective constant border: the user's value, or OpenCV's
        // morphologyDefaultBorderValue substitution (type max for erode, type
        // min for dilate) — morph.dispatch.cpp:111-128.
        Scalar constant = borderValue ?? Scalar.All(erode ? TypeMax<T>() : TypeMin<T>());

        using PooledMat<T> pooled = Borders.CopyMakeBorderPooled(
            source,
            top: anchor.Y,
            bottom: kh - anchor.Y - 1,
            left: anchor.X,
            right: kw - anchor.X - 1,
            borderType,
            constant);
        Mat<T> padded = pooled.Mat;

        bool rect = CountNonZero(kernel) == kw * kh;

        if (erode)
        {
            if (rect)
            {
                ApplySeparableRect<T, MinOp>(padded, dst, kw, kh);
            }
            else
            {
                ApplyArbitrary<T, MinOp>(padded, dst, kernel);
            }
        }
        else if (rect)
        {
            ApplySeparableRect<T, MaxOp>(padded, dst, kw, kh);
        }
        else
        {
            ApplyArbitrary<T, MaxOp>(padded, dst, kernel);
        }
    }

    /// <summary>
    /// All-ones kernel: horizontal min/max over every padded row, then
    /// vertical over the intermediate — identical to the full-window result.
    /// Each output folds its operands left to right, starting from the first.
    /// </summary>
    private static void ApplySeparableRect<T, TOp>(Mat<T> padded, Mat<T> dst, int kw, int kh)
        where T : unmanaged
        where TOp : struct, IMinMaxOp
    {
        int cn = dst.Channels;
        int rowLength = dst.Width * cn;
        int paddedStride = padded.Stride;
        T[] rowPass = ArrayPool<T>.Shared.Rent(rowLength * padded.Height);
        try
        {
            // One min/max instruction covers Vector<T>.Count elements; the passes are memory-bound.
            long lanes = Vector<T>.Count;
            ParallelRows.For(padded.Height, rowLength * kw / lanes, (y0, y1) =>
            {
                ref T src = ref MemoryMarshal.GetReference(padded.GetReadOnlySpan());
                ref T mid = ref rowPass[0];
                for (int y = y0; y < y1; y++)
                {
                    FoldTaps<T, TOp>(
                        ref Unsafe.Add(ref src, y * paddedStride), cn, kw,
                        ref Unsafe.Add(ref mid, y * rowLength), rowLength);
                }
            });

            ParallelRows.For(dst.Height, rowLength * kh / lanes, (y0, y1) =>
            {
                ref T mid = ref rowPass[0];
                for (int y = y0; y < y1; y++)
                {
                    FoldTaps<T, TOp>(
                        ref Unsafe.Add(ref mid, y * rowLength), rowLength, kh,
                        ref MemoryMarshal.GetReference(dst.GetRowSpan(y)), rowLength);
                }
            });
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rowPass);
        }
    }

    /// <summary>
    /// <c>d[i] = op(...op(op(s[i], s[i + step]), s[i + 2·step])..., s[i + (taps-1)·step])</c>
    /// for <c>i</c> in <c>[0, length)</c>.
    /// </summary>
    private static void FoldTaps<T, TOp>(ref T s, int step, int taps, ref T d, int length)
        where T : unmanaged
        where TOp : struct, IMinMaxOp
    {
        int i = 0;
        if (MinMaxOps.IsVectorized<T>())
        {
            int vc = Vector<T>.Count;
            for (; i <= length - vc; i += vc)
            {
                Vector<T> acc = Vector.LoadUnsafe(ref s, (nuint)i);
                for (int k = 1; k < taps; k++)
                {
                    acc = TOp.Apply(acc, Vector.LoadUnsafe(ref s, (nuint)(i + (k * step))));
                }

                acc.StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < length; i++)
        {
            T acc = Unsafe.Add(ref s, i);
            for (int k = 1; k < taps; k++)
            {
                acc = TOp.Apply(acc, Unsafe.Add(ref s, i + (k * step)));
            }

            Unsafe.Add(ref d, i) = acc;
        }
    }

    /// <summary>
    /// Arbitrary structuring element: fold over the offsets of the non-zero
    /// kernel cells, gathered in row-major order like OpenCV's
    /// <c>preprocess2DKernel</c>.
    /// </summary>
    private static void ApplyArbitrary<T, TOp>(Mat<T> padded, Mat<T> dst, Mat<sbyte> kernel)
        where T : unmanaged
        where TOp : struct, IMinMaxOp
    {
        int cn = dst.Channels;
        int rowLength = dst.Width * cn;
        int paddedStride = padded.Stride;

        int nz = CountNonZero(kernel);
        var offsets = new int[nz];
        int n = 0;
        for (int ky = 0; ky < kernel.Height; ky++)
        {
            ReadOnlySpan<sbyte> krow = kernel.GetReadOnlyRowSpan(ky);
            for (int kx = 0; kx < kernel.Width; kx++)
            {
                if (krow[kx] != 0)
                {
                    offsets[n++] = (ky * paddedStride) + (kx * cn);
                }
            }
        }

        ParallelRows.For(dst.Height, (long)rowLength * nz / Vector<T>.Count, (y0, y1) => ArbitraryRows<T, TOp>(padded, dst, offsets, y0, y1));
    }

    private static void ArbitraryRows<T, TOp>(Mat<T> padded, Mat<T> dst, int[] offsets, int y0, int y1)
        where T : unmanaged
        where TOp : struct, IMinMaxOp
    {
        int nz = offsets.Length;
        int rowLength = dst.Width * dst.Channels;
        int paddedStride = padded.Stride;
        ref T src = ref MemoryMarshal.GetReference(padded.GetReadOnlySpan());
        for (int y = y0; y < y1; y++)
        {
            ref T s = ref Unsafe.Add(ref src, y * paddedStride);
            ref T d = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
            int i = 0;
            if (MinMaxOps.IsVectorized<T>())
            {
                int vc = Vector<T>.Count;
                for (; i <= rowLength - vc; i += vc)
                {
                    Vector<T> acc = Vector.LoadUnsafe(ref s, (nuint)(i + offsets[0]));
                    for (int k = 1; k < nz; k++)
                    {
                        acc = TOp.Apply(acc, Vector.LoadUnsafe(ref s, (nuint)(i + offsets[k])));
                    }

                    acc.StoreUnsafe(ref d, (nuint)i);
                }
            }

            for (; i < rowLength; i++)
            {
                T acc = Unsafe.Add(ref s, i + offsets[0]);
                for (int k = 1; k < nz; k++)
                {
                    acc = TOp.Apply(acc, Unsafe.Add(ref s, i + offsets[k]));
                }

                Unsafe.Add(ref d, i) = acc;
            }
        }
    }

    /// <summary>Counts non-zero structuring-element cells.</summary>
    public static int CountNonZero(Mat<sbyte> kernel)
    {
        int count = 0;
        for (int y = 0; y < kernel.Height; y++)
        {
            ReadOnlySpan<sbyte> row = kernel.GetReadOnlyRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                if (row[x] != 0)
                {
                    count++;
                }
            }
        }

        return count;
    }

    /// <summary>
    /// Saturating element-wise <c>a - b</c> (cv::subtract semantics: integer
    /// types clamp to the type range; floating types subtract plainly).
    /// </summary>
    /// <remarks><paramref name="dst"/> can be <paramref name="a"/> or <paramref name="b"/>.</remarks>
    public static void SubtractInto<T>(Mat<T> a, Mat<T> b, Mat<T> dst)
        where T : unmanaged
    {
        for (int y = 0; y < a.Height; y++)
        {
            ReadOnlySpan<T> ra = a.GetReadOnlyRowSpan(y);
            ReadOnlySpan<T> rb = b.GetReadOnlyRowSpan(y);
            Span<T> rd = dst.GetRowSpan(y);
            int i = 0;
            if (MinMaxOps.IsVectorized<T>() && typeof(T) != typeof(short))
            {
                ref T a0 = ref MemoryMarshal.GetReference(ra);
                ref T b0 = ref MemoryMarshal.GetReference(rb);
                ref T d0 = ref MemoryMarshal.GetReference(rd);
                for (; i <= ra.Length - Vector<T>.Count; i += Vector<T>.Count)
                {
                    Vector<T> va = Vector.LoadUnsafe(ref a0, (nuint)i);
                    Vector<T> vb = Vector.LoadUnsafe(ref b0, (nuint)i);

                    // Unsigned saturation: a - min(a, b) is a - b clamped at zero.
                    Vector<T> r = MinMaxOps.IsFloating<T>() ? va - vb : va - Vector.Min(va, vb);
                    r.StoreUnsafe(ref d0, (nuint)i);
                }
            }

            for (; i < ra.Length; i++)
            {
                rd[i] = SubtractSaturate(ra[i], rb[i]);
            }
        }

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T SubtractSaturate<T>(T a, T b)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            byte r = (byte)Math.Max(Unsafe.As<T, byte>(ref a) - Unsafe.As<T, byte>(ref b), 0);
            return Unsafe.As<byte, T>(ref r);
        }

        if (typeof(T) == typeof(ushort))
        {
            ushort r = (ushort)Math.Max(Unsafe.As<T, ushort>(ref a) - Unsafe.As<T, ushort>(ref b), 0);
            return Unsafe.As<ushort, T>(ref r);
        }

        if (typeof(T) == typeof(short))
        {
            short r = (short)Math.Clamp(
                Unsafe.As<T, short>(ref a) - Unsafe.As<T, short>(ref b),
                short.MinValue,
                short.MaxValue);
            return Unsafe.As<short, T>(ref r);
        }

        if (typeof(T) == typeof(float))
        {
            float r = Unsafe.As<T, float>(ref a) - Unsafe.As<T, float>(ref b);
            return Unsafe.As<float, T>(ref r);
        }

        if (typeof(T) == typeof(double))
        {
            double r = Unsafe.As<T, double>(ref a) - Unsafe.As<T, double>(ref b);
            return Unsafe.As<double, T>(ref r);
        }

        throw new NotSupportedException(
            $"Morphology does not support the matrix element type '{typeof(T)}'.");
    }

    /// <summary>Type maximum used for the default erode constant border.</summary>
    public static double TypeMax<T>()
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return byte.MaxValue;
        }

        if (typeof(T) == typeof(ushort))
        {
            return ushort.MaxValue;
        }

        if (typeof(T) == typeof(short))
        {
            return short.MaxValue;
        }

        if (typeof(T) == typeof(float))
        {
            return float.MaxValue;
        }

        if (typeof(T) == typeof(double))
        {
            return double.MaxValue;
        }

        throw new NotSupportedException(
            $"Morphology does not support the matrix element type '{typeof(T)}'.");
    }

    /// <summary>Type minimum used for the default dilate constant border.</summary>
    public static double TypeMin<T>()
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort))
        {
            return 0;
        }

        if (typeof(T) == typeof(short))
        {
            return short.MinValue;
        }

        if (typeof(T) == typeof(float))
        {
            return float.MinValue; // -FLT_MAX
        }

        if (typeof(T) == typeof(double))
        {
            return double.MinValue; // -DBL_MAX
        }

        throw new NotSupportedException(
            $"Morphology does not support the matrix element type '{typeof(T)}'.");
    }
}
