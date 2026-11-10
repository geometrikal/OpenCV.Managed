// This implementation is ported from the OpenCV non-separable linear filter.
//
// Original sources:
//   modules/imgproc/src/filter.dispatch.cpp (preprocess2DKernel, hal::filter2D)
//   modules/imgproc/src/filter.simd.hpp (Filter2D template, getLinearFilter)
//
// Kernels with an area of at least 130 (byte→byte, byte→short, float→float)
// or 50 (all other pairs) use the DFT path of dftFilter2D (CrossCorrEngine),
// as OpenCV does on SSE3 hardware for whole images.
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
/// OpenCV <c>hal::filter2D</c>: the DFT path for large kernels, otherwise direct correlation
/// with OpenCV's <c>Filter2D</c> scalar template (accumulation in the kernel type, float or double,
/// over the kernel's nonzero
/// coefficients in row-major order, starting from <c>(float)delta</c>).
/// Supported pairs follow <c>getLinearFilter</c>: <c>byte→byte/ushort/short/float/double</c>,
/// <c>ushort→ushort/float/double</c>, <c>short→short/float/double</c>, <c>float→float</c>
/// and <c>double→double</c>.
/// </summary>
internal static class Filter2DEngine
{
    public static Mat<TDst> Apply<TSrc, TDst>(
        Mat<TSrc> source,
        Mat<float> kernel,
        Point anchor,
        double delta,
        BorderType borderType,
        Mat<TDst>? destination = null)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(kernel);
        if (kernel.Channels != 1)
        {
            throw new ArgumentException("The kernel must be single-channel.", nameof(kernel));
        }

        var values = new double[kernel.Width * kernel.Height];
        for (int i = 0; i < kernel.Height; i++)
        {
            ReadOnlySpan<float> krow = kernel.GetReadOnlyRowSpan(i);
            for (int j = 0; j < kernel.Width; j++)
            {
                values[(i * kernel.Width) + j] = krow[j];
            }
        }

        return Apply<TSrc, TDst>(source, values, kernel.Width, kernel.Height, anchor, delta, borderType, destination);
    }

    /// <summary>
    /// Applies a row-major <paramref name="kw"/> × <paramref name="kh"/> kernel. Values are
    /// rounded to float32 unless the source or destination is double. A non-null
    /// <paramref name="destination"/> of the source size receives the result and is returned; it
    /// must not share its array with the source.
    /// </summary>
    public static Mat<TDst> Apply<TSrc, TDst>(
        Mat<TSrc> source,
        double[] kernel,
        int kw,
        int kh,
        Point anchor,
        double delta,
        BorderType borderType,
        Mat<TDst>? destination = null)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (borderType == BorderType.Transparent)
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "BorderType.Transparent is not supported by filters.");
        }

        int anchorX = anchor.X < 0 ? kw / 2 : anchor.X;
        int anchorY = anchor.Y < 0 ? kh / 2 : anchor.Y;
        if (anchorX >= kw || anchorY >= kh)
        {
            throw new ArgumentOutOfRangeException(nameof(anchor), anchor, "The anchor must lie inside the kernel.");
        }

        EnsureSupported<TSrc, TDst>();
        bool doubleKernel = typeof(TSrc) == typeof(double) || typeof(TDst) == typeof(double);

        if ((long)kw * kh >= CrossCorrEngine.Threshold<TSrc, TDst>())
        {
            return ToDestination(CrossCorrEngine.Filter2D<TSrc, TDst>(source, kernel, kw, kh, anchorX, anchorY, delta, borderType), destination);
        }

        if (borderType == BorderType.Wrap)
        {
            throw new NotSupportedException(
                "BORDER_WRAP is supported only by the DFT path of filter2D (large kernels), as in OpenCV.");
        }

        // preprocess2DKernel: nonzero coefficients (in the kernel type) in row-major order;
        // an all-zero kernel keeps a single zero tap at (0, 0).
        var coords = new List<Point>(kw * kh);
        var coeffs = new List<double>(kw * kh);
        for (int i = 0; i < kh; i++)
        {
            for (int j = 0; j < kw; j++)
            {
                double k = doubleKernel ? kernel[(i * kw) + j] : (float)kernel[(i * kw) + j];
                if (k == 0)
                {
                    continue;
                }

                coords.Add(new Point(j, i));
                coeffs.Add(k);
            }
        }

        if (coords.Count == 0)
        {
            coords.Add(new Point(0, 0));
            coeffs.Add(0);
        }

        if (!doubleKernel && SmallIntFilter2D.Supports<TSrc, TDst>(coeffs, delta))
        {
            return ToDestination(
                SmallIntFilter2D.Apply<TDst>((Mat<byte>)(object)source, kw, kh, anchorX, anchorY, borderType, coords, coeffs, delta),
                destination);
        }

        using PooledMat<TSrc> padded = Borders.CopyMakeBorderPooled(
            source,
            top: anchorY,
            bottom: kh - anchorY - 1,
            left: anchorX,
            right: kw - anchorX - 1,
            borderType,
            default);

        if (doubleKernel)
        {
            using PooledMat<double> paddedD = Widen<TSrc, double>(padded.Mat);
            return ConvolveDouble(paddedD.Mat, source.Width, source.Height, source.Channels, coords, coeffs, delta, destination);
        }

        if (typeof(TSrc) == typeof(float))
        {
            return Convolve((Mat<float>)(object)padded.Mat, source.Width, source.Height, source.Channels, coords, coeffs, delta, destination);
        }

        using PooledMat<float> paddedF = Widen<TSrc, float>(padded.Mat);
        return Convolve(paddedF.Mat, source.Width, source.Height, source.Channels, coords, coeffs, delta, destination);
    }

    private static Mat<TDst> ToDestination<TDst>(Mat<TDst> result, Mat<TDst>? destination)
        where TDst : unmanaged
    {
        if (destination is null)
        {
            return result;
        }

        result.CopyTo(destination);
        return destination;
    }

    /// <summary>Rejects the pairs that OpenCV's <c>getLinearFilter</c> does not provide (<c>ddepth &gt;= sdepth</c>).</summary>
    private static void EnsureSupported<TSrc, TDst>()
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        bool supported =
            typeof(TSrc) == typeof(byte)
                ? SepFilterEngine.IsFilterDepth<TDst>()
                : typeof(TSrc) == typeof(ushort) || typeof(TSrc) == typeof(short)
                    ? typeof(TDst) == typeof(TSrc) || typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double)
                    : (typeof(TSrc) == typeof(float) || typeof(TSrc) == typeof(double)) && typeof(TDst) == typeof(TSrc);
        if (!supported)
        {
            throw new NotSupportedException(
                $"Unsupported filter2D type combination ({typeof(TSrc).Name} → {typeof(TDst).Name}). " +
                "Supported: byte→byte/ushort/short/float/double, ushort→ushort/float/double, " +
                "short→short/float/double, float→float, double→double.");
        }
    }

    /// <summary>Converts a matrix to float or double; the conversion is exact.</summary>
    private static PooledMat<TBuf> Widen<TSrc, TBuf>(Mat<TSrc> source)
        where TSrc : unmanaged
        where TBuf : unmanaged
    {
        PooledMat<TBuf> result = PooledMat<TBuf>.Rent(source.Width, source.Height, source.Channels);
        for (int y = 0; y < source.Height; y++)
        {
            if (typeof(TBuf) == typeof(float))
            {
                FilterCasts.WidenRow(source.GetReadOnlyRowSpan(y), MemoryMarshal.Cast<TBuf, float>(result.Mat.GetRowSpan(y)));
            }
            else
            {
                FilterCasts.WidenRow(source.GetReadOnlyRowSpan(y), MemoryMarshal.Cast<TBuf, double>(result.Mat.GetRowSpan(y)));
            }
        }

        return result;
    }

    /// <summary>
    /// <c>sums[i] = delta + c[0]·s[i + off[0]] + c[1]·s[i + off[1]] + …</c> in double, accumulated
    /// left to right per element.
    /// </summary>
    private static Mat<TDst> ConvolveDouble<TDst>(
        Mat<double> padded,
        int dstWidth,
        int dstHeight,
        int cn,
        List<Point> coords,
        List<double> coeffs,
        double delta,
        Mat<TDst>? destination)
        where TDst : unmanaged
    {
        int nz = coords.Count;
        int width = dstWidth * cn;
        int stride = padded.Stride;
        var tapOffset = new int[nz];
        var tapCoeff = new double[nz];
        for (int k = 0; k < nz; k++)
        {
            tapOffset[k] = (coords[k].Y * stride) + (coords[k].X * cn);
            tapCoeff[k] = coeffs[k];
        }

        Mat<TDst> dst = destination ?? new Mat<TDst>(dstWidth, dstHeight, cn, clear: false);
        ParallelRows.For(dstHeight, (long)width * nz, (y0, y1) => ConvolveDoubleRows(padded, dst, tapOffset, tapCoeff, delta, width, y0, y1));
        return dst;
    }

    private static void ConvolveDoubleRows<TDst>(
        Mat<double> padded, Mat<TDst> dst, int[] tapOffset, double[] tapCoeff, double delta, int width, int y0, int y1)
        where TDst : unmanaged
    {
        int nz = tapOffset.Length;
        int stride = padded.Stride;
        double[] sumsArray = ArrayPool<double>.Shared.Rent(width);
        try
        {
            ref double src = ref MemoryMarshal.GetReference(padded.GetReadOnlySpan());
            ref double sum = ref sumsArray[0];
            int vd = Vector<double>.Count;
            for (int y = y0; y < y1; y++)
            {
                ref double row = ref Unsafe.Add(ref src, y * stride);
                int i = 0;
                if (Vector.IsHardwareAccelerated)
                {
                    var vdelta = new Vector<double>(delta);
                    for (; i <= width - vd; i += vd)
                    {
                        Vector<double> a = vdelta;
                        for (int k = 0; k < nz; k++)
                        {
                            a += new Vector<double>(tapCoeff[k]) * Vector.LoadUnsafe(ref Unsafe.Add(ref row, tapOffset[k] + i));
                        }

                        a.StoreUnsafe(ref sum, (nuint)i);
                    }
                }

                for (; i < width; i++)
                {
                    double acc = delta;
                    for (int k = 0; k < nz; k++)
                    {
                        acc += tapCoeff[k] * Unsafe.Add(ref row, tapOffset[k] + i);
                    }

                    Unsafe.Add(ref sum, i) = acc;
                }

                FilterCasts.StoreRow(dst.GetRowSpan(y), sumsArray.AsSpan(0, width));
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(sumsArray);
        }
    }

    /// <summary>
    /// <c>sums[i] = delta + c[0]·s[i + off[0]] + c[1]·s[i + off[1]] + …</c>, accumulated left to
    /// right per element. A block of four vectors stays in registers across all taps.
    /// </summary>
    private static Mat<TDst> Convolve<TDst>(
        Mat<float> padded,
        int dstWidth,
        int dstHeight,
        int cn,
        List<Point> coords,
        List<double> coeffs,
        double delta,
        Mat<TDst>? destination)
        where TDst : unmanaged
    {
        int nz = coords.Count;
        int width = dstWidth * cn;
        int stride = padded.Stride;
        float fdelta = (float)delta;

        var tapOffset = new int[nz];
        var tapCoeff = new float[nz];
        for (int k = 0; k < nz; k++)
        {
            tapOffset[k] = (coords[k].Y * stride) + (coords[k].X * cn);
            tapCoeff[k] = (float)coeffs[k];
        }

        Mat<TDst> dst = destination ?? new Mat<TDst>(dstWidth, dstHeight, cn, clear: false);
        ParallelRows.For(dstHeight, (long)width * nz, (y0, y1) => ConvolveRows(padded, dst, tapOffset, tapCoeff, fdelta, width, y0, y1));
        return dst;
    }

    private static void ConvolveRows<TDst>(
        Mat<float> padded, Mat<TDst> dst, int[] tapOffset, float[] tapCoeff, float fdelta, int width, int y0, int y1)
        where TDst : unmanaged
    {
        int nz = tapOffset.Length;
        int stride = padded.Stride;
        float[] sumsArray = ArrayPool<float>.Shared.Rent(width);
        try
        {
            Span<float> sums = sumsArray.AsSpan(0, width);
            ref float src = ref MemoryMarshal.GetReference(padded.GetReadOnlySpan());
            ref float sum = ref sumsArray[0];
            ref int off = ref tapOffset[0];
            ref float coeff = ref tapCoeff[0];
            int vf = Vector<float>.Count;
            int block = 4 * vf;
            for (int y = y0; y < y1; y++)
            {
                ref float row = ref Unsafe.Add(ref src, y * stride);
                int i = 0;
                if (Vector.IsHardwareAccelerated && width >= block)
                {
                    var vdelta = new Vector<float>(fdelta);
                    for (; i < width; i += block)
                    {
                        // The final block overlaps the previous one; it rewrites equal values.
                        i = Math.Min(i, width - block);
                        Vector<float> a0 = vdelta, a1 = vdelta, a2 = vdelta, a3 = vdelta;
                        for (int k = 0; k < nz; k++)
                        {
                            var c = new Vector<float>(Unsafe.Add(ref coeff, k));
                            ref float t = ref Unsafe.Add(ref row, Unsafe.Add(ref off, k) + i);
                            a0 += c * Vector.LoadUnsafe(ref t);
                            a1 += c * Vector.LoadUnsafe(ref t, (nuint)vf);
                            a2 += c * Vector.LoadUnsafe(ref t, (nuint)(2 * vf));
                            a3 += c * Vector.LoadUnsafe(ref t, (nuint)(3 * vf));
                        }

                        a0.StoreUnsafe(ref sum, (nuint)i);
                        a1.StoreUnsafe(ref sum, (nuint)(i + vf));
                        a2.StoreUnsafe(ref sum, (nuint)(i + (2 * vf)));
                        a3.StoreUnsafe(ref sum, (nuint)(i + (3 * vf)));
                    }
                }

                for (; i < width; i++)
                {
                    float acc = fdelta;
                    for (int k = 0; k < nz; k++)
                    {
                        acc += Unsafe.Add(ref coeff, k) * Unsafe.Add(ref row, Unsafe.Add(ref off, k) + i);
                    }

                    Unsafe.Add(ref sum, i) = acc;
                }

                FilterCasts.StoreRow(dst.GetRowSpan(y), sums);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(sumsArray);
        }
    }
}
