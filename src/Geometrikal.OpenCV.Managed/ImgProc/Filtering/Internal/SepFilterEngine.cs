// This implementation is ported from the OpenCV separable linear filter.
//
// Original sources:
//   modules/imgproc/src/filter.dispatch.cpp (createSeparableLinearFilter,
//     bit-exact mode selection, hal::sepFilter2D)
//   modules/imgproc/src/filter.simd.hpp (RowFilter, SymmRowSmallFilter,
//     ColumnFilter, SymmColumnFilter, SymmColumnSmallFilter, Cast,
//     FixedPtCastEx, getLinearRowFilter, getLinearColumnFilter)
//
// OpenCV's streaming FilterEngine is replaced by a border-padded working
// image (see MorphEngine for the precedent); the row and column passes below
// reproduce the arithmetic of OpenCV's *scalar* filter kernels exactly.
// Integer-buffer paths collapse OpenCV's symmetric special cases into one
// generic loop because exact integer arithmetic makes them equal; float
// paths replicate OpenCV's branch structure operation-for-operation because
// float rounding depends on the association order.
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
/// Padded-image separable convolution replicating OpenCV's
/// <c>createSeparableLinearFilter</c> + <c>FilterEngine</c> pipeline. Sources and
/// destinations are <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or
/// <c>double</c>; <c>double→float</c> is rejected, as in OpenCV. The buffer is int32
/// (8-bit bit-exact modes), double (any double operand) or float.
/// </summary>
internal static class SepFilterEngine
{
    /// <summary>
    /// Applies the separable filter with float32 kernels. A non-null <paramref name="destination"/>
    /// of the source size receives the result and is returned.
    /// </summary>
    public static Mat<TDst> Apply<TSrc, TDst>(
        Mat<TSrc> source,
        float[] kernelX,
        float[] kernelY,
        Point anchor,
        double delta,
        BorderType borderType,
        Mat<TDst>? destination = null)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(kernelX);
        ArgumentNullException.ThrowIfNull(kernelY);
        return Apply<TSrc, TDst>(source, ToDouble(kernelX), ToDouble(kernelY), anchor, delta, borderType, destination);
    }

    /// <summary>
    /// Applies the separable filter. The kernels hold values of OpenCV's kernel type
    /// (float32 values unless the buffer is double). <paramref name="anchor"/> components
    /// may be negative for the centred default. A non-null <paramref name="destination"/>
    /// of the source size receives the result and is returned.
    /// </summary>
    public static Mat<TDst> Apply<TSrc, TDst>(
        Mat<TSrc> source,
        double[] kernelX,
        double[] kernelY,
        Point anchor,
        double delta,
        BorderType borderType,
        Mat<TDst>? destination)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(kernelX);
        ArgumentNullException.ThrowIfNull(kernelY);
        if (kernelX.Length == 0 || kernelY.Length == 0)
        {
            throw new ArgumentException("Filter kernels must be non-empty.");
        }

        if (borderType == BorderType.Wrap)
        {
            throw new NotSupportedException(
                "BORDER_WRAP is not supported by OpenCV's filter engine and is likewise rejected here.");
        }

        int kw = kernelX.Length;
        int kh = kernelY.Length;
        int anchorX = anchor.X < 0 ? kw / 2 : anchor.X;
        int anchorY = anchor.Y < 0 ? kh / 2 : anchor.Y;
        if (anchorX >= kw || anchorY >= kh)
        {
            throw new ArgumentOutOfRangeException(nameof(anchor), anchor, "The anchor must lie inside the kernel.");
        }

        EnsureSupported<TSrc, TDst>();
        KernelKind rtype = KernelClassifier.GetKernelType(kernelX, anchorX);
        KernelKind ctype = KernelClassifier.GetKernelType(kernelY, anchorY);
        int cn = source.Channels;
        int rowLength = source.Width * cn;
        Mat<TDst> dst = destination ?? new Mat<TDst>(source.Width, source.Height, cn, clear: false);

        if (typeof(TSrc) == typeof(byte))
        {
            // Bit-exact integer mode selection (createSeparableLinearFilter,
            // filter.dispatch.cpp:342). Note the *exact* equality test for the
            // 8U→8U case: smooth+symmetrical only, no other flags.
            bool byteDst = typeof(TDst) == typeof(byte);
            bool shortDst = typeof(TDst) == typeof(short);
            bool tryBitExact =
                (byteDst &&
                 rtype == (KernelKind.Smooth | KernelKind.Symmetrical) &&
                 ctype == (KernelKind.Smooth | KernelKind.Symmetrical)) ||
                (shortDst &&
                 (rtype & (KernelKind.Symmetrical | KernelKind.Asymmetrical)) != 0 &&
                 (ctype & (KernelKind.Symmetrical | KernelKind.Asymmetrical)) != 0 &&
                 (rtype & ctype & KernelKind.Integer) != 0);

            if (tryBitExact)
            {
                int quantBits = byteDst ? 8 : 0;
                int[]? kxi = KernelClassifier.TryCreateBitExactKernel32S(kernelX, quantBits);
                int[]? kyi = KernelClassifier.TryCreateBitExactKernel32S(kernelY, quantBits);
                if (kxi != null && kyi != null)
                {
                    int bits = quantBits * 2;
                    int ideltaAcc = ScalarConverter.CvRound(delta * (1L << bits));
                    var src8 = (Mat<byte>)(object)source;
                    if (shortDst && Int16Kernel.Fits(kxi, kyi, ideltaAcc))
                    {
                        RowStream.Run<byte, short, short, Int16Kernel>(
                            src8, (Mat<short>)(object)dst, kw, kh, anchorX, anchorY, borderType, rowLength,
                            new Int16Kernel(kxi, kyi, cn, ideltaAcc),
                            (long)rowLength * (kw + kh) / Vector<short>.Count);
                    }
                    else
                    {
                        RowStream.Run<byte, int, TDst, IntKernel<TDst>>(
                            src8, dst, kw, kh, anchorX, anchorY, borderType, rowLength,
                            new IntKernel<TDst>(kxi, kyi, cn, bits, ideltaAcc),
                            (long)rowLength * (kw + kh) * 2 / Vector<int>.Count);
                    }

                    return dst;
                }
            }
        }

        if (typeof(TSrc) == typeof(double) || typeof(TDst) == typeof(double))
        {
            // bdepth = CV_64F: RowFilter<ST, double> and (Symm)ColumnFilter<Cast<double, DT>>, scalar code only.
            RowStream.Run<TSrc, double, TDst, DoubleKernel<TSrc, TDst>>(
                source, dst, kw, kh, anchorX, anchorY, borderType, rowLength,
                new DoubleKernel<TSrc, TDst>(kernelX, kernelY, ctype, cn, delta),
                (long)rowLength * (kw + kh) / Vector<double>.Count);
            return dst;
        }

        RowStream.Run<TSrc, float, TDst, FloatKernel<TSrc, TDst>>(
            source, dst, kw, kh, anchorX, anchorY, borderType, rowLength,
            new FloatKernel<TSrc, TDst>(ToSingle(kernelX), ToSingle(kernelY), rtype, ctype, cn, (float)delta),
            (long)rowLength * (kw + kh) / Vector<float>.Count);
        return dst;
    }

    /// <summary>
    /// Rejects the combinations that OpenCV's <c>getLinearRowFilter</c> and
    /// <c>getLinearColumnFilter</c> do not provide.
    /// </summary>
    private static void EnsureSupported<TSrc, TDst>()
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        if (!IsFilterDepth<TSrc>() || !IsFilterDepth<TDst>() ||
            (typeof(TSrc) == typeof(double) && typeof(TDst) == typeof(float)))
        {
            throw new NotSupportedException(
                $"Unsupported separable filter type combination ({typeof(TSrc).Name} → {typeof(TDst).Name}). " +
                "Sources and destinations are byte, ushort, short, float or double; double → float is not supported.");
        }
    }

    /// <summary>Element types of OpenCV's linear filter engine.</summary>
    internal static bool IsFilterDepth<T>()
        where T : unmanaged
        => typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) || typeof(T) == typeof(short) ||
           typeof(T) == typeof(float) || typeof(T) == typeof(double);

    private static double[] ToDouble(float[] kernel)
    {
        var result = new double[kernel.Length];
        for (int i = 0; i < kernel.Length; i++)
        {
            result[i] = kernel[i];
        }

        return result;
    }

    private static float[] ToSingle(double[] kernel)
    {
        var result = new float[kernel.Length];
        for (int i = 0; i < kernel.Length; i++)
        {
            result[i] = (float)kernel[i];
        }

        return result;
    }

    // ------------------------------------------------------------- kernels

    /// <summary>Integer row and column filters (8-bit bit-exact modes): int32 buffer.</summary>
    private readonly struct IntKernel<TDst>(int[] kx, int[] ky, int cn, int bits, int ideltaAcc) : IRowStreamKernel<byte, int, TDst>
        where TDst : unmanaged
    {
        public void Row(ReadOnlySpan<byte> line, Span<int> row, Span<int> wide) => RowByteToInt(line, row, kx, cn);

        public void Column(ref int rows, int rowStride, Span<TDst> dst, Span<int> scratch)
        {
            if (typeof(TDst) == typeof(byte))
            {
                ColumnIntToByte(ref rows, rowStride, ky, bits, ideltaAcc, MemoryMarshal.Cast<TDst, byte>(dst));
            }
            else
            {
                ColumnIntToShort(ref rows, rowStride, ky, ideltaAcc, MemoryMarshal.Cast<TDst, short>(dst));
            }
        }
    }

    /// <summary>
    /// The 8U→16S integer filters in 16-bit lanes. <see cref="Fits"/> admits only kernels whose
    /// sums stay inside the int16 range, so the results equal the int32 sums without saturation.
    /// </summary>
    private readonly struct Int16Kernel : IRowStreamKernel<byte, short, short>
    {
        private readonly short[] _kx;
        private readonly int[] _xOffsets;
        private readonly short[] _ky;
        private readonly int[] _yRows;
        private readonly short _delta;

        public Int16Kernel(int[] kx, int[] ky, int cn, int delta)
        {
            (_kx, _xOffsets) = NonZero(kx, cn);
            (_ky, _yRows) = NonZero(ky, 1);
            _delta = (short)delta;
        }

        public static bool Fits(int[] kx, int[] ky, int delta)
        {
            long sx = 0, sy = 0;
            foreach (int k in kx)
            {
                sx += Math.Abs((long)k);
            }

            foreach (int k in ky)
            {
                sy += Math.Abs((long)k);
            }

            return (sx * sy * 255) + Math.Abs((long)delta) <= short.MaxValue;
        }

        private static (short[] Coeffs, int[] Offsets) NonZero(int[] k, int step)
        {
            int count = 0;
            foreach (int v in k)
            {
                count += v != 0 ? 1 : 0;
            }

            // Zero taps pad the list to 3 or 5 entries for the unrolled loops.
            int length = count <= 3 ? 3 : count <= 5 ? 5 : count;
            var coeffs = new short[length];
            var offsets = new int[length];
            for (int i = 0, j = 0; i < k.Length; i++)
            {
                if (k[i] != 0)
                {
                    coeffs[j] = (short)k[i];
                    offsets[j++] = i * step;
                }
            }

            return (coeffs, offsets);
        }

        public void Row(ReadOnlySpan<byte> line, Span<short> row, Span<short> wide)
        {
            ref byte s = ref MemoryMarshal.GetReference(line);
            ref short d = ref MemoryMarshal.GetReference(row);
            int n = row.Length;
            int taps = _kx.Length;
            int i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<byte>.Count)
            {
                if (taps == 3)
                {
                    Row3(ref s, ref d, n, _kx, _xOffsets);
                }
                else if (taps == 5)
                {
                    Row5(ref s, ref d, n, _kx, _xOffsets);
                }
                else
                {
                    RowAny(ref s, ref d, n, _kx, _xOffsets);
                }

                return;
            }

            for (; i < n; i++)
            {
                int sum = 0;
                for (int k = 0; k < taps; k++)
                {
                    sum += _kx[k] * Unsafe.Add(ref s, i + _xOffsets[k]);
                }

                Unsafe.Add(ref d, i) = (short)sum;
            }
        }

        public void Column(ref short rows, int rowStride, Span<short> dst, Span<short> scratch)
        {
            ref short d = ref MemoryMarshal.GetReference(dst);
            int n = dst.Length;
            int taps = _ky.Length;
            int i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<short>.Count)
            {
                if (taps == 3)
                {
                    Column3(ref rows, rowStride, ref d, n, _ky, _yRows, _delta);
                }
                else if (taps == 5)
                {
                    Column5(ref rows, rowStride, ref d, n, _ky, _yRows, _delta);
                }
                else
                {
                    ColumnAny(ref rows, rowStride, ref d, n, _ky, _yRows, _delta);
                }

                return;
            }

            for (; i < n; i++)
            {
                int sum = _delta;
                for (int k = 0; k < taps; k++)
                {
                    sum += _ky[k] * Unsafe.Add(ref rows, (_yRows[k] * rowStride) + i);
                }

                Unsafe.Add(ref d, i) = (short)sum;
            }
        }

        private static void Widen(Vector<byte> v, out Vector<short> lo, out Vector<short> hi)
        {
            Vector.Widen(v, out Vector<ushort> l, out Vector<ushort> h);
            lo = Vector.AsVectorInt16(l);
            hi = Vector.AsVectorInt16(h);
        }

        private static void Row3(ref byte s, ref short d, int n, short[] k, int[] o)
        {
            int vb = Vector<byte>.Count, vs = Vector<short>.Count;
            Vector<short> c0 = new(k[0]), c1 = new(k[1]), c2 = new(k[2]);
            ref byte s0 = ref Unsafe.Add(ref s, o[0]);
            ref byte s1 = ref Unsafe.Add(ref s, o[1]);
            ref byte s2 = ref Unsafe.Add(ref s, o[2]);
            for (int i = 0; i < n; i += vb)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vb);
                Widen(Vector.LoadUnsafe(ref s0, (nuint)i), out Vector<short> l0, out Vector<short> h0);
                Widen(Vector.LoadUnsafe(ref s1, (nuint)i), out Vector<short> l1, out Vector<short> h1);
                Widen(Vector.LoadUnsafe(ref s2, (nuint)i), out Vector<short> l2, out Vector<short> h2);
                ((l0 * c0) + (l1 * c1) + (l2 * c2)).StoreUnsafe(ref d, (nuint)i);
                ((h0 * c0) + (h1 * c1) + (h2 * c2)).StoreUnsafe(ref d, (nuint)(i + vs));
            }
        }

        private static void Row5(ref byte s, ref short d, int n, short[] k, int[] o)
        {
            int vb = Vector<byte>.Count, vs = Vector<short>.Count;
            Vector<short> c0 = new(k[0]), c1 = new(k[1]), c2 = new(k[2]), c3 = new(k[3]), c4 = new(k[4]);
            ref byte s0 = ref Unsafe.Add(ref s, o[0]);
            ref byte s1 = ref Unsafe.Add(ref s, o[1]);
            ref byte s2 = ref Unsafe.Add(ref s, o[2]);
            ref byte s3 = ref Unsafe.Add(ref s, o[3]);
            ref byte s4 = ref Unsafe.Add(ref s, o[4]);
            for (int i = 0; i < n; i += vb)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vb);
                Widen(Vector.LoadUnsafe(ref s0, (nuint)i), out Vector<short> l0, out Vector<short> h0);
                Widen(Vector.LoadUnsafe(ref s1, (nuint)i), out Vector<short> l1, out Vector<short> h1);
                Widen(Vector.LoadUnsafe(ref s2, (nuint)i), out Vector<short> l2, out Vector<short> h2);
                Widen(Vector.LoadUnsafe(ref s3, (nuint)i), out Vector<short> l3, out Vector<short> h3);
                Widen(Vector.LoadUnsafe(ref s4, (nuint)i), out Vector<short> l4, out Vector<short> h4);
                ((l0 * c0) + (l1 * c1) + (l2 * c2) + (l3 * c3) + (l4 * c4)).StoreUnsafe(ref d, (nuint)i);
                ((h0 * c0) + (h1 * c1) + (h2 * c2) + (h3 * c3) + (h4 * c4)).StoreUnsafe(ref d, (nuint)(i + vs));
            }
        }

        private static void RowAny(ref byte s, ref short d, int n, short[] k, int[] o)
        {
            int vb = Vector<byte>.Count, vs = Vector<short>.Count;
            for (int i = 0; i < n; i += vb)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vb);
                Vector<short> a0 = Vector<short>.Zero, a1 = Vector<short>.Zero;
                for (int t = 0; t < k.Length; t++)
                {
                    Widen(Vector.LoadUnsafe(ref s, (nuint)(i + o[t])), out Vector<short> lo, out Vector<short> hi);
                    var c = new Vector<short>(k[t]);
                    a0 += lo * c;
                    a1 += hi * c;
                }

                a0.StoreUnsafe(ref d, (nuint)i);
                a1.StoreUnsafe(ref d, (nuint)(i + vs));
            }
        }

        private static void Column3(ref short rows, int stride, ref short d, int n, short[] k, int[] r, short delta)
        {
            int vs = Vector<short>.Count;
            Vector<short> c0 = new(k[0]), c1 = new(k[1]), c2 = new(k[2]), start = new(delta);
            ref short r0 = ref Unsafe.Add(ref rows, r[0] * stride);
            ref short r1 = ref Unsafe.Add(ref rows, r[1] * stride);
            ref short r2 = ref Unsafe.Add(ref rows, r[2] * stride);
            for (int i = 0; i < n; i += vs)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vs);
                (start + (Vector.LoadUnsafe(ref r0, (nuint)i) * c0) + (Vector.LoadUnsafe(ref r1, (nuint)i) * c1)
                    + (Vector.LoadUnsafe(ref r2, (nuint)i) * c2)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        private static void Column5(ref short rows, int stride, ref short d, int n, short[] k, int[] r, short delta)
        {
            int vs = Vector<short>.Count;
            Vector<short> c0 = new(k[0]), c1 = new(k[1]), c2 = new(k[2]), c3 = new(k[3]), c4 = new(k[4]), start = new(delta);
            ref short r0 = ref Unsafe.Add(ref rows, r[0] * stride);
            ref short r1 = ref Unsafe.Add(ref rows, r[1] * stride);
            ref short r2 = ref Unsafe.Add(ref rows, r[2] * stride);
            ref short r3 = ref Unsafe.Add(ref rows, r[3] * stride);
            ref short r4 = ref Unsafe.Add(ref rows, r[4] * stride);
            for (int i = 0; i < n; i += vs)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vs);
                (start + (Vector.LoadUnsafe(ref r0, (nuint)i) * c0) + (Vector.LoadUnsafe(ref r1, (nuint)i) * c1)
                    + (Vector.LoadUnsafe(ref r2, (nuint)i) * c2) + (Vector.LoadUnsafe(ref r3, (nuint)i) * c3)
                    + (Vector.LoadUnsafe(ref r4, (nuint)i) * c4)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        private static void ColumnAny(ref short rows, int stride, ref short d, int n, short[] k, int[] r, short delta)
        {
            int vs = Vector<short>.Count;
            for (int i = 0; i < n; i += vs)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, n - vs);
                var a = new Vector<short>(delta);
                for (int t = 0; t < k.Length; t++)
                {
                    a += Vector.LoadUnsafe(ref rows, (nuint)((r[t] * stride) + i)) * new Vector<short>(k[t]);
                }

                a.StoreUnsafe(ref d, (nuint)i);
            }
        }
    }

    /// <summary>Float buffer: RowFilter / SymmRowSmallFilter and ColumnFilter / SymmColumn(Small)Filter.</summary>
    private readonly struct FloatKernel<TSrc, TDst> : IRowStreamKernel<TSrc, float, TDst>
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        private readonly float[] _kx;
        private readonly float[] _ky;
        private readonly KernelKind _rtype;
        private readonly int _cn;
        private readonly bool _anySymmetry;
        private readonly bool _symmetrical;
        private readonly bool _small;
        private readonly float _delta;

        public FloatKernel(float[] kx, float[] ky, KernelKind rtype, KernelKind ctype, int cn, float delta)
        {
            _kx = kx;
            _ky = ky;
            _rtype = rtype;
            _cn = cn;
            _anySymmetry = (ctype & (KernelKind.Symmetrical | KernelKind.Asymmetrical)) != 0;
            _symmetrical = (ctype & KernelKind.Symmetrical) != 0;

            // The small-kernel (ksize 3) column variant exists only for float destinations.
            _small = _anySymmetry && ky.Length == 3 && typeof(TDst) == typeof(float);
            _delta = delta;
        }

        public void Row(ReadOnlySpan<TSrc> line, Span<float> row, Span<float> wide)
        {
            if (typeof(TSrc) == typeof(byte))
            {
                // getLinearRowFilter has no symmetric small-kernel variant for 8U→32F.
                RowByteToFloat(MemoryMarshal.Cast<TSrc, byte>(line), row, _kx, _cn);
            }
            else if (typeof(TSrc) == typeof(float))
            {
                RowFloat(MemoryMarshal.Cast<TSrc, float>(line), row, _kx, _rtype, _cn);
            }
            else
            {
                // 16U/16S → 32F use the generic RowFilter only.
                FilterCasts.WidenRow(line, wide);
                RowFloat(wide, row, _kx, KernelKind.General, _cn);
            }
        }

        public void Column(ref float rows, int rowStride, Span<TDst> dst, Span<float> scratch)
        {
            int vf = Vector.IsHardwareAccelerated ? Vector<float>.Count : int.MaxValue;
            Span<float> sums = typeof(TDst) == typeof(float) ? MemoryMarshal.Cast<TDst, float>(dst) : scratch;
            ref float s = ref MemoryMarshal.GetReference(sums);
            int ksize2 = _ky.Length / 2;
            if (_small)
            {
                ColumnSmall(ref rows, rowStride, ref s, _ky[ksize2], _ky[ksize2 + 1], _symmetrical, _delta, vf);
            }
            else
            {
                ColumnGeneric(ref rows, rowStride, ref s, _ky, _anySymmetry, _symmetrical, _delta, vf);
            }

            if (typeof(TDst) != typeof(float))
            {
                FilterCasts.StoreRow(dst, scratch);
            }
        }
    }

    /// <summary>Double buffer: RowFilter&lt;ST, double&gt; and (Symm)ColumnFilter&lt;Cast&lt;double, DT&gt;&gt;.</summary>
    private readonly struct DoubleKernel<TSrc, TDst>(double[] kx, double[] ky, KernelKind ctype, int cn, double delta) : IRowStreamKernel<TSrc, double, TDst>
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        public void Row(ReadOnlySpan<TSrc> line, Span<double> row, Span<double> wide)
        {
            if (typeof(TSrc) == typeof(double))
            {
                RowDouble(MemoryMarshal.Cast<TSrc, double>(line), row, kx, cn);
            }
            else
            {
                FilterCasts.WidenRow(line, wide);
                RowDouble(wide, row, kx, cn);
            }
        }

        public void Column(ref double rows, int rowStride, Span<TDst> dst, Span<double> scratch)
        {
            bool anySymmetry = (ctype & (KernelKind.Symmetrical | KernelKind.Asymmetrical)) != 0;
            bool symmetrical = (ctype & KernelKind.Symmetrical) != 0;
            Span<double> sums = typeof(TDst) == typeof(double) ? MemoryMarshal.Cast<TDst, double>(dst) : scratch;
            ColumnGenericDouble(ref rows, rowStride, ref MemoryMarshal.GetReference(sums), ky, anySymmetry, symmetrical, delta);
            if (typeof(TDst) != typeof(double))
            {
                FilterCasts.StoreRow(dst, scratch);
            }
        }
    }

    // ---------------------------------------------------------------- rows

    /// <summary>
    /// Integer row filter (RowFilter&lt;uchar,int&gt;). Exact integer arithmetic makes OpenCV's
    /// symmetric small-kernel variants equal to this generic loop.
    /// </summary>
    private static void RowByteToInt(ReadOnlySpan<byte> line, Span<int> row, int[] kx, int cn)
    {
        ref byte s = ref MemoryMarshal.GetReference(line);
        ref int d = ref MemoryMarshal.GetReference(row);
        int rowLength = row.Length;
        int ksize = kx.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vi = Vector<int>.Count;
            for (; i <= rowLength - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                Vector<int> a0 = default, a1 = default, a2 = default, a3 = default;
                for (int k = 0; k < ksize; k++)
                {
                    FilterVectors.WidenToInt32(
                        Vector.LoadUnsafe(ref s, (nuint)(i + (k * cn))),
                        out Vector<int> w0, out Vector<int> w1, out Vector<int> w2, out Vector<int> w3);
                    var kk = new Vector<int>(kx[k]);
                    a0 += kk * w0;
                    a1 += kk * w1;
                    a2 += kk * w2;
                    a3 += kk * w3;
                }

                a0.StoreUnsafe(ref d, (nuint)i);
                a1.StoreUnsafe(ref d, (nuint)(i + vi));
                a2.StoreUnsafe(ref d, (nuint)(i + (2 * vi)));
                a3.StoreUnsafe(ref d, (nuint)(i + (3 * vi)));
            }
        }

        for (; i < rowLength; i++)
        {
            int sum = 0;
            for (int k = 0; k < ksize; k++)
            {
                sum += kx[k] * Unsafe.Add(ref s, i + (k * cn));
            }

            Unsafe.Add(ref d, i) = sum;
        }
    }

    /// <summary>Generic float row filter from bytes (RowFilter&lt;uchar,float&gt;).</summary>
    private static void RowByteToFloat(ReadOnlySpan<byte> line, Span<float> row, float[] kx, int cn)
    {
        ref byte s = ref MemoryMarshal.GetReference(line);
        ref float d = ref MemoryMarshal.GetReference(row);
        int rowLength = row.Length;
        int ksize = kx.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vf = Vector<float>.Count;
            for (; i <= rowLength - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                FilterVectors.WidenToSingle(
                    Vector.LoadUnsafe(ref s, (nuint)i),
                    out Vector<float> a0, out Vector<float> a1, out Vector<float> a2, out Vector<float> a3);
                var k0 = new Vector<float>(kx[0]);
                a0 = k0 * a0;
                a1 = k0 * a1;
                a2 = k0 * a2;
                a3 = k0 * a3;
                for (int k = 1; k < ksize; k++)
                {
                    FilterVectors.WidenToSingle(
                        Vector.LoadUnsafe(ref s, (nuint)(i + (k * cn))),
                        out Vector<float> f0, out Vector<float> f1, out Vector<float> f2, out Vector<float> f3);
                    var kk = new Vector<float>(kx[k]);
                    a0 += kk * f0;
                    a1 += kk * f1;
                    a2 += kk * f2;
                    a3 += kk * f3;
                }

                a0.StoreUnsafe(ref d, (nuint)i);
                a1.StoreUnsafe(ref d, (nuint)(i + vf));
                a2.StoreUnsafe(ref d, (nuint)(i + (2 * vf)));
                a3.StoreUnsafe(ref d, (nuint)(i + (3 * vf)));
            }
        }

        for (; i < rowLength; i++)
        {
            float sum = kx[0] * Unsafe.Add(ref s, i);
            for (int k = 1; k < ksize; k++)
            {
                sum += kx[k] * Unsafe.Add(ref s, i + (k * cn));
            }

            Unsafe.Add(ref d, i) = sum;
        }
    }

    /// <summary><c>d[i] = kx[0]·s[i] + kx[1]·s[i + cn] + …</c>, accumulated left to right per element.</summary>
    private static void RowDouble(ReadOnlySpan<double> row, Span<double> dst, double[] kx, int cn)
    {
        ref double s = ref MemoryMarshal.GetReference(row);
        ref double d = ref MemoryMarshal.GetReference(dst);
        int n = dst.Length;
        int ksize = kx.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vd = Vector<double>.Count;
            if (n >= 4 * vd)
            {
                for (; i < n; i += 4 * vd)
                {
                    // The final block overlaps the previous one; it rewrites equal values.
                    i = Math.Min(i, n - (4 * vd));
                    var f = new Vector<double>(kx[0]);
                    Vector<double> a0 = f * Vector.LoadUnsafe(ref s, (nuint)i);
                    Vector<double> a1 = f * Vector.LoadUnsafe(ref s, (nuint)(i + vd));
                    Vector<double> a2 = f * Vector.LoadUnsafe(ref s, (nuint)(i + (2 * vd)));
                    Vector<double> a3 = f * Vector.LoadUnsafe(ref s, (nuint)(i + (3 * vd)));
                    for (int k = 1; k < ksize; k++)
                    {
                        f = new Vector<double>(kx[k]);
                        int o = i + (k * cn);
                        a0 += f * Vector.LoadUnsafe(ref s, (nuint)o);
                        a1 += f * Vector.LoadUnsafe(ref s, (nuint)(o + vd));
                        a2 += f * Vector.LoadUnsafe(ref s, (nuint)(o + (2 * vd)));
                        a3 += f * Vector.LoadUnsafe(ref s, (nuint)(o + (3 * vd)));
                    }

                    a0.StoreUnsafe(ref d, (nuint)i);
                    a1.StoreUnsafe(ref d, (nuint)(i + vd));
                    a2.StoreUnsafe(ref d, (nuint)(i + (2 * vd)));
                    a3.StoreUnsafe(ref d, (nuint)(i + (3 * vd)));
                }

                return;
            }
        }

        for (; i < n; i++)
        {
            double sum = kx[0] * Unsafe.Add(ref s, i);
            for (int k = 1; k < ksize; k++)
            {
                sum += kx[k] * Unsafe.Add(ref s, i + (k * cn));
            }

            Unsafe.Add(ref d, i) = sum;
        }
    }

    private static void RowFloat(ReadOnlySpan<float> row, Span<float> dst, float[] kx, KernelKind rtype, int cn)
    {
        ref float s = ref MemoryMarshal.GetReference(row);
        ref float d = ref MemoryMarshal.GetReference(dst);
        int n = dst.Length;
        int vf = Vector.IsHardwareAccelerated ? Vector<float>.Count : int.MaxValue;
        int ksize = kx.Length;
        int ksize2 = ksize / 2;
        int c0 = ksize2 * cn;
        int cn2 = cn * 2;
        int i = 0;
        bool small = (rtype & (KernelKind.Symmetrical | KernelKind.Asymmetrical)) != 0 && ksize <= 5;

        if (!small)
        {
            if (vf != int.MaxValue && n >= 4 * vf)
            {
                for (; i < n; i += 4 * vf)
                {
                    // The final block overlaps the previous one; it rewrites equal values.
                    i = Math.Min(i, n - (4 * vf));
                    var f = new Vector<float>(kx[0]);
                    Vector<float> a0 = f * L(ref s, i);
                    Vector<float> a1 = f * L(ref s, i + vf);
                    Vector<float> a2 = f * L(ref s, i + (2 * vf));
                    Vector<float> a3 = f * L(ref s, i + (3 * vf));
                    for (int k = 1; k < ksize; k++)
                    {
                        f = new Vector<float>(kx[k]);
                        int o = i + (k * cn);
                        a0 += f * L(ref s, o);
                        a1 += f * L(ref s, o + vf);
                        a2 += f * L(ref s, o + (2 * vf));
                        a3 += f * L(ref s, o + (3 * vf));
                    }

                    a0.StoreUnsafe(ref d, (nuint)i);
                    a1.StoreUnsafe(ref d, (nuint)(i + vf));
                    a2.StoreUnsafe(ref d, (nuint)(i + (2 * vf)));
                    a3.StoreUnsafe(ref d, (nuint)(i + (3 * vf)));
                }
            }

            for (; i <= n - vf; i += vf)
            {
                Vector<float> sum = new Vector<float>(kx[0]) * L(ref s, i);
                for (int k = 1; k < ksize; k++)
                {
                    sum += new Vector<float>(kx[k]) * L(ref s, i + (k * cn));
                }

                sum.StoreUnsafe(ref d, (nuint)i);
            }

            for (; i < n; i++)
            {
                float sum = kx[0] * At(ref s, i);
                for (int k = 1; k < ksize; k++)
                {
                    sum += kx[k] * At(ref s, i + (k * cn));
                }

                At(ref d, i) = sum;
            }

            return;
        }

        if ((rtype & KernelKind.Symmetrical) != 0)
        {
            if (ksize == 1)
            {
                float k0 = kx[0];
                if (k0 == 1f)
                {
                    row[..n].CopyTo(dst);
                    return;
                }

                for (; i <= n - vf; i += vf)
                {
                    (k0 * L(ref s, i)).StoreUnsafe(ref d, (nuint)i);
                }

                for (; i < n; i++)
                {
                    At(ref d, i) = k0 * At(ref s, i);
                }
            }
            else if (ksize == 3)
            {
                float k0 = kx[ksize2];
                float k1 = kx[ksize2 + 1];
                if (k0 == 2f && k1 == 1f)
                {
                    for (; i <= n - vf; i += vf)
                    {
                        int c = i + c0;
                        (L(ref s, c - cn) + (L(ref s, c) * 2f) + L(ref s, c + cn)).StoreUnsafe(ref d, (nuint)i);
                    }

                    for (; i < n; i++)
                    {
                        int c = i + c0;
                        At(ref d, i) = At(ref s, c - cn) + (At(ref s, c) * 2f) + At(ref s, c + cn);
                    }
                }
                else if (k0 == -2f && k1 == 1f)
                {
                    for (; i <= n - vf; i += vf)
                    {
                        int c = i + c0;
                        (L(ref s, c - cn) - (L(ref s, c) * 2f) + L(ref s, c + cn)).StoreUnsafe(ref d, (nuint)i);
                    }

                    for (; i < n; i++)
                    {
                        int c = i + c0;
                        At(ref d, i) = At(ref s, c - cn) - (At(ref s, c) * 2f) + At(ref s, c + cn);
                    }
                }
                else
                {
                    for (; i <= n - vf; i += vf)
                    {
                        int c = i + c0;
                        ((L(ref s, c) * k0) + ((L(ref s, c - cn) + L(ref s, c + cn)) * k1)).StoreUnsafe(ref d, (nuint)i);
                    }

                    for (; i < n; i++)
                    {
                        int c = i + c0;
                        At(ref d, i) = (At(ref s, c) * k0) + ((At(ref s, c - cn) + At(ref s, c + cn)) * k1);
                    }
                }
            }
            else
            {
                // ksize == 5
                float k0 = kx[ksize2];
                float k1 = kx[ksize2 + 1];
                float k2 = kx[ksize2 + 2];
                if (k0 == -2f && k1 == 0f && k2 == 1f)
                {
                    for (; i <= n - vf; i += vf)
                    {
                        int c = i + c0;
                        ((-2f * L(ref s, c)) + L(ref s, c - cn2) + L(ref s, c + cn2)).StoreUnsafe(ref d, (nuint)i);
                    }

                    for (; i < n; i++)
                    {
                        int c = i + c0;
                        At(ref d, i) = (-2f * At(ref s, c)) + At(ref s, c - cn2) + At(ref s, c + cn2);
                    }
                }
                else
                {
                    for (; i <= n - vf; i += vf)
                    {
                        int c = i + c0;
                        ((L(ref s, c) * k0) + ((L(ref s, c - cn) + L(ref s, c + cn)) * k1)
                            + ((L(ref s, c - cn2) + L(ref s, c + cn2)) * k2)).StoreUnsafe(ref d, (nuint)i);
                    }

                    for (; i < n; i++)
                    {
                        int c = i + c0;
                        At(ref d, i) = (At(ref s, c) * k0) + ((At(ref s, c - cn) + At(ref s, c + cn)) * k1)
                            + ((At(ref s, c - cn2) + At(ref s, c + cn2)) * k2);
                    }
                }
            }
        }
        else if (ksize == 3)
        {
            float k0 = kx[ksize2];
            float k1 = kx[ksize2 + 1];
            if (k0 == 0f && k1 == 1f)
            {
                for (; i <= n - vf; i += vf)
                {
                    int c = i + c0;
                    (L(ref s, c + cn) - L(ref s, c - cn)).StoreUnsafe(ref d, (nuint)i);
                }

                for (; i < n; i++)
                {
                    int c = i + c0;
                    At(ref d, i) = At(ref s, c + cn) - At(ref s, c - cn);
                }
            }
            else
            {
                for (; i <= n - vf; i += vf)
                {
                    int c = i + c0;
                    ((L(ref s, c + cn) - L(ref s, c - cn)) * k1).StoreUnsafe(ref d, (nuint)i);
                }

                for (; i < n; i++)
                {
                    int c = i + c0;
                    At(ref d, i) = (At(ref s, c + cn) - At(ref s, c - cn)) * k1;
                }
            }
        }
        else if (ksize == 5)
        {
            float k1 = kx[ksize2 + 1];
            float k2 = kx[ksize2 + 2];
            for (; i <= n - vf; i += vf)
            {
                int c = i + c0;
                (((L(ref s, c + cn) - L(ref s, c - cn)) * k1) + ((L(ref s, c + cn2) - L(ref s, c - cn2)) * k2))
                    .StoreUnsafe(ref d, (nuint)i);
            }

            for (; i < n; i++)
            {
                int c = i + c0;
                At(ref d, i) = ((At(ref s, c + cn) - At(ref s, c - cn)) * k1) + ((At(ref s, c + cn2) - At(ref s, c - cn2)) * k2);
            }
        }
        else
        {
            // ksize == 1 asymmetric: the kernel is [0]; OpenCV's tail
            // loop computes kx[0]*S.
            float k0 = kx[0];
            for (; i <= n - vf; i += vf)
            {
                (k0 * L(ref s, i)).StoreUnsafe(ref d, (nuint)i);
            }

            for (; i < n; i++)
            {
                At(ref d, i) = k0 * At(ref s, i);
            }
        }
    }

    // ------------------------------------------------------------- columns

    /// <summary>
    /// Integer column filter to bytes (FixedPtCastEx: round-half-up shift).
    /// Exact modular integer arithmetic collapses all of OpenCV's symmetric
    /// variants into this generic accumulation.
    /// </summary>
    private static void ColumnIntToByte(ref int r0, int width, int[] ky, int bits, int ideltaAcc, Span<byte> dst)
    {
        int ksize = ky.Length;
        int shiftDelta = bits != 0 ? 1 << (bits - 1) : 0;
        ref byte d = ref MemoryMarshal.GetReference(dst);
        int n = dst.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vi = Vector<int>.Count;
            var start = new Vector<int>(ideltaAcc);
            var round = new Vector<int>(shiftDelta);
            for (; i <= n - Vector<byte>.Count; i += Vector<byte>.Count)
            {
                Vector<int> a0 = start, a1 = start, a2 = start, a3 = start;
                for (int k = 0; k < ksize; k++)
                {
                    ref int r = ref Unsafe.Add(ref r0, k * width);
                    var kk = new Vector<int>(ky[k]);
                    a0 += kk * Vector.LoadUnsafe(ref r, (nuint)i);
                    a1 += kk * Vector.LoadUnsafe(ref r, (nuint)(i + vi));
                    a2 += kk * Vector.LoadUnsafe(ref r, (nuint)(i + (2 * vi)));
                    a3 += kk * Vector.LoadUnsafe(ref r, (nuint)(i + (3 * vi)));
                }

                FilterVectors.SaturateToByte(
                    Vector.ShiftRightArithmetic(a0 + round, bits),
                    Vector.ShiftRightArithmetic(a1 + round, bits),
                    Vector.ShiftRightArithmetic(a2 + round, bits),
                    Vector.ShiftRightArithmetic(a3 + round, bits)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            int sum = ideltaAcc;
            for (int k = 0; k < ksize; k++)
            {
                sum += ky[k] * Unsafe.Add(ref r0, (k * width) + i);
            }

            int shifted = (sum + shiftDelta) >> bits;
            Unsafe.Add(ref d, i) = (byte)Math.Clamp(shifted, 0, 255);
        }
    }

    /// <summary>Integer column filter to shorts (bits == 0, plain saturation).</summary>
    private static void ColumnIntToShort(ref int r0, int width, int[] ky, int ideltaAcc, Span<short> dst)
    {
        int ksize = ky.Length;
        ref short d = ref MemoryMarshal.GetReference(dst);
        int n = dst.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vi = Vector<int>.Count;
            var start = new Vector<int>(ideltaAcc);
            for (; i <= n - Vector<short>.Count; i += Vector<short>.Count)
            {
                Vector<int> a0 = start, a1 = start;
                for (int k = 0; k < ksize; k++)
                {
                    ref int r = ref Unsafe.Add(ref r0, k * width);
                    var kk = new Vector<int>(ky[k]);
                    a0 += kk * Vector.LoadUnsafe(ref r, (nuint)i);
                    a1 += kk * Vector.LoadUnsafe(ref r, (nuint)(i + vi));
                }

                FilterVectors.SaturateToInt16(a0, a1).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            int sum = ideltaAcc;
            for (int k = 0; k < ksize; k++)
            {
                sum += ky[k] * Unsafe.Add(ref r0, (k * width) + i);
            }

            Unsafe.Add(ref d, i) = (short)Math.Clamp(sum, short.MinValue, short.MaxValue);
        }
    }

    /// <summary>
    /// ColumnFilter (no symmetry): <c>s = k0·r0 + delta; s += k·rk</c> for k = 1..ksize-1.
    /// SymmColumnFilter: <c>s = f0·rc + delta</c> (symmetric) or <c>s = delta</c> (antisymmetric),
    /// then <c>s += f·(r[c+k] ± r[c-k])</c> for k = 1..ksize/2. Each element keeps this order.
    /// </summary>
    private static void ColumnGeneric(
        ref float r0, int width, ref float sums, float[] ky, bool anySymmetry, bool symmetrical, float fdelta, int vf)
    {
        int ksize = ky.Length;
        int ksize2 = ksize / 2;
        int i = 0;
        if (vf != int.MaxValue && width >= 4 * vf)
        {
            var vd = new Vector<float>(fdelta);
            for (; i < width; i += 4 * vf)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, width - (4 * vf));
                Vector<float> a0, a1, a2, a3;
                if (!anySymmetry)
                {
                    var f = new Vector<float>(ky[0]);
                    a0 = (f * L(ref r0, i)) + vd;
                    a1 = (f * L(ref r0, i + vf)) + vd;
                    a2 = (f * L(ref r0, i + (2 * vf))) + vd;
                    a3 = (f * L(ref r0, i + (3 * vf))) + vd;
                    for (int k = 1; k < ksize; k++)
                    {
                        ref float sk = ref Unsafe.Add(ref r0, k * width);
                        f = new Vector<float>(ky[k]);
                        a0 += f * L(ref sk, i);
                        a1 += f * L(ref sk, i + vf);
                        a2 += f * L(ref sk, i + (2 * vf));
                        a3 += f * L(ref sk, i + (3 * vf));
                    }
                }
                else
                {
                    if (symmetrical)
                    {
                        ref float sc = ref Unsafe.Add(ref r0, ksize2 * width);
                        var f = new Vector<float>(ky[ksize2]);
                        a0 = (f * L(ref sc, i)) + vd;
                        a1 = (f * L(ref sc, i + vf)) + vd;
                        a2 = (f * L(ref sc, i + (2 * vf))) + vd;
                        a3 = (f * L(ref sc, i + (3 * vf))) + vd;
                    }
                    else
                    {
                        a0 = a1 = a2 = a3 = vd;
                    }

                    for (int k = 1; k <= ksize2; k++)
                    {
                        ref float sp = ref Unsafe.Add(ref r0, (ksize2 + k) * width);
                        ref float sm = ref Unsafe.Add(ref r0, (ksize2 - k) * width);
                        var f = new Vector<float>(ky[ksize2 + k]);
                        if (symmetrical)
                        {
                            a0 += f * (L(ref sp, i) + L(ref sm, i));
                            a1 += f * (L(ref sp, i + vf) + L(ref sm, i + vf));
                            a2 += f * (L(ref sp, i + (2 * vf)) + L(ref sm, i + (2 * vf)));
                            a3 += f * (L(ref sp, i + (3 * vf)) + L(ref sm, i + (3 * vf)));
                        }
                        else
                        {
                            a0 += f * (L(ref sp, i) - L(ref sm, i));
                            a1 += f * (L(ref sp, i + vf) - L(ref sm, i + vf));
                            a2 += f * (L(ref sp, i + (2 * vf)) - L(ref sm, i + (2 * vf)));
                            a3 += f * (L(ref sp, i + (3 * vf)) - L(ref sm, i + (3 * vf)));
                        }
                    }
                }

                a0.StoreUnsafe(ref sums, (nuint)i);
                a1.StoreUnsafe(ref sums, (nuint)(i + vf));
                a2.StoreUnsafe(ref sums, (nuint)(i + (2 * vf)));
                a3.StoreUnsafe(ref sums, (nuint)(i + (3 * vf)));
            }
        }

        for (; i < width; i++)
        {
            float acc;
            if (!anySymmetry)
            {
                acc = (ky[0] * At(ref r0, i)) + fdelta;
                for (int k = 1; k < ksize; k++)
                {
                    acc += ky[k] * At(ref Unsafe.Add(ref r0, k * width), i);
                }
            }
            else
            {
                acc = symmetrical ? (ky[ksize2] * At(ref Unsafe.Add(ref r0, ksize2 * width), i)) + fdelta : fdelta;
                for (int k = 1; k <= ksize2; k++)
                {
                    float p = At(ref Unsafe.Add(ref r0, (ksize2 + k) * width), i);
                    float m = At(ref Unsafe.Add(ref r0, (ksize2 - k) * width), i);
                    acc += ky[ksize2 + k] * (symmetrical ? p + m : p - m);
                }
            }

            At(ref sums, i) = acc;
        }
    }

    /// <summary>The double form of <see cref="ColumnGeneric"/>, with the same per-element order.</summary>
    private static void ColumnGenericDouble(
        ref double r0, int width, ref double sums, double[] ky, bool anySymmetry, bool symmetrical, double delta)
    {
        int ksize = ky.Length;
        int ksize2 = ksize / 2;
        int i = 0;
        int vd = Vector<double>.Count;
        if (Vector.IsHardwareAccelerated && width >= 2 * vd)
        {
            var vdelta = new Vector<double>(delta);
            for (; i < width; i += 2 * vd)
            {
                // The final block overlaps the previous one; it rewrites equal values.
                i = Math.Min(i, width - (2 * vd));
                Vector<double> a0, a1;
                if (!anySymmetry)
                {
                    var f = new Vector<double>(ky[0]);
                    a0 = (f * Vector.LoadUnsafe(ref r0, (nuint)i)) + vdelta;
                    a1 = (f * Vector.LoadUnsafe(ref r0, (nuint)(i + vd))) + vdelta;
                    for (int k = 1; k < ksize; k++)
                    {
                        ref double sk = ref Unsafe.Add(ref r0, k * width);
                        f = new Vector<double>(ky[k]);
                        a0 += f * Vector.LoadUnsafe(ref sk, (nuint)i);
                        a1 += f * Vector.LoadUnsafe(ref sk, (nuint)(i + vd));
                    }
                }
                else
                {
                    if (symmetrical)
                    {
                        ref double sc = ref Unsafe.Add(ref r0, ksize2 * width);
                        var f = new Vector<double>(ky[ksize2]);
                        a0 = (f * Vector.LoadUnsafe(ref sc, (nuint)i)) + vdelta;
                        a1 = (f * Vector.LoadUnsafe(ref sc, (nuint)(i + vd))) + vdelta;
                    }
                    else
                    {
                        a0 = a1 = vdelta;
                    }

                    for (int k = 1; k <= ksize2; k++)
                    {
                        ref double sp = ref Unsafe.Add(ref r0, (ksize2 + k) * width);
                        ref double sm = ref Unsafe.Add(ref r0, (ksize2 - k) * width);
                        var f = new Vector<double>(ky[ksize2 + k]);
                        if (symmetrical)
                        {
                            a0 += f * (Vector.LoadUnsafe(ref sp, (nuint)i) + Vector.LoadUnsafe(ref sm, (nuint)i));
                            a1 += f * (Vector.LoadUnsafe(ref sp, (nuint)(i + vd)) + Vector.LoadUnsafe(ref sm, (nuint)(i + vd)));
                        }
                        else
                        {
                            a0 += f * (Vector.LoadUnsafe(ref sp, (nuint)i) - Vector.LoadUnsafe(ref sm, (nuint)i));
                            a1 += f * (Vector.LoadUnsafe(ref sp, (nuint)(i + vd)) - Vector.LoadUnsafe(ref sm, (nuint)(i + vd)));
                        }
                    }
                }

                a0.StoreUnsafe(ref sums, (nuint)i);
                a1.StoreUnsafe(ref sums, (nuint)(i + vd));
            }

            return;
        }

        for (; i < width; i++)
        {
            double acc;
            if (!anySymmetry)
            {
                acc = (ky[0] * Unsafe.Add(ref r0, i)) + delta;
                for (int k = 1; k < ksize; k++)
                {
                    acc += ky[k] * Unsafe.Add(ref r0, (k * width) + i);
                }
            }
            else
            {
                acc = symmetrical ? (ky[ksize2] * Unsafe.Add(ref r0, (ksize2 * width) + i)) + delta : delta;
                for (int k = 1; k <= ksize2; k++)
                {
                    double p = Unsafe.Add(ref r0, ((ksize2 + k) * width) + i);
                    double m = Unsafe.Add(ref r0, ((ksize2 - k) * width) + i);
                    acc += ky[ksize2 + k] * (symmetrical ? p + m : p - m);
                }
            }

            Unsafe.Add(ref sums, i) = acc;
        }
    }

    /// <summary>SymmColumnSmallFilter (ksize 3, float destination).</summary>
    private static void ColumnSmall(
        ref float r0, int width, ref float sums, float f0, float f1, bool symmetrical, float fdelta, int vf)
    {
        ref float s0 = ref r0;
        ref float s1 = ref Unsafe.Add(ref r0, width);
        ref float s2 = ref Unsafe.Add(ref r0, 2 * width);
        var vd = new Vector<float>(fdelta);
        int i = 0;

        if (symmetrical)
        {
            if (f0 == 2f && f1 == 1f)
            {
                for (; i <= width - vf; i += vf)
                {
                    (L(ref s0, i) + (L(ref s1, i) * 2f) + L(ref s2, i) + vd).StoreUnsafe(ref sums, (nuint)i);
                }

                for (; i < width; i++)
                {
                    At(ref sums, i) = At(ref s0, i) + (At(ref s1, i) * 2f) + At(ref s2, i) + fdelta;
                }
            }
            else if (f0 == -2f && f1 == 1f)
            {
                for (; i <= width - vf; i += vf)
                {
                    (L(ref s0, i) - (L(ref s1, i) * 2f) + L(ref s2, i) + vd).StoreUnsafe(ref sums, (nuint)i);
                }

                for (; i < width; i++)
                {
                    At(ref sums, i) = At(ref s0, i) - (At(ref s1, i) * 2f) + At(ref s2, i) + fdelta;
                }
            }
            else
            {
                for (; i <= width - vf; i += vf)
                {
                    (((L(ref s0, i) + L(ref s2, i)) * f1) + (L(ref s1, i) * f0) + vd).StoreUnsafe(ref sums, (nuint)i);
                }

                for (; i < width; i++)
                {
                    At(ref sums, i) = ((At(ref s0, i) + At(ref s2, i)) * f1) + (At(ref s1, i) * f0) + fdelta;
                }
            }
        }
        else if (f0 == 0f && (f1 == 1f || f1 == -1f))
        {
            ref float lo = ref f1 < 0 ? ref s2 : ref s0;
            ref float hi = ref f1 < 0 ? ref s0 : ref s2;
            for (; i <= width - vf; i += vf)
            {
                (L(ref hi, i) - L(ref lo, i) + vd).StoreUnsafe(ref sums, (nuint)i);
            }

            for (; i < width; i++)
            {
                At(ref sums, i) = At(ref hi, i) - At(ref lo, i) + fdelta;
            }
        }
        else
        {
            for (; i <= width - vf; i += vf)
            {
                (((L(ref s2, i) - L(ref s0, i)) * f1) + vd).StoreUnsafe(ref sums, (nuint)i);
            }

            for (; i < width; i++)
            {
                At(ref sums, i) = ((At(ref s2, i) - At(ref s0, i)) * f1) + fdelta;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> L(ref float s, int i) => Vector.LoadUnsafe(ref s, (nuint)i);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref float At(ref float s, int i) => ref Unsafe.Add(ref s, i);
}
