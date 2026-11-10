// This implementation is ported from the OpenCV reduction functions.
//
// Original sources:
//   modules/core/src/sum.dispatch.cpp (sum)
//   modules/core/src/mean.dispatch.cpp (mean, meanStdDev)
//   modules/core/src/minmax.dispatch.cpp (minMaxLoc)
//   modules/core/src/count_non_zero.dispatch.cpp (countNonZero)
//   modules/core/src/norm.dispatch.cpp (norm)
//   modules/core/src/matrix_operations.cpp (reduce)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV's reductions: <c>sum</c>, <c>mean</c>, <c>meanStdDev</c>,
/// <c>minMaxLoc</c>, <c>countNonZero</c>, <c>norm</c> and <c>reduce</c>.
/// </summary>
/// <remarks>
/// <para>Integer sources accumulate exactly. Floating sources accumulate in double; the
/// summation order differs from OpenCV's vector code, so the last bits of a floating result can
/// differ from OpenCV.</para>
/// <para>A mask is a single-channel <c>byte</c> matrix of the source size; nonzero mask elements
/// select whole pixels.</para>
/// </remarks>
public static class Reductions
{
    /// <summary>Per-channel sum, as OpenCV <c>sum</c>. Channels above 4 are not supported.</summary>
    public static Scalar Sum<T>(Mat<T> source)
        where T : unmanaged
    {
        ValidateSource(source, maxChannels: 4);
        ReductionKernels.SumRows(source, null, squares: false, out double[] sum, out _, out _);
        return ToScalar(sum);
    }

    /// <summary>
    /// Per-channel mean of the selected pixels, as OpenCV <c>mean</c>: <c>sum * (1.0 / n)</c>.
    /// Returns zeros when no pixel is selected.
    /// </summary>
    public static Scalar Mean<T>(Mat<T> source, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateSource(source, maxChannels: 4);
        Elementwise.ValidateMask(source, mask);
        ReductionKernels.SumRows(source, mask, squares: false, out double[] sum, out _, out long n);
        double scale = n != 0 ? 1.0 / n : 0;
        for (int c = 0; c < sum.Length; c++)
        {
            sum[c] *= scale;
        }

        return ToScalar(sum);
    }

    /// <summary>
    /// Per-channel mean and population standard deviation of the selected pixels, as OpenCV
    /// <c>meanStdDev</c>: <c>stddev = sqrt(max(sqsum * (1/n) - mean², 0))</c>.
    /// </summary>
    public static void MeanStdDev<T>(Mat<T> source, out Scalar mean, out Scalar stdDev, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateSource(source, maxChannels: 4);
        Elementwise.ValidateMask(source, mask);
        ReductionKernels.SumRows(source, mask, squares: true, out double[] sum, out double[] sq, out long n);
        double scale = n != 0 ? 1.0 / n : 0;
        var dev = new double[sum.Length];
        for (int c = 0; c < sum.Length; c++)
        {
            sum[c] *= scale;
            dev[c] = Math.Sqrt(Math.Max((sq[c] * scale) - (sum[c] * sum[c]), 0.0));
        }

        mean = ToScalar(sum);
        stdDev = ToScalar(dev);
    }

    /// <summary>
    /// Minimum and maximum values and their first positions in row-major order, as OpenCV
    /// <c>minMaxLoc</c>. The source must have one channel.
    /// </summary>
    /// <remarks>
    /// NaN elements are skipped. If every element is NaN, the result is (+∞, −∞) at (0, 0).
    /// If the mask selects no pixel, the result is (0, 0) at (−1, −1). OpenCV 5 takes its
    /// starting value from the first element, so a leading NaN propagates there; this port
    /// follows the NaN-skipping behaviour of OpenCV 4.
    /// </remarks>
    public static MinMaxLocResult MinMaxLoc<T>(Mat<T> source, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateSource(source, maxChannels: 1);
        Elementwise.ValidateMask(source, mask);
        return ReductionKernels.MinMaxLoc(source, mask);
    }

    /// <summary>
    /// Number of nonzero elements, as OpenCV <c>countNonZero</c>. The source must have one
    /// channel. NaN counts as nonzero; -0.0 counts as zero.
    /// </summary>
    public static int CountNonZero<T>(Mat<T> source)
        where T : unmanaged
    {
        ValidateSource(source, maxChannels: 1);
        return ReductionKernels.CountNonZero(source);
    }

    /// <summary>
    /// Norm of a matrix, as OpenCV <c>norm(src, normType, mask)</c>. <see cref="NormType.Inf"/>
    /// ignores NaN; <see cref="NormType.L1"/> and <see cref="NormType.L2"/> propagate it.
    /// </summary>
    public static double Norm<T>(Mat<T> source, NormType normType = NormType.L2, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateSource(source, maxChannels: int.MaxValue);
        Elementwise.ValidateMask(source, mask);
        int type = ValidateNormType(normType, allowRelative: false);
        return Finish(ReductionKernels.Norm(source, null, type, mask), type);
    }

    /// <summary>
    /// Norm of the difference <c>a - b</c>, as OpenCV <c>norm(a, b, normType, mask)</c>. With
    /// <see cref="NormType.Relative"/>, the result is divided by <c>norm(b) + DBL_EPSILON</c>.
    /// </summary>
    /// <remarks>
    /// For <c>int</c> the difference wraps in 32 bits for <see cref="NormType.Inf"/> and
    /// <see cref="NormType.L1"/>, as in OpenCV; <see cref="NormType.L2"/> subtracts in double.
    /// </remarks>
    public static double Norm<T>(Mat<T> a, Mat<T> b, NormType normType = NormType.L2, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ValidateSource(a, maxChannels: int.MaxValue);
        Elementwise.ValidateSameShape(a, b, nameof(b));
        Elementwise.ValidateMask(a, mask);
        int type = ValidateNormType(normType, allowRelative: true);
        double diff = Finish(ReductionKernels.Norm(a, b, type, mask), type);
        if ((normType & NormType.Relative) == 0)
        {
            return diff;
        }

        return diff / (Finish(ReductionKernels.Norm(b, null, type, mask), type) + Conversions.DblEpsilon);
    }

    /// <summary>
    /// Reduces a matrix to one row (<paramref name="dim"/> = 0) or one column
    /// (<paramref name="dim"/> = 1), as OpenCV <c>reduce</c>. Channels reduce independently.
    /// </summary>
    /// <remarks>
    /// <para>Supported pairs: <see cref="ReduceType.Sum"/>, <see cref="ReduceType.Avg"/> and
    /// <see cref="ReduceType.Sum2"/>: <c>byte</c> to <c>int</c>/<c>float</c>/<c>double</c>,
    /// <c>ushort</c> and <c>short</c> to <c>float</c>/<c>double</c>, <c>float</c> to
    /// <c>float</c>/<c>double</c>, <c>double</c> to <c>double</c>; <see cref="ReduceType.Avg"/>
    /// also <c>byte</c> to <c>byte</c>. <see cref="ReduceType.Max"/> and
    /// <see cref="ReduceType.Min"/>: same type for <c>byte</c>, <c>ushort</c>, <c>short</c>,
    /// <c>float</c> and <c>double</c>.</para>
    /// <para>Accumulation: integer sources sum exactly (16-bit sources to <c>float</c> convert
    /// the exact sum once); <c>float</c> to <c>float</c> sums in float in index order; double
    /// destinations sum in double. <see cref="ReduceType.Avg"/> converts the sum with scale
    /// <c>1/N</c> as <see cref="Conversions.ConvertTo{TSrc, TDst}"/>.</para>
    /// </remarks>
    public static Mat<TDst> Reduce<TSrc, TDst>(Mat<TSrc> source, int dim, ReduceType op)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (dim is not (0 or 1))
        {
            throw new ArgumentOutOfRangeException(nameof(dim));
        }

        int n = dim == 0 ? source.Height : source.Width;
        int outW = dim == 0 ? source.Width : 1;
        int outH = dim == 0 ? 1 : source.Height;

        if (op == ReduceType.Avg)
        {
            if (IsSmallerThanInt<TSrc>() && IsSmallerThanInt<TDst>())
            {
                EnsureReduceSupported<TSrc, int>(ReduceType.Sum);
                Mat<int> sum = ReduceSum<TSrc, int>(source, dim, squares: false, outW, outH);
                return Conversions.ConvertTo<int, TDst>(sum, 1.0 / n);
            }

            EnsureReduceSupported<TSrc, TDst>(ReduceType.Sum);
            Mat<TDst> total = ReduceSum<TSrc, TDst>(source, dim, squares: false, outW, outH);
            return Conversions.ConvertTo<TDst, TDst>(total, 1.0 / n);
        }

        EnsureReduceSupported<TSrc, TDst>(op);
        return op switch
        {
            ReduceType.Sum => ReduceSum<TSrc, TDst>(source, dim, squares: false, outW, outH),
            ReduceType.Sum2 => ReduceSum<TSrc, TDst>(source, dim, squares: true, outW, outH),
            ReduceType.Max => (Mat<TDst>)(object)ReduceMinMax<TSrc, MaxOp>(source, dim, outW, outH),
            _ => (Mat<TDst>)(object)ReduceMinMax<TSrc, MinOp>(source, dim, outW, outH),
        };
    }

    private static Mat<TDst> ReduceSum<TSrc, TDst>(Mat<TSrc> source, int dim, bool squares, int outW, int outH)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        int cn = source.Channels;
        var dst = new Mat<TDst>(outW, outH, cn, clear: false);
        AccKind kind = typeof(TDst) == typeof(double)
            ? AccKind.Double
            : typeof(TDst) == typeof(float) &&
              (typeof(TSrc) == typeof(float) || (squares && typeof(TSrc) != typeof(byte)))
                ? AccKind.Float
                : AccKind.Long;

        if (dim == 0)
        {
            var acc = new Accumulator(source.RowLength, kind);
            for (int y = 0; y < source.Height; y++)
            {
                ReadOnlySpan<TSrc> row = source.GetReadOnlyRowSpan(y);
                for (int i = 0; i < row.Length; i++)
                {
                    acc.Add(i, row[i], squares);
                }
            }

            acc.Store(dst.GetRowSpan(0));
            return dst;
        }

        var rowAcc = new Accumulator(cn, kind);
        for (int y = 0; y < source.Height; y++)
        {
            rowAcc.Clear();
            ReadOnlySpan<TSrc> row = source.GetReadOnlyRowSpan(y);
            for (int i = 0, c = 0; i < row.Length; i++, c = c + 1 == cn ? 0 : c + 1)
            {
                rowAcc.Add(c, row[i], squares);
            }

            rowAcc.Store(dst.GetRowSpan(y));
        }

        return dst;
    }

    private enum AccKind
    {
        Long,
        Float,
        Double,
    }

    /// <summary>Sum accumulators in the type OpenCV uses for a (source, destination) pair.</summary>
    private sealed class Accumulator(int length, AccKind kind)
    {
        private readonly long[] _long = kind == AccKind.Long ? new long[length] : [];
        private readonly float[] _float = kind == AccKind.Float ? new float[length] : [];
        private readonly double[] _double = kind == AccKind.Double ? new double[length] : [];

        public void Clear()
        {
            Array.Clear(_long);
            Array.Clear(_float);
            Array.Clear(_double);
        }

        public void Add<T>(int i, T value, bool squares)
            where T : unmanaged
        {
            switch (kind)
            {
                case AccKind.Long:
                    long l = Saturate.ToInt(value);
                    _long[i] += squares ? l * l : l;
                    break;
                case AccKind.Float:
                    float f = Saturate.ToFloat(value);
                    _float[i] += squares ? f * f : f;
                    break;
                default:
                    double d = Saturate.ToDouble(value);
                    _double[i] += squares ? d * d : d;
                    break;
            }
        }

        public void Store<TDst>(Span<TDst> dst)
            where TDst : unmanaged
        {
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = kind switch
                {
                    AccKind.Float => Saturate.FromFloat<TDst>(_float[i]),
                    AccKind.Double => Saturate.FromDouble<TDst>(_double[i]),
                    _ => typeof(TDst) == typeof(int)
                        ? Saturate.FromInt<TDst>(unchecked((int)_long[i]))
                        : Saturate.FromDouble<TDst>(_long[i]),
                };
            }
        }
    }

    private static Mat<T> ReduceMinMax<T, TOp>(Mat<T> source, int dim, int outW, int outH)
        where T : unmanaged
        where TOp : struct, IMinMaxOp
    {
        int cn = source.Channels;
        var dst = new Mat<T>(outW, outH, cn, clear: false);
        if (dim == 0)
        {
            Span<T> acc = dst.GetRowSpan(0);
            source.GetReadOnlyRowSpan(0).CopyTo(acc);
            for (int y = 1; y < source.Height; y++)
            {
                ReadOnlySpan<T> row = source.GetReadOnlyRowSpan(y);
                for (int i = 0; i < acc.Length; i++)
                {
                    acc[i] = TOp.Apply(acc[i], row[i]);
                }
            }

            return dst;
        }

        for (int y = 0; y < source.Height; y++)
        {
            ReadOnlySpan<T> row = source.GetReadOnlyRowSpan(y);
            Span<T> d = dst.GetRowSpan(y);
            for (int c = 0; c < cn; c++)
            {
                T acc = row[c];
                for (int x = 1; x < source.Width; x++)
                {
                    acc = TOp.Apply(acc, row[(x * cn) + c]);
                }

                d[c] = acc;
            }
        }

        return dst;
    }

    private static void EnsureReduceSupported<TSrc, TDst>(ReduceType op)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        bool ok = op switch
        {
            ReduceType.Sum or ReduceType.Sum2 =>
                (typeof(TSrc) == typeof(byte) && (typeof(TDst) == typeof(int) || typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double))) ||
                ((typeof(TSrc) == typeof(ushort) || typeof(TSrc) == typeof(short)) && (typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double))) ||
                (typeof(TSrc) == typeof(float) && (typeof(TDst) == typeof(float) || typeof(TDst) == typeof(double))) ||
                (typeof(TSrc) == typeof(double) && typeof(TDst) == typeof(double)),
            ReduceType.Max or ReduceType.Min =>
                typeof(TSrc) == typeof(TDst) &&
                (typeof(TSrc) == typeof(byte) || typeof(TSrc) == typeof(ushort) || typeof(TSrc) == typeof(short) ||
                 typeof(TSrc) == typeof(float) || typeof(TSrc) == typeof(double)),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };

        if (!ok)
        {
            throw new NotSupportedException(
                $"reduce {op} from '{typeof(TSrc).Name}' to '{typeof(TDst).Name}' is not supported.");
        }
    }

    private static bool IsSmallerThanInt<T>()
        where T : unmanaged
        => Saturate.IsSmallInteger<T>();

    private static void ValidateSource<T>(Mat<T> source, int maxChannels)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Saturate.EnsureSupported<T>();
        if (source.Channels > maxChannels)
        {
            throw new ArgumentException(
                maxChannels == 1
                    ? "The source must have one channel."
                    : $"The source must have at most {maxChannels} channels.",
                nameof(source));
        }
    }

    private static int ValidateNormType(NormType normType, bool allowRelative)
    {
        int type = (int)normType & 7;
        int flags = (int)normType & ~7;
        if (type is not (1 or 2 or 4 or 5) || (flags != 0 && !(allowRelative && flags == (int)NormType.Relative)))
        {
            throw new ArgumentOutOfRangeException(nameof(normType));
        }

        return type;
    }

    private static double Finish(double value, int type) => type == 4 ? Math.Sqrt(value) : value;

    private static Scalar ToScalar(double[] v) => new(
        v.Length > 0 ? v[0] : 0,
        v.Length > 1 ? v[1] : 0,
        v.Length > 2 ? v[2] : 0,
        v.Length > 3 ? v[3] : 0);
}
