// This implementation is ported from the OpenCV integral image functions.
//
// Original source:
//   modules/imgproc/src/sumpixels.dispatch.cpp (integral)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed port of OpenCV's <c>integral</c>: sum, squared sum and 45-degree tilted sum images.
/// The source is not modified.
/// </summary>
/// <remarks>
/// <para>Outputs have <c>(width + 1) x (height + 1)</c> pixels and the source channel count. Row 0
/// and column 0 are zero; <c>sum(X, Y)</c> is the sum of the source pixels above and to the left of
/// <c>(X, Y)</c>. The tilted sum <c>tilted(X, Y)</c> is the sum of <c>src(x, y)</c> over
/// <c>y &lt; Y</c> and <c>|x - X + 1| &lt;= Y - 1 - y</c>.</para>
/// <para>Supported (source, sum, squared sum) element types, as OpenCV:
/// <c>byte</c>: (<c>int</c>; <c>double</c>, <c>float</c>, <c>int</c>), (<c>float</c>;
/// <c>double</c>, <c>float</c>), (<c>double</c>; <c>double</c>);
/// <c>ushort</c> and <c>short</c>: (<c>double</c>; <c>double</c>);
/// <c>float</c>: (<c>float</c>; <c>double</c>, <c>float</c>), (<c>double</c>; <c>double</c>);
/// <c>double</c>: (<c>double</c>; <c>double</c>). The overloads without a squared sum validate the
/// pair with a <c>double</c> squared sum. Other combinations throw
/// <see cref="NotSupportedException"/>.</para>
/// <para>Accumulation follows the OpenCV scalar loop: <c>s += src[x]; sum[y+1][x+1] = sum[y][x+1] + s</c>
/// in the sum type, and <c>sq += (TSqSum)src * src</c> for the squared sum. <c>int</c> sums wrap on
/// overflow. For <c>byte</c> to <c>float</c> with row sums above 2^24, OpenCV 5.0 without IPP uses
/// a vector block order whose rounding depends on the CPU; this port keeps the scalar order.</para>
/// <para>Overloads without destinations return new matrices. Overloads with destinations write
/// every element of them. A destination must have <c>(width + 1) x (height + 1)</c> pixels and the
/// source channel count; it may be an ROI. Destinations must not share memory with the source or
/// with each other.</para>
/// </remarks>
public static class IntegralImages
{
    /// <summary>Computes the sum image, as OpenCV <c>integral(src, sum, sdepth)</c>.</summary>
    public static Mat<TSum> Integral<TSrc, TSum>(Mat<TSrc> source)
        where TSrc : unmanaged
        where TSum : unmanaged
    {
        Validate<TSrc, TSum, double>(source);
        Mat<TSum> sum = Create<TSum>(source.Width, source.Height, source.Channels);
        Dispatch<TSrc, TSum, double>(source, sum, null, null);
        return sum;
    }

    /// <summary>
    /// Computes the sum and squared sum images, as OpenCV
    /// <c>integral(src, sum, sqsum, sdepth, sqdepth)</c>.
    /// </summary>
    public static Mat<TSum> Integral<TSrc, TSum, TSqSum>(Mat<TSrc> source, out Mat<TSqSum> sqsum)
        where TSrc : unmanaged
        where TSum : unmanaged
        where TSqSum : unmanaged
    {
        Validate<TSrc, TSum, TSqSum>(source);
        Mat<TSum> sum = Create<TSum>(source.Width, source.Height, source.Channels);
        sqsum = Create<TSqSum>(source.Width, source.Height, source.Channels);
        Dispatch<TSrc, TSum, TSqSum>(source, sum, sqsum, null);
        return sum;
    }

    /// <summary>
    /// Computes the sum, squared sum and tilted sum images, as OpenCV
    /// <c>integral(src, sum, sqsum, tilted, sdepth, sqdepth)</c>. The tilted sum has the sum type
    /// and is computed on one thread.
    /// </summary>
    public static Mat<TSum> Integral<TSrc, TSum, TSqSum>(Mat<TSrc> source, out Mat<TSqSum> sqsum, out Mat<TSum> tilted)
        where TSrc : unmanaged
        where TSum : unmanaged
        where TSqSum : unmanaged
    {
        Validate<TSrc, TSum, TSqSum>(source);
        Mat<TSum> sum = Create<TSum>(source.Width, source.Height, source.Channels);
        sqsum = Create<TSqSum>(source.Width, source.Height, source.Channels);
        tilted = Create<TSum>(source.Width, source.Height, source.Channels);
        Dispatch<TSrc, TSum, TSqSum>(source, sum, sqsum, tilted);
        return sum;
    }

    /// <summary>
    /// Computes the sum image into <paramref name="sum"/>, as OpenCV <c>integral(src, sum, sdepth)</c>
    /// with a preallocated <c>sum</c>.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="sum"/> does not have <c>(width + 1) x (height + 1)</c> pixels and the source channel count, or shares memory with the source.</exception>
    /// <exception cref="NotSupportedException">The type combination is not supported.</exception>
    public static void Integral<TSrc, TSum>(Mat<TSrc> source, Mat<TSum> sum)
        where TSrc : unmanaged
        where TSum : unmanaged
    {
        Validate<TSrc, TSum, double>(source);
        ValidateDestination(source, sum, nameof(sum));
        Dispatch<TSrc, TSum, double>(source, sum, null, null);
    }

    /// <summary>
    /// Computes the sum and squared sum images into <paramref name="sum"/> and
    /// <paramref name="sqsum"/>, as OpenCV <c>integral(src, sum, sqsum, sdepth, sqdepth)</c> with
    /// preallocated outputs.
    /// </summary>
    /// <exception cref="ArgumentException">A destination does not have <c>(width + 1) x (height + 1)</c> pixels and the source channel count, or destinations share memory.</exception>
    /// <exception cref="NotSupportedException">The type combination is not supported.</exception>
    public static void Integral<TSrc, TSum, TSqSum>(Mat<TSrc> source, Mat<TSum> sum, Mat<TSqSum> sqsum)
        where TSrc : unmanaged
        where TSum : unmanaged
        where TSqSum : unmanaged
    {
        Validate<TSrc, TSum, TSqSum>(source);
        ValidateDestination(source, sum, nameof(sum));
        ValidateDestination(source, sqsum, nameof(sqsum));
        ValidateDistinct(sum, sqsum, nameof(sqsum));
        Dispatch<TSrc, TSum, TSqSum>(source, sum, sqsum, null);
    }

    /// <summary>
    /// Computes the sum, squared sum and tilted sum images into <paramref name="sum"/>,
    /// <paramref name="sqsum"/> and <paramref name="tilted"/>, as OpenCV
    /// <c>integral(src, sum, sqsum, tilted, sdepth, sqdepth)</c> with preallocated outputs. The
    /// tilted sum is computed on one thread.
    /// </summary>
    /// <exception cref="ArgumentException">A destination does not have <c>(width + 1) x (height + 1)</c> pixels and the source channel count, or destinations share memory.</exception>
    /// <exception cref="NotSupportedException">The type combination is not supported.</exception>
    public static void Integral<TSrc, TSum, TSqSum>(Mat<TSrc> source, Mat<TSum> sum, Mat<TSqSum> sqsum, Mat<TSum> tilted)
        where TSrc : unmanaged
        where TSum : unmanaged
        where TSqSum : unmanaged
    {
        Validate<TSrc, TSum, TSqSum>(source);
        ValidateDestination(source, sum, nameof(sum));
        ValidateDestination(source, sqsum, nameof(sqsum));
        ValidateDestination(source, tilted, nameof(tilted));
        ValidateDistinct(sum, sqsum, nameof(sqsum));
        ValidateDistinct(sum, tilted, nameof(tilted));
        ValidateDistinct(sqsum, tilted, nameof(tilted));
        Dispatch<TSrc, TSum, TSqSum>(source, sum, sqsum, tilted);
    }

    private static void ValidateDestination<TSrc, TDst>(Mat<TSrc> source, Mat<TDst> destination, string paramName)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(destination, paramName);
        if (destination.Width != source.Width + 1 || destination.Height != source.Height + 1
            || destination.Channels != source.Channels)
        {
            throw new ArgumentException(
                $"The destination must have {source.Width + 1} x {source.Height + 1} pixels and {source.Channels} channels.",
                paramName);
        }

        ValidateDistinct(source, destination, paramName);
    }

    private static void ValidateDistinct<TA, TB>(Mat<TA> a, Mat<TB> b, string paramName)
        where TA : unmanaged
        where TB : unmanaged
    {
        if (ReferenceEquals(a.DangerousGetArray(), b.DangerousGetArray()))
        {
            throw new ArgumentException("The matrices must not share memory.", paramName);
        }
    }

    private static Mat<T> Create<T>(int width, int height, int channels)
        where T : unmanaged
        => new(checked(width + 1), checked(height + 1), channels, clear: false);

    private static bool IsSupported<TSrc, TSum, TSq>()
    {
        static bool Is<TA, TB>() => typeof(TA) == typeof(TB);

        return Is<TSrc, byte>()
            ? (Is<TSum, int>() && (Is<TSq, double>() || Is<TSq, float>() || Is<TSq, int>()))
              || (Is<TSum, float>() && (Is<TSq, double>() || Is<TSq, float>()))
              || (Is<TSum, double>() && Is<TSq, double>())
            : Is<TSrc, ushort>() || Is<TSrc, short>() || Is<TSrc, double>()
                ? Is<TSum, double>() && Is<TSq, double>()
                : Is<TSrc, float>()
                  && ((Is<TSum, float>() && (Is<TSq, double>() || Is<TSq, float>())) || (Is<TSum, double>() && Is<TSq, double>()));
    }

    private static void Validate<TSrc, TSum, TSq>(Mat<TSrc> source)
        where TSrc : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!IsSupported<TSrc, TSum, TSq>())
        {
            throw new NotSupportedException(
                $"integral from '{typeof(TSrc).Name}' to sum '{typeof(TSum).Name}' and squared sum '{typeof(TSq).Name}' is not supported.");
        }
    }

    private static void Dispatch<TSrc, TSum, TSq>(Mat<TSrc> src, Mat<TSum> sum, Mat<TSq>? sq, Mat<TSum>? tilted)
        where TSrc : unmanaged
        where TSum : unmanaged
        where TSq : unmanaged
    {
        static bool Is<TA, TB>() => typeof(TA) == typeof(TB);

        if (Is<TSrc, byte>() && Is<TSum, int>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<byte>(src), As<int>(sum), AsOrNull<double>(sq), AsOrNull<int>(tilted));
        }
        else if (Is<TSrc, byte>() && Is<TSum, int>() && Is<TSq, float>())
        {
            IntegralKernels.Run(As<byte>(src), As<int>(sum), AsOrNull<float>(sq), AsOrNull<int>(tilted));
        }
        else if (Is<TSrc, byte>() && Is<TSum, int>() && Is<TSq, int>())
        {
            IntegralKernels.Run(As<byte>(src), As<int>(sum), AsOrNull<int>(sq), AsOrNull<int>(tilted));
        }
        else if (Is<TSrc, byte>() && Is<TSum, float>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<byte>(src), As<float>(sum), AsOrNull<double>(sq), AsOrNull<float>(tilted));
        }
        else if (Is<TSrc, byte>() && Is<TSum, float>() && Is<TSq, float>())
        {
            IntegralKernels.Run(As<byte>(src), As<float>(sum), AsOrNull<float>(sq), AsOrNull<float>(tilted));
        }
        else if (Is<TSrc, byte>() && Is<TSum, double>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<byte>(src), As<double>(sum), AsOrNull<double>(sq), AsOrNull<double>(tilted));
        }
        else if (Is<TSrc, ushort>() && Is<TSum, double>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<ushort>(src), As<double>(sum), AsOrNull<double>(sq), AsOrNull<double>(tilted));
        }
        else if (Is<TSrc, short>() && Is<TSum, double>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<short>(src), As<double>(sum), AsOrNull<double>(sq), AsOrNull<double>(tilted));
        }
        else if (Is<TSrc, float>() && Is<TSum, float>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<float>(src), As<float>(sum), AsOrNull<double>(sq), AsOrNull<float>(tilted));
        }
        else if (Is<TSrc, float>() && Is<TSum, float>() && Is<TSq, float>())
        {
            IntegralKernels.Run(As<float>(src), As<float>(sum), AsOrNull<float>(sq), AsOrNull<float>(tilted));
        }
        else if (Is<TSrc, float>() && Is<TSum, double>() && Is<TSq, double>())
        {
            IntegralKernels.Run(As<float>(src), As<double>(sum), AsOrNull<double>(sq), AsOrNull<double>(tilted));
        }
        else
        {
            IntegralKernels.Run(As<double>(src), As<double>(sum), AsOrNull<double>(sq), AsOrNull<double>(tilted));
        }
    }

    private static Mat<T> As<T>(object m)
        where T : unmanaged
        => (Mat<T>)m;

    private static Mat<T>? AsOrNull<T>(object? m)
        where T : unmanaged
        => (Mat<T>?)m;
}
