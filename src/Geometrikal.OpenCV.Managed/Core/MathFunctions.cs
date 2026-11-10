// This file is a managed port of parts of OpenCV modules/core/src/mathfuncs.cpp
// (magnitude, phase, cartToPolar, polarToCart, exp, log, pow, sqrt).
// OpenCV is licensed under the Apache License 2.0. Copyright (C) 2000-2024, Intel Corporation,
// all rights reserved, and OpenCV contributors.

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Per-element math functions, as OpenCV <c>magnitude</c>, <c>phase</c>, <c>cartToPolar</c>,
/// <c>polarToCart</c>, <c>exp</c>, <c>log</c>, <c>pow</c> and <c>sqrt</c>. No function modifies
/// its sources; each returns new matrices of the source size and channel count. Channels are
/// processed as independent elements.
/// </summary>
/// <remarks>
/// <para>Element blocks follow OpenCV: when all inputs are continuous, a matrix is one plane;
/// otherwise each row is a plane. <c>cartToPolar</c> and <c>polarToCart</c> process each plane in
/// blocks of 1024 elements. Kernels whose vector and scalar forms differ (<c>polarToCart</c>,
/// <c>pow(x, −0.5)</c>) follow the SSE-baseline split of <see cref="MathKernels"/>, so their
/// results depend on the block structure, as in OpenCV.</para>
/// <para>Angles are in [0, 360) degrees or [0, 2π) radians. <c>phase</c> and <c>cartToPolar</c> use
/// the OpenCV polynomial <c>atan</c> (about 0.3° accuracy before the radian scale is applied,
/// 0.01° typical); <c>double</c> inputs are computed in <c>float</c>, as OpenCV.</para>
/// </remarks>
public static class MathFunctions
{
    private const int BlockSize = 1024;
    private const int DoubleBlock = 128;

    /// <summary>Magnitude of 2-D vectors: <c>sqrt(x² + y²)</c>, as OpenCV <c>magnitude</c>.</summary>
    /// <param name="x"><c>float</c> or <c>double</c> X components.</param>
    /// <param name="y">Y components of the type and size of <paramref name="x"/>.</param>
    public static Mat<T> Magnitude<T>(Mat<T> x, Mat<T> y)
        where T : unmanaged
    {
        ValidatePair(x, y);
        var mag = new Mat<T>(x.Width, x.Height, x.Channels, clear: false);
        ForEachPlane(x, y, mag, null, (a, b, m, _) =>
        {
            if (typeof(T) == typeof(float))
            {
                FastMath.Magnitude32f(Cast<float>(a), Cast<float>(b), CastMut<float>(m));
            }
            else
            {
                var xd = Cast<double>(a);
                var yd = Cast<double>(b);
                var md = CastMut<double>(m);
                for (int i = 0; i < md.Length; i++)
                {
                    md[i] = Math.Sqrt((xd[i] * xd[i]) + (yd[i] * yd[i]));
                }
            }
        });
        return mag;
    }

    /// <summary>Angle of 2-D vectors, as OpenCV <c>phase</c>.</summary>
    /// <param name="x"><c>float</c> or <c>double</c> X components.</param>
    /// <param name="y">Y components of the type and size of <paramref name="x"/>.</param>
    /// <param name="angleInDegrees">Degrees in [0, 360) when true; radians in [0, 2π) when false.</param>
    public static Mat<T> Phase<T>(Mat<T> x, Mat<T> y, bool angleInDegrees = false)
        where T : unmanaged
    {
        ValidatePair(x, y);
        var angle = new Mat<T>(x.Width, x.Height, x.Channels, clear: false);
        ForEachPlane(x, y, angle, null, (a, b, r, _) =>
        {
            if (typeof(T) == typeof(float))
            {
                MathKernels.FastAtan32f(Cast<float>(b), Cast<float>(a), CastMut<float>(r), angleInDegrees);
            }
            else
            {
                ThroughFloat(Cast<double>(b), Cast<double>(a), CastMut<double>(r), null, (yf, xf, rf, _) => MathKernels.FastAtan32f(yf, xf, rf, angleInDegrees));
            }
        });
        return angle;
    }

    /// <summary>Magnitude and angle of 2-D vectors, as OpenCV <c>cartToPolar</c>.</summary>
    /// <param name="x"><c>float</c> or <c>double</c> X components.</param>
    /// <param name="y">Y components of the type and size of <paramref name="x"/>.</param>
    /// <param name="angleInDegrees">Degrees in [0, 360) when true; radians in [0, 2π) when false.</param>
    public static (Mat<T> Magnitude, Mat<T> Angle) CartToPolar<T>(Mat<T> x, Mat<T> y, bool angleInDegrees = false)
        where T : unmanaged
    {
        ValidatePair(x, y);
        var mag = new Mat<T>(x.Width, x.Height, x.Channels, clear: false);
        var angle = new Mat<T>(x.Width, x.Height, x.Channels, clear: false);
        int block = ChannelBlock(x.Channels);
        ForEachPlane(x, y, mag, angle, (a, b, m, r) =>
        {
            int total = m.Length / System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            for (int j = 0; j < total; j += block)
            {
                int len = Math.Min(block, total - j);
                if (typeof(T) == typeof(float))
                {
                    MathKernels.CartToPolar32f(Cast<float>(a).Slice(j, len), Cast<float>(b).Slice(j, len),
                        CastMut<float>(m).Slice(j, len), CastMut<float>(r).Slice(j, len), angleInDegrees);
                }
                else
                {
                    ThroughFloat(Cast<double>(a).Slice(j, len), Cast<double>(b).Slice(j, len), CastMut<double>(m).Slice(j, len),
                        CastMut<double>(r).Slice(j, len), (xf, yf, mf, rf) => MathKernels.CartToPolar32f(xf, yf, mf, rf, angleInDegrees));
                }
            }
        });
        return (mag, angle);
    }

    /// <summary>X and Y components from magnitudes and angles, as OpenCV <c>polarToCart</c>.</summary>
    /// <param name="magnitude">Magnitudes of the type and size of <paramref name="angle"/>, or null for unit vectors.</param>
    /// <param name="angle"><c>float</c> or <c>double</c> angles.</param>
    /// <param name="angleInDegrees">The angles are in degrees when true and in radians when false.</param>
    /// <remarks>
    /// Blocks of at least 8 elements use the OpenCV 5.0 vector <c>sin</c>/<c>cos</c> (Cephes
    /// polynomials); shorter blocks use the 64-entry table of the scalar code. Both are accurate to
    /// about 1e-7 relative to the magnitude, but they differ in the last bits.
    /// </remarks>
    public static (Mat<T> X, Mat<T> Y) PolarToCart<T>(Mat<T>? magnitude, Mat<T> angle, bool angleInDegrees = false)
        where T : unmanaged
    {
        ValidateFloating(angle, nameof(angle));
        if (magnitude is not null)
        {
            ValidatePair(magnitude, angle);
        }

        var xOut = new Mat<T>(angle.Width, angle.Height, angle.Channels, clear: false);
        var yOut = new Mat<T>(angle.Width, angle.Height, angle.Channels, clear: false);
        int block = ChannelBlock(angle.Channels);
        ForEachPlane(magnitude ?? angle, angle, xOut, yOut, (m, a, cx, cy) =>
        {
            int total = a.Length / System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            for (int j = 0; j < total; j += block)
            {
                int len = Math.Min(block, total - j);
                if (typeof(T) == typeof(float))
                {
                    MathKernels.SinCos32f(Cast<float>(m).Slice(j, len), magnitude is not null, Cast<float>(a).Slice(j, len),
                        CastMut<float>(cx).Slice(j, len), CastMut<float>(cy).Slice(j, len), angleInDegrees);
                }
                else
                {
                    PolarToCart64f(Cast<double>(m).Slice(j, len), magnitude is not null, Cast<double>(a).Slice(j, len),
                        CastMut<double>(cx).Slice(j, len), CastMut<double>(cy).Slice(j, len), angleInDegrees);
                }
            }
        });
        return (xOut, yOut);
    }

    /// <summary>Exponent of each element, as OpenCV <c>exp</c> (the C runtime <c>exp</c>, as the MSVC build).</summary>
    /// <param name="source"><c>float</c> or <c>double</c> elements.</param>
    public static Mat<T> Exp<T>(Mat<T> source)
        where T : unmanaged
    {
        ValidateFloating(source, nameof(source));
        var dst = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
        Unary(source, dst, v => MathF.Exp(v), Math.Exp);
        return dst;
    }

    /// <summary>
    /// Natural logarithm of each element, as OpenCV <c>log</c> (the C runtime <c>log</c>, as the MSVC
    /// build). Zero gives −∞; negative elements give NaN.
    /// </summary>
    /// <param name="source"><c>float</c> or <c>double</c> elements.</param>
    public static Mat<T> Log<T>(Mat<T> source)
        where T : unmanaged
    {
        ValidateFloating(source, nameof(source));
        var dst = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
        Unary(source, dst, v => MathF.Log(v), Math.Log);
        return dst;
    }

    /// <summary>Square root of each element, as OpenCV <c>sqrt</c> (<c>pow(src, 0.5)</c>).</summary>
    /// <param name="source"><c>float</c> or <c>double</c> elements.</param>
    public static Mat<T> Sqrt<T>(Mat<T> source)
        where T : unmanaged
    {
        return Pow(source, 0.5);
    }

    /// <summary>
    /// Raises each element to a power, as OpenCV <c>pow</c>.
    /// </summary>
    /// <param name="source">
    /// <c>byte</c>, <c>sbyte</c>, <c>ushort</c>, <c>short</c>, <c>int</c>, <c>float</c> or
    /// <c>double</c> elements. Integer elements support integer powers only.
    /// </param>
    /// <param name="power">The exponent.</param>
    /// <remarks>
    /// <para>An integer power (within <c>DBL_EPSILON</c>) uses repeated squaring in the OpenCV working
    /// type, saturated to the element type: 0 gives ones, 1 a copy and 2 <c>multiply(src, src)</c>.
    /// A negative integer power of an integer element is 0 except for |x| ≤ 2 (OpenCV table).</para>
    /// <para>±0.5 uses <c>sqrt</c> or the reciprocal square root (vector blocks use the
    /// <c>rsqrtps</c> estimate with one Newton step, as OpenCV with SSE). Other powers compute
    /// <c>exp(log(x)·power)</c>; <c>x = 0</c> gives 0 (or +∞ for a negative power) and <c>x &lt; 0</c>
    /// gives the NaN with bits <c>0x7fffffff</c>.</para>
    /// </remarks>
    /// <exception cref="NotSupportedException">The element type is not supported, or a non-integer power of integer elements.</exception>
    public static Mat<T> Pow<T>(Mat<T> source, double power)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        bool floating = typeof(T) == typeof(float) || typeof(T) == typeof(double);
        if (!floating && typeof(T) != typeof(byte) && typeof(T) != typeof(sbyte) && typeof(T) != typeof(ushort)
            && typeof(T) != typeof(short) && typeof(T) != typeof(int))
        {
            throw new NotSupportedException($"pow does not support {typeof(T).Name} elements.");
        }

        int ipower = ScalarConverter.CvRound(power);
        bool isIntegerPower = Math.Abs(ipower - power) < 2.220446049250313e-16;
        if (isIntegerPower)
        {
            switch (ipower)
            {
                case 0:
                {
                    var ones = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
                    ones.SetTo(new Scalar(1, 1, 1, 1));
                    return ones;
                }

                case 1:
                    return source.Clone();
                case 2:
                    return Arithmetic.Multiply(source, source);
            }
        }
        else if (!floating)
        {
            throw new NotSupportedException("pow of integer elements needs an integer power.");
        }

        var dst = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
        ForEachPlane(source, source, dst, null, (s, _, d, _) =>
        {
            if (isIntegerPower)
            {
                IntegerPowerPlane<T>(s, d, ipower);
            }
            else if (Math.Abs(Math.Abs(power) - 0.5) < 2.220446049250313e-16)
            {
                SquareRootPlane<T>(s, d, power < 0);
            }
            else
            {
                RealPowerPlane<T>(s, d, power);
            }
        });
        return dst;
    }

    private static void IntegerPowerPlane<T>(ReadOnlySpan<byte> s, Span<byte> d, int power)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            var src = Cast<float>(s);
            var dst = CastMut<float>(d);
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = MathKernels.IntegerPower(src[i], power);
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var src = Cast<double>(s);
            var dst = CastMut<double>(d);
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = MathKernels.IntegerPower(src[i], power);
            }
        }
        else
        {
            var src = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(s);
            var dst = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(d);
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = MathKernels.IntegerPower(src[i], power);
            }
        }
    }

    private static void SquareRootPlane<T>(ReadOnlySpan<byte> s, Span<byte> d, bool inverse)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            var src = Cast<float>(s);
            var dst = CastMut<float>(d);
            if (inverse)
            {
                MathKernels.InvSqrt32f(src, dst);
                return;
            }

            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = MathF.Sqrt(src[i]);
            }
        }
        else
        {
            var src = Cast<double>(s);
            var dst = CastMut<double>(d);
            for (int i = 0; i < dst.Length; i++)
            {
                dst[i] = inverse ? 1 / Math.Sqrt(src[i]) : Math.Sqrt(src[i]);
            }
        }
    }

    private static void RealPowerPlane<T>(ReadOnlySpan<byte> s, Span<byte> d, double power)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            var src = Cast<float>(s);
            var dst = CastMut<float>(d);
            float infinity = BitConverter.Int32BitsToSingle(0x7f800000);
            float nan = BitConverter.Int32BitsToSingle(0x7fffffff);
            for (int i = 0; i < dst.Length; i++)
            {
                float x = src[i];
                float y = MathF.Exp((float)(MathF.Log(x) * power));
                if (x <= 0)
                {
                    y = x == 0 ? (power < 0 ? infinity : y) : nan;
                }

                dst[i] = y;
            }
        }
        else
        {
            var src = Cast<double>(s);
            var dst = CastMut<double>(d);
            double infinity = BitConverter.Int64BitsToDouble(0x7FF0000000000000);
            double nan = BitConverter.Int64BitsToDouble(0x7FFFFFFFFFFFFFFF);
            for (int i = 0; i < dst.Length; i++)
            {
                double x = src[i];
                double y = Math.Exp(Math.Log(x) * power);
                if (x <= 0)
                {
                    y = x == 0 ? (power < 0 ? infinity : y) : nan;
                }

                dst[i] = y;
            }
        }
    }

    /// <summary>OpenCV <c>hal::polarToCart64f</c>: 128-element blocks converted to <c>float</c>.</summary>
    private static void PolarToCart64f(ReadOnlySpan<double> mag, bool hasMagnitude, ReadOnlySpan<double> angle, Span<double> x, Span<double> y, bool degrees)
    {
        Span<float> mf = stackalloc float[DoubleBlock];
        Span<float> af = stackalloc float[DoubleBlock];
        Span<float> xf = stackalloc float[DoubleBlock];
        Span<float> yf = stackalloc float[DoubleBlock];
        for (int i = 0; i < angle.Length; i += DoubleBlock)
        {
            int n = Math.Min(DoubleBlock, angle.Length - i);
            for (int j = 0; j < n; j++)
            {
                if (hasMagnitude)
                {
                    mf[j] = (float)mag[i + j];
                }

                af[j] = (float)angle[i + j];
            }

            MathKernels.SinCos32f(mf[..n], hasMagnitude, af[..n], xf[..n], yf[..n], degrees);
            for (int j = 0; j < n; j++)
            {
                x[i + j] = xf[j];
                y[i + j] = yf[j];
            }
        }
    }

    /// <summary>OpenCV block size rounded up to whole pixels: <c>((BLOCK_SIZE + cn − 1)/cn)·cn</c>.</summary>
    private static int ChannelBlock(int channels) => (BlockSize + channels - 1) / channels * channels;

    private delegate void FloatKernel(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, Span<float> d);

    /// <summary>
    /// Runs a <c>float</c> kernel on <c>double</c> data in 128-element blocks, as the OpenCV 64-bit
    /// wrappers (<c>cartToPolar64f</c>, <c>fastAtan64f</c>).
    /// </summary>
    private static void ThroughFloat(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c, Span<double> d, FloatKernel kernel)
    {
        Span<float> af = stackalloc float[DoubleBlock];
        Span<float> bf = stackalloc float[DoubleBlock];
        Span<float> cf = stackalloc float[DoubleBlock];
        Span<float> df = stackalloc float[DoubleBlock];
        for (int i = 0; i < c.Length; i += DoubleBlock)
        {
            int n = Math.Min(DoubleBlock, c.Length - i);
            for (int j = 0; j < n; j++)
            {
                af[j] = (float)a[i + j];
                bf[j] = (float)b[i + j];
            }

            kernel(af[..n], bf[..n], cf[..n], df[..n]);
            for (int j = 0; j < n; j++)
            {
                c[i + j] = cf[j];
            }

            if (!d.IsEmpty)
            {
                for (int j = 0; j < n; j++)
                {
                    d[i + j] = df[j];
                }
            }
        }
    }

    private static void Unary<T>(Mat<T> source, Mat<T> dst, Func<float, float> f32, Func<double, double> f64)
        where T : unmanaged
    {
        ForEachPlane(source, source, dst, null, (s, _, d, _) =>
        {
            if (typeof(T) == typeof(float))
            {
                var src = Cast<float>(s);
                var dd = CastMut<float>(d);
                for (int i = 0; i < dd.Length; i++)
                {
                    dd[i] = f32(src[i]);
                }
            }
            else
            {
                var src = Cast<double>(s);
                var dd = CastMut<double>(d);
                for (int i = 0; i < dd.Length; i++)
                {
                    dd[i] = f64(src[i]);
                }
            }
        });
    }

    private delegate void PlaneAction(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> c, Span<byte> d);

    /// <summary>
    /// Calls <paramref name="action"/> once with all elements when every matrix is continuous, and
    /// once per row otherwise (OpenCV <c>NAryMatIterator</c> planes). Spans are raw bytes.
    /// </summary>
    private static void ForEachPlane<T>(Mat<T> a, Mat<T> b, Mat<T> c, Mat<T>? d, PlaneAction action)
        where T : unmanaged
    {
        bool continuous = a.IsContinuous && b.IsContinuous && c.IsContinuous && (d is null || d.IsContinuous);
        if (continuous)
        {
            action(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.GetReadOnlySpan()),
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(b.GetReadOnlySpan()),
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.GetSpan()),
                d is null ? [] : System.Runtime.InteropServices.MemoryMarshal.AsBytes(d.GetSpan()));
            return;
        }

        for (int y = 0; y < a.Height; y++)
        {
            action(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.GetReadOnlyRowSpan(y)),
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(b.GetReadOnlyRowSpan(y)),
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(c.GetRowSpan(y)),
                d is null ? [] : System.Runtime.InteropServices.MemoryMarshal.AsBytes(d.GetRowSpan(y)));
        }
    }

    private static ReadOnlySpan<TOut> Cast<TOut>(ReadOnlySpan<byte> bytes)
        where TOut : unmanaged => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, TOut>(bytes);

    private static Span<TOut> CastMut<TOut>(Span<byte> bytes)
        where TOut : unmanaged => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, TOut>(bytes);

    private static void ValidateFloating<T>(Mat<T> m, string name)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(m, name);
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException("The elements must be float or double.");
        }
    }

    private static void ValidatePair<T>(Mat<T> a, Mat<T> b)
        where T : unmanaged
    {
        ValidateFloating(a, nameof(a));
        ArgumentNullException.ThrowIfNull(b);
        if (a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels)
        {
            throw new ArgumentException("The matrices must have the same size and channel count.");
        }
    }
}
