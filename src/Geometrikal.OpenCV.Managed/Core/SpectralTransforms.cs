// This implementation is ported from the OpenCV discrete Fourier and cosine transforms.
//
// Original source:
//   modules/core/src/dxt.cpp (cv::dft, cv::idft, cv::dct, cv::idct, cv::mulSpectrums,
//   cv::getOptimalDFTSize)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/dxt.cpp
//
// OpenCV is licensed under the Apache License 2.0.
// See the OpenCV license and this repository's licensing documentation.

namespace Geometrikal.OpenCV.Managed.Core;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV <c>dft</c>, <c>idft</c>, <c>dct</c>, <c>idct</c>,
/// <c>mulSpectrums</c> and <c>getOptimalDFTSize</c>. The transforms are the mixed-radix
/// algorithm of <c>dxt.cpp</c> without IPP; results are bit-exact to OpenCV with IPP disabled.
/// Every operation returns a new matrix and does not modify its sources.
/// </summary>
/// <remarks>
/// <para>Element type: <see cref="float"/> or <see cref="double"/>.</para>
/// <para>A real spectrum uses the CCS packed layout of OpenCV: in each 1D row,
/// <c>Re0, Re1, Im1, Re2, Im2, ...</c> with a final <c>Re(n/2)</c> for even n. In 2D, columns 0
/// and n-1 (even width) hold the same packing along the column, and the other column pairs hold
/// the real and imaginary parts of complex column spectra.</para>
/// <para>Rows and columns are transformed in parallel (see <see cref="CvParallel"/>). The
/// result does not depend on the thread count.</para>
/// </remarks>
public static class SpectralTransforms
{
    private static readonly int[] OptimalDftSizes = BuildOptimalSizes();

    /// <summary>
    /// Forward or inverse discrete Fourier transform of a 1D or 2D array, as OpenCV
    /// <c>cv::dft(src, dst, flags, nonzeroRows)</c>.
    /// </summary>
    /// <param name="source">1-channel real or 2-channel complex (re, im) matrix.</param>
    /// <param name="flags">Combination of <see cref="DftFlags"/>.</param>
    /// <param name="nonzeroRows">
    /// When positive and not above the row count: forward transforms process only the first
    /// <paramref name="nonzeroRows"/> input rows (the other rows count as zero); inverse transforms
    /// compute only the first <paramref name="nonzeroRows"/> output rows and set the other rows to zero.
    /// </param>
    /// <returns>
    /// A new matrix of the source size. A forward transform of 1-channel input with
    /// <see cref="DftFlags.ComplexOutput"/> gives 2 channels; an inverse transform of 2-channel
    /// input with <see cref="DftFlags.RealOutput"/> gives 1 channel; otherwise the channel count
    /// of the source. 1-channel input of an inverse transform is read as CCS.
    /// </returns>
    /// <remarks>
    /// <para>The transform is 1D when <see cref="DftFlags.Rows"/> is set, when the source has one
    /// row, or when it has one column and is continuous; otherwise it is 2D. A non-continuous
    /// single column is transformed with the column algorithm, as in OpenCV, and its forward CCS
    /// result can differ in the last bits from the continuous case.</para>
    /// <para>With <see cref="DftFlags.Rows"/>, <see cref="DftFlags.ComplexOutput"/> and
    /// <paramref name="nonzeroRows"/>, the rows after <paramref name="nonzeroRows"/> are zero. OpenCV
    /// leaves the upper half of those rows uninitialized.</para>
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// <typeparamref name="T"/> is not float or double; or the source has one column and
    /// <paramref name="nonzeroRows"/> is positive.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// The source has neither 1 nor 2 channels, or <see cref="DftFlags.ComplexInput"/> is set for
    /// a 1-channel source.
    /// </exception>
    public static Mat<T> Dft<T>(Mat<T> source, DftFlags flags = DftFlags.None, int nonzeroRows = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        EnsureFloating<T>();
        int cn = source.Channels;
        if (cn != 1 && cn != 2)
        {
            throw new ArgumentException("The source must have 1 or 2 channels.", nameof(source));
        }

        if ((flags & DftFlags.ComplexInput) != 0 && cn != 2)
        {
            throw new ArgumentException("ComplexInput requires a 2-channel source.", nameof(flags));
        }

        if (source.Width == 1 && nonzeroRows > 0)
        {
            throw new NotSupportedException("nonzeroRows is not supported for a single-column matrix.");
        }

        bool inv = (flags & DftFlags.Inverse) != 0;
        int dstCn = cn;
        if (!inv && cn == 1 && (flags & DftFlags.ComplexOutput) != 0)
        {
            dstCn = 2;
        }
        else if (inv && cn == 2 && (flags & DftFlags.RealOutput) != 0)
        {
            dstCn = 1;
        }

        // Every element is written, except the upper halves of complex rows after nonzeroRows.
        bool clear = cn == 1 && dstCn == 2 && nonzeroRows > 0 && nonzeroRows < source.Height;
        var dst = new Mat<T>(source.Width, source.Height, dstCn, clear);
        bool rows = (flags & DftFlags.Rows) != 0;
        bool scale = (flags & DftFlags.Scale) != 0;
        if (typeof(T) == typeof(float))
        {
            DftEngine<float>.Dft((Mat<float>)(object)source, (Mat<float>)(object)dst, inv, rows, scale, nonzeroRows);
        }
        else
        {
            DftEngine<double>.Dft((Mat<double>)(object)source, (Mat<double>)(object)dst, inv, rows, scale, nonzeroRows);
        }

        return dst;
    }

    /// <summary>
    /// Inverse discrete Fourier transform, as OpenCV <c>cv::idft</c>:
    /// <see cref="Dft{T}"/> with <see cref="DftFlags.Inverse"/> added. The result is not scaled
    /// unless <see cref="DftFlags.Scale"/> is set.
    /// </summary>
    public static Mat<T> Idft<T>(Mat<T> source, DftFlags flags = DftFlags.None, int nonzeroRows = 0)
        where T : unmanaged
        => Dft(source, flags | DftFlags.Inverse, nonzeroRows);

    /// <summary>
    /// Forward or inverse discrete cosine transform of a 1D or 2D array, as OpenCV
    /// <c>cv::dct(src, dst, flags)</c>. The forward transform is the orthonormal DCT-II; the
    /// inverse is its inverse (DCT-III).
    /// </summary>
    /// <param name="source">1-channel matrix. Each transformed length must be even or 1.</param>
    /// <param name="flags">Combination of <see cref="DctFlags"/>.</param>
    /// <returns>A new matrix of the source size and type.</returns>
    /// <remarks>
    /// The transform is 1D when <see cref="DctFlags.Rows"/> is set, when the source has one
    /// row, or when it has one column and is continuous; otherwise rows are transformed, then
    /// columns.
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// <typeparamref name="T"/> is not float or double, or a transformed length is odd and
    /// greater than 1.
    /// </exception>
    /// <exception cref="ArgumentException">The source does not have 1 channel.</exception>
    public static Mat<T> Dct<T>(Mat<T> source, DctFlags flags = DctFlags.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        EnsureFloating<T>();
        if (source.Channels != 1)
        {
            throw new ArgumentException("The source must have 1 channel.", nameof(source));
        }

        var dst = new Mat<T>(source.Width, source.Height, 1, clear: false);
        bool inv = (flags & DctFlags.Inverse) != 0;
        bool rows = (flags & DctFlags.Rows) != 0;
        if (typeof(T) == typeof(float))
        {
            DftEngine<float>.Dct((Mat<float>)(object)source, (Mat<float>)(object)dst, inv, rows);
        }
        else
        {
            DftEngine<double>.Dct((Mat<double>)(object)source, (Mat<double>)(object)dst, inv, rows);
        }

        return dst;
    }

    /// <summary>
    /// Inverse discrete cosine transform, as OpenCV <c>cv::idct</c>: <see cref="Dct{T}"/> with
    /// <see cref="DctFlags.Inverse"/> added.
    /// </summary>
    public static Mat<T> Idct<T>(Mat<T> source, DctFlags flags = DctFlags.None)
        where T : unmanaged
        => Dct(source, flags | DctFlags.Inverse);

    /// <summary>
    /// Per-element multiplication of two Fourier spectra, as OpenCV
    /// <c>cv::mulSpectrums(a, b, dst, flags, conjB)</c>.
    /// </summary>
    /// <param name="a">First spectrum: 1-channel CCS or 2-channel complex.</param>
    /// <param name="b">Second spectrum of the same size and channel count.</param>
    /// <param name="flags">Only <see cref="DftFlags.Rows"/> is used: each row is a separate 1D spectrum.</param>
    /// <param name="conjB">Multiplies by the conjugate of <paramref name="b"/>.</param>
    /// <returns>A new matrix of the size and type of <paramref name="a"/>.</returns>
    /// <remarks>
    /// Complex products are computed in double and rounded to <typeparamref name="T"/>; the real
    /// CCS elements are multiplied in <typeparamref name="T"/>. The spectra are 1D when
    /// <see cref="DftFlags.Rows"/> is set, when they have one row, or when they have one column
    /// and both are continuous.
    /// </remarks>
    public static Mat<T> MulSpectrums<T>(Mat<T> a, Mat<T> b, DftFlags flags = DftFlags.None, bool conjB = false)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        EnsureFloating<T>();
        if (a.Channels != 1 && a.Channels != 2)
        {
            throw new ArgumentException("The spectra must have 1 or 2 channels.", nameof(a));
        }

        if (a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels)
        {
            throw new ArgumentException("The spectra must have the same size and channel count.", nameof(b));
        }

        var dst = new Mat<T>(a.Width, a.Height, a.Channels, clear: false);
        bool rows = (flags & DftFlags.Rows) != 0;
        if (typeof(T) == typeof(float))
        {
            SpectrumKernels.Mul((Mat<float>)(object)a, (Mat<float>)(object)b, (Mat<float>)(object)dst, rows, conjB);
        }
        else
        {
            SpectrumKernels.Mul((Mat<double>)(object)a, (Mat<double>)(object)b, (Mat<double>)(object)dst, rows, conjB);
        }

        return dst;
    }

    /// <summary>
    /// Per-element division of two Fourier spectra, as OpenCV
    /// <c>cv::divSpectrums(a, b, dst, flags, conjB)</c>.
    /// </summary>
    /// <param name="a">Numerator spectrum: 1-channel CCS or 2-channel complex.</param>
    /// <param name="b">Denominator spectrum of the same size and channel count.</param>
    /// <param name="flags">Only <see cref="DftFlags.Rows"/> is used: each row is a separate 1D spectrum.</param>
    /// <param name="conjB">Divides by the conjugate of <paramref name="b"/>.</param>
    /// <returns>A new matrix of the size and type of <paramref name="a"/>.</returns>
    /// <remarks>
    /// The complex quotient is <c>a·conj(b) / (|b|² + ε)</c> (or <c>a·b / (|b|² + ε)</c> with
    /// <paramref name="conjB"/>), computed in double with ε = <c>FLT_EPSILON</c> for float and
    /// <c>DBL_EPSILON</c> for double. Real CCS elements are <c>a / (b + ε)</c> in
    /// <typeparamref name="T"/>. The 1D rules are those of <see cref="MulSpectrums{T}"/>.
    /// </remarks>
    public static Mat<T> DivSpectrums<T>(Mat<T> a, Mat<T> b, DftFlags flags = DftFlags.None, bool conjB = false)
        where T : unmanaged =>
        DivSpectrums(a, b, flags, conjB, legacy455: false);

    // legacy455 reproduces cv2 4.5.5: float row pairs computed in float before the double quotient.
    internal static Mat<T> DivSpectrums<T>(Mat<T> a, Mat<T> b, DftFlags flags, bool conjB, bool legacy455)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        EnsureFloating<T>();
        if (a.Channels != 1 && a.Channels != 2)
        {
            throw new ArgumentException("The spectra must have 1 or 2 channels.", nameof(a));
        }

        if (a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels)
        {
            throw new ArgumentException("The spectra must have the same size and channel count.", nameof(b));
        }

        var dst = new Mat<T>(a.Width, a.Height, a.Channels, clear: false);
        if (typeof(T) == typeof(float))
        {
            SpectrumKernels.Div((Mat<float>)(object)a, (Mat<float>)(object)b, (Mat<float>)(object)dst, (flags & DftFlags.Rows) != 0, conjB, legacy455);
        }
        else
        {
            SpectrumKernels.Div((Mat<double>)(object)a, (Mat<double>)(object)b, (Mat<double>)(object)dst, (flags & DftFlags.Rows) != 0, conjB);
        }

        return dst;
    }

    /// <summary>
    /// Smallest size not below <paramref name="size"/> of the form 2^a·3^b·5^c (b ≤ 14), as OpenCV
    /// <c>cv::getOptimalDFTSize</c>. Returns 1 for 0, and -1 for negative sizes and sizes of
    /// 2125764000 or more.
    /// </summary>
    public static int GetOptimalDftSize(int size)
    {
        int[] tab = OptimalDftSizes;
        int a = 0, b = tab.Length - 1;
        if ((uint)size >= (uint)tab[b])
        {
            return -1;
        }

        while (a < b)
        {
            int c = (a + b) >> 1;
            if (size <= tab[c])
            {
                b = c;
            }
            else
            {
                a = c + 1;
            }
        }

        return tab[b];
    }

    private static void EnsureFloating<T>()
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException("Spectral transforms support float and double elements only.");
        }
    }

    /// <summary>The <c>optimalDFTSizeTab</c> values: sorted 2^a·3^b·5^c ≤ 2125764000 with b ≤ 14.</summary>
    private static int[] BuildOptimalSizes()
    {
        const long Limit = 2125764000;
        var sizes = new List<int>(1651);
        for (long p3 = 1, b = 0; b <= 14 && p3 <= Limit; p3 *= 3, b++)
        {
            for (long p5 = p3; p5 <= Limit; p5 *= 5)
            {
                for (long p2 = p5; p2 <= Limit; p2 *= 2)
                {
                    sizes.Add((int)p2);
                }
            }
        }

        sizes.Sort();
        return [.. sizes];
    }
}
