// This implementation is ported from the OpenCV image filtering functions.
//
// Original sources:
//   modules/imgproc/src/filter.dispatch.cpp (filter2D, sepFilter2D)
//   modules/imgproc/src/box_filter.dispatch.cpp (blur, boxFilter)
//   modules/imgproc/src/smooth.dispatch.cpp (GaussianBlur, getGaussianKernel)
//   modules/imgproc/src/median_blur.dispatch.cpp (medianBlur)
//   modules/imgproc/src/bilateral_filter.dispatch.cpp (bilateralFilter)
//   modules/imgproc/src/stackblur.cpp (stackBlur)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's image smoothing and linear filtering functions:
/// <c>blur</c>, <c>boxFilter</c>, <c>sqrBoxFilter</c>, <c>GaussianBlur</c>,
/// <c>medianBlur</c>, <c>bilateralFilter</c>, <c>stackBlur</c>, <c>filter2D</c>,
/// <c>sepFilter2D</c> and <c>getGaussianKernel</c>.
///
/// <para><b>Non-mutating.</b> Every operation returns a new matrix; the
/// source is never modified.</para>
///
/// <para><b>Element types.</b> Matrices have 1..4 channels. Box, Gaussian and
/// median filters take <c>byte</c> or <c>float</c> sources; box filters support
/// <c>byte→byte/short/float</c> and <c>float→float</c>. <c>sepFilter2D</c>
/// takes <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>
/// sources and destinations except <c>double→float</c>; <c>filter2D</c> requires a
/// destination depth not below the source depth (<c>byte→any</c>,
/// <c>ushort→ushort/float/double</c>, <c>short→short/float/double</c>,
/// <c>float→float</c>, <c>double→double</c>). OpenCV's <c>ddepth</c> becomes the
/// <c>TDst</c> type parameter. The 8-bit Gaussian uses OpenCV's bit-exact
/// fixed-point path; 8-bit separable filters use OpenCV's bit-exact integer
/// path when the kernels qualify.</para>
///
/// <para><b>Borders.</b> The default is <see cref="BorderType.Reflect101"/>.
/// <see cref="BorderType.Constant"/> pads with zeros (OpenCV exposes no
/// border value for these functions). <see cref="BorderType.Wrap"/> is
/// rejected, exactly as OpenCV's filter engine rejects it; the DFT path of
/// <c>filter2D</c> and <c>bilateralFilter</c> accept it, as in OpenCV. Region-of-interest
/// views are always treated as isolated images (OpenCV samples the parent
/// image outside a submatrix ROI unless <c>BORDER_ISOLATED</c> is set; this
/// port always behaves as if isolated).</para>
///
/// <para><b>Kernels.</b> Filter kernels are float32. For a double source or
/// destination they are converted to double (exactly), as OpenCV converts them
/// to its double buffer type.</para>
/// </summary>
public static class Filters
{
    /// <summary>
    /// Computes the Gaussian kernel exactly as OpenCV's
    /// <c>getGaussianKernel(n, sigma, CV_64F)</c>, including its softdouble
    /// bit-exact arithmetic. For <paramref name="sigma"/> &lt;= 0 the sigma is
    /// derived from <paramref name="n"/> (with hardcoded kernels for n ≤ 9).
    /// </summary>
    /// <param name="n">Kernel size; positive.</param>
    /// <param name="sigma">Gaussian standard deviation, or non-positive to derive it from <paramref name="n"/>.</param>
    /// <returns>The kernel coefficients (they sum to 1).</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="n"/> is not positive.</exception>
    public static double[] GetGaussianKernel(int n, double sigma)
        => GaussianKernels.GetKernelBitExact(n, sigma);

    /// <summary>
    /// Gaussian blur of an 8-bit image via OpenCV's bit-exact fixed-point
    /// path (<c>GaussianBlurFixedPoint</c>).
    /// </summary>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="ksize">
    /// Kernel size; both components odd and positive, or zero/negative to
    /// derive them from the sigmas.
    /// </param>
    /// <param name="sigmaX">Gaussian standard deviation in X. Non-positive derives it from the kernel size.</param>
    /// <param name="sigmaY">Gaussian standard deviation in Y; zero means "same as sigmaX".</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new blurred matrix.</returns>
    public static Mat<byte> GaussianBlur(
        Mat<byte> source,
        Size ksize,
        double sigmaX,
        double sigmaY = 0,
        BorderType borderType = BorderType.Reflect101)
    {
        ValidateSource(source);
        ksize = GaussianPrologue(source.Width, source.Height, ksize, borderType);
        if (ksize is { Width: 1, Height: 1 })
        {
            return source.Clone();
        }

        if (sigmaY <= 0)
        {
            sigmaY = sigmaX;
        }

        ksize = GaussianKernels.ComputeKernelSize(ksize, sigmaX, sigmaY, byteDepth: true);

        ushort[] fkx = GaussianKernels.GetKernelFixedPointQ8(ksize.Width, sigmaX);
        ushort[] fky = GaussianKernels.GetKernelFixedPointQ8(ksize.Height, sigmaY);
        return GaussianFixedPoint.Apply(source, fkx, fky, CheckGaussianBorder(borderType));
    }

    /// <summary>
    /// Gaussian blur of a float32 image (OpenCV's <c>sepFilter2D</c> float
    /// path with bit-exact kernel coefficients).
    /// </summary>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="ksize">
    /// Kernel size; both components odd and positive, or zero/negative to
    /// derive them from the sigmas.
    /// </param>
    /// <param name="sigmaX">Gaussian standard deviation in X. Non-positive derives it from the kernel size.</param>
    /// <param name="sigmaY">Gaussian standard deviation in Y; zero means "same as sigmaX".</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new blurred matrix.</returns>
    public static Mat<float> GaussianBlur(
        Mat<float> source,
        Size ksize,
        double sigmaX,
        double sigmaY = 0,
        BorderType borderType = BorderType.Reflect101)
    {
        ValidateSource(source);
        return GaussianBlurCore(source, ksize, sigmaX, sigmaY, borderType, null);
    }

    /// <summary>
    /// <see cref="GaussianBlur(Mat{float}, Size, double, double, BorderType)"/> into
    /// <paramref name="destination"/>, which has the source size and channel count and does not
    /// share its array with the source.
    /// </summary>
    internal static void GaussianBlurInto(
        Mat<float> source,
        Mat<float> destination,
        Size ksize,
        double sigmaX,
        double sigmaY = 0,
        BorderType borderType = BorderType.Reflect101)
    {
        ValidateSource(source);
        GaussianBlurCore(source, ksize, sigmaX, sigmaY, borderType, destination);
    }

    private static Mat<float> GaussianBlurCore(
        Mat<float> source, Size ksize, double sigmaX, double sigmaY, BorderType borderType, Mat<float>? destination)
    {
        ksize = GaussianPrologue(source.Width, source.Height, ksize, borderType);
        if (ksize is { Width: 1, Height: 1 })
        {
            if (destination is null)
            {
                return source.Clone();
            }

            source.CopyTo(destination);
            return destination;
        }

        if (sigmaY <= 0)
        {
            sigmaY = sigmaX;
        }

        ksize = GaussianKernels.ComputeKernelSize(ksize, sigmaX, sigmaY, byteDepth: false);

        float[] kx = GaussianKernels.GetKernelSingle(ksize.Width, sigmaX);
        float[] ky = GaussianKernels.GetKernelSingle(ksize.Height, sigmaY);
        return SepFilterEngine.Apply<float, float>(
            source, kx, ky, new Point(-1, -1), 0, CheckGaussianBorder(borderType), destination);
    }

    /// <summary>
    /// Normalized box blur, mirroring OpenCV's <c>blur</c> (i.e.
    /// <c>boxFilter</c> with <c>normalize=true</c> and matching types).
    /// </summary>
    /// <typeparam name="T">Element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="ksize">Kernel size; positive.</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new blurred matrix.</returns>
    public static Mat<T> Blur<T>(
        Mat<T> source,
        Size ksize,
        Point? anchor = null,
        BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
        => BoxFilter<T, T>(source, ksize, anchor, normalize: true, borderType);

    /// <summary>
    /// Box filter, mirroring OpenCV's <c>boxFilter</c>. Accumulation types
    /// follow OpenCV exactly (16-bit sums for 8-bit blurs with kernel area
    /// ≤ 256, 32-bit integer sums for other 8-bit inputs, double sums for
    /// float inputs).
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type; <c>byte→byte/short/float</c> and
    /// <c>float→float</c> are supported.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="ksize">Kernel size; positive.</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="normalize">Whether the sum is scaled by <c>1/(ksize.Width*ksize.Height)</c>.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new filtered matrix.</returns>
    public static Mat<TDst> BoxFilter<TSrc, TDst>(
        Mat<TSrc> source,
        Size ksize,
        Point? anchor = null,
        bool normalize = true,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ValidateSource(source);
        return BoxFilterEngine.Apply<TSrc, TDst>(
            source, ksize, anchor ?? new Point(-1, -1), normalize, borderType);
    }

    /// <summary>
    /// Median filter, mirroring OpenCV's <c>medianBlur</c>: sorting networks
    /// for kernel sizes 3 and 5, histogram filtering for larger odd sizes
    /// (8-bit, 1/3/4 channels only). Borders replicate edge pixels.
    /// </summary>
    /// <typeparam name="T">Element type: <c>byte</c> or <c>float</c> (float supports ksize 3 and 5 only).</typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="ksize">Aperture size; odd and positive. 1 returns a copy.</param>
    /// <returns>A new filtered matrix.</returns>
    public static Mat<T> MedianBlur<T>(Mat<T> source, int ksize)
        where T : unmanaged
    {
        ValidateSource(source);
        if (ksize % 2 != 1 || ksize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(ksize), ksize, "The aperture size must be odd and positive.");
        }

        if (ksize == 1)
        {
            return source.Clone();
        }

        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException(
                $"Unsupported median element type '{typeof(T)}'. Supported: byte, float.");
        }

        if (ksize <= 5)
        {
            return MedianEngine.SortNet(source, ksize);
        }

        if (typeof(T) != typeof(byte))
        {
            throw new NotSupportedException("Median apertures larger than 5 require an 8-bit image, as in OpenCV.");
        }

        if (source.Channels == 2)
        {
            throw new NotSupportedException(
                "Median apertures larger than 5 support 1, 3 or 4 channels, as in OpenCV.");
        }

        return (Mat<T>)(object)MedianEngine.Histogram((Mat<byte>)(object)source, ksize);
    }

    /// <summary>
    /// Convolution with an arbitrary 2-D kernel, mirroring OpenCV's
    /// <c>filter2D</c> (which correlates rather than convolves — the kernel
    /// is not mirrored). Kernels with an area of at least 130
    /// (<c>byte→byte</c>, <c>byte→short</c>, <c>float→float</c>) or 50
    /// (every other pair) use OpenCV's DFT path (<c>dftFilter2D</c>/<c>crossCorr</c>:
    /// float32 spectra for byte sources except <c>byte→double</c>, float64 spectra
    /// otherwise, blocks with the border extrapolated from each clipped source window).
    /// Smaller kernels use the direct path with float accumulation.
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type with a depth not below the source depth:
    /// <c>byte→byte/ushort/short/float/double</c>, <c>ushort→ushort/float/double</c>,
    /// <c>short→short/float/double</c>, <c>float→float</c>, <c>double→double</c>.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="kernel">Single-channel float32 kernel.</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="delta">Value added to each filtered result before saturation.</param>
    /// <param name="borderType">
    /// Border extrapolation (constant borders use value 0). <see cref="BorderType.Wrap"/> is
    /// accepted only when the DFT path is selected, as in OpenCV.
    /// </param>
    /// <returns>A new filtered matrix.</returns>
    /// <exception cref="NotSupportedException">
    /// The type combination is unsupported, or <see cref="BorderType.Wrap"/> is used with a
    /// kernel below the DFT threshold.
    /// </exception>
    public static Mat<TDst> Filter2D<TSrc, TDst>(
        Mat<TSrc> source,
        Mat<float> kernel,
        Point? anchor = null,
        double delta = 0,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ValidateSource(source);
        return Filter2DEngine.Apply<TSrc, TDst>(
            source, kernel, anchor ?? new Point(-1, -1), delta, borderType);
    }

    /// <summary>
    /// <see cref="Filter2D{TSrc, TDst}"/> into <paramref name="destination"/>, which has the source
    /// size and channel count and does not share its array with the source.
    /// </summary>
    internal static void Filter2DInto<TSrc, TDst>(
        Mat<TSrc> source,
        Mat<TDst> destination,
        Mat<float> kernel,
        Point? anchor = null,
        double delta = 0,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ValidateSource(source);
        Filter2DEngine.Apply(source, kernel, anchor ?? new Point(-1, -1), delta, borderType, destination);
    }

    /// <summary>
    /// Separable linear filter, mirroring OpenCV's <c>sepFilter2D</c>: the
    /// image is filtered with <paramref name="kernelX"/> along rows, then
    /// <paramref name="kernelY"/> along columns. 8-bit sources use OpenCV's
    /// bit-exact int32 path when the kernels qualify (smooth symmetric
    /// kernels for byte outputs; integer symmetric/antisymmetric kernels for
    /// short outputs), float32 accumulation otherwise, and double accumulation
    /// when the source or the destination is double.
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or
    /// <c>double</c>; <c>double→float</c> is not supported, as in OpenCV.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="kernelX">Row kernel (applied horizontally).</param>
    /// <param name="kernelY">Column kernel (applied vertically).</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="delta">Value added to each filtered result before saturation.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new filtered matrix.</returns>
    public static Mat<TDst> SepFilter2D<TSrc, TDst>(
        Mat<TSrc> source,
        float[] kernelX,
        float[] kernelY,
        Point? anchor = null,
        double delta = 0,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ValidateSource(source);
        return SepFilterEngine.Apply<TSrc, TDst>(
            source, kernelX, kernelY, anchor ?? new Point(-1, -1), delta, borderType);
    }

    /// <summary>
    /// Squared box filter, mirroring OpenCV's <c>sqrBoxFilter</c>: the sum (or mean, when
    /// <paramref name="normalize"/> is set) of the squared source values over the kernel.
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type: <c>float</c>, <c>double</c> or <c>int</c> for byte sources;
    /// <c>float</c> or <c>double</c> for float sources.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels. It is not modified.</param>
    /// <param name="ksize">Kernel size; positive.</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="normalize">Whether the sum is scaled by <c>1/(ksize.Width*ksize.Height)</c>.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0); <see cref="BorderType.Wrap"/> is rejected.</param>
    /// <returns>A new filtered matrix.</returns>
    /// <remarks>
    /// Byte sources sum squares in int32; float sources sum in double with OpenCV's running
    /// sums. For byte sources with float or int outputs, OpenCV scales the first
    /// <c>4·⌊W·cn/4⌋</c> elements of each row with a float32 factor and the others with a
    /// double factor; the port does the same. A normalized filter of a single-row or
    /// single-column image uses a kernel height or width of 1.
    /// </remarks>
    /// <exception cref="NotSupportedException">The type combination or border type is unsupported.</exception>
    public static Mat<TDst> SqrBoxFilter<TSrc, TDst>(
        Mat<TSrc> source,
        Size ksize,
        Point? anchor = null,
        bool normalize = true,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ValidateSource(source);
        return BoxFilterEngine.ApplySqr<TSrc, TDst>(source, ksize, anchor ?? new Point(-1, -1), normalize, borderType);
    }

    /// <summary>
    /// <see cref="SqrBoxFilter{TSrc, TDst}"/> with OpenCV's default output depth for byte
    /// sources (<c>ddepth = -1</c> gives <c>CV_32F</c>).
    /// </summary>
    /// <param name="source">Source matrix, 1..4 channels. It is not modified.</param>
    /// <param name="ksize">Kernel size; positive.</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="normalize">Whether the sum is scaled by <c>1/(ksize.Width*ksize.Height)</c>.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new float matrix.</returns>
    public static Mat<float> SqrBoxFilter(
        Mat<byte> source,
        Size ksize,
        Point? anchor = null,
        bool normalize = true,
        BorderType borderType = BorderType.Reflect101)
        => SqrBoxFilter<byte, float>(source, ksize, anchor, normalize, borderType);

    /// <summary>
    /// <see cref="SqrBoxFilter{TSrc, TDst}"/> with OpenCV's default output depth for float
    /// sources (<c>ddepth = -1</c> gives <c>CV_64F</c>).
    /// </summary>
    /// <param name="source">Source matrix, 1..4 channels. It is not modified.</param>
    /// <param name="ksize">Kernel size; positive.</param>
    /// <param name="anchor">Anchor inside the kernel, or null for the centre.</param>
    /// <param name="normalize">Whether the sum is scaled by <c>1/(ksize.Width*ksize.Height)</c>.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new double matrix.</returns>
    public static Mat<double> SqrBoxFilter(
        Mat<float> source,
        Size ksize,
        Point? anchor = null,
        bool normalize = true,
        BorderType borderType = BorderType.Reflect101)
        => SqrBoxFilter<float, double>(source, ksize, anchor, normalize, borderType);

    /// <summary>
    /// Bilateral filter, mirroring OpenCV 5.0's <c>bilateralFilter</c> with the arithmetic of
    /// its SSE2 build.
    /// </summary>
    /// <typeparam name="T">Element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <param name="source">Source matrix with 1 or 3 channels. It is not modified.</param>
    /// <param name="d">
    /// Neighbourhood diameter. A non-positive value derives the radius from
    /// <paramref name="sigmaSpace"/> (<c>cvRound(1.5·sigmaSpace)</c>); the radius is at least 1.
    /// </param>
    /// <param name="sigmaColor">Colour standard deviation.</param>
    /// <param name="sigmaSpace">Spatial standard deviation.</param>
    /// <param name="borderType">
    /// Border extrapolation; every type except <see cref="BorderType.Transparent"/>, including
    /// <see cref="BorderType.Wrap"/>. Constant borders use value 0.
    /// </param>
    /// <returns>A new filtered matrix.</returns>
    /// <remarks>
    /// <para>A sigma of at most 1e-6 returns a copy (OpenCV 5.0; OpenCV 4.x replaced a
    /// non-positive sigma with 1). A float image whose minimum and maximum differ by less than
    /// <c>FLT_EPSILON</c> also returns a copy.</para>
    /// <para>Byte images use OpenCV's colour-weight table (a vectorized exponential plus
    /// <c>expf</c> for the last four entries); float images use its interpolated exponential
    /// table. Float NaN neighbours are skipped and a NaN centre is excluded, as in OpenCV.
    /// Infinite float values are not supported: OpenCV reads outside its table for them.</para>
    /// <para>OpenCV builds with AVX2 or AVX-512 use fused multiply-add and other block sizes,
    /// so their float results depend on the processor. The port always reproduces the SSE2
    /// result (vector formula on the first <c>16·⌊W/16⌋</c> byte columns or <c>8·⌊W/8⌋</c>
    /// float columns, scalar formula on the others).</para>
    /// </remarks>
    /// <exception cref="NotSupportedException">The element type or channel count is unsupported.</exception>
    public static Mat<T> BilateralFilter<T>(
        Mat<T> source,
        int d,
        double sigmaColor,
        double sigmaSpace,
        BorderType borderType = BorderType.Reflect101)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels != 1 && source.Channels != 3)
        {
            throw new NotSupportedException("bilateralFilter supports 1 or 3 channels, as in OpenCV.");
        }

        if (borderType == BorderType.Transparent)
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "BorderType.Transparent is not supported by filters.");
        }

        if (typeof(T) == typeof(byte))
        {
            return (Mat<T>)(object)BilateralEngine.Apply((Mat<byte>)(object)source, d, sigmaColor, sigmaSpace, borderType);
        }

        if (typeof(T) == typeof(float))
        {
            return (Mat<T>)(object)BilateralEngine.Apply((Mat<float>)(object)source, d, sigmaColor, sigmaSpace, borderType);
        }

        throw new NotSupportedException($"Unsupported bilateralFilter element type '{typeof(T)}'. Supported: byte, float.");
    }

    /// <summary>
    /// Stack blur, mirroring OpenCV's <c>stackBlur</c> (OpenCV 4.7 and later): an approximation
    /// of a Gaussian blur with triangular weights <c>1, 2, …, r+1, …, 2, 1</c>, normalized by
    /// <c>(r+1)²</c> per pass.
    /// </summary>
    /// <typeparam name="T">Element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <param name="source">Source matrix with any channel count. It is not modified.</param>
    /// <param name="ksize">
    /// Kernel size; both components odd and positive. A component larger than the image is
    /// reduced to the image size (to the size minus 1 when the size is even), as in OpenCV.
    /// </param>
    /// <returns>A new blurred matrix.</returns>
    /// <remarks>
    /// <para>Borders replicate the edge pixels. Kernel widths up to 9 on an image wider than
    /// the kernel use direct weighted row sums; other row passes and all column passes use
    /// running sums.</para>
    /// <para>Byte results reproduce OpenCV's integer division approximation
    /// <c>(s·mul) &gt;&gt; shr</c> in its vector part and truncation of <c>s·(1/(r+1)²)</c> in its
    /// scalar part, including the 32-bit overflow of the vector multiply for large radii.</para>
    /// <para>OpenCV's column pass splits the columns into one stripe per thread, and its result
    /// depends on the thread count. The port always reproduces OpenCV with one thread, for any
    /// <see cref="CvParallel.MaxDegreeOfParallelism"/>.</para>
    /// <para>16-bit element types are not ported: OpenCV's vector code overflows for them.</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">A kernel component is not odd and positive.</exception>
    /// <exception cref="NotSupportedException">The element type is unsupported.</exception>
    public static Mat<T> StackBlur<T>(Mat<T> source, Size ksize)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (ksize.Width <= 0 || ksize.Width % 2 != 1 || ksize.Height <= 0 || ksize.Height % 2 != 1)
        {
            throw new ArgumentOutOfRangeException(nameof(ksize), ksize, "Both kernel size components must be odd and positive.");
        }

        if (typeof(T) != typeof(byte) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"Unsupported stackBlur element type '{typeof(T)}'. Supported: byte, float.");
        }

        return StackBlurEngine.Apply(source, ksize);
    }

    internal static void ValidateSource<T>(Mat<T> source)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels is < 1 or > 4)
        {
            throw new ArgumentException("Filtering supports 1 to 4 channels.", nameof(source));
        }
    }

    /// <summary>
    /// GaussianBlur's pre-kernel steps: the single-pixel-dimension collapse
    /// (smooth.dispatch.cpp:540; applies to non-constant borders — this port
    /// is always "isolated").
    /// </summary>
    private static Size GaussianPrologue(int width, int height, Size ksize, BorderType borderType)
    {
        if (borderType != BorderType.Constant)
        {
            if (height == 1)
            {
                ksize = new Size(ksize.Width, 1);
            }

            if (width == 1)
            {
                ksize = new Size(1, ksize.Height);
            }
        }

        return ksize;
    }

    private static BorderType CheckGaussianBorder(BorderType borderType)
        => borderType == BorderType.Wrap
            ? throw new NotSupportedException(
                "BORDER_WRAP is not supported by OpenCV's filter engine and is likewise rejected here.")
            : borderType;
}
