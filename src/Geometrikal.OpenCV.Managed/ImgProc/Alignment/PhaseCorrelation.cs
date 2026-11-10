// This API is based on OpenCV phase correlation.
//
// Original sources:
//   modules/imgproc/include/opencv2/imgproc.hpp (declarations)
//   modules/imgproc/src/phasecorr.cpp (phaseCorrelate, createHanningWindow)
//   modules/imgproc/src/phasecorr_iterative.cpp (phaseCorrelateIterative)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Sub-pixel translation estimation with the Fourier shift theorem, as OpenCV
/// <c>phaseCorrelate</c>, <c>phaseCorrelateIterative</c> and <c>createHanningWindow</c>. The
/// functions read the inputs and do not change them.
/// </summary>
public static class PhaseCorrelation
{
    /// <summary>
    /// Detects the translation between two images of the same size, as OpenCV
    /// <c>phaseCorrelate(src1, src2, window, &amp;response)</c>.
    /// </summary>
    /// <param name="src1">First image: 1-channel <c>float</c> or <c>double</c>.</param>
    /// <param name="src2">Second image of the type and size of <paramref name="src1"/>.</param>
    /// <param name="window">
    /// Optional window of the same type and size (for example from
    /// <see cref="CreateHanningWindow{T}(Size)"/>). Both images are multiplied by it.
    /// </param>
    /// <param name="response">
    /// Sum of the inverse cross-power spectrum in the 5×5 window around the peak, divided by the
    /// padded area. A value near 1 indicates a reliable shift.
    /// </param>
    /// <returns>The shift of <paramref name="src2"/> relative to <paramref name="src1"/>.</returns>
    /// <remarks>
    /// The images are zero-padded to <see cref="SpectralTransforms.GetOptimalDftSize(int)"/>.
    /// When no padding is necessary and a window is given, OpenCV multiplies the window into the
    /// caller's source arrays; the port multiplies into copies.
    /// </remarks>
    /// <exception cref="NotSupportedException">The images have one column (OpenCV fails for this shape).</exception>
    public static Point2d PhaseCorrelate<T>(Mat<T> src1, Mat<T> src2, Mat<T>? window, out double response)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(src1);
        ArgumentNullException.ThrowIfNull(src2);
        EnsureFloating<T>();
        if (src1.Channels != 1 || src2.Channels != 1 || src1.Width != src2.Width || src1.Height != src2.Height)
        {
            throw new ArgumentException("The images must be 1-channel and of the same size.", nameof(src2));
        }

        if (window is not null && (window.Channels != 1 || window.Width != src1.Width || window.Height != src1.Height))
        {
            throw new ArgumentException("The window must be 1-channel and of the image size.", nameof(window));
        }

        int m = SpectralTransforms.GetOptimalDftSize(src1.Height);
        int n = SpectralTransforms.GetOptimalDftSize(src1.Width);
        using PooledMat<T> padded1Pool = PooledMat<T>.Rent(n, m, 1);
        using PooledMat<T> padded2Pool = PooledMat<T>.Rent(n, m, 1);
        using PooledMat<T> fft1Pool = PooledMat<T>.Rent(n, m, 1);
        using PooledMat<T> fft2Pool = PooledMat<T>.Rent(n, m, 1);
        Mat<T> padded1 = padded1Pool.Mat, padded2 = padded2Pool.Mat, fft1 = fft1Pool.Mat, fft2 = fft2Pool.Mat;
        PadInto(src1, padded1);
        PadInto(src2, padded2);
        if (window is not null)
        {
            Mat<T> paddedWin = fft1;
            PadInto(window, paddedWin);
            Arithmetic.Multiply(paddedWin, padded1, padded1);
            Arithmetic.Multiply(paddedWin, padded2, padded2);
        }

        DftEngine<T>.Dft(padded1, fft1, inv: false, rowTransform: false, scaled: false, nonzeroRows: 0);
        DftEngine<T>.Dft(padded2, fft2, inv: false, rowTransform: false, scaled: false, nonzeroRows: 0);
        Mat<T> p = fft1;
        SpectrumKernels.Mul(fft1, fft2, p, rowsFlag: false, conjB: true);
        Mat<T> pm = fft2;
        PhaseCorrKernels.MagSpectrums(p, pm);
        Mat<T> c = p;
        PhaseCorrKernels.DivSpectrums(p, pm, c);
        Mat<T> shifted = padded1;
        DftEngine<T>.Dft(c, shifted, inv: true, rowTransform: false, scaled: false, nonzeroRows: 0);
        PhaseCorrKernels.FftShift(shifted, padded2);

        Point peak = Reductions.MinMaxLoc(shifted).MaxLoc;
        Point2d t = PhaseCorrKernels.WeightedCentroid(shifted, peak, new Size(5, 5), out response);
        response /= m * n;
        return new Point2d((n / 2.0) - t.X, (m / 2.0) - t.Y);
    }

    /// <summary>Detects the translation between two images without a window.</summary>
    /// <inheritdoc cref="PhaseCorrelate{T}(Mat{T}, Mat{T}, Mat{T}?, out double)"/>
    public static Point2d PhaseCorrelate<T>(Mat<T> src1, Mat<T> src2, Mat<T>? window = null)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
        => PhaseCorrelate(src1, src2, window, out _);

    /// <summary>
    /// Computes Hanning window coefficients in two dimensions, as OpenCV
    /// <c>createHanningWindow(dst, winSize, type)</c>:
    /// <c>sqrt((0.5·(1 - cos(2πx/(W-1)))) · (0.5·(1 - cos(2πy/(H-1)))))</c>.
    /// </summary>
    /// <param name="winSize">Window size; both dimensions must be greater than 1.</param>
    /// <returns>A new 1-channel window of type <typeparamref name="T"/> (<c>float</c> or <c>double</c>).</returns>
    /// <remarks>
    /// The column factors of the even-length prefix use the OpenCV vector cosine
    /// (<c>v_cos</c>, Cephes polynomial); the last column of an odd width and the row factors use
    /// the C library cosine, as OpenCV does.
    /// </remarks>
    public static Mat<T> CreateHanningWindow<T>(Size winSize)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        EnsureFloating<T>();
        if (winSize.Width <= 1 || winSize.Height <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(winSize), "Both window dimensions must be greater than 1.");
        }

        int rows = winSize.Height, cols = winSize.Width;
        var wc = new double[cols];
        double coeff0 = 2.0 * Math.PI / (cols - 1), coeff1 = 2.0 * Math.PI / (rows - 1);
        int c = 0;
        for (; c <= cols - 2; c++)
        {
            wc[c] = 0.5 * (1.0 - PhaseCorrKernels.SimdCos(coeff0 * c));
        }

        for (; c < cols; c++)
        {
            wc[c] = 0.5 * (1.0 - Math.Cos(coeff0 * c));
        }

        var dst = new Mat<T>(cols, rows, 1, clear: false);
        for (int i = 0; i < rows; i++)
        {
            Span<T> d = dst.GetRowSpan(i);
            double wr = 0.5 * (1.0 - Math.Cos(coeff1 * i));
            for (int j = 0; j < cols; j++)
            {
                T v = T.CreateTruncating(wr * wc[j]);
                d[j] = T.Sqrt(v);
            }
        }

        return dst;
    }

    /// <summary>
    /// Detects the translation between two images with the iterative sub-pixel refinement of
    /// OpenCV 5.0 <c>phaseCorrelateIterative(src1, src2, L2size, maxIters)</c>.
    /// </summary>
    /// <param name="src1">First image: 1-channel <c>float</c> or <c>double</c>.</param>
    /// <param name="src2">Second image of the type and size of <paramref name="src1"/>.</param>
    /// <param name="l2Size">Size of the peak neighbourhood that is upsampled (odd, at least 3).</param>
    /// <param name="maxIters">Maximum number of refinement iterations per L1 ratio.</param>
    /// <returns>The shift of <paramref name="src2"/> relative to <paramref name="src1"/>.</returns>
    public static Point2d PhaseCorrelateIterative<T>(Mat<T> src1, Mat<T> src2, int l2Size = 7, int maxIters = 10)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(src1);
        ArgumentNullException.ThrowIfNull(src2);
        EnsureFloating<T>();
        if (src1.Channels != 1 || src2.Channels != 1 || src1.Width != src2.Width || src1.Height != src2.Height)
        {
            throw new ArgumentException("The images must be 1-channel and of the same size.", nameof(src2));
        }

        return PhaseCorrIterative.Run(src1, src2, l2Size, maxIters);
    }

    private static void PadInto<T>(Mat<T> src, Mat<T> dst)
        where T : unmanaged
    {
        dst.Clear();
        src.CopyTo(dst.Roi(0, 0, src.Width, src.Height));
    }

    private static void EnsureFloating<T>()
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException("Phase correlation supports float and double images only.");
        }
    }
}
