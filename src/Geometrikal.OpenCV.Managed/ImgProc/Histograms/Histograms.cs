// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (calcHist, calcBackProject, compareHist, equalizeHist)
//   modules/imgproc/include/opencv2/imgproc.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's dense histogram functions: <c>calcHist</c>, <c>calcBackProject</c>,
/// <c>compareHist</c> and <c>equalizeHist</c>. No method modifies its source images.
/// </summary>
/// <remarks>
/// <para><b>Sources.</b> Histograms and back projections accept <c>byte</c>, <c>ushort</c> and
/// <c>float</c> images with any channel count. Other element types throw
/// <see cref="NotSupportedException"/>. All images must have the same size.</para>
/// <para><b>Channels.</b> <c>channels[i]</c> selects the plane of dimension <c>i</c> by a global
/// index over the channels of all images: the first image has indices
/// <c>0 .. images[0].Channels - 1</c>, the second image continues from there. An empty
/// <c>channels</c> requires one single-channel image per dimension.</para>
/// <para><b>Ranges.</b> With uniform bins, <c>ranges[i]</c> holds <c>[lo, hi)</c> of dimension
/// <c>i</c>; the upper bound is exclusive. The bin is <c>cvFloor(v * a + b)</c> in double with
/// <c>a = size / (hi - lo)</c> and <c>b = -a * lo</c>. With non-uniform bins, <c>ranges[i]</c>
/// holds <c>size + 1</c> strictly increasing boundaries and bin <c>k</c> covers
/// <c>[ranges[i][k], ranges[i][k + 1])</c>. Null ranges mean <c>[0, 256)</c> per dimension and
/// are valid for <c>byte</c> images only.</para>
/// <para><b>Mask.</b> A single-channel <c>byte</c> matrix of the image size; nonzero elements
/// select pixels. It replaces both OpenCV <c>CV_8U</c> and <c>CV_Bool</c> masks.</para>
/// <para><b>Threads.</b> Row stripes count into local integer histograms that are added at the
/// end. Results do not depend on <see cref="CvParallel.MaxDegreeOfParallelism"/>.</para>
/// <para>Sparse histograms and OpenCL, IPP and HAL paths are not ported.</para>
/// </remarks>
public static class Histograms
{
    /// <summary>
    /// Computes a dense histogram, as OpenCV <c>calcHist</c>.
    /// </summary>
    /// <param name="images">Source images of the same size and element type.</param>
    /// <param name="channels">Global channel index per dimension, or empty (see remarks of <see cref="Histograms"/>).</param>
    /// <param name="mask">Optional pixel mask.</param>
    /// <param name="histSize">Number of bins per dimension; 1 to <see cref="DenseHistogram.MaxDims"/> dimensions.</param>
    /// <param name="ranges">Bin ranges per dimension, or null for the implicit 8-bit range.</param>
    /// <param name="uniform">True for uniform bins; false for explicit bin boundaries.</param>
    /// <returns>A new histogram of bin counts.</returns>
    /// <remarks>
    /// Uniform bins of <c>ushort</c> and <c>float</c> images drop values outside <c>[lo, hi)</c>,
    /// then clamp the bin index to <c>[0, size - 1]</c>. A NaN value passes the range test and
    /// counts in bin 0, as OpenCV on x86. With non-uniform bins NaN is dropped. Counts above 2^24
    /// lose precision in the <c>float</c> result, as OpenCV.
    /// </remarks>
    public static DenseHistogram CalcHist<T>(
        ReadOnlySpan<Mat<T>> images,
        ReadOnlySpan<int> channels,
        Mat<byte>? mask,
        ReadOnlySpan<int> histSize,
        float[][]? ranges,
        bool uniform = true)
        where T : unmanaged
    {
        var hist = new DenseHistogram(histSize);
        CalcHist(images, channels, mask, hist, ranges, uniform, accumulate: false);
        return hist;
    }

    /// <summary>
    /// Computes a one-dimensional histogram of one channel with uniform bins over
    /// <c>[low, high)</c>.
    /// </summary>
    public static DenseHistogram CalcHist<T>(Mat<T> image, int channel, int histSize, float low, float high, Mat<byte>? mask = null)
        where T : unmanaged
        => CalcHist([image], [channel], mask, [histSize], [[low, high]]);

    /// <summary>
    /// Computes a dense histogram into <paramref name="hist"/>, as OpenCV <c>calcHist</c> with an
    /// existing output. The histogram sizes come from <paramref name="hist"/>.
    /// </summary>
    /// <param name="images">Source images of the same size and element type.</param>
    /// <param name="channels">Global channel index per dimension, or empty.</param>
    /// <param name="mask">Optional pixel mask.</param>
    /// <param name="hist">Destination histogram.</param>
    /// <param name="ranges">Bin ranges per dimension, or null for the implicit 8-bit range.</param>
    /// <param name="uniform">True for uniform bins; false for explicit bin boundaries.</param>
    /// <param name="accumulate">
    /// True to add the new counts to the existing bins. As OpenCV, every existing bin is first
    /// rounded to an integer (<c>cvRound</c>, half to even); fractional values are lost.
    /// False clears the histogram first.
    /// </param>
    public static void CalcHist<T>(
        ReadOnlySpan<Mat<T>> images,
        ReadOnlySpan<int> channels,
        Mat<byte>? mask,
        DenseHistogram hist,
        float[][]? ranges,
        bool uniform = true,
        bool accumulate = false)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(hist);
        HistLayout<T> layout = HistLayout<T>.Create(images, channels, hist.Sizes, ranges, uniform);
        ValidateMask(layout, mask);

        float[] data = hist.Data;
        var counts = new int[data.Length];
        if (accumulate)
        {
            for (int i = 0; i < data.Length; i++)
            {
                counts[i] = ScalarConverter.CvRound(data[i]);
            }
        }

        CalcHistEngine.Count(layout, mask, counts);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = counts[i];
        }
    }

    /// <summary>
    /// Computes the back projection of a histogram, as OpenCV <c>calcBackProject</c>: each output
    /// pixel is <c>saturate_cast&lt;T&gt;(hist[bin] * (float)scale)</c>, or 0 when the pixel
    /// falls outside the histogram.
    /// </summary>
    /// <param name="images">Source images of the same size and element type.</param>
    /// <param name="channels">Global channel index per histogram dimension, or empty.</param>
    /// <param name="hist">
    /// Histogram. A two-dimensional histogram with one row or one column is treated as
    /// one-dimensional (OpenCV 5.0).
    /// </param>
    /// <param name="ranges">Bin ranges per dimension, or null for the implicit 8-bit range.</param>
    /// <param name="scale">Scale factor; converted to <c>float</c>.</param>
    /// <param name="uniform">True for uniform bins; false for explicit bin boundaries.</param>
    /// <returns>A single-channel image of the source size and element type.</returns>
    /// <remarks>
    /// <c>byte</c> images use the <c>calcHist</c> bin rule (range test, then clamp). Uniform bins
    /// of <c>ushort</c> and <c>float</c> images do not test the range: a pixel belongs to the
    /// histogram only if <c>0 &lt;= cvFloor(v * a + b) &lt; size</c>. A value just below the upper
    /// bound whose index rounds to <c>size</c> therefore gives 0, and NaN gives 0.
    /// </remarks>
    public static Mat<T> CalcBackProject<T>(
        ReadOnlySpan<Mat<T>> images,
        ReadOnlySpan<int> channels,
        DenseHistogram hist,
        float[][]? ranges,
        double scale = 1,
        bool uniform = true)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(hist);
        ReadOnlySpan<int> sizes = hist.Sizes;
        if (sizes.Length == 2 && (sizes[0] == 1 || sizes[1] == 1))
        {
            sizes = new[] { sizes[0] + sizes[1] - 1 };
        }

        HistLayout<T> layout = HistLayout<T>.Create(images, channels, sizes, ranges, uniform);
        var dst = new Mat<T>(layout.Width, layout.Height, 1, clear: false);
        BackProjectEngine.Run(layout, hist.Data, (float)scale, dst);
        return dst;
    }

    /// <summary>
    /// Compares two histograms of the same sizes, as OpenCV <c>compareHist</c>.
    /// </summary>
    /// <remarks>
    /// The bins are processed as one flat array. The accumulation order is the OpenCV 5.0 order
    /// of a 128-bit vector build: blocks of four bins add into two double lanes, then a scalar
    /// tail adds the last <c>Length mod 4</c> bins. <see cref="HistCompMethods.Chisqr"/> and
    /// <see cref="HistCompMethods.ChisqrAlt"/> use the OpenCV 5.0 term
    /// <c>(a * a) * (1 / b)</c> in the blocks; their last bits differ from OpenCV 4.x.
    /// Bins with <c>|b| &lt;= DBL_EPSILON</c> are skipped. <see cref="HistCompMethods.KLDiv"/>
    /// replaces a second-histogram bin with <c>|q| &lt;= DBL_EPSILON</c> by <c>1e-10</c>.
    /// </remarks>
    public static double CompareHist(DenseHistogram h1, DenseHistogram h2, HistCompMethods method)
    {
        ArgumentNullException.ThrowIfNull(h1);
        ArgumentNullException.ThrowIfNull(h2);
        if (!h1.SameSizes(h2))
        {
            throw new ArgumentException("The histograms must have the same sizes.", nameof(h2));
        }

        var k = new CompareHistKernels(method);
        k.Plane(h1.Data, h2.Data);
        return k.Finish(h1.Length);
    }

    /// <summary>
    /// Compares two histograms stored as matrices of the same shape, as OpenCV
    /// <c>compareHist</c>. All channels are compared. Continuous matrices are processed as one
    /// array; otherwise each row is one plane, as the OpenCV plane iterator.
    /// </summary>
    public static double CompareHist(Mat<float> h1, Mat<float> h2, HistCompMethods method)
    {
        ArgumentNullException.ThrowIfNull(h1);
        Elementwise.ValidateSameShape(h1, h2, nameof(h2));
        var k = new CompareHistKernels(method);
        if (h1.IsContinuous && h2.IsContinuous)
        {
            k.Plane(h1.GetReadOnlySpan(), h2.GetReadOnlySpan());
        }
        else
        {
            for (int y = 0; y < h1.Height; y++)
            {
                k.Plane(h1.GetReadOnlyRowSpan(y), h2.GetReadOnlyRowSpan(y));
            }
        }

        return k.Finish(h1.PixelCount);
    }

    /// <summary>
    /// Equalizes the histogram of a single-channel <c>byte</c> image, as OpenCV
    /// <c>equalizeHist</c>.
    /// </summary>
    /// <returns>A new equalized image.</returns>
    public static Mat<byte> EqualizeHist(Mat<byte> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<byte>(source.Width, source.Height, 1, clear: false);
        EqualizeHist(source, dst);
        return dst;
    }

    /// <summary>
    /// Equalizes the histogram of a single-channel <c>byte</c> image into
    /// <paramref name="destination"/>, as OpenCV <c>equalizeHist</c>.
    /// </summary>
    /// <param name="source">Single-channel source image.</param>
    /// <param name="destination">
    /// Destination of the source size; may be the same matrix as <paramref name="source"/>.
    /// </param>
    /// <remarks>
    /// With <c>i</c> the smallest present value: if every pixel equals <c>i</c>, the destination is
    /// filled with <c>i</c>. Otherwise <c>scale = 255f / (total - hist[i])</c> in float and
    /// <c>lut[k] = saturate_cast&lt;byte&gt;((float)cumsum(hist[i+1..k]) * scale)</c>, with
    /// <c>lut[i] = 0</c>.
    /// </remarks>
    public static void EqualizeHist(Mat<byte> source, Mat<byte> destination)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (source.Channels != 1)
        {
            throw new ArgumentException("The source must have one channel.", nameof(source));
        }

        Elementwise.ValidateSameShape(source, destination, nameof(destination));

        var hist = new int[256];
        CalcHistEngine.Histogram256(source, 0, 1, null, hist);
        int i = 0;
        while (hist[i] == 0)
        {
            i++;
        }

        int total = source.PixelCount;
        if (hist[i] == total)
        {
            destination.Fill((byte)i);
            return;
        }

        float scale = 255f / (total - hist[i]);
        var lut = new byte[256];
        int sum = 0;
        for (int k = i + 1; k < 256; k++)
        {
            sum += hist[k];
            lut[k] = Saturate.FromFloat<byte>(sum * scale);
        }

        ParallelRows.For(
            source.Height,
            Elementwise.WorkPerRow<byte>(source.Width),
            (y0, y1) => ApplyLut(source, destination, lut, y0, y1));
    }

    private static void ApplyLut(Mat<byte> source, Mat<byte> destination, byte[] lut, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            ByteLut.Apply(source.GetReadOnlyRowSpan(y), destination.GetRowSpan(y), lut);
        }
    }

    private static void ValidateMask<T>(HistLayout<T> layout, Mat<byte>? mask)
        where T : unmanaged
    {
        if (mask is null)
        {
            return;
        }

        if (mask.Width != layout.Width || mask.Height != layout.Height || mask.Channels != 1)
        {
            throw new ArgumentException("The mask must be single-channel with the image size.", nameof(mask));
        }
    }
}
