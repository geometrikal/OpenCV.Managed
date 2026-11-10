// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (histPrepareImages, calcHistLookupTables_8u)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Plane selection and bin parameters of one <c>calcHist</c> or <c>calcBackProject</c> call,
/// as OpenCV <c>histPrepareImages</c>.
/// </summary>
internal sealed class HistLayout<T>
    where T : unmanaged
{
    /// <summary>8U table entry for an out-of-range value; a sum of three entries stays below 2^31.</summary>
    public const int OutOfRange = 1 << 28;

    private HistLayout(
        Mat<T>[] images, int[] offsets, int[] steps, int[] sizes, int[] hsteps, double[]? a, double[]? b,
        double[]? lo, double[]? hi, float[][]? ranges)
    {
        Images = images;
        Offsets = offsets;
        Steps = steps;
        Sizes = sizes;
        HSteps = hsteps;
        A = a;
        B = b;
        Lo = lo;
        Hi = hi;
        Ranges = ranges;
        Width = images[0].Width;
        Height = images[0].Height;
        int total = 1;
        foreach (int s in sizes)
        {
            total *= s;
        }

        Total = total;
    }

    /// <summary>Image of each dimension.</summary>
    public Mat<T>[] Images { get; }

    /// <summary>Channel of each dimension inside its image.</summary>
    public int[] Offsets { get; }

    /// <summary>Channel count (element stride) of each dimension's image.</summary>
    public int[] Steps { get; }

    public int[] Sizes { get; }

    /// <summary>Element stride of each histogram dimension.</summary>
    public int[] HSteps { get; }

    /// <summary>Uniform bin scale per dimension: <c>bin = cvFloor(v * A + B)</c>. Null for non-uniform bins.</summary>
    public double[]? A { get; }

    public double[]? B { get; }

    /// <summary>Uniform range bounds per dimension, widened from <c>float</c>.</summary>
    public double[]? Lo { get; }

    public double[]? Hi { get; }

    /// <summary>Non-uniform bin boundaries; <c>Sizes[i] + 1</c> values per dimension.</summary>
    public float[][]? Ranges { get; }

    public bool Uniform => A is not null;

    public int Dims => Sizes.Length;

    public int Width { get; }

    public int Height { get; }

    public int Total { get; }

    /// <summary>
    /// Validates the arguments and selects the planes. <paramref name="ranges"/> null means the
    /// implicit 8U range <c>[0, 256)</c> per dimension (byte sources only).
    /// </summary>
    public static HistLayout<T> Create(
        ReadOnlySpan<Mat<T>> images, ReadOnlySpan<int> channels, ReadOnlySpan<int> sizes, float[][]? ranges, bool uniform)
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"Histograms of '{typeof(T).Name}' images are not supported.");
        }

        if (images.Length == 0)
        {
            throw new ArgumentException("At least one image is required.", nameof(images));
        }

        int dims = sizes.Length;
        if (dims is < 1 or > DenseHistogram.MaxDims)
        {
            throw new ArgumentException($"A histogram has 1 to {DenseHistogram.MaxDims} dimensions.", nameof(sizes));
        }

        Mat<T> first = images[0] ?? throw new ArgumentNullException(nameof(images));
        foreach (Mat<T> im in images)
        {
            ArgumentNullException.ThrowIfNull(im, nameof(images));
            if (im.Width != first.Width || im.Height != first.Height)
            {
                throw new ArgumentException("All images must have the same size.", nameof(images));
            }
        }

        if (channels.Length != 0 && channels.Length != dims)
        {
            throw new ArgumentException("Give one channel index per histogram dimension, or none.", nameof(channels));
        }

        var dimImages = new Mat<T>[dims];
        var offsets = new int[dims];
        var steps = new int[dims];
        for (int i = 0; i < dims; i++)
        {
            int j;
            int c;
            if (channels.Length == 0)
            {
                if (images.Length != dims || images[i].Channels != 1)
                {
                    throw new ArgumentException(
                        "Without channel indices, give one single-channel image per dimension.", nameof(channels));
                }

                j = i;
                c = 0;
            }
            else
            {
                c = channels[i];
                if (c < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(channels));
                }

                for (j = 0; j < images.Length && c >= images[j].Channels; j++)
                {
                    c -= images[j].Channels;
                }

                if (j == images.Length)
                {
                    throw new ArgumentOutOfRangeException(nameof(channels), "A channel index exceeds the total channel count.");
                }
            }

            dimImages[i] = images[j];
            offsets[i] = c;
            steps[i] = images[j].Channels;
        }

        long total = 1;
        var sizeArray = new int[dims];
        for (int i = 0; i < dims; i++)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sizes[i], nameof(sizes));
            sizeArray[i] = sizes[i];
            total *= sizes[i];
            if (total >= OutOfRange)
            {
                throw new ArgumentException("The histogram has too many bins.", nameof(sizes));
            }
        }

        var hsteps = new int[dims];
        int step = 1;
        for (int i = dims - 1; i >= 0; i--)
        {
            hsteps[i] = step;
            step *= sizeArray[i];
        }

        if (ranges is null)
        {
            if (typeof(T) != typeof(byte))
            {
                throw new ArgumentException("Ranges are required for non-byte images.", nameof(ranges));
            }

            var ia = new double[dims];
            for (int i = 0; i < dims; i++)
            {
                ia[i] = sizeArray[i] / 256.0;
            }

            return new HistLayout<T>(
                dimImages, offsets, steps, sizeArray, hsteps, ia, new double[dims], Filled(dims, 0), Filled(dims, 256), null);
        }

        if (ranges.Length != dims)
        {
            throw new ArgumentException("Give one range array per histogram dimension.", nameof(ranges));
        }

        if (uniform)
        {
            var a = new double[dims];
            var b = new double[dims];
            var lo = new double[dims];
            var hi = new double[dims];
            for (int i = 0; i < dims; i++)
            {
                float[] r = ranges[i] ?? throw new ArgumentNullException(nameof(ranges));
                if (r.Length < 2 || !(r[0] < r[1]))
                {
                    throw new ArgumentException("Each uniform range needs a lower bound below the upper bound.", nameof(ranges));
                }

                lo[i] = r[0];
                hi[i] = r[1];
                double t = sizeArray[i] / (hi[i] - lo[i]);
                a[i] = t;
                b[i] = -t * lo[i];
            }

            return new HistLayout<T>(dimImages, offsets, steps, sizeArray, hsteps, a, b, lo, hi, null);
        }

        var bounds = new float[dims][];
        for (int i = 0; i < dims; i++)
        {
            float[] r = ranges[i] ?? throw new ArgumentNullException(nameof(ranges));
            if (r.Length < sizeArray[i] + 1)
            {
                throw new ArgumentException("A non-uniform range needs histSize + 1 boundaries.", nameof(ranges));
            }

            for (int k = 0; k < sizeArray[i]; k++)
            {
                if (!(r[k] < r[k + 1]))
                {
                    throw new ArgumentException("Non-uniform boundaries must increase strictly.", nameof(ranges));
                }
            }

            bounds[i] = r[..(sizeArray[i] + 1)];
        }

        return new HistLayout<T>(dimImages, offsets, steps, sizeArray, hsteps, null, null, null, null, bounds);
    }

    /// <summary>
    /// <c>calcHistLookupTables_8u</c>: per dimension and byte value, the element offset of the bin
    /// or <see cref="OutOfRange"/>. Uniform bins test the range on the value and clamp the index.
    /// </summary>
    public int[] BuildTable8u()
    {
        int dims = Dims;
        var tab = new int[dims * 256];
        for (int i = 0; i < dims; i++)
        {
            Span<int> t = tab.AsSpan(i * 256, 256);
            int sz = Sizes[i];
            int hstep = HSteps[i];
            if (Uniform)
            {
                double a = A![i];
                double b = B![i];
                double vlo = Lo![i];
                double vhi = Hi![i];
                for (int j = 0; j < 256; j++)
                {
                    int idx = ScalarConverter.CvFloor((j * a) + b);
                    t[j] = j >= vlo && j < vhi ? Math.Clamp(idx, 0, sz - 1) * hstep : OutOfRange;
                }

                continue;
            }

            float[] r = Ranges![i];
            int limit = Math.Min(ScalarConverter.CvCeil(r[0]), 256);
            int bin = -1;
            int written = OutOfRange;
            int v = 0;
            while (true)
            {
                for (; v < limit; v++)
                {
                    t[v] = written;
                }

                if ((uint)(++bin) < (uint)sz)
                {
                    limit = Math.Min(ScalarConverter.CvCeil(r[bin + 1]), 256);
                    written = bin * hstep;
                }
                else
                {
                    for (; v < 256; v++)
                    {
                        t[v] = OutOfRange;
                    }

                    break;
                }
            }
        }

        return tab;
    }

    private static double[] Filled(int n, double v)
    {
        var a = new double[n];
        Array.Fill(a, v);
        return a;
    }
}
