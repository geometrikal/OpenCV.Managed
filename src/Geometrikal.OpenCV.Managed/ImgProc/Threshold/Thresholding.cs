// This implementation is ported from the OpenCV thresholding functions.
//
// Original sources:
//   modules/imgproc/src/thresh.cpp (threshold, thresholdWithMask, adaptiveThreshold)
//   modules/imgproc/include/opencv2/imgproc.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>threshold</c>, <c>thresholdWithMask</c> and
/// <c>adaptiveThreshold</c>.
/// </summary>
/// <remarks>
/// <para><b>Element types.</b> <see cref="Threshold{T}(Mat{T}, Mat{T}, double, double, ThresholdType, Mat{byte})"/>
/// accepts <c>byte</c>, <c>short</c>, <c>ushort</c>, <c>float</c> and <c>double</c> with any
/// channel count. The rule applies to every element; one threshold applies to all channels.
/// Other element types throw <see cref="NotSupportedException"/>.</para>
/// <para><b>Integer depths.</b> The threshold becomes <c>cvFloor(thresh)</c> and the maximum
/// value becomes <c>saturate_cast(cvRound(maxValue))</c>, with x86 semantics: NaN and +inf
/// thresholds give <see cref="int.MinValue"/>, -inf gives <see cref="int.MaxValue"/>. A floored
/// threshold outside <c>[min, max - 1]</c> of the element type fills or copies the whole
/// matrix, as OpenCV.</para>
/// <para><b>Floating depths.</b> <c>float</c> uses <c>(float)thresh</c> and
/// <c>(float)maxValue</c>. A NaN source or threshold fails every comparison: binary rules give
/// +0, and <see cref="ThresholdType.Trunc"/> gives the threshold for a NaN source. These are the
/// OpenCV 5.0 vector rules, applied to every element.</para>
/// <para><b>Results.</b> Overloads without a destination return a new matrix and never modify
/// the source. The destination overload writes into the destination; it may be the same matrix
/// as the source.</para>
/// </remarks>
public static class Thresholding
{
    private const ThresholdType KnownBits =
        ThresholdType.Mask | ThresholdType.Otsu | ThresholdType.Triangle | ThresholdType.DryRun;

    /// <summary>Applies a fixed-level or automatic threshold, as OpenCV <c>threshold</c>.</summary>
    /// <param name="source">Source matrix.</param>
    /// <param name="thresh">Threshold. Ignored with <see cref="ThresholdType.Otsu"/> or <see cref="ThresholdType.Triangle"/>.</param>
    /// <param name="maxValue">Value of passing elements for the binary rules.</param>
    /// <param name="type">Rule and flags. <see cref="ThresholdType.DryRun"/> is not accepted.</param>
    /// <returns>A new thresholded matrix.</returns>
    public static Mat<T> Threshold<T>(Mat<T> source, double thresh, double maxValue, ThresholdType type)
        where T : unmanaged
        => Threshold(source, thresh, maxValue, type, out _);

    /// <summary>Applies a fixed-level or automatic threshold, as OpenCV <c>threshold</c>.</summary>
    /// <param name="source">Source matrix.</param>
    /// <param name="thresh">Threshold. Ignored with <see cref="ThresholdType.Otsu"/> or <see cref="ThresholdType.Triangle"/>.</param>
    /// <param name="maxValue">Value of passing elements for the binary rules.</param>
    /// <param name="type">Rule and flags. <see cref="ThresholdType.DryRun"/> is not accepted.</param>
    /// <param name="usedThreshold">
    /// The threshold that was applied: <c>cvFloor</c> of the (computed) threshold for integer
    /// depths, <paramref name="thresh"/> for floating depths.
    /// </param>
    /// <returns>A new thresholded matrix.</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> contains <see cref="ThresholdType.DryRun"/>.</exception>
    public static Mat<T> Threshold<T>(
        Mat<T> source, double thresh, double maxValue, ThresholdType type, out double usedThreshold)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if ((type & ThresholdType.DryRun) != 0)
        {
            throw new ArgumentException(
                "DryRun requires the overload with a destination parameter.", nameof(type));
        }

        var destination = new Mat<T>(source.Width, source.Height, source.Channels, clear: false);
        usedThreshold = Threshold(source, destination, thresh, maxValue, type);
        return destination;
    }

    /// <summary>
    /// Applies a fixed-level or automatic threshold into <paramref name="destination"/>, as OpenCV
    /// <c>threshold</c>, or with a mask as OpenCV 5.0 <c>thresholdWithMask</c>.
    /// </summary>
    /// <param name="source">Source matrix.</param>
    /// <param name="destination">
    /// Destination with the shape of <paramref name="source"/>; it may be the same matrix. It may be
    /// null only when <paramref name="type"/> contains <see cref="ThresholdType.DryRun"/>.
    /// </param>
    /// <param name="thresh">Threshold. Ignored with <see cref="ThresholdType.Otsu"/> or <see cref="ThresholdType.Triangle"/>.</param>
    /// <param name="maxValue">Value of passing elements for the binary rules.</param>
    /// <param name="type">
    /// One rule, at most one of <see cref="ThresholdType.Otsu"/> and
    /// <see cref="ThresholdType.Triangle"/>, and optionally <see cref="ThresholdType.DryRun"/>
    /// (compute and return the threshold; the destination is not written).
    /// </param>
    /// <param name="mask">
    /// Optional single-channel mask of the source size. Where the mask is zero, the destination
    /// keeps its values, and Otsu and triangle use only the pixels where the mask is nonzero.
    /// Unlike OpenCV 5.0, the mask also applies when the floored threshold is outside the element
    /// range (OpenCV then fills or copies the whole destination).
    /// </param>
    /// <returns>
    /// The threshold that was applied: <c>cvFloor</c> of the (computed) threshold for integer
    /// depths, <paramref name="thresh"/> for floating depths.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Unknown rule or flag; both Otsu and triangle; Otsu on a source other than single-channel
    /// <c>byte</c> or <c>ushort</c>; triangle on a source other than single-channel <c>byte</c>;
    /// shape mismatch.
    /// </exception>
    /// <exception cref="NotSupportedException">Unsupported element type.</exception>
    public static double Threshold<T>(
        Mat<T> source,
        Mat<T>? destination,
        double thresh,
        double maxValue,
        ThresholdType type,
        Mat<byte>? mask = null)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        EnsureSupported<T>();
        bool dryRun = (type & ThresholdType.DryRun) != 0;
        if (destination is not null)
        {
            Elementwise.ValidateSameShape(source, destination, nameof(destination));
        }
        else if (!dryRun)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        Elementwise.ValidateMask(source, mask);
        int op = Decode(type, out ThresholdType automatic);

        if (automatic == ThresholdType.Otsu)
        {
            if (source.Channels != 1 || (typeof(T) != typeof(byte) && typeof(T) != typeof(ushort)))
            {
                throw new ArgumentException("Otsu requires a single-channel byte or ushort source.", nameof(type));
            }

            thresh = AutoThreshold.Otsu(source, mask);
        }
        else if (automatic == ThresholdType.Triangle)
        {
            if (source is not Mat<byte> bytes || source.Channels != 1)
            {
                throw new ArgumentException("Triangle requires a single-channel byte source.", nameof(type));
            }

            thresh = AutoThreshold.Triangle(bytes, mask);
        }

        if (Saturate.IsFloating<T>())
        {
            if (!dryRun)
            {
                ThresholdKernels.Run(
                    source, destination!, mask, Saturate.FromDouble<T>(thresh), Saturate.FromDouble<T>(maxValue), op);
            }

            return thresh;
        }

        int ithresh = ScalarConverter.CvFloor(thresh);
        if (dryRun)
        {
            return ithresh;
        }

        int lo = (int)Saturate.MinValue<T>();
        int hi = (int)Saturate.MaxValue<T>();
        int imaxval = op == 2 ? ithresh : ScalarConverter.CvRound(maxValue);
        imaxval = Math.Clamp(imaxval, lo, hi);

        if (ithresh < lo || ithresh >= hi)
        {
            bool fill = op is 0 or 1 || (op is 2 or 4 && ithresh < lo) || (op == 3 && ithresh >= hi);
            if (fill)
            {
                int v = op == 0 ? (ithresh >= hi ? 0 : imaxval)
                    : op == 1 ? (ithresh >= hi ? imaxval : 0)
                    : 0;
                Fill(destination!, Saturate.FromInt<T>(v), mask);
            }
            else if (mask is not null)
            {
                source.CopyTo(destination!, mask);
            }
            else if (!ReferenceEquals(source, destination))
            {
                source.CopyTo(destination!);
            }

            return ithresh;
        }

        ThresholdKernels.Run(
            source, destination!, mask, Saturate.FromInt<T>(ithresh), Saturate.FromInt<T>(imaxval), op);
        return ithresh;
    }

    /// <summary>
    /// Thresholds each pixel against the mean of its <paramref name="blockSize"/> ×
    /// <paramref name="blockSize"/> neighbourhood minus <paramref name="c"/>, as OpenCV
    /// <c>adaptiveThreshold</c>.
    /// </summary>
    /// <param name="source">Single-channel source.</param>
    /// <param name="maxValue">
    /// Value of passing pixels, converted with <c>saturate_cast&lt;byte&gt;</c>. A negative value
    /// returns an all-zero matrix before the method and type are checked, as OpenCV.
    /// </param>
    /// <param name="method">
    /// <see cref="AdaptiveThresholdType.MeanC"/>: box mean (<see cref="Filters.Blur{T}"/>).
    /// <see cref="AdaptiveThresholdType.GaussianC"/>: float32 Gaussian blur with sigma derived
    /// from the block size, rounded half to even to <c>byte</c>. Borders replicate edge pixels.
    /// </param>
    /// <param name="type"><see cref="ThresholdType.Binary"/> or <see cref="ThresholdType.BinaryInv"/>.</param>
    /// <param name="blockSize">Odd neighbourhood size, at least 3.</param>
    /// <param name="c">
    /// Constant subtracted from the mean. Binary uses <c>cvCeil(c)</c>, inverted binary uses
    /// <c>cvFloor(c)</c>, with x86 semantics for NaN and out-of-range values.
    /// </param>
    /// <returns>
    /// A new matrix: <c>src &gt; mean - ic ? maxValue : 0</c> (binary) or
    /// <c>src &lt;= mean - ic ? maxValue : 0</c> (inverted).
    /// </returns>
    /// <exception cref="ArgumentException">More than one channel, unknown method, or unsupported type.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="blockSize"/> is even or less than 3.</exception>
    public static Mat<byte> AdaptiveThreshold(
        Mat<byte> source,
        double maxValue,
        AdaptiveThresholdType method,
        ThresholdType type,
        int blockSize,
        double c)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.Channels != 1)
        {
            throw new ArgumentException("The source must have one channel.", nameof(source));
        }

        if (blockSize % 2 != 1 || blockSize <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(blockSize), "The block size must be odd and at least 3.");
        }

        if (maxValue < 0)
        {
            return new Mat<byte>(source.Width, source.Height);
        }

        if (method is not (AdaptiveThresholdType.MeanC or AdaptiveThresholdType.GaussianC))
        {
            throw new ArgumentException("Unknown adaptive threshold method.", nameof(method));
        }

        if (type is not (ThresholdType.Binary or ThresholdType.BinaryInv))
        {
            throw new ArgumentException("The threshold type must be Binary or BinaryInv.", nameof(type));
        }

        var ksize = new Size(blockSize, blockSize);
        Mat<byte> mean = method == AdaptiveThresholdType.MeanC
            ? Filters.Blur(source, ksize, null, BorderType.Replicate)
            : Conversions.ConvertTo<float, byte>(
                Filters.GaussianBlur(Conversions.ConvertTo<byte, float>(source), ksize, 0, 0, BorderType.Replicate));

        byte imaxval = ScalarConverter.SaturateCast<byte>(maxValue);
        int idelta = type == ThresholdType.Binary ? ScalarConverter.CvCeil(c) : ScalarConverter.CvFloor(c);
        ThresholdKernels.AdaptiveCompare(source, mean, imaxval, unchecked(-idelta), type == ThresholdType.BinaryInv);
        return mean;
    }

    private static int Decode(ThresholdType type, out ThresholdType automatic)
    {
        if ((type & ~KnownBits) != 0)
        {
            throw new ArgumentException("Unknown threshold flag.", nameof(type));
        }

        int op = (int)(type & ThresholdType.Mask);
        if (op > (int)ThresholdType.ToZeroInv)
        {
            throw new ArgumentException("Unknown threshold type.", nameof(type));
        }

        automatic = type & (ThresholdType.Otsu | ThresholdType.Triangle);
        if (automatic == (ThresholdType.Otsu | ThresholdType.Triangle))
        {
            throw new ArgumentException("Otsu and Triangle cannot be combined.", nameof(type));
        }

        return op;
    }

    private static void EnsureSupported<T>()
        where T : unmanaged
    {
        if (typeof(T) != typeof(byte) && typeof(T) != typeof(short) && typeof(T) != typeof(ushort) &&
            typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"The matrix element type '{typeof(T)}' is not supported.");
        }
    }

    private static void Fill<T>(Mat<T> destination, T value, Mat<byte>? mask)
        where T : unmanaged
    {
        if (mask is null)
        {
            destination.Fill(value);
            return;
        }

        int cn = destination.Channels;
        for (int y = 0; y < destination.Height; y++)
        {
            ReadOnlySpan<byte> m = mask.GetReadOnlyRowSpan(y);
            Span<T> d = destination.GetRowSpan(y);
            for (int x = 0; x < m.Length; x++)
            {
                if (m[x] != 0)
                {
                    d.Slice(x * cn, cn).Fill(value);
                }
            }
        }
    }
}
