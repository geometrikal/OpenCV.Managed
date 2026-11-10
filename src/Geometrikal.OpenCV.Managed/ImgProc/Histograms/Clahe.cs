// This implementation is ported from the OpenCV CLAHE implementation.
//
// Original sources:
//   modules/imgproc/src/clahe.cpp (createCLAHE, CLAHE_Impl)
//   modules/imgproc/include/opencv2/imgproc.hpp (class CLAHE)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Contrast Limited Adaptive Histogram Equalization, as OpenCV <c>createCLAHE</c> and
/// <c>CLAHE::apply</c>.
/// </summary>
/// <remarks>
/// <para>The image is divided into <see cref="TilesGridSize"/> tiles. If the width or the height
/// is not divisible by the grid, the image is padded at the bottom and right with
/// <see cref="BorderType.Reflect101"/>; in that case a divisible dimension is also padded by one
/// whole tile, as OpenCV. Each tile histogram is clipped at
/// <c>max((int)(ClipLimit * tileArea / histSize), 1)</c> and the excess is redistributed once.
/// Each pixel interpolates bilinearly between the tables of the four nearest tile centres.</para>
/// <para>The class has no internal buffers, so <see cref="Apply(Mat{byte}, Mat{byte})"/> is safe to
/// call from several threads with different destinations. OpenCV <c>collectGarbage</c> has no
/// equivalent.</para>
/// </remarks>
public sealed class Clahe
{
    private Size _tilesGridSize;
    private int _bitShift;

    /// <summary>Creates a CLAHE operator, as OpenCV <c>createCLAHE</c>.</summary>
    /// <param name="clipLimit">Contrast limit; 0 or a negative value disables clipping.</param>
    /// <param name="tilesGridSize">Number of tiles in x and y; null gives 8 x 8.</param>
    public Clahe(double clipLimit = 40.0, Size? tilesGridSize = null)
    {
        ClipLimit = clipLimit;
        TilesGridSize = tilesGridSize ?? new Size(8, 8);
    }

    /// <summary>Creates a CLAHE operator, as OpenCV <c>createCLAHE</c>.</summary>
    public static Clahe Create(double clipLimit = 40.0, Size? tilesGridSize = null) => new(clipLimit, tilesGridSize);

    /// <summary>Contrast limit; 0 or a negative value disables clipping.</summary>
    public double ClipLimit { get; set; }

    /// <summary>Number of tiles in x (<see cref="Size.Width"/>) and y (<see cref="Size.Height"/>); both positive.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A component is not positive.</exception>
    public Size TilesGridSize
    {
        get => _tilesGridSize;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value.Width, nameof(value));
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value.Height, nameof(value));
            _tilesGridSize = value;
        }
    }

    /// <summary>
    /// OpenCV 5.0 bit shift: values are shifted right by this amount before the histogram and the
    /// table lookup, and the result is shifted left. The histogram has <c>256 &gt;&gt; BitShift</c>
    /// (byte) or <c>65536 &gt;&gt; BitShift</c> (ushort) bins. The default 0 gives the OpenCV 4.x
    /// result.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is outside <c>[0, 15]</c>.</exception>
    public int BitShift
    {
        get => _bitShift;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(value, 15);
            _bitShift = value;
        }
    }

    /// <summary>Applies CLAHE to a single-channel <c>byte</c> image and returns a new image.</summary>
    public Mat<byte> Apply(Mat<byte> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<byte>(source.Width, source.Height, 1, clear: false);
        Apply(source, dst);
        return dst;
    }

    /// <summary>Applies CLAHE to a single-channel <c>ushort</c> image and returns a new image.</summary>
    public Mat<ushort> Apply(Mat<ushort> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        var dst = new Mat<ushort>(source.Width, source.Height, 1, clear: false);
        Apply(source, dst);
        return dst;
    }

    /// <summary>
    /// Applies CLAHE to a single-channel <c>byte</c> image. <paramref name="destination"/> has the
    /// source size and may be the same matrix as <paramref name="source"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A matrix has more than one channel, the sizes differ, or <see cref="BitShift"/> is 8 or more.</exception>
    public void Apply(Mat<byte> source, Mat<byte> destination) => Run(source, destination, 8);

    /// <summary>
    /// Applies CLAHE to a single-channel <c>ushort</c> image. <paramref name="destination"/> has the
    /// source size and may be the same matrix as <paramref name="source"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A matrix has more than one channel or the sizes differ.</exception>
    public void Apply(Mat<ushort> source, Mat<ushort> destination) => Run(source, destination, 16);

    private void Run<T>(Mat<T> source, Mat<T> destination, int bits)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        if (source.Channels != 1)
        {
            throw new ArgumentException("The source must have one channel.", nameof(source));
        }

        Core.Internal.Elementwise.ValidateSameShape(source, destination, nameof(destination));
        int shift = _bitShift;
        if (shift >= bits)
        {
            throw new ArgumentException($"BitShift must be below {bits} for this element type.");
        }

        ClaheEngine.Apply(source, destination, ClipLimit, _tilesGridSize.Width, _tilesGridSize.Height, shift);
    }
}
