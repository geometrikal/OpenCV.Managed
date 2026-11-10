// This implementation is ported from the OpenCV remap and convertMaps.
//
// Original source:
//   modules/imgproc/src/imgwarp.cpp (cv::remap, cv::convertMaps)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <content>Generic remap and map conversion.</content>
public static partial class Geometry
{
    /// <summary>
    /// Remaps an image with floating-point maps, as OpenCV <c>remap</c>:
    /// <c>dst(x, y) = src(mapX(x, y), mapY(x, y))</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>Maps.</b> Either two single-channel maps (<paramref name="mapY"/> not null), or one
    /// two-channel map with interleaved (x, y) and <paramref name="mapY"/> null. The destination
    /// has the map size. Coordinates are rounded to the nearest pixel for
    /// <see cref="InterpolationFlags.Nearest"/> and to 1/32 pixel (<c>cvRound(m·32)</c>) otherwise.
    /// Integer coordinates saturate to the 16-bit range.</para>
    /// <para><b>Borders.</b> <see cref="BorderType.Constant"/> uses <paramref name="borderValue"/>
    /// for taps outside the source, and writes the value without blending when the whole
    /// neighbourhood is outside. <see cref="BorderType.Transparent"/> leaves destination pixels
    /// unchanged when their neighbourhood leaves the source. For bilinear interpolation it follows
    /// OpenCV 5.0: a pixel whose base tap is inside the source but whose right or lower taps are
    /// outside uses the inside taps with weights renormalised by
    /// <c>Σw / Σw_inside</c>; OpenCV 4.x leaves such pixels unchanged. Other rules extrapolate each
    /// tap coordinate with <see cref="Borders.BorderInterpolate"/>.</para>
    /// <para><b>Relative maps.</b> With <see cref="WarpFlags.RelativeMap"/> (OpenCV 5.0) the map
    /// values are offsets: the destination coordinates are added to the integer part of the
    /// converted coordinates.</para>
    /// <para>Other rules as
    /// <see cref="WarpAffine{T}(Mat{T}, Mat{double}, Size, InterpolationFlags, WarpFlags, BorderType, Scalar)"/>.
    /// Source and destination sizes must be below 32767.</para>
    /// </remarks>
    /// <param name="source">Source image.</param>
    /// <param name="mapX">x map (one channel), or the interleaved (x, y) map (two channels).</param>
    /// <param name="mapY">y map (one channel), or null for an interleaved map.</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <param name="borderType">Border rule.</param>
    /// <param name="borderValue">Value for <see cref="BorderType.Constant"/>.</param>
    /// <param name="flags"><see cref="WarpFlags.RelativeMap"/> or <see cref="WarpFlags.None"/>.</param>
    /// <returns>A new image of the map size.</returns>
    public static Mat<T> Remap<T>(
        Mat<T> source,
        Mat<float> mapX,
        Mat<float>? mapY,
        InterpolationFlags interpolation,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default,
        WarpFlags flags = WarpFlags.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateFloatMaps(mapX, mapY);
        var dst = new Mat<T>(mapX.Width, mapX.Height, source.Channels, clear: borderType == BorderType.Transparent);
        RemapCore(source, dst, new FloatMapCoords(mapX, mapY), interpolation, borderType, borderValue, flags);
        return dst;
    }

    /// <summary>Remaps an image with floating-point maps into <paramref name="destination"/>, as OpenCV <c>remap</c>.</summary>
    /// <inheritdoc cref="Remap{T}(Mat{T}, Mat{float}, Mat{float}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>
    /// <param name="destination">Destination with the map size and the source channel count. It may be the source matrix.</param>
    public static void Remap<T>(
        Mat<T> source,
        Mat<T> destination,
        Mat<float> mapX,
        Mat<float>? mapY,
        InterpolationFlags interpolation,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default,
        WarpFlags flags = WarpFlags.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ValidateFloatMaps(mapX, mapY);
        RequireMapShape(destination, source, mapX.Width, mapX.Height);
        WithDetachedSource(source, destination, src => RemapCore(src, destination, new FloatMapCoords(mapX, mapY), interpolation, borderType, borderValue, flags));
    }

    /// <summary>
    /// Remaps an image with fixed-point maps from <see cref="ConvertMapsToFixedPoint"/>, as OpenCV
    /// <c>remap</c> with <c>CV_16SC2</c> and <c>CV_16UC1</c> maps.
    /// </summary>
    /// <remarks>
    /// <para><paramref name="xy"/> holds integer (x, y) pairs; <paramref name="fractions"/> holds the
    /// table index <c>(y fraction) · 32 + (x fraction)</c> in its low 10 bits.
    /// <paramref name="fractions"/> may be null only for <see cref="InterpolationFlags.Nearest"/>.</para>
    /// <para>Nearest-neighbour remap with fractions adds 1 to a coordinate whose fraction is
    /// below one half (OpenCV <c>NNDeltaTab_i</c>), so it can differ from rounding the original
    /// float map. Other rules as
    /// <see cref="Remap{T}(Mat{T}, Mat{float}, Mat{float}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>.</para>
    /// </remarks>
    /// <param name="source">Source image.</param>
    /// <param name="xy">Two-channel integer coordinate map.</param>
    /// <param name="fractions">One-channel table-index map of the same size, or null.</param>
    /// <param name="interpolation">Interpolation method.</param>
    /// <param name="borderType">Border rule.</param>
    /// <param name="borderValue">Value for <see cref="BorderType.Constant"/>.</param>
    /// <param name="flags"><see cref="WarpFlags.RelativeMap"/> or <see cref="WarpFlags.None"/>.</param>
    /// <returns>A new image of the map size.</returns>
    public static Mat<T> Remap<T>(
        Mat<T> source,
        Mat<short> xy,
        Mat<ushort>? fractions,
        InterpolationFlags interpolation,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default,
        WarpFlags flags = WarpFlags.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateFixedMaps(xy, fractions, interpolation);
        var dst = new Mat<T>(xy.Width, xy.Height, source.Channels, clear: borderType == BorderType.Transparent);
        RemapCore(source, dst, new FixedMapCoords(xy, fractions), interpolation, borderType, borderValue, flags);
        return dst;
    }

    /// <summary>Remaps an image with fixed-point maps into <paramref name="destination"/>, as OpenCV <c>remap</c>.</summary>
    /// <inheritdoc cref="Remap{T}(Mat{T}, Mat{short}, Mat{ushort}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>
    /// <param name="destination">Destination with the map size and the source channel count. It may be the source matrix.</param>
    public static void Remap<T>(
        Mat<T> source,
        Mat<T> destination,
        Mat<short> xy,
        Mat<ushort>? fractions,
        InterpolationFlags interpolation,
        BorderType borderType = BorderType.Constant,
        Scalar borderValue = default,
        WarpFlags flags = WarpFlags.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ValidateFixedMaps(xy, fractions, interpolation);
        RequireMapShape(destination, source, xy.Width, xy.Height);
        WithDetachedSource(source, destination, src => RemapCore(src, destination, new FixedMapCoords(xy, fractions), interpolation, borderType, borderValue, flags));
    }

    /// <summary>
    /// Converts floating-point maps to fixed-point maps, as OpenCV <c>convertMaps</c> with
    /// <c>dstmap1type = CV_16SC2</c>.
    /// </summary>
    /// <param name="mapX">x map (one channel), or the interleaved (x, y) map (two channels).</param>
    /// <param name="mapY">y map (one channel), or null for an interleaved map.</param>
    /// <param name="nearestInterpolation">
    /// True: integer pairs <c>cvRound(m)</c> and no fractions. False: pairs <c>cvRound(m·32) &gt;&gt; 5</c>
    /// and fractions <c>(iy &amp; 31)·32 + (ix &amp; 31)</c>.
    /// </param>
    /// <returns>The two-channel integer map and the fraction map (null when <paramref name="nearestInterpolation"/> is true).</returns>
    public static (Mat<short> Xy, Mat<ushort>? Fractions) ConvertMapsToFixedPoint(Mat<float> mapX, Mat<float>? mapY, bool nearestInterpolation = false)
    {
        ValidateFloatMaps(mapX, mapY);
        var xy = new Mat<short>(mapX.Width, mapX.Height, 2, clear: false);
        Mat<ushort>? fractions = nearestInterpolation ? null : new Mat<ushort>(mapX.Width, mapX.Height, 1, clear: false);
        var coords = new FloatMapCoords(mapX, mapY);
        Span<ushort> scratch = stackalloc ushort[0];
        for (int y = 0; y < mapX.Height; y++)
        {
            coords.Row(y, xy.GetRowSpan(y), fractions is null ? scratch : fractions.GetRowSpan(y), nearestInterpolation);
        }

        return (xy, fractions);
    }

    /// <summary>
    /// Converts fixed-point maps to two single-channel floating-point maps, as OpenCV
    /// <c>convertMaps</c> with <c>dstmap1type = CV_32FC1</c>:
    /// <c>x + (f &amp; 31)/32</c> and <c>y + (f &gt;&gt; 5)/32</c>.
    /// </summary>
    /// <param name="xy">Two-channel integer coordinate map.</param>
    /// <param name="fractions">Fraction map (low 10 bits used), or null for zero fractions.</param>
    /// <returns>The x and y maps.</returns>
    public static (Mat<float> MapX, Mat<float> MapY) ConvertMapsToFloat(Mat<short> xy, Mat<ushort>? fractions)
    {
        ValidateFixedPair(xy, fractions);
        var mx = new Mat<float>(xy.Width, xy.Height, 1, clear: false);
        var my = new Mat<float>(xy.Width, xy.Height, 1, clear: false);
        for (int y = 0; y < xy.Height; y++)
        {
            ReadOnlySpan<short> s = xy.GetReadOnlyRowSpan(y);
            ReadOnlySpan<ushort> f = fractions is null ? default : fractions.GetReadOnlyRowSpan(y);
            Span<float> dx = mx.GetRowSpan(y), dy = my.GetRowSpan(y);
            for (int x = 0; x < dx.Length; x++)
            {
                int fxy = fractions is null ? 0 : f[x] & (InterTables.Size2 - 1);
                dx[x] = s[2 * x] + ((fxy & (InterTables.Size - 1)) * (1f / InterTables.Size));
                dy[x] = s[(2 * x) + 1] + ((fxy >> InterTables.Bits) * (1f / InterTables.Size));
            }
        }

        return (mx, my);
    }

    /// <summary>
    /// Converts fixed-point maps to one interleaved floating-point map, as OpenCV
    /// <c>convertMaps</c> with <c>dstmap1type = CV_32FC2</c>.
    /// </summary>
    /// <param name="xy">Two-channel integer coordinate map.</param>
    /// <param name="fractions">Fraction map (low 10 bits used), or null for zero fractions.</param>
    /// <returns>The two-channel (x, y) map.</returns>
    public static Mat<float> ConvertMapsToFloatInterleaved(Mat<short> xy, Mat<ushort>? fractions)
    {
        (Mat<float> mx, Mat<float> my) = ConvertMapsToFloat(xy, fractions);
        return Conversions.Merge(mx, my);
    }

    /// <summary>Splits an interleaved floating-point map into x and y maps, as OpenCV <c>convertMaps</c> from <c>CV_32FC2</c> to <c>CV_32FC1</c>.</summary>
    /// <param name="map">Two-channel (x, y) map.</param>
    /// <returns>The x and y maps.</returns>
    public static (Mat<float> MapX, Mat<float> MapY) ConvertMapsToFloat(Mat<float> map)
    {
        ValidateFloatMaps(map, null);
        Mat<float>[] planes = Conversions.Split(map);
        return (planes[0], planes[1]);
    }

    /// <summary>Interleaves x and y maps, as OpenCV <c>convertMaps</c> from <c>CV_32FC1</c> to <c>CV_32FC2</c>.</summary>
    /// <param name="mapX">x map.</param>
    /// <param name="mapY">y map of the same size.</param>
    /// <returns>The two-channel (x, y) map.</returns>
    public static Mat<float> ConvertMapsToFloatInterleaved(Mat<float> mapX, Mat<float> mapY)
    {
        ArgumentNullException.ThrowIfNull(mapY);
        ValidateFloatMaps(mapX, mapY);
        return Conversions.Merge(mapX, mapY);
    }

    private static void RemapCore<T, TCoords>(
        Mat<T> src, Mat<T> dst, TCoords coords, InterpolationFlags interpolation, BorderType borderType, Scalar borderValue, WarpFlags flags)
        where T : unmanaged
        where TCoords : struct, IRowCoords
    {
        interpolation = WarpEngine.Validate<T>(interpolation, src.Channels);
        WarpEngine.ValidateBorder(borderType);
        if ((flags & ~WarpFlags.RelativeMap) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(flags), flags, "Remap accepts only WarpFlags.RelativeMap.");
        }

        if (src.Width >= short.MaxValue || src.Height >= short.MaxValue || dst.Width >= short.MaxValue || dst.Height >= short.MaxValue)
        {
            throw new ArgumentException("Remap requires source and destination sizes below 32767.");
        }

        WarpEngine.Run(src, dst, coords, interpolation, borderType, borderValue, (flags & WarpFlags.RelativeMap) != 0);
    }

    private static void ValidateFloatMaps(Mat<float> mapX, Mat<float>? mapY)
    {
        ArgumentNullException.ThrowIfNull(mapX);
        if (mapY is null)
        {
            if (mapX.Channels != 2)
            {
                throw new ArgumentException("A single map must have two channels (x, y).", nameof(mapX));
            }

            return;
        }

        if (mapX.Channels != 1 || mapY.Channels != 1 || mapX.Width != mapY.Width || mapX.Height != mapY.Height)
        {
            throw new ArgumentException("Planar maps must be single-channel and of equal size.", nameof(mapY));
        }
    }

    private static void ValidateFixedPair(Mat<short> xy, Mat<ushort>? fractions)
    {
        ArgumentNullException.ThrowIfNull(xy);
        if (xy.Channels != 2)
        {
            throw new ArgumentException("The coordinate map must have two channels.", nameof(xy));
        }

        if (fractions is not null && (fractions.Channels != 1 || fractions.Width != xy.Width || fractions.Height != xy.Height))
        {
            throw new ArgumentException("The fraction map must be single-channel with the size of the coordinate map.", nameof(fractions));
        }
    }

    private static void ValidateFixedMaps(Mat<short> xy, Mat<ushort>? fractions, InterpolationFlags interpolation)
    {
        ValidateFixedPair(xy, fractions);
        if (fractions is null && interpolation != InterpolationFlags.Nearest)
        {
            throw new ArgumentException("Interpolating remap with fixed-point maps requires the fraction map.", nameof(fractions));
        }
    }

    private static void RequireMapShape<T>(Mat<T> destination, Mat<T> source, int width, int height)
        where T : unmanaged
    {
        if (destination.Width != width || destination.Height != height || destination.Channels != source.Channels)
        {
            throw new ArgumentException("The destination must have the map size and the source channel count.", nameof(destination));
        }
    }
}
