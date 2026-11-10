// This implementation is ported from the opencv_contrib guided filter.
//
// Original source (opencv_contrib, module ximgproc, tag 5.x):
//   modules/ximgproc/src/guided_filter.cpp (GuidedFilter, createGuidedFilter)
//
// opencv_contrib is not part of this repository; the sources are at
// https://github.com/opencv/opencv_contrib.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Guided filter with precomputed guide statistics, mirroring OpenCV ximgproc
/// <c>GuidedFilter</c> (<c>createGuidedFilter</c> and <c>GuidedFilter::filter</c>).
/// </summary>
/// <remarks>
/// <para>The guide has 1 to 3 channels. The filter keeps the guide planes, their
/// box means and the inverse of the regularized guide covariance, so repeated
/// <see cref="Filter{TSrc, TDst}"/> calls with the same guide skip that work. An instance can
/// be used from several threads.</para>
/// <para>Box means use <c>boxFilter</c> with a <c>(2·radius+1)²</c> kernel, normalization and
/// <see cref="BorderType.Reflect"/>. A <c>scale</c> below 1 selects the fast guided
/// filter: the guide and the sources are resized with linear interpolation, and the
/// coefficients are resized back.</para>
/// <para>With a 1-channel guide, OpenCV inverts the variance with the x86 <c>RCPPS</c>
/// approximation on the first <c>4·⌊w/4⌋</c> columns of each row (the column count after
/// scaling) and with <c>1/x</c> on the others. <c>RCPPS</c> results depend on the processor
/// vendor and model; the port calls the same instruction, so its 1-channel-guide output equals
/// OpenCV's output on the same processor. Without SSE (for example ARM64) every column uses
/// <c>1/x</c>, as OpenCV built without SSE does. 2- and 3-channel guides are bit-exact on every
/// processor.</para>
/// </remarks>
public sealed class GuidedFilter
{
    private readonly GuidedFilterEngine _engine;

    private GuidedFilter(GuidedFilterEngine engine) => _engine = engine;

    /// <summary>
    /// Computes the guide statistics, as OpenCV <c>createGuidedFilter(guide, radius, eps, scale)</c>.
    /// </summary>
    /// <typeparam name="TGuide">Guide element type: <c>byte</c>, <c>ushort</c> or <c>float</c>.</typeparam>
    /// <param name="guide">Guide image with 1 to 3 channels. It is not modified.</param>
    /// <param name="radius">Box radius; the kernel is <c>(2·radius+1)²</c>. Non-negative.</param>
    /// <param name="eps">Regularization added to the guide variance. Non-negative.</param>
    /// <param name="scale">Subsampling factor of the fast guided filter, in (0, 1].</param>
    /// <returns>A filter for sources of the guide size.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A parameter is outside its range.</exception>
    /// <exception cref="NotSupportedException">The guide element type is unsupported.</exception>
    public static GuidedFilter Create<TGuide>(Mat<TGuide> guide, int radius, double eps, double scale = 1.0)
        where TGuide : unmanaged
        => CreateCore(guide, radius, eps, scale, pooled: false);

    /// <summary>One-shot filter whose planes come from the array pool and return to it afterwards.</summary>
    internal static Mat<TDst> FilterOnce<TGuide, TSrc, TDst>(Mat<TGuide> guide, Mat<TSrc> source, int radius, double eps, double scale)
        where TGuide : unmanaged
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        GuidedFilter filter = CreateCore(guide, radius, eps, scale, pooled: true);
        try
        {
            return filter.Filter<TSrc, TDst>(source);
        }
        finally
        {
            filter._engine.Release();
        }
    }

    private static GuidedFilter CreateCore<TGuide>(Mat<TGuide> guide, int radius, double eps, double scale, bool pooled)
        where TGuide : unmanaged
    {
        ArgumentNullException.ThrowIfNull(guide);
        if (typeof(TGuide) != typeof(byte) && typeof(TGuide) != typeof(ushort) && typeof(TGuide) != typeof(float))
        {
            throw new NotSupportedException($"Unsupported guide element type '{typeof(TGuide)}'. Supported: byte, ushort, float.");
        }

        if (guide.Channels > 3)
        {
            throw new NotSupportedException("The guide must have 1 to 3 channels, as in OpenCV.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(radius);
        if (!(eps >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(eps), eps, "eps must be non-negative.");
        }

        if (!(scale > 0 && scale <= 1.0))
        {
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "scale must be in (0, 1].");
        }

        return new GuidedFilter(GuidedFilterEngine.Create(guide, radius, eps, scale, pooled));
    }

    /// <summary>
    /// Filters a source of the guide size, as OpenCV <c>GuidedFilter::filter(src, dst, dDepth)</c>.
    /// Each source channel is filtered separately.
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type (OpenCV <c>dDepth</c>): <c>byte</c>, <c>ushort</c>, <c>short</c>,
    /// <c>float</c> or <c>double</c>. Integer types use <c>cvRound</c> and saturation.
    /// </typeparam>
    /// <param name="source">Source image with any channel count. It is not modified.</param>
    /// <returns>A new matrix with the source size and channel count.</returns>
    /// <exception cref="ArgumentException">The source size differs from the guide size.</exception>
    /// <exception cref="NotSupportedException">An element type is unsupported.</exception>
    public Mat<TDst> Filter<TSrc, TDst>(Mat<TSrc> source)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (typeof(TSrc) != typeof(byte) && typeof(TSrc) != typeof(float))
        {
            throw new NotSupportedException($"Unsupported source element type '{typeof(TSrc)}'. Supported: byte, float.");
        }

        if (typeof(TDst) != typeof(byte) && typeof(TDst) != typeof(ushort) && typeof(TDst) != typeof(short)
            && typeof(TDst) != typeof(float) && typeof(TDst) != typeof(double))
        {
            throw new NotSupportedException($"Unsupported destination element type '{typeof(TDst)}'.");
        }

        if (source.Width != _engine.Width || source.Height != _engine.Height)
        {
            throw new ArgumentException("The source size must equal the guide size.", nameof(source));
        }

        return _engine.Filter<TSrc, TDst>(source);
    }

    /// <summary>
    /// <see cref="Filter{TSrc, TDst}"/> with the source element type as the destination type
    /// (OpenCV <c>dDepth = -1</c>).
    /// </summary>
    /// <typeparam name="T">Element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <param name="source">Source image with any channel count. It is not modified.</param>
    /// <returns>A new matrix with the source size, channel count and element type.</returns>
    public Mat<T> Filter<T>(Mat<T> source)
        where T : unmanaged
        => Filter<T, T>(source);
}
