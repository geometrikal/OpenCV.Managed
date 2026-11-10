// This implementation is ported from the opencv_contrib guided filter.
//
// Original source (opencv_contrib, module ximgproc, tag 5.x):
//   modules/ximgproc/src/guided_filter.cpp (guidedFilter)
//
// opencv_contrib is not part of this repository; the sources are at
// https://github.com/opencv/opencv_contrib.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Managed ports of OpenCV ximgproc edge-preserving filters: <c>guidedFilter</c>. The
/// bilateral filter of the imgproc module is <see cref="Filters.BilateralFilter{T}"/>.
/// </summary>
/// <remarks>Every operation returns a new matrix; the sources are not modified.</remarks>
public static class EdgePreservingFilters
{
    /// <summary>
    /// Guided filter with the source element type as the destination type, as OpenCV
    /// <c>ximgproc::guidedFilter(guide, src, dst, radius, eps, -1, scale)</c>.
    /// </summary>
    /// <typeparam name="TGuide">Guide element type: <c>byte</c>, <c>ushort</c> or <c>float</c>.</typeparam>
    /// <typeparam name="TSrc">Source element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <param name="guide">Guide image with 1 to 3 channels.</param>
    /// <param name="source">Source image of the guide size with any channel count.</param>
    /// <param name="radius">Box radius; the kernel is <c>(2·radius+1)²</c>. Non-negative.</param>
    /// <param name="eps">Regularization added to the guide variance. Non-negative.</param>
    /// <param name="scale">Subsampling factor of the fast guided filter, in (0, 1].</param>
    /// <returns>A new filtered matrix.</returns>
    /// <remarks>See <see cref="GuidedFilter"/> for the arithmetic and the processor dependence of 1-channel guides.</remarks>
    public static Mat<TSrc> GuidedFilter<TGuide, TSrc>(Mat<TGuide> guide, Mat<TSrc> source, int radius, double eps, double scale = 1.0)
        where TGuide : unmanaged
        where TSrc : unmanaged
        => ImgProc.GuidedFilter.FilterOnce<TGuide, TSrc, TSrc>(guide, source, radius, eps, scale);

    /// <summary>
    /// Guided filter with an explicit destination type (OpenCV <c>dDepth</c>), as OpenCV
    /// <c>ximgproc::guidedFilter(guide, src, dst, radius, eps, dDepth, scale)</c>.
    /// </summary>
    /// <typeparam name="TGuide">Guide element type: <c>byte</c>, <c>ushort</c> or <c>float</c>.</typeparam>
    /// <typeparam name="TSrc">Source element type: <c>byte</c> or <c>float</c>.</typeparam>
    /// <typeparam name="TDst">Destination element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <param name="guide">Guide image with 1 to 3 channels.</param>
    /// <param name="source">Source image of the guide size with any channel count.</param>
    /// <param name="radius">Box radius; the kernel is <c>(2·radius+1)²</c>. Non-negative.</param>
    /// <param name="eps">Regularization added to the guide variance. Non-negative.</param>
    /// <param name="scale">Subsampling factor of the fast guided filter, in (0, 1].</param>
    /// <returns>A new filtered matrix.</returns>
    public static Mat<TDst> GuidedFilter<TGuide, TSrc, TDst>(Mat<TGuide> guide, Mat<TSrc> source, int radius, double eps, double scale = 1.0)
        where TGuide : unmanaged
        where TSrc : unmanaged
        where TDst : unmanaged
        => ImgProc.GuidedFilter.FilterOnce<TGuide, TSrc, TDst>(guide, source, radius, eps, scale);
}
