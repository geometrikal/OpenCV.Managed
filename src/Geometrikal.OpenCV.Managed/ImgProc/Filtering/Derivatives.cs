// This implementation is ported from the OpenCV derivative filters.
//
// Original sources:
//   modules/imgproc/src/deriv.cpp (Sobel, Scharr, Laplacian, getDerivKernels)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed ports of OpenCV's derivative filters: <c>Sobel</c>,
/// <c>Scharr</c>, <c>Laplacian</c> and <c>getDerivKernels</c>.
///
/// <para>All operations are non-mutating and share the filtering
/// conventions of <see cref="Filters"/> (always-isolated ROIs, zero-valued
/// constant borders, <c>BORDER_WRAP</c> rejected). Element types are
/// <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> and <c>double</c>
/// (OpenCV's <c>CV_8U</c>, <c>CV_16U</c>, <c>CV_16S</c>, <c>CV_32F</c>,
/// <c>CV_64F</c>). The kernel type is OpenCV's
/// <c>ktype = max(CV_32F, sdepth, ddepth)</c>: double when the source or the
/// destination is double, otherwise float. 8-bit sources with <c>short</c>
/// destinations ride OpenCV's bit-exact integer path automatically because the
/// derivative kernels are integer-valued and symmetric/antisymmetric.</para>
/// </summary>
public static class Derivatives
{
    /// <summary>
    /// Returns the row and column kernels for a Sobel (or Scharr) derivative
    /// operator, mirroring OpenCV's <c>getDerivKernels</c>.
    /// </summary>
    /// <param name="dx">Derivative order in X; non-negative.</param>
    /// <param name="dy">Derivative order in Y; non-negative.</param>
    /// <param name="ksize">
    /// Aperture size: odd and ≤ 31, or any non-positive value
    /// (OpenCV's <c>FILTER_SCHARR</c>) for the 3×3 Scharr kernels
    /// (which require <c>dx + dy == 1</c>).
    /// </param>
    /// <param name="normalize">Whether the kernels are scaled to compensate for the aperture's gain.</param>
    /// <returns>The horizontal (X) and vertical (Y) kernels.</returns>
    public static (float[] KernelX, float[] KernelY) GetDerivKernels(
        int dx, int dy, int ksize, bool normalize = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(dx);
        ArgumentOutOfRangeException.ThrowIfNegative(dy);

        if (ksize <= 0)
        {
            if (dx + dy != 1)
            {
                throw new ArgumentException("Scharr kernels require dx >= 0, dy >= 0 and dx + dy == 1.");
            }

            return (DerivKernels.GetScharrKernel(dx, normalize), DerivKernels.GetScharrKernel(dy, normalize));
        }

        if (dx + dy <= 0)
        {
            throw new ArgumentException("Sobel kernels require dx + dy > 0.");
        }

        int ksizeX = ksize == 1 && dx > 0 ? 3 : ksize;
        int ksizeY = ksize == 1 && dy > 0 ? 3 : ksize;
        return (DerivKernels.GetSobelKernel(ksizeX, dx, normalize), DerivKernels.GetSobelKernel(ksizeY, dy, normalize));
    }

    /// <summary>
    /// Sobel derivative filter, mirroring OpenCV's <c>Sobel</c>. Passing a
    /// non-positive <paramref name="ksize"/> selects the Scharr kernels, as
    /// OpenCV's <c>FILTER_SCHARR</c> does.
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or
    /// <c>double</c>. <c>double→float</c> is not supported (OpenCV has no such column filter).
    /// Integer destinations round half to even and saturate; byte and ushort destinations
    /// saturate negative derivatives to zero, exactly as in OpenCV.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="dx">Derivative order in X.</param>
    /// <param name="dy">Derivative order in Y.</param>
    /// <param name="ksize">Aperture size: odd, ≤ 31; non-positive for Scharr.</param>
    /// <param name="scale">Scale factor folded into the smoothing kernel (OpenCV semantics).</param>
    /// <param name="delta">Value added to each result before saturation.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new derivative matrix.</returns>
    /// <exception cref="NotSupportedException">The type combination is not supported.</exception>
    public static Mat<TDst> Sobel<TSrc, TDst>(
        Mat<TSrc> source,
        int dx,
        int dy,
        int ksize = 3,
        double scale = 1,
        double delta = 0,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
        => SobelInto<TSrc, TDst>(source, dx, dy, ksize, scale, delta, borderType, null);

    /// <summary>
    /// Scharr derivative filter (3×3), mirroring OpenCV's <c>Scharr</c>.
    /// Requires <c>dx + dy == 1</c>.
    /// </summary>
    /// <typeparam name="TSrc">Source element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or
    /// <c>double</c>; <c>double→float</c> is not supported.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="dx">Derivative order in X (0 or 1).</param>
    /// <param name="dy">Derivative order in Y (0 or 1).</param>
    /// <param name="scale">Scale factor folded into the smoothing kernel.</param>
    /// <param name="delta">Value added to each result before saturation.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new derivative matrix.</returns>
    /// <exception cref="NotSupportedException">The type combination is not supported.</exception>
    public static Mat<TDst> Scharr<TSrc, TDst>(
        Mat<TSrc> source,
        int dx,
        int dy,
        double scale = 1,
        double delta = 0,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
        => SobelInto<TSrc, TDst>(source, dx, dy, 0, scale, delta, borderType, null);

    /// <summary>
    /// Laplacian filter, mirroring OpenCV's <c>Laplacian</c>: aperture 1 or 3
    /// applies the fixed 3×3 kernel via <c>filter2D</c>; larger apertures sum
    /// second derivatives computed separably in OpenCV's working depth
    /// (<c>short</c> for 8-bit sources with ksize ≤ 5, <c>double</c> for double
    /// sources, else <c>float</c>), then convert with <c>convertTo(scale, delta)</c>.
    /// </summary>
    /// <remarks>
    /// The aperture-1/3 kernel has OpenCV 5.0's kernel type: double when the source or the
    /// destination is double. OpenCV 4.x multiplied a float kernel by <paramref name="scale"/>
    /// for double images, so results for a scale that is not exact in float differ from 4.x.
    /// </remarks>
    /// <typeparam name="TSrc">Source element type: <c>byte</c>, <c>ushort</c>, <c>short</c>, <c>float</c> or <c>double</c>.</typeparam>
    /// <typeparam name="TDst">
    /// Destination element type. Aperture 1/3 requires a destination depth not below the source
    /// depth (<c>byte→any</c>, <c>ushort→ushort/float/double</c>,
    /// <c>short→short/float/double</c>, <c>float→float</c>, <c>double→double</c>); larger
    /// apertures accept every destination type.
    /// </typeparam>
    /// <param name="source">Source matrix, 1..4 channels.</param>
    /// <param name="ksize">Aperture size: odd, positive and ≤ 31.</param>
    /// <param name="scale">Scale factor applied to the result.</param>
    /// <param name="delta">Value added to each result before saturation.</param>
    /// <param name="borderType">Border extrapolation (constant borders use value 0).</param>
    /// <returns>A new filtered matrix.</returns>
    /// <exception cref="NotSupportedException">The type combination is not supported.</exception>
    public static Mat<TDst> Laplacian<TSrc, TDst>(
        Mat<TSrc> source,
        int ksize = 1,
        double scale = 1,
        double delta = 0,
        BorderType borderType = BorderType.Reflect101)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        Filters.ValidateSource(source);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ksize);
        bool doubleKernel = typeof(TSrc) == typeof(double) || typeof(TDst) == typeof(double);

        if (ksize == 1 || ksize == 3)
        {
            // The fixed aperture-1/3 kernels (deriv.cpp:723), multiplied by scale in the kernel type.
            double[] k = ksize == 3
                ? [2, 0, 2, 0, -8, 0, 2, 0, 2]
                : [0, 1, 0, 1, -4, 1, 0, 1, 0];
            if (scale != 1)
            {
                for (int i = 0; i < k.Length; i++)
                {
                    k[i] = doubleKernel ? k[i] * scale : (float)k[i] * (float)scale;
                }
            }

            return Filter2DEngine.Apply<TSrc, TDst>(source, k, 3, 3, new Point(-1, -1), delta, borderType);
        }

        double[] kd = ToDouble(DerivKernels.GetSobelKernel(ksize, 2, normalize: false));
        double[] ks = ToDouble(DerivKernels.GetSobelKernel(ksize, 0, normalize: false));

        if (typeof(TSrc) == typeof(byte) && ksize <= 5)
        {
            return LaplacianSeparable<TSrc, short, TDst>(source, kd, ks, scale, delta, borderType);
        }

        if (typeof(TSrc) == typeof(double))
        {
            return LaplacianSeparable<TSrc, double, TDst>(source, kd, ks, scale, delta, borderType);
        }

        return LaplacianSeparable<TSrc, float, TDst>(source, kd, ks, scale, delta, borderType);
    }

    /// <summary>
    /// Sobel/Scharr with an optional preallocated destination of the source size.
    /// </summary>
    internal static Mat<TDst> SobelInto<TSrc, TDst>(
        Mat<TSrc> source,
        int dx,
        int dy,
        int ksize,
        double scale,
        double delta,
        BorderType borderType,
        Mat<TDst>? destination)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        Filters.ValidateSource(source);
        (float[] fx, float[] fy) = GetDerivKernels(dx, dy, ksize);
        double[] kx = ToDouble(fx);
        double[] ky = ToDouble(fy);
        if (scale != 1)
        {
            // Sobel scales the smoothing kernel (kx when dx == 0) in the kernel type.
            bool doubleKernel = typeof(TSrc) == typeof(double) || typeof(TDst) == typeof(double);
            double[] target = dx == 0 ? kx : ky;
            for (int i = 0; i < target.Length; i++)
            {
                target[i] = doubleKernel ? target[i] * scale : (float)target[i] * (float)scale;
            }
        }

        return SepFilterEngine.Apply(source, kx, ky, new Point(-1, -1), delta, borderType, destination);
    }

    /// <summary>
    /// <c>d²x + d²y</c> in the work type <typeparamref name="TWork"/> (saturating for
    /// <c>short</c>), then <c>convertTo(dst, ddepth, scale, delta)</c> (deriv.cpp:771).
    /// </summary>
    private static Mat<TDst> LaplacianSeparable<TSrc, TWork, TDst>(
        Mat<TSrc> source, double[] kd, double[] ks, double scale, double delta, BorderType borderType)
        where TSrc : unmanaged
        where TWork : unmanaged
        where TDst : unmanaged
    {
        if (!SepFilterEngine.IsFilterDepth<TDst>())
        {
            throw new NotSupportedException($"Unsupported Laplacian destination type '{typeof(TDst)}'.");
        }

        Mat<TWork> d2x = SepFilterEngine.Apply<TSrc, TWork>(source, kd, ks, new Point(-1, -1), 0, borderType, null);
        Mat<TWork> d2y = SepFilterEngine.Apply<TSrc, TWork>(source, ks, kd, new Point(-1, -1), 0, borderType, null);
        Arithmetic.Add(d2x, d2y, d2x);
        return Conversions.ConvertTo<TWork, TDst>(d2x, scale, delta);
    }

    private static double[] ToDouble(float[] kernel)
    {
        var result = new double[kernel.Length];
        for (int i = 0; i < kernel.Length; i++)
        {
            result[i] = kernel[i];
        }

        return result;
    }
}
