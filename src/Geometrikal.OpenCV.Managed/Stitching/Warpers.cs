// This implementation is ported from the OpenCV stitching warpers.
//
// Original sources:
//   modules/stitching/src/warpers.cpp
//   modules/stitching/include/opencv2/stitching/detail/warpers.hpp
//   modules/stitching/include/opencv2/stitching/detail/warpers_inl.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>
/// Base class of the rotation warpers, as OpenCV <c>detail::RotationWarper</c> and
/// <c>detail::RotationWarperBase</c>.
/// </summary>
/// <remarks>
/// <para><b>Camera parameters.</b> <c>K</c> is the 3×3 <c>float</c> intrinsic matrix. <c>R</c> is
/// the 3×3 <c>float</c> camera rotation. The warper computes <c>K⁻¹</c> (LU), <c>R·K⁻¹</c> and
/// <c>K·Rᵀ</c> in <c>float</c>, as OpenCV.</para>
/// <para><b>Arithmetic.</b> All projections compute in <c>float</c> with the OpenCV operation order.
/// The trigonometric projections use <see cref="MathF"/>.</para>
/// <para><b>State.</b> Each call sets the camera parameters of the instance. An instance is not
/// thread-safe.</para>
/// </remarks>
public abstract class RotationWarper
{
    private protected readonly ProjectorParams Projector = new();

    private protected RotationWarper(float scale)
    {
        Projector.Scale = scale;
    }

    /// <summary>Gets or sets the projection scale.</summary>
    public float Scale
    {
        get => Projector.Scale;
        set => Projector.Scale = value;
    }

    /// <summary>Projects a source image point to the warped image, as OpenCV <c>warpPoint</c>.</summary>
    /// <param name="point">Source image point.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="r">3×3 camera rotation.</param>
    /// <returns>The projected point.</returns>
    public virtual Point2f WarpPoint(Point2f point, Mat<float> k, Mat<float> r)
    {
        SetCameraParams(k, r);
        MapForward(point.X, point.Y, out float u, out float v);
        return new Point2f(u, v);
    }

    /// <summary>Projects a warped image point back to the source image, as OpenCV <c>warpPointBackward</c>.</summary>
    /// <param name="point">Warped image point.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="r">3×3 camera rotation.</param>
    /// <returns>The source image point.</returns>
    public virtual Point2f WarpPointBackward(Point2f point, Mat<float> k, Mat<float> r)
    {
        SetCameraParams(k, r);
        MapBackward(point.X, point.Y, out float x, out float y);
        return new Point2f(x, y);
    }

    /// <summary>Builds the backward maps of the warp, as OpenCV <c>buildMaps</c>.</summary>
    /// <param name="sourceSize">Source image size.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="r">3×3 camera rotation.</param>
    /// <param name="mapX">Receives the source x coordinate of each warped pixel.</param>
    /// <param name="mapY">Receives the source y coordinate of each warped pixel.</param>
    /// <returns>
    /// The warped region: its top-left corner, and a width and height one smaller than the map
    /// size, as OpenCV <c>Rect(dst_tl, dst_br)</c>.
    /// </returns>
    public virtual Rect BuildMaps(Size sourceSize, Mat<float> k, Mat<float> r, out Mat<float> mapX, out Mat<float> mapY)
    {
        SetCameraParams(k, r);
        return BuildMapsCore(sourceSize, out mapX, out mapY);
    }

    /// <summary>Warps an image, as OpenCV <c>warp</c> (<c>buildMaps</c> and <c>remap</c>).</summary>
    /// <typeparam name="T">Element type supported by <see cref="Geometry.Remap{T}(Mat{T}, Mat{float}, Mat{float}, InterpolationFlags, BorderType, Scalar, WarpFlags)"/>.</typeparam>
    /// <param name="source">Source image. It is not modified.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="r">3×3 camera rotation.</param>
    /// <param name="interpolation">Remap interpolation.</param>
    /// <param name="borderType">Remap border rule; constant borders use 0.</param>
    /// <param name="destination">Receives the warped image.</param>
    /// <returns>The top-left corner of the warped image in the warped space.</returns>
    public virtual Point Warp<T>(Mat<T> source, Mat<float> k, Mat<float> r, InterpolationFlags interpolation, BorderType borderType, out Mat<T> destination)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        SetCameraParams(k, r);
        return WarpCore(source, interpolation, borderType, out destination);
    }

    /// <summary>Projects a warped image back to the source image plane, as OpenCV <c>warpBackward</c>.</summary>
    /// <typeparam name="T">Element type supported by remap.</typeparam>
    /// <param name="source">Warped image. Its size must equal the warped region of <paramref name="destinationSize"/>.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="r">3×3 camera rotation.</param>
    /// <param name="interpolation">Remap interpolation.</param>
    /// <param name="borderType">Remap border rule; constant borders use 0.</param>
    /// <param name="destinationSize">Size of the source image plane.</param>
    /// <returns>The image in the source image plane.</returns>
    /// <exception cref="ArgumentException">The source size does not match the warped region.</exception>
    public virtual Mat<T> WarpBackward<T>(Mat<T> source, Mat<float> k, Mat<float> r, InterpolationFlags interpolation, BorderType borderType, Size destinationSize)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        SetCameraParams(k, r);
        DetectResultRoi(destinationSize, out Point srcTl, out Point srcBr);
        if (srcBr.X - srcTl.X + 1 != source.Width || srcBr.Y - srcTl.Y + 1 != source.Height)
        {
            throw new ArgumentException("The source size does not match the warped region of the destination size.", nameof(source));
        }

        using PooledMat<float> mapX = PooledMat<float>.Rent(destinationSize.Width, destinationSize.Height, 1);
        using PooledMat<float> mapY = PooledMat<float>.Rent(destinationSize.Width, destinationSize.Height, 1);
        BuildBackwardMapsCore(srcTl, mapX.Mat, mapY.Mat);
        return Geometry.Remap(source, mapX.Mat, mapY.Mat, interpolation, borderType);
    }

    /// <summary>Returns the warped region of a source image, as OpenCV <c>warpRoi</c>.</summary>
    /// <param name="sourceSize">Source image size.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="r">3×3 camera rotation.</param>
    /// <returns>The warped region; its size equals the size of the warped image.</returns>
    public virtual Rect WarpRoi(Size sourceSize, Mat<float> k, Mat<float> r)
    {
        SetCameraParams(k, r);
        DetectResultRoi(sourceSize, out Point tl, out Point br);
        return new Rect(tl.X, tl.Y, br.X + 1 - tl.X, br.Y + 1 - tl.Y);
    }

    private protected virtual void SetCameraParams(Mat<float> k, Mat<float> r) => Projector.SetCameraParams(k, r, null);

    private protected abstract void MapForward(float x, float y, out float u, out float v);

    private protected abstract void MapBackward(float u, float v, out float x, out float y);

    private protected abstract void DetectResultRoi(Size sourceSize, out Point tl, out Point br);

    private protected abstract void FillMaps(Point tl, Mat<float> mapX, Mat<float> mapY);

    private protected abstract void BuildBackwardMapsCore(Point srcTl, Mat<float> mapX, Mat<float> mapY);

    /// <summary>Warps with the current camera parameters; the maps are pooled temporaries.</summary>
    private protected Point WarpCore<T>(Mat<T> source, InterpolationFlags interpolation, BorderType borderType, out Mat<T> destination)
        where T : unmanaged
    {
        Size sourceSize = new(source.Width, source.Height);
        DetectResultRoi(sourceSize, out Point tl, out Point br);
        int width = br.X - tl.X + 1;
        int height = br.Y - tl.Y + 1;
        if (width <= 0 || height <= 0)
        {
            throw new InvalidOperationException("The warped region is empty.");
        }

        using PooledMat<float> mapX = PooledMat<float>.Rent(width, height, 1);
        using PooledMat<float> mapY = PooledMat<float>.Rent(width, height, 1);
        FillMaps(tl, mapX.Mat, mapY.Mat);
        destination = Geometry.Remap(source, mapX.Mat, mapY.Mat, interpolation, borderType);
        return tl;
    }

    private protected Rect BuildMapsCore(Size sourceSize, out Mat<float> mapX, out Mat<float> mapY)
    {
        if (sourceSize.Width <= 0 || sourceSize.Height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceSize), "The source size must be positive.");
        }

        DetectResultRoi(sourceSize, out Point tl, out Point br);
        int width = br.X - tl.X + 1;
        int height = br.Y - tl.Y + 1;
        if (width <= 0 || height <= 0)
        {
            throw new InvalidOperationException("The warped region is empty.");
        }

        mapX = new Mat<float>(width, height, 1, clear: false);
        mapY = new Mat<float>(width, height, 1, clear: false);
        FillMaps(tl, mapX, mapY);
        return new Rect(tl.X, tl.Y, br.X - tl.X, br.Y - tl.Y);
    }
}

/// <summary>
/// Warper that projects an image onto the plane <c>z = 1</c>, as OpenCV <c>detail::PlaneWarper</c>.
/// </summary>
/// <remarks>
/// The overloads with a translation <c>T</c> (3×1 or 1×3 <c>float</c>) project onto the plane
/// shifted by <c>T</c>. The result region uses the four projected source corners only.
/// <c>BuildMaps</c> computes the maps in <c>Vector&lt;float&gt;</c> lanes with the OpenCV operation
/// order; the result is bit-exact with the scalar OpenCV loop.
/// </remarks>
public class PlaneWarper : RotationWarper
{
    /// <summary>Creates a plane warper.</summary>
    /// <param name="scale">Projection scale.</param>
    public PlaneWarper(float scale = 1f)
        : base(scale)
    {
    }

    /// <summary>Projects a source point with a translation, as OpenCV <c>warpPoint(pt, K, R, T)</c>.</summary>
    public Point2f WarpPoint(Point2f point, Mat<float> k, Mat<float> r, Mat<float> t)
    {
        Projector.SetCameraParams(k, r, t);
        PlaneProjector.MapForward(Projector, point.X, point.Y, out float u, out float v);
        return new Point2f(u, v);
    }

    /// <summary>Projects a warped point back with a translation, as OpenCV <c>warpPointBackward(pt, K, R, T)</c>.</summary>
    public Point2f WarpPointBackward(Point2f point, Mat<float> k, Mat<float> r, Mat<float> t)
    {
        Projector.SetCameraParams(k, r, t);
        PlaneProjector.MapBackward(Projector, point.X, point.Y, out float x, out float y);
        return new Point2f(x, y);
    }

    /// <summary>Builds the backward maps with a translation, as OpenCV <c>buildMaps(src_size, K, R, T, ...)</c>.</summary>
    public Rect BuildMaps(Size sourceSize, Mat<float> k, Mat<float> r, Mat<float> t, out Mat<float> mapX, out Mat<float> mapY)
    {
        Projector.SetCameraParams(k, r, t);
        return BuildMapsCore(sourceSize, out mapX, out mapY);
    }

    /// <summary>Warps an image with a translation, as OpenCV <c>warp(src, K, R, T, ...)</c>.</summary>
    public Point Warp<T>(Mat<T> source, Mat<float> k, Mat<float> r, Mat<float> t, InterpolationFlags interpolation, BorderType borderType, out Mat<T> destination)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Projector.SetCameraParams(k, r, t);
        return WarpCore(source, interpolation, borderType, out destination);
    }

    /// <summary>Returns the warped region with a translation, as OpenCV <c>warpRoi(src_size, K, R, T)</c>.</summary>
    public Rect WarpRoi(Size sourceSize, Mat<float> k, Mat<float> r, Mat<float> t)
    {
        Projector.SetCameraParams(k, r, t);
        DetectResultRoi(sourceSize, out Point tl, out Point br);
        return new Rect(tl.X, tl.Y, br.X + 1 - tl.X, br.Y + 1 - tl.Y);
    }

    private protected override void MapForward(float x, float y, out float u, out float v) => PlaneProjector.MapForward(Projector, x, y, out u, out v);

    private protected override void MapBackward(float u, float v, out float x, out float y) => PlaneProjector.MapBackward(Projector, u, v, out x, out y);

    private protected override void DetectResultRoi(Size sourceSize, out Point tl, out Point br) => WarpMaps.DetectRoiPlane(Projector, sourceSize, out tl, out br);

    private protected override void FillMaps(Point tl, Mat<float> mapX, Mat<float> mapY) => WarpMaps.BuildMaps<PlaneProjector>(Projector, tl, mapX, mapY);

    private protected override void BuildBackwardMapsCore(Point srcTl, Mat<float> mapX, Mat<float> mapY) => WarpMaps.BuildBackwardMaps<PlaneProjector>(Projector, srcTl, mapX, mapY);
}

/// <summary>
/// Plane warper driven by a 3×3 affine matrix <c>H</c> in place of the rotation, as OpenCV
/// <c>detail::AffineWarper</c>.
/// </summary>
/// <remarks>
/// <c>WarpPoint</c>, <c>WarpPointBackward</c>, <c>BuildMaps</c>, <c>Warp</c> and <c>WarpRoi</c>
/// split <c>H</c> into <c>R = Hₗᵀ</c> (with <c>H[0,2] = H[1,2] = 0</c>) and
/// <c>T = −R·(H[0,2], H[1,2], 0)ᵀ</c>, as OpenCV <c>getRTfromHomogeneous</c>, and call the plane
/// warper. <c>WarpBackward</c> is not overridden in OpenCV: it uses <c>H</c> as the rotation and
/// a zero translation. The port keeps this behaviour.
/// </remarks>
public class AffineWarper : PlaneWarper
{
    /// <summary>Creates an affine warper.</summary>
    /// <param name="scale">Projection scale.</param>
    public AffineWarper(float scale = 1f)
        : base(scale)
    {
    }

    /// <inheritdoc/>
    /// <param name="point">Source image point.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="h">3×3 affine transform.</param>
    public override Point2f WarpPoint(Point2f point, Mat<float> k, Mat<float> h)
    {
        GetRtFromHomogeneous(h, out Mat<float> r, out Mat<float> t);
        return WarpPoint(point, k, r, t);
    }

    /// <inheritdoc/>
    /// <param name="point">Warped image point.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="h">3×3 affine transform.</param>
    public override Point2f WarpPointBackward(Point2f point, Mat<float> k, Mat<float> h)
    {
        GetRtFromHomogeneous(h, out Mat<float> r, out Mat<float> t);
        return WarpPointBackward(point, k, r, t);
    }

    /// <inheritdoc/>
    /// <param name="sourceSize">Source image size.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="h">3×3 affine transform.</param>
    /// <param name="mapX">Receives the source x coordinate of each warped pixel.</param>
    /// <param name="mapY">Receives the source y coordinate of each warped pixel.</param>
    public override Rect BuildMaps(Size sourceSize, Mat<float> k, Mat<float> h, out Mat<float> mapX, out Mat<float> mapY)
    {
        GetRtFromHomogeneous(h, out Mat<float> r, out Mat<float> t);
        return BuildMaps(sourceSize, k, r, t, out mapX, out mapY);
    }

    /// <inheritdoc/>
    /// <param name="source">Source image. It is not modified.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="h">3×3 affine transform.</param>
    /// <param name="interpolation">Remap interpolation.</param>
    /// <param name="borderType">Remap border rule.</param>
    /// <param name="destination">Receives the warped image.</param>
    public override Point Warp<T>(Mat<T> source, Mat<float> k, Mat<float> h, InterpolationFlags interpolation, BorderType borderType, out Mat<T> destination)
    {
        GetRtFromHomogeneous(h, out Mat<float> r, out Mat<float> t);
        return Warp(source, k, r, t, interpolation, borderType, out destination);
    }

    /// <inheritdoc/>
    /// <param name="sourceSize">Source image size.</param>
    /// <param name="k">3×3 camera intrinsic matrix.</param>
    /// <param name="h">3×3 affine transform.</param>
    public override Rect WarpRoi(Size sourceSize, Mat<float> k, Mat<float> h)
    {
        GetRtFromHomogeneous(h, out Mat<float> r, out Mat<float> t);
        return WarpRoi(sourceSize, k, r, t);
    }

    /// <summary>OpenCV <c>AffineWarper::getRTfromHomogeneous</c>.</summary>
    private static void GetRtFromHomogeneous(Mat<float> h, out Mat<float> r, out Mat<float> t)
    {
        ProjectorParams.Validate3x3(h, nameof(h));
        var rr = new Mat<float>(3, 3);
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                rr.At(x, y) = h.At(y, x);
            }
        }

        var t0 = new Mat<float>(1, 3);
        t0.At(0, 0) = h.At(0, 2);
        t0.At(1, 0) = h.At(1, 2);
        rr.At(2, 0) = 0f;
        rr.At(2, 1) = 0f;
        r = rr;
        t = LinearAlgebra.Gemm(rr, t0, -1, null, 0);
    }
}

/// <summary>
/// Warper that projects an image onto the cylinder <c>x² + z² = 1</c>, as OpenCV
/// <c>detail::CylindricalWarper</c>.
/// </summary>
/// <remarks>The result region uses every pixel of the source border (<c>detectResultRoiByBorder</c>).</remarks>
public class CylindricalWarper : RotationWarper
{
    /// <summary>Creates a cylindrical warper.</summary>
    /// <param name="scale">Projection scale (cylinder radius in pixels).</param>
    public CylindricalWarper(float scale)
        : base(scale)
    {
    }

    private protected override void MapForward(float x, float y, out float u, out float v) => CylindricalProjector.MapForward(Projector, x, y, out u, out v);

    private protected override void MapBackward(float u, float v, out float x, out float y) => CylindricalProjector.MapBackward(Projector, u, v, out x, out y);

    private protected override void DetectResultRoi(Size sourceSize, out Point tl, out Point br) => WarpMaps.DetectRoiByBorder<CylindricalProjector>(Projector, sourceSize, out tl, out br);

    private protected override void FillMaps(Point tl, Mat<float> mapX, Mat<float> mapY) => WarpMaps.BuildMaps<CylindricalProjector>(Projector, tl, mapX, mapY);

    private protected override void BuildBackwardMapsCore(Point srcTl, Mat<float> mapX, Mat<float> mapY) => WarpMaps.BuildBackwardMaps<CylindricalProjector>(Projector, srcTl, mapX, mapY);
}

/// <summary>
/// Warper that projects an image onto the unit sphere, as OpenCV <c>detail::SphericalWarper</c>.
/// </summary>
/// <remarks>
/// A 360° panorama has a width of <c>2·π·scale</c>. The result region uses the source border and
/// extends to a pole when the pole projects inside the source image, as OpenCV
/// <c>SphericalWarper::detectResultRoi</c>.
/// </remarks>
public class SphericalWarper : RotationWarper
{
    /// <summary>Creates a spherical warper.</summary>
    /// <param name="scale">Projection scale (sphere radius in pixels).</param>
    public SphericalWarper(float scale)
        : base(scale)
    {
    }

    private protected override void MapForward(float x, float y, out float u, out float v) => SphericalProjector.MapForward(Projector, x, y, out u, out v);

    private protected override void MapBackward(float u, float v, out float x, out float y) => SphericalProjector.MapBackward(Projector, u, v, out x, out y);

    private protected override void DetectResultRoi(Size sourceSize, out Point tl, out Point br) => WarpMaps.DetectRoiSpherical(Projector, sourceSize, out tl, out br);

    private protected override void FillMaps(Point tl, Mat<float> mapX, Mat<float> mapY) => WarpMaps.BuildMaps<SphericalProjector>(Projector, tl, mapX, mapY);

    private protected override void BuildBackwardMapsCore(Point srcTl, Mat<float> mapX, Mat<float> mapY) => WarpMaps.BuildBackwardMaps<SphericalProjector>(Projector, srcTl, mapX, mapY);
}
