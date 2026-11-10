// This implementation is ported from the OpenCV stitching blenders.
//
// Original sources:
//   modules/stitching/src/blenders.cpp
//   modules/stitching/include/opencv2/stitching/detail/blenders.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>Weight map element type of <see cref="MultiBandBlender"/>.</summary>
public enum BlendWeightType
{
    /// <summary><c>CV_32F</c> weights in [0, 1].</summary>
    Float = 5,

    /// <summary><c>CV_16S</c> fixed-point weights in [0, 256].</summary>
    Short = 3,
}

/// <summary>
/// Simple blender that copies the masked pixels of each image into the panorama, as OpenCV
/// <c>detail::Blender</c> (<c>Blender::NO</c>).
/// </summary>
/// <remarks>
/// <para><b>Images.</b> The panorama accumulates 3-channel <c>short</c> pixels (<c>CV_16SC3</c>).
/// Masks are 1-channel <c>byte</c> images of the image size.</para>
/// <para><b>Sequence.</b> Call <c>Prepare</c>, then <c>Feed</c> for each image, then <c>Blend</c>.
/// <c>Blend</c> releases the panorama; call <c>Prepare</c> again before the next sequence.
/// Later images overwrite earlier images where their masks are set.</para>
/// </remarks>
public class Blender
{
    private Mat<short>? _dst;
    private Mat<byte>? _dstMask;

    /// <summary>Gets the panorama region set by <c>Prepare</c>.</summary>
    public Rect DstRoi { get; private protected set; }

    private protected Mat<short> Dst => _dst ?? throw new InvalidOperationException("Prepare must be called first.");

    private protected Mat<byte> DstMask => _dstMask ?? throw new InvalidOperationException("Prepare must be called first.");

    /// <summary>Prepares the panorama for images placed at <paramref name="corners"/>, as OpenCV <c>prepare(corners, sizes)</c>.</summary>
    /// <param name="corners">Top-left corners of the images.</param>
    /// <param name="sizes">Image sizes.</param>
    public void Prepare(IReadOnlyList<Point> corners, IReadOnlyList<Size> sizes)
        => Prepare(StitchingUtils.ResultRoi(corners, sizes));

    /// <summary>Prepares a zero panorama of the region <paramref name="dstRoi"/>, as OpenCV <c>prepare(dst_roi)</c>.</summary>
    /// <param name="dstRoi">Panorama region.</param>
    public virtual void Prepare(Rect dstRoi) => PrepareCore(dstRoi, true);

    /// <summary>Adds an image to the panorama, as OpenCV <c>feed</c>.</summary>
    /// <param name="image">3-channel image. It is not modified.</param>
    /// <param name="mask">1-channel mask of the image size.</param>
    /// <param name="tl">Top-left corner of the image in panorama coordinates. The image must lie inside the panorama region.</param>
    public virtual void Feed(Mat<short> image, Mat<byte> mask, Point tl)
    {
        ValidateFeed(image, mask, tl);
        BlendKernels.FeedNo(image, mask, Dst, DstMask, tl.X - DstRoi.X, tl.Y - DstRoi.Y);
    }

    /// <summary>Returns the panorama and its mask, as OpenCV <c>blend</c>. Pixels outside the mask are zero.</summary>
    /// <param name="dst">Receives the 3-channel <c>short</c> panorama.</param>
    /// <param name="dstMask">Receives the panorama mask (0 or 255).</param>
    public virtual void Blend(out Mat<short> dst, out Mat<byte> dstMask)
    {
        dst = Dst;
        dstMask = DstMask;
        BlendKernels.ZeroOutsideMask(dst, dstMask);
        _dst = null;
        _dstMask = null;
    }

    /// <summary>Returns <paramref name="spare"/> cleared when its shape matches, otherwise a new zero matrix.</summary>
    private protected static Mat<TElem> ReuseOrCreate<TElem>(Mat<TElem>? spare, int width, int height, int channels)
        where TElem : unmanaged
    {
        if (spare is not null && spare.Width == width && spare.Height == height && spare.Channels == channels)
        {
            spare.Clear();
            return spare;
        }

        return new Mat<TElem>(width, height, channels);
    }

    /// <summary>Allocates the zero panorama; <paramref name="withMask"/> false leaves the mask to <see cref="SetResult"/>.</summary>
    private protected void PrepareCore(Rect dstRoi, bool withMask)
    {
        if (dstRoi.Width <= 0 || dstRoi.Height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dstRoi), "The panorama region must not be empty.");
        }

        _dst = new Mat<short>(dstRoi.Width, dstRoi.Height, 3);
        _dstMask = withMask ? new Mat<byte>(dstRoi.Width, dstRoi.Height, 1) : null;
        DstRoi = dstRoi;
    }

    private protected void SetResult(Mat<short> dst, Mat<byte> dstMask)
    {
        _dst = dst;
        _dstMask = dstMask;
    }

    private protected void ValidateFeed<T>(Mat<T> image, Mat<byte> mask, Point tl)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(mask);
        _ = Dst;
        if (image.Channels != 3)
        {
            throw new ArgumentException("The image must have 3 channels.", nameof(image));
        }

        if (mask.Channels != 1 || mask.Width != image.Width || mask.Height != image.Height)
        {
            throw new ArgumentException("The mask must be 1-channel with the image size.", nameof(mask));
        }

        if (tl.X < DstRoi.X || tl.Y < DstRoi.Y || tl.X + image.Width > DstRoi.X + DstRoi.Width || tl.Y + image.Height > DstRoi.Y + DstRoi.Height)
        {
            throw new ArgumentOutOfRangeException(nameof(tl), "The image must lie inside the panorama region.");
        }
    }

    /// <summary>
    /// Creates a feather weight map, as OpenCV <c>detail::createWeightMap</c>:
    /// <c>min(1, sharpness · distanceTransform(mask, L1, 3))</c>.
    /// </summary>
    /// <param name="mask">1-channel mask.</param>
    /// <param name="sharpness">Weight slope per pixel of distance to the mask border.</param>
    /// <returns>The weight map.</returns>
    public static Mat<float> CreateWeightMap(Mat<byte> mask, float sharpness)
    {
        ArgumentNullException.ThrowIfNull(mask);
        var dist = new Mat<float>(mask.Width, mask.Height, 1, clear: false);
        CreateWeightMap(mask, sharpness, dist);
        return dist;
    }

    /// <summary><see cref="CreateWeightMap(Mat{byte}, float)"/> into <paramref name="weight"/> of the mask size.</summary>
    private protected static void CreateWeightMap(Mat<byte> mask, float sharpness, Mat<float> weight)
    {
        DistanceTransforms.DistanceTransform(mask, weight, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        BlendKernels.ScaleAndTruncate(weight, sharpness);
    }

    /// <summary>
    /// Divides each pixel by its weight, in place, as OpenCV <c>detail::normalizeUsingWeightMap</c>:
    /// <c>(short)(v / (w + 1e-5f))</c>.
    /// </summary>
    /// <param name="weight">1-channel weight map of the image size.</param>
    /// <param name="image">3-channel image. It is modified.</param>
    public static void NormalizeUsingWeightMap(Mat<float> weight, Mat<short> image)
    {
        ValidateWeight(weight, image);
        BlendKernels.NormalizeFloat(weight, image);
    }

    /// <summary>
    /// Divides each pixel by its fixed-point weight, in place, as OpenCV
    /// <c>detail::normalizeUsingWeightMap</c> with a <c>CV_16S</c> map: <c>(v · 256) / (w + 1)</c>.
    /// </summary>
    /// <param name="weight">1-channel weight map of the image size.</param>
    /// <param name="image">3-channel image. It is modified.</param>
    public static void NormalizeUsingWeightMap(Mat<short> weight, Mat<short> image)
    {
        ValidateWeight(weight, image);
        BlendKernels.NormalizeShort(weight, image);
    }

    /// <summary>Builds a Laplacian pyramid of a 3-channel image, as OpenCV <c>detail::createLaplacePyr</c>.</summary>
    /// <param name="image">3-channel <c>byte</c> image. It is not modified.</param>
    /// <param name="levels">Number of band-pass levels.</param>
    /// <returns><paramref name="levels"/> + 1 <c>short</c> levels; the last level is the low-pass image.</returns>
    public static Mat<short>[] CreateLaplacePyr(Mat<byte> image, int levels)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegative(levels);
        return BlendPyramids.LaplaceFromBytes(image, levels);
    }

    /// <summary>Builds a Laplacian pyramid of a <c>short</c> image, as OpenCV <c>detail::createLaplacePyr</c>.</summary>
    /// <param name="image"><c>short</c> image. It is not modified.</param>
    /// <param name="levels">Number of band-pass levels.</param>
    /// <returns><paramref name="levels"/> + 1 levels; the last level is the low-pass image.</returns>
    public static Mat<short>[] CreateLaplacePyr(Mat<short> image, int levels)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentOutOfRangeException.ThrowIfNegative(levels);
        return BlendPyramids.LaplaceFromShorts(image.Clone(), levels);
    }

    /// <summary>
    /// Collapses a Laplacian pyramid in place, as OpenCV <c>detail::restoreImageFromLaplacePyr</c>.
    /// Level 0 receives the image.
    /// </summary>
    /// <param name="pyramid">Pyramid levels. Levels 0 to n − 1 are modified.</param>
    public static void RestoreImageFromLaplacePyr(Mat<short>[] pyramid)
    {
        ArgumentNullException.ThrowIfNull(pyramid);
        BlendPyramids.Restore(pyramid);
    }

    private static void ValidateWeight<TW>(Mat<TW> weight, Mat<short> image)
        where TW : unmanaged
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels != 3)
        {
            throw new ArgumentException("The image must have 3 channels.", nameof(image));
        }

        if (weight.Channels != 1 || weight.Width != image.Width || weight.Height != image.Height)
        {
            throw new ArgumentException("The weight map must be 1-channel with the image size.", nameof(weight));
        }
    }
}

/// <summary>
/// Blender that mixes images with weights that rise linearly from the mask border, as OpenCV
/// <c>detail::FeatherBlender</c>.
/// </summary>
/// <remarks>
/// Each image adds <c>(short)(v · w)</c> to the panorama and <c>w</c> to the weight sum, with
/// <c>w = min(1, sharpness · L1 distance to the mask border)</c>. <c>Blend</c> divides by
/// <c>sum + 1e-5f</c> and sets the mask where the sum is above <c>1e-5f</c>. The results are
/// bit-exact with OpenCV.
/// </remarks>
public class FeatherBlender : Blender
{
    /// <summary>C++ <c>std::numeric_limits&lt;float&gt;::epsilon()</c>.</summary>
    private const float FltEpsilon = 1.1920929e-07f;

    private Mat<float>? _dstWeightMap;
    private Mat<float>? _spareWeightMap;

    /// <summary>Creates a feather blender.</summary>
    /// <param name="sharpness">Weight slope per pixel of distance to the mask border.</param>
    public FeatherBlender(float sharpness = 0.02f)
    {
        Sharpness = sharpness;
    }

    /// <summary>Gets or sets the weight slope.</summary>
    public float Sharpness { get; set; }

    /// <inheritdoc/>
    public override void Prepare(Rect dstRoi)
    {
        PrepareCore(dstRoi, false);
        _dstWeightMap = ReuseOrCreate(_spareWeightMap, dstRoi.Width, dstRoi.Height, 1);
        _spareWeightMap = null;
    }

    /// <inheritdoc/>
    public override void Feed(Mat<short> image, Mat<byte> mask, Point tl)
    {
        ValidateFeed(image, mask, tl);
        using PooledMat<float> weight = PooledMat<float>.Rent(mask.Width, mask.Height, 1);
        CreateWeightMap(mask, Sharpness, weight.Mat);
        var roi = new Rect(tl.X - DstRoi.X, tl.Y - DstRoi.Y, image.Width, image.Height);
        BlendKernels.AccumulateFloat(image, weight.Mat, Dst.Roi(roi), _dstWeightMap!.Roi(roi));
    }

    /// <inheritdoc/>
    public override void Blend(out Mat<short> dst, out Mat<byte> dstMask)
    {
        Mat<float> weights = _dstWeightMap ?? throw new InvalidOperationException("Prepare must be called first.");
        BlendKernels.NormalizeFloat(weights, Dst);
        SetResult(Dst, BlendKernels.WeightMask(weights));
        _spareWeightMap = weights;
        _dstWeightMap = null;
        base.Blend(out dst, out dstMask);
    }

    /// <summary>
    /// Creates normalised weight maps for a set of masks, as OpenCV
    /// <c>FeatherBlender::createWeightMaps</c>: each map divided by the sum of the overlapping maps
    /// (a sum below <c>FLT_EPSILON</c> counts as 1).
    /// </summary>
    /// <param name="masks">Image masks.</param>
    /// <param name="corners">Top-left corners, one per mask.</param>
    /// <param name="weightMaps">Receives the weight maps.</param>
    /// <returns>The panorama region.</returns>
    public Rect CreateWeightMaps(IReadOnlyList<Mat<byte>> masks, IReadOnlyList<Point> corners, out Mat<float>[] weightMaps)
    {
        ArgumentNullException.ThrowIfNull(masks);
        ArgumentNullException.ThrowIfNull(corners);
        weightMaps = new Mat<float>[masks.Count];
        var sizes = new Size[masks.Count];
        for (int i = 0; i < masks.Count; i++)
        {
            weightMaps[i] = CreateWeightMap(masks[i], Sharpness);
            sizes[i] = new Size(masks[i].Width, masks[i].Height);
        }

        Rect dstRoi = StitchingUtils.ResultRoi(corners, sizes);
        var sum = new Mat<float>(dstRoi.Width, dstRoi.Height, 1);
        for (int i = 0; i < weightMaps.Length; i++)
        {
            Mat<float> s = sum.Roi(corners[i].X - dstRoi.X, corners[i].Y - dstRoi.Y, sizes[i].Width, sizes[i].Height);
            Arithmetic.Add(s, weightMaps[i], s);
        }

        for (int i = 0; i < weightMaps.Length; i++)
        {
            Mat<float> s = sum.Roi(corners[i].X - dstRoi.X, corners[i].Y - dstRoi.Y, sizes[i].Width, sizes[i].Height);
            for (int y = 0; y < s.Height; y++)
            {
                Span<float> row = s.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    if (row[x] < FltEpsilon)
                    {
                        row[x] = 1f;
                    }
                }
            }

            Arithmetic.Divide(weightMaps[i], s, weightMaps[i]);
        }

        return dstRoi;
    }
}

/// <summary>
/// Blender that mixes the Laplacian pyramid bands of the images with Gaussian-pyramid weights,
/// as OpenCV <c>detail::MultiBandBlender</c> (CPU path).
/// </summary>
/// <remarks>
/// <para><b>Bands.</b> <c>Prepare</c> limits the band count to <c>ceil(log2(max(width, height)))</c>
/// and pads the panorama to a multiple of <c>2^bands</c>. Each image is padded by a reflected border
/// of <c>3 · 2^bands</c> pixels, aligned to the band grid.</para>
/// <para><b>Images.</b> <c>Feed</c> accepts 3-channel <c>byte</c> or <c>short</c> images. The result
/// is a 3-channel <c>short</c> panorama of the <c>Prepare</c> region.</para>
/// <para><b>Exactness.</b> The pyramids use <see cref="Pyramids"/>; the accumulation and
/// normalisation loops follow OpenCV. The results are bit-exact with OpenCV.</para>
/// </remarks>
public class MultiBandBlender : Blender
{
    private readonly BlendWeightType _weightType;
    private int _numBands;
    private Rect _dstRoiFinal;
    private Mat<short>[]? _dstPyrLaplace;
    private Mat<float>[]? _bandWeightsF;
    private Mat<short>[]? _bandWeightsS;
    private Mat<short>?[]? _spareLevels;
    private Mat<float>[]? _spareWeightsF;
    private Mat<short>[]? _spareWeightsS;

    /// <summary>Creates a multi-band blender.</summary>
    /// <param name="numBands">Requested number of bands.</param>
    /// <param name="weightType">Weight element type.</param>
    public MultiBandBlender(int numBands = 5, BlendWeightType weightType = BlendWeightType.Float)
    {
        if (weightType is not (BlendWeightType.Float or BlendWeightType.Short))
        {
            throw new ArgumentOutOfRangeException(nameof(weightType));
        }

        NumBands = numBands;
        _weightType = weightType;
    }

    /// <summary>Gets or sets the requested number of bands.</summary>
    public int NumBands { get; set; }

    /// <summary>Gets the weight element type.</summary>
    public BlendWeightType WeightType => _weightType;

    /// <inheritdoc/>
    public override void Prepare(Rect dstRoi)
    {
        _dstRoiFinal = dstRoi;
        double maxLen = Math.Max(dstRoi.Width, dstRoi.Height);
        _numBands = Math.Min(NumBands, (int)Math.Ceiling(Math.Log(maxLen) / Math.Log(2.0)));
        int unit = 1 << _numBands;
        var padded = new Rect(dstRoi.X, dstRoi.Y, dstRoi.Width + ((unit - (dstRoi.Width % unit)) % unit), dstRoi.Height + ((unit - (dstRoi.Height % unit)) % unit));
        PrepareCore(padded, false);

        _dstPyrLaplace = new Mat<short>[_numBands + 1];
        _dstPyrLaplace[0] = Dst;
        _bandWeightsF = _weightType == BlendWeightType.Float ? new Mat<float>[_numBands + 1] : null;
        _bandWeightsS = _weightType == BlendWeightType.Short ? new Mat<short>[_numBands + 1] : null;
        int w = padded.Width, h = padded.Height;
        for (int i = 0; i <= _numBands; i++)
        {
            if (i > 0)
            {
                w = (w + 1) / 2;
                h = (h + 1) / 2;
                _dstPyrLaplace[i] = ReuseOrCreate(Spare(_spareLevels, i), w, h, 3);
            }

            if (_bandWeightsF is not null)
            {
                _bandWeightsF[i] = ReuseOrCreate(Spare(_spareWeightsF, i), w, h, 1);
            }
            else
            {
                _bandWeightsS![i] = ReuseOrCreate(Spare(_spareWeightsS, i), w, h, 1);
            }
        }

        _spareLevels = null;
        _spareWeightsF = null;
        _spareWeightsS = null;
    }

    private static Mat<TElem>? Spare<TElem>(Mat<TElem>?[]? spares, int i)
        where TElem : unmanaged
        => spares is not null && i < spares.Length ? spares[i] : null;

    /// <inheritdoc/>
    public override void Feed(Mat<short> image, Mat<byte> mask, Point tl) => FeedCore(image, mask, tl);

    /// <summary>Adds a 3-channel <c>byte</c> image to the panorama, as OpenCV <c>feed</c> with <c>CV_8UC3</c>.</summary>
    /// <param name="image">3-channel image. It is not modified.</param>
    /// <param name="mask">1-channel mask of the image size.</param>
    /// <param name="tl">Top-left corner of the image in panorama coordinates.</param>
    public void Feed(Mat<byte> image, Mat<byte> mask, Point tl) => FeedCore(image, mask, tl);

    private void FeedCore<T>(Mat<T> image, Mat<byte> mask, Point tl)
        where T : unmanaged
    {
        ValidateFeed(image, mask, tl);
        Rect roi = DstRoi;
        int unit = 1 << _numBands;
        int gap = 3 * unit;
        int tlx = Math.Max(roi.X, tl.X - gap), tly = Math.Max(roi.Y, tl.Y - gap);
        int brx = Math.Min(roi.X + roi.Width, tl.X + image.Width + gap), bry = Math.Min(roi.Y + roi.Height, tl.Y + image.Height + gap);

        tlx = roi.X + (((tlx - roi.X) >> _numBands) << _numBands);
        tly = roi.Y + (((tly - roi.Y) >> _numBands) << _numBands);
        int width = brx - tlx, height = bry - tly;
        width += (unit - (width % unit)) % unit;
        height += (unit - (height % unit)) % unit;
        brx = tlx + width;
        bry = tly + height;
        int dy = Math.Max(bry - (roi.Y + roi.Height), 0);
        int dx = Math.Max(brx - (roi.X + roi.Width), 0);
        tlx -= dx; brx -= dx;
        tly -= dy; bry -= dy;

        int top = tl.Y - tly, left = tl.X - tlx;
        int bottom = bry - tl.Y - image.Height, right = brx - tl.X - image.Width;

        var pool = new List<IDisposable>();
        try
        {
            PooledMat<T> bordered = Borders.CopyMakeBorderPooled(image, top, bottom, left, right, BorderType.Reflect);
            pool.Add(bordered);
            Mat<short>[] srcPyr = typeof(T) == typeof(byte)
                ? BlendPyramids.LaplaceFromBytes((Mat<byte>)(object)bordered.Mat, _numBands, pool)
                : BlendPyramids.LaplaceFromShorts((Mat<short>)(object)bordered.Mat, _numBands, pool);

            int xTl = tlx - roi.X, yTl = tly - roi.Y, xBr = brx - roi.X, yBr = bry - roi.Y;
            if (_bandWeightsF is not null)
            {
                PooledMat<float> weight = PooledMat<float>.Rent(mask.Width, mask.Height, 1);
                pool.Add(weight);
                Elementwise.Unary(mask, weight.Mat, null, new ConvertScaleFloatOp<byte, float>((float)(1.0 / 255.0), 0f));
                PooledMat<float> weightBordered = Borders.CopyMakeBorderPooled(weight.Mat, top, bottom, left, right, BorderType.Constant);
                pool.Add(weightBordered);
                Mat<float> level = weightBordered.Mat;
                for (int i = 0; i <= _numBands; i++)
                {
                    if (i > 0)
                    {
                        level = BlendPyramids.Down(level, pool);
                    }

                    var rc = new Rect(xTl, yTl, xBr - xTl, yBr - yTl);
                    BlendKernels.AccumulateFloat(srcPyr[i], level, _dstPyrLaplace![i].Roi(rc), _bandWeightsF[i].Roi(rc));
                    xTl /= 2; yTl /= 2; xBr /= 2; yBr /= 2;
                }
            }
            else
            {
                PooledMat<short> weightPooled = PooledMat<short>.Rent(mask.Width, mask.Height, 1);
                pool.Add(weightPooled);
                Mat<short> weight = weightPooled.Mat;
                for (int y = 0; y < mask.Height; y++)
                {
                    ReadOnlySpan<byte> m = mask.GetReadOnlyRowSpan(y);
                    Span<short> w = weight.GetRowSpan(y);
                    for (int x = 0; x < m.Length; x++)
                    {
                        w[x] = (short)(m[x] + (m[x] != 0 ? 1 : 0));
                    }
                }

                PooledMat<short> weightBordered = Borders.CopyMakeBorderPooled(weight, top, bottom, left, right, BorderType.Constant);
                pool.Add(weightBordered);
                Mat<short> level = weightBordered.Mat;
                for (int i = 0; i <= _numBands; i++)
                {
                    if (i > 0)
                    {
                        level = BlendPyramids.Down(level, pool);
                    }

                    var rc = new Rect(xTl, yTl, xBr - xTl, yBr - yTl);
                    BlendKernels.AccumulateShort(srcPyr[i], level, _dstPyrLaplace![i].Roi(rc), _bandWeightsS![i].Roi(rc));
                    xTl /= 2; yTl /= 2; xBr /= 2; yBr /= 2;
                }
            }
        }
        finally
        {
            foreach (IDisposable pooled in pool)
            {
                pooled.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public override void Blend(out Mat<short> dst, out Mat<byte> dstMask)
    {
        Mat<short>[] pyr = _dstPyrLaplace ?? throw new InvalidOperationException("Prepare must be called first.");
        for (int i = 0; i <= _numBands; i++)
        {
            if (_bandWeightsF is not null)
            {
                BlendKernels.NormalizeFloat(_bandWeightsF[i], pyr[i]);
            }
            else
            {
                BlendKernels.NormalizeShort(_bandWeightsS![i], pyr[i]);
            }
        }

        BlendPyramids.Restore(pyr);
        var rc = new Rect(0, 0, _dstRoiFinal.Width, _dstRoiFinal.Height);
        Mat<byte> mask = _bandWeightsF is not null
            ? BlendKernels.WeightMask(_bandWeightsF[0].Roi(rc))
            : BlendKernels.WeightMask(_bandWeightsS![0].Roi(rc));
        bool whole = rc.Width == pyr[0].Width && rc.Height == pyr[0].Height;
        SetResult(whole ? pyr[0] : pyr[0].Roi(rc).Clone(), mask);
        _spareLevels = pyr;
        _spareLevels[0] = null;
        _spareWeightsF = _bandWeightsF;
        _spareWeightsS = _bandWeightsS;
        _dstPyrLaplace = null;
        _bandWeightsF = null;
        _bandWeightsS = null;
        base.Blend(out dst, out dstMask);
    }
}
