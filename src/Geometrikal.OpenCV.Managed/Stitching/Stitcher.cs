// This implementation is ported from the OpenCV 5.0 high-level stitcher.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching.hpp (Stitcher)
//   modules/stitching/src/stitcher.cpp (create, estimateTransform, composePanorama, stitch,
//   matchImages, estimateCameraParams, setTransform)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>Stitching scenario, as OpenCV <c>Stitcher::Mode</c>.</summary>
public enum StitcherMode
{
    /// <summary>
    /// Photo panorama from a rotating camera (<c>PANORAMA</c>): homography matcher and estimator,
    /// ray bundle adjustment, horizontal wave correction, spherical warper, block gain compensation.
    /// </summary>
    Panorama = 0,

    /// <summary>
    /// Scans of a plane under affine motion (<c>SCANS</c>), for example a downward-looking camera:
    /// affine (similarity) matcher and estimator, partial affine bundle adjustment, affine warper,
    /// no exposure compensation.
    /// </summary>
    Scans = 1,
}

/// <summary>Result status, as OpenCV <c>Stitcher::Status</c>.</summary>
public enum StitcherStatus
{
    /// <summary>Success (<c>OK</c>).</summary>
    Ok = 0,

    /// <summary>Fewer than two images, or fewer than two connected images (<c>ERR_NEED_MORE_IMGS</c>).</summary>
    ErrNeedMoreImgs = 1,

    /// <summary>The estimator failed (<c>ERR_HOMOGRAPHY_EST_FAIL</c>).</summary>
    ErrHomographyEstFail = 2,

    /// <summary>The bundle adjustment failed (<c>ERR_CAMERA_PARAMS_ADJUST_FAIL</c>).</summary>
    ErrCameraParamsAdjustFail = 3,
}

/// <summary>
/// High-level image stitcher, as OpenCV <c>cv::Stitcher</c>: features → pairwise matching →
/// largest connected component → camera estimation → bundle adjustment → wave correction →
/// warping → exposure compensation → seam finding → blending.
/// </summary>
/// <remarks>
/// <para>
/// Registration runs at <see cref="RegistrationResol"/> megapixels, seam estimation at
/// <see cref="SeamEstimationResol"/> megapixels and compositing at
/// <see cref="CompositingResol"/> megapixels (a negative value keeps the original size). The
/// scales are computed from the first image. Images are 8-bit with 1 or 3 channels; the
/// components decide the supported channel counts.
/// </para>
/// <para>
/// The stitcher keeps state between <see cref="EstimateTransform"/> and
/// <see cref="ComposePanorama(out Mat{byte}?)"/>. An instance is not thread-safe. The matcher
/// uses exhaustive matching instead of FLANN; see <see cref="BestOf2NearestMatcher"/>.
/// </para>
/// </remarks>
public sealed class Stitcher
{
    /// <summary>Value of the resolution settings that keeps the original resolution (OpenCV <c>ORIG_RESOL</c>).</summary>
    public const double OrigResol = -1.0;

    private List<Mat<byte>> _imgs = [];
    private List<Mat<byte>?> _masks = [];
    private List<Size> _fullImgSizes = [];
    private ImageFeatures[] _features = [];
    private MatchesInfo[] _pairwiseMatches = [];
    private List<Mat<byte>> _seamEstImgs = [];
    private int[] _indices = [];
    private List<CameraParams> _cameras = [];
    private double _workScale = 1;
    private double _seamScale = 1;
    private double _seamWorkAspect = 1;
    private double _warpedImageScale = 1;

    /// <summary>Gets or sets the registration resolution in megapixels. The default is 0.6.</summary>
    public double RegistrationResol { get; set; } = 0.6;

    /// <summary>Gets or sets the seam estimation resolution in megapixels. The default is 0.1.</summary>
    public double SeamEstimationResol { get; set; } = 0.1;

    /// <summary>Gets or sets the compositing resolution in megapixels. The default is <see cref="OrigResol"/>.</summary>
    public double CompositingResol { get; set; } = OrigResol;

    /// <summary>Gets or sets the minimum pair confidence for the connected component and the bundle adjustment. The default is 1.</summary>
    public double PanoConfidenceThresh { get; set; } = 1;

    /// <summary>Gets or sets a value indicating whether wave correction runs.</summary>
    public bool WaveCorrection { get; set; }

    /// <summary>Gets or sets the wave correction kind.</summary>
    public WaveCorrectKind WaveCorrectKind { get; set; } = WaveCorrectKind.Horizontal;

    /// <summary>Gets or sets the interpolation of the image warps. The default is <see cref="InterpolationFlags.Linear"/>.</summary>
    public InterpolationFlags InterpolationFlags { get; set; } = InterpolationFlags.Linear;

    /// <summary>Gets or sets the features finder. The default is ORB with the default settings.</summary>
    public IFeaturesFinder FeaturesFinder { get; set; } = new OrbFeaturesFinder();

    /// <summary>Gets or sets the pairwise matcher.</summary>
    public FeaturesMatcher FeaturesMatcher { get; set; } = new BestOf2NearestMatcher();

    /// <summary>Gets or sets the optional n × n matching mask (see <see cref="FeaturesMatcher.Match(IReadOnlyList{ImageFeatures}, Mat{byte}?)"/>).</summary>
    public Mat<byte>? MatchingMask { get; set; }

    /// <summary>Gets or sets the camera estimator.</summary>
    public Estimator Estimator { get; set; } = new HomographyBasedEstimator();

    /// <summary>Gets or sets the bundle adjuster. Its <see cref="BundleAdjusterBase.ConfThresh"/> is set to <see cref="PanoConfidenceThresh"/>.</summary>
    public BundleAdjusterBase BundleAdjuster { get; set; } = new BundleAdjusterRay();

    /// <summary>Gets or sets the warper creator.</summary>
    public WarperCreator Warper { get; set; } = new SphericalWarperCreator();

    /// <summary>Gets or sets the exposure compensator.</summary>
    public ExposureCompensator ExposureCompensator { get; set; } = new BlocksGainCompensator();

    /// <summary>Gets or sets the seam finder.</summary>
    public SeamFinder SeamFinder { get; set; } = new GraphCutSeamFinder(GraphCutCostType.Color);

    /// <summary>Gets or sets the blender.</summary>
    public Blender Blender { get; set; } = new MultiBandBlender();

    /// <summary>Gets the indices of the input images used in the panorama (OpenCV <c>component</c>).</summary>
    public IReadOnlyList<int> Component => _indices;

    /// <summary>Gets the estimated cameras of the images in <see cref="Component"/> (OpenCV <c>cameras</c>).</summary>
    public IReadOnlyList<CameraParams> Cameras => _cameras;

    /// <summary>Gets the registration scale (OpenCV <c>workScale</c>).</summary>
    public double WorkScale => _workScale;

    /// <summary>Gets the panorama mask of the last composition (OpenCV <c>resultMask</c>).</summary>
    public Mat<byte>? ResultMask { get; private set; }

    /// <summary>
    /// Creates a stitcher with the OpenCV settings of a mode, as OpenCV <c>Stitcher::create</c>.
    /// Both modes use registration 0.6 MP, seam estimation 0.1 MP, the original compositing
    /// resolution, confidence threshold 1, the colour graph-cut seam finder, the multi-band
    /// blender, ORB features and linear interpolation.
    /// </summary>
    /// <param name="mode">Stitching scenario.</param>
    /// <returns>A new stitcher.</returns>
    public static Stitcher Create(StitcherMode mode = StitcherMode.Panorama)
    {
        var s = new Stitcher();
        switch (mode)
        {
            case StitcherMode.Panorama:
                s.Estimator = new HomographyBasedEstimator();
                s.WaveCorrection = true;
                s.WaveCorrectKind = WaveCorrectKind.Horizontal;
                s.FeaturesMatcher = new BestOf2NearestMatcher();
                s.BundleAdjuster = new BundleAdjusterRay();
                s.Warper = new SphericalWarperCreator();
                s.ExposureCompensator = new BlocksGainCompensator();
                break;
            case StitcherMode.Scans:
                s.Estimator = new AffineBasedEstimator();
                s.WaveCorrection = false;
                s.FeaturesMatcher = new AffineBestOf2NearestMatcher(false);
                s.BundleAdjuster = new BundleAdjusterAffinePartial();
                s.Warper = new AffineWarperCreator();
                s.ExposureCompensator = new NoExposureCompensator();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode), "Invalid stitching mode.");
        }

        return s;
    }

    /// <summary>
    /// Finds features, matches the images and estimates the cameras, as OpenCV
    /// <c>Stitcher::estimateTransform</c>.
    /// </summary>
    /// <param name="images">Input images. They are not modified and are kept for <see cref="ComposePanorama(out Mat{byte}?)"/>.</param>
    /// <param name="masks">Optional masks for the feature detection, one per image.</param>
    /// <returns>The status.</returns>
    public StitcherStatus EstimateTransform(IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>?>? masks = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        _imgs = [.. images];
        _masks = masks is null ? [] : [.. masks];
        StitcherStatus status = MatchImages();
        return status != StitcherStatus.Ok ? status : EstimateCameraParams();
    }

    /// <summary>
    /// Sets the cameras directly instead of estimating them, as OpenCV <c>Stitcher::setTransform</c>.
    /// </summary>
    /// <param name="images">Input images.</param>
    /// <param name="cameras">One camera per image of <paramref name="component"/>, at the registration scale.</param>
    /// <param name="component">Indices of the images to use, or null for all images.</param>
    /// <returns>The status.</returns>
    public StitcherStatus SetTransform(IReadOnlyList<Mat<byte>> images, IReadOnlyList<CameraParams> cameras, IReadOnlyList<int>? component = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(cameras);
        _imgs = [.. images];
        _masks = [];
        if (_imgs.Count < 2)
        {
            return StitcherStatus.ErrNeedMoreImgs;
        }

        ComputeScales(resizeForFeatures: false, null);
        _features = [];
        _pairwiseMatches = [];
        _indices = component is null ? Enumerable.Range(0, _imgs.Count).ToArray() : [.. component];
        TakeSubset();
        if (_imgs.Count < 2)
        {
            return StitcherStatus.ErrNeedMoreImgs;
        }

        _cameras = cameras.Select(c => c.Clone()).ToList();
        _warpedImageScale = MedianFocal(_cameras);
        return StitcherStatus.Ok;
    }

    /// <summary>
    /// Composes the panorama of the images given to <see cref="EstimateTransform"/>, as OpenCV
    /// <c>Stitcher::composePanorama(pano)</c>.
    /// </summary>
    /// <param name="pano">The 8-bit panorama.</param>
    /// <returns>The status.</returns>
    public StitcherStatus ComposePanorama(out Mat<byte>? pano) => ComposePanorama(null, out pano);

    /// <summary>
    /// Composes the panorama of new images with the estimated cameras, as OpenCV
    /// <c>Stitcher::composePanorama(images, pano)</c>.
    /// </summary>
    /// <param name="images">Replacement images, one per input image of <see cref="EstimateTransform"/>, or null to keep the input images.</param>
    /// <param name="pano">The 8-bit panorama.</param>
    /// <returns>The status.</returns>
    public StitcherStatus ComposePanorama(IReadOnlyList<Mat<byte>>? images, out Mat<byte>? pano)
    {
        if (images is not null && images.Count > 0)
        {
            if (images.Count != _fullCount)
            {
                throw new ArgumentException("The number of images differs from the estimated set.", nameof(images));
            }

            var seam = new List<Mat<byte>>();
            var imgs = new List<Mat<byte>>();
            foreach (int idx in _indices)
            {
                imgs.Add(images[idx]);
                seam.Add(Geometry.Resize(images[idx], default, _seamScale, _seamScale, InterpolationFlags.LinearExact));
            }

            _imgs = imgs;
            _seamEstImgs = seam;
        }

        int n = _imgs.Count;
        var corners = new Point[n];
        var masksWarped = new Mat<byte>[n];
        var imagesWarped = new Mat<byte>[n];
        var sizes = new Size[n];

        RotationWarper w = Warper.Create((float)(_warpedImageScale * _seamWorkAspect));
        for (int i = 0; i < n; i++)
        {
            Mat<byte> seamImg = _seamEstImgs[i];
            var mask = new Mat<byte>(seamImg.Width, seamImg.Height);
            mask.Fill(255);
            Mat<float> k = SmallMat.ToFloat(_cameras[i].K());
            float a = (float)_seamWorkAspect;
            k.At(0, 0) *= a;
            k.At(0, 2) *= a;
            k.At(1, 1) *= a;
            k.At(1, 2) *= a;
            Mat<float> r = SmallMat.ToFloat(_cameras[i].R);
            corners[i] = w.Warp(seamImg, k, r, InterpolationFlags, BorderType.Reflect, out imagesWarped[i]);
            sizes[i] = new Size(imagesWarped[i].Width, imagesWarped[i].Height);
            w.Warp(mask, k, r, InterpolationFlags.Nearest, BorderType.Constant, out masksWarped[i]);
        }

        ExposureCompensator.Feed(corners, imagesWarped, masksWarped);
        for (int i = 0; i < n; i++)
        {
            ExposureCompensator.Apply(i, corners[i], imagesWarped[i], masksWarped[i]);
        }

        var imagesWarpedF = new Mat<float>[n];
        for (int i = 0; i < n; i++)
        {
            imagesWarpedF[i] = Conversions.ConvertTo<byte, float>(imagesWarped[i]);
        }

        SeamFinder.Find(imagesWarpedF, corners, masksWarped);

        bool blenderPrepared = false;
        double composeScale = 1;
        bool composeScaleSet = false;
        var camerasScaled = _cameras.Select(c => c.Clone()).ToList();
        for (int imgIdx = 0; imgIdx < n; imgIdx++)
        {
            Mat<byte> fullImg = _imgs[imgIdx];
            if (!composeScaleSet)
            {
                if (CompositingResol > 0)
                {
                    composeScale = Math.Min(1.0, Math.Sqrt(CompositingResol * 1e6 / ((double)fullImg.Width * fullImg.Height)));
                }

                composeScaleSet = true;
                double composeWorkAspect = composeScale / _workScale;
                w = Warper.Create((float)(_warpedImageScale * composeWorkAspect));
                for (int i = 0; i < n; i++)
                {
                    camerasScaled[i].Ppx *= composeWorkAspect;
                    camerasScaled[i].Ppy *= composeWorkAspect;
                    camerasScaled[i].Focal *= composeWorkAspect;
                    Size sz = _fullImgSizes[i];
                    if (Math.Abs(composeScale - 1) > 1e-1)
                    {
                        sz = new Size(
                            (int)Math.Round(_fullImgSizes[i].Width * composeScale, MidpointRounding.ToEven),
                            (int)Math.Round(_fullImgSizes[i].Height * composeScale, MidpointRounding.ToEven));
                    }

                    Rect roi = w.WarpRoi(sz, SmallMat.ToFloat(camerasScaled[i].K()), SmallMat.ToFloat(camerasScaled[i].R));
                    corners[i] = new Point(roi.X, roi.Y);
                    sizes[i] = new Size(roi.Width, roi.Height);
                }
            }

            Mat<byte> img = Math.Abs(composeScale - 1) > 1e-1
                ? Geometry.Resize(fullImg, default, composeScale, composeScale, InterpolationFlags.LinearExact)
                : fullImg;
            Mat<float> kc = SmallMat.ToFloat(camerasScaled[imgIdx].K());
            Mat<float> rc = SmallMat.ToFloat(_cameras[imgIdx].R);
            w.Warp(img, kc, rc, InterpolationFlags, BorderType.Reflect, out Mat<byte> imgWarped);
            var fullMask = new Mat<byte>(img.Width, img.Height);
            fullMask.Fill(255);
            w.Warp(fullMask, kc, rc, InterpolationFlags.Nearest, BorderType.Constant, out Mat<byte> maskWarped);

            ExposureCompensator.Apply(imgIdx, corners[imgIdx], imgWarped, maskWarped);
            Mat<short> imgWarpedS = Conversions.ConvertTo<byte, short>(imgWarped);

            Mat<byte> dilated = Morphology.Dilate(masksWarped[imgIdx]);
            Mat<byte> seamMask = Geometry.Resize(dilated, new Size(maskWarped.Width, maskWarped.Height), 0, 0, InterpolationFlags.LinearExact);
            Bitwise.And(seamMask, maskWarped, maskWarped);

            if (!blenderPrepared)
            {
                Blender.Prepare(corners, sizes);
                blenderPrepared = true;
            }

            Blender.Feed(imgWarpedS, maskWarped, corners[imgIdx]);
        }

        Blender.Blend(out Mat<short> result, out Mat<byte> resultMask);
        ResultMask = resultMask;
        pano = Conversions.ConvertTo<short, byte>(result);
        return StitcherStatus.Ok;
    }

    /// <summary>
    /// Estimates the cameras and composes the panorama, as OpenCV <c>Stitcher::stitch</c>.
    /// </summary>
    /// <param name="images">Input images. They are not modified.</param>
    /// <param name="pano">The 8-bit panorama, or null when the status is not <see cref="StitcherStatus.Ok"/>.</param>
    /// <returns>The status.</returns>
    public StitcherStatus Stitch(IReadOnlyList<Mat<byte>> images, out Mat<byte>? pano) => Stitch(images, null, out pano);

    /// <summary>
    /// Estimates the cameras with feature masks and composes the panorama, as OpenCV
    /// <c>Stitcher::stitch(images, masks, pano)</c>.
    /// </summary>
    /// <param name="images">Input images. They are not modified.</param>
    /// <param name="masks">Optional feature detection masks, one per image.</param>
    /// <param name="pano">The 8-bit panorama, or null when the status is not <see cref="StitcherStatus.Ok"/>.</param>
    /// <returns>The status.</returns>
    public StitcherStatus Stitch(IReadOnlyList<Mat<byte>> images, IReadOnlyList<Mat<byte>?>? masks, out Mat<byte>? pano)
    {
        StitcherStatus status = EstimateTransform(images, masks);
        if (status != StitcherStatus.Ok)
        {
            pano = null;
            return status;
        }

        return ComposePanorama(out pano);
    }

    private int _fullCount;

    private StitcherStatus MatchImages()
    {
        if (_imgs.Count < 2)
        {
            return StitcherStatus.ErrNeedMoreImgs;
        }

        var featureImgs = new List<Mat<byte>>(_imgs.Count);
        var featureMasks = new List<Mat<byte>?>(_masks.Count);
        ComputeScales(resizeForFeatures: true, featureImgs);
        for (int i = 0; i < _masks.Count; i++)
        {
            Mat<byte>? m = _masks[i];
            featureMasks.Add(m is null ? null : Geometry.Resize(m, default, _workScale, _workScale, InterpolationFlags.Nearest));
        }

        _features = FeaturesFinder.Compute(featureImgs, featureMasks.Count == 0 ? null : featureMasks);
        for (int i = 0; i < _features.Length; i++)
        {
            _features[i].ImgIdx = i;
        }

        _pairwiseMatches = FeaturesMatcher.Match(_features, MatchingMask);
        FeaturesMatcher.CollectGarbage();
        _indices = MatchesGraph.LeaveBiggestComponent(ref _features, ref _pairwiseMatches, (float)PanoConfidenceThresh);
        TakeSubset();
        return _imgs.Count < 2 ? StitcherStatus.ErrNeedMoreImgs : StitcherStatus.Ok;
    }

    private StitcherStatus EstimateCameraParams()
    {
        if (!Estimator.Estimate(_features, _pairwiseMatches, _cameras))
        {
            return StitcherStatus.ErrHomographyEstFail;
        }

        foreach (CameraParams c in _cameras)
        {
            c.R = SmallMat.RoundToFloat(c.R);
        }

        BundleAdjuster.ConfThresh = PanoConfidenceThresh;
        if (!BundleAdjuster.Estimate(_features, _pairwiseMatches, _cameras))
        {
            return StitcherStatus.ErrCameraParamsAdjustFail;
        }

        _warpedImageScale = MedianFocal(_cameras);
        if (WaveCorrection)
        {
            var rmats = _cameras.Select(c => c.R.Clone()).ToList();
            Stitching.WaveCorrection.WaveCorrect(rmats, WaveCorrectKind);
            for (int i = 0; i < _cameras.Count; i++)
            {
                _cameras[i].R = rmats[i];
            }
        }

        return StitcherStatus.Ok;
    }

    private void ComputeScales(bool resizeForFeatures, List<Mat<byte>>? featureImgs)
    {
        _fullCount = _imgs.Count;
        _workScale = 1;
        _seamWorkAspect = 1;
        _seamScale = 1;
        bool workScaleSet = false, seamScaleSet = false;
        _seamEstImgs = new List<Mat<byte>>(_imgs.Count);
        _fullImgSizes = new List<Size>(_imgs.Count);
        foreach (Mat<byte> img in _imgs)
        {
            var size = new Size(img.Width, img.Height);
            double area = (double)size.Width * size.Height;
            _fullImgSizes.Add(size);
            if (RegistrationResol < 0)
            {
                featureImgs?.Add(img);
                _workScale = 1;
                workScaleSet = true;
            }
            else
            {
                if (!workScaleSet)
                {
                    _workScale = Math.Min(1.0, Math.Sqrt(RegistrationResol * 1e6 / area));
                    workScaleSet = true;
                }

                if (resizeForFeatures)
                {
                    featureImgs!.Add(Geometry.Resize(img, default, _workScale, _workScale, InterpolationFlags.LinearExact));
                }
            }

            if (!seamScaleSet)
            {
                _seamScale = Math.Min(1.0, Math.Sqrt(SeamEstimationResol * 1e6 / area));
                _seamWorkAspect = _seamScale / _workScale;
                seamScaleSet = true;
            }

            _seamEstImgs.Add(Geometry.Resize(img, default, _seamScale, _seamScale, InterpolationFlags.LinearExact));
        }
    }

    private void TakeSubset()
    {
        var imgs = new List<Mat<byte>>(_indices.Length);
        var seam = new List<Mat<byte>>(_indices.Length);
        var sizes = new List<Size>(_indices.Length);
        foreach (int idx in _indices)
        {
            imgs.Add(_imgs[idx]);
            seam.Add(_seamEstImgs[idx]);
            sizes.Add(_fullImgSizes[idx]);
        }

        _imgs = imgs;
        _seamEstImgs = seam;
        _fullImgSizes = sizes;
    }

    private static double MedianFocal(List<CameraParams> cameras)
    {
        double[] focals = cameras.Select(c => c.Focal).ToArray();
        Array.Sort(focals);
        int m = focals.Length;
        return m % 2 == 1
            ? (float)focals[m / 2]
            : (float)(focals[(m / 2) - 1] + focals[m / 2]) * 0.5f;
    }
}
