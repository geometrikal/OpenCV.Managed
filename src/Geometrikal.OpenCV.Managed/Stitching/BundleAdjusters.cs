// This implementation is ported from the OpenCV 5.0 stitching bundle adjusters.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp
//   modules/stitching/src/motion_estimators.cpp (BundleAdjusterBase, BundleAdjusterReproj,
//   BundleAdjusterRay, BundleAdjusterAffine, BundleAdjusterAffinePartial, NoBundleAdjuster,
//   calcDeriv)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>
/// Base class of the bundle adjusters, as OpenCV <c>detail::BundleAdjusterBase</c>.
/// </summary>
/// <remarks>
/// <para>
/// The adjuster uses the pairs (i, j), i &lt; j, with a confidence above
/// <see cref="ConfThresh"/> and their inlier matches. It minimizes the residuals with the OpenCV
/// 5.0 <c>LevMarq</c> solver (geodesic acceleration, at most <see cref="TermCriteria"/>
/// <c>MaxCount</c> iterations, step norm tolerance <c>Epsilon</c>, small energy tolerance
/// <c>Epsilon²</c>). The Jacobian uses central differences. After the optimization every
/// rotation is multiplied by the inverse rotation of the spanning tree centre.
/// </para>
/// <para>
/// The adjusters read <see cref="CameraParams.R"/> as <c>float</c> values (OpenCV requires
/// <c>CV_32F</c>) and write <c>float</c> values.
/// </para>
/// </remarks>
public abstract class BundleAdjusterBase : Estimator
{
    private protected readonly Mat<double> TmpA = new(3, 3);
    private protected readonly Mat<double> TmpB = new(3, 3);
    private protected readonly Mat<double> TmpC = new(3, 3);
    private protected readonly Mat<double> TmpH1 = new(3, 3);
    private protected readonly Mat<double> TmpH2 = new(3, 3);
    private const double DblEpsilon = 2.220446049250313e-16;

    private int _numImages;
    private IReadOnlyList<ImageFeatures> _features = [];
    private IReadOnlyList<MatchesInfo> _pairwise = [];

    /// <summary>Initializes a new instance.</summary>
    /// <param name="numParamsPerCam">Number of parameters per camera.</param>
    /// <param name="numErrsPerMeasurement">Number of residuals per inlier match.</param>
    protected BundleAdjusterBase(int numParamsPerCam, int numErrsPerMeasurement)
    {
        NumParamsPerCam = numParamsPerCam;
        NumErrsPerMeasurement = numErrsPerMeasurement;
    }

    /// <summary>Gets the number of parameters per camera.</summary>
    public int NumParamsPerCam { get; }

    /// <summary>Gets the number of residuals per inlier match.</summary>
    public int NumErrsPerMeasurement { get; }

    /// <summary>
    /// Gets or sets the 3 × 3 refinement mask (OpenCV <c>refinement_mask_</c>). A zero element
    /// fixes the matching intrinsic parameter: (0,0) focal, (0,2) ppx, (1,2) ppy, (1,1) aspect.
    /// Only <see cref="BundleAdjusterReproj"/> uses it. The default is all ones.
    /// </summary>
    public byte[,] RefinementMask { get; set; } = new byte[,] { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };

    /// <summary>Gets or sets the minimum pair confidence (OpenCV <c>conf_thresh_</c>). The default is 1.</summary>
    public double ConfThresh { get; set; } = 1.0;

    /// <summary>Gets or sets the termination criteria. The default is (EPS + COUNT, 1000, DBL_EPSILON).</summary>
    public TermCriteria TermCriteria { get; set; } = new(TermCriteriaType.Eps | TermCriteriaType.MaxIter, 1000, DblEpsilon);

    /// <summary>Gets the camera parameter vector (OpenCV <c>cam_params_</c>).</summary>
    private protected double[] CamParams { get; set; } = [];

    /// <summary>Gets the pairs (i, j) used by the adjustment.</summary>
    private protected List<(int I, int J)> Edges { get; } = [];

    /// <summary>Gets the total number of inlier matches of <see cref="Edges"/>.</summary>
    private protected int TotalNumMatches { get; private set; }

    private protected int NumImages => _numImages;

    private protected IReadOnlyList<ImageFeatures> FeaturesList => _features;

    private protected MatchesInfo PairAt(int i, int j) => _pairwise[(i * _numImages) + j];

    /// <inheritdoc/>
    protected override bool EstimateCore(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches, List<CameraParams> cameras)
    {
        _numImages = features.Count;
        _features = features;
        _pairwise = pairwiseMatches;
        if (cameras.Count != _numImages)
        {
            throw new ArgumentException("There must be one camera per image.", nameof(cameras));
        }

        SetUpInitialCameraParams(cameras);
        Edges.Clear();
        for (int i = 0; i < _numImages - 1; i++)
        {
            for (int j = i + 1; j < _numImages; j++)
            {
                if (PairAt(i, j).Confidence > ConfThresh)
                {
                    Edges.Add((i, j));
                }
            }
        }

        TotalNumMatches = 0;
        foreach ((int i, int j) in Edges)
        {
            TotalNumMatches += PairAt(i, j).NumInliers;
        }

        int nerrs = TotalNumMatches * NumErrsPerMeasurement;
        if (nerrs > 0 && CamParams.Length > 0)
        {
            var problem = new Problem(this, nerrs);
            LevMarq.Optimize(CamParams.AsSpan(), problem, TermCriteria.MaxCount, TermCriteria.Epsilon, TermCriteria.Epsilon * TermCriteria.Epsilon);
        }

        foreach (double v in CamParams)
        {
            if (double.IsNaN(v))
            {
                return false;
            }
        }

        ObtainRefinedCameraParams(cameras);

        MatchesGraph.FindMaxSpanningTree(_numImages, pairwiseMatches, out List<int> centers);
        Mat<float> rInv = SmallMat.Inv(SmallMat.ToFloat(cameras[centers[0]].R));
        for (int i = 0; i < _numImages; i++)
        {
            cameras[i].R = SmallMat.ToDouble(SmallMat.Mul(rInv, SmallMat.ToFloat(cameras[i].R)));
        }

        return true;
    }

    /// <summary>Fills <see cref="CamParams"/> from the cameras.</summary>
    private protected abstract void SetUpInitialCameraParams(List<CameraParams> cameras);

    /// <summary>Writes <see cref="CamParams"/> to the cameras.</summary>
    private protected abstract void ObtainRefinedCameraParams(List<CameraParams> cameras);

    /// <summary>Computes the residuals at <see cref="CamParams"/>.</summary>
    private protected abstract void CalcError(Span<double> err);

    /// <summary>Computes the row-major Jacobian at <see cref="CamParams"/> by central differences.</summary>
    private protected abstract void CalcJacobian(Span<double> jac);

    /// <summary>OpenCV <c>calcDeriv</c> into column <paramref name="col"/>.</summary>
    private protected void NumericColumn(int index, double step, Span<double> jac, int col, double[] err1, double[] err2)
    {
        double val = CamParams[index];
        CamParams[index] = val - step;
        CalcError(err1);
        CamParams[index] = val + step;
        CalcError(err2);
        double h = 2 * step;
        int p = CamParams.Length;
        for (int r = 0; r < err1.Length; r++)
        {
            jac[(r * p) + col] = (err2[r] - err1[r]) / h;
        }

        CamParams[index] = val;
    }

    /// <summary>Rodrigues vector of the orthogonalized <c>float</c> rotation, as the OpenCV <c>setUpInitialCameraParams</c>.</summary>
    private protected static (float X, float Y, float Z) RotationToVector(Mat<double> r)
    {
        Mat<float> rf = SmallMat.ToFloat(r);
        var svd = new Svd<float>(rf, SvdFlags.FullUV);
        Mat<float> rr = SmallMat.Mul(svd.U, svd.Vt);
        if (LinearAlgebra.Determinant(rr) < 0)
        {
            Span<float> s = rr.GetSpan();
            for (int i = 0; i < s.Length; i++)
            {
                s[i] = (s[i] * -1f) + 0f;
            }
        }

        Span<double> d = stackalloc double[9];
        for (int i = 0; i < 9; i++)
        {
            d[i] = rr.At(i / 3, i % 3);
        }

        Rodrigues.ToVector(d, out double x, out double y, out double z);
        return ((float)x, (float)y, (float)z);
    }

    /// <summary>Rotation matrix of a rotation vector, rounded to <c>float</c>.</summary>
    private protected static Mat<double> VectorToRotationF(double x, double y, double z)
    {
        var m = new Mat<double>(3, 3);
        Span<double> d = stackalloc double[9];
        Rodrigues.ToMatrix(x, y, z, d);
        for (int i = 0; i < 9; i++)
        {
            m.At(i / 3, i % 3) = (float)d[i];
        }

        return m;
    }

    private readonly struct Problem : ILevMarqProblem
    {
        private readonly BundleAdjusterBase _owner;

        public Problem(BundleAdjusterBase owner, int nerrs)
        {
            _owner = owner;
            ErrorCount = nerrs;
        }

        public int ParamCount => _owner.CamParams.Length;

        public int ErrorCount { get; }

        public void Compute(ReadOnlySpan<double> x, Span<double> err, Span<double> jac)
        {
            double[] camParams = _owner.CamParams;
            double[] backup = ArrayPool<double>.Shared.Rent(camParams.Length);
            try
            {
                camParams.CopyTo(backup, 0);
                x.CopyTo(camParams);
                if (!jac.IsEmpty)
                {
                    _owner.CalcJacobian(jac);
                }

                _owner.CalcError(err);
                backup.AsSpan(0, camParams.Length).CopyTo(camParams);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(backup);
            }
        }
    }
}

/// <summary>
/// Bundle adjuster that keeps the initial cameras, as OpenCV <c>detail::NoBundleAdjuster</c>.
/// </summary>
public sealed class NoBundleAdjuster : BundleAdjusterBase
{
    /// <summary>Initializes a new instance.</summary>
    public NoBundleAdjuster()
        : base(0, 0)
    {
    }

    /// <inheritdoc/>
    protected override bool EstimateCore(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches, List<CameraParams> cameras) => true;

    private protected override void SetUpInitialCameraParams(List<CameraParams> cameras)
    {
    }

    private protected override void ObtainRefinedCameraParams(List<CameraParams> cameras)
    {
    }

    private protected override void CalcError(Span<double> err)
    {
    }

    private protected override void CalcJacobian(Span<double> jac)
    {
    }
}

/// <summary>
/// Bundle adjuster that minimizes the reprojection error, as OpenCV
/// <c>detail::BundleAdjusterReproj</c>. It refines the focal length, principal point, aspect
/// ratio (selected by <see cref="BundleAdjusterBase.RefinementMask"/>) and the rotation of each
/// camera: 7 parameters per camera, 2 residuals per inlier match.
/// </summary>
public sealed class BundleAdjusterReproj : BundleAdjusterBase
{
    private double[] _err1 = [], _err2 = [];

    /// <summary>Initializes a new instance.</summary>
    public BundleAdjusterReproj()
        : base(7, 2)
    {
    }

    private protected override void SetUpInitialCameraParams(List<CameraParams> cameras)
    {
        CamParams = new double[NumImages * 7];
        for (int i = 0; i < NumImages; i++)
        {
            CamParams[i * 7] = cameras[i].Focal;
            CamParams[(i * 7) + 1] = cameras[i].Ppx;
            CamParams[(i * 7) + 2] = cameras[i].Ppy;
            CamParams[(i * 7) + 3] = cameras[i].Aspect;
            (float x, float y, float z) = RotationToVector(cameras[i].R);
            CamParams[(i * 7) + 4] = x;
            CamParams[(i * 7) + 5] = y;
            CamParams[(i * 7) + 6] = z;
        }
    }

    private protected override void ObtainRefinedCameraParams(List<CameraParams> cameras)
    {
        for (int i = 0; i < NumImages; i++)
        {
            cameras[i].Focal = CamParams[i * 7];
            cameras[i].Ppx = CamParams[(i * 7) + 1];
            cameras[i].Ppy = CamParams[(i * 7) + 2];
            cameras[i].Aspect = CamParams[(i * 7) + 3];
            cameras[i].R = VectorToRotationF(CamParams[(i * 7) + 4], CamParams[(i * 7) + 5], CamParams[(i * 7) + 6]);
        }
    }

    private protected override void CalcError(Span<double> err)
    {
        Span<double> r1 = stackalloc double[9];
        Span<double> r2 = stackalloc double[9];
        int matchIdx = 0;
        foreach ((int i, int j) in Edges)
        {
            double f1 = CamParams[i * 7], f2 = CamParams[j * 7];
            double ppx1 = CamParams[(i * 7) + 1], ppx2 = CamParams[(j * 7) + 1];
            double ppy1 = CamParams[(i * 7) + 2], ppy2 = CamParams[(j * 7) + 2];
            double a1 = CamParams[(i * 7) + 3], a2 = CamParams[(j * 7) + 3];
            Rodrigues.ToMatrix(CamParams[(i * 7) + 4], CamParams[(i * 7) + 5], CamParams[(i * 7) + 6], r1);
            Rodrigues.ToMatrix(CamParams[(j * 7) + 4], CamParams[(j * 7) + 5], CamParams[(j * 7) + 6], r2);
            // h = k2 · inv(r2) · r1 · inv(k1), evaluated left to right.
            r2.CopyTo(TmpA.GetSpan());
            SmallMat.Inv3Into(TmpA, TmpB);
            SmallMat.SetK(TmpA, f2, f2 * a2, ppx2, ppy2);
            SmallMat.MulInto(TmpA, TmpB, TmpC);
            r1.CopyTo(TmpA.GetSpan());
            SmallMat.MulInto(TmpC, TmpA, TmpH2);
            SmallMat.SetK(TmpA, f1, f1 * a1, ppx1, ppy1);
            SmallMat.Inv3Into(TmpA, TmpB);
            SmallMat.MulInto(TmpH2, TmpB, TmpH1);
            ReadOnlySpan<double> h = TmpH1.GetReadOnlySpan();

            ImageFeatures features1 = FeaturesList[i], features2 = FeaturesList[j];
            MatchesInfo info = PairAt(i, j);
            for (int k = 0; k < info.Matches.Count; k++)
            {
                if (info.InliersMask[k] == 0)
                {
                    continue;
                }

                DMatch m = info.Matches[k];
                Point2f p1 = features1.Keypoints[m.QueryIdx].Pt;
                Point2f p2 = features2.Keypoints[m.TrainIdx].Pt;
                double x = (h[0] * p1.X) + (h[1] * p1.Y) + h[2];
                double y = (h[3] * p1.X) + (h[4] * p1.Y) + h[5];
                double z = (h[6] * p1.X) + (h[7] * p1.Y) + h[8];
                err[2 * matchIdx] = p2.X - (x / z);
                err[(2 * matchIdx) + 1] = p2.Y - (y / z);
                matchIdx++;
            }
        }
    }

    private protected override void CalcJacobian(Span<double> jac)
    {
        jac.Clear();
        int nerr = TotalNumMatches * 2;
        if (_err1.Length != nerr)
        {
            _err1 = new double[nerr];
            _err2 = new double[nerr];
        }

        const double Step = 1e-4;
        for (int i = 0; i < NumImages; i++)
        {
            if (RefinementMask[0, 0] != 0)
            {
                NumericColumn(i * 7, Step, jac, i * 7, _err1, _err2);
            }

            if (RefinementMask[0, 2] != 0)
            {
                NumericColumn((i * 7) + 1, Step, jac, (i * 7) + 1, _err1, _err2);
            }

            if (RefinementMask[1, 2] != 0)
            {
                NumericColumn((i * 7) + 2, Step, jac, (i * 7) + 2, _err1, _err2);
            }

            if (RefinementMask[1, 1] != 0)
            {
                NumericColumn((i * 7) + 3, Step, jac, (i * 7) + 3, _err1, _err2);
            }

            for (int j = 4; j < 7; j++)
            {
                NumericColumn((i * 7) + j, Step, jac, (i * 7) + j, _err1, _err2);
            }
        }
    }
}

/// <summary>
/// Bundle adjuster that minimizes the distance between the rays through the matched points, as
/// OpenCV <c>detail::BundleAdjusterRay</c>. It refines the focal length and the rotation of each
/// camera: 4 parameters per camera, 3 residuals per inlier match. The principal points are the
/// image centres.
/// </summary>
public sealed class BundleAdjusterRay : BundleAdjusterBase
{
    private double[] _err1 = [], _err2 = [];

    /// <summary>Initializes a new instance.</summary>
    public BundleAdjusterRay()
        : base(4, 3)
    {
    }

    private protected override void SetUpInitialCameraParams(List<CameraParams> cameras)
    {
        CamParams = new double[NumImages * 4];
        for (int i = 0; i < NumImages; i++)
        {
            CamParams[i * 4] = cameras[i].Focal;
            (float x, float y, float z) = RotationToVector(cameras[i].R);
            CamParams[(i * 4) + 1] = x;
            CamParams[(i * 4) + 2] = y;
            CamParams[(i * 4) + 3] = z;
        }
    }

    private protected override void ObtainRefinedCameraParams(List<CameraParams> cameras)
    {
        for (int i = 0; i < NumImages; i++)
        {
            cameras[i].Focal = CamParams[i * 4];
            cameras[i].R = VectorToRotationF(CamParams[(i * 4) + 1], CamParams[(i * 4) + 2], CamParams[(i * 4) + 3]);
        }
    }

    private protected override void CalcError(Span<double> err)
    {
        Span<double> r1 = stackalloc double[9];
        Span<double> r2 = stackalloc double[9];
        int matchIdx = 0;
        foreach ((int i, int j) in Edges)
        {
            double f1 = CamParams[i * 4], f2 = CamParams[j * 4];
            Rodrigues.ToMatrix(CamParams[(i * 4) + 1], CamParams[(i * 4) + 2], CamParams[(i * 4) + 3], r1);
            Rodrigues.ToMatrix(CamParams[(j * 4) + 1], CamParams[(j * 4) + 2], CamParams[(j * 4) + 3], r2);
            ImageFeatures features1 = FeaturesList[i], features2 = FeaturesList[j];
            SmallMat.SetK(TmpA, f1, f1, features1.ImgSize.Width * 0.5, features1.ImgSize.Height * 0.5);
            SmallMat.Inv3Into(TmpA, TmpB);
            r1.CopyTo(TmpC.GetSpan());
            SmallMat.MulInto(TmpC, TmpB, TmpH1);
            SmallMat.SetK(TmpA, f2, f2, features2.ImgSize.Width * 0.5, features2.ImgSize.Height * 0.5);
            SmallMat.Inv3Into(TmpA, TmpB);
            r2.CopyTo(TmpC.GetSpan());
            SmallMat.MulInto(TmpC, TmpB, TmpH2);
            ReadOnlySpan<double> h1 = TmpH1.GetReadOnlySpan();
            ReadOnlySpan<double> h2 = TmpH2.GetReadOnlySpan();
            double mult = Math.Sqrt(f1 * f2);

            MatchesInfo info = PairAt(i, j);
            for (int k = 0; k < info.Matches.Count; k++)
            {
                if (info.InliersMask[k] == 0)
                {
                    continue;
                }

                DMatch m = info.Matches[k];
                Point2f p1 = features1.Keypoints[m.QueryIdx].Pt;
                double x1 = (h1[0] * p1.X) + (h1[1] * p1.Y) + h1[2];
                double y1 = (h1[3] * p1.X) + (h1[4] * p1.Y) + h1[5];
                double z1 = (h1[6] * p1.X) + (h1[7] * p1.Y) + h1[8];
                double len = Math.Sqrt((x1 * x1) + (y1 * y1) + (z1 * z1));
                x1 /= len;
                y1 /= len;
                z1 /= len;

                Point2f p2 = features2.Keypoints[m.TrainIdx].Pt;
                double x2 = (h2[0] * p2.X) + (h2[1] * p2.Y) + h2[2];
                double y2 = (h2[3] * p2.X) + (h2[4] * p2.Y) + h2[5];
                double z2 = (h2[6] * p2.X) + (h2[7] * p2.Y) + h2[8];
                len = Math.Sqrt((x2 * x2) + (y2 * y2) + (z2 * z2));
                x2 /= len;
                y2 /= len;
                z2 /= len;

                err[3 * matchIdx] = mult * (x1 - x2);
                err[(3 * matchIdx) + 1] = mult * (y1 - y2);
                err[(3 * matchIdx) + 2] = mult * (z1 - z2);
                matchIdx++;
            }
        }
    }

    private protected override void CalcJacobian(Span<double> jac)
    {
        int nerr = TotalNumMatches * 3;
        if (_err1.Length != nerr)
        {
            _err1 = new double[nerr];
            _err2 = new double[nerr];
        }

        const double Step = 1e-3;
        for (int i = 0; i < NumImages; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                NumericColumn((i * 4) + j, Step, jac, (i * 4) + j, _err1, _err2);
            }
        }
    }
}

/// <summary>
/// Bundle adjuster for full affine transforms, as OpenCV <c>detail::BundleAdjusterAffine</c>:
/// 6 parameters per camera (a, b, tx, c, d, ty), 2 residuals per inlier match. Use it with
/// <see cref="AffineBasedEstimator"/>.
/// </summary>
public sealed class BundleAdjusterAffine : BundleAdjusterBase
{
    private double[] _err1 = [], _err2 = [];

    /// <summary>Initializes a new instance.</summary>
    public BundleAdjusterAffine()
        : base(6, 2)
    {
    }

    private protected override void SetUpInitialCameraParams(List<CameraParams> cameras)
    {
        CamParams = new double[NumImages * 6];
        for (int i = 0; i < NumImages; i++)
        {
            for (int k = 0; k < 6; k++)
            {
                CamParams[(i * 6) + k] = (float)cameras[i].R.At(k / 3, k % 3);
            }
        }
    }

    private protected override void ObtainRefinedCameraParams(List<CameraParams> cameras)
    {
        for (int i = 0; i < NumImages; i++)
        {
            Mat<double> r = SmallMat.Eye3();
            for (int k = 0; k < 6; k++)
            {
                r.At(k / 3, k % 3) = (float)CamParams[(i * 6) + k];
            }

            cameras[i].R = r;
        }
    }

    private protected override void CalcError(Span<double> err)
    {
        int matchIdx = 0;
        foreach ((int i, int j) in Edges)
        {
            SmallMat.SetEye3(TmpA);
            SmallMat.SetEye3(TmpB);
            Geometry.InvertAffineTransform(CamParams.AsSpan(i * 6, 6), TmpA.GetSpan()[..6]);
            CamParams.AsSpan(j * 6, 6).CopyTo(TmpB.GetSpan());
            SmallMat.MulInto(TmpA, TmpB, TmpH1);
            ReadOnlySpan<double> h = TmpH1.GetReadOnlySpan();
            ImageFeatures features1 = FeaturesList[i], features2 = FeaturesList[j];
            MatchesInfo info = PairAt(i, j);
            for (int k = 0; k < info.Matches.Count; k++)
            {
                if (info.InliersMask[k] == 0)
                {
                    continue;
                }

                DMatch m = info.Matches[k];
                Point2f p1 = features1.Keypoints[m.QueryIdx].Pt;
                Point2f p2 = features2.Keypoints[m.TrainIdx].Pt;
                double x = (h[0] * p1.X) + (h[1] * p1.Y) + h[2];
                double y = (h[3] * p1.X) + (h[4] * p1.Y) + h[5];
                err[2 * matchIdx] = p2.X - x;
                err[(2 * matchIdx) + 1] = p2.Y - y;
                matchIdx++;
            }
        }
    }

    private protected override void CalcJacobian(Span<double> jac)
    {
        int nerr = TotalNumMatches * 2;
        if (_err1.Length != nerr)
        {
            _err1 = new double[nerr];
            _err2 = new double[nerr];
        }

        const double Step = 1e-4;
        for (int i = 0; i < NumImages; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                NumericColumn((i * 6) + j, Step, jac, (i * 6) + j, _err1, _err2);
            }
        }
    }
}

/// <summary>
/// Bundle adjuster for similarity transforms, as OpenCV <c>detail::BundleAdjusterAffinePartial</c>:
/// 4 parameters per camera (a, b, tx, ty for <c>[a -b tx; b a ty]</c>), 2 residuals per inlier
/// match. The <see cref="Stitcher"/> SCANS mode uses it.
/// </summary>
public sealed class BundleAdjusterAffinePartial : BundleAdjusterBase
{
    private double[] _err1 = [], _err2 = [];

    /// <summary>Initializes a new instance.</summary>
    public BundleAdjusterAffinePartial()
        : base(4, 2)
    {
    }

    private protected override void SetUpInitialCameraParams(List<CameraParams> cameras)
    {
        CamParams = new double[NumImages * 4];
        for (int i = 0; i < NumImages; i++)
        {
            Mat<double> r = cameras[i].R;
            CamParams[i * 4] = (float)r.At(0, 0);
            CamParams[(i * 4) + 1] = (float)r.At(1, 0);
            CamParams[(i * 4) + 2] = (float)r.At(0, 2);
            CamParams[(i * 4) + 3] = (float)r.At(1, 2);
        }
    }

    private protected override void ObtainRefinedCameraParams(List<CameraParams> cameras)
    {
        for (int i = 0; i < NumImages; i++)
        {
            ReadOnlySpan<double> p = CamParams.AsSpan(i * 4, 4);
            Mat<double> r = SmallMat.Eye3();
            r.At(0, 0) = (float)p[0];
            r.At(0, 1) = (float)-p[1];
            r.At(0, 2) = (float)p[2];
            r.At(1, 0) = (float)p[1];
            r.At(1, 1) = (float)p[0];
            r.At(1, 2) = (float)p[3];
            cameras[i].R = r;
        }
    }

    private protected override void CalcError(Span<double> err)
    {
        int matchIdx = 0;
        foreach ((int i, int j) in Edges)
        {
            Similarity(CamParams.AsSpan(i * 4, 4), TmpC);
            Similarity(CamParams.AsSpan(j * 4, 4), TmpB);
            SmallMat.SetEye3(TmpA);
            Geometry.InvertAffineTransform(TmpC.GetReadOnlySpan()[..6], TmpA.GetSpan()[..6]);
            SmallMat.MulInto(TmpA, TmpB, TmpH1);
            ReadOnlySpan<double> h = TmpH1.GetReadOnlySpan();
            ImageFeatures features1 = FeaturesList[i], features2 = FeaturesList[j];
            MatchesInfo info = PairAt(i, j);
            for (int k = 0; k < info.Matches.Count; k++)
            {
                if (info.InliersMask[k] == 0)
                {
                    continue;
                }

                DMatch m = info.Matches[k];
                Point2f p1 = features1.Keypoints[m.QueryIdx].Pt;
                Point2f p2 = features2.Keypoints[m.TrainIdx].Pt;
                double x = (h[0] * p1.X) + (h[1] * p1.Y) + h[2];
                double y = (h[3] * p1.X) + (h[4] * p1.Y) + h[5];
                err[2 * matchIdx] = p2.X - x;
                err[(2 * matchIdx) + 1] = p2.Y - y;
                matchIdx++;
            }
        }
    }

    private protected override void CalcJacobian(Span<double> jac)
    {
        int nerr = TotalNumMatches * 2;
        if (_err1.Length != nerr)
        {
            _err1 = new double[nerr];
            _err2 = new double[nerr];
        }

        const double Step = 1e-4;
        for (int i = 0; i < NumImages; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                NumericColumn((i * 4) + j, Step, jac, (i * 4) + j, _err1, _err2);
            }
        }
    }

    private static void Similarity(ReadOnlySpan<double> p, Mat<double> m)
    {
        SmallMat.SetEye3(m);
        m.At(0, 0) = p[0];
        m.At(0, 1) = -p[1];
        m.At(0, 2) = p[2];
        m.At(1, 0) = p[1];
        m.At(1, 1) = p[0];
        m.At(1, 2) = p[3];
    }
}
