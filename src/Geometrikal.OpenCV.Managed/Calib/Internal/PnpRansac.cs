// This implementation is ported from the OpenCV RANSAC PnP solver.
//
// Original sources:
//   modules/geometry/src/solvepnp.cpp (solvePnPRansac, PnPRansacCallback)
//   modules/geometry/src/ptsetreg.cpp (RANSACPointSetRegistrator::run, getSubset, findInliers)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>solvePnPRansac</c> with <c>RANSACPointSetRegistrator</c> and <c>PnPRansacCallback</c>.</summary>
internal static class PnpRansac
{
    public static PnPRansacResult Run(
        ReadOnlySpan<Point3f> objectPoints,
        ReadOnlySpan<Point2f> imagePoints,
        Mat<double> cameraMatrix,
        double[] dist,
        SolvePnPMethod method,
        int iterationsCount,
        float reprojectionError,
        double confidence,
        PnPPose? extrinsicGuess)
    {
        ArgumentNullException.ThrowIfNull(cameraMatrix);
        int npoints = objectPoints.Length;
        if (npoints < 4 || npoints != imagePoints.Length)
        {
            throw new ArgumentException("At least 4 points are necessary and the object and image point counts must be equal.", nameof(objectPoints));
        }

        CameraDistortion.Expand(dist, nameof(dist));
        bool useExtrinsicGuess = extrinsicGuess.HasValue;
        PnPPose guess = extrinsicGuess ?? default;

        int modelPoints = 5;
        SolvePnPMethod kernelMethod = SolvePnPMethod.Epnp;
        if (method is SolvePnPMethod.P3p or SolvePnPMethod.Ap3p)
        {
            modelPoints = 4;
            kernelMethod = method;
        }
        else if (npoints == 4)
        {
            modelPoints = 4;
            kernelMethod = SolvePnPMethod.P3p;
        }

        Point3d[] obj = Widen(objectPoints);
        Point2d[] img = Widen(imagePoints);

        if (modelPoints == npoints)
        {
            List<PnPPose> direct = PnP.Generic(obj, img, cameraMatrix, dist, kernelMethod, useExtrinsicGuess, guess, true);
            if (direct.Count == 0)
            {
                return new PnPRansacResult(false, default, default, []);
            }

            int[] all = new int[npoints];
            for (int i = 0; i < npoints; i++)
            {
                all[i] = i;
            }

            return new PnPRansacResult(true, direct[0].RotationVector, direct[0].TranslationVector, all);
        }

        if (!(confidence > 0 && confidence < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(confidence), "The confidence must be in (0, 1).");
        }

        var mask = new byte[npoints];
        if (!RunRansac(objectPoints, imagePoints, obj, img, cameraMatrix, dist, kernelMethod, modelPoints, reprojectionError, confidence, iterationsCount, out PnPPose model, mask))
        {
            return new PnPRansacResult(false, default, default, []);
        }

        var inObj = new List<Point3d>(npoints);
        var inImg = new List<Point2d>(npoints);
        var inliers = new List<int>(npoints);
        for (int i = 0; i < npoints; i++)
        {
            if (mask[i] != 0)
            {
                inObj.Add(obj[i]);
                inImg.Add(img[i]);
                inliers.Add(i);
            }
        }

        if (method == SolvePnPMethod.Iterative && !useExtrinsicGuess)
        {
            guess = model;
            useExtrinsicGuess = true;
        }

        SolvePnPMethod finalMethod = method is SolvePnPMethod.P3p or SolvePnPMethod.Ap3p ? SolvePnPMethod.Epnp : method;
        List<PnPPose> final = PnP.Generic(inObj.ToArray(), inImg.ToArray(), cameraMatrix, dist, finalMethod, useExtrinsicGuess, guess, false);
        if (final.Count == 0)
        {
            return new PnPRansacResult(false, model.RotationVector, model.TranslationVector, []);
        }

        return new PnPRansacResult(true, final[0].RotationVector, final[0].TranslationVector, inliers.ToArray());
    }

    private static bool RunRansac(
        ReadOnlySpan<Point3f> m1,
        ReadOnlySpan<Point2f> m2,
        Point3d[] obj,
        Point2d[] img,
        Mat<double> cameraMatrix,
        double[] dist,
        SolvePnPMethod kernelMethod,
        int modelPoints,
        double threshold,
        double confidence,
        int maxIters,
        out PnPPose bestModel,
        byte[] bestMask)
    {
        int count = m1.Length;
        var rng = new CvRng(ulong.MaxValue);
        int niters = Math.Max(maxIters, 1);
        int maxGoodCount = 0;
        float t = (float)(threshold * threshold);
        bestModel = default;

        var ms1 = new Point3d[modelPoints];
        var ms2 = new Point2d[modelPoints];
        Span<int> idx = stackalloc int[modelPoints];
        var mask = new byte[count];
        Point2d[] projArray = ArrayPool<Point2d>.Shared.Rent(count);
        Span<Point2d> proj = projArray.AsSpan(0, count);
        try
        {
            for (int iter = 0; iter < niters; iter++)
            {
                // PnPRansacCallback has no checkSubset, so the first sample is accepted.
                for (int i = 0; i < modelPoints; i++)
                {
                    int idxI;
                    do
                    {
                        idxI = rng.Uniform(0, count);
                    }
                    while (idx[..i].Contains(idxI));

                    idx[i] = idxI;
                    ms1[i] = obj[idxI];
                    ms2[i] = img[idxI];
                }

                List<PnPPose> models = PnP.Generic(ms1, ms2, cameraMatrix, dist, kernelMethod, false, default, true);
                if (models.Count == 0)
                {
                    continue;
                }

                PnPPose model = models[0];
                int goodCount = FindInliers(obj, m2, model, cameraMatrix, dist, t, mask, proj);
                if (goodCount > Math.Max(maxGoodCount, modelPoints - 1))
                {
                    mask.CopyTo(bestMask, 0);
                    bestModel = model;
                    maxGoodCount = goodCount;
                    niters = PointSetRegistrator.UpdateNumIters(confidence, (double)(count - goodCount) / count, modelPoints, niters);
                }
            }
        }
        finally
        {
            ArrayPool<Point2d>.Shared.Return(projArray);
        }

        return maxGoodCount > 0;
    }

    private static int FindInliers(Point3d[] obj, ReadOnlySpan<Point2f> m2, PnPPose model, Mat<double> cameraMatrix, double[] dist, float t, byte[] mask, Span<Point2d> proj)
    {
        CameraModel.ProjectPointsInto(obj, model.RotationVector, model.TranslationVector, cameraMatrix, dist, proj);
        int nz = 0;
        for (int i = 0; i < m2.Length; i++)
        {
            float dx = m2[i].X - (float)proj[i].X;
            float dy = m2[i].Y - (float)proj[i].Y;
            float s = 0f + (dx * dx);
            s += dy * dy;
            float err = (float)(double)s;
            int f = err <= t ? 1 : 0;
            mask[i] = (byte)f;
            nz += f;
        }

        return nz;
    }

    private static Point3d[] Widen(ReadOnlySpan<Point3f> p)
    {
        var d = new Point3d[p.Length];
        for (int i = 0; i < p.Length; i++)
        {
            d[i] = new Point3d(p[i].X, p[i].Y, p[i].Z);
        }

        return d;
    }

    private static Point2d[] Widen(ReadOnlySpan<Point2f> p)
    {
        var d = new Point2d[p.Length];
        for (int i = 0; i < p.Length; i++)
        {
            d[i] = new Point2d(p[i].X, p[i].Y);
        }

        return d;
    }
}
