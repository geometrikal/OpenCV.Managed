// This implementation is ported from the OpenCV 5.0 stitching camera autocalibration.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/autocalib.hpp
//   modules/stitching/src/autocalib.cpp (focalsFromHomography, estimateFocal,
//   calibrateRotatingCamera)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Focal length estimation for rotating cameras, as the OpenCV <c>detail</c> autocalibration
/// functions.
/// </summary>
public static class Autocalibration
{
    /// <summary>
    /// Estimates the focal lengths of two cameras from the homography between their images, as
    /// OpenCV <c>focalsFromHomography</c>. The cameras rotate around their centres and have the
    /// principal point at the image origin.
    /// </summary>
    /// <param name="h">3 × 3 homography. It is not modified.</param>
    /// <param name="f0">Focal length estimate of the first camera.</param>
    /// <param name="f1">Focal length estimate of the second camera.</param>
    /// <param name="f0Ok">True when <paramref name="f0"/> is valid.</param>
    /// <param name="f1Ok">True when <paramref name="f1"/> is valid.</param>
    /// <exception cref="ArgumentException"><paramref name="h"/> is not 3 × 3.</exception>
    public static void FocalsFromHomography(Mat<double> h, out double f0, out double f1, out bool f0Ok, out bool f1Ok)
    {
        ArgumentNullException.ThrowIfNull(h);
        if (h.Width != 3 || h.Height != 3 || h.Channels != 1)
        {
            throw new ArgumentException("The homography must be 3 × 3.", nameof(h));
        }

        Span<double> m = stackalloc double[9];
        for (int i = 0; i < 9; i++)
        {
            m[i] = h.At(i / 3, i % 3);
        }

        f0 = f1 = 0;
        f1Ok = true;
        double d1 = m[6] * m[7];
        double d2 = (m[7] - m[6]) * (m[7] + m[6]);
        double v1 = -((m[0] * m[1]) + (m[3] * m[4])) / d1;
        double v2 = ((m[0] * m[0]) + (m[3] * m[3]) - (m[1] * m[1]) - (m[4] * m[4])) / d2;
        if (v1 < v2)
        {
            (v1, v2) = (v2, v1);
            (d1, d2) = (d2, d1);
        }

        if (v1 > 0 && v2 > 0)
        {
            f1 = Math.Sqrt(Math.Abs(d1) > Math.Abs(d2) ? v1 : v2);
        }
        else if (v1 > 0)
        {
            f1 = Math.Sqrt(v1);
        }
        else
        {
            f1Ok = false;
        }

        f0Ok = true;
        d1 = (m[0] * m[3]) + (m[1] * m[4]);
        d2 = (m[0] * m[0]) + (m[1] * m[1]) - (m[3] * m[3]) - (m[4] * m[4]);
        v1 = -m[2] * m[5] / d1;
        v2 = ((m[5] * m[5]) - (m[2] * m[2])) / d2;
        if (v1 < v2)
        {
            (v1, v2) = (v2, v1);
            (d1, d2) = (d2, d1);
        }

        if (v1 > 0 && v2 > 0)
        {
            f0 = Math.Sqrt(Math.Abs(d1) > Math.Abs(d2) ? v1 : v2);
        }
        else if (v1 > 0)
        {
            f0 = Math.Sqrt(v1);
        }
        else
        {
            f0Ok = false;
        }
    }

    /// <summary>
    /// Estimates one focal length for all cameras, as OpenCV <c>estimateFocal</c>: the median of
    /// <c>sqrt(f0·f1)</c> over all pairwise homographies with two valid estimates. When there are
    /// fewer than n - 1 estimates, every focal length is the mean of <c>width + height</c>.
    /// </summary>
    /// <param name="features">Features of the n images (only the image sizes are used).</param>
    /// <param name="pairwiseMatches">The n × n pairwise matches.</param>
    /// <returns>n focal lengths.</returns>
    public static double[] EstimateFocal(IReadOnlyList<ImageFeatures> features, IReadOnlyList<MatchesInfo> pairwiseMatches)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(pairwiseMatches);
        int n = features.Count;
        var focals = new double[n];
        var all = new List<double>();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                MatchesInfo m = pairwiseMatches[(i * n) + j];
                if (m.H is null)
                {
                    continue;
                }

                FocalsFromHomography(m.H, out double f0, out double f1, out bool ok0, out bool ok1);
                if (ok0 && ok1)
                {
                    all.Add(Math.Sqrt(f0 * f1));
                }
            }
        }

        if (all.Count >= n - 1)
        {
            all.Sort();
            double median = all.Count % 2 == 1
                ? all[all.Count / 2]
                : (all[(all.Count / 2) - 1] + all[all.Count / 2]) * 0.5;
            Array.Fill(focals, median);
        }
        else
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += features[i].ImgSize.Width + features[i].ImgSize.Height;
            }

            Array.Fill(focals, sum / n);
        }

        return focals;
    }

    /// <summary>
    /// Estimates the intrinsic matrix of a rotating camera from homographies, as OpenCV
    /// <c>calibrateRotatingCamera</c>.
    /// </summary>
    /// <param name="hs">One or more 3 × 3 homographies. They are not modified.</param>
    /// <param name="k">The upper-triangular 3 × 3 intrinsic matrix, or null when the Cholesky factorization fails.</param>
    /// <returns>True when <paramref name="k"/> was computed.</returns>
    public static bool CalibrateRotatingCamera(IReadOnlyList<Mat<double>> hs, out Mat<double>? k)
    {
        ArgumentNullException.ThrowIfNull(hs);
        int m = hs.Count;
        if (m < 1)
        {
            throw new ArgumentException("At least one homography is required.", nameof(hs));
        }

        ReadOnlySpan<int> idx = [0, 1, 2, 1, 3, 4, 2, 4, 5];
        var a = new Mat<double>(6, 6 * m);
        int eq = 0;
        Span<double> h = stackalloc double[9];
        for (int q = 0; q < m; q++)
        {
            Mat<double> hq = hs[q];
            if (hq.Width != 3 || hq.Height != 3)
            {
                throw new ArgumentException("Each homography must be 3 × 3.", nameof(hs));
            }

            double inv = 1.0 / Math.Cbrt(LinearAlgebra.Determinant(hq));
            for (int i = 0; i < 9; i++)
            {
                h[i] = (hq.At(i / 3, i % 3) * inv) + 0.0;
            }

            for (int i = 0; i < 3; i++)
            {
                for (int j = i; j < 3; j++, eq++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        for (int s = 0; s < 3; s++)
                        {
                            a.At(eq, idx[(l * 3) + s]) += h[(i * 3) + l] * h[(j * 3) + s];
                        }
                    }

                    a.At(eq, idx[(i * 3) + j]) -= 1;
                }
            }
        }

        Mat<double> wcoef = Svd<double>.SolveZ(a);
        var w = new double[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = i; j < 3; j++)
            {
                w[(i * 3) + j] = w[(j * 3) + i] = wcoef.At(idx[(i * 3) + j], 0) / wcoef.At(5, 0);
            }
        }

        if (!LapackDecomp.Cholesky<double>(w, 3, 3, default, 0, 0))
        {
            k = null;
            return false;
        }

        w[1] = w[2] = w[5] = 0;
        k = new Mat<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                k.At(i, j) = w[(j * 3) + i];
            }
        }

        return true;
    }
}
