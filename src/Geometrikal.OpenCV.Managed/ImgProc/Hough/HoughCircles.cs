// This file is a managed port of OpenCV modules/imgproc/src/hough.cpp (HoughCircles,
// HOUGH_GRADIENT).
// OpenCV is licensed under the Apache License 2.0. Copyright (C) 2000-2024, Intel Corporation,
// all rights reserved, and OpenCV contributors.

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>A circle found by <see cref="HoughTransforms.HoughCircles"/>.</summary>
/// <param name="X">Centre X in pixels.</param>
/// <param name="Y">Centre Y in pixels.</param>
/// <param name="Radius">Radius in pixels (0 in centres-only mode).</param>
/// <param name="Votes">Support of the circle (OpenCV <c>CV_32FC4</c> fourth value): the edge points
/// near the radius, or the accumulator index of the centre in centres-only mode.</param>
public readonly record struct HoughCircle(float X, float Y, float Radius, float Votes);

public static partial class HoughTransforms
{
    /// <summary>C <c>FLT_EPSILON</c>; <see cref="float.Epsilon"/> is the smallest subnormal instead.</summary>
    private const float FltEpsilon = 1.1920929e-7f;

    /// <summary>
    /// Finds circles with the Hough gradient method, as OpenCV
    /// <c>HoughCircles(image, HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius)</c>.
    /// Circles are sorted by decreasing support.
    /// </summary>
    /// <param name="image">8-bit single-channel image (usually blurred).</param>
    /// <param name="dp">Inverse ratio of the accumulator resolution to the image resolution (at least 1).</param>
    /// <param name="minDist">Smallest distance between circle centres.</param>
    /// <param name="cannyThreshold">Upper Canny threshold (<c>param1</c>); the lower is half of it.</param>
    /// <param name="accumulatorThreshold">Votes a centre needs, and support a circle needs (<c>param2</c>).</param>
    /// <param name="minRadius">Smallest radius (negative values are 0).</param>
    /// <param name="maxRadius">
    /// Largest radius. 0 uses the larger image side; a negative value returns centres only (radius 0);
    /// a value not above <paramref name="minRadius"/> becomes <c>minRadius + 2</c>.
    /// </param>
    /// <param name="maxCircles">Largest number of circles returned; negative for no limit.</param>
    /// <param name="sobelKernelSize">Sobel aperture: 3, 5, 7, or -1 for Scharr (<c>param3</c>).</param>
    /// <remarks>
    /// The OpenCV <c>HOUGH_GRADIENT_ALT</c> method is not ported. The accumulator, centres and radius
    /// estimation are computed on one thread; OpenCV's parallel results do not depend on the threads.
    /// </remarks>
    public static HoughCircle[] HoughCircles(
        Mat<byte> image,
        double dp,
        double minDist,
        double cannyThreshold,
        double accumulatorThreshold,
        int minRadius = 0,
        int maxRadius = 0,
        int maxCircles = -1,
        int sobelKernelSize = 3)
    {
        ValidateImage(image);
        if (dp <= 0 || minDist <= 0 || cannyThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(dp), "dp, min_dist and canny_threshold must be all positive numbers");
        }

        if (accumulatorThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(accumulatorThreshold), "acc_threshold must be a positive number");
        }

        if (sobelKernelSize is not (-1 or 3 or 5 or 7))
        {
            throw new ArgumentOutOfRangeException(nameof(sobelKernelSize));
        }

        int cannyThresh = ScalarConverter.CvRound(cannyThreshold);
        int accThresh = ScalarConverter.CvRound(accumulatorThreshold);
        minRadius = Math.Max(0, minRadius);
        if (maxCircles < 0)
        {
            maxCircles = int.MaxValue;
        }

        bool centersOnly = maxRadius < 0;
        if (maxRadius <= 0)
        {
            maxRadius = Math.Max(image.Height, image.Width);
        }
        else if (maxRadius <= minRadius)
        {
            maxRadius = minRadius + 2;
        }

        return CirclesGradient(image, (float)dp, (float)minDist, minRadius, maxRadius, cannyThresh, accThresh, maxCircles, sobelKernelSize, centersOnly);
    }

    private static HoughCircle[] CirclesGradient(
        Mat<byte> image, float dp, float minDist, int minRadius, int maxRadius, int cannyThreshold, int accThreshold, int maxCircles, int kernelSize, bool centersOnly)
    {
        dp = Math.Max(dp, 1f);
        float idp = 1f / dp;
        var dx = Derivatives.Sobel<byte, short>(image, 1, 0, kernelSize, 1, 0, BorderType.Replicate);
        var dy = Derivatives.Sobel<byte, short>(image, 0, 1, kernelSize, 1, 0, BorderType.Replicate);
        var edges = EdgeDetection.Canny(dx, dy, Math.Max(1, cannyThreshold / 2), cannyThreshold);

        int width = image.Width, height = image.Height;
        int acols = ScalarConverter.CvCeil(width * idp), arows = ScalarConverter.CvCeil(height * idp);
        int astep = acols + 2;
        var accum = new int[(arows + 2) * astep];
        var nz = new List<Point>();
        var isNonZero = new bool[width * height];

        for (int y = 0; y < height; y++)
        {
            var e = edges.GetReadOnlyRowSpan(y);
            var dxRow = dx.GetReadOnlyRowSpan(y);
            var dyRow = dy.GetReadOnlyRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                if (e[x] == 0)
                {
                    continue;
                }

                float vx = dxRow[x], vy = dyRow[x];
                if (vx == 0 && vy == 0)
                {
                    continue;
                }

                float mag = MathF.Sqrt((vx * vx) + (vy * vy));
                if (mag < 1f)
                {
                    continue;
                }

                isNonZero[(y * width) + x] = true;
                int sx = ScalarConverter.CvRound(vx * idp * 1024 / mag);
                int sy = ScalarConverter.CvRound(vy * idp * 1024 / mag);
                int x0 = ScalarConverter.CvRound(x * idp * 1024);
                int y0 = ScalarConverter.CvRound(y * idp * 1024);
                for (int k1 = 0; k1 < 2; k1++)
                {
                    int x1 = x0 + (minRadius * sx);
                    int y1 = y0 + (minRadius * sy);
                    for (int r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, r++)
                    {
                        int x2 = x1 >> 10, y2 = y1 >> 10;
                        if ((uint)x2 >= (uint)acols || (uint)y2 >= (uint)arows)
                        {
                            break;
                        }

                        accum[(y2 * astep) + x2]++;
                    }

                    sx = -sx;
                    sy = -sy;
                }
            }
        }

        // The non-zero set in raster order (OpenCV NZPointSet::toList).
        for (int i = 0; i < isNonZero.Length; i++)
        {
            if (isNonZero[i])
            {
                nz.Add(new Point(i % width, i / width));
            }
        }

        if (nz.Count == 0)
        {
            return [];
        }

        int accumRows = arows + 2;
        var centers = new List<int>();
        for (int y = 1; y < accumRows - 1; y++)
        {
            for (int x = 1, b = (y * astep) + 1; x < astep - 1; x++, b++)
            {
                if (accum[b] > accThreshold && accum[b] > accum[b - 1] && accum[b] >= accum[b + 1]
                    && accum[b] > accum[b - astep] && accum[b] >= accum[b + astep])
                {
                    centers.Add(b);
                }
            }
        }

        if (centers.Count == 0)
        {
            return [];
        }

        centers.Sort((l1, l2) => accum[l1] != accum[l2] ? accum[l2].CompareTo(accum[l1]) : l1.CompareTo(l2));
        var circles = new List<HoughCircle>();
        float minDist2 = minDist * minDist;
        if (centersOnly)
        {
            foreach (int center in centers)
            {
                int y = center / astep;
                int x = center - (y * astep);
                var circle = new HoughCircle((x + 0.5f) * dp, (y + 0.5f) * dp, 0, center);
                if (FarFromAll(circles, circles.Count, circle, minDist2))
                {
                    circles.Add(circle);
                }
            }
        }
        else
        {
            var estimated = EstimateRadii(nz, centers, astep, accThreshold, minRadius, maxRadius, dp);
            estimated.Sort(CompareSupport);
            circles.AddRange(estimated);

            // RemoveOverlaps: keep a circle only when it is far from all kept circles before it.
            int end = Math.Min(1, circles.Count);
            for (int i = 1; i < circles.Count; i++)
            {
                if (FarFromAll(circles, end, circles[i], minDist2))
                {
                    circles[end++] = circles[i];
                }
            }

            circles.RemoveRange(end, circles.Count - end);
        }

        return circles.Take(Math.Min(maxCircles, circles.Count)).ToArray();
    }

    /// <summary>OpenCV <c>HoughCircleEstimateRadiusInvoker</c>: the radius with the best support per centre.</summary>
    private static List<HoughCircle> EstimateRadii(List<Point> nz, List<int> centers, int acols, int accThreshold, int minRadius, int maxRadius, float dr)
    {
        const int nBinsPerDr = 10;
        float minRadius2 = (float)minRadius * minRadius;
        float maxRadius2 = (float)maxRadius * maxRadius;
        int nBins = ScalarConverter.CvRound((maxRadius - minRadius) / dr * nBinsPerDr);
        var bins = new int[nBins];
        var distances = new float[nz.Count];
        var result = new List<HoughCircle>();
        foreach (int ofs in centers)
        {
            int y = ofs / acols;
            int x = ofs - (y * acols);
            float cx = (x + 0.5f) * dr, cy = (y + 0.5f) * dr;
            int count = 0;
            foreach (var pt in nz)
            {
                float ddx = cx - pt.X, ddy = cy - pt.Y;
                float r2 = (ddx * ddx) + (ddy * ddy);
                if (minRadius2 <= r2 && r2 <= maxRadius2)
                {
                    distances[count++] = r2;
                }
            }

            int maxCount = 0;
            float rBest = 0;
            if (count > 0)
            {
                Array.Clear(bins);
                for (int k = 0; k < count; k++)
                {
                    int bin = Math.Max(0, Math.Min(nBins - 1, ScalarConverter.CvRound((MathF.Sqrt(distances[k]) - minRadius) / dr * nBinsPerDr)));
                    bins[bin]++;
                }

                for (int j = nBins - 1; j > 0; j--)
                {
                    if (bins[j] == 0)
                    {
                        continue;
                    }

                    int upbin = j;
                    int curCount = 0;
                    for (; j > upbin - nBinsPerDr && j >= 0; j--)
                    {
                        curCount += bins[j];
                    }

                    float rCur = ((upbin + j) / 2f / nBinsPerDr * dr) + minRadius;
                    if (curCount * rBest >= maxCount * rCur || (rBest < FltEpsilon && curCount >= maxCount))
                    {
                        rBest = rCur;
                        maxCount = curCount;
                    }
                }
            }

            if (maxCount > accThreshold)
            {
                result.Add(new HoughCircle(cx, cy, rBest, maxCount));
            }
        }

        return result;
    }

    /// <summary>OpenCV <c>cmpAccum</c>: more support, then larger radius, smaller X, smaller Y.</summary>
    private static int CompareSupport(HoughCircle a, HoughCircle b)
    {
        if (a.Votes != b.Votes)
        {
            return b.Votes.CompareTo(a.Votes);
        }

        if (a.Radius != b.Radius)
        {
            return b.Radius.CompareTo(a.Radius);
        }

        return a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y);
    }

    private static bool FarFromAll(List<HoughCircle> circles, int end, HoughCircle circle, float minDist2)
    {
        for (int j = 0; j < end; j++)
        {
            float distX = circle.X - circles[j].X, distY = circle.Y - circles[j].Y;
            if ((distX * distX) + (distY * distY) < minDist2)
            {
                return false;
            }
        }

        return true;
    }
}
