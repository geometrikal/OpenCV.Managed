// This implementation is ported from OpenCV's point-in-contour test.
//
// Original source:
//   modules/geometry/src/geometry.cpp (pointPolygonTest)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/geometry.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>pointPolygonTest</c> over interleaved (x, y) coordinates.</summary>
internal static class FitPointPolygonTest
{
    /// <summary>Returns the signed distance, or +1/-1/0, as OpenCV <c>pointPolygonTest</c>.</summary>
    public static double Compute<T>(ReadOnlySpan<T> xy, Point2f pt, bool measureDist)
        where T : unmanaged, INumber<T>
    {
        int total = xy.Length / 2;
        if (total == 0)
        {
            return measureDist ? -double.MaxValue : -1;
        }

        int ipx = ScalarConverter.CvRound(pt.X);
        int ipy = ScalarConverter.CvRound(pt.Y);
        int counter = 0;

        if (typeof(T) == typeof(int) && !measureDist && ipx == pt.X && ipy == pt.Y)
        {
            int vx = int.CreateTruncating(xy[2 * (total - 1)]);
            int vy = int.CreateTruncating(xy[(2 * total) - 1]);
            for (int i = 0; i < total; i++)
            {
                int v0x = vx, v0y = vy;
                vx = int.CreateTruncating(xy[2 * i]);
                vy = int.CreateTruncating(xy[(2 * i) + 1]);

                if ((v0y <= ipy && vy <= ipy) || (v0y > ipy && vy > ipy) || (v0x < ipx && vx < ipx))
                {
                    if (ipy == vy && (ipx == vx || (ipy == v0y && ((v0x <= ipx && ipx <= vx) || (vx <= ipx && ipx <= v0x)))))
                    {
                        return 0;
                    }

                    continue;
                }

                long dist = ((long)(ipy - v0y) * (vx - v0x)) - ((long)(ipx - v0x) * (vy - v0y));
                if (dist == 0)
                {
                    return 0;
                }

                if (vy < v0y)
                {
                    dist = -dist;
                }

                counter += dist > 0 ? 1 : 0;
            }

            return counter % 2 == 0 ? -1 : 1;
        }

        float px = pt.X, py = pt.Y;
        float fx = F(xy[2 * (total - 1)]);
        float fy = F(xy[(2 * total) - 1]);

        if (!measureDist)
        {
            for (int i = 0; i < total; i++)
            {
                float f0x = fx, f0y = fy;
                fx = F(xy[2 * i]);
                fy = F(xy[(2 * i) + 1]);

                if ((f0y <= py && fy <= py) || (f0y > py && fy > py) || (f0x < px && fx < px))
                {
                    if (py == fy && (px == fx || (py == f0y && ((f0x <= px && px <= fx) || (fx <= px && px <= f0x)))))
                    {
                        return 0;
                    }

                    continue;
                }

                double dist = ((double)(py - f0y) * (fx - f0x)) - ((double)(px - f0x) * (fy - f0y));
                if (dist == 0)
                {
                    return 0;
                }

                if (fy < f0y)
                {
                    dist = -dist;
                }

                counter += dist > 0 ? 1 : 0;
            }

            return counter % 2 == 0 ? -1 : 1;
        }

        double minDistNum = float.MaxValue, minDistDenom = 1;
        for (int i = 0; i < total; i++)
        {
            float f0x = fx, f0y = fy;
            fx = F(xy[2 * i]);
            fy = F(xy[(2 * i) + 1]);

            double dx = fx - f0x, dy = fy - f0y;
            double dx1 = px - f0x, dy1 = py - f0y;
            double dx2 = px - fx, dy2 = py - fy;
            double distNum, distDenom = 1;

            if ((dx1 * dx) + (dy1 * dy) <= 0)
            {
                distNum = (dx1 * dx1) + (dy1 * dy1);
            }
            else if ((dx2 * dx) + (dy2 * dy) >= 0)
            {
                distNum = (dx2 * dx2) + (dy2 * dy2);
            }
            else
            {
                distNum = (dy1 * dx) - (dx1 * dy);
                distNum *= distNum;
                distDenom = (dx * dx) + (dy * dy);
            }

            if (distNum * minDistDenom < minDistNum * distDenom)
            {
                minDistNum = distNum;
                minDistDenom = distDenom;
                if (minDistNum == 0)
                {
                    break;
                }
            }

            if ((f0y <= py && fy <= py) || (f0y > py && fy > py) || (f0x < px && fx < px))
            {
                continue;
            }

            distNum = (dy1 * dx) - (dx1 * dy);
            if (dy < 0)
            {
                distNum = -distNum;
            }

            counter += distNum > 0 ? 1 : 0;
        }

        double result = Math.Sqrt(minDistNum / minDistDenom);
        return counter % 2 == 0 ? -result : result;
    }

    private static float F<T>(T v)
        where T : unmanaged, INumber<T> => float.CreateTruncating(v);
}
