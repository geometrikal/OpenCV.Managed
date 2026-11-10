// This implementation is ported from OpenCV's rotating calipers.
//
// Original source:
//   modules/geometry/src/rotcalipers.cpp (rotatingCalipers, minAreaRect)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/rotcalipers.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Minimum-area bounding rectangle with the rotating calipers of OpenCV 5.0.</summary>
internal static class CalipersMinAreaRect
{
    /// <summary>OpenCV <c>minAreaRect</c> for a counter-clockwise convex hull.</summary>
    public static RotatedRect FromHull(ReadOnlySpan<Point2f> hull)
    {
        int n = hull.Length;
        double angle = -Math.PI / 2;
        Point2f center = default;
        float width = 0, height = 0;

        if (n > 2)
        {
            Span<float> outBuf = stackalloc float[6];
            RotatingCalipers(hull, 1.0f, outBuf);
            float o0x = outBuf[0], o0y = outBuf[1], o1x = outBuf[2], o1y = outBuf[3], o2x = outBuf[4], o2y = outBuf[5];
            center = new Point2f(o0x + ((o1x + o2x) * 0.5f), o0y + ((o1y + o2y) * 0.5f));
            width = (float)Math.Sqrt(((double)o2x * o2x) + ((double)o2y * o2y));
            height = (float)Math.Sqrt(((double)o1x * o1x) + ((double)o1y * o1y));
            if (o1x == 0f && o1y > 0f)
            {
                (width, height) = (height, width);
            }
            else
            {
                angle = -Math.Atan2(o1x, o1y);
            }
        }
        else if (n == 2)
        {
            center = new Point2f((hull[0].X + hull[1].X) * 0.5f, (hull[0].Y + hull[1].Y) * 0.5f);
            double dx = hull[0].X - hull[1].X;
            double dy = hull[0].Y - hull[1].Y;
            width = 0;
            height = (float)Math.Sqrt((dx * dx) + (dy * dy));
            if (dx == 0)
            {
                (width, height) = (height, width);
            }
            else if (dy < 0)
            {
                angle = Math.Atan2(dy, dx);
                (width, height) = (height, width);
            }
            else if (dy > 0)
            {
                angle = -Math.Atan2(dx, dy);
            }
        }
        else if (n == 1)
        {
            center = hull[0];
        }

        return new RotatedRect(center, new Size2f(width, height), (float)(angle * 180 / Math.PI));
    }

    /// <summary>
    /// <c>rotatingCalipers</c> in <c>CALIPERS_MINAREARECT</c> mode. Writes the corner and the two
    /// side vectors to <paramref name="output"/> (six floats).
    /// </summary>
    private static void RotatingCalipers(ReadOnlySpan<Point2f> points, float orientation, Span<float> output)
    {
        int n = points.Length;
        float minarea = float.MaxValue;
        int bufLeft = 0, bufBottom = 0;
        float bufBaseA = 0, bufWidth = 0, bufBaseB = 0, bufHeight = 0;
        float[] rented = ArrayPool<float>.Shared.Rent(n * 3);
        try
        {
            Span<float> invVectLength = rented.AsSpan(0, n);
            Span<float> vect = rented.AsSpan(n, n * 2);
            int left = 0, bottom = 0, right = 0, top = 0;
            Span<int> seq = stackalloc int[4];

            float baseA;
            float baseB = 0;

            Point2f pt0 = points[0];
            float leftX = pt0.X, rightX = pt0.X;
            float topY = pt0.Y, bottomY = pt0.Y;

            for (int i = 0; i < n; i++)
            {
                if (pt0.X < leftX)
                {
                    leftX = pt0.X;
                    left = i;
                }

                if (pt0.X > rightX)
                {
                    rightX = pt0.X;
                    right = i;
                }

                if (pt0.Y > topY)
                {
                    topY = pt0.Y;
                    top = i;
                }

                if (pt0.Y < bottomY)
                {
                    bottomY = pt0.Y;
                    bottom = i;
                }

                Point2f pt = points[i + 1 < n ? i + 1 : 0];
                double dx = pt.X - pt0.X;
                double dy = pt.Y - pt0.Y;
                vect[2 * i] = (float)dx;
                vect[(2 * i) + 1] = (float)dy;
                invVectLength[i] = (float)(1.0 / Math.Sqrt((dx * dx) + (dy * dy)));
                pt0 = pt;
            }

            if (orientation == 0f)
            {
                double ax = vect[2 * (n - 1)];
                double ay = vect[(2 * (n - 1)) + 1];
                for (int i = 0; i < n; i++)
                {
                    double bx = vect[2 * i];
                    double by = vect[(2 * i) + 1];
                    double convexity = (ax * by) - (ay * bx);
                    if (convexity != 0)
                    {
                        orientation = convexity > 0 ? 1f : -1f;
                        break;
                    }

                    ax = bx;
                    ay = by;
                }

                if (orientation == 0f)
                {
                    throw new InvalidOperationException("The hull orientation is undefined.");
                }
            }

            baseA = orientation;

            seq[0] = bottom;
            seq[1] = right;
            seq[2] = top;
            seq[3] = left;

            Span<float> rot = stackalloc float[8];
            for (int k = 0; k < n; k++)
            {
                int mainElement = 0;

                rot[0] = vect[2 * seq[0]];
                rot[1] = vect[(2 * seq[0]) + 1];

                // rotate90CW
                rot[2] = vect[(2 * seq[1]) + 1];
                rot[3] = -vect[2 * seq[1]];

                // rotate180
                rot[4] = -vect[2 * seq[2]];
                rot[5] = -vect[(2 * seq[2]) + 1];

                // rotate90CCW
                rot[6] = -vect[(2 * seq[3]) + 1];
                rot[7] = vect[2 * seq[3]];

                for (int i = 1; i < 4; i++)
                {
                    // firstVecIsRight(rot[i], rot[main])
                    float tx = rot[(2 * i) + 1];
                    float ty = -rot[2 * i];
                    if ((tx * rot[2 * mainElement]) + (ty * rot[(2 * mainElement) + 1]) < 0)
                    {
                        mainElement = i;
                    }
                }

                int pindex = seq[mainElement];
                float leadX = vect[2 * pindex] * invVectLength[pindex];
                float leadY = vect[(2 * pindex) + 1] * invVectLength[pindex];
                switch (mainElement)
                {
                    case 0:
                        baseA = leadX;
                        baseB = leadY;
                        break;
                    case 1:
                        baseA = leadY;
                        baseB = -leadX;
                        break;
                    case 2:
                        baseA = -leadX;
                        baseB = -leadY;
                        break;
                    default:
                        baseA = -leadY;
                        baseB = leadX;
                        break;
                }

                seq[mainElement] += 1;
                seq[mainElement] = seq[mainElement] == n ? 0 : seq[mainElement];

                float wdx = points[seq[1]].X - points[seq[3]].X;
                float wdy = points[seq[1]].Y - points[seq[3]].Y;
                float width = (wdx * baseA) + (wdy * baseB);

                float hdx = points[seq[2]].X - points[seq[0]].X;
                float hdy = points[seq[2]].Y - points[seq[0]].Y;
                float height = (-hdx * baseB) + (hdy * baseA);

                float area = width * height;
                if (area <= minarea)
                {
                    minarea = area;
                    bufLeft = seq[3];
                    bufBaseA = baseA;
                    bufWidth = width;
                    bufBaseB = baseB;
                    bufHeight = height;
                    bufBottom = seq[0];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }

        float a1 = bufBaseA;
        float b1 = bufBaseB;
        float a2 = -bufBaseB;
        float b2 = bufBaseA;

        float c1 = (a1 * points[bufLeft].X) + (points[bufLeft].Y * b1);
        float c2 = (a2 * points[bufBottom].X) + (points[bufBottom].Y * b2);

        float idet = 1.0f / ((a1 * b2) - (a2 * b1));

        output[0] = ((c1 * b2) - (c2 * b1)) * idet;
        output[1] = ((a1 * c2) - (a2 * c1)) * idet;
        output[2] = a1 * bufWidth;
        output[3] = b1 * bufWidth;
        output[4] = a2 * bufHeight;
        output[5] = b2 * bufHeight;
    }
}
