// This implementation is ported from the OpenCV sub-pixel corner refinement.
//
// Original sources:
//   modules/imgproc/src/cornersubpix.cpp (cornerSubPix)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>OpenCV <c>cornerSubPix</c>.</summary>
internal static class SubPixRefiner
{
    private const int MaxIters = 100;
    private const double DblEpsilon = 2.220446049250313e-16;

    public static void Refine<T>(Mat<T> src, Span<Point2f> corners, Size win, Size zeroZone, TermCriteria criteria)
        where T : unmanaged
    {
        int winW = (win.Width * 2) + 1, winH = (win.Height * 2) + 1;
        int maxIters = (criteria.Type & TermCriteriaType.MaxIter) != 0
            ? Math.Min(Math.Max(criteria.MaxCount, 1), MaxIters)
            : MaxIters;
        double eps = (criteria.Type & TermCriteriaType.Eps) != 0 ? Math.Max(criteria.Epsilon, 0.0) : 0;
        eps *= eps;

        if (corners.Length == 0)
        {
            return;
        }

        if (win.Width <= 0 || win.Height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(win), "The window half size must be positive.");
        }

        if (src.Width < (win.Width * 2) + 5 || src.Height < (win.Height * 2) + 5)
        {
            throw new ArgumentOutOfRangeException(nameof(win), "The image is smaller than 2*winSize + 5.");
        }

        float[] mask = ArrayPool<float>.Shared.Rent(winW * winH);
        float[] buf = ArrayPool<float>.Shared.Rent((winW + 2) * (winH + 2));
        try
        {
            RefineAll(src, corners, win, zeroZone, maxIters, eps, mask, buf);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buf);
            ArrayPool<float>.Shared.Return(mask);
        }
    }

    private static void RefineAll<T>(Mat<T> src, Span<Point2f> corners, Size win, Size zeroZone, int maxIters, double eps, float[] mask, float[] buf)
        where T : unmanaged
    {
        int winW = (win.Width * 2) + 1, winH = (win.Height * 2) + 1;
        for (int i = 0; i < winH; i++)
        {
            float y = (float)(i - win.Height) / win.Height;
            float vy = MathF.Exp(-y * y);
            for (int j = 0; j < winW; j++)
            {
                float x = (float)(j - win.Width) / win.Width;
                mask[(i * winW) + j] = vy * MathF.Exp(-x * x);
            }
        }

        if (zeroZone.Width >= 0 && zeroZone.Height >= 0 &&
            (zeroZone.Width * 2) + 1 < winW && (zeroZone.Height * 2) + 1 < winH)
        {
            for (int i = win.Height - zeroZone.Height; i <= win.Height + zeroZone.Height; i++)
            {
                for (int j = win.Width - zeroZone.Width; j <= win.Width + zeroZone.Width; j++)
                {
                    mask[(i * winW) + j] = 0;
                }
            }
        }

        int bufW = winW + 2, bufH = winH + 2;
        int cols = src.Width, rows = src.Height;

        for (int pt = 0; pt < corners.Length; pt++)
        {
            Point2f cT = corners[pt], cI = cT;
            if (!Contains(cols, rows, cT))
            {
                throw new ArgumentOutOfRangeException(nameof(corners), "cornerSubPix: initial corner is outside the image.");
            }

            int iter = 0;
            double err;
            do
            {
                double a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;
                RectSubPix.Extract(src, bufW, bufH, cI, buf.AsSpan(0, bufW * bufH));

                int k = 0;
                for (int i = 0; i < winH; i++)
                {
                    int row = ((i + 1) * bufW) + 1;
                    double py = i - win.Height;
                    for (int j = 0; j < winW; j++, k++)
                    {
                        double m = mask[k];
                        double tgx = buf[row + j + 1] - buf[row + j - 1];
                        double tgy = buf[row + j + bufW] - buf[row + j - bufW];
                        double gxx = tgx * tgx * m;
                        double gxy = tgx * tgy * m;
                        double gyy = tgy * tgy * m;
                        double px = j - win.Width;

                        a += gxx;
                        b += gxy;
                        c += gyy;

                        bb1 += (gxx * px) + (gxy * py);
                        bb2 += (gxy * px) + (gyy * py);
                    }
                }

                double det = (a * c) - (b * b);
                if (Math.Abs(det) <= DblEpsilon * DblEpsilon)
                {
                    break;
                }

                double scale = 1.0 / det;
                var cI2 = new Point2f(
                    (float)(cI.X + (c * scale * bb1) - (b * scale * bb2)),
                    (float)(cI.Y - (b * scale * bb1) + (a * scale * bb2)));
                float ex = cI2.X - cI.X, ey = cI2.Y - cI.Y;
                err = (ex * ex) + (ey * ey);
                if (!Contains(cols, rows, cI2))
                {
                    break;
                }

                cI = cI2;
            }
            while (++iter < maxIters && err > eps);

            if (MathF.Abs(cI.X - cT.X) > win.Width || MathF.Abs(cI.Y - cT.Y) > win.Height)
            {
                cI = cT;
            }

            corners[pt] = cI;
        }
    }

    private static bool Contains(int cols, int rows, Point2f p) => 0 <= p.X && p.X < cols && 0 <= p.Y && p.Y < rows;
}
