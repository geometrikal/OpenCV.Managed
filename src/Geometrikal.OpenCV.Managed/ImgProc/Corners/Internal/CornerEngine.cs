// This implementation is ported from the OpenCV corner detectors.
//
// Original sources:
//   modules/imgproc/src/corner.cpp (cornerEigenValsVecs, calcMinEigenVal, calcHarris, eigen2x2,
//     calcEigenValsVecs)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Gradient covariance and per-pixel eigenvalue functions of OpenCV <c>corner.cpp</c>.
/// </summary>
internal static class CornerEngine
{
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>
    /// Box-summed gradient covariance <c>(dx·dx, dx·dy, dy·dy)</c> as a pooled 3-channel float
    /// matrix, as the first part of OpenCV <c>cornerEigenValsVecs</c>.
    /// </summary>
    public static PooledMat<float> Covariance<T>(Mat<T> source, int blockSize, int ksize, BorderType borderType)
        where T : unmanaged
    {
        double scale = (double)(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;
        if (ksize < 0)
        {
            scale *= 2.0;
        }

        if (typeof(T) == typeof(byte))
        {
            scale *= 255.0;
        }

        scale = 1.0 / scale;

        int w = source.Width, h = source.Height;
        using PooledMat<float> dxPool = PooledMat<float>.Rent(w, h, 1);
        using PooledMat<float> dyPool = PooledMat<float>.Rent(w, h, 1);
        using PooledMat<float> covPool = PooledMat<float>.Rent(w, h, 3);
        Mat<float> dx = dxPool.Mat, dy = dyPool.Mat, cov = covPool.Mat;
        int aperture = ksize > 0 ? ksize : 0;
        Derivatives.SobelInto(source, 1, 0, aperture, scale, 0, borderType, dx);
        Derivatives.SobelInto(source, 0, 1, aperture, scale, 0, borderType, dy);
        ParallelRows.For(h, (long)w * 3, (y0, y1) => CovRows(dx, dy, cov, y0, y1));
        PooledMat<float> sum = PooledMat<float>.Rent(w, h, 3);
        try
        {
            BoxFilterEngine.ApplyInto(cov, new Size(blockSize, blockSize), new Point(-1, -1), false, borderType, sum.Mat);
        }
        catch
        {
            sum.Dispose();
            throw;
        }

        return sum;
    }

    private static void CovRows(Mat<float> dx, Mat<float> dy, Mat<float> cov, int y0, int y1)
    {
        int w = dx.Width;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<float> gx = dx.GetReadOnlyRowSpan(y);
            ReadOnlySpan<float> gy = dy.GetReadOnlyRowSpan(y);
            Span<float> c = cov.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                float a = gx[x], b = gy[x];
                c[3 * x] = a * a;
                c[(3 * x) + 1] = a * b;
                c[(3 * x) + 2] = b * b;
            }
        }
    }

    /// <summary>OpenCV <c>calcMinEigenVal</c>. The vector and scalar OpenCV loops give equal results.</summary>
    public static void MinEigenVal(Mat<float> cov, Mat<float> dst)
    {
        int w = cov.Width, h = cov.Height;
        ParallelRows.For(h, (long)w * 4, (y0, y1) => MinEigenRows(cov, dst, y0, y1));
    }

    private static void MinEigenRows(Mat<float> cov, Mat<float> dst, int y0, int y1)
    {
        int w = cov.Width;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<float> c = cov.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                float a = c[3 * x] * 0.5f;
                float b = c[(3 * x) + 1];
                float cc = c[(3 * x) + 2] * 0.5f;
                float t = a - cc;
                d[x] = (a + cc) - MathF.Sqrt((t * t) + (b * b));
            }
        }
    }

    /// <summary>
    /// OpenCV <c>calcHarris</c> over the continuous matrix: the elements of the 4-lane vector
    /// loop use float arithmetic, and the last <c>(width·height) mod 4</c> elements use the scalar
    /// loop, which evaluates <c>k·(a + c)·(a + c)</c> in double.
    /// </summary>
    public static void Harris(Mat<float> cov, double k, Mat<float> dst)
    {
        int w = cov.Width, h = cov.Height;
        long total = (long)w * h;
        long vectorEnd = total - (total % 4);
        ParallelRows.For(h, (long)w * 4, (y0, y1) => HarrisRows(cov, dst, k, vectorEnd, y0, y1));
    }

    private static void HarrisRows(Mat<float> cov, Mat<float> dst, double k, long vectorEnd, int y0, int y1)
    {
        int w = cov.Width;
        float kf = (float)k;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<float> c = cov.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            long rowStart = (long)y * w;
            int split = (int)Math.Clamp(vectorEnd - rowStart, 0, w);
            for (int x = 0; x < split; x++)
            {
                float a = c[3 * x], b = c[(3 * x) + 1], cc = c[(3 * x) + 2];
                float acbb = (a * cc) - (b * b);
                float ac = a + cc;
                d[x] = acbb - ((kf * ac) * ac);
            }

            for (int x = split; x < w; x++)
            {
                float a = c[3 * x], b = c[(3 * x) + 1], cc = c[(3 * x) + 2];
                d[x] = (float)((a * cc) - (b * b) - (k * (a + cc) * (a + cc)));
            }
        }
    }

    /// <summary>OpenCV <c>calcEigenValsVecs</c> (<c>eigen2x2</c> in double).</summary>
    public static void EigenValsVecs(Mat<float> cov, Mat<float> dst)
    {
        int w = cov.Width, h = cov.Height;
        ParallelRows.For(h, (long)w * 32, (y0, y1) => EigenRows(cov, dst, y0, y1));
    }

    private static void EigenRows(Mat<float> cov, Mat<float> dst, int y0, int y1)
    {
        int w = cov.Width;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<float> c = cov.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                Eigen2x2(c.Slice(3 * x, 3), d.Slice(6 * x, 6));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Eigen2x2(ReadOnlySpan<float> cov, Span<float> dst)
    {
        double a = cov[0];
        double b = cov[1];
        double c = cov[2];

        double u = (a + c) * 0.5;
        double v = Math.Sqrt(((a - c) * (a - c) * 0.25) + (b * b));
        double l1 = u + v;
        double l2 = u - v;

        (double x, double y) = Vector(a, b, c, l1);
        double dd = 1.0 / Math.Sqrt((x * x) + (y * y) + DblEpsilon);
        dst[0] = (float)l1;
        dst[2] = (float)(x * dd);
        dst[3] = (float)(y * dd);

        (x, y) = Vector(a, b, c, l2);
        dd = 1.0 / Math.Sqrt((x * x) + (y * y) + DblEpsilon);
        dst[1] = (float)l2;
        dst[4] = (float)(x * dd);
        dst[5] = (float)(y * dd);
    }

    private static (double X, double Y) Vector(double a, double b, double c, double l)
    {
        double x = b;
        double y = l - a;
        double e = Math.Abs(x);

        if (e + Math.Abs(y) < 1e-4)
        {
            y = b;
            x = l - c;
            e = Math.Abs(x);
            if (e + Math.Abs(y) < 1e-4)
            {
                e = 1.0 / (e + Math.Abs(y) + 1.1920928955078125e-07);
                x *= e;
                y *= e;
            }
        }

        return (x, y);
    }
}
