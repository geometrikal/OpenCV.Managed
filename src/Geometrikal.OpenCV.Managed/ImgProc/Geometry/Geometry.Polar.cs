// This implementation is ported from the OpenCV warpPolar.
//
// Original source:
//   modules/imgproc/src/imgwarp.cpp (cv::warpPolar)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Radius scale of <see cref="Geometry.WarpPolar{T}"/>, as OpenCV <c>WarpPolarMode</c>.</summary>
public enum WarpPolarMode
{
    /// <summary><c>WARP_POLAR_LINEAR</c>: ρ = Kmag·column.</summary>
    Linear = 0,

    /// <summary><c>WARP_POLAR_LOG</c>: ρ = exp(Kmag·column) − 1 (semi-log polar).</summary>
    Log = 256,
}

/// <content>Polar remapping.</content>
public static partial class Geometry
{
    /// <summary>
    /// Remaps an image to polar or semi-log polar coordinates, or back, as OpenCV <c>warpPolar</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>Forward.</b> Destination column ρ and row φ sample the source at
    /// <c>center + r(ρ)·(cos, sin)(2π·φ/height)</c>, with <c>r = maxRadius·ρ/width</c> (linear) or
    /// <c>r = exp(ln(maxRadius)·ρ/width) − 1</c> (log). The maps are computed in <c>double</c> and
    /// stored as <c>float</c>.</para>
    /// <para><b>Inverse</b> (<see cref="WarpFlags.InverseMap"/>). The source is a polar image. It is
    /// extended by one wrapped row at the top and bottom; each destination pixel samples the radius
    /// and angle from <see cref="MathFunctions.CartToPolar{T}"/> (log: <c>ln(r + 1)</c>).</para>
    /// <para><b>Size.</b> When both <paramref name="dsize"/> sides are not positive, the size is
    /// <c>(cvRound(maxRadius), cvRound(maxRadius·π))</c>; when only the height is not positive, it is
    /// <c>cvRound(width·π)</c>.</para>
    /// <para><b>Outliers.</b> <see cref="WarpFlags.FillOutliers"/> sets pixels that map outside the
    /// source to zero (<see cref="BorderType.Constant"/>). Without it the remap uses
    /// <see cref="BorderType.Transparent"/>, and those pixels keep the zero of the new destination;
    /// OpenCV leaves them uninitialised.</para>
    /// </remarks>
    /// <param name="source">Source image.</param>
    /// <param name="dsize">Destination size (see remarks).</param>
    /// <param name="center">Centre of the polar transform in the Cartesian image.</param>
    /// <param name="maxRadius">Radius of the bounding circle; also the inverse radius scale.</param>
    /// <param name="interpolation">Remap interpolation.</param>
    /// <param name="mode">Linear or semi-log polar radius.</param>
    /// <param name="flags"><see cref="WarpFlags.InverseMap"/> and <see cref="WarpFlags.FillOutliers"/>.</param>
    /// <returns>A new image of size <paramref name="dsize"/>.</returns>
    public static Mat<T> WarpPolar<T>(
        Mat<T> source,
        Size dsize,
        Point2f center,
        double maxRadius,
        InterpolationFlags interpolation,
        WarpPolarMode mode = WarpPolarMode.Linear,
        WarpFlags flags = WarpFlags.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        if (dsize.Width <= 0 && dsize.Height <= 0)
        {
            dsize = new Size(ScalarConverter.CvRound(maxRadius), ScalarConverter.CvRound(maxRadius * Math.PI));
        }
        else if (dsize.Height <= 0)
        {
            dsize = new Size(dsize.Width, ScalarConverter.CvRound(dsize.Width * Math.PI));
        }

        if (dsize.Width <= 0 || dsize.Height <= 0)
        {
            throw new ArgumentException("The destination size is empty.", nameof(dsize));
        }

        bool semiLog = mode == WarpPolarMode.Log;
        var border = (flags & WarpFlags.FillOutliers) != 0 ? BorderType.Constant : BorderType.Transparent;
        if ((flags & WarpFlags.InverseMap) == 0)
        {
            var (mapX, mapY) = ForwardPolarMaps(dsize, center, maxRadius, semiLog);
            return Remap(source, mapX, mapY, interpolation, border);
        }

        const int angleBorder = 1;
        if (source.Height == 0 || source.Width == 0)
        {
            throw new ArgumentException("The source is empty.", nameof(source));
        }

        var padded = Borders.CopyMakeBorder(source, angleBorder, angleBorder, 0, 0, BorderType.Wrap);
        var (invX, invY) = InversePolarMaps(dsize, center, maxRadius, semiLog, source.Width, source.Height, angleBorder);
        return Remap(padded, invX, invY, interpolation, border);
    }

    internal static (Mat<float> MapX, Mat<float> MapY) ForwardPolarMaps(Size dsize, Point2f center, double maxRadius, bool semiLog)
    {
        var mapX = new Mat<float>(dsize.Width, dsize.Height, clear: false);
        var mapY = new Mat<float>(dsize.Width, dsize.Height, clear: false);
        double kAngle = 2 * Math.PI / dsize.Height;
        var rhos = new float[dsize.Width];
        if (semiLog)
        {
            double kMag = Math.Log(maxRadius) / dsize.Width;
            for (int rho = 0; rho < rhos.Length; rho++)
            {
                rhos[rho] = (float)(Math.Exp(rho * kMag) - 1.0);
            }
        }
        else
        {
            double kMag = maxRadius / dsize.Width;
            for (int rho = 0; rho < rhos.Length; rho++)
            {
                rhos[rho] = (float)(rho * kMag);
            }
        }

        for (int phi = 0; phi < dsize.Height; phi++)
        {
            double angle = kAngle * phi;
            double cp = Math.Cos(angle), sp = Math.Sin(angle);
            var mx = mapX.GetRowSpan(phi);
            var my = mapY.GetRowSpan(phi);
            for (int rho = 0; rho < rhos.Length; rho++)
            {
                mx[rho] = (float)((rhos[rho] * cp) + center.X);
                my[rho] = (float)((rhos[rho] * sp) + center.Y);
            }
        }

        return (mapX, mapY);
    }

    internal static (Mat<float> MapX, Mat<float> MapY) InversePolarMaps(
        Size dsize, Point2f center, double maxRadius, bool semiLog, int polarWidth, int polarHeight, int angleBorder)
    {
        var mapX = new Mat<float>(dsize.Width, dsize.Height, clear: false);
        var mapY = new Mat<float>(dsize.Width, dsize.Height, clear: false);
        double kAngle = 2 * Math.PI / polarHeight;
        double kMag = semiLog ? Math.Log(maxRadius) / polarWidth : maxRadius / polarWidth;
        int width = dsize.Width;
        var bufX = new float[width];
        for (int x = 0; x < width; x++)
        {
            bufX[x] = x - center.X;
        }

        ParallelRows.For(dsize.Height, width, (start, end) =>
        {
            var bufY = new float[width];
            var rho = new float[width];
            var angle = new float[width];
            for (int y = start; y < end; y++)
            {
                Array.Fill(bufY, y - center.Y);

                // cartToPolar on a 1 × width matrix: blocks of 1024 elements.
                for (int j = 0; j < width; j += 1024)
                {
                    int len = Math.Min(1024, width - j);
                    MathKernels.CartToPolar32f(bufX.AsSpan(j, len), bufY.AsSpan(j, len), rho.AsSpan(j, len), angle.AsSpan(j, len), false);
                }

                if (semiLog)
                {
                    for (int x = 0; x < width; x++)
                    {
                        rho[x] = MathF.Log(rho[x] + 1f);
                    }
                }

                var mx = mapX.GetRowSpan(y);
                var my = mapY.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    mx[x] = (float)(rho[x] / kMag);
                    my[x] = (float)(angle[x] / kAngle) + angleBorder;
                }
            }
        });

        return (mapX, mapY);
    }
}
