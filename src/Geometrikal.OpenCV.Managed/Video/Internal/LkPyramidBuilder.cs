// This implementation is ported from the OpenCV sparse optical flow module.
//
// Original sources:
//   modules/video/src/lkpyramid.cpp (buildOpticalFlowPyramid)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Video.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>OpenCV <c>buildOpticalFlowPyramid</c> with padded level buffers.</summary>
internal static class LkPyramidBuilder
{
    public static OpticalFlowPyramid Build(
        Mat<byte> img,
        Size winSize,
        int maxLevel,
        bool withDerivatives,
        BorderType pyrBorder,
        BorderType derivBorder,
        bool tryReuseInputImage,
        List<IDisposable>? rentals = null)
    {
        int ww = winSize.Width, wh = winSize.Height, cn = img.Channels;
        var images = new List<Mat<byte>>();
        var derivs = withDerivatives ? new List<Mat<short>>() : null;

        Mat<byte>? level0 = null;
        if (tryReuseInputImage && LkBorders.IsSubmatrix(img))
        {
            (Size whole, Point ofs) = LkBorders.LocateRoi(img);
            if (ofs.X >= ww && ofs.Y >= wh && ofs.X + img.Width + ww <= whole.Width && ofs.Y + img.Height + wh <= whole.Height)
            {
                level0 = img;
            }
        }

        if (level0 == null)
        {
            var temp = NewBuffer<byte>(img.Width + (2 * ww), img.Height + (2 * wh), cn, pyrBorder == BorderType.Transparent, rentals);
            if (pyrBorder == BorderType.Transparent)
            {
                img.CopyTo(temp.Roi(ww, wh, img.Width, img.Height));
            }
            else
            {
                LkBorders.CopyMakeBorder(img, temp, wh, ww, pyrBorder);
            }

            level0 = temp.Roi(ww, wh, img.Width, img.Height);
        }

        var sz = new Size(img.Width, img.Height);
        Mat<byte> prevLevel = level0;
        Mat<byte> thisLevel = level0;
        for (int level = 0; level <= maxLevel; level++)
        {
            if (level != 0)
            {
                var temp = NewBuffer<byte>(sz.Width + (2 * ww), sz.Height + (2 * wh), cn, pyrBorder == BorderType.Transparent, rentals);
                thisLevel = temp.Roi(ww, wh, sz.Width, sz.Height);
                Pyramids.PyrDown(prevLevel, thisLevel);
                if (pyrBorder != BorderType.Transparent)
                {
                    LkBorders.FillBorder(temp, wh, ww, sz.Width, sz.Height, pyrBorder);
                }
            }

            images.Add(thisLevel);

            if (derivs != null)
            {
                derivs.Add(Derivative(thisLevel, winSize, derivBorder, rentals));
            }

            sz = new Size((sz.Width + 1) / 2, (sz.Height + 1) / 2);
            if (sz.Width <= ww || sz.Height <= wh)
            {
                break;
            }

            prevLevel = thisLevel;
        }

        return new OpticalFlowPyramid(winSize, images.ToArray(), derivs?.ToArray());
    }

    /// <summary>
    /// Scharr derivatives of <paramref name="level"/> in a buffer padded by the window size,
    /// with <paramref name="border"/> in the padding; returns the unpadded view.
    /// </summary>
    public static Mat<short> Derivative(Mat<byte> level, Size winSize, BorderType border, List<IDisposable>? rentals = null)
    {
        int ww = winSize.Width, wh = winSize.Height;
        var deriv = NewBuffer<short>(level.Width + (2 * ww), level.Height + (2 * wh), level.Channels * 2, border == BorderType.Transparent, rentals);
        Mat<short> derivI = deriv.Roi(ww, wh, level.Width, level.Height);
        LkScharrDeriv.Run(level, derivI);
        if (border != BorderType.Transparent)
        {
            LkBorders.FillBorder(deriv, wh, ww, level.Width, level.Height, border);
        }

        return derivI;
    }

    /// <summary>
    /// Allocates a packed buffer, or rents it and adds the rental to <paramref name="rentals"/>.
    /// A rented buffer has a larger backing array, so <see cref="LkBorders.LocateRoi"/> does not apply to it.
    /// </summary>
    private static Mat<T> NewBuffer<T>(int width, int height, int channels, bool clear, List<IDisposable>? rentals)
        where T : unmanaged
    {
        if (rentals == null)
        {
            return new Mat<T>(width, height, channels, clear: clear);
        }

        var pooled = PooledMat<T>.Rent(width, height, channels);
        rentals.Add(pooled);
        if (clear)
        {
            pooled.Mat.DangerousGetArray().AsSpan(0, width * height * channels).Clear();
        }

        return pooled.Mat;
    }
}
