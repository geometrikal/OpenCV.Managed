// This implementation is ported from the OpenCV Voronoi seam finder.
//
// Original sources:
//   modules/stitching/src/seam_finders.cpp (VoronoiSeamFinder::findInPair)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>OpenCV <c>VoronoiSeamFinder::findInPair</c>.</summary>
internal static class SeamVoronoi
{
    private const int Gap = 10;

    public static void FindInPair(Mat<byte> mask1, Mat<byte> mask2, Point tl1, Point tl2, Rect roi)
    {
        int w = roi.Width + (2 * Gap);
        int h = roi.Height + (2 * Gap);
        using PooledMat<byte> sub1 = PooledMat<byte>.Rent(w, h, 1);
        using PooledMat<byte> sub2 = PooledMat<byte>.Rent(w, h, 1);
        using PooledMat<float> dist1 = PooledMat<float>.Rent(w, h, 1);
        using PooledMat<float> dist2 = PooledMat<float>.Rent(w, h, 1);

        Span<byte> s1 = sub1.Mat.GetSpan();
        Span<byte> s2 = sub2.Mat.GetSpan();
        SeamRoi.CutWindow(mask1, roi.X - tl1.X - Gap, roi.Y - tl1.Y - Gap, s1, w, h);
        SeamRoi.CutWindow(mask2, roi.X - tl2.X - Gap, roi.Y - tl2.Y - Gap, s2, w, h);

        // The sources of the distance transforms are (unique1 == 0) and (unique2 == 0).
        int n = w * h;
        for (int i = 0; i < n; i++)
        {
            bool a = s1[i] != 0, b = s2[i] != 0;
            s1[i] = a && !b ? (byte)0 : (byte)255;
            s2[i] = b && !a ? (byte)0 : (byte)255;
        }

        DistanceTransforms.DistanceTransform(sub1.Mat, dist1.Mat, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        DistanceTransforms.DistanceTransform(sub2.Mat, dist2.Mat, DistanceTypes.L1, DistanceTransformMasks.Mask3);

        ReadOnlySpan<float> d1 = dist1.Mat.GetReadOnlySpan();
        ReadOnlySpan<float> d2 = dist2.Mat.GetReadOnlySpan();
        int x1 = roi.X - tl1.X, y1 = roi.Y - tl1.Y;
        int x2 = roi.X - tl2.X, y2 = roi.Y - tl2.Y;
        for (int y = 0; y < roi.Height; y++)
        {
            Span<byte> row1 = mask1.GetRowSpan(y1 + y).Slice(x1, roi.Width);
            Span<byte> row2 = mask2.GetRowSpan(y2 + y).Slice(x2, roi.Width);
            int o = ((y + Gap) * w) + Gap;
            for (int x = 0; x < roi.Width; x++)
            {
                if (d1[o + x] < d2[o + x])
                {
                    row2[x] = 0;
                }
                else
                {
                    row1[x] = 0;
                }
            }
        }
    }
}
