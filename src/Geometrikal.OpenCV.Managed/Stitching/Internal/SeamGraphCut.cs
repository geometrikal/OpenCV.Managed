// This implementation is ported from the OpenCV graph-cut seam finder.
//
// Original sources:
//   modules/stitching/src/seam_finders.cpp (GraphCutSeamFinder::Impl)
//   modules/stitching/include/opencv2/stitching/detail/util_inl.hpp (normL2)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>OpenCV <c>GraphCutSeamFinder::Impl</c>.</summary>
internal static class SeamGraphCut
{
    private const int Gap = 10;
    private const float WeightEps = 1f;

    /// <summary>
    /// OpenCV <c>Impl::find</c> gradient step: the squared L2 norm of the 3-channel Sobel
    /// derivatives per pixel.
    /// </summary>
    public static (float[] Dx, float[] Dy) Gradients(Mat<float> src)
    {
        Mat<float> dx = Derivatives.Sobel<float, float>(src, 1, 0);
        Mat<float> dy = Derivatives.Sobel<float, float>(src, 0, 1);
        int w = src.Width;
        float[] ndx = GC.AllocateUninitializedArray<float>(w * src.Height);
        float[] ndy = GC.AllocateUninitializedArray<float>(w * src.Height);
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<float> rx = dx.GetReadOnlyRowSpan(y);
            ReadOnlySpan<float> ry = dy.GetReadOnlyRowSpan(y);
            Span<float> ox = ndx.AsSpan(y * w, w);
            Span<float> oy = ndy.AsSpan(y * w, w);
            for (int x = 0; x < w; x++)
            {
                int k = 3 * x;
                ox[x] = (rx[k] * rx[k]) + (rx[k + 1] * rx[k + 1]) + (rx[k + 2] * rx[k + 2]);
                oy[x] = (ry[k] * ry[k]) + (ry[k + 1] * ry[k + 1]) + (ry[k + 2] * ry[k + 2]);
            }
        }

        return (ndx, ndy);
    }

    /// <summary>OpenCV <c>Impl::findInPair</c>.</summary>
    public static void FindInPair(
        Mat<float> img1,
        Mat<float> img2,
        float[] dx1,
        float[] dy1,
        float[] dx2,
        float[] dy2,
        Mat<byte> mask1,
        Mat<byte> mask2,
        Point tl1,
        Point tl2,
        Rect roi,
        GraphCutCostType costType,
        float terminalCost,
        float badRegionPenalty)
    {
        int w = roi.Width + (2 * Gap);
        int h = roi.Height + (2 * Gap);
        int n = w * h;
        float[] diff = ArrayPool<float>.Shared.Rent(n);
        byte[] m1 = ArrayPool<byte>.Shared.Rent(n);
        byte[] m2 = ArrayPool<byte>.Shared.Rent(n);
        bool grad = costType == GraphCutCostType.ColorGrad;
        float[] sdx1 = grad ? ArrayPool<float>.Shared.Rent(n) : [];
        float[] sdy1 = grad ? ArrayPool<float>.Shared.Rent(n) : [];
        float[] sdx2 = grad ? ArrayPool<float>.Shared.Rent(n) : [];
        float[] sdy2 = grad ? ArrayPool<float>.Shared.Rent(n) : [];
        try
        {
            int ox1 = roi.X - tl1.X - Gap, oy1 = roi.Y - tl1.Y - Gap;
            int ox2 = roi.X - tl2.X - Gap, oy2 = roi.Y - tl2.Y - Gap;
            SeamRoi.CutWindow(mask1, ox1, oy1, m1, w, h);
            SeamRoi.CutWindow(mask2, ox2, oy2, m2, w, h);
            ColorDiff(img1, img2, ox1, oy1, ox2, oy2, diff, w, h);
            if (grad)
            {
                CutGradient(dx1, img1.Width, img1.Height, ox1, oy1, sdx1, w, h);
                CutGradient(dy1, img1.Width, img1.Height, ox1, oy1, sdy1, w, h);
                CutGradient(dx2, img2.Width, img2.Height, ox2, oy2, sdx2, w, h);
                CutGradient(dy2, img2.Width, img2.Height, ox2, oy2, sdy2, w, h);
            }

            int edgeCount = ((h - 1) * w) + ((w - 1) * h);
            using var graph = new GcGraph(n, edgeCount);
            for (int i = 0; i < n; i++)
            {
                int v = graph.AddVtx();
                graph.AddTermWeights(v, m1[i] != 0 ? terminalCost : 0f, m2[i] != 0 ? terminalCost : 0f);
            }

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int v = (y * w) + x;
                    if (x < w - 1)
                    {
                        float weight = grad
                            ? ((diff[v] + diff[v + 1]) / ((((sdx1[v] + sdx1[v + 1]) + sdx2[v]) + sdx2[v + 1]) + WeightEps)) + WeightEps
                            : (diff[v] + diff[v + 1]) + WeightEps;
                        if (m1[v] == 0 || m1[v + 1] == 0 || m2[v] == 0 || m2[v + 1] == 0)
                        {
                            weight += badRegionPenalty;
                        }

                        graph.AddEdges(v, v + 1, weight, weight);
                    }

                    if (y < h - 1)
                    {
                        int b = v + w;
                        float weight = grad
                            ? ((diff[v] + diff[b]) / ((((sdy1[v] + sdy1[b]) + sdy2[v]) + sdy2[b]) + WeightEps)) + WeightEps
                            : (diff[v] + diff[b]) + WeightEps;
                        if (m1[v] == 0 || m1[b] == 0 || m2[v] == 0 || m2[b] == 0)
                        {
                            weight += badRegionPenalty;
                        }

                        graph.AddEdges(v, b, weight, weight);
                    }
                }
            }

            graph.MaxFlow();

            int x1 = roi.X - tl1.X, y1 = roi.Y - tl1.Y;
            int x2 = roi.X - tl2.X, y2 = roi.Y - tl2.Y;
            for (int y = 0; y < roi.Height; y++)
            {
                Span<byte> row1 = mask1.GetRowSpan(y1 + y).Slice(x1, roi.Width);
                Span<byte> row2 = mask2.GetRowSpan(y2 + y).Slice(x2, roi.Width);
                int o = ((y + Gap) * w) + Gap;
                for (int x = 0; x < roi.Width; x++)
                {
                    if (graph.InSourceSegment(o + x))
                    {
                        if (row1[x] != 0)
                        {
                            row2[x] = 0;
                        }
                    }
                    else if (row2[x] != 0)
                    {
                        row1[x] = 0;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(diff);
            ArrayPool<byte>.Shared.Return(m1);
            ArrayPool<byte>.Shared.Return(m2);
            if (grad)
            {
                ArrayPool<float>.Shared.Return(sdx1);
                ArrayPool<float>.Shared.Return(sdy1);
                ArrayPool<float>.Shared.Return(sdx2);
                ArrayPool<float>.Shared.Return(sdy2);
            }
        }
    }

    /// <summary>
    /// Writes <c>normL2(subimg1(p), subimg2(p))</c> per window pixel; pixels outside an image are
    /// (0, 0, 0) as in OpenCV.
    /// </summary>
    private static void ColorDiff(Mat<float> img1, Mat<float> img2, int ox1, int oy1, int ox2, int oy2, Span<float> diff, int w, int h)
    {
        for (int y = 0; y < h; y++)
        {
            int sy1 = oy1 + y, sy2 = oy2 + y;
            bool in1 = (uint)sy1 < (uint)img1.Height;
            bool in2 = (uint)sy2 < (uint)img2.Height;
            ReadOnlySpan<float> r1 = in1 ? img1.GetReadOnlyRowSpan(sy1) : default;
            ReadOnlySpan<float> r2 = in2 ? img2.GetReadOnlyRowSpan(sy2) : default;
            Span<float> d = diff.Slice(y * w, w);
            for (int x = 0; x < w; x++)
            {
                int sx1 = ox1 + x, sx2 = ox2 + x;
                float a0 = 0f, a1 = 0f, a2 = 0f, b0 = 0f, b1 = 0f, b2 = 0f;
                if (in1 && (uint)sx1 < (uint)img1.Width)
                {
                    int k = 3 * sx1;
                    a0 = r1[k];
                    a1 = r1[k + 1];
                    a2 = r1[k + 2];
                }

                if (in2 && (uint)sx2 < (uint)img2.Width)
                {
                    int k = 3 * sx2;
                    b0 = r2[k];
                    b1 = r2[k + 1];
                    b2 = r2[k + 2];
                }

                float e0 = a0 - b0, e1 = a1 - b1, e2 = a2 - b2;
                d[x] = (e0 * e0) + (e1 * e1) + (e2 * e2);
            }
        }
    }

    private static void CutGradient(float[] g, int gw, int gh, int ox, int oy, Span<float> dst, int w, int h)
    {
        dst[..(w * h)].Clear();
        int xs = Math.Max(0, -ox);
        int xe = Math.Min(w, gw - ox);
        if (xs >= xe)
        {
            return;
        }

        int ys = Math.Max(0, -oy);
        int ye = Math.Min(h, gh - oy);
        for (int y = ys; y < ye; y++)
        {
            g.AsSpan(((oy + y) * gw) + ox + xs, xe - xs).CopyTo(dst.Slice((y * w) + xs, xe - xs));
        }
    }
}
