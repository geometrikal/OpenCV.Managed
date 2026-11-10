// This implementation is ported from the OpenCV VNG demosaicing.
//
// Original source:
//   modules/imgproc/src/demosaicing.cpp (Bayer2RGB_VNG_8u)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <content>Variable number of gradients (VNG) demosaicing.</content>
public static partial class Demosaicing
{
    private static readonly float[] VngScalarScale = [0f, 0.5f, 0.25f, 0.1666666666667f, 0.125f, 0.1f, 0.08333333333f, 0.0714286f, 0.0625f];

    // The SSE kernel multiplies by 0.5f / ng; it differs from the scalar table for ng = 7.
    private static readonly float[] VngVectorScale = [0f, 0.5f / 1, 0.5f / 2, 0.5f / 3, 0.5f / 4, 0.5f / 5, 0.5f / 6, 0.5f / 7, 0.5f / 8];

    private static void BayerToBgrVng(Mat<byte> source, Mat<byte> dst, BayerPattern pattern)
    {
        if (Math.Min(source.Width, source.Height) < 8)
        {
            BayerToBgr(source, dst, pattern);
            return;
        }

        var padded = Borders.CopyMakeBorder(source, 2, 2, 2, 2, BorderType.Reflect101);
        int n = padded.Width, height = padded.Height;
        int blueIdx = pattern is BayerPattern.BG or BayerPattern.GB ? 0 : 2;
        bool greenCell0 = pattern is not (BayerPattern.BG or BayerPattern.RG);

        // Three rolling rows of seven gradient planes, each of n columns.
        const int brows = 3, planes = 7;
        int bufStep = n * planes;
        var buf = new ushort[bufStep * brows];

        for (int y = 2; y < height - 2; y++)
        {
            for (int dy = y == 2 ? -1 : 1; dy <= 1; dy++)
            {
                VngGradientRow(padded, y + dy, buf.AsSpan(((y + dy - 1) % brows) * bufStep, bufStep), n);
            }

            VngPixelRow(padded, y, buf, bufStep, n, dst.GetRowSpan(y - 2), greenCell0, blueIdx);
            greenCell0 = !greenCell0;
            blueIdx ^= 2;
        }
    }

    private static void VngGradientRow(Mat<byte> padded, int row, Span<ushort> brow, int n)
    {
        var up = padded.GetReadOnlyRowSpan(row - 1);
        var mid = padded.GetReadOnlyRowSpan(row);
        var down = padded.GetReadOnlyRowSpan(row + 1);
        for (int p = 0; p < 7; p++)
        {
            brow[n * p] = 0;
            brow[(n * p) + n - 1] = 0;
        }

        for (int c = 1; c < n - 1; c++)
        {
            int nw = up[c - 1], nn = up[c], ne = up[c + 1];
            int w = mid[c - 1], e = mid[c + 1];
            int sw = down[c - 1], s = down[c], se = down[c + 1];
            int b2 = Math.Abs(ne - sw) * 2;
            int b3 = Math.Abs(nw - se) * 2;
            brow[c] = (ushort)(Math.Abs(nw - sw) + (Math.Abs(nn - s) * 2) + Math.Abs(ne - se));
            brow[n + c] = (ushort)(Math.Abs(nw - ne) + (Math.Abs(w - e) * 2) + Math.Abs(sw - se));
            brow[(2 * n) + c] = (ushort)b2;
            brow[(3 * n) + c] = (ushort)b3;
            brow[(4 * n) + c] = (ushort)(b2 + Math.Abs(nn - w) + Math.Abs(s - e));
            brow[(5 * n) + c] = (ushort)(b3 + Math.Abs(nn - e) + Math.Abs(s - w));
            brow[(6 * n) + c] = (ushort)((nn + w + e + s) >> 1);
        }
    }

    private static void VngPixelRow(Mat<byte> padded, int y, ushort[] buf, int bufStep, int n, Span<byte> dstRow, bool greenCell, int blueIdx)
    {
        const int brows = 3;
        ReadOnlySpan<ushort> b0 = buf.AsSpan(((y - 2) % brows) * bufStep, bufStep);
        ReadOnlySpan<ushort> b1 = buf.AsSpan(((y - 1) % brows) * bufStep, bufStep);
        ReadOnlySpan<ushort> b2 = buf.AsSpan((y % brows) * bufStep, bufStep);
        var r2u = padded.GetReadOnlyRowSpan(y - 2);
        var r1u = padded.GetReadOnlyRowSpan(y - 1);
        var r0 = padded.GetReadOnlyRowSpan(y);
        var r1d = padded.GetReadOnlyRowSpan(y + 1);
        var r2d = padded.GetReadOnlyRowSpan(y + 2);

        // OpenCV SSE layout: one scalar pixel when the row starts on green, then blocks of 8 while
        // i <= n - 10, then scalar pixels.
        int i = 2;
        int vectorStart = greenCell ? Math.Min(3, n - 2) : 2;
        int vectorEnd = vectorStart;
        while (vectorEnd <= n - 10)
        {
            vectorEnd += 8;
        }

        for (; i < n - 2; i++)
        {
            float[] scale = i >= vectorStart && i < vectorEnd ? VngVectorScale : VngScalarScale;
            int c = i;
            int gradN = b0[c] + b1[c];
            int gradS = b1[c] + b2[c];
            int gradW = b1[n + c - 1] + b1[n + c];
            int gradE = b1[n + c] + b1[n + c + 1];
            int minGrad = Math.Min(Math.Min(Math.Min(gradN, gradS), gradW), gradE);
            int maxGrad = Math.Max(Math.Max(Math.Max(gradN, gradS), gradW), gradE);
            int r, g, b;
            int n2 = 2 * n, n3 = 3 * n, n4 = 4 * n, n5 = 5 * n, n6 = 6 * n;
            if (!greenCell)
            {
                int gradNE = b0[n4 + c + 1] + b1[n4 + c];
                int gradSW = b1[n4 + c] + b2[n4 + c - 1];
                int gradNW = b0[n5 + c - 1] + b1[n5 + c];
                int gradSE = b1[n5 + c] + b2[n5 + c + 1];
                minGrad = Math.Min(Math.Min(Math.Min(Math.Min(minGrad, gradNE), gradSW), gradNW), gradSE);
                maxGrad = Math.Max(Math.Max(Math.Max(Math.Max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                int t = minGrad + Math.Max(maxGrad / 2, 1);

                int rs = 0, gs = 0, bs = 0, ng = 0;
                if (gradN < t)
                {
                    rs += r2u[c] + r0[c];
                    gs += r1u[c] * 2;
                    bs += r1u[c - 1] + r1u[c + 1];
                    ng++;
                }

                if (gradS < t)
                {
                    rs += r2d[c] + r0[c];
                    gs += r1d[c] * 2;
                    bs += r1d[c - 1] + r1d[c + 1];
                    ng++;
                }

                if (gradW < t)
                {
                    rs += r0[c - 2] + r0[c];
                    gs += r0[c - 1] * 2;
                    bs += r1u[c - 1] + r1d[c - 1];
                    ng++;
                }

                if (gradE < t)
                {
                    rs += r0[c + 2] + r0[c];
                    gs += r0[c + 1] * 2;
                    bs += r1u[c + 1] + r1d[c + 1];
                    ng++;
                }

                if (gradNE < t)
                {
                    rs += r2u[c + 2] + r0[c];
                    gs += b0[n6 + c + 1];
                    bs += r1u[c + 1] * 2;
                    ng++;
                }

                if (gradSW < t)
                {
                    rs += r2d[c - 2] + r0[c];
                    gs += b2[n6 + c - 1];
                    bs += r1d[c - 1] * 2;
                    ng++;
                }

                if (gradNW < t)
                {
                    rs += r2u[c - 2] + r0[c];
                    gs += b0[n6 + c - 1];
                    bs += r1u[c - 1] * 2;
                    ng++;
                }

                if (gradSE < t)
                {
                    rs += r2d[c + 2] + r0[c];
                    gs += b2[n6 + c + 1];
                    bs += r1d[c + 1] * 2;
                    ng++;
                }

                r = r0[c];
                g = r + ScalarConverter.CvRound((gs - rs) * scale[ng]);
                b = r + ScalarConverter.CvRound((bs - rs) * scale[ng]);
            }
            else
            {
                int gradNE = b0[n2 + c] + b0[n2 + c + 1] + b1[n2 + c] + b1[n2 + c + 1];
                int gradSW = b1[n2 + c] + b1[n2 + c - 1] + b2[n2 + c] + b2[n2 + c - 1];
                int gradNW = b0[n3 + c] + b0[n3 + c - 1] + b1[n3 + c] + b1[n3 + c - 1];
                int gradSE = b1[n3 + c] + b1[n3 + c + 1] + b2[n3 + c] + b2[n3 + c + 1];
                minGrad = Math.Min(Math.Min(Math.Min(Math.Min(minGrad, gradNE), gradSW), gradNW), gradSE);
                maxGrad = Math.Max(Math.Max(Math.Max(Math.Max(maxGrad, gradNE), gradSW), gradNW), gradSE);
                int t = minGrad + Math.Max(maxGrad / 2, 1);

                int rs = 0, gs = 0, bs = 0, ng = 0;
                if (gradN < t)
                {
                    rs += r2u[c - 1] + r2u[c + 1];
                    gs += r2u[c] + r0[c];
                    bs += r1u[c] * 2;
                    ng++;
                }

                if (gradS < t)
                {
                    rs += r2d[c - 1] + r2d[c + 1];
                    gs += r2d[c] + r0[c];
                    bs += r1d[c] * 2;
                    ng++;
                }

                if (gradW < t)
                {
                    rs += r0[c - 1] * 2;
                    gs += r0[c - 2] + r0[c];
                    bs += r1u[c - 2] + r1d[c - 2];
                    ng++;
                }

                if (gradE < t)
                {
                    rs += r0[c + 1] * 2;
                    gs += r0[c + 2] + r0[c];
                    bs += r1u[c + 2] + r1d[c + 2];
                    ng++;
                }

                if (gradNE < t)
                {
                    rs += r2u[c + 1] + r0[c + 1];
                    gs += r1u[c + 1] * 2;
                    bs += r1u[c] + r1u[c + 2];
                    ng++;
                }

                if (gradSW < t)
                {
                    rs += r2d[c - 1] + r0[c - 1];
                    gs += r1d[c - 1] * 2;
                    bs += r1d[c] + r1d[c - 2];
                    ng++;
                }

                if (gradNW < t)
                {
                    rs += r2u[c - 1] + r0[c - 1];
                    gs += r1u[c - 1] * 2;
                    bs += r1u[c - 2] + r1u[c];
                    ng++;
                }

                if (gradSE < t)
                {
                    rs += r2d[c + 1] + r0[c + 1];
                    gs += r1d[c + 1] * 2;
                    bs += r1d[c + 2] + r1d[c];
                    ng++;
                }

                g = r0[c];
                r = g + ScalarConverter.CvRound((rs - gs) * scale[ng]);
                b = g + ScalarConverter.CvRound((bs - gs) * scale[ng]);
            }

            int d = (i - 2) * 3;
            dstRow[d + blueIdx] = (byte)Math.Clamp(b, 0, 255);
            dstRow[d + 1] = (byte)Math.Clamp(g, 0, 255);
            dstRow[d + (blueIdx ^ 2)] = (byte)Math.Clamp(r, 0, 255);
            greenCell = !greenCell;
        }
    }
}
