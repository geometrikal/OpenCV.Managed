// This implementation is ported from the OpenCV ORB detector and descriptor.
//
// Original sources:
//   modules/features/src/orb.cpp (ORB_Impl::detectAndCompute, computeKeyPoints, HarrisResponses,
//     ICAngles, computeOrbDescriptors, getScale)
//   modules/core/src/mathfuncs_core.simd.hpp (fastAtan2, atan_f32)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>Settings of one ORB run, as the members of OpenCV <c>ORB_Impl</c>.</summary>
internal readonly record struct OrbSettings(
    int NFeatures,
    double ScaleFactor,
    int NLevels,
    int EdgeThreshold,
    int FirstLevel,
    int WtaK,
    OrbScoreType ScoreType,
    int PatchSize,
    int FastThreshold);

/// <summary>
/// OpenCV <c>ORB_Impl::detectAndCompute</c> on the CPU path: one pyramid buffer with all levels
/// and their borders, FAST per level, Harris response, intensity-centroid angle and the binary
/// descriptor.
/// </summary>
internal static class OrbEngine
{
    internal const int DescriptorSize = 32;
    private const float HarrisK = 0.04f;
    private const int HarrisBlockSize = 9;
    private const float DblEpsilon = (float)2.220446049250313E-16;

    /// <summary>OpenCV <c>getGaussianKernel(7, 2, CV_32F)</c>: the bit-exact kernel rounded to float.</summary>
    private static readonly Lazy<double[]> BlurKernel = new(() => Array.ConvertAll(Filters.GetGaussianKernel(7, 2), v => (double)(float)v));

    private static readonly float Atan2P1 = 0.9997878412794807f * (float)(180 / Math.PI);
    private static readonly float Atan2P3 = -0.3258083974640975f * (float)(180 / Math.PI);
    private static readonly float Atan2P5 = 0.1555786518463281f * (float)(180 / Math.PI);
    private static readonly float Atan2P7 = -0.04432655554792128f * (float)(180 / Math.PI);

    /// <summary>
    /// Runs ORB. <paramref name="keypoints"/> is the output for detection, or the input (changed in
    /// place) when <paramref name="useProvidedKeypoints"/> is true.
    /// </summary>
    /// <returns>Descriptors (one 32-byte row per keypoint), or null when no descriptors are requested or no keypoint remains.</returns>
    internal static Mat<byte>? Run(
        in OrbSettings s,
        Mat<byte> image,
        Mat<byte>? mask,
        List<KeyPoint> keypoints,
        bool useProvidedKeypoints,
        bool doDescriptors)
    {
        bool doKeypoints = !useProvidedKeypoints;
        if (!doKeypoints && !doDescriptors)
        {
            return null;
        }

        int halfPatchSize = s.PatchSize / 2;
        int descPatchSize = ScalarConverter.CvCeil(halfPatchSize * Math.Sqrt(2.0));
        int border = Math.Max(s.EdgeThreshold, Math.Max(descPatchSize, HarrisBlockSize / 2)) + 1;

        PooledMat<byte>? binMask = null;
        PooledMat<byte>? maskPyramid = null;
        try
        {
            if (mask is not null)
            {
                binMask = BinaryMask(mask);
            }

            return RunCore(s, image, binMask?.Mat, keypoints, useProvidedKeypoints, doDescriptors, border, ref maskPyramid);
        }
        finally
        {
            maskPyramid?.Dispose();
            binMask?.Dispose();
        }
    }

    private static Mat<byte>? RunCore(
        in OrbSettings s,
        Mat<byte> image,
        Mat<byte>? binMask,
        List<KeyPoint> keypoints,
        bool useProvidedKeypoints,
        bool doDescriptors,
        int border,
        ref PooledMat<byte>? maskPyramid)
    {
        bool doKeypoints = !useProvidedKeypoints;
        int nLevels = s.NLevels;
        bool sortedByLevel = true;
        if (!doKeypoints)
        {
            nLevels = 0;
            for (int i = 0; i < keypoints.Count; i++)
            {
                int level = keypoints[i].Octave;
                if (level < 0)
                {
                    throw new ArgumentException("A keypoint octave is negative.", nameof(keypoints));
                }

                if (i > 0 && level < keypoints[i - 1].Octave)
                {
                    sortedByLevel = false;
                }

                nLevels = Math.Max(nLevels, level);
            }

            nLevels++;
        }

        var layerInfo = new Rect[nLevels];
        var layerScale = new float[nLevels];
        Size bufSize = Layout(s, image.Width, image.Height, border, layerInfo, layerScale);

        // Pixels between the bordered levels stay uninitialized, as in OpenCV; no step reads them.
        using var pyramidBuffer = PooledMat<byte>.Rent(bufSize.Width, bufSize.Height, 1);
        Mat<byte> pyramid = pyramidBuffer.Mat;
        if (binMask is not null)
        {
            maskPyramid = PooledMat<byte>.Rent(bufSize.Width, bufSize.Height, 1);
        }

        BuildPyramid(s, image, binMask, border, layerInfo, pyramid, maskPyramid?.Mat);

        if (doKeypoints)
        {
            ComputeKeyPoints(s, pyramid, maskPyramid?.Mat, layerInfo, layerScale, keypoints);
        }
        else
        {
            KeyPointsFilter.RunByImageBorder(keypoints, new Size(image.Width, image.Height), s.EdgeThreshold);
            if (!sortedByLevel)
            {
                KeyPoint[] all = keypoints.ToArray();
                keypoints.Clear();
                for (int level = 0; level < nLevels; level++)
                {
                    foreach (KeyPoint kp in all)
                    {
                        if (kp.Octave == level)
                        {
                            keypoints.Add(kp);
                        }
                    }
                }
            }
        }

        if (!doDescriptors || keypoints.Count == 0)
        {
            return null;
        }

        if (s.WtaK is < 2 or > 4)
        {
            throw new ArgumentOutOfRangeException(nameof(s), s.WtaK, "WTA_K must be 2, 3 or 4.");
        }

        int[] pattern = OrbPatterns.Get(s.PatchSize, s.WtaK);
        double[] kernel = BlurKernel.Value;
        for (int level = 0; level < nLevels; level++)
        {
            // OpenCV blurs a submatrix with sepFilter2D and float kernels, not with the fixed-point GaussianBlur path.
            Mat<byte> working = pyramid.Roi(layerInfo[level]);
            using var blurred = PooledMat<byte>.Rent(working.Width, working.Height, 1);
            SepFilterEngine.Apply<byte, byte>(working, kernel, kernel, new Point(-1, -1), 0, BorderType.Reflect101, blurred.Mat);
            blurred.Mat.CopyTo(working);
        }

        var descriptors = new Mat<byte>(DescriptorSize, keypoints.Count);
        int wtaK = s.WtaK;
        ParallelRows.For(keypoints.Count, 2048, (start, end) =>
            ComputeDescriptors(pyramid, layerInfo, layerScale, keypoints, descriptors, pattern, wtaK, start, end));
        return descriptors;
    }

    /// <summary>OpenCV <c>getScale</c>: <c>(float)pow(scaleFactor, level − firstLevel)</c>.</summary>
    private static float GetScale(int level, int firstLevel, double scaleFactor)
        => (float)Math.Pow(scaleFactor, level - firstLevel);

    /// <summary>OpenCV <c>threshold(mask, 0, 255, THRESH_BINARY)</c>.</summary>
    private static PooledMat<byte> BinaryMask(Mat<byte> mask)
    {
        var pooled = PooledMat<byte>.Rent(mask.Width, mask.Height, 1);
        Mat<byte> dst = pooled.Mat;
        for (int y = 0; y < mask.Height; y++)
        {
            ReadOnlySpan<byte> src = mask.GetReadOnlyRowSpan(y);
            Span<byte> d = dst.GetRowSpan(y);
            for (int x = 0; x < src.Length; x++)
            {
                d[x] = src[x] > 0 ? (byte)255 : (byte)0;
            }
        }

        return pooled;
    }

    /// <summary>
    /// Level rectangles inside the pyramid buffer, as OpenCV: levels are placed left to right with
    /// a border on each side; a level that does not fit starts a new band below.
    /// </summary>
    private static Size Layout(in OrbSettings s, int cols, int rows, int border, Rect[] layerInfo, float[] layerScale)
    {
        float level0InvScale = 1.0f / GetScale(0, s.FirstLevel, s.ScaleFactor);
        int level0Width = ScalarConverter.CvRound(cols * level0InvScale);
        int level0Height = ScalarConverter.CvRound(rows * level0InvScale);
        int bufWidth = (level0Width + (border * 2) + 15) & -16;

        int levelDy = level0Height + (border * 2);
        int ofsX = 0;
        int ofsY = 0;
        for (int level = 0; level < layerInfo.Length; level++)
        {
            float scale = GetScale(level, s.FirstLevel, s.ScaleFactor);
            layerScale[level] = scale;
            float invScale = 1.0f / scale;
            int w = ScalarConverter.CvRound(cols * invScale);
            int h = ScalarConverter.CvRound(rows * invScale);
            if (w <= 0 || h <= 0)
            {
                throw new ArgumentException($"Pyramid level {level} has an empty size; OpenCV resize fails for this image size and scale.");
            }

            int wholeW = w + (border * 2);
            int wholeH = h + (border * 2);
            if (ofsX + wholeW > bufWidth)
            {
                ofsX = 0;
                ofsY += levelDy;
                levelDy = wholeH;
            }

            layerInfo[level] = new Rect(ofsX + border, ofsY + border, w, h);
            ofsX += wholeW;
        }

        return new Size(bufWidth, ofsY + levelDy);
    }

    private static void BuildPyramid(
        in OrbSettings s,
        Mat<byte> image,
        Mat<byte>? mask,
        int border,
        Rect[] layerInfo,
        Mat<byte> pyramid,
        Mat<byte>? maskPyramid)
    {
        Mat<byte> prevImg = image;
        Mat<byte>? prevMask = mask;
        for (int level = 0; level < layerInfo.Length; level++)
        {
            Rect li = layerInfo[level];
            var whole = new Rect(li.X - border, li.Y - border, li.Width + (2 * border), li.Height + (2 * border));
            Mat<byte> extImg = pyramid.Roi(whole);
            Mat<byte> currImg = extImg.Roi(border, border, li.Width, li.Height);
            Mat<byte>? extMask = null;
            Mat<byte>? currMask = null;
            if (maskPyramid is not null)
            {
                extMask = maskPyramid.Roi(whole);
                currMask = extMask.Roi(border, border, li.Width, li.Height);
            }

            if (level != s.FirstLevel)
            {
                using (var resized = PooledMat<byte>.Rent(li.Width, li.Height, 1))
                {
                    Geometry.Resize(prevImg, resized.Mat, InterpolationFlags.LinearExact);
                    Borders.CopyMakeBorder(resized.Mat, extImg, border, border, border, border, BorderType.Reflect101);
                }

                if (currMask is not null)
                {
                    using var resized = PooledMat<byte>.Rent(li.Width, li.Height, 1);
                    Geometry.Resize(prevMask!, resized.Mat, InterpolationFlags.LinearExact);
                    if (level > s.FirstLevel)
                    {
                        ToZero254(resized.Mat);
                    }

                    Borders.CopyMakeBorder(resized.Mat, extMask!, border, border, border, border, BorderType.Constant);
                }
            }
            else
            {
                Borders.CopyMakeBorder(image, extImg, border, border, border, border, BorderType.Reflect101);
                if (mask is not null)
                {
                    Borders.CopyMakeBorder(mask, extMask!, border, border, border, border, BorderType.Constant);
                }
            }

            if (level > s.FirstLevel)
            {
                prevImg = currImg;
                prevMask = currMask;
            }
        }
    }

    /// <summary>OpenCV <c>threshold(m, m, 254, 0, THRESH_TOZERO)</c>: only 255 stays.</summary>
    private static void ToZero254(Mat<byte> m)
    {
        for (int y = 0; y < m.Height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                if (row[x] <= 254)
                {
                    row[x] = 0;
                }
            }
        }
    }

    private static void ComputeKeyPoints(
        in OrbSettings s,
        Mat<byte> pyramid,
        Mat<byte>? maskPyramid,
        Rect[] layerInfo,
        float[] layerScale,
        List<KeyPoint> allKeypoints)
    {
        int nlevels = layerInfo.Length;
        int[] nfeaturesPerLevel = FeaturesPerLevel(s.NFeatures, s.ScaleFactor, nlevels);
        int halfPatchSize = s.PatchSize / 2;
        int[] umax = Umax(halfPatchSize);

        allKeypoints.Clear();
        int[] counters = new int[nlevels];
        var levelKeypoints = FastEngine.KeyPointBuffer.Rent();
        try
        {
            for (int level = 0; level < nlevels; level++)
            {
                int featuresNum = nfeaturesPerLevel[level];
                Mat<byte> img = pyramid.Roi(layerInfo[level]);

                DetectFastInsideBorder(img, ref levelKeypoints, s.FastThreshold, s.EdgeThreshold);
                Span<KeyPoint> span = levelKeypoints.Span;
                if (maskPyramid is not null)
                {
                    span = span[..KeyPointsFilter.RunByPixelsMaskSpan(span, maskPyramid.Roi(layerInfo[level]))];
                }

                span = span[..KeyPointsFilter.RunByImageBorderSpan(span, new Size(img.Width, img.Height), s.EdgeThreshold)];
                span = span[..KeyPointsFilter.RetainBestSpan(span, s.ScoreType == OrbScoreType.Harris ? 2 * featuresNum : featuresNum)];

                counters[level] = span.Length;
                float sf = layerScale[level];
                for (int i = 0; i < span.Length; i++)
                {
                    span[i].Octave = level;
                    span[i].Size = s.PatchSize * sf;
                }

                allKeypoints.AddRange(span);
            }
        }
        finally
        {
            levelKeypoints.Return();
        }

        if (allKeypoints.Count == 0)
        {
            return;
        }

        if (s.ScoreType == OrbScoreType.Harris)
        {
            ParallelRows.For(allKeypoints.Count, 49 * 16, (start, end) =>
                HarrisResponses(pyramid, layerInfo, CollectionsMarshal.AsSpan(allKeypoints)[start..end], 7, HarrisK));

            // Each level is reduced in place and moved down; the write position never passes the read position.
            Span<KeyPoint> harris = CollectionsMarshal.AsSpan(allKeypoints);
            int offset = 0;
            int kept = 0;
            for (int level = 0; level < nlevels; level++)
            {
                Span<KeyPoint> levelSpan = harris.Slice(offset, counters[level]);
                offset += counters[level];
                int count = KeyPointsFilter.RetainBestSpan(levelSpan, nfeaturesPerLevel[level]);
                levelSpan[..count].CopyTo(harris[kept..]);
                kept += count;
            }

            CollectionsMarshal.SetCount(allKeypoints, kept);
        }

        ParallelRows.For(allKeypoints.Count, 4 * halfPatchSize * halfPatchSize, (start, end) =>
            IcAngles(pyramid, layerInfo, CollectionsMarshal.AsSpan(allKeypoints)[start..end], umax, halfPatchSize));
        Span<KeyPoint> all = CollectionsMarshal.AsSpan(allKeypoints);
        for (int i = 0; i < all.Length; i++)
        {
            float scale = layerScale[all[i].Octave];
            all[i].Pt = new Point2f(all[i].Pt.X * scale, all[i].Pt.Y * scale);
        }
    }

    /// <summary>
    /// FAST 9_16 with non-maximum suppression on <paramref name="img"/>. When the border filter
    /// that follows removes every keypoint within <paramref name="edgeThreshold"/> pixels of an
    /// edge, FAST runs only on the part of the image that can give the other keypoints. A pixel
    /// at least 4 pixels inside that part has the same corner test, score and neighbour scores
    /// as in the full image, so the kept keypoints are the same.
    /// </summary>
    private static void DetectFastInsideBorder(Mat<byte> img, ref FastEngine.KeyPointBuffer keypoints, int threshold, int edgeThreshold)
    {
        int o = edgeThreshold - 4;
        if (o <= 0 || img.Width <= 2 * edgeThreshold || img.Height <= 2 * edgeThreshold)
        {
            FastEngine.DetectInto(img, threshold, true, FastDetectorType.Type9_16, ref keypoints);
            return;
        }

        FastEngine.DetectInto(img.Roi(o, o, img.Width - (2 * o), img.Height - (2 * o)), threshold, true, FastDetectorType.Type9_16, ref keypoints);
        Span<KeyPoint> span = keypoints.Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i].Pt = new Point2f(span[i].Pt.X + o, span[i].Pt.Y + o);
        }
    }

    /// <summary>OpenCV feature distribution over levels: a geometric series with ratio <c>1/scaleFactor</c> in single precision.</summary>
    internal static int[] FeaturesPerLevel(int nfeatures, double scaleFactor, int nlevels)
    {
        int[] perLevel = new int[nlevels];
        float factor = (float)(1.0 / scaleFactor);
        float ndesired = nfeatures * (1 - factor) / (1 - (float)Math.Pow(factor, nlevels));
        int sum = 0;
        for (int level = 0; level < nlevels - 1; level++)
        {
            perLevel[level] = ScalarConverter.CvRound(ndesired);
            sum += perLevel[level];
            ndesired *= factor;
        }

        perLevel[nlevels - 1] = Math.Max(nfeatures - sum, 0);
        return perLevel;
    }

    /// <summary>OpenCV <c>umax</c>: half-width of each row of the circular patch, made symmetric.</summary>
    internal static int[] Umax(int halfPatchSize)
    {
        int[] umax = new int[halfPatchSize + 2];
        float sqrt2 = MathF.Sqrt(2f);
        int vmax = ScalarConverter.CvFloor((halfPatchSize * sqrt2 / 2) + 1);
        int vmin = ScalarConverter.CvCeil(halfPatchSize * sqrt2 / 2);
        for (int v = 0; v <= vmax; ++v)
        {
            umax[v] = ScalarConverter.CvRound(Math.Sqrt(((double)halfPatchSize * halfPatchSize) - (v * v)));
        }

        for (int v = halfPatchSize, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
            {
                ++v0;
            }

            umax[v] = v0;
            ++v0;
        }

        return umax;
    }

    /// <summary>OpenCV <c>HarrisResponses</c> with integer Sobel-like gradients and a single-precision response.</summary>
    private static void HarrisResponses(Mat<byte> img, Rect[] layerInfo, Span<KeyPoint> pts, int blockSize, float harrisK)
    {
        ReadOnlySpan<byte> data = img.DangerousGetArray();
        int baseOffset = img.DangerousGetOffset();
        int step = img.Stride;
        int r = blockSize / 2;
        float scale = 1f / ((1 << 2) * blockSize * 255f);
        float scaleSqSq = scale * scale * scale * scale;

        for (int p = 0; p < pts.Length; p++)
        {
            int x0 = ScalarConverter.CvRound(pts[p].Pt.X);
            int y0 = ScalarConverter.CvRound(pts[p].Pt.Y);
            Rect layer = layerInfo[pts[p].Octave];
            int ofs0 = baseOffset + ((y0 - r + layer.Y) * step) + (x0 - r + layer.X);
            int a = 0, b = 0, c = 0;
            for (int i = 0; i < blockSize; i++)
            {
                for (int j = 0; j < blockSize; j++)
                {
                    int o = ofs0 + (i * step) + j;
                    int ix = ((data[o + 1] - data[o - 1]) * 2) + (data[o - step + 1] - data[o - step - 1]) + (data[o + step + 1] - data[o + step - 1]);
                    int iy = ((data[o + step] - data[o - step]) * 2) + (data[o + step - 1] - data[o - step - 1]) + (data[o + step + 1] - data[o - step + 1]);
                    a += ix * ix;
                    b += iy * iy;
                    c += ix * iy;
                }
            }

            pts[p].Response = (((float)a * b) - ((float)c * c) - (harrisK * ((float)a + b) * ((float)a + b))) * scaleSqSq;
        }
    }

    /// <summary>OpenCV <c>ICAngles</c>: intensity-centroid orientation over the circular patch.</summary>
    private static void IcAngles(Mat<byte> img, Rect[] layerInfo, Span<KeyPoint> pts, int[] umax, int halfK)
    {
        ReadOnlySpan<byte> data = img.DangerousGetArray();
        int baseOffset = img.DangerousGetOffset();
        int step = img.Stride;
        for (int p = 0; p < pts.Length; p++)
        {
            Rect layer = layerInfo[pts[p].Octave];
            int center = baseOffset + ((ScalarConverter.CvRound(pts[p].Pt.Y) + layer.Y) * step) + ScalarConverter.CvRound(pts[p].Pt.X) + layer.X;
            int m01 = 0, m10 = 0;
            for (int u = -halfK; u <= halfK; ++u)
            {
                m10 += u * data[center + u];
            }

            for (int v = 1; v <= halfK; ++v)
            {
                int vSum = 0;
                int d = umax[v];
                for (int u = -d; u <= d; ++u)
                {
                    int valPlus = data[center + u + (v * step)];
                    int valMinus = data[center + u - (v * step)];
                    vSum += valPlus - valMinus;
                    m10 += u * (valPlus + valMinus);
                }

                m01 += v * vSum;
            }

            pts[p].Angle = FastAtan2(m01, m10);
        }
    }

    /// <summary>OpenCV <c>fastAtan2</c> (polynomial <c>atan_f32</c>), in degrees in [0, 360).</summary>
    internal static float FastAtan2(float y, float x)
    {
        float ax = MathF.Abs(x), ay = MathF.Abs(y);
        float a, c, c2;
        if (ax >= ay)
        {
            c = ay / (ax + DblEpsilon);
            c2 = c * c;
            a = ((((((Atan2P7 * c2) + Atan2P5) * c2) + Atan2P3) * c2) + Atan2P1) * c;
        }
        else
        {
            c = ax / (ay + DblEpsilon);
            c2 = c * c;
            a = 90f - (((((((Atan2P7 * c2) + Atan2P5) * c2) + Atan2P3) * c2) + Atan2P1) * c);
        }

        if (x < 0)
        {
            a = 180f - a;
        }

        if (y < 0)
        {
            a = 360f - a;
        }

        return a;
    }

    private static void ComputeDescriptors(
        Mat<byte> pyramid,
        Rect[] layerInfo,
        float[] layerScale,
        List<KeyPoint> keypoints,
        Mat<byte> descriptors,
        int[] pattern,
        int wtaK,
        int start,
        int end)
    {
        ReadOnlySpan<byte> data = pyramid.DangerousGetArray();
        int baseOffset = pyramid.DangerousGetOffset();
        int step = pyramid.Stride;
        int npoints = pattern.Length / 2;
        int padded = (npoints + 7) & ~7;
        float[] pxBuffer = ArrayPool<float>.Shared.Rent(padded);
        float[] pyBuffer = ArrayPool<float>.Shared.Rent(padded);
        int[] offsetBuffer = ArrayPool<int>.Shared.Rent(padded);
        byte[] valueBuffer = ArrayPool<byte>.Shared.Rent(npoints);
        try
        {
            Span<float> px = pxBuffer.AsSpan(0, padded);
            Span<float> py = pyBuffer.AsSpan(0, padded);
            px.Clear();
            py.Clear();
            for (int i = 0; i < npoints; i++)
            {
                px[i] = pattern[2 * i];
                py[i] = pattern[(2 * i) + 1];
            }

            ComputeDescriptorRows(data, baseOffset, step, layerInfo, layerScale, keypoints, descriptors, px, py, offsetBuffer.AsSpan(0, padded), valueBuffer.AsSpan(0, npoints), wtaK, start, end);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(valueBuffer);
            ArrayPool<int>.Shared.Return(offsetBuffer);
            ArrayPool<float>.Shared.Return(pyBuffer);
            ArrayPool<float>.Shared.Return(pxBuffer);
        }
    }

    private static void ComputeDescriptorRows(
        ReadOnlySpan<byte> data,
        int baseOffset,
        int step,
        Rect[] layerInfo,
        float[] layerScale,
        List<KeyPoint> keypoints,
        Mat<byte> descriptors,
        ReadOnlySpan<float> px,
        ReadOnlySpan<float> py,
        Span<int> offsets,
        Span<byte> values,
        int wtaK,
        int start,
        int end)
    {
        int npoints = values.Length;
        for (int j = start; j < end; j++)
        {
            KeyPoint kpt = keypoints[j];
            Rect layer = layerInfo[kpt.Octave];
            float scale = 1f / layerScale[kpt.Octave];
            float angle = kpt.Angle;
            angle *= (float)(Math.PI / 180.0);
            float a = MathF.Cos(angle), b = MathF.Sin(angle);
            int center = baseOffset + ((ScalarConverter.CvRound(kpt.Pt.Y * scale) + layer.Y) * step) + ScalarConverter.CvRound(kpt.Pt.X * scale) + layer.X;
            Span<byte> desc = descriptors.GetRowSpan(j);
            var sampler = new Sampler(values);
            RotatedOffsets(px, py, a, b, step, offsets);
            for (int i = 0; i < npoints; i++)
            {
                values[i] = data[center + offsets[i]];
            }

            switch (wtaK)
            {
                case 2:
                    for (int i = 0; i < DescriptorSize; ++i)
                    {
                        int p0 = i * 16;
                        int val = 0;
                        for (int bit = 0; bit < 8; bit++)
                        {
                            int t0 = sampler.Get(p0 + (2 * bit));
                            int t1 = sampler.Get(p0 + (2 * bit) + 1);
                            val |= (t0 < t1 ? 1 : 0) << bit;
                        }

                        desc[i] = (byte)val;
                    }

                    break;
                case 3:
                    for (int i = 0; i < DescriptorSize; ++i)
                    {
                        int p0 = i * 12;
                        int val = 0;
                        for (int q = 0; q < 4; q++)
                        {
                            int t0 = sampler.Get(p0 + (3 * q));
                            int t1 = sampler.Get(p0 + (3 * q) + 1);
                            int t2 = sampler.Get(p0 + (3 * q) + 2);
                            int k = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0 ? 1 : 0);
                            val |= k << (2 * q);
                        }

                        desc[i] = (byte)val;
                    }

                    break;
                default:
                    for (int i = 0; i < DescriptorSize; ++i)
                    {
                        int p0 = i * 16;
                        int val = 0;
                        for (int q = 0; q < 4; q++)
                        {
                            int t0 = sampler.Get(p0 + (4 * q));
                            int t1 = sampler.Get(p0 + (4 * q) + 1);
                            int t2 = sampler.Get(p0 + (4 * q) + 2);
                            int t3 = sampler.Get(p0 + (4 * q) + 3);
                            int u = 0, v = 2;
                            if (t1 > t0)
                            {
                                t0 = t1;
                                u = 1;
                            }

                            if (t3 > t2)
                            {
                                t2 = t3;
                                v = 3;
                            }

                            int k = t0 > t2 ? u : v;
                            val |= k << (2 * q);
                        }

                        desc[i] = (byte)val;
                    }

                    break;
            }
        }
    }

    /// <summary>
    /// Offsets of the pattern points rotated by <c>(a, b)</c>, as OpenCV <c>GET_VALUE</c>:
    /// <c>cvRound(x·a − y·b)</c> and <c>cvRound(x·b + y·a)</c> in single precision.
    /// </summary>
    private static void RotatedOffsets(ReadOnlySpan<float> px, ReadOnlySpan<float> py, float a, float b, int step, Span<int> offsets)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var va = Vector256.Create(a);
            var vb = Vector256.Create(b);
            var vstep = Vector256.Create(step);
            for (; i <= px.Length - 8; i += 8)
            {
                Vector256<float> x = Vector256.Create(px.Slice(i, 8));
                Vector256<float> y = Vector256.Create(py.Slice(i, 8));
                Vector256<int> ix = Vector256.ConvertToInt32(Vector256.Round((x * va) - (y * vb)));
                Vector256<int> iy = Vector256.ConvertToInt32(Vector256.Round((x * vb) + (y * va)));
                ((iy * vstep) + ix).CopyTo(offsets.Slice(i, 8));
            }
        }

        for (; i < px.Length; i++)
        {
            float x = (px[i] * a) - (py[i] * b);
            float y = (px[i] * b) + (py[i] * a);
            offsets[i] = ((int)MathF.Round(y) * step) + (int)MathF.Round(x);
        }
    }

    /// <summary>Pattern point values of one keypoint.</summary>
    private readonly ref struct Sampler
    {
        private readonly ReadOnlySpan<byte> _values;

        public Sampler(ReadOnlySpan<byte> values) => _values = values;

        public int Get(int idx) => _values[idx];
    }
}
