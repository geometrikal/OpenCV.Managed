// This implementation is ported from OpenCV template matching.
//
// Original sources:
//   modules/imgproc/src/templmatch.cpp (matchTemplateMask)
//   modules/core/src/matrix_expressions.cpp (MatExpr evaluation order)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV 5.0 <c>matchTemplateMask</c> on float inputs. Each step is the core operation that the
/// OpenCV matrix expression evaluates to, in the same order.
/// </summary>
internal static class TemplateMatchMask
{
    public static Mat<float> Run(Mat<float> img, Mat<float> templ, Mat<float> mask, TemplateMatchModes method)
    {
        if (img.Width < templ.Width || img.Height < templ.Height)
        {
            throw new ArgumentException("A masked template must not be larger than the image.", nameof(templ));
        }

        int cn = img.Channels;
        int rw = img.Width - templ.Width + 1, rh = img.Height - templ.Height + 1;
        var result = new Mat<float>(rw, rh, 1, clear: false);

        if (mask.Channels != cn)
        {
            var planes = new Mat<float>[cn];
            Array.Fill(planes, mask);
            mask = Conversions.Merge(planes);
        }

        if (method is TemplateMatchModes.SqDiff or TemplateMatchModes.SqDiffNormed)
        {
            using PooledMat<float> tempResult = PooledMat<float>.Rent(rw, rh, 1);
            Mat<float> mask2 = Arithmetic.Multiply(mask, mask);
            double templ2Mask2Sum = FloatReductions.NormL2Sqr(Arithmetic.Multiply(templ, mask));
            ImageSquareCorr(img, mask2, tempResult.Mat);
            TemplateCrossCorr.Run(img, Arithmetic.Multiply(templ, mask2), result);

            // -2 * result + tempResult + s: in-place addWeighted, or scaleAdd when s is zero.
            if (templ2Mask2Sum != 0)
            {
                AddWeightedInPlace(result, -2, tempResult.Mat, templ2Mask2Sum);
            }
            else
            {
                Arithmetic.ScaleAdd(result, -2, tempResult.Mat, result);
            }

            if (method == TemplateMatchModes.SqDiffNormed)
            {
                DivideBySqrtScaled(result, tempResult.Mat, templ2Mask2Sum);
            }

            return result;
        }

        if (method is TemplateMatchModes.CCorr or TemplateMatchModes.CCorrNormed)
        {
            var templMask2 = Arithmetic.Multiply(templ, Arithmetic.Multiply(mask, mask));
            TemplateCrossCorr.Run(img, templMask2, result);
            if (method == TemplateMatchModes.CCorrNormed)
            {
                using PooledMat<float> tempResult = PooledMat<float>.Rent(rw, rh, 1);
                Mat<float> mask2 = Arithmetic.Multiply(mask, mask);
                double templ2Mask2Sum = FloatReductions.NormL2Sqr(Arithmetic.Multiply(templ, mask));
                ImageSquareCorr(img, mask2, tempResult.Mat);
                DivideBySqrtScaled(result, tempResult.Mat, templ2Mask2Sum);
            }

            return result;
        }

        Scalar maskSum = Sum(mask);
        Scalar templMean = Div(Sum(Arithmetic.Multiply(mask, templ)), maskSum);
        var templCentered = SubtractScalar(templ, templMean);
        var templxMask = Arithmetic.Multiply(mask, Arithmetic.Multiply(mask, templCentered));
        using PooledMat<float> imgMaskCorrPool = PooledMat<float>.Rent(rw, rh, cn);
        using PooledMat<float> tempResPool = PooledMat<float>.Rent(rw, rh, cn);
        Mat<float> imgMaskCorr = imgMaskCorrPool.Mat, tempRes = tempResPool.Mat;

        TemplateCrossCorr.Run(img, templxMask, result);
        TemplateCrossCorr.Run(img, mask, imgMaskCorr);

        Arithmetic.Multiply(imgMaskCorr, Div(Sum(templxMask), maskSum), tempRes);
        SubtractChannelSum(result, tempRes);

        if (method == TemplateMatchModes.CCoeffNormed)
        {
            double normTemplx = Math.Sqrt(FloatReductions.NormL2Sqr(Arithmetic.Multiply(mask, templCentered)));
            using PooledMat<float> normImgxPool = PooledMat<float>.Rent(rw, rh, 1);
            using PooledMat<float> imgMask2CorrPool = PooledMat<float>.Rent(rw, rh, cn);
            using PooledMat<float> tempRes1Pool = PooledMat<float>.Rent(rw, rh, cn);
            Mat<float> normImgx = normImgxPool.Mat, imgMask2Corr = imgMask2CorrPool.Mat, tempRes1 = tempRes1Pool.Mat;
            Mat<float> mask2 = Arithmetic.Multiply(mask, mask);
            Scalar mask2Sum = Sum(mask2);
            ImageSquareCorr(img, mask2, normImgx);
            TemplateCrossCorr.Run(img, mask2, imgMask2Corr);

            Arithmetic.Multiply(imgMaskCorr, Div(Scalar.All(1.0), maskSum), tempRes1);
            Mat<float> tempRes2 = imgMaskCorr;
            Arithmetic.Multiply(imgMaskCorr, Div(mask2Sum, maskSum), tempRes2);

            // temp_res2 - 2 * img_mask2_corr evaluates as scaleAdd(img_mask2_corr, -2, temp_res2).
            Mat<float> inner = imgMask2Corr;
            Arithmetic.ScaleAdd(imgMask2Corr, -2, tempRes2, inner);
            Arithmetic.Multiply(tempRes1, inner, tempRes);
            AddChannelSum(normImgx, tempRes);

            SqrtInPlace(normImgx);

            // result /= norm_imgx * norm_templx evaluates the scaled product first.
            FloatReductions.ConvertScale(normImgx, normTemplx, normImgx);
            Arithmetic.Divide(result, normImgx, result);
        }

        return result;
    }

    /// <summary><c>crossCorr(img·img, mask2, dst)</c> with a pooled squared image.</summary>
    private static void ImageSquareCorr(Mat<float> img, Mat<float> mask2, Mat<float> dst)
    {
        using PooledMat<float> img2 = PooledMat<float>.Rent(img.Width, img.Height, img.Channels);
        Arithmetic.Multiply(img, img, img2.Mat);
        TemplateCrossCorr.Run(img2.Mat, mask2, dst);
    }

    /// <summary><c>result /= sqrt(src * s)</c>.</summary>
    private static void DivideBySqrtScaled(Mat<float> result, Mat<float> src, double s)
    {
        using PooledMat<float> denom = PooledMat<float>.Rent(src.Width, src.Height, src.Channels);
        FloatReductions.ConvertScale(src, s, denom.Mat);
        SqrtInPlace(denom.Mat);
        Arithmetic.Divide(result, denom.Mat, result);
    }

    /// <summary>
    /// OpenCV <c>addWeighted32f</c> with <c>dst == src1</c> and <c>beta = 1</c>: 4-lane blocks compute
    /// <c>a·w1 + (b·w2 + d)</c>; the in-place call computes the remaining elements of the
    /// (continuous) array as <c>(a·w1 + b·w2) + d</c>.
    /// </summary>
    private static void AddWeightedInPlace(Mat<float> a, double alpha, Mat<float> b, double gamma)
    {
        float w1 = (float)alpha, w2 = 1f, d = (float)gamma;
        Span<float> av = a.GetSpan();
        ReadOnlySpan<float> bv = b.GetReadOnlySpan();
        int n = a.Width * a.Height, vecEnd = n - (n % 4), x = 0;
        for (; x < vecEnd; x++)
        {
            av[x] = (av[x] * w1) + ((bv[x] * w2) + d);
        }

        for (; x < n; x++)
        {
            av[x] = (av[x] * w1) + (bv[x] * w2) + d;
        }
    }

    private static Scalar Sum(Mat<float> m)
    {
        double[] s = FloatReductions.Sum(m);
        return new Scalar(s[0], s[1], s[2], s[3]);
    }

    private static Scalar Div(Scalar a, Scalar b)
        => new(a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]);

    /// <summary>
    /// <c>templ - s</c>: OpenCV evaluates <c>convertTo(alpha = 1, beta = -s[0])</c> for a real scalar
    /// (1 channel) and <c>add(templ, -s)</c> otherwise; both add <c>(float)(-s[k])</c> in float.
    /// </summary>
    private static Mat<float> SubtractScalar(Mat<float> templ, Scalar s)
    {
        int cn = templ.Channels;
        var dst = new Mat<float>(templ.Width, templ.Height, cn, clear: false);
        Span<float> neg = [(float)-s[0], (float)-s[1], (float)-s[2], (float)-s[3]];
        for (int y = 0; y < templ.Height; y++)
        {
            ReadOnlySpan<float> a = templ.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            for (int i = 0; i < a.Length; i++)
            {
                d[i] = a[i] + neg[i % cn];
            }
        }

        return dst;
    }

    private static void SqrtInPlace(Mat<float> m)
    {
        for (int y = 0; y < m.Height; y++)
        {
            Span<float> d = m.GetRowSpan(y);
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = MathF.Sqrt(d[i]);
            }
        }
    }

    /// <summary><c>result -= Σ_k t[k]</c>; the channel sum is the float <c>REDUCE_SUM</c> in channel order.</summary>
    private static void SubtractChannelSum(Mat<float> result, Mat<float> t)
    {
        int cn = t.Channels;
        for (int y = 0; y < result.Height; y++)
        {
            Span<float> r = result.GetRowSpan(y);
            ReadOnlySpan<float> s = t.GetReadOnlyRowSpan(y);
            for (int x = 0; x < r.Length; x++)
            {
                r[x] -= ChannelSum(s, x * cn, cn);
            }
        }
    }

    private static void AddChannelSum(Mat<float> result, Mat<float> t)
    {
        int cn = t.Channels;
        for (int y = 0; y < result.Height; y++)
        {
            Span<float> r = result.GetRowSpan(y);
            ReadOnlySpan<float> s = t.GetReadOnlyRowSpan(y);
            for (int x = 0; x < r.Length; x++)
            {
                r[x] += ChannelSum(s, x * cn, cn);
            }
        }
    }

    private static float ChannelSum(ReadOnlySpan<float> s, int i, int cn)
    {
        float acc = s[i];
        for (int k = 1; k < cn; k++)
        {
            acc += s[i + k];
        }

        return acc;
    }
}
