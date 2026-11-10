// This implementation is ported from OpenCV template matching.
//
// Original sources:
//   modules/imgproc/src/templmatch.cpp (common_matchTemplate)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>common_matchTemplate</c>: converts the raw cross-correlation in place into the
/// requested measure with float64 window sums from integral images.
/// </summary>
internal static class TemplateNormalization
{
    private const double DblEpsilon = 2.220446049250313E-16;
    private const double FltEpsilon = 1.1920928955078125E-07;

    public static void Run<T>(Mat<T> img, Mat<T> templ, Mat<float> result, TemplateMatchModes method)
        where T : unmanaged
    {
        if (method == TemplateMatchModes.CCorr)
        {
            return;
        }

        int cn = img.Channels;
        int numType = method is TemplateMatchModes.CCorrNormed ? 0
            : method is TemplateMatchModes.CCoeff or TemplateMatchModes.CCoeffNormed ? 1
            : 2;
        bool isNormed = method is TemplateMatchModes.CCorrNormed or TemplateMatchModes.SqDiffNormed or TemplateMatchModes.CCoeffNormed;

        double invArea = 1.0 / ((double)templ.Height * templ.Width);
        using PooledMat<double> sumPool = PooledMat<double>.Rent(img.Width + 1, img.Height + 1, cn);
        using PooledMat<double> sqsumPool = PooledMat<double>.Rent(method == TemplateMatchModes.CCoeff ? 1 : img.Width + 1, method == TemplateMatchModes.CCoeff ? 1 : img.Height + 1, cn);
        Mat<double> sum = sumPool.Mat;
        Mat<double>? sqsum = null;
        Scalar templMean;
        double templNorm = 0, templSum2 = 0;

        if (method == TemplateMatchModes.CCoeff)
        {
            IntegralImages.Integral(img, sum);
            templMean = typeof(T) == typeof(float) ? FloatReductions.Mean((Mat<float>)(object)templ) : Reductions.Mean(templ);
        }
        else
        {
            sqsum = sqsumPool.Mat;
            IntegralImages.Integral(img, sum, sqsum);
            Scalar templSdv;
            if (typeof(T) == typeof(float))
            {
                FloatReductions.MeanStdDev((Mat<float>)(object)templ, null, out templMean, out templSdv);
            }
            else
            {
                Reductions.MeanStdDev(templ, out templMean, out templSdv);
            }

            templNorm = (templSdv[0] * templSdv[0]) + (templSdv[1] * templSdv[1]) + (templSdv[2] * templSdv[2]) + (templSdv[3] * templSdv[3]);

            if (templNorm < DblEpsilon && method == TemplateMatchModes.CCoeffNormed)
            {
                result.Fill(1f);
                return;
            }

            templSum2 = templNorm + (templMean[0] * templMean[0]) + (templMean[1] * templMean[1]) + (templMean[2] * templMean[2]) + (templMean[3] * templMean[3]);
            if (numType != 1)
            {
                templMean = default;
                templNorm = templSum2;
            }

            templSum2 /= invArea;
            templNorm = Math.Sqrt(templNorm);
            templNorm /= Math.Sqrt(invArea);
        }

        var p = new Params
        {
            Cn = cn,
            NumType = numType,
            IsNormed = isNormed,
            IsSqDiffNormed = method == TemplateMatchModes.SqDiffNormed,
            InvArea = invArea,
            TemplNorm = templNorm,
            TemplSum2 = templSum2,
            M0 = templMean[0],
            M1 = templMean[1],
            M2 = templMean[2],
            M3 = templMean[3],
            Tw = templ.Width * cn,
            Th = templ.Height,
        };

        Mat<double> sqs = sqsum ?? sum;
        ParallelRows.For(result.Height, (long)result.Width * cn * 4, (y0, y1) => Rows(sum, sqs, result, p, y0, y1));
    }

    private struct Params
    {
        public int Cn, NumType, Tw, Th;
        public bool IsNormed, IsSqDiffNormed;
        public double InvArea, TemplNorm, TemplSum2, M0, M1, M2, M3;
    }

    private static void Rows(Mat<double> sum, Mat<double> sqsum, Mat<float> result, Params p, int y0, int y1)
    {
        int cn = p.Cn;
        Span<double> mean = [p.M0, p.M1, p.M2, p.M3];
        for (int i = y0; i < y1; i++)
        {
            ReadOnlySpan<double> s0 = sum.GetReadOnlyRowSpan(i);
            ReadOnlySpan<double> s2 = sum.GetReadOnlyRowSpan(i + p.Th);
            ReadOnlySpan<double> q0 = sqsum.GetReadOnlyRowSpan(i);
            ReadOnlySpan<double> q2 = sqsum.GetReadOnlyRowSpan(i + p.Th);
            Span<float> rrow = result.GetRowSpan(i);

            int idx = 0;
            for (int j = 0; j < rrow.Length; j++, idx += cn)
            {
                double num = rrow[j], t;
                double wndMean2 = 0, wndSum2 = 0;

                if (p.NumType == 1)
                {
                    for (int k = 0; k < cn; k++)
                    {
                        t = s0[idx + k] - s0[idx + p.Tw + k] - s2[idx + k] + s2[idx + p.Tw + k];
                        wndMean2 += t * t;
                        num -= t * mean[k];
                    }

                    wndMean2 *= p.InvArea;
                }

                if (p.IsNormed || p.NumType == 2)
                {
                    for (int k = 0; k < cn; k++)
                    {
                        t = q0[idx + k] - q0[idx + p.Tw + k] - q2[idx + k] + q2[idx + p.Tw + k];
                        wndSum2 += t;
                    }

                    if (p.NumType == 2)
                    {
                        num = wndSum2 - (2 * num) + p.TemplSum2;
                        num = Math.Max(num, 0.0);
                    }
                }

                if (p.IsNormed)
                {
                    double diff2 = Math.Max(wndSum2 - wndMean2, 0);
                    if (diff2 <= Math.Min(0.5, 10 * FltEpsilon * wndSum2))
                    {
                        t = 0;
                    }
                    else
                    {
                        t = Math.Sqrt(diff2) * p.TemplNorm;
                    }

                    if (Math.Abs(num) < t)
                    {
                        num /= t;
                    }
                    else if (Math.Abs(num) < t * 1.125)
                    {
                        num = num > 0 ? 1 : -1;
                    }
                    else
                    {
                        num = p.IsSqDiffNormed ? 1 : 0;
                    }
                }

                rrow[j] = (float)num;
            }
        }
    }
}
