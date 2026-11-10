// This implementation is ported from the OpenCV ECC image alignment.
//
// Original sources:
//   modules/video/src/ecc.cpp (findTransformECCWithMask, computeECC and the image_jacobian_*,
//   project_onto_jacobian_ECC and update_warping_matrix_ECC helpers)
//   modules/core/src/matrix_expressions.cpp (MatExpr evaluation order)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV 5.0 <c>findTransformECCWithMask</c> and <c>computeECC</c>. Each matrix expression of
/// ecc.cpp is evaluated with the core operation that OpenCV selects for it, in the same order.
/// </summary>
internal static class EccEngine
{
    public static double ComputeEcc<T>(Mat<T> templ, Mat<T> input, Mat<byte>? mask)
        where T : unmanaged
    {
        int activePixels = mask is null ? templ.Width * templ.Height : Reductions.CountNonZero(mask);
        FloatReductions.MeanStdDev(templ, mask, out Scalar meanT, out Scalar sdT);
        FloatReductions.MeanStdDev(input, mask, out Scalar meanI, out Scalar sdI);
        double templNorm = Math.Sqrt(activePixels * ScalarNormL2Sqr(sdT));
        double inputNorm = Math.Sqrt(activePixels * ScalarNormL2Sqr(sdI));

        double dot;
        if (typeof(T) == typeof(float))
        {
            using PooledMat<float> zt = PooledMat<float>.Rent(templ.Width, templ.Height, templ.Channels);
            using PooledMat<float> zi = PooledMat<float>.Rent(input.Width, input.Height, input.Channels);
            ZeroMeanFloat((Mat<float>)(object)templ, meanT, mask, zt.Mat);
            ZeroMeanFloat((Mat<float>)(object)input, meanI, mask, zi.Mat);
            dot = FloatReductions.Dot(zt.Mat, zi.Mat);
        }
        else if (typeof(T) == typeof(double))
        {
            dot = ZeroMeanDotDouble((Mat<double>)(object)templ, meanT, (Mat<double>)(object)input, meanI, mask);
        }
        else
        {
            dot = ZeroMeanDotInteger(templ, meanT, input, meanI, mask);
        }

        return dot / (templNorm * inputNorm);
    }

    /// <summary>
    /// <c>norm(Scalar, NORM_L2SQR)</c>: the four components of a 4×1 double matrix summed in order.
    /// </summary>
    private static double ScalarNormL2Sqr(Scalar s)
    {
        double r = 0;
        for (int k = 0; k < 4; k++)
        {
            r += s[k] * s[k];
        }

        return r;
    }

    /// <summary><c>subtract(m, mean, zeros, mask)</c> for float: <c>v - (float)mean</c> at selected pixels, zero elsewhere.</summary>
    private static void ZeroMeanFloat(Mat<float> m, Scalar mean, Mat<byte>? mask, Mat<float> dst)
    {
        int cn = m.Channels;
        dst.Clear();
        Span<float> mv = [(float)mean[0], (float)mean[1], (float)mean[2], (float)mean[3]];
        for (int y = 0; y < m.Height; y++)
        {
            ReadOnlySpan<float> s = m.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            ReadOnlySpan<byte> mk = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            for (int x = 0; x < m.Width; x++)
            {
                if (mask is not null && mk[x] == 0)
                {
                    continue;
                }

                for (int k = 0; k < cn; k++)
                {
                    d[(x * cn) + k] = s[(x * cn) + k] - mv[k];
                }
            }
        }
    }

    /// <summary>Double sources: <c>v - mean</c> in double, then <c>dotProd_64f</c> (groups of four).</summary>
    private static double ZeroMeanDotDouble(Mat<double> a, Scalar ma, Mat<double> b, Scalar mb, Mat<byte>? mask)
    {
        int cn = a.Channels;
        int n = a.Width * a.Height * cn;
        double[] za = ArrayPool<double>.Shared.Rent(n);
        double[] zb = ArrayPool<double>.Shared.Rent(n);
        try
        {
            return ZeroMeanDotDouble(a, ma, b, mb, mask, za.AsSpan(0, n), zb.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<double>.Shared.Return(zb);
            ArrayPool<double>.Shared.Return(za);
        }
    }

    private static double ZeroMeanDotDouble(Mat<double> a, Scalar ma, Mat<double> b, Scalar mb, Mat<byte>? mask, Span<double> za, Span<double> zb)
    {
        int cn = a.Channels;
        int i = 0;
        for (int y = 0; y < a.Height; y++)
        {
            ReadOnlySpan<double> ra = a.GetReadOnlyRowSpan(y);
            ReadOnlySpan<double> rb = b.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> mk = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            for (int x = 0; x < a.Width; x++)
            {
                bool on = mask is null || mk[x] != 0;
                for (int k = 0; k < cn; k++, i++)
                {
                    za[i] = on ? ra[(x * cn) + k] - ma[k] : 0;
                    zb[i] = on ? rb[(x * cn) + k] - mb[k] : 0;
                }
            }
        }

        double result = 0;
        int j = 0;
        for (; j <= za.Length - 4; j += 4)
        {
            result += (za[j] * zb[j]) + (za[j + 1] * zb[j + 1]) + (za[j + 2] * zb[j + 2]) + (za[j + 3] * zb[j + 3]);
        }

        for (; j < za.Length; j++)
        {
            result += za[j] * zb[j];
        }

        return result;
    }

    /// <summary>
    /// 8-bit and 16-bit sources: OpenCV subtracts the mean rounded to an integer (<c>cvRound</c>);
    /// the integer dot product is exact. 1-channel images are first converted to a signed type.
    /// Multi-channel images are not converted (OpenCV tests the type <c>CV_8U</c>/<c>CV_16U</c>,
    /// which excludes them), so the differences saturate to the unsigned range.
    /// </summary>
    private static double ZeroMeanDotInteger<T>(Mat<T> a, Scalar ma, Mat<T> b, Scalar mb, Mat<byte>? mask)
        where T : unmanaged
    {
        int cn = a.Channels;
        Span<long> ia = stackalloc long[4];
        Span<long> ib = stackalloc long[4];
        for (int k = 0; k < 4; k++)
        {
            ia[k] = (long)Math.Round(ma[k], MidpointRounding.ToEven);
            ib[k] = (long)Math.Round(mb[k], MidpointRounding.ToEven);
        }

        long max = cn == 1 ? long.MaxValue : typeof(T) == typeof(byte) ? byte.MaxValue : ushort.MaxValue;
        long min = cn == 1 ? long.MinValue : 0;
        long dot = 0;
        for (int y = 0; y < a.Height; y++)
        {
            ReadOnlySpan<T> ra = a.GetReadOnlyRowSpan(y);
            ReadOnlySpan<T> rb = b.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> mk = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            for (int x = 0; x < a.Width; x++)
            {
                if (mask is not null && mk[x] == 0)
                {
                    continue;
                }

                for (int k = 0; k < cn; k++)
                {
                    long va = Math.Clamp((long)Saturate.ToDouble(ra[(x * cn) + k]) - ia[k], min, max);
                    long vb = Math.Clamp((long)Saturate.ToDouble(rb[(x * cn) + k]) - ib[k], min, max);
                    dot += va * vb;
                }
            }
        }

        return dot;
    }

    public static double FindTransform<T>(
        Mat<T> src, Mat<T> dst, Mat<byte>? templateMask, Mat<byte>? inputMask, Mat<float> map,
        MotionTypes motionType, TermCriteria criteria, int gaussFiltSize)
        where T : unmanaged
    {
        int numberOfIterations = (criteria.Type & TermCriteriaType.MaxIter) != 0 ? criteria.MaxCount : 200;
        double terminationEps = (criteria.Type & TermCriteriaType.Eps) != 0 ? criteria.Epsilon : -1;
        int np = motionType switch
        {
            MotionTypes.Translation => 2,
            MotionTypes.Euclidean => 3,
            MotionTypes.Homography => 8,
            _ => 6,
        };

        int ws = src.Width, hs = src.Height, cn = src.Channels;
        using PooledMat<float> xgridPool = PooledMat<float>.Rent(ws, hs, cn);
        using PooledMat<float> ygridPool = PooledMat<float>.Rent(ws, hs, cn);
        Mat<float> xgrid = xgridPool.Mat, ygrid = ygridPool.Mat;
        for (int y = 0; y < hs; y++)
        {
            Span<float> xr = xgrid.GetRowSpan(y);
            Span<float> yr = ygrid.GetRowSpan(y);
            for (int x = 0; x < ws; x++)
            {
                for (int k = 0; k < cn; k++)
                {
                    xr[(x * cn) + k] = x;
                    yr[(x * cn) + k] = y;
                }
            }
        }

        var gsize = new Size(gaussFiltSize, gaussFiltSize);
        int wd = dst.Width, hd = dst.Height;
        using PooledMat<float> templateFloatPool = PooledMat<float>.Rent(ws, hs, cn);
        using PooledMat<float> imageFloatPool = PooledMat<float>.Rent(wd, hd, cn);
        using PooledMat<float> gradientXPool = PooledMat<float>.Rent(wd, hd, cn);
        using PooledMat<float> gradientYPool = PooledMat<float>.Rent(wd, hd, cn);
        Mat<float> templateFloat = templateFloatPool.Mat;
        Mat<float> imageFloat = imageFloatPool.Mat;
        Mat<float> gradientX = gradientXPool.Mat;
        Mat<float> gradientY = gradientYPool.Mat;
        BlurAsFloat(src, gsize, templateFloat);
        BlurAsFloat(dst, gsize, imageFloat);

        var dx = new Mat<float>(new[] { -0.5f, 0f, 0.5f }, 3, 1);
        var dy = new Mat<float>(new[] { -0.5f, 0f, 0.5f }, 1, 3);
        Filters.Filter2DInto(imageFloat, gradientX, dx);
        Filters.Filter2DInto(imageFloat, gradientY, dy);

        using PooledMat<byte> templtOnes = PooledMat<byte>.Rent(templateMask is null ? ws : 1, templateMask is null ? hs : 1, 1);
        using PooledMat<byte> preOnes = PooledMat<byte>.Rent(inputMask is null ? dst.Width : 1, inputMask is null ? dst.Height : 1, 1);
        Mat<byte> templtMask = templateMask is null ? Ones(templtOnes.Mat) : BlurredMask(templateMask, gsize);
        Mat<byte> preMask;
        if (inputMask is null)
        {
            preMask = Ones(preOnes.Mat);
        }
        else
        {
            preMask = BlurredMask(inputMask, gsize);
            if (templateMask is null)
            {
                Mat<byte> zeroMask = ZeroMask(preMask);
                gradientX.SetTo(new Scalar(0), zeroMask);
                gradientY.SetTo(new Scalar(0), zeroMask);
            }
        }

        using PooledMat<float> jacobianPool = PooledMat<float>.Rent(ws * np, hs, cn);
        Mat<float> jacobian = jacobianPool.Mat;
        var blocks = new Mat<float>[np];
        for (int i = 0; i < np; i++)
        {
            blocks[i] = jacobian.Roi(i * ws, 0, ws, hs);
        }

        var hessian = new Mat<float>(np, np, 1, clear: false);
        var imageProjection = new Mat<float>(1, np);
        var templateProjection = new Mat<float>(1, np);
        var errorProjection = new Mat<float>(1, np);

        using PooledMat<float> imageWarpedPool = PooledMat<float>.Rent(ws, hs, cn);
        using PooledMat<float> gradientXWarpedPool = PooledMat<float>.Rent(ws, hs, cn);
        using PooledMat<float> gradientYWarpedPool = PooledMat<float>.Rent(ws, hs, cn);
        using PooledMat<byte> imageMaskPool = PooledMat<byte>.Rent(ws, hs, 1);
        using PooledMat<float> templateZMPool = PooledMat<float>.Rent(ws, hs, cn);
        using PooledMat<float> errorPool = PooledMat<float>.Rent(ws, hs, cn);
        Mat<float> imageWarped = imageWarpedPool.Mat;
        Mat<float> gradientXWarped = gradientXWarpedPool.Mat;
        Mat<float> gradientYWarped = gradientYWarpedPool.Mat;
        Mat<byte> imageMask = imageMaskPool.Mat;
        Mat<float> templateZM = templateZMPool.Mat;
        Mat<float> error = errorPool.Mat;

        double rho = -1;
        double lastRho = -terminationEps;
        for (int i = 1; i <= numberOfIterations && Math.Abs(rho - lastRho) >= terminationEps; i++)
        {
            bool perspective = motionType == MotionTypes.Homography;
            EccWarp.Linear(imageFloat, imageWarped, map, perspective);
            EccWarp.Linear(gradientX, gradientXWarped, map, perspective);
            EccWarp.Linear(gradientY, gradientYWarped, map, perspective);
            EccWarp.Nearest(preMask, imageMask, map, perspective);

            if (templateMask is not null)
            {
                Bitwise.And(imageMask, templtMask, imageMask);
                Mat<byte> zeroMask = ZeroMask(imageMask);
                gradientXWarped.SetTo(new Scalar(0), zeroMask);
                gradientYWarped.SetTo(new Scalar(0), zeroMask);
            }

            FloatReductions.MeanStdDev(imageWarped, imageMask, out Scalar imgMean, out Scalar imgStd);
            FloatReductions.MeanStdDev(templateFloat, imageMask, out Scalar tmpMean, out Scalar tmpStd);

            SubtractMasked(imageWarped, imgMean, imageMask, imageWarped);
            templateZM.Clear();
            SubtractMasked(templateFloat, tmpMean, imageMask, templateZM);

            int validPixels = Reductions.CountNonZero(imageMask);
            double tmpNorm = Math.Sqrt(validPixels * ScalarNormL2Sqr(tmpStd));
            double imgNorm = Math.Sqrt(validPixels * ScalarNormL2Sqr(imgStd));

            switch (motionType)
            {
                case MotionTypes.Affine:
                    JacobianAffine(gradientXWarped, gradientYWarped, xgrid, ygrid, blocks);
                    break;
                case MotionTypes.Homography:
                    JacobianHomography(gradientXWarped, gradientYWarped, xgrid, ygrid, map, blocks);
                    break;
                case MotionTypes.Translation:
                    gradientXWarped.CopyTo(blocks[0]);
                    gradientYWarped.CopyTo(blocks[1]);
                    break;
                default:
                    JacobianEuclidean(gradientXWarped, gradientYWarped, xgrid, ygrid, map, blocks);
                    break;
            }

            ProjectHessian(blocks, hessian);
            LinearAlgebra.Invert(hessian, out Mat<float> hessianInv, DecompTypes.Lu);

            double correlation = FloatReductions.Dot(templateZM, imageWarped);
            lastRho = rho;
            rho = correlation / (imgNorm * tmpNorm);
            if (double.IsNaN(rho))
            {
                throw new InvalidOperationException("NaN encountered.");
            }

            Project(blocks, imageWarped, imageProjection);
            Project(blocks, templateZM, templateProjection);

            Mat<float> imageProjectionHessian = LinearAlgebra.Gemm(hessianInv, imageProjection, 1, null, 0);
            double lambdaN = (imgNorm * imgNorm) - FloatReductions.Dot(imageProjection, imageProjectionHessian);
            double lambdaD = correlation - FloatReductions.Dot(templateProjection, imageProjectionHessian);
            if (lambdaD <= 0.0)
            {
                throw new InvalidOperationException(
                    "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped.");
            }

            double lambda = lambdaN / lambdaD;
            AddEx(templateZM, lambda, imageWarped, -1, error);
            Project(blocks, error, errorProjection);
            Mat<float> deltaP = LinearAlgebra.Gemm(hessianInv, errorProjection, 1, null, 0);
            UpdateWarp(map, deltaP, motionType);
        }

        return rho;
    }

    /// <summary>
    /// <c>convertTo(CV_32F)</c> of an ECC input, then <c>GaussianBlur</c> into
    /// <paramref name="destination"/>. A float input is not copied: the Gaussian blur reads a
    /// region of interest as an isolated image, as it reads the copy.
    /// </summary>
    private static void BlurAsFloat<T>(Mat<T> m, Size gsize, Mat<float> destination)
        where T : unmanaged
    {
        if (typeof(T) == typeof(float))
        {
            Filters.GaussianBlurInto((Mat<float>)(object)m, destination, gsize, 0, 0);
            return;
        }

        using PooledMat<float> f = PooledMat<float>.Rent(m.Width, m.Height, m.Channels);
        Conversions.ConvertInto(m, f.Mat, 1, 0, null);
        Filters.GaussianBlurInto(f.Mat, destination, gsize, 0, 0);
    }

    private static Mat<byte> Ones(Mat<byte> m)
    {
        m.Fill(1);
        return m;
    }

    private static Mat<byte> ZeroMask(Mat<byte> m)
        => Comparison.Compare(m, 0, CmpType.Eq);

    /// <summary>
    /// threshold(mask, 0, 1, BINARY), convert to float, GaussianBlur, scale by 0.5/0.95, convert to
    /// byte with rounding.
    /// </summary>
    private static Mat<byte> BlurredMask(Mat<byte> mask, Size gsize)
    {
        Mat<byte> bin = Thresholding.Threshold(mask, 0, 1, ThresholdType.Binary);
        Mat<float> f = Filters.GaussianBlur(Conversions.ConvertTo<byte, float>(bin), gsize, 0, 0);
        f = Conversions.ConvertTo<float, float>(f, 0.5 / 0.95);
        return Conversions.ConvertTo<float, byte>(f);
    }

    /// <summary><c>subtract(src, mean, dst, mask)</c> for float: selected pixels receive <c>v - (float)mean</c>.</summary>
    private static void SubtractMasked(Mat<float> src, Scalar mean, Mat<byte> mask, Mat<float> dst)
    {
        int cn = src.Channels;
        Span<float> mv = [(float)mean[0], (float)mean[1], (float)mean[2], (float)mean[3]];
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<float> s = src.GetReadOnlyRowSpan(y);
            Span<float> d = dst.GetRowSpan(y);
            ReadOnlySpan<byte> mk = mask.GetReadOnlyRowSpan(y);
            for (int x = 0; x < src.Width; x++)
            {
                if (mk[x] == 0)
                {
                    continue;
                }

                for (int k = 0; k < cn; k++)
                {
                    d[(x * cn) + k] = s[(x * cn) + k] - mv[k];
                }
            }
        }
    }

    /// <summary>
    /// <c>alpha·a + beta·b</c> as <c>MatOp_AddEx::assign</c> evaluates it: add or subtract for unit
    /// weights, <c>scaleAdd</c> when one weight is 1, <c>addWeighted</c> otherwise.
    /// </summary>
    private static void AddEx(Mat<float> a, double alpha, Mat<float> b, double beta, Mat<float> dst)
    {
        if (alpha == 1)
        {
            if (beta == 1)
            {
                Arithmetic.Add(a, b, dst);
            }
            else if (beta == -1)
            {
                Arithmetic.Subtract(a, b, dst);
            }
            else
            {
                Arithmetic.ScaleAdd(b, beta, a, dst);
            }
        }
        else if (beta == 1)
        {
            if (alpha == -1)
            {
                Arithmetic.Subtract(b, a, dst);
            }
            else
            {
                Arithmetic.ScaleAdd(a, alpha, b, dst);
            }
        }
        else
        {
            Arithmetic.AddWeighted(a, alpha, b, beta, 0, dst);
        }
    }

    private static void JacobianAffine(Mat<float> gx, Mat<float> gy, Mat<float> x, Mat<float> y, Mat<float>[] blocks)
    {
        Arithmetic.Multiply(gx, x, blocks[0]);
        Arithmetic.Multiply(gy, x, blocks[1]);
        Arithmetic.Multiply(gx, y, blocks[2]);
        Arithmetic.Multiply(gy, y, blocks[3]);
        gx.CopyTo(blocks[4]);
        gy.CopyTo(blocks[5]);
    }

    private static void JacobianEuclidean(Mat<float> gx, Mat<float> gy, Mat<float> x, Mat<float> y, Mat<float> map, Mat<float>[] blocks)
    {
        double h0 = map.At(0, 0);
        double h1 = map.At(1, 0);
        using PooledMat<float> hatXPool = PooledMat<float>.Rent(x.Width, x.Height, x.Channels);
        using PooledMat<float> hatYPool = PooledMat<float>.Rent(x.Width, x.Height, x.Channels);
        Mat<float> hatX = hatXPool.Mat, hatY = hatYPool.Mat;
        AddEx(x, -h1, y, -h0, hatX);
        AddEx(x, h0, y, -h1, hatY);
        Arithmetic.Multiply(gx, hatX, hatX);
        Arithmetic.Multiply(gy, hatY, hatY);
        Arithmetic.Add(hatX, hatY, blocks[0]);
        gx.CopyTo(blocks[1]);
        gy.CopyTo(blocks[2]);
    }

    private static void JacobianHomography(Mat<float> gx, Mat<float> gy, Mat<float> x, Mat<float> y, Mat<float> map, Mat<float>[] blocks)
    {
        float h0 = map.At(0, 0), h1 = map.At(1, 0), h2 = map.At(2, 0);
        float h3 = map.At(0, 1), h4 = map.At(1, 1), h5 = map.At(2, 1);
        float h6 = map.At(0, 2), h7 = map.At(1, 2);

        int w = x.Width, h = x.Height, cn = x.Channels;
        using PooledMat<float> denPool = PooledMat<float>.Rent(w, h, cn);
        using PooledMat<float> hatXPool = PooledMat<float>.Rent(w, h, cn);
        using PooledMat<float> hatYPool = PooledMat<float>.Rent(w, h, cn);
        using PooledMat<float> g1Pool = PooledMat<float>.Rent(w, h, cn);
        using PooledMat<float> g2Pool = PooledMat<float>.Rent(w, h, cn);
        Mat<float> den = denPool.Mat, hatX = hatXPool.Mat, hatY = hatYPool.Mat, g1 = g1Pool.Mat, g2 = g2Pool.Mat;

        Arithmetic.AddWeighted(x, h2, y, h5, 1.0, den);
        Arithmetic.AddWeighted(x, h0, y, h3, 0.0, hatX);
        Arithmetic.Add(hatX, new Scalar(h6), hatX);
        Arithmetic.AddWeighted(x, h1, y, h4, 0.0, hatY);
        Arithmetic.Add(hatY, new Scalar(h7), hatY);

        FloatReductions.ConvertScale(hatY, -1, hatY);
        Arithmetic.Divide(hatY, den, hatY);
        FloatReductions.ConvertScale(hatX, -1, hatX);
        Arithmetic.Divide(hatX, den, hatX);

        Arithmetic.Divide(gx, den, g1);
        Arithmetic.Divide(gy, den, g2);

        Arithmetic.Multiply(g1, x, blocks[0]);
        Arithmetic.Multiply(g2, x, blocks[1]);
        Arithmetic.Multiply(hatX, g1, hatX);
        Arithmetic.Multiply(hatY, g2, hatY);
        Mat<float> temp = hatX;
        Arithmetic.Add(hatX, hatY, temp);
        Arithmetic.Multiply(temp, x, blocks[2]);
        Arithmetic.Multiply(g1, y, blocks[3]);
        Arithmetic.Multiply(g2, y, blocks[4]);
        Arithmetic.Multiply(temp, y, blocks[5]);
        g1.CopyTo(blocks[6]);
        g2.CopyTo(blocks[7]);
    }

    /// <summary><c>project_onto_jacobian_ECC(jacobian, jacobian, hessian)</c>.</summary>
    private static void ProjectHessian(Mat<float>[] blocks, Mat<float> hessian)
    {
        int n = blocks.Length;
        for (int i = 0; i < n; i++)
        {
            double norm = Math.Sqrt(FloatReductions.NormL2Sqr(blocks[i]));
            hessian.At(i, i) = (float)Math.Pow(norm, 2);
            for (int j = i + 1; j < n; j++)
            {
                hessian.At(i, j) = (float)FloatReductions.Dot(blocks[i], blocks[j]);
                hessian.At(j, i) = hessian.At(i, j);
            }
        }
    }

    /// <summary><c>project_onto_jacobian_ECC(jacobian, image, dst)</c>: <c>dst[i] = image · block i</c>.</summary>
    private static void Project(Mat<float>[] blocks, Mat<float> image, Mat<float> dst)
    {
        for (int i = 0; i < blocks.Length; i++)
        {
            dst.At(i, 0) = (float)FloatReductions.Dot(image, blocks[i]);
        }
    }

    private static void UpdateWarp(Mat<float> map, Mat<float> update, MotionTypes motionType)
    {
        float U(int i) => update.At(i, 0);
        ref float M(int i) => ref map.At(i / 3, i % 3);

        switch (motionType)
        {
            case MotionTypes.Translation:
                M(2) += U(0);
                M(5) += U(1);
                break;
            case MotionTypes.Affine:
                M(0) += U(0);
                M(3) += U(1);
                M(1) += U(2);
                M(4) += U(3);
                M(2) += U(4);
                M(5) += U(5);
                break;
            case MotionTypes.Homography:
                M(0) += U(0);
                M(3) += U(1);
                M(6) += U(2);
                M(1) += U(3);
                M(4) += U(4);
                M(7) += U(5);
                M(2) += U(6);
                M(5) += U(7);
                break;
            default:
                double theta = U(0);
                // ecc.cpp calls asin with a float argument, which selects the float overload.
                theta += MathF.Asin(M(3));
                M(2) += U(1);
                M(5) += U(2);
                M(0) = M(4) = (float)Math.Cos(theta);
                M(3) = (float)Math.Sin(theta);
                M(1) = -M(3);
                break;
        }
    }
}
