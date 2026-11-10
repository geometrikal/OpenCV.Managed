// This implementation is ported from the OpenCV Gaussian kernel computation.
//
// Original sources:
//   modules/imgproc/src/smooth.dispatch.cpp
//     (getGaussianKernelBitExact, getGaussianKernelFixedPoint_ED,
//      getGaussianKernel, createGaussianKernels)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Bit-exact Gaussian kernel computation. OpenCV performs these calculations
/// in <c>softdouble</c>; every operation used is a correctly-rounded IEEE-754
/// binary64 op (see <see cref="SoftDoubleMath"/>), so plain <c>double</c>
/// arithmetic reproduces the results bit-for-bit.
/// </summary>
internal static class GaussianKernels
{
    /// <summary>
    /// OpenCV's <c>getGaussianKernelBitExact</c> (smooth.dispatch.cpp:79).
    /// The kernel sums to 1 (in the bit-exact sense of the original).
    /// </summary>
    public static double[] GetKernelBitExact(int n, double sigma)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);

        if (sigma <= 0)
        {
            // Hardcoded "small gaussian" kernels; all values are exact dyadic
            // rationals, identical to OpenCV's raw-bit constants.
            switch (n)
            {
                case 1:
                    return [1.0];
                case 3:
                    return [0.25, 0.5, 0.25];
                case 5:
                    return [0.0625, 0.25, 0.375, 0.25, 0.0625];
                case 7:
                    return [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125];
                case 9:
                    return
                    [
                        0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375,
                        0.19921875, 0.1171875, 0.05078125, 0.015625,
                    ];
                default:
                    break;
            }
        }

        // sigmaX = sigma > 0 ? sigma : n*0.15 + 0.35 — OpenCV uses softdouble
        // mulAdd (a fused multiply-add).
        double sigmaX = sigma > 0 ? sigma : Math.FusedMultiplyAdd(n, 0.15, 0.35);
        double scale2X = -0.125 / (sigmaX * sigmaX);

        int n2 = (n - 1) / 2;
        var values = new double[n2];
        double sum = 0.0;
        for (int i = 0, x = 1 - n; i < n2; i++, x += 2)
        {
            double t = SoftDoubleMath.Exp(x * x * scale2X);
            values[i] = t;
            sum += t;
        }

        sum *= 2.0;
        sum += 1.0;
        if ((n & 1) == 0)
        {
            sum += 1.0;
        }

        double mul1 = 1.0 / sum;

        var result = new double[n];
        for (int i = 0; i < n2; i++)
        {
            double t = values[i] * mul1;
            result[i] = t;
            result[n - 1 - i] = t;
        }

        result[n2] = mul1; // softdouble::one() * mul1
        if ((n & 1) == 0)
        {
            result[n2 + 1] = result[n2];
        }

        return result;
    }

    /// <summary>The bit-exact kernel cast to float32 (OpenCV's CV_32F ktype).</summary>
    public static float[] GetKernelSingle(int n, double sigma)
    {
        double[] kd = GetKernelBitExact(n, sigma);
        var result = new float[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = (float)kd[i];
        }

        return result;
    }

    /// <summary>
    /// OpenCV's <c>getGaussianKernelFixedPoint_ED</c> with 8 fraction bits
    /// (the <c>ufixedpoint16</c> kernel used by the bit-exact 8-bit Gaussian):
    /// error-diffusion rounding of the sides, with the centre coefficient
    /// forced to <c>256 − Σ(sides)</c> so the raw kernel sums to exactly 256.
    /// Returns Q8.8 raw values.
    /// </summary>
    public static ushort[] GetKernelFixedPointQ8(int n, double sigma)
    {
        if ((n & 1) == 0)
        {
            throw new ArgumentException("The fixed-point Gaussian kernel size must be odd.", nameof(n));
        }

        double[] kd = GetKernelBitExact(n, sigma);
        const long FractionMultiplier = 1L << 8;

        var fixedValues = new long[n];
        int n2 = n / 2;
        double err = 0.0;
        long sum = 0;
        for (int i = 0; i < n2; i++)
        {
            double adj = (kd[i] * FractionMultiplier) + err;
            long v0 = ScalarConverter.CvRound(adj);
            err = adj - v0;
            fixedValues[i] = v0;
            fixedValues[n - 1 - i] = v0;
            sum += v0;
        }

        sum *= 2;
        fixedValues[n2] = FractionMultiplier - sum;

        var raw = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            // FT::fromRaw((raw_t)value) — truncating like the original.
            raw[i] = unchecked((ushort)fixedValues[i]);
        }

        return raw;
    }

    /// <summary>
    /// The kernel-size-from-sigma rule of OpenCV's
    /// <c>createGaussianKernels</c> (smooth.dispatch.cpp:277), including the
    /// odd/positive validation.
    /// </summary>
    public static Size ComputeKernelSize(Size ksize, double sigma1, double sigma2, bool byteDepth)
    {
        if (sigma2 <= 0)
        {
            sigma2 = sigma1;
        }

        if (ksize.Width <= 0 && sigma1 > 0)
        {
            ksize = new Size(ScalarConverter.CvRound((sigma1 * (byteDepth ? 3 : 4) * 2) + 1) | 1, ksize.Height);
        }

        if (ksize.Height <= 0 && sigma2 > 0)
        {
            ksize = new Size(ksize.Width, ScalarConverter.CvRound((sigma2 * (byteDepth ? 3 : 4) * 2) + 1) | 1);
        }

        if (ksize.Width <= 0 || ksize.Width % 2 != 1 ||
            ksize.Height <= 0 || ksize.Height % 2 != 1)
        {
            throw new ArgumentException(
                "The Gaussian kernel size must be positive and odd (or zero with a positive sigma).",
                nameof(ksize));
        }

        return ksize;
    }
}
