// This implementation is ported from the OpenCV derivative kernel generators.
//
// Original sources:
//   modules/imgproc/src/deriv.cpp (getSobelKernels, getScharrKernels)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Sobel and Scharr 1-D kernel generation (<c>getDerivKernels</c>).
/// Kernels are produced at float32 precision — OpenCV's ktype for 8-bit and
/// float32 images.
/// </summary>
internal static class DerivKernels
{
    /// <summary>
    /// One axis of OpenCV's <c>getScharrKernels</c> (deriv.cpp:55).
    /// </summary>
    public static float[] GetScharrKernel(int order, bool normalize)
    {
        if (order != 0 && order != 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(order), order, "Scharr derivative orders must satisfy dx >= 0, dy >= 0, dx + dy == 1.");
        }

        int[] kerI = order == 0 ? [3, 10, 3] : [-1, 0, 1];
        double scale = !normalize || order == 1 ? 1.0 : 1.0 / 32;
        return Convert(kerI, scale);
    }

    /// <summary>
    /// One axis of OpenCV's <c>getSobelKernels</c> (deriv.cpp:86). The caller
    /// passes the per-axis kernel size (after the <c>ksize==1 &amp;&amp;
    /// order&gt;0 → 3</c> substitution) and derivative order.
    /// </summary>
    public static float[] GetSobelKernel(int ksize, int order, bool normalize)
    {
        if (ksize % 2 == 0 || ksize > 31)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ksize), ksize, "The kernel size must be odd and not larger than 31.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(order);
        if (ksize <= order)
        {
            throw new ArgumentOutOfRangeException(
                nameof(order), order, "The kernel size must be larger than the derivative order.");
        }

        int[] kerI = new int[ksize + 1];
        if (ksize == 1)
        {
            kerI[0] = 1;
        }
        else if (ksize == 3)
        {
            if (order == 0)
            {
                kerI[0] = 1;
                kerI[1] = 2;
                kerI[2] = 1;
            }
            else if (order == 1)
            {
                kerI[0] = -1;
                kerI[1] = 0;
                kerI[2] = 1;
            }
            else
            {
                kerI[0] = 1;
                kerI[1] = -2;
                kerI[2] = 1;
            }
        }
        else
        {
            // Pascal-triangle construction, verbatim from OpenCV.
            kerI[0] = 1;
            for (int i = 0; i < ksize; i++)
            {
                kerI[i + 1] = 0;
            }

            int oldval;
            int newval;
            for (int i = 0; i < ksize - order - 1; i++)
            {
                oldval = kerI[0];
                for (int j = 1; j <= ksize; j++)
                {
                    newval = kerI[j] + kerI[j - 1];
                    kerI[j - 1] = oldval;
                    oldval = newval;
                }
            }

            for (int i = 0; i < order; i++)
            {
                oldval = -kerI[0];
                for (int j = 1; j <= ksize; j++)
                {
                    newval = kerI[j - 1] - kerI[j];
                    kerI[j - 1] = oldval;
                    oldval = newval;
                }
            }
        }

        double scale = !normalize ? 1.0 : 1.0 / (1 << (ksize - order - 1));
        return Convert(kerI.AsSpan(0, ksize), scale);
    }

    private static float[] Convert(ReadOnlySpan<int> kerI, double scale)
    {
        var kernel = new float[kerI.Length];
        for (int i = 0; i < kerI.Length; i++)
        {
            // temp.convertTo(kernel, CV_32F, scale); the scales used here are
            // exact powers of two, so the rounding mode of the intermediate
            // multiply cannot matter.
            kernel[i] = (float)(kerI[i] * scale);
        }

        return kernel;
    }
}
