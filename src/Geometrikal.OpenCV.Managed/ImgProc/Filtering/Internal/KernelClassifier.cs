// This implementation is ported from the OpenCV separable-filter kernel
// classification.
//
// Original sources:
//   modules/imgproc/src/filter.dispatch.cpp (getKernelType, createBitExactKernel_32S)
//   modules/imgproc/include/opencv2/imgproc.hpp (KERNEL_* constants)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Kernel property flags mirroring OpenCV's <c>KERNEL_GENERAL</c> /
/// <c>KERNEL_SYMMETRICAL</c> / <c>KERNEL_ASYMMETRICAL</c> /
/// <c>KERNEL_SMOOTH</c> / <c>KERNEL_INTEGER</c>.
/// </summary>
[Flags]
internal enum KernelKind
{
    General = 0,
    Symmetrical = 1,
    Asymmetrical = 2,
    Smooth = 4,
    Integer = 8,
}

/// <summary>
/// Ports OpenCV's <c>getKernelType</c> and <c>createBitExactKernel_32S</c>
/// for 1-D separable filter kernels.
/// </summary>
internal static class KernelClassifier
{
    /// <summary>FLT_EPSILON (2^-23), used by OpenCV's smooth-kernel test.</summary>
    private const double FltEpsilon = 1.1920928955078125e-7;

    /// <summary>
    /// Classifies a 1-D kernel exactly as OpenCV's <c>getKernelType</c>
    /// (filter.dispatch.cpp:233). The symmetry candidate flags are only set
    /// when the anchor is centred, matching the 1-D case of the original.
    /// </summary>
    public static KernelKind GetKernelType(ReadOnlySpan<double> kernel, int anchor)
    {
        int sz = kernel.Length;
        KernelKind type = KernelKind.Smooth | KernelKind.Integer;
        if ((anchor * 2) + 1 == sz)
        {
            type |= KernelKind.Symmetrical | KernelKind.Asymmetrical;
        }

        double sum = 0;
        for (int i = 0; i < sz; i++)
        {
            double a = kernel[i];
            double b = kernel[sz - i - 1];
            if (a != b)
            {
                type &= ~KernelKind.Symmetrical;
            }

            if (a != -b)
            {
                type &= ~KernelKind.Asymmetrical;
            }

            if (a < 0)
            {
                type &= ~KernelKind.Smooth;
            }

            if (a != ScalarConverter.CvRound(a))
            {
                type &= ~KernelKind.Integer;
            }

            sum += a;
        }

        if (Math.Abs(sum - 1) > FltEpsilon * (Math.Abs(sum) + 1))
        {
            type &= ~KernelKind.Smooth;
        }

        return type;
    }

    /// <summary>
    /// Quantizes a kernel to int32 with <c>2^bits</c> scaling and validates the
    /// quantization error, exactly as OpenCV's <c>createBitExactKernel_32S</c>
    /// (filter.dispatch.cpp:296). Returns null when the kernel cannot be
    /// represented bit-exactly.
    /// </summary>
    public static int[]? TryCreateBitExactKernel32S(ReadOnlySpan<double> kernel, int bits)
    {
        int ksize = kernel.Length;
        var result = new int[ksize];
        double eps = 10 * FltEpsilon * (1 << bits);
        for (int i = 0; i < ksize; i++)
        {
            double approxValue = kernel[i] * (1 << bits);
            int bitExactValue = ScalarConverter.CvRound(approxValue);
            if (Math.Abs(approxValue - bitExactValue) > eps)
            {
                return null;
            }

            result[i] = bitExactValue;
        }

        return result;
    }
}
