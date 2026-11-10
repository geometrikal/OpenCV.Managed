namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins OpenCV 5.0 behaviour where cv2 4.5.5 (the filtering golden reference) differs.
/// OpenCV 5.0 builds the aperture-1/3 Laplacian kernel in double when the source or the
/// destination is double and multiplies it by <c>scale</c> in double (deriv.cpp:718-745, commit
/// <c>1d5f28a44f</c>). cv2 4.5.5 multiplies a float kernel by <c>scale</c> and converts it to double.
/// </summary>
public sealed class FiltersOpenCv5DifferenceTests
{
    [Theory]
    [InlineData("lap_f64_k3_scale01")]
    [InlineData("lap_u8f64_k1_scale01")]
    public void LaplacianDouble_ScaleNotExactInFloat_UsesDoubleKernel(string caseName)
    {
        FiltersGoldenTests.FilterCase c = FiltersGoldenTests.AllCases[caseName];
        Assert.True(c.Suspect);
        double[] input = c.Dtype == "f64"
            ? Doubles(c.Input!)
            : Array.ConvertAll(Convert.FromBase64String(c.Input!), b => (double)b);
        double[] golden = Doubles(c.Expected!);
        int ksize = c.Ksize![0];

        double[] actual = Doubles(Convert.ToBase64String(FiltersGoldenTests.Run(c)));
        double[] opencv5 = Laplacian3x3(input, c.Width, c.Height, c.Channels, ksize, c.Scale, floatKernel: false);
        double[] opencv455 = Laplacian3x3(input, c.Width, c.Height, c.Channels, ksize, c.Scale, floatKernel: true);

        Assert.Equal(opencv5, actual);
        Assert.Equal(opencv455, golden);
        Assert.NotEqual(golden, actual);
    }

    /// <summary>
    /// <c>filter2D</c> with the fixed Laplacian kernel times <paramref name="scale"/>: double sums
    /// over the nonzero taps in row-major order, Reflect101 border, zero delta.
    /// </summary>
    private static double[] Laplacian3x3(
        double[] src, int width, int height, int cn, int ksize, double scale, bool floatKernel)
    {
        double[] k = ksize == 3 ? [2, 0, 2, 0, -8, 0, 2, 0, 2] : [0, 1, 0, 1, -4, 1, 0, 1, 0];
        for (int i = 0; i < 9; i++)
        {
            k[i] = floatKernel ? (float)k[i] * (float)scale : k[i] * scale;
        }

        var dst = new double[src.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int ch = 0; ch < cn; ch++)
                {
                    double acc = 0;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int sy = Borders.BorderInterpolate(y + ky - 1, height, BorderType.Reflect101);
                        for (int kx = 0; kx < 3; kx++)
                        {
                            double coeff = k[(ky * 3) + kx];
                            if (coeff == 0)
                            {
                                continue;
                            }

                            int sx = Borders.BorderInterpolate(x + kx - 1, width, BorderType.Reflect101);
                            acc += coeff * src[(((sy * width) + sx) * cn) + ch];
                        }
                    }

                    dst[(((y * width) + x) * cn) + ch] = acc;
                }
            }
        }

        return dst;
    }

    private static double[] Doubles(string b64)
    {
        byte[] raw = Convert.FromBase64String(b64);
        var result = new double[raw.Length / 8];
        raw.AsSpan().CopyTo(MemoryMarshal.AsBytes(result.AsSpan()));
        return result;
    }
}
