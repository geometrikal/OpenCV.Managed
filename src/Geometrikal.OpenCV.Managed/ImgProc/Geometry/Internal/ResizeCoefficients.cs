// This implementation is ported from the OpenCV resize coefficient generation.
//
// Original source:
//   modules/imgproc/src/resize.cpp (interpolateCubic, interpolateLanczos4, hal::resize)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Kernel shape of the separable resize path.</summary>
internal enum ResizeKernel
{
    Linear,
    Cubic,
    Lanczos4,
}

/// <summary>
/// Offsets and weights of the separable resize path (<c>hal::resize</c>). Arrays are rented;
/// <see cref="Dispose"/> returns them.
/// </summary>
internal sealed class ResizeCoefficients : IDisposable
{
    public const int CoefBits = 11;
    public const int CoefScale = 1 << CoefBits;

    private ResizeCoefficients(int ksize, int width, int dheight)
    {
        Ksize = ksize;
        XOffsets = ArrayPool<int>.Shared.Rent(width);
        Alpha = ArrayPool<float>.Shared.Rent(width * ksize);
        IAlpha = ArrayPool<short>.Shared.Rent(width * ksize);
        YOffsets = ArrayPool<int>.Shared.Rent(dheight);
        Beta = ArrayPool<float>.Shared.Rent(dheight * ksize);
        IBeta = ArrayPool<short>.Shared.Rent(dheight * ksize);
    }

    public int Ksize { get; }

    /// <summary>Source element of tap <c>ksize/2 - 1</c> for each destination element.</summary>
    public int[] XOffsets { get; }

    public float[] Alpha { get; }

    public short[] IAlpha { get; }

    public int[] YOffsets { get; }

    public float[] Beta { get; }

    public short[] IBeta { get; }

    /// <summary>First destination pixel whose taps are all inside the source.</summary>
    public int XMin { get; private set; }

    /// <summary>First destination pixel at the right whose taps leave the source.</summary>
    public int XMax { get; private set; }

    public static ResizeCoefficients Create(
        int srcWidth, int dstWidth, int dstHeight, int cn, double invScaleX, double invScaleY, ResizeKernel kernel, bool areaMode)
    {
        int ksize = kernel switch { ResizeKernel.Cubic => 4, ResizeKernel.Lanczos4 => 8, _ => 2 };
        int ksize2 = ksize / 2;
        int width = dstWidth * cn;
        var c = new ResizeCoefficients(ksize, width, dstHeight);
        double scaleX = 1.0 / invScaleX, scaleY = 1.0 / invScaleY;
        bool clampTaps = kernel == ResizeKernel.Linear;
        Span<float> cbuf = stackalloc float[8];
        int xmin = 0, xmax = dstWidth;

        for (int dx = 0; dx < dstWidth; dx++)
        {
            Position(dx, scaleX, invScaleX, areaMode, out int sx, out float fx);
            if (sx < ksize2 - 1)
            {
                xmin = dx + 1;
                if (sx < 0 && clampTaps)
                {
                    fx = 0;
                    sx = 0;
                }
            }

            if (sx + ksize2 >= srcWidth)
            {
                xmax = Math.Min(xmax, dx);
                if (sx >= srcWidth - 1 && clampTaps)
                {
                    fx = 0;
                    sx = srcWidth - 1;
                }
            }

            for (int k = 0; k < cn; k++)
            {
                c.XOffsets[(dx * cn) + k] = (sx * cn) + k;
            }

            Weights(kernel, fx, cbuf);
            for (int k = 0; k < cn; k++)
            {
                int o = ((dx * cn) + k) * ksize;
                for (int t = 0; t < ksize; t++)
                {
                    c.Alpha[o + t] = cbuf[t];
                    c.IAlpha[o + t] = ToFixed(cbuf[t]);
                }
            }
        }

        for (int dy = 0; dy < dstHeight; dy++)
        {
            Position(dy, scaleY, invScaleY, areaMode, out int sy, out float fy);
            c.YOffsets[dy] = sy;
            Weights(kernel, fy, cbuf);
            for (int t = 0; t < ksize; t++)
            {
                c.Beta[(dy * ksize) + t] = cbuf[t];
                c.IBeta[(dy * ksize) + t] = ToFixed(cbuf[t]);
            }
        }

        c.XMin = xmin;
        c.XMax = xmax;
        return c;
    }

    public void Dispose()
    {
        ArrayPool<int>.Shared.Return(XOffsets);
        ArrayPool<float>.Shared.Return(Alpha);
        ArrayPool<short>.Shared.Return(IAlpha);
        ArrayPool<int>.Shared.Return(YOffsets);
        ArrayPool<float>.Shared.Return(Beta);
        ArrayPool<short>.Shared.Return(IBeta);
    }

    /// <summary><c>saturate_cast&lt;short&gt;(c * INTER_RESIZE_COEF_SCALE)</c>.</summary>
    private static short ToFixed(float c) => (short)Math.Clamp(ScalarConverter.CvRound(c * CoefScale), short.MinValue, short.MaxValue);

    private static void Position(int d, double scale, double invScale, bool areaMode, out int s, out float f)
    {
        if (!areaMode)
        {
            f = (float)(((d + 0.5) * scale) - 0.5);
            s = CvFloor(f);
            f -= s;
        }
        else
        {
            s = ScalarConverter.CvFloor(d * scale);
            f = (float)((d + 1) - ((s + 1) * invScale));
            f = f <= 0 ? 0f : f - CvFloor(f);
        }
    }

    /// <summary><c>cvFloor(float)</c>.</summary>
    internal static int CvFloor(float v) => ScalarConverter.CvFloor(v);

    private static void Weights(ResizeKernel kernel, float x, Span<float> c)
    {
        switch (kernel)
        {
            case ResizeKernel.Cubic:
                InterpolateCubic(x, c);
                break;
            case ResizeKernel.Lanczos4:
                InterpolateLanczos4(x, c);
                break;
            default:
                c[0] = 1f - x;
                c[1] = x;
                break;
        }
    }

    /// <summary><c>interpolateCubic</c> with <c>A = -0.75</c>, in single precision.</summary>
    public static void InterpolateCubic(float x, Span<float> c)
    {
        const float a = -0.75f;
        c[0] = ((((a * (x + 1)) - (5 * a)) * (x + 1)) + (8 * a)) * (x + 1) - (4 * a);
        c[1] = (((((a + 2) * x) - (a + 3)) * x) * x) + 1;
        c[2] = (((((a + 2) * (1 - x)) - (a + 3)) * (1 - x)) * (1 - x)) + 1;
        c[3] = 1f - c[0] - c[1] - c[2];
    }

    private static readonly double[] Lanczos4Cs = BuildLanczosCs();

    private static double[] BuildLanczosCs()
    {
        const double s45 = 0.70710678118654752440084436210485;
        return [1, 0, -s45, -s45, 0, 1, s45, -s45, -1, 0, s45, s45, 0, -1, -s45, s45];
    }

    /// <summary>
    /// <c>interpolateLanczos4</c> of resize: the sum is normalised in single precision and a tap at
    /// distance below <c>1e-6</c> gets the weight <c>1e30</c> before normalisation.
    /// </summary>
    public static void InterpolateLanczos4(float x, Span<float> c)
    {
        double[] cs = Lanczos4Cs;
        float sum = 0;
        double y0 = -(x + 3) * Math.PI * 0.25;
        double s0 = Math.Sin(y0), c0 = Math.Cos(y0);
        for (int i = 0; i < 8; i++)
        {
            float yi = x + 3 - i;
            if (MathF.Abs(yi) >= 1e-6f)
            {
                double y = -yi * Math.PI * 0.25;
                c[i] = (float)(((cs[2 * i] * s0) + (cs[(2 * i) + 1] * c0)) / (y * y));
            }
            else
            {
                c[i] = 1e30f;
            }

            sum += c[i];
        }

        sum = 1f / sum;
        for (int i = 0; i < 8; i++)
        {
            c[i] *= sum;
        }
    }
}
