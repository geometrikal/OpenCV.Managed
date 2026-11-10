// This implementation is ported from the OpenCV warp interpolation tables.
//
// Original sources:
//   modules/imgproc/src/imgwarp.cpp (initInterTab1D, initInterTab2D, interpolateLanczos4)
//   OpenCV 4.5.5 modules/imgproc/src/imgwarp.cpp (BicubicTab, interpolateCubic)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// 2-D interpolation weights of the warps and remap on a 1/32-pixel grid. Entry
/// <c>i * 32 + j</c> holds the <c>k × k</c> weights for the fractions <c>i/32</c> (y) and
/// <c>j/32</c> (x), row-major. The 16-bit tables are scaled by 32768 and corrected so that each
/// entry sums to 32768.
/// </summary>
internal sealed class InterTables
{
    public const int Bits = 5;
    public const int Size = 1 << Bits;
    public const int Size2 = Size * Size;
    public const int CoefScale = 1 << 15;

    private const float FltEpsilon = 1.1920929E-07f;

    private static readonly Lazy<InterTables> LinearTables = new(() => new InterTables(2));
    private static readonly Lazy<InterTables> CubicTables = new(() => new InterTables(4));
    private static readonly Lazy<InterTables> Lanczos4Tables = new(() => new InterTables(8));

    private InterTables(int ksize)
    {
        Ksize = ksize;
        var tab1 = new float[Size * ksize];
        float scale = 1f / Size;
        Span<float> c = stackalloc float[8];
        for (int i = 0; i < Size; i++)
        {
            Interpolate(ksize, i * scale, c);
            c[..ksize].CopyTo(tab1.AsSpan(i * ksize));
        }

        int kk = ksize * ksize;
        Float = new float[Size2 * kk];

        // The sum correction reads past the current entry for ksize 2 (as OpenCV does with its flat array).
        short[] itab = new short[(Size2 * kk) + 16];
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                int o = ((i * Size) + j) * kk;
                int isum = 0;
                for (int k1 = 0; k1 < ksize; k1++)
                {
                    float vy = tab1[(i * ksize) + k1];
                    for (int k2 = 0; k2 < ksize; k2++)
                    {
                        float v = vy * tab1[(j * ksize) + k2];
                        Float[o + (k1 * ksize) + k2] = v;
                        short iv = (short)Math.Clamp(Core.Internal.ScalarConverter.CvRound(v * CoefScale), short.MinValue, short.MaxValue);
                        itab[o + (k1 * ksize) + k2] = iv;
                        isum += iv;
                    }
                }

                if (isum != CoefScale)
                {
                    int diff = isum - CoefScale;
                    int h = ksize / 2;
                    int mk1 = h, mk2 = h, bk1 = h, bk2 = h;
                    for (int k1 = h; k1 < h + 2; k1++)
                    {
                        for (int k2 = h; k2 < h + 2; k2++)
                        {
                            if (itab[o + (k1 * ksize) + k2] < itab[o + (mk1 * ksize) + mk2])
                            {
                                mk1 = k1;
                                mk2 = k2;
                            }
                            else if (itab[o + (k1 * ksize) + k2] > itab[o + (bk1 * ksize) + bk2])
                            {
                                bk1 = k1;
                                bk2 = k2;
                            }
                        }
                    }

                    if (diff < 0)
                    {
                        itab[o + (bk1 * ksize) + bk2] = (short)(itab[o + (bk1 * ksize) + bk2] - diff);
                    }
                    else
                    {
                        itab[o + (mk1 * ksize) + mk2] = (short)(itab[o + (mk1 * ksize) + mk2] - diff);
                    }
                }
            }
        }

        Fixed = itab;
    }

    public int Ksize { get; }

    /// <summary>Single-precision weights.</summary>
    public float[] Float { get; }

    /// <summary>Weights scaled by 32768 (16-bit), with the sum correction.</summary>
    public short[] Fixed { get; }

    public static InterTables Linear => LinearTables.Value;

    public static InterTables Cubic => CubicTables.Value;

    public static InterTables Lanczos4 => Lanczos4Tables.Value;

    private static void Interpolate(int ksize, float x, Span<float> c)
    {
        if (ksize == 2)
        {
            c[0] = 1f - x;
            c[1] = x;
        }
        else if (ksize == 4)
        {
            ResizeCoefficients.InterpolateCubic(x, c);
        }
        else
        {
            Lanczos4Weights(x, c);
        }
    }

    /// <summary>
    /// <c>interpolateLanczos4</c> of imgwarp: no guard for single taps; <c>x &lt; FLT_EPSILON</c>
    /// gives the unit impulse at tap 3.
    /// </summary>
    private static void Lanczos4Weights(float x, Span<float> c)
    {
        const double s45 = 0.70710678118654752440084436210485;
        ReadOnlySpan<double> cs = [1, 0, -s45, -s45, 0, 1, s45, -s45, -1, 0, s45, s45, 0, -1, -s45, s45];
        if (x < FltEpsilon)
        {
            c[..8].Clear();
            c[3] = 1;
            return;
        }

        float sum = 0;
        double y0 = -(x + 3) * Math.PI * 0.25;
        double s0 = Math.Sin(y0), c0 = Math.Cos(y0);
        for (int i = 0; i < 8; i++)
        {
            double y = -(x + 3 - i) * Math.PI * 0.25;
            c[i] = (float)(((cs[2 * i] * s0) + (cs[(2 * i) + 1] * c0)) / (y * y));
            sum += c[i];
        }

        sum = 1f / sum;
        for (int i = 0; i < 8; i++)
        {
            c[i] *= sum;
        }
    }
}
