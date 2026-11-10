// This file is a managed port of parts of OpenCV:
//   modules/core/src/mathfuncs.cpp (pow, iPow_i, iPow_f)
//   modules/core/src/mathfuncs_core.simd.hpp (cartToPolar32f, SinCos_32f, polarToCart, invSqrt32f, exp, log)
//   modules/core/include/opencv2/core/hal/intrin_math.hpp (v_sincos_default_32f)
//   modules/core/include/opencv2/core/hal/intrin_sse.hpp (v_invsqrt)
// OpenCV is licensed under the Apache License 2.0. Copyright (C) 2000-2024, Intel Corporation,
// all rights reserved, and OpenCV contributors.

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Element kernels of the OpenCV math functions with the OpenCV split between the vector body and
/// the scalar tail. The vector body runs on blocks of two SSE registers (8 floats), as the
/// SSE-baseline build: when a span has at least 8 elements, the last block overlaps the previous
/// one; a span of fewer than 8 elements uses the scalar code only.
/// </summary>
internal static class MathKernels
{
    /// <summary>Elements per vector iteration of the SSE-baseline build (two 4-lane registers).</summary>
    public const int FloatBlock = 8;

    private const int SinCosTableSize = 64;

    private static readonly double[] SinTable =
    [
        0.00000000000000000000, 0.09801714032956060400,
        0.19509032201612825000, 0.29028467725446233000,
        0.38268343236508978000, 0.47139673682599764000,
        0.55557023301960218000, 0.63439328416364549000,
        0.70710678118654746000, 0.77301045336273699000,
        0.83146961230254524000, 0.88192126434835494000,
        0.92387953251128674000, 0.95694033573220894000,
        0.98078528040323043000, 0.99518472667219682000,
        1.00000000000000000000, 0.99518472667219693000,
        0.98078528040323043000, 0.95694033573220894000,
        0.92387953251128674000, 0.88192126434835505000,
        0.83146961230254546000, 0.77301045336273710000,
        0.70710678118654757000, 0.63439328416364549000,
        0.55557023301960218000, 0.47139673682599786000,
        0.38268343236508989000, 0.29028467725446239000,
        0.19509032201612861000, 0.09801714032956082600,
        0.00000000000000012246, -0.09801714032956059000,
        -0.19509032201612836000, -0.29028467725446211000,
        -0.38268343236508967000, -0.47139673682599764000,
        -0.55557023301960196000, -0.63439328416364527000,
        -0.70710678118654746000, -0.77301045336273666000,
        -0.83146961230254524000, -0.88192126434835494000,
        -0.92387953251128652000, -0.95694033573220882000,
        -0.98078528040323032000, -0.99518472667219693000,
        -1.00000000000000000000, -0.99518472667219693000,
        -0.98078528040323043000, -0.95694033573220894000,
        -0.92387953251128663000, -0.88192126434835505000,
        -0.83146961230254546000, -0.77301045336273688000,
        -0.70710678118654768000, -0.63439328416364593000,
        -0.55557023301960218000, -0.47139673682599792000,
        -0.38268343236509039000, -0.29028467725446250000,
        -0.19509032201612872000, -0.09801714032956050600,
    ];

    private static readonly double K2 = 2 * Math.PI / SinCosTableSize;
    private static readonly double SinA0 = -0.166630293345647 * K2 * K2 * K2;
    private static readonly double SinA2 = K2;
    private static readonly double CosA0 = -0.499818138450326 * K2 * K2;

    /// <summary>Radians per degree as the OpenCV <c>(float)(CV_PI/180)</c>.</summary>
    public static readonly float RadiansPerDegree = (float)(Math.PI / 180);

    /// <summary>
    /// OpenCV <c>hal::cartToPolar32f</c>: magnitude <c>sqrt(x·x + y·y)</c> and the
    /// <c>atan_f32</c> angle. The vector and scalar forms give the same results without FMA.
    /// </summary>
    public static void CartToPolar32f(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> mag, Span<float> angle, bool degrees)
    {
        float scale = degrees ? 1f : RadiansPerDegree;
        for (int i = 0; i < mag.Length; i++)
        {
            float x0 = x[i], y0 = y[i];
            mag[i] = MathF.Sqrt((x0 * x0) + (y0 * y0));
            angle[i] = FastMath.FastAtan2(y0, x0) * scale;
        }
    }

    /// <summary>OpenCV <c>hal::fastAtan32f</c> in degrees or radians.</summary>
    public static void FastAtan32f(ReadOnlySpan<float> y, ReadOnlySpan<float> x, Span<float> angle, bool degrees)
    {
        float scale = degrees ? 1f : RadiansPerDegree;
        for (int i = 0; i < angle.Length; i++)
        {
            angle[i] = FastMath.FastAtan2(y[i], x[i]) * scale;
        }
    }

    /// <summary>
    /// OpenCV <c>SinCos_32f</c>: <c>cos</c> and <c>sin</c> of the angles, times the magnitudes when
    /// given. Vector blocks use the Cephes <c>v_sincos</c>; the scalar code uses a 64-entry table
    /// with a cubic correction. The two forms differ in the last bits.
    /// </summary>
    public static void SinCos32f(ReadOnlySpan<float> mag, bool hasMagnitude, ReadOnlySpan<float> angle, Span<float> cos, Span<float> sin, bool degrees)
    {
        int len = angle.Length;
        int i = 0;
        if (len >= FloatBlock)
        {
            float scale = degrees ? (float)Math.PI / 180f : 1f;
            for (; i < len; i += FloatBlock)
            {
                if (i + FloatBlock > len)
                {
                    i = len - FloatBlock;
                }

                for (int k = i; k < i + FloatBlock; k++)
                {
                    SinCosCephes(angle[k] * scale, out float s, out float c);
                    if (hasMagnitude)
                    {
                        c *= mag[k];
                        s *= mag[k];
                    }

                    cos[k] = c;
                    sin[k] = s;
                }
            }

            return;
        }

        double k1 = degrees ? SinCosTableSize / 360.0 : SinCosTableSize / (2 * Math.PI);
        for (; i < len; i++)
        {
            double t = angle[i] * k1;
            int it = ScalarConverter.CvRound(t);
            t -= it;
            int sinIndex = it & (SinCosTableSize - 1);
            int cosIndex = ((SinCosTableSize / 4) - sinIndex) & (SinCosTableSize - 1);

            double sinB = ((SinA0 * t * t) + SinA2) * t;
            double cosB = (CosA0 * t * t) + 1;
            double sinA = SinTable[sinIndex];
            double cosA = SinTable[cosIndex];
            double sinValue = (sinA * cosB) + (cosA * sinB);
            double cosValue = (cosA * cosB) - (sinA * sinB);
            if (hasMagnitude)
            {
                double m = mag[i];
                sinValue *= m;
                cosValue *= m;
            }

            sin[i] = (float)sinValue;
            cos[i] = (float)cosValue;
        }
    }

    /// <summary>
    /// One lane of OpenCV <c>v_sincos_default_32f</c> (Cephes single precision) without FMA:
    /// every <c>v_fma(a, b, c)</c> is <c>a·b + c</c> with two roundings.
    /// </summary>
    public static void SinCosCephes(float x, out float ysin, out float ycos)
    {
        const float fopi = 1.27323954473516f;
        const float minusDp1 = -0.78515625f;
        const float minusDp2 = -2.4187564849853515625E-4f;
        const float minusDp3 = -3.77489497744594108E-8f;
        const float sincofP0 = -1.9515295891E-4f;
        const float sincofP1 = 8.3321608736E-3f;
        const float sincofP2 = -1.6666654611E-1f;
        const float coscofP0 = 2.443315711809948E-5f;
        const float coscofP1 = -1.388731625493765E-3f;
        const float coscofP2 = 4.166664568298827E-2f;

        bool signSin = x < 0;
        float vx = MathF.Abs(x);
        float vy = vx * fopi;

        int emm2 = TruncateToInt32(vy);
        emm2 += 1;
        emm2 &= ~1;
        vy = emm2;

        bool polyMask = (emm2 & 2) == 0;
        vx = (vy * minusDp1) + vx;
        vx = (vy * minusDp2) + vx;
        vx = (vy * minusDp3) + vx;

        // sign_mask_sin ^= (emm2 & 4) == 0; the select keeps ysin where the mask is set.
        bool keepSin = signSin ^ ((emm2 & 4) == 0);
        bool negateCos = ((emm2 - 2) & 4) == 0;

        float xx = vx * vx;
        float y1 = (coscofP0 * xx) + coscofP1;
        y1 = (y1 * xx) + coscofP2;
        y1 = (y1 * xx) + -0.5f;
        y1 = (y1 * xx) + 1f;

        float y2 = (sincofP0 * xx) + sincofP1;
        y2 = (y2 * xx) + sincofP2;
        y2 *= xx;
        y2 = (y2 * vx) + vx;

        float s = polyMask ? y2 : y1;
        float c = polyMask ? y1 : y2;
        s = keepSin ? s : FlipSign(s);
        c = negateCos ? FlipSign(c) : c;

        if (vx == float.PositiveInfinity || float.IsNaN(x))
        {
            s = c = BitConverter.Int32BitsToSingle(0x7fc00000);
        }

        ysin = s;
        ycos = c;
    }

    /// <summary>
    /// OpenCV <c>hal::invSqrt32f</c>: vector blocks use <c>rsqrtps</c> with one Newton step (as the
    /// SSE <c>v_invsqrt</c>); the scalar code is <c>1/sqrt(x)</c>.
    /// </summary>
    public static void InvSqrt32f(ReadOnlySpan<float> src, Span<float> dst)
    {
        int len = dst.Length;
        int i = 0;
        if (len >= FloatBlock)
        {
            for (; i < len; i += FloatBlock)
            {
                if (i + FloatBlock > len)
                {
                    i = len - FloatBlock;
                }

                for (int k = i; k < i + FloatBlock; k++)
                {
                    float t = src[k];
                    float h = t * 0.5f;
                    t = ReciprocalSqrtEstimate(t);
                    dst[k] = t * (1.5f - ((t * t) * h));
                }
            }

            return;
        }

        for (; i < len; i++)
        {
            dst[i] = 1 / MathF.Sqrt(src[i]);
        }
    }

    /// <summary>
    /// OpenCV <c>iPow_i</c>: integer power of integer elements in the working type
    /// (<c>unsigned</c> for unsigned elements, <c>int</c> for signed), saturated to the element type.
    /// </summary>
    public static T IntegerPower<T>(T value, int power)
        where T : unmanaged
    {
        if (power < 0)
        {
            int v = Saturate.ToInt(value);
            if (Math.Abs(v) > 2)
            {
                return default;
            }

            // tab[v + 2] for v in [-2, 2].
            return (v + 2) switch
            {
                0 => Saturate.FromInt<T>(power == -1 ? -1 : 0),
                1 => Saturate.FromInt<T>((power & 1) != 0 ? -1 : 1),
                2 => Saturate.FromDouble<T>(Saturate.MaxValue<T>()),
                3 => Saturate.FromInt<T>(1),
                _ => Saturate.FromInt<T>(power == -1 ? 1 : 0),
            };
        }

        if (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort))
        {
            uint a = 1, b = (uint)Saturate.ToInt(value);
            int p = power;
            unchecked
            {
                while (p > 1)
                {
                    if ((p & 1) != 0)
                    {
                        a *= b;
                    }

                    b *= b;
                    p >>= 1;
                }

                a *= b;
            }

            // saturate_cast<T>(unsigned) is min(a, max).
            return Saturate.FromInt<T>((int)Math.Min(a, int.MaxValue));
        }
        else
        {
            int a = 1, b = Saturate.ToInt(value);
            int p = power;
            unchecked
            {
                while (p > 1)
                {
                    if ((p & 1) != 0)
                    {
                        a *= b;
                    }

                    b *= b;
                    p >>= 1;
                }

                a *= b;
            }

            return Saturate.FromInt<T>(a);
        }
    }

    /// <summary>OpenCV <c>iPow_f</c> for <c>float</c>: repeated squaring, reciprocal first for negative powers.</summary>
    public static float IntegerPower(float value, int power0)
    {
        float a = 1, b = value;
        int p = Math.Abs(power0);
        if (power0 < 0)
        {
            b = 1 / b;
        }

        while (p > 1)
        {
            if ((p & 1) != 0)
            {
                a *= b;
            }

            b *= b;
            p >>= 1;
        }

        return a * b;
    }

    /// <summary>OpenCV <c>iPow_f</c> for <c>double</c>.</summary>
    public static double IntegerPower(double value, int power0)
    {
        double a = 1, b = value;
        int p = Math.Abs(power0);
        if (power0 < 0)
        {
            b = 1 / b;
        }

        while (p > 1)
        {
            if ((p & 1) != 0)
            {
                a *= b;
            }

            b *= b;
            p >>= 1;
        }

        return a * b;
    }

    /// <summary><c>cvttps2dq</c>: truncation, with <c>int.MinValue</c> for NaN and values out of range.</summary>
    private static int TruncateToInt32(float v)
    {
        return float.IsNaN(v) || v >= 2147483648f || v < -2147483648f ? int.MinValue : (int)v;
    }

    private static float FlipSign(float v) => BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(v) ^ int.MinValue);

    private static float ReciprocalSqrtEstimate(float v)
    {
        if (Sse.IsSupported)
        {
            return Sse.ReciprocalSqrtScalar(Vector128.CreateScalarUnsafe(v)).ToScalar();
        }

        return 1 / MathF.Sqrt(v);
    }
}
