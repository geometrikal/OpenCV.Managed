// This implementation is ported from the OpenCV Lab conversion tables.
//
// Original sources:
//   modules/imgproc/src/color_lab.cpp (createLabTabs, initLUTforABXZ,
//     initLUTforLABLUVs16, fill_one, splineBuild, splineInterpolate,
//     applyGamma, applyInvGamma, trilinearInterpolate, colour constants)
//   modules/core/src/softfloat.cpp (f32_cbrt)
//   modules/core/src/mathfuncs.cpp (cubeRoot)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Constants and lookup tables of OpenCV's Lab conversions. All tables are built once, with
/// the float and double operations of OpenCV's softfloat code in the same order.
/// </summary>
internal static class LabTables
{
    public const int LabShift = 12;
    public const int GammaShift = 3;
    public const int LabShift2 = 15;
    public const int CbrtTabSize = 1024;
    public const int GammaTabSize = 1024;
    public const int CbrtTabSizeB = 256 * 3 / 2 * (1 << GammaShift);
    public const int InvGammaShift = 12;
    public const int InvGammaTabSize = 1 << InvGammaShift;
    public const int LutDim = 33;
    public const int LabBase = 1 << 14;
    public const int TrilinearShift = 4;
    public const int TrilinearBase = 1 << TrilinearShift;
    public const int MinAbValue = -8145;

    public const float LabCbrtTabScale = 2048f / 3f;
    public const float GammaTabScale = GammaTabSize;

    /// <summary><c>sRGB2XYZ_D65</c>: rows X, Y, Z; columns R, G, B.</summary>
    public static readonly double[] SRgb2XyzD65 =
    [
        D(0x3fda65a14488c60d), D(0x3fd6e297396d0918), D(0x3fc71819d2391d58),
        D(0x3fcb38cda6e75ff6), D(0x3fe6e297396d0918), D(0x3fb279aae6c8f755),
        D(0x3f93cc4ac6cdaf4b), D(0x3fbe836eb4e98138), D(0x3fee68427418d691),
    ];

    /// <summary><c>XYZ2sRGB_D65</c>: rows R, G, B; columns X, Y, Z.</summary>
    public static readonly double[] Xyz2SRgbD65 =
    [
        D(0x4009ec804102ff8f), D(unchecked((long)0xbff8982a9930be0e)), D(unchecked((long)0xbfdfe7ff583a53b9)),
        D(unchecked((long)0xbfef042528ae74f3)), D(0x3ffe040f23897204), D(0x3fa546d3f9e7b80b),
        D(0x3fac7de5082cf52c), D(unchecked((long)0xbfca1e14bdfd2631)), D(0x3ff0eabef06b3786),
    ];

    public static readonly double[] D65 = [D(0x3fee6a22b3892ee8), 1.0, D(0x3ff16b8950763a19)];

    public static readonly float LThresh = 216f / 24389f;
    public static readonly float LScale = 841f / 108f;
    public static readonly float LBias = 16f / 116f;

    private static readonly double GammaThreshold = 809.0 / 20000.0;
    private static readonly double GammaInvThreshold = 7827.0 / 2500000.0;
    private static readonly double GammaLowScale = 323.0 / 25.0;
    private static readonly double GammaPower = 12.0 / 5.0;
    private static readonly double GammaXShift = 11.0 / 200.0;

    private static readonly Lazy<Tables> Instance = new(Build, LazyThreadSafetyMode.ExecutionAndPublication);

    public static Tables Get => Instance.Value;

    private static double D(long bits) => BitConverter.Int64BitsToDouble(bits);

    /// <summary>OpenCV <c>applyGamma</c> (sRGB to linear).</summary>
    public static float ApplyGamma(float x)
    {
        double xd = x;
        return (float)(xd <= GammaThreshold
            ? xd / GammaLowScale
            : Math.Pow((xd + GammaXShift) / (1.0 + GammaXShift), GammaPower));
    }

    /// <summary>OpenCV <c>applyInvGamma</c> (linear to sRGB).</summary>
    public static float ApplyInvGamma(float x)
    {
        double xd = x;
        return (float)(xd <= GammaInvThreshold
            ? xd * GammaLowScale
            : (Math.Pow(xd, 1.0 / GammaPower) * (1.0 + GammaXShift)) - GammaXShift);
    }

    /// <summary>
    /// softfloat <c>f32_cbrt</c>: Turkowski's rational approximation in double; the result
    /// mantissa is truncated, not rounded.
    /// </summary>
    public static float SoftCbrt(float x)
    {
        if (float.IsNaN(x))
        {
            return float.NaN;
        }

        if (float.IsInfinity(x))
        {
            return x;
        }

        uint bits = BitConverter.SingleToUInt32Bits(x);
        uint sign = bits >> 31;
        int ex = (int)((bits >> 23) & 0xFF) - 127;
        int shx = ex % 3;
        shx -= shx >= 0 ? 3 : 0;
        ex = ((ex - shx) / 3) - 1;
        double fr = BitConverter.Int64BitsToDouble(((long)(shx + 1023) << 52) | ((long)(bits & 0x7FFFFF) << 29));
        fr = ((((((((D(0x4046a09e6653ba70) * fr) + D(0x406808f46c6116e0)) * fr) + D(0x405dca97439cae14)) * fr)
                  + D(0x402add70d2827500)) * fr) + D(0x3fc4f15f83f55d2d))
             / ((((((((D(0x402d9e20660edb21) * fr) + D(0x4062ff15c0285815)) * fr) + D(0x406510d06a8112ce)) * fr)
                   + D(0x4040fecbc9e2c375)) * fr) + 1.0);
        if ((bits & 0x7FFFFFFF) == 0)
        {
            return 0f;
        }

        ulong frBits = (ulong)BitConverter.DoubleToInt64Bits(fr);
        uint mant = (uint)((frBits & 0xFFFFFFFFFFFFFUL) >> 29);
        return BitConverter.UInt32BitsToSingle((sign << 31) + ((uint)(ex + 127) << 23) + mant);
    }

    /// <summary>OpenCV <c>cv::cubeRoot(float)</c>, used by the scalar linear-RGB-to-Lab path.</summary>
    public static float CubeRoot(float value)
    {
        int vi = BitConverter.SingleToInt32Bits(value);
        int ix = vi & 0x7fffffff;
        int s = (int)(vi & 0x80000000);
        int ex = (ix >> 23) - 127;
        int shx = ex % 3;
        shx -= shx >= 0 ? 3 : 0;
        ex = (ex - shx) / 3;
        float fr = BitConverter.Int32BitsToSingle((ix & ((1 << 23) - 1)) | ((shx + 127) << 23));
        fr = (float)(((((((((45.2548339756803022511987494 * fr) + 192.2798368355061050458134625) * fr)
                         + 119.1654824285581628956914143) * fr) + 13.43250139086239872172837314) * fr)
                      + 0.1636161226585754240958355063)
                     / ((((((((14.80884093219134573786480845 * fr) + 151.9714051044435648658557668) * fr)
                           + 168.5254414101568283957668343) * fr) + 33.9905941350215598754191872) * fr) + 1.0));
        int r = unchecked(BitConverter.SingleToInt32Bits(fr) + (ex << 23) + s) & (unchecked(vi * 2) != 0 ? -1 : 0);
        return BitConverter.Int32BitsToSingle(r);
    }

    /// <summary>OpenCV <c>splineBuild</c>: natural cubic spline coefficients in float.</summary>
    public static float[] SplineBuild(float[] f, int n)
    {
        var tab = new float[n * 4];
        tab[0] = tab[1] = 0f;
        for (int i = 1; i < n; i++)
        {
            float t = (f[i + 1] - (f[i] * 2) + f[i - 1]) * 3;
            float l = 1 / (4 - tab[(i - 1) * 4]);
            tab[i * 4] = l;
            tab[(i * 4) + 1] = (t - tab[((i - 1) * 4) + 1]) * l;
        }

        float cn = 0;
        for (int i = n - 1; i >= 0; i--)
        {
            float c = tab[(i * 4) + 1] - (tab[i * 4] * cn);
            float b = f[i + 1] - f[i] - ((cn + (c * 2)) / 3);
            float d = (cn - c) / 3;
            tab[i * 4] = f[i];
            tab[(i * 4) + 1] = b;
            tab[(i * 4) + 2] = c;
            tab[(i * 4) + 3] = d;
            cn = c;
        }

        return tab;
    }

    /// <summary>OpenCV <c>splineInterpolate</c>; the index is clamped to <c>[0, n - 1]</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SplineInterpolate(float x, float[] tab, int n)
    {
        int ix = Math.Clamp((int)x, 0, n - 1);
        x -= ix;
        int k = ix * 4;
        return (((((tab[k + 3] * x) + tab[k + 2]) * x) + tab[k + 1]) * x) + tab[k];
    }

    /// <summary>OpenCV <c>trilinearInterpolate</c> on the packed RGB-to-Lab table.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TrilinearInterpolate(int cx, int cy, int cz, short[] lut, short[] weights, out int a, out int b, out int c)
    {
        int tx = cx >> 9, ty = cy >> 9, tz = cz >> 9;
        int baseIdx = (24 * tx) + (24 * LutDim * ty) + (24 * LutDim * LutDim * tz);
        int x = (cx >> 5) & (TrilinearBase - 1);
        int y = (cy >> 5) & (TrilinearBase - 1);
        int z = (cz >> 5) & (TrilinearBase - 1);
        int w = 8 * (x + (TrilinearBase * y) + (TrilinearBase * TrilinearBase * z));
        int sa = 0, sb = 0, sc = 0;
        for (int i = 0; i < 8; i++)
        {
            int wi = weights[w + i];
            sa += lut[baseIdx + i] * wi;
            sb += lut[baseIdx + i + 8] * wi;
            sc += lut[baseIdx + i + 16] * wi;
        }

        a = (sa + (1 << 11)) >> 12;
        b = (sb + (1 << 11)) >> 12;
        c = (sc + (1 << 11)) >> 12;
    }

    private static Tables Build()
    {
        var t = new Tables();

        var f = new float[CbrtTabSize + 1];
        float scale = 1f / LabCbrtTabScale;
        for (int i = 0; i <= CbrtTabSize; i++)
        {
            float x = scale * i;
            f[i] = x < LThresh ? MathF.FusedMultiplyAdd(x, LScale, LBias) : SoftCbrt(x);
        }

        t.LabCbrtTab = SplineBuild(f, CbrtTabSize);

        var g = new float[GammaTabSize + 1];
        var ig = new float[GammaTabSize + 1];
        scale = 1f / GammaTabScale;
        for (int i = 0; i <= GammaTabSize; i++)
        {
            float x = scale * i;
            g[i] = ApplyGamma(x);
            ig[i] = ApplyInvGamma(x);
        }

        t.SRgbGammaTab = SplineBuild(g, GammaTabSize);
        t.SRgbInvGammaTab = SplineBuild(ig, GammaTabSize);

        const float intScale = 255 * (1 << GammaShift);
        for (int i = 0; i < 256; i++)
        {
            float x = i / 255f;
            t.SRgbGammaTabB[i] = (ushort)ScalarConverter.CvRound(intScale * ApplyGamma(x));
            t.LinearGammaTabB[i] = (ushort)(i * (1 << GammaShift));
        }

        const float invScale = 1f / InvGammaTabSize;
        for (int i = 0; i < InvGammaTabSize; i++)
        {
            float x = invScale * i;
            t.SRgbInvGammaTabB[i] = (ushort)ScalarConverter.CvRound(255f * ApplyInvGamma(x));
        }

        float cbTabScale = 1f / (255f * (1 << GammaShift));
        const float lshift2 = 1 << LabShift2;
        for (int i = 0; i < CbrtTabSizeB; i++)
        {
            float x = cbTabScale * i;
            t.LabCbrtTabB[i] = (ushort)ScalarConverter.CvRound(
                lshift2 * (x < LThresh ? MathF.FusedMultiplyAdd(x, LScale, LBias) : SoftCbrt(x)));
        }

        for (int i = 0; i < 256; i++)
        {
            int y, ify;
            if (i <= 20)
            {
                y = ScalarConverter.CvRound((float)(i * LabBase * 20 * 9) / (float)(17 * 29 * 29 * 29));
                ify = ScalarConverter.CvRound(LabBase * ((16f / 116f) + ((float)(i * 5) / (float)(3 * 17 * 29))));
            }
            else
            {
                float fy = ((float)(i * 100 * LabBase) / (float)(255 * 116)) + ((float)(16 * LabBase) / 116f);
                ify = ScalarConverter.CvRound(fy);
                y = ScalarConverter.CvRound(fy * fy * fy / (float)(LabBase * LabBase));
            }

            t.LabToYFB[i * 2] = (ushort)y;
            t.LabToYFB[(i * 2) + 1] = (ushort)ify;
        }

        for (int i = MinAbValue; i < (LabBase * 9 / 4) + MinAbValue; i++)
        {
            t.AbToXzB[i - MinAbValue] = i <= 3390
                ? (i * 108 / 841) - (LabBase * 16 / 116 * 108 / 841)
                : i * i / LabBase * i / LabBase;
        }

        BuildTrilinear(t);
        return t;
    }

    private static void BuildTrilinear(Tables t)
    {
        double[] m = SRgb2XyzD65;
        double[] white = [1.0 / D65[0], 1.0, 1.0 / D65[2]];
        var s = new float[9];
        for (int i = 0; i < 3; i++)
        {
            s[(i * 3) + 0] = (float)(m[(i * 3) + 2] * white[i]);
            s[(i * 3) + 1] = (float)(m[(i * 3) + 1] * white[i]);
            s[(i * 3) + 2] = (float)(m[(i * 3) + 0] * white[i]);
        }

        const float f9033 = (float)(29 * 29 * 29) / 27f;
        var prev = new short[LutDim * LutDim * LutDim * 3];
        for (int p = 0; p < LutDim; p++)
        {
            for (int q = 0; q < LutDim; q++)
            {
                for (int r = 0; r < LutDim; r++)
                {
                    int idx = (p * 3) + (q * LutDim * 3) + (r * LutDim * LutDim * 3);
                    float rr = ApplyGamma(p / 32f);
                    float gg = ApplyGamma(q / 32f);
                    float bb = ApplyGamma(r / 32f);
                    float x = (rr * s[0]) + (gg * s[1]) + (bb * s[2]);
                    float y = (rr * s[3]) + (gg * s[4]) + (bb * s[5]);
                    float z = (rr * s[6]) + (gg * s[7]) + (bb * s[8]);
                    float fx = x > LThresh ? SoftCbrt(x) : MathF.FusedMultiplyAdd(x, LScale, LBias);
                    float fy = y > LThresh ? SoftCbrt(y) : MathF.FusedMultiplyAdd(y, LScale, LBias);
                    float fz = z > LThresh ? SoftCbrt(z) : MathF.FusedMultiplyAdd(z, LScale, LBias);
                    float l = y > LThresh ? (116f * fy) - 16f : f9033 * y;
                    float a = 500f * (fx - fy);
                    float b = 200f * (fy - fz);
                    prev[idx] = (short)ScalarConverter.CvRound(LabBase * l / 100f);
                    prev[idx + 1] = (short)ScalarConverter.CvRound(LabBase * (a + 128f) / 256f);
                    prev[idx + 2] = (short)ScalarConverter.CvRound(LabBase * (b + 128f) / 256f);
                }
            }
        }

        var lut = new short[LutDim * LutDim * LutDim * 3 * 8];
        for (int p = 0; p < LutDim; p++)
        {
            for (int q = 0; q < LutDim; q++)
            {
                for (int r = 0; r < LutDim; r++)
                {
                    for (int dp = 0; dp < 2; dp++)
                    {
                        for (int dq = 0; dq < 2; dq++)
                        {
                            for (int dr = 0; dr < 2; dr++)
                            {
                                int old = (Math.Min(p + dp, LutDim - 1) * 3) +
                                          (Math.Min(q + dq, LutDim - 1) * LutDim * 3) +
                                          (Math.Min(r + dr, LutDim - 1) * LutDim * LutDim * 3);
                                int idxNew = (p * 24) + (q * LutDim * 24) + (r * LutDim * LutDim * 24) + (4 * dp) + (2 * dq) + dr;
                                lut[idxNew] = prev[old];
                                lut[idxNew + 8] = prev[old + 1];
                                lut[idxNew + 16] = prev[old + 2];
                            }
                        }
                    }
                }
            }
        }

        t.RgbToLabLut = lut;
        for (int p = 0; p < TrilinearBase; p++)
        {
            int pp = TrilinearBase - p;
            for (int q = 0; q < TrilinearBase; q++)
            {
                int qq = TrilinearBase - q;
                for (int r = 0; r < TrilinearBase; r++)
                {
                    int rr = TrilinearBase - r;
                    int w = (8 * p) + (8 * TrilinearBase * q) + (8 * TrilinearBase * TrilinearBase * r);
                    t.TrilinearWeights[w + 0] = (short)(pp * qq * rr);
                    t.TrilinearWeights[w + 1] = (short)(pp * qq * r);
                    t.TrilinearWeights[w + 2] = (short)(pp * q * rr);
                    t.TrilinearWeights[w + 3] = (short)(pp * q * r);
                    t.TrilinearWeights[w + 4] = (short)(p * qq * rr);
                    t.TrilinearWeights[w + 5] = (short)(p * qq * r);
                    t.TrilinearWeights[w + 6] = (short)(p * q * rr);
                    t.TrilinearWeights[w + 7] = (short)(p * q * r);
                }
            }
        }
    }

    internal sealed class Tables
    {
        public float[] LabCbrtTab { get; set; } = [];

        public float[] SRgbGammaTab { get; set; } = [];

        public float[] SRgbInvGammaTab { get; set; } = [];

        public ushort[] SRgbGammaTabB { get; } = new ushort[256];

        public ushort[] LinearGammaTabB { get; } = new ushort[256];

        public ushort[] SRgbInvGammaTabB { get; } = new ushort[InvGammaTabSize];

        public ushort[] LabCbrtTabB { get; } = new ushort[CbrtTabSizeB];

        public ushort[] LabToYFB { get; } = new ushort[512];

        public int[] AbToXzB { get; } = new int[LabBase * 9 / 4];

        public short[] RgbToLabLut { get; set; } = [];

        public short[] TrilinearWeights { get; } = new short[8 * TrilinearBase * TrilinearBase * TrilinearBase];
    }
}
