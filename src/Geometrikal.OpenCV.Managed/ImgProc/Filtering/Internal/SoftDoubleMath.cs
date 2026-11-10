// This implementation is a transcription of OpenCV's software-float
// exponential onto IEEE-754 binary64 hardware arithmetic.
//
// Original sources:
//   modules/core/src/softfloat.cpp (f64_exp, expTab, EXPPOLY_32F_A0)
//
// OpenCV computes bit-exact Gaussian kernels with `softdouble`, a software
// IEEE-754 binary64 implementation whose +, -, *, / and round-to-integral
// operations are correctly rounded — i.e. they produce exactly the same bit
// patterns as .NET's hardware `double` operations. The only softdouble
// routine that is not a single correctly-rounded operation is `exp`, which
// OpenCV implements as a table+polynomial built from those primitive
// operations. Transcribing that routine verbatim onto `double` therefore
// reproduces `cv::exp(softdouble)` bit-for-bit.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Bit-exact transcription of OpenCV's <c>cv::exp(softdouble)</c>
/// (<c>f64_exp</c>, softfloat.cpp:3535) used by the Gaussian kernel
/// computation.
/// </summary>
internal static class SoftDoubleMath
{
    private const int ExpTabScale = 6;
    private const int ExpTabMask = (1 << ExpTabScale) - 1;

    // .9670371139572337719125840413672004409288e-2
    private static readonly double ExpPolyA0 = BitConverter.Int64BitsToDouble(0x3f83ce0f3e46f431);

    private static readonly double A5 = 1.0 / ExpPolyA0;
    private static readonly double A4 = BitConverter.Int64BitsToDouble(0x3fe62e42fefa39f1) / ExpPolyA0;
    private static readonly double A3 = BitConverter.Int64BitsToDouble(0x3fcebfbdff82a45a) / ExpPolyA0;
    private static readonly double A2 = BitConverter.Int64BitsToDouble(0x3fac6b08d81fec75) / ExpPolyA0;
    private static readonly double A1 = BitConverter.Int64BitsToDouble(0x3f83b2a72b4f3cd3) / ExpPolyA0;
    private static readonly double A0 = BitConverter.Int64BitsToDouble(0x3f55e7aa1566c2a4) / ExpPolyA0;

    // 1/ln(2) * 2^EXPTAB_SCALE (the multiplication by a power of two is exact).
    private static readonly double ExpPrescale =
        BitConverter.Int64BitsToDouble(0x3ff71547652b82fe) * (1 << ExpTabScale);

    private static readonly double ExpPostscale = 1.0 / (1 << ExpTabScale);

    private const double ExpMaxVal = 3000.0 * (1 << ExpTabScale); // log10(DBL_MAX) < 3000

    private static readonly long[] ExpTab =
    [
        0x3ff0000000000000, 0x3ff02c9a3e778061, 0x3ff059b0d3158574, 0x3ff0874518759bc8,
        0x3ff0b5586cf9890f, 0x3ff0e3ec32d3d1a2, 0x3ff11301d0125b51, 0x3ff1429aaea92de0,
        0x3ff172b83c7d517b, 0x3ff1a35beb6fcb75, 0x3ff1d4873168b9aa, 0x3ff2063b88628cd6,
        0x3ff2387a6e756238, 0x3ff26b4565e27cdd, 0x3ff29e9df51fdee1, 0x3ff2d285a6e4030b,
        0x3ff306fe0a31b715, 0x3ff33c08b26416ff, 0x3ff371a7373aa9cb, 0x3ff3a7db34e59ff7,
        0x3ff3dea64c123422, 0x3ff4160a21f72e2a, 0x3ff44e086061892d, 0x3ff486a2b5c13cd0,
        0x3ff4bfdad5362a27, 0x3ff4f9b2769d2ca7, 0x3ff5342b569d4f82, 0x3ff56f4736b527da,
        0x3ff5ab07dd485429, 0x3ff5e76f15ad2148, 0x3ff6247eb03a5585, 0x3ff6623882552225,
        0x3ff6a09e667f3bcd, 0x3ff6dfb23c651a2f, 0x3ff71f75e8ec5f74, 0x3ff75feb564267c9,
        0x3ff7a11473eb0187, 0x3ff7e2f336cf4e62, 0x3ff82589994cce13, 0x3ff868d99b4492ed,
        0x3ff8ace5422aa0db, 0x3ff8f1ae99157736, 0x3ff93737b0cdc5e5, 0x3ff97d829fde4e50,
        0x3ff9c49182a3f090, 0x3ffa0c667b5de565, 0x3ffa5503b23e255d, 0x3ffa9e6b5579fdbf,
        0x3ffae89f995ad3ad, 0x3ffb33a2b84f15fb, 0x3ffb7f76f2fb5e47, 0x3ffbcc1e904bc1d2,
        0x3ffc199bdd85529c, 0x3ffc67f12e57d14b, 0x3ffcb720dcef9069, 0x3ffd072d4a07897c,
        0x3ffd5818dcfba487, 0x3ffda9e603db3285, 0x3ffdfc97337b9b5f, 0x3ffe502ee78b3ff6,
        0x3ffea4afa2a490da, 0x3ffefa1bee615a27, 0x3fff50765b6e4540, 0x3fffa7c1819e90d8,
    ];

    /// <summary>
    /// <c>cv::exp(softdouble)</c>: table-driven exponential with the exact
    /// operation sequence of OpenCV's <c>f64_exp</c>.
    /// </summary>
    public static double Exp(double x)
    {
        // Special cases mirror f64_exp.
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (double.IsInfinity(x))
        {
            return double.IsPositiveInfinity(x) ? x : 0.0;
        }

        long bits = BitConverter.DoubleToInt64Bits(x);
        double x0;
        if (((bits >> 52) & 0x7FF) > 1023 + 10)
        {
            x0 = bits < 0 ? -ExpMaxVal : ExpMaxVal;
        }
        else
        {
            x0 = x * ExpPrescale;
        }

        // cvRound(softdouble) == round-half-to-even; x0 is within int range here.
        int val0 = (int)Math.Round(x0, MidpointRounding.ToEven);
        int t = (val0 >> ExpTabScale) + 1023;
        t = t < 0 ? 0 : (t > 2047 ? 2047 : t);
        double buf = BitConverter.Int64BitsToDouble((long)t << 52);

        x0 = (x0 - Math.Round(x0, MidpointRounding.ToEven)) * ExpPostscale;

        return buf * ExpPolyA0 * BitConverter.Int64BitsToDouble(ExpTab[val0 & ExpTabMask])
            * ((((((A0 * x0) + A1) * x0 + A2) * x0 + A3) * x0 + A4) * x0 + A5);
    }
}
