// This implementation is ported from the OpenCV discrete Fourier transform.
//
// Original source:
//   modules/core/src/dxt.cpp (DFTFactorize, DFTInit, DCTInit, bitrevTab, DFTTab, DctScale)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/dxt.cpp
//
// OpenCV is licensed under the Apache License 2.0.
// See the OpenCV license and this repository's licensing documentation.

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Collections.Concurrent;
using System.Numerics;

/// <summary>
/// Tables of one DFT length: factorization, digit-reversal permutation and twiddles, as
/// <c>DFTFactorize</c> and <c>DFTInit</c>. Instances are immutable and shared between threads.
/// </summary>
internal sealed class DftPlan<T>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
{
    private const int MaxCachedPlans = 128;

    private static readonly ConcurrentDictionary<(int N, bool InvItab), DftPlan<T>> Cache = new();

    private DftPlan(int n, bool invItab)
    {
        N = n;
        Factors = DftTables.Factorize(n);
        Itab = new int[n];
        Wave = new T[2 * n];
        DftTables.Init(n, Factors, Itab, Wave, invItab);
    }

    /// <summary>Transform length (<c>tab_size</c>).</summary>
    public int N { get; }

    public int[] Factors { get; }

    public int[] Itab { get; }

    /// <summary>Twiddles <c>exp(-2πik/N)</c> as interleaved (re, im) pairs.</summary>
    public T[] Wave { get; }

    public static DftPlan<T> Get(int n, bool invItab)
    {
        // The inverse table equals the forward table when the permutation is an involution.
        if (invItab && n > 5)
        {
            int[] f = DftTables.Factorize(n);
            invItab = f[0] != f[^1];
        }
        else
        {
            invItab = false;
        }

        if (Cache.TryGetValue((n, invItab), out DftPlan<T>? plan))
        {
            return plan;
        }

        if (Cache.Count >= MaxCachedPlans)
        {
            Cache.Clear();
        }

        return Cache.GetOrAdd((n, invItab), static k => new DftPlan<T>(k.N, k.InvItab));
    }
}

/// <summary>DCT twiddles <c>scale·exp(-iπk/(2N))</c> for k = 0..N/2, as <c>DCTInit</c>.</summary>
internal sealed class DctPlan<T>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
{
    private const int MaxCachedPlans = 64;

    private static readonly ConcurrentDictionary<(int N, bool Inv), DctPlan<T>> Cache = new();

    private DctPlan(int n, bool inv)
    {
        Dft = DftPlan<T>.Get(n, inv);
        Wave = new T[2 * ((n / 2) + 1)];
        DftTables.DctInit(n, Wave, inv);
    }

    public DftPlan<T> Dft { get; }

    public T[] Wave { get; }

    public static DctPlan<T> Get(int n, bool inv)
    {
        if (Cache.TryGetValue((n, inv), out DctPlan<T>? plan))
        {
            return plan;
        }

        if (Cache.Count >= MaxCachedPlans)
        {
            Cache.Clear();
        }

        return Cache.GetOrAdd((n, inv), static k => new DctPlan<T>(k.N, k.Inv));
    }
}

internal static class DftTables
{
    private static ReadOnlySpan<byte> BitrevTab =>
    [
        0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
        0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
        0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
        0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
        0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
        0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
        0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
        0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
        0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
        0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
        0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
        0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
        0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
        0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
        0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
        0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
    ];

    /// <summary>(cos, sin)(2π/2^m), copied from OpenCV; not recomputed.</summary>
    private static readonly double[] DftTab =
    [
        1.00000000000000000, 0.00000000000000000,
        -1.00000000000000000, 0.00000000000000000,
        0.00000000000000000, 1.00000000000000000,
        0.70710678118654757, 0.70710678118654746,
        0.92387953251128674, 0.38268343236508978,
        0.98078528040323043, 0.19509032201612825,
        0.99518472667219693, 0.09801714032956060,
        0.99879545620517241, 0.04906767432741802,
        0.99969881869620425, 0.02454122852291229,
        0.99992470183914450, 0.01227153828571993,
        0.99998117528260111, 0.00613588464915448,
        0.99999529380957619, 0.00306795676296598,
        0.99999882345170188, 0.00153398018628477,
        0.99999970586288223, 0.00076699031874270,
        0.99999992646571789, 0.00038349518757140,
        0.99999998161642933, 0.00019174759731070,
        0.99999999540410733, 0.00009587379909598,
        0.99999999885102686, 0.00004793689960307,
        0.99999999971275666, 0.00002396844980842,
        0.99999999992818922, 0.00001198422490507,
        0.99999999998204725, 0.00000599211245264,
        0.99999999999551181, 0.00000299605622633,
        0.99999999999887801, 0.00000149802811317,
        0.99999999999971945, 0.00000074901405658,
        0.99999999999992983, 0.00000037450702829,
        0.99999999999998246, 0.00000018725351415,
        0.99999999999999567, 0.00000009362675707,
        0.99999999999999889, 0.00000004681337854,
        0.99999999999999978, 0.00000002340668927,
        0.99999999999999989, 0.00000001170334463,
        1.00000000000000000, 0.00000000585167232,
        1.00000000000000000, 0.00000000292583616,
    ];

    private static readonly double[] DctScale =
    [
        0.707106781186547570, 0.500000000000000000, 0.353553390593273790,
        0.250000000000000000, 0.176776695296636890, 0.125000000000000000,
        0.088388347648318447, 0.062500000000000000, 0.044194173824159223,
        0.031250000000000000, 0.022097086912079612, 0.015625000000000000,
        0.011048543456039806, 0.007812500000000000, 0.005524271728019903,
        0.003906250000000000, 0.002762135864009952, 0.001953125000000000,
        0.001381067932004976, 0.000976562500000000, 0.000690533966002488,
        0.000488281250000000, 0.000345266983001244, 0.000244140625000000,
        0.000172633491500622, 0.000122070312500000, 0.000086316745750311,
        0.000061035156250000, 0.000043158372875155, 0.000030517578125000,
    ];

    /// <summary>
    /// <c>DFTFactorize</c>: the power-of-2 block first (if any), then the odd factors in
    /// descending order.
    /// </summary>
    public static int[] Factorize(int n)
    {
        if (n <= 5)
        {
            return [n];
        }

        Span<int> factors = stackalloc int[34];
        int nf = 0;
        int f = (((n - 1) ^ n) + 1) >> 1;
        if (f > 1)
        {
            factors[nf++] = f;
            n = f == n ? 1 : n / f;
        }

        for (f = 3; n > 1;)
        {
            int d = n / f;
            if (d * f == n)
            {
                factors[nf++] = f;
                n = d;
            }
            else
            {
                f += 2;
                if (f * f > n)
                {
                    break;
                }
            }
        }

        if (n > 1)
        {
            factors[nf++] = n;
        }

        f = (factors[0] & 1) == 0 ? 1 : 0;
        for (int i = f; i < (nf + f) / 2; i++)
        {
            (factors[i], factors[nf - i - 1 + f]) = (factors[nf - i - 1 + f], factors[i]);
        }

        return factors[..nf].ToArray();
    }

    /// <summary><c>DFTInit</c>. <paramref name="wave"/> holds interleaved (re, im) pairs.</summary>
    public static void Init<T>(int n0, int[] factors, int[] itab, T[] wave, bool invItab)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        int nf = factors.Length;
        int n = factors[0], m = 0;

        if (n0 <= 5)
        {
            itab[0] = 0;
            itab[n0 - 1] = n0 - 1;
            if (n0 != 4)
            {
                for (int i = 1; i < n0 - 1; i++)
                {
                    itab[i] = i;
                }
            }
            else
            {
                itab[1] = 2;
                itab[2] = 1;
            }

            if (n0 == 5)
            {
                wave[0] = T.One;
                wave[1] = T.Zero;
            }

            if (n0 != 4)
            {
                return;
            }

            m = 2;
        }
        else
        {
            Span<int> digits = stackalloc int[34];
            Span<int> radix = stackalloc int[34];
            radix[nf] = 1;
            digits[nf] = 0;
            for (int i = 0; i < nf; i++)
            {
                digits[i] = 0;
                radix[nf - i - 1] = radix[nf - i] * factors[nf - i - 1];
            }

            int[] tab = invItab && factors[0] != factors[nf - 1] ? new int[n0] : itab;

            if ((n & 1) == 0)
            {
                int a = radix[1], na2 = (n * a) >> 1, na4 = na2 >> 1;
                for (m = 0; (uint)(1 << m) < (uint)n; m++)
                {
                }

                if (n <= 2)
                {
                    tab[0] = 0;
                    tab[1] = na2;
                }
                else if (n <= 256)
                {
                    int shift = 10 - m;
                    for (int i = 0; i <= n - 4; i += 4)
                    {
                        int j = (BitrevTab[i >> 2] >> shift) * a;
                        tab[i] = j;
                        tab[i + 1] = j + na2;
                        tab[i + 2] = j + na4;
                        tab[i + 3] = j + na2 + na4;
                    }
                }
                else
                {
                    int shift = 34 - m;
                    for (int i = 0; i < n; i += 4)
                    {
                        int j = BitRev(i >> 2, shift) * a;
                        tab[i] = j;
                        tab[i + 1] = j + na2;
                        tab[i + 2] = j + na4;
                        tab[i + 3] = j + na2 + na4;
                    }
                }

                digits[1]++;

                if (nf >= 2)
                {
                    for (int i = n, j = radix[2]; i < n0;)
                    {
                        for (int k = 0; k < n; k++)
                        {
                            tab[i + k] = tab[k] + j;
                        }

                        if ((i += n) >= n0)
                        {
                            break;
                        }

                        j += radix[2];
                        for (int k = 1; ++digits[k] >= factors[k]; k++)
                        {
                            digits[k] = 0;
                            j += radix[k + 2] - radix[k];
                        }
                    }
                }
            }
            else
            {
                for (int i = 0, j = 0; ;)
                {
                    tab[i] = j;
                    if (++i >= n0)
                    {
                        break;
                    }

                    j += radix[1];
                    for (int k = 0; ++digits[k] >= factors[k]; k++)
                    {
                        digits[k] = 0;
                        j += radix[k + 2] - radix[k];
                    }
                }
            }

            if (!ReferenceEquals(tab, itab))
            {
                itab[0] = 0;
                for (int i = n0 & 1; i < n0; i += 2)
                {
                    itab[tab[i]] = i;
                    itab[tab[i + 1]] = i + 1;
                }
            }
        }

        double wre, wim, w1re, w1im;
        if ((n0 & (n0 - 1)) == 0)
        {
            wre = w1re = DftTab[2 * m];
            wim = w1im = -DftTab[(2 * m) + 1];
        }
        else
        {
            double t = -Math.PI * 2 / n0;
            wim = w1im = Math.Sin(t);
            wre = w1re = Math.Sqrt(1.0 - (w1im * w1im));
        }

        n = (n0 + 1) / 2;
        wave[0] = T.One;
        wave[1] = T.Zero;
        if ((n0 & 1) == 0)
        {
            wave[2 * n] = -T.One;
            wave[(2 * n) + 1] = T.Zero;
        }

        for (int i = 1; i < n; i++)
        {
            wave[2 * i] = T.CreateTruncating(wre);
            wave[(2 * i) + 1] = T.CreateTruncating(wim);
            wave[2 * (n0 - i)] = T.CreateTruncating(wre);
            wave[(2 * (n0 - i)) + 1] = T.CreateTruncating(-wim);

            double t = (wre * w1re) - (wim * w1im);
            wim = (wre * w1im) + (wim * w1re);
            wre = t;
        }
    }

    /// <summary><c>DCTInit</c> for even <paramref name="n"/>; the recurrence starts from the scale rounded to T.</summary>
    public static void DctInit<T>(int n, T[] wave, bool inv)
        where T : unmanaged, IBinaryFloatingPointIeee754<T>
    {
        if (n == 1)
        {
            return;
        }

        double scale, w1re, w1im;
        if ((n & (n - 1)) == 0)
        {
            int m;
            for (m = 0; (uint)(1 << m) < (uint)n; m++)
            {
            }

            scale = (!inv ? 2 : 1) * DctScale[m];
            w1re = DftTab[2 * (m + 2)];
            w1im = -DftTab[(2 * (m + 2)) + 1];
        }
        else
        {
            double t = 1.0 / (2 * n);
            scale = (!inv ? 2 : 1) * Math.Sqrt(t);
            w1im = Math.Sin(-Math.PI * t);
            w1re = Math.Sqrt(1.0 - (w1im * w1im));
        }

        n >>= 1;
        double wre = double.CreateTruncating(T.CreateTruncating(scale));
        double wim = 0;
        for (int i = 0; i <= n; i++)
        {
            wave[2 * i] = T.CreateTruncating(wre);
            wave[(2 * i) + 1] = T.CreateTruncating(wim);
            double t = (wre * w1re) - (wim * w1im);
            wim = (wre * w1im) + (wim * w1re);
            wre = t;
        }
    }

    private static int BitRev(int i, int shift)
        => (int)((((uint)BitrevTab[i & 255] << 24) +
                  ((uint)BitrevTab[(i >> 8) & 255] << 16) +
                  ((uint)BitrevTab[(i >> 16) & 255] << 8) +
                  BitrevTab[(i >> 24) & 255]) >> shift);
}
