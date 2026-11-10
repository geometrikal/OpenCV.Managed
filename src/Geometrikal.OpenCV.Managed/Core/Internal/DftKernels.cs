// This implementation is ported from the OpenCV discrete Fourier and cosine transforms.
//
// Original source:
//   modules/core/src/dxt.cpp (DFT, DFT_R2, DFT_R3, DFT_R5, RealDFT, CCSIDFT, ExpandCCS, DCT, IDCT)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/core/src/dxt.cpp
//
// OpenCV is licensed under the Apache License 2.0.
// See the OpenCV license and this repository's licensing documentation.

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Arguments of one complex transform call (OpenCV <c>OcvDftOptions</c>). A real transform of
/// even length runs a half-length transform over the full-length tables; <see cref="Factor0"/>
/// then replaces the first factor.
/// </summary>
internal readonly struct DftCall<T>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
{
    public DftCall(DftPlan<T> plan, bool inv, T scale)
    {
        N = plan.N;
        TabSize = plan.N;
        Factors = plan.Factors;
        FactorStart = 0;
        Nf = plan.Factors.Length;
        Factor0 = plan.Factors[0];
        Itab = plan.Itab;
        Wave = plan.Wave;
        Inv = inv;
        Scale = scale;
    }

    private DftCall(in DftCall<T> c, int n, int factorStart, int nf, int factor0, bool inv, T scale)
    {
        N = n;
        TabSize = c.TabSize;
        Factors = c.Factors;
        FactorStart = factorStart;
        Nf = nf;
        Factor0 = factor0;
        Itab = c.Itab;
        Wave = c.Wave;
        Inv = inv;
        Scale = scale;
    }

    public int N { get; }

    public int TabSize { get; }

    public int[] Factors { get; }

    public int FactorStart { get; }

    public int Nf { get; }

    public int Factor0 { get; }

    public int[] Itab { get; }

    public T[] Wave { get; }

    public bool Inv { get; }

    public T Scale { get; }

    /// <summary>Largest odd factor; sizes the scratch of the generic odd radix.</summary>
    public int MaxFactor
    {
        get
        {
            int m = 1;
            foreach (int f in Factors)
            {
                m = Math.Max(m, f);
            }

            return m;
        }
    }

    public int Factor(int k) => k == 0 ? Factor0 : Factors[FactorStart + k];

    /// <summary>Same length and tables, forward, unscaled.</summary>
    public DftCall<T> Plain() => new(this, N, FactorStart, Nf, Factor0, false, T.One);

    /// <summary>The half-length forward transform of a real transform of even length.</summary>
    public DftCall<T> Half()
    {
        int f0 = Factors[0] >> 1;
        return f0 == 1
            ? new DftCall<T>(this, N >> 1, 1, Nf - 1, Factors[1], false, T.One)
            : new DftCall<T>(this, N >> 1, 0, Nf, f0, false, T.One);
    }
}

/// <summary>
/// The dxt.cpp kernels over interleaved complex data: element k has its real part at index
/// 2k and its imaginary part at 2k+1. With <see cref="VectorLanes{T}"/> every lane holds an
/// independent transform of the same length.
/// </summary>
internal static class DftKernels<T, TV, TL>
    where T : unmanaged, IBinaryFloatingPointIeee754<T>
    where TV : unmanaged
    where TL : struct, IDftLanes<TV, T>
{
    private static readonly T Sin120 = T.CreateTruncating(0.86602540378443864676372317075294);
    private static readonly T Fft5_2 = T.CreateTruncating(0.559016994374947424102293417182819);
    private static readonly T Fft5_3 = T.CreateTruncating(-0.951056516295153572116439333379382);
    private static readonly T Fft5_4 = T.CreateTruncating(-1.538841768587626701285145288018455);
    private static readonly T Fft5_5 = T.CreateTruncating(0.363271264002680442947733378740309);
    private static readonly T Half = T.CreateTruncating(0.5);
    private static readonly T Quarter = T.CreateTruncating(0.25);
    private static readonly T Two = T.CreateTruncating(2.0);
    private static readonly T Sin45 = T.CreateTruncating(0.70710678118654752440084436210485);

    /// <summary>Scratch elements that <see cref="Dft"/> needs for a call.</summary>
    public static int ScratchLength(in DftCall<T> c) => 2 * (c.MaxFactor + 1);

    /// <summary>
    /// <c>DFT&lt;T&gt;</c>. With a non-empty <paramref name="src"/> the input is gathered through
    /// the permutation (out of place); with an empty <paramref name="src"/> the transform runs in
    /// place on already permuted data (<c>noPermute</c>).
    /// </summary>
    public static void Dft(in DftCall<T> c, ReadOnlySpan<TV> src, Span<TV> dst, Span<TV> scratch)
    {
        int n = c.N;
        bool inv = c.Inv;
        ref TV d = ref MemoryMarshal.GetReference(dst);
        ref T wave = ref MemoryMarshal.GetArrayDataReference(c.Wave);

        if (!src.IsEmpty)
        {
            Permute(c, src, ref d);
        }
        else if (inv)
        {
            for (int i = 0; i < n; i++)
            {
                Im(ref d, i) = TL.Neg(Im(ref d, i));
            }
        }

        int nn = 1;
        int dw0 = c.TabSize;
        int f0 = c.Factor0;

        if ((f0 & 1) == 0)
        {
            while (nn * 4 <= f0)
            {
                int nx = nn;
                nn *= 4;
                dw0 /= 4;
                Radix4(ref d, ref wave, n, nn, nx, dw0);
            }

            while (nn < f0)
            {
                nn *= 2;
                dw0 /= 2;
                Radix2(ref d, ref wave, n, nn, dw0);
            }
        }

        for (int fIdx = (f0 & 1) != 0 ? 0 : 1; fIdx < c.Nf; fIdx++)
        {
            int factor = c.Factor(fIdx);
            int nx = nn;
            nn *= factor;
            dw0 /= factor;

            if (factor == 3)
            {
                Radix3(ref d, ref wave, n, nn, dw0);
            }
            else if (factor == 5)
            {
                Radix5(ref d, ref wave, n, nn, dw0);
            }
            else
            {
                RadixOdd(ref d, ref wave, n, nn, nx, dw0, factor, c.TabSize, scratch);
            }
        }

        if (c.Scale != T.One)
        {
            TV reScale = TL.Splat(c.Scale);
            TV imScale = TL.Splat(inv ? -c.Scale : c.Scale);
            for (int i = 0; i < n; i++)
            {
                Re(ref d, i) = TL.Mul(Re(ref d, i), reScale);
                Im(ref d, i) = TL.Mul(Im(ref d, i), imScale);
            }
        }
        else if (inv)
        {
            for (int i = 0; i < n; i++)
            {
                Im(ref d, i) = TL.Neg(Im(ref d, i));
            }
        }
    }

    /// <summary>
    /// <c>RealDFT</c>: real input of length <c>c.N</c> to CCS, or to the first half of the
    /// complex spectrum when <paramref name="complexOutput"/> is set. <paramref name="dst"/> needs
    /// <c>2·N</c> elements for odd N and <c>N + 2</c> otherwise.
    /// </summary>
    public static void RealDft(in DftCall<T> c, ReadOnlySpan<TV> src, Span<TV> dst, bool complexOutput, Span<TV> scratch)
    {
        int n = c.N;
        TV scale = TL.Splat(c.Scale);
        int off = complexOutput ? 1 : 0;

        if (n == 1)
        {
            dst[off] = TL.Mul(src[0], scale);
        }
        else if (n == 2)
        {
            TV t = TL.Mul(TL.Add(src[0], src[1]), scale);
            dst[off + 1] = TL.Mul(TL.Sub(src[0], src[1]), scale);
            dst[off] = t;
        }
        else if ((n & 1) != 0)
        {
            int[] itab = c.Itab;
            TV zero = default;
            dst[0] = TL.Mul(src[0], scale);
            dst[1] = zero;
            for (int j = 1; j < n; j += 2)
            {
                TV t0 = TL.Mul(src[itab[j]], scale);
                TV t1 = TL.Mul(src[itab[j + 1]], scale);
                dst[2 * j] = t0;
                dst[(2 * j) + 1] = zero;
                dst[(2 * j) + 2] = t1;
                dst[(2 * j) + 3] = zero;
            }

            Dft(c.Plain(), default, dst, scratch);
            if (!complexOutput)
            {
                dst[1] = dst[0];
            }
        }
        else
        {
            TV scale2 = TL.Splat(c.Scale * Half);
            int n2 = n >> 1;
            Span<TV> o = dst[off..];

            Dft(c.Half(), src, o, scratch);

            TV t = TL.Sub(o[0], o[1]);
            o[0] = TL.Mul(TL.Add(o[0], o[1]), scale);
            o[1] = TL.Mul(t, scale);

            TV t0 = o[n2];
            t = o[n - 1];
            o[n - 1] = o[1];

            ReadOnlySpan<T> wave = c.Wave;
            int j, w;
            for (j = 2, w = 1; j < n2; j += 2, w++)
            {
                TV h2Re = TL.Mul(scale2, TL.Add(o[j + 1], t));
                TV h2Im = TL.Mul(scale2, TL.Sub(o[n - j], o[j]));
                TV h1Re = TL.Mul(scale2, TL.Add(o[j], o[n - j]));
                TV h1Im = TL.Mul(scale2, TL.Sub(o[j + 1], t));

                TV wr = TL.Splat(wave[2 * w]);
                TV wi = TL.Splat(wave[(2 * w) + 1]);
                t = TL.Sub(TL.Mul(h2Re, wr), TL.Mul(h2Im, wi));
                h2Im = TL.Add(TL.Mul(h2Re, wi), TL.Mul(h2Im, wr));
                h2Re = t;
                t = o[n - j - 1];

                o[j - 1] = TL.Add(h1Re, h2Re);
                o[n - j - 1] = TL.Sub(h1Re, h2Re);
                o[j] = TL.Add(h1Im, h2Im);
                o[n - j] = TL.Sub(h2Im, h1Im);
            }

            if (j <= n2)
            {
                o[n2 - 1] = TL.Mul(t0, scale);
                o[n2] = TL.Mul(TL.Neg(t), scale);
            }
        }

        if (complexOutput && ((n & 1) == 0 || n == 1))
        {
            dst[0] = dst[1];
            dst[1] = default;
            if (n > 1)
            {
                dst[n + 1] = default;
            }
        }
    }

    /// <summary>
    /// <c>CCSIDFT</c> with distinct source and destination: CCS input of length <c>c.N</c> to
    /// real output. <paramref name="dst"/> needs <c>2·N</c> elements for odd N.
    /// </summary>
    public static void CcsIdft(in DftCall<T> c, ReadOnlySpan<TV> src, Span<TV> dst, Span<TV> scratch)
    {
        int n = c.N;
        TV scale = TL.Splat(c.Scale);
        int[] itab = c.Itab;

        if (n == 1)
        {
            dst[0] = TL.Mul(src[0], scale);
        }
        else if (n == 2)
        {
            TV t = TL.Mul(TL.Add(src[0], src[1]), scale);
            dst[1] = TL.Mul(TL.Sub(src[0], src[1]), scale);
            dst[0] = t;
        }
        else if ((n & 1) != 0)
        {
            dst[0] = src[0];
            dst[1] = default;
            int n2 = (n + 1) >> 1;
            for (int j = 1; j < n2; j++)
            {
                int k0 = itab[j], k1 = itab[n - j];
                TV t0 = src[(2 * j) - 1];
                TV t1 = src[2 * j];
                dst[2 * k0] = t0;
                dst[(2 * k0) + 1] = TL.Neg(t1);
                dst[2 * k1] = t0;
                dst[(2 * k1) + 1] = t1;
            }

            Dft(c.Plain(), default, dst, scratch);
            dst[0] = TL.Mul(dst[0], scale);
            for (int j = 1; j < n; j += 2)
            {
                TV t0 = TL.Mul(dst[j * 2], scale);
                TV t1 = TL.Mul(dst[(j * 2) + 2], scale);
                dst[j] = t0;
                dst[j + 1] = t1;
            }
        }
        else
        {
            ReadOnlySpan<T> wave = c.Wave;
            TV t = src[1];
            dst[0] = TL.Add(src[0], src[n - 1]);
            dst[1] = TL.Sub(src[n - 1], src[0]);

            int n2 = (n + 1) >> 1;
            int j, w;
            for (j = 2, w = 1; j < n2; j += 2, w++)
            {
                TV h1Re = TL.Add(t, src[n - j - 1]);
                TV h1Im = TL.Sub(src[j], src[n - j]);
                TV h2Re = TL.Sub(t, src[n - j - 1]);
                TV h2Im = TL.Add(src[j], src[n - j]);

                TV wr = TL.Splat(wave[2 * w]);
                TV wi = TL.Splat(wave[(2 * w) + 1]);
                t = TL.Add(TL.Mul(h2Re, wr), TL.Mul(h2Im, wi));
                h2Im = TL.Sub(TL.Mul(h2Im, wr), TL.Mul(h2Re, wi));
                h2Re = t;

                t = src[j + 1];
                int j2 = j >> 1;
                int k = itab[j2];
                dst[k] = TL.Sub(h1Re, h2Im);
                dst[k + 1] = TL.Sub(TL.Neg(h1Im), h2Re);
                k = itab[n2 - j2];
                dst[k] = TL.Add(h1Re, h2Im);
                dst[k + 1] = TL.Sub(h1Im, h2Re);
            }

            if (j <= n2)
            {
                TV two = TL.Splat(Two);
                int k = itab[n2];
                dst[k * 2] = TL.Mul(t, two);
                dst[(k * 2) + 1] = TL.Mul(src[n2], two);
            }

            Dft(c.Half(), default, dst, scratch);

            TV negScale = TL.Splat(-c.Scale);
            for (j = 0; j < n; j += 2)
            {
                dst[j] = TL.Mul(dst[j], scale);
                dst[j + 1] = TL.Mul(dst[j + 1], negScale);
            }
        }
    }

    /// <summary>
    /// <c>ExpandCCS</c>: converts a CCS vector of length <paramref name="n"/> stored at the start
    /// of <paramref name="p"/> to <paramref name="n"/> conjugate-symmetric complex values in place.
    /// </summary>
    public static void ExpandCcs(Span<TV> p, int n)
    {
        for (int i = 1; i < (n + 1) / 2; i++)
        {
            p[(n - i) * 2] = p[(i * 2) - 1];
            p[((n - i) * 2) + 1] = TL.Neg(p[i * 2]);
        }

        if ((n & 1) == 0)
        {
            p[n] = p[n - 1];
            p[n + 1] = default;
            n--;
        }

        for (int i = n - 1; i > 0; i--)
        {
            p[i + 1] = p[i];
        }

        p[1] = default;
    }

    /// <summary>
    /// <c>DCT</c> of even length <c>c.N</c> (or 1): <paramref name="x"/> to <paramref name="y"/>,
    /// both natural order. <paramref name="buf"/> needs <c>2·N + 2</c> elements.
    /// </summary>
    public static void Dct(in DftCall<T> c, T[] dctWave, ReadOnlySpan<TV> x, Span<TV> y, Span<TV> buf, Span<TV> scratch)
    {
        int n = c.N;
        if (n == 1)
        {
            y[0] = x[0];
            return;
        }

        int n2 = n >> 1;
        Span<TV> dftSrc = buf[..n];
        Span<TV> dftDst = buf.Slice(n, n + 2);
        for (int j = 0; j < n2; j++)
        {
            dftSrc[j] = x[2 * j];
            dftSrc[n - j - 1] = x[(2 * j) + 1];
        }

        RealDft(c, dftSrc, dftDst, false, scratch);

        y[0] = TL.Mul(TL.Mul(dftDst[0], TL.Splat(dctWave[0])), TL.Splat(Sin45));
        for (int j = 1; j < n2; j++)
        {
            TV wr = TL.Splat(dctWave[2 * j]);
            TV wi = TL.Splat(dctWave[(2 * j) + 1]);
            TV negWi = TL.Splat(-dctWave[(2 * j) + 1]);
            y[j] = TL.Sub(TL.Mul(wr, dftDst[(j * 2) - 1]), TL.Mul(wi, dftDst[j * 2]));
            y[n - j] = TL.Sub(TL.Mul(negWi, dftDst[(j * 2) - 1]), TL.Mul(wr, dftDst[j * 2]));
        }

        y[n2] = TL.Mul(dftDst[n - 1], TL.Splat(dctWave[2 * n2]));
    }

    /// <summary><c>IDCT</c>; arguments as <see cref="Dct"/>.</summary>
    public static void Idct(in DftCall<T> c, T[] dctWave, ReadOnlySpan<TV> x, Span<TV> y, Span<TV> buf, Span<TV> scratch)
    {
        int n = c.N;
        if (n == 1)
        {
            y[0] = x[0];
            return;
        }

        int n2 = n >> 1;
        Span<TV> dftSrc = buf[..n];
        Span<TV> dftDst = buf.Slice(n, n + 2);
        TV two = TL.Splat(Two);

        dftSrc[0] = TL.Mul(TL.Mul(TL.Mul(x[0], two), TL.Splat(dctWave[0])), TL.Splat(Sin45));
        for (int j = 1; j < n2; j++)
        {
            TV wr = TL.Splat(dctWave[2 * j]);
            TV wi = TL.Splat(dctWave[(2 * j) + 1]);
            TV negWi = TL.Splat(-dctWave[(2 * j) + 1]);
            dftSrc[(j * 2) - 1] = TL.Sub(TL.Mul(wr, x[j]), TL.Mul(wi, x[n - j]));
            dftSrc[j * 2] = TL.Sub(TL.Mul(negWi, x[j]), TL.Mul(wr, x[n - j]));
        }

        dftSrc[n - 1] = TL.Mul(TL.Mul(x[n2], two), TL.Splat(dctWave[2 * n2]));
        CcsIdft(c, dftSrc, dftDst, scratch);

        for (int j = 0; j < n2; j++)
        {
            y[2 * j] = dftDst[j];
            y[(2 * j) + 1] = dftDst[n - j - 1];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref TV Re(ref TV d, int k) => ref Unsafe.Add(ref d, 2 * k);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref TV Im(ref TV d, int k) => ref Unsafe.Add(ref d, (2 * k) + 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV WRe(ref T wave, int k) => TL.Splat(Unsafe.Add(ref wave, 2 * k));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV WIm(ref T wave, int k) => TL.Splat(Unsafe.Add(ref wave, (2 * k) + 1));

    private static void Permute(in DftCall<T> c, ReadOnlySpan<TV> src, ref TV d)
    {
        int n = c.N;
        int tabStep = c.TabSize == n ? 1 : c.TabSize == n * 2 ? 2 : c.TabSize / n;
        ReadOnlySpan<int> itab = c.Itab;
        ref TV s = ref MemoryMarshal.GetReference(src);
        int i, it = 0;
        if (!c.Inv)
        {
            for (i = 0; i <= n - 2; i += 2, it += 2 * tabStep)
            {
                int k0 = itab[it], k1 = itab[it + tabStep];
                Re(ref d, i) = Re(ref s, k0);
                Im(ref d, i) = Im(ref s, k0);
                Re(ref d, i + 1) = Re(ref s, k1);
                Im(ref d, i + 1) = Im(ref s, k1);
            }

            if (i < n)
            {
                Re(ref d, n - 1) = Re(ref s, n - 1);
                Im(ref d, n - 1) = Im(ref s, n - 1);
            }
        }
        else
        {
            for (i = 0; i <= n - 2; i += 2, it += 2 * tabStep)
            {
                int k0 = itab[it], k1 = itab[it + tabStep];
                Re(ref d, i) = Re(ref s, k0);
                Im(ref d, i) = TL.Neg(Im(ref s, k0));
                Re(ref d, i + 1) = Re(ref s, k1);
                Im(ref d, i + 1) = TL.Neg(Im(ref s, k1));
            }

            if (i < n)
            {
                Re(ref d, i) = Re(ref s, n - 1);
                Im(ref d, i) = TL.Neg(Im(ref s, n - 1));
            }
        }
    }

    private static void Radix4(ref TV d, ref T wave, int n, int nn, int nx, int dw0)
    {
        for (int i = 0; i < n; i += nn)
        {
            int v0 = i, v1 = i + (nx * 2);

            TV r0 = Re(ref d, v1), i0 = Im(ref d, v1);
            TV r4 = Re(ref d, v1 + nx), i4 = Im(ref d, v1 + nx);
            TV r1 = TL.Add(r0, r4), i1 = TL.Add(i0, i4);
            TV r3 = TL.Sub(i0, i4), i3 = TL.Sub(r4, r0);

            TV r2 = Re(ref d, v0), i2 = Im(ref d, v0);
            r4 = Re(ref d, v0 + nx);
            i4 = Im(ref d, v0 + nx);
            r0 = TL.Add(r2, r4);
            i0 = TL.Add(i2, i4);
            r2 = TL.Sub(r2, r4);
            i2 = TL.Sub(i2, i4);

            Re(ref d, v0) = TL.Add(r0, r1);
            Im(ref d, v0) = TL.Add(i0, i1);
            Re(ref d, v1) = TL.Sub(r0, r1);
            Im(ref d, v1) = TL.Sub(i0, i1);
            Re(ref d, v0 + nx) = TL.Add(r2, r3);
            Im(ref d, v0 + nx) = TL.Add(i2, i3);
            Re(ref d, v1 + nx) = TL.Sub(r2, r3);
            Im(ref d, v1 + nx) = TL.Sub(i2, i3);

            ref TV a0 = ref Unsafe.Add(ref d, 2 * (i + 1));
            ref T w1 = ref Unsafe.Add(ref wave, 2 * dw0);
            ref T w2 = ref Unsafe.Add(ref wave, 4 * dw0);
            ref T w3 = ref Unsafe.Add(ref wave, 6 * dw0);
            int s = 2 * nx;
            for (int j = 1; j < nx; j++)
            {
                ref TV a1 = ref Unsafe.Add(ref a0, s);
                ref TV b0 = ref Unsafe.Add(ref a0, 2 * s);
                ref TV b1 = ref Unsafe.Add(ref a0, 3 * s);

                TV w2r = TL.Splat(w2), w2i = TL.Splat(Unsafe.Add(ref w2, 1));
                TV w1r = TL.Splat(w1), w1i = TL.Splat(Unsafe.Add(ref w1, 1));
                TV w3r = TL.Splat(w3), w3i = TL.Splat(Unsafe.Add(ref w3, 1));

                TV ar = a1, ai = Unsafe.Add(ref a1, 1);
                r2 = TL.Sub(TL.Mul(ar, w2r), TL.Mul(ai, w2i));
                i2 = TL.Add(TL.Mul(ar, w2i), TL.Mul(ai, w2r));
                ar = b0;
                ai = Unsafe.Add(ref b0, 1);
                r0 = TL.Add(TL.Mul(ar, w1i), TL.Mul(ai, w1r));
                i0 = TL.Sub(TL.Mul(ar, w1r), TL.Mul(ai, w1i));
                ar = b1;
                ai = Unsafe.Add(ref b1, 1);
                r3 = TL.Add(TL.Mul(ar, w3i), TL.Mul(ai, w3r));
                i3 = TL.Sub(TL.Mul(ar, w3r), TL.Mul(ai, w3i));

                r1 = TL.Add(i0, i3);
                i1 = TL.Add(r0, r3);
                r3 = TL.Sub(r0, r3);
                i3 = TL.Sub(i3, i0);
                r4 = a0;
                i4 = Unsafe.Add(ref a0, 1);

                r0 = TL.Add(r4, r2);
                i0 = TL.Add(i4, i2);
                r2 = TL.Sub(r4, r2);
                i2 = TL.Sub(i4, i2);

                a0 = TL.Add(r0, r1);
                Unsafe.Add(ref a0, 1) = TL.Add(i0, i1);
                b0 = TL.Sub(r0, r1);
                Unsafe.Add(ref b0, 1) = TL.Sub(i0, i1);
                a1 = TL.Add(r2, r3);
                Unsafe.Add(ref a1, 1) = TL.Add(i2, i3);
                b1 = TL.Sub(r2, r3);
                Unsafe.Add(ref b1, 1) = TL.Sub(i2, i3);

                a0 = ref Unsafe.Add(ref a0, 2);
                w1 = ref Unsafe.Add(ref w1, 2 * dw0);
                w2 = ref Unsafe.Add(ref w2, 4 * dw0);
                w3 = ref Unsafe.Add(ref w3, 6 * dw0);
            }
        }
    }

    private static void Radix2(ref TV d, ref T wave, int cn, int n, int dw0)
    {
        int nx = n / 2;
        for (int i = 0; i < cn; i += n)
        {
            int v = i;
            TV r0 = TL.Add(Re(ref d, v), Re(ref d, v + nx));
            TV i0 = TL.Add(Im(ref d, v), Im(ref d, v + nx));
            TV r1 = TL.Sub(Re(ref d, v), Re(ref d, v + nx));
            TV i1 = TL.Sub(Im(ref d, v), Im(ref d, v + nx));
            Re(ref d, v) = r0;
            Im(ref d, v) = i0;
            Re(ref d, v + nx) = r1;
            Im(ref d, v + nx) = i1;

            ref TV a = ref Unsafe.Add(ref d, 2 * (i + 1));
            ref T w = ref Unsafe.Add(ref wave, 2 * dw0);
            for (int j = 1; j < nx; j++)
            {
                ref TV b = ref Unsafe.Add(ref a, 2 * nx);
                TV wr = TL.Splat(w), wi = TL.Splat(Unsafe.Add(ref w, 1));
                TV br = b, bi = Unsafe.Add(ref b, 1);
                r1 = TL.Sub(TL.Mul(br, wr), TL.Mul(bi, wi));
                i1 = TL.Add(TL.Mul(bi, wr), TL.Mul(br, wi));
                r0 = a;
                i0 = Unsafe.Add(ref a, 1);

                a = TL.Add(r0, r1);
                Unsafe.Add(ref a, 1) = TL.Add(i0, i1);
                b = TL.Sub(r0, r1);
                Unsafe.Add(ref b, 1) = TL.Sub(i0, i1);

                a = ref Unsafe.Add(ref a, 2);
                w = ref Unsafe.Add(ref w, 2 * dw0);
            }
        }
    }

    private static void Radix3(ref TV d, ref T wave, int cn, int n, int dw0)
    {
        int nx = n / 3;
        TV sin120 = TL.Splat(Sin120);
        TV half = TL.Splat(Half);
        for (int i = 0; i < cn; i += n)
        {
            int v = i;
            TV r1 = TL.Add(Re(ref d, v + nx), Re(ref d, v + (nx * 2)));
            TV i1 = TL.Add(Im(ref d, v + nx), Im(ref d, v + (nx * 2)));
            TV r0 = Re(ref d, v);
            TV i0 = Im(ref d, v);
            TV r2 = TL.Mul(sin120, TL.Sub(Im(ref d, v + nx), Im(ref d, v + (nx * 2))));
            TV i2 = TL.Mul(sin120, TL.Sub(Re(ref d, v + (nx * 2)), Re(ref d, v + nx)));
            Re(ref d, v) = TL.Add(r0, r1);
            Im(ref d, v) = TL.Add(i0, i1);
            r0 = TL.Sub(r0, TL.Mul(half, r1));
            i0 = TL.Sub(i0, TL.Mul(half, i1));
            Re(ref d, v + nx) = TL.Add(r0, r2);
            Im(ref d, v + nx) = TL.Add(i0, i2);
            Re(ref d, v + (nx * 2)) = TL.Sub(r0, r2);
            Im(ref d, v + (nx * 2)) = TL.Sub(i0, i2);

            for (int j = 1, dw = dw0; j < nx; j++, dw += dw0)
            {
                v = i + j;
                TV w1r = WRe(ref wave, dw), w1i = WIm(ref wave, dw);
                TV w2r = WRe(ref wave, dw * 2), w2i = WIm(ref wave, dw * 2);
                TV ar = Re(ref d, v + nx), ai = Im(ref d, v + nx);
                r0 = TL.Sub(TL.Mul(ar, w1r), TL.Mul(ai, w1i));
                i0 = TL.Add(TL.Mul(ar, w1i), TL.Mul(ai, w1r));
                ar = Re(ref d, v + (nx * 2));
                ai = Im(ref d, v + (nx * 2));
                i2 = TL.Sub(TL.Mul(ar, w2r), TL.Mul(ai, w2i));
                r2 = TL.Add(TL.Mul(ar, w2i), TL.Mul(ai, w2r));
                r1 = TL.Add(r0, i2);
                i1 = TL.Add(i0, r2);

                r2 = TL.Mul(sin120, TL.Sub(i0, r2));
                i2 = TL.Mul(sin120, TL.Sub(i2, r0));
                r0 = Re(ref d, v);
                i0 = Im(ref d, v);
                Re(ref d, v) = TL.Add(r0, r1);
                Im(ref d, v) = TL.Add(i0, i1);
                r0 = TL.Sub(r0, TL.Mul(half, r1));
                i0 = TL.Sub(i0, TL.Mul(half, i1));
                Re(ref d, v + nx) = TL.Add(r0, r2);
                Im(ref d, v + nx) = TL.Add(i0, i2);
                Re(ref d, v + (nx * 2)) = TL.Sub(r0, r2);
                Im(ref d, v + (nx * 2)) = TL.Sub(i0, i2);
            }
        }
    }

    private static void Radix5(ref TV d, ref T wave, int cn, int n, int dw0)
    {
        int nx = n / 5;
        TV quarter = TL.Splat(Quarter);
        TV f2 = TL.Splat(Fft5_2);
        TV f3 = TL.Splat(Fft5_3);
        TV negF3 = TL.Splat(-Fft5_3);
        TV negF5 = TL.Splat(-Fft5_5);
        TV f5 = TL.Splat(Fft5_5);
        TV negF4 = TL.Splat(-Fft5_4);
        TV f4 = TL.Splat(Fft5_4);
        for (int i = 0; i < cn; i += n)
        {
            for (int j = 0, dw = 0; j < nx; j++, dw += dw0)
            {
                int v0 = i + j;
                int v1 = v0 + (nx * 2);
                int v2 = v1 + (nx * 2);

                TV wr = WRe(ref wave, dw), wi = WIm(ref wave, dw);
                TV ar = Re(ref d, v0 + nx), ai = Im(ref d, v0 + nx);
                TV r3 = TL.Sub(TL.Mul(ar, wr), TL.Mul(ai, wi));
                TV i3 = TL.Add(TL.Mul(ar, wi), TL.Mul(ai, wr));
                wr = WRe(ref wave, dw * 4);
                wi = WIm(ref wave, dw * 4);
                ar = Re(ref d, v2);
                ai = Im(ref d, v2);
                TV r2 = TL.Sub(TL.Mul(ar, wr), TL.Mul(ai, wi));
                TV i2 = TL.Add(TL.Mul(ar, wi), TL.Mul(ai, wr));

                TV r1 = TL.Add(r3, r2), i1 = TL.Add(i3, i2);
                r3 = TL.Sub(r3, r2);
                i3 = TL.Sub(i3, i2);

                wr = WRe(ref wave, dw * 3);
                wi = WIm(ref wave, dw * 3);
                ar = Re(ref d, v1 + nx);
                ai = Im(ref d, v1 + nx);
                TV r4 = TL.Sub(TL.Mul(ar, wr), TL.Mul(ai, wi));
                TV i4 = TL.Add(TL.Mul(ar, wi), TL.Mul(ai, wr));
                wr = WRe(ref wave, dw * 2);
                wi = WIm(ref wave, dw * 2);
                ar = Re(ref d, v1);
                ai = Im(ref d, v1);
                TV r0 = TL.Sub(TL.Mul(ar, wr), TL.Mul(ai, wi));
                TV i0 = TL.Add(TL.Mul(ar, wi), TL.Mul(ai, wr));

                r2 = TL.Add(r4, r0);
                i2 = TL.Add(i4, i0);
                r4 = TL.Sub(r4, r0);
                i4 = TL.Sub(i4, i0);

                r0 = Re(ref d, v0);
                i0 = Im(ref d, v0);
                TV r5 = TL.Add(r1, r2), i5 = TL.Add(i1, i2);

                Re(ref d, v0) = TL.Add(r0, r5);
                Im(ref d, v0) = TL.Add(i0, i5);

                r0 = TL.Sub(r0, TL.Mul(quarter, r5));
                i0 = TL.Sub(i0, TL.Mul(quarter, i5));
                r1 = TL.Mul(f2, TL.Sub(r1, r2));
                i1 = TL.Mul(f2, TL.Sub(i1, i2));
                r2 = TL.Mul(negF3, TL.Add(i3, i4));
                i2 = TL.Mul(f3, TL.Add(r3, r4));

                i3 = TL.Mul(i3, negF5);
                r3 = TL.Mul(r3, f5);
                i4 = TL.Mul(i4, negF4);
                r4 = TL.Mul(r4, f4);

                r5 = TL.Add(r2, i3);
                i5 = TL.Add(i2, r3);
                r2 = TL.Sub(r2, i4);
                i2 = TL.Sub(i2, r4);

                r3 = TL.Add(r0, r1);
                i3 = TL.Add(i0, i1);
                r0 = TL.Sub(r0, r1);
                i0 = TL.Sub(i0, i1);

                Re(ref d, v0 + nx) = TL.Add(r3, r2);
                Im(ref d, v0 + nx) = TL.Add(i3, i2);
                Re(ref d, v2) = TL.Sub(r3, r2);
                Im(ref d, v2) = TL.Sub(i3, i2);

                Re(ref d, v1) = TL.Add(r0, r5);
                Im(ref d, v1) = TL.Add(i0, i5);
                Re(ref d, v1 + nx) = TL.Sub(r0, r5);
                Im(ref d, v1 + nx) = TL.Sub(i0, i5);
            }
        }
    }

    private static void RadixOdd(ref TV d, ref T wave, int cn, int n, int nx, int dw0, int factor, int tabSize, Span<TV> scratch)
    {
        int factor2 = (factor - 1) / 2;
        int dwF = tabSize / factor;
        ref TV a = ref MemoryMarshal.GetReference(scratch);
        ref TV b = ref Unsafe.Add(ref a, 2 * factor2);

        for (int i = 0; i < cn; i += n)
        {
            for (int j = 0, dw = 0; j < nx; j++, dw += dw0)
            {
                int v = i + j;
                TV v0r = Re(ref d, v), v0i = Im(ref d, v);
                TV vnr = v0r, vni = v0i;

                if (j == 0)
                {
                    for (int p = 1, k = nx; p <= factor2; p++, k += nx)
                    {
                        TV r0 = TL.Add(Re(ref d, v + k), Re(ref d, v + n - k));
                        TV i0 = TL.Sub(Im(ref d, v + k), Im(ref d, v + n - k));
                        TV r1 = TL.Sub(Re(ref d, v + k), Re(ref d, v + n - k));
                        TV i1 = TL.Add(Im(ref d, v + k), Im(ref d, v + n - k));

                        vnr = TL.Add(vnr, r0);
                        vni = TL.Add(vni, i1);
                        Re(ref a, p - 1) = r0;
                        Im(ref a, p - 1) = i0;
                        Re(ref b, p - 1) = r1;
                        Im(ref b, p - 1) = i1;
                    }
                }
                else
                {
                    int waveBase = dw * factor;
                    int dd = dw;
                    for (int p = 1, k = nx; p <= factor2; p++, k += nx, dd += dw)
                    {
                        TV wr = WRe(ref wave, dd), wi = WIm(ref wave, dd);
                        TV xr = Re(ref d, v + k), xi = Im(ref d, v + k);
                        TV r2 = TL.Sub(TL.Mul(xr, wr), TL.Mul(xi, wi));
                        TV i2 = TL.Add(TL.Mul(xr, wi), TL.Mul(xi, wr));

                        wr = WRe(ref wave, waveBase - dd);
                        wi = WIm(ref wave, waveBase - dd);
                        xr = Re(ref d, v + n - k);
                        xi = Im(ref d, v + n - k);
                        TV r1 = TL.Sub(TL.Mul(xr, wr), TL.Mul(xi, wi));
                        TV i1 = TL.Add(TL.Mul(xr, wi), TL.Mul(xi, wr));

                        TV r0 = TL.Add(r2, r1);
                        TV i0 = TL.Sub(i2, i1);
                        r1 = TL.Sub(r2, r1);
                        i1 = TL.Add(i2, i1);

                        vnr = TL.Add(vnr, r0);
                        vni = TL.Add(vni, i1);
                        Re(ref a, p - 1) = r0;
                        Im(ref a, p - 1) = i0;
                        Re(ref b, p - 1) = r1;
                        Im(ref b, p - 1) = i1;
                    }
                }

                Re(ref d, v) = vnr;
                Im(ref d, v) = vni;

                for (int p = 1, k = nx; p <= factor2; p++, k += nx)
                {
                    TV s0r = v0r, s0i = v0i, s1r = v0r, s1i = v0i;
                    int di = dwF * p, dd = di;

                    for (int q = 0; q < factor2; q++)
                    {
                        TV wr = WRe(ref wave, di), wi = WIm(ref wave, di);
                        TV r0 = TL.Mul(wr, Re(ref a, q));
                        TV i0 = TL.Mul(wi, Im(ref a, q));
                        TV r1 = TL.Mul(wr, Im(ref b, q));
                        TV i1 = TL.Mul(wi, Re(ref b, q));

                        s1r = TL.Add(s1r, TL.Add(r0, i0));
                        s0r = TL.Add(s0r, TL.Sub(r0, i0));
                        s1i = TL.Add(s1i, TL.Sub(r1, i1));
                        s0i = TL.Add(s0i, TL.Add(r1, i1));

                        di += dd;
                        if (di >= tabSize)
                        {
                            di -= tabSize;
                        }
                    }

                    Re(ref d, v + k) = s0r;
                    Im(ref d, v + k) = s0i;
                    Re(ref d, v + n - k) = s1r;
                    Im(ref d, v + n - k) = s1i;
                }
            }
        }
    }
}
