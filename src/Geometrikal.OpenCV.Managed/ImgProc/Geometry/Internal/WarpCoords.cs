// This implementation is ported from the OpenCV warp coordinate generators and map conversion.
//
// Original sources:
//   modules/imgproc/src/imgwarp.cpp (WarpAffineInvoker, hal::warpAffineBlocklineNN,
//     hal::warpAffineBlockline, WarpPerspectiveInvoker, hal::warpPerspectiveBlocklineNN,
//     hal::warpPerspectiveBlockline, RemapInvoker, NNDeltaTab_i)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Produces the integer source coordinates <c>xy</c> (pairs) and, for interpolating kernels, the
/// table index <c>fxy</c> of one destination row.
/// </summary>
internal interface IRowCoords
{
    void Row(int y, Span<short> xy, Span<ushort> fxy, bool nearest);
}

internal static class CoordMath
{
    public const int AbBits = 10;
    public const int AbScale = 1 << AbBits;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static short SatShort(int v) => (short)Math.Clamp(v, short.MinValue, short.MaxValue);

    /// <summary><c>saturate_cast&lt;int&gt;(double)</c>, which is <c>cvRound</c>: out-of-range values give <see cref="int.MinValue"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int SatInt(double v) => ScalarConverter.CvRound(v);

    /// <summary>Splits fixed-point coordinates with 5 fraction bits into <c>xy</c> and the table index.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Split(int x, int y, Span<short> xy, Span<ushort> fxy, int i)
    {
        xy[2 * i] = SatShort(x >> InterTables.Bits);
        xy[(2 * i) + 1] = SatShort(y >> InterTables.Bits);
        fxy[i] = (ushort)(((y & (InterTables.Size - 1)) * InterTables.Size) + (x & (InterTables.Size - 1)));
    }

    /// <summary>
    /// <see cref="Split"/> for 8 pixels: <c>x</c>/<c>y</c> hold fixed-point coordinates with 5
    /// fraction bits, or integer coordinates for nearest-neighbour (no table index).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store8(
        Vector128<int> x0, Vector128<int> x1, Vector128<int> y0, Vector128<int> y1, ref short xy, ref ushort fxy, int i, bool nearest)
    {
        Vector128<short> xs, ys;
        if (nearest)
        {
            xs = Sse2.PackSignedSaturate(x0, x1);
            ys = Sse2.PackSignedSaturate(y0, y1);
        }
        else
        {
            xs = Sse2.PackSignedSaturate(Sse2.ShiftRightArithmetic(x0, InterTables.Bits), Sse2.ShiftRightArithmetic(x1, InterTables.Bits));
            ys = Sse2.PackSignedSaturate(Sse2.ShiftRightArithmetic(y0, InterTables.Bits), Sse2.ShiftRightArithmetic(y1, InterTables.Bits));
            var mask = Vector128.Create(InterTables.Size - 1);
            Vector128<int> a0 = Sse2.ShiftLeftLogical(y0 & mask, InterTables.Bits) | (x0 & mask);
            Vector128<int> a1 = Sse2.ShiftLeftLogical(y1 & mask, InterTables.Bits) | (x1 & mask);
            Sse2.PackSignedSaturate(a0, a1).AsUInt16().StoreUnsafe(ref fxy, (nuint)i);
        }

        Sse2.UnpackLow(xs, ys).StoreUnsafe(ref xy, (nuint)(2 * i));
        Sse2.UnpackHigh(xs, ys).StoreUnsafe(ref xy, (nuint)((2 * i) + 8));
    }
}

/// <summary>Affine coordinates on a 1/1024 grid: <c>X = round((M1·y + M2)·1024) + M0·x·1024</c>.</summary>
internal readonly struct AffineCoords : IRowCoords, IDisposable
{
    private readonly int[] _adelta;
    private readonly int[] _bdelta;
    private readonly double[] _m;
    private readonly int _width;

    public AffineCoords(double[] m, int width)
    {
        _m = m;
        _width = width;
        _adelta = ArrayPool<int>.Shared.Rent(width);
        _bdelta = ArrayPool<int>.Shared.Rent(width);
        for (int x = 0; x < width; x++)
        {
            _adelta[x] = CoordMath.SatInt(m[0] * x * CoordMath.AbScale);
            _bdelta[x] = CoordMath.SatInt(m[3] * x * CoordMath.AbScale);
        }
    }

    public void Dispose()
    {
        ArrayPool<int>.Shared.Return(_adelta);
        ArrayPool<int>.Shared.Return(_bdelta);
    }

    public void Row(int y, Span<short> xy, Span<ushort> fxy, bool nearest)
    {
        int roundDelta = nearest ? CoordMath.AbScale / 2 : CoordMath.AbScale / InterTables.Size / 2;
        int x0 = unchecked(CoordMath.SatInt(((_m[1] * y) + _m[2]) * CoordMath.AbScale) + roundDelta);
        int y0 = unchecked(CoordMath.SatInt(((_m[4] * y) + _m[5]) * CoordMath.AbScale) + roundDelta);
        ref int ad = ref MemoryMarshal.GetArrayDataReference(_adelta);
        ref int bd = ref MemoryMarshal.GetArrayDataReference(_bdelta);
        int width = _width;
        int x = 0;
        if (nearest)
        {
            if (Sse2.IsSupported)
            {
                ref short xyr = ref MemoryMarshal.GetReference(xy);
                ref ushort fr = ref MemoryMarshal.GetReference(fxy);
                Vector128<int> vx0 = Vector128.Create(x0), vy0 = Vector128.Create(y0);
                for (; x <= width - 8; x += 8)
                {
                    Vector128<int> xa = Sse2.ShiftRightArithmetic(vx0 + Vector128.LoadUnsafe(ref ad, (nuint)x), CoordMath.AbBits);
                    Vector128<int> xb = Sse2.ShiftRightArithmetic(vx0 + Vector128.LoadUnsafe(ref ad, (nuint)(x + 4)), CoordMath.AbBits);
                    Vector128<int> ya = Sse2.ShiftRightArithmetic(vy0 + Vector128.LoadUnsafe(ref bd, (nuint)x), CoordMath.AbBits);
                    Vector128<int> yb = Sse2.ShiftRightArithmetic(vy0 + Vector128.LoadUnsafe(ref bd, (nuint)(x + 4)), CoordMath.AbBits);
                    CoordMath.Store8(xa, xb, ya, yb, ref xyr, ref fr, x, nearest: true);
                }
            }

            for (; x < width; x++)
            {
                xy[2 * x] = CoordMath.SatShort(unchecked(x0 + Unsafe.Add(ref ad, x)) >> CoordMath.AbBits);
                xy[(2 * x) + 1] = CoordMath.SatShort(unchecked(y0 + Unsafe.Add(ref bd, x)) >> CoordMath.AbBits);
            }

            return;
        }

        const int shift = CoordMath.AbBits - InterTables.Bits;
        if (Sse2.IsSupported)
        {
            ref short xyr = ref MemoryMarshal.GetReference(xy);
            ref ushort fr = ref MemoryMarshal.GetReference(fxy);
            Vector128<int> vx0 = Vector128.Create(x0), vy0 = Vector128.Create(y0);
            for (; x <= width - 8; x += 8)
            {
                Vector128<int> xa = Sse2.ShiftRightArithmetic(vx0 + Vector128.LoadUnsafe(ref ad, (nuint)x), shift);
                Vector128<int> xb = Sse2.ShiftRightArithmetic(vx0 + Vector128.LoadUnsafe(ref ad, (nuint)(x + 4)), shift);
                Vector128<int> ya = Sse2.ShiftRightArithmetic(vy0 + Vector128.LoadUnsafe(ref bd, (nuint)x), shift);
                Vector128<int> yb = Sse2.ShiftRightArithmetic(vy0 + Vector128.LoadUnsafe(ref bd, (nuint)(x + 4)), shift);
                CoordMath.Store8(xa, xb, ya, yb, ref xyr, ref fr, x, nearest: false);
            }
        }

        for (; x < width; x++)
        {
            int xs = unchecked(x0 + Unsafe.Add(ref ad, x)) >> shift;
            int ys = unchecked(y0 + Unsafe.Add(ref bd, x)) >> shift;
            CoordMath.Split(xs, ys, xy, fxy, x);
        }
    }
}

/// <summary>
/// Perspective coordinates. The row is split into blocks of <c>bw0</c> columns as in
/// <c>WarpPerspectiveInvoker</c>; the block start enters the double expression.
/// </summary>
internal readonly struct PerspectiveCoords : IRowCoords
{
    private readonly double[] _m;
    private readonly int _width;
    private readonly int _blockWidth;

    public PerspectiveCoords(double[] m, int width, int height)
    {
        _m = m;
        _width = width;
        int bh0 = Math.Min(16, height);
        _blockWidth = Math.Min(1024 / bh0, width);
    }

    public void Row(int y, Span<short> xy, Span<ushort> fxy, bool nearest)
    {
        double[] m = _m;
        double num = nearest ? 1.0 : InterTables.Size;
        for (int xb = 0; xb < _width; xb += _blockWidth)
        {
            double x0 = (m[0] * xb) + (m[1] * y) + m[2];
            double y0 = (m[3] * xb) + (m[4] * y) + m[5];
            double w0 = (m[6] * xb) + (m[7] * y) + m[8];
            int bw = Math.Min(_blockWidth, _width - xb);
            int x1 = 0;
            if (Avx.IsSupported)
            {
                ref short xyr = ref MemoryMarshal.GetReference(xy);
                ref ushort fr = ref MemoryMarshal.GetReference(fxy);
                Vector256<double> vm0 = Vector256.Create(m[0]), vm3 = Vector256.Create(m[3]), vm6 = Vector256.Create(m[6]);
                Vector256<double> vx0 = Vector256.Create(x0), vy0 = Vector256.Create(y0), vw0 = Vector256.Create(w0);
                Vector256<double> vnum = Vector256.Create(num);
                Vector256<double> lo = Vector256.Create((double)int.MinValue), hi = Vector256.Create((double)int.MaxValue);
                Vector256<double> lanes = Vector256.Create(0.0, 1, 2, 3);
                for (; x1 <= bw - 8; x1 += 8)
                {
                    Vector256<double> xa = Vector256.Create((double)x1) + lanes;
                    Vector256<double> xb4 = xa + Vector256.Create(4.0);
                    PerspectiveLanes(xa, vm0, vm3, vm6, vx0, vy0, vw0, vnum, lo, hi, out Vector128<int> ia, out Vector128<int> ja);
                    PerspectiveLanes(xb4, vm0, vm3, vm6, vx0, vy0, vw0, vnum, lo, hi, out Vector128<int> ib, out Vector128<int> jb);
                    CoordMath.Store8(ia, ib, ja, jb, ref xyr, ref fr, xb + x1, nearest);
                }
            }

            for (; x1 < bw; x1++)
            {
                double w = w0 + (m[6] * x1);
                w = w != 0 ? num / w : 0;
                double fx = MaxD(int.MinValue, MinD(int.MaxValue, (x0 + (m[0] * x1)) * w));
                double fy = MaxD(int.MinValue, MinD(int.MaxValue, (y0 + (m[3] * x1)) * w));
                int xs = CoordMath.SatInt(fx);
                int ys = CoordMath.SatInt(fy);
                int i = xb + x1;
                if (nearest)
                {
                    xy[2 * i] = CoordMath.SatShort(xs);
                    xy[(2 * i) + 1] = CoordMath.SatShort(ys);
                }
                else
                {
                    CoordMath.Split(xs, ys, xy, fxy, i);
                }
            }
        }
    }

    /// <summary>
    /// Four pixels of the scalar loop: the same double operations per lane; <c>minpd</c>/<c>maxpd</c>
    /// order the operands as <c>std::min</c>/<c>std::max</c> and <c>cvtpd2dq</c> rounds as <c>cvRound</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PerspectiveLanes(
        Vector256<double> x1, Vector256<double> m0, Vector256<double> m3, Vector256<double> m6, Vector256<double> x0, Vector256<double> y0,
        Vector256<double> w0, Vector256<double> num, Vector256<double> lo, Vector256<double> hi, out Vector128<int> xs, out Vector128<int> ys)
    {
        Vector256<double> w = w0 + (m6 * x1);
        Vector256<double> nonzero = Avx.CompareNotEqual(w, Vector256<double>.Zero);
        w = Avx.And(num / w, nonzero);
        Vector256<double> fx = Avx.Max(Avx.Min((x0 + (m0 * x1)) * w, hi), lo);
        Vector256<double> fy = Avx.Max(Avx.Min((y0 + (m3 * x1)) * w, hi), lo);
        xs = Avx.ConvertToVector128Int32(fx);
        ys = Avx.ConvertToVector128Int32(fy);
    }

    /// <summary><c>std::min(a, b)</c>: <c>b &lt; a ? b : a</c>.</summary>
    private static double MinD(double a, double b) => b < a ? b : a;

    /// <summary><c>std::max(a, b)</c>: <c>a &lt; b ? b : a</c>.</summary>
    private static double MaxD(double a, double b) => a < b ? b : a;
}

/// <summary>Floating-point maps, planar (two 1-channel maps) or interleaved (one 2-channel map).</summary>
internal readonly struct FloatMapCoords : IRowCoords
{
    private readonly Mat<float> _map1;
    private readonly Mat<float>? _map2;

    public FloatMapCoords(Mat<float> map1, Mat<float>? map2)
    {
        _map1 = map1;
        _map2 = map2;
    }

    public void Row(int y, Span<short> xy, Span<ushort> fxy, bool nearest)
    {
        ReadOnlySpan<float> m1 = _map1.GetReadOnlyRowSpan(y);
        ReadOnlySpan<float> m2 = _map2 is null ? default : _map2.GetReadOnlyRowSpan(y);
        bool planar = _map2 is not null;
        int width = planar ? m1.Length : m1.Length / 2;
        int x = 0;
        if (Sse2.IsSupported)
        {
            ref float p1 = ref MemoryMarshal.GetReference(m1);
            ref float p2 = ref MemoryMarshal.GetReference(m2);
            ref short xyr = ref MemoryMarshal.GetReference(xy);
            ref ushort fr = ref MemoryMarshal.GetReference(fxy);
            Vector128<float> scale = Vector128.Create(nearest ? 1f : InterTables.Size);
            for (; x <= width - 8; x += 8)
            {
                Vector128<float> xa, xb, ya, yb;
                if (planar)
                {
                    xa = Vector128.LoadUnsafe(ref p1, (nuint)x);
                    xb = Vector128.LoadUnsafe(ref p1, (nuint)(x + 4));
                    ya = Vector128.LoadUnsafe(ref p2, (nuint)x);
                    yb = Vector128.LoadUnsafe(ref p2, (nuint)(x + 4));
                }
                else
                {
                    Vector128<float> q0 = Vector128.LoadUnsafe(ref p1, (nuint)(2 * x));
                    Vector128<float> q1 = Vector128.LoadUnsafe(ref p1, (nuint)((2 * x) + 4));
                    Vector128<float> q2 = Vector128.LoadUnsafe(ref p1, (nuint)((2 * x) + 8));
                    Vector128<float> q3 = Vector128.LoadUnsafe(ref p1, (nuint)((2 * x) + 12));
                    xa = Sse.Shuffle(q0, q1, 0b10_00_10_00);
                    ya = Sse.Shuffle(q0, q1, 0b11_01_11_01);
                    xb = Sse.Shuffle(q2, q3, 0b10_00_10_00);
                    yb = Sse.Shuffle(q2, q3, 0b11_01_11_01);
                }

                // cvtps2dq rounds half to even and gives int.MinValue out of range, as cvRound(float).
                CoordMath.Store8(
                    Sse2.ConvertToVector128Int32(xa * scale),
                    Sse2.ConvertToVector128Int32(xb * scale),
                    Sse2.ConvertToVector128Int32(ya * scale),
                    Sse2.ConvertToVector128Int32(yb * scale),
                    ref xyr,
                    ref fr,
                    x,
                    nearest);
            }
        }

        for (; x < width; x++)
        {
            float mx = planar ? m1[x] : m1[2 * x];
            float my = planar ? m2[x] : m1[(2 * x) + 1];
            if (nearest)
            {
                xy[2 * x] = CoordMath.SatShort(ScalarConverter.CvRound(mx));
                xy[(2 * x) + 1] = CoordMath.SatShort(ScalarConverter.CvRound(my));
            }
            else
            {
                CoordMath.Split(ScalarConverter.CvRound(mx * InterTables.Size), ScalarConverter.CvRound(my * InterTables.Size), xy, fxy, x);
            }
        }
    }
}

/// <summary>
/// Fixed-point maps: integer pairs and optional table indices. Nearest-neighbour remap adds 1 to
/// a coordinate whose fraction is below one half (<c>NNDeltaTab_i</c>), as OpenCV does.
/// </summary>
internal readonly struct FixedMapCoords : IRowCoords
{
    private readonly Mat<short> _map1;
    private readonly Mat<ushort>? _map2;

    public FixedMapCoords(Mat<short> map1, Mat<ushort>? map2)
    {
        _map1 = map1;
        _map2 = map2;
    }

    public void Row(int y, Span<short> xy, Span<ushort> fxy, bool nearest)
    {
        ReadOnlySpan<short> m1 = _map1.GetReadOnlyRowSpan(y);
        int width = m1.Length / 2;
        if (_map2 is null)
        {
            m1.CopyTo(xy);
            return;
        }

        ReadOnlySpan<ushort> m2 = _map2.GetReadOnlyRowSpan(y);
        if (nearest)
        {
            for (int x = 0; x < width; x++)
            {
                int a = m2[x] & (InterTables.Size2 - 1);
                xy[2 * x] = (short)(m1[2 * x] + ((a & (InterTables.Size - 1)) < InterTables.Size / 2 ? 1 : 0));
                xy[(2 * x) + 1] = (short)(m1[(2 * x) + 1] + ((a >> InterTables.Bits) < InterTables.Size / 2 ? 1 : 0));
            }

            return;
        }

        m1.CopyTo(xy);
        for (int x = 0; x < width; x++)
        {
            fxy[x] = (ushort)(m2[x] & (InterTables.Size2 - 1));
        }
    }
}
