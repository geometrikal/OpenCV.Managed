// This implementation is ported from the OpenCV initUndistortRectifyMap computer.
//
// Original sources:
//   modules/imgproc/src/undistort.dispatch.cpp (initUndistortRectifyMap)
//   modules/imgproc/src/undistort.simd.hpp (initUndistortRectifyMapComputer)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Output layout of an undistortion map.</summary>
internal enum UndistortMapKind
{
    /// <summary>Two <c>CV_32FC1</c> maps.</summary>
    FloatPair,

    /// <summary>One <c>CV_32FC2</c> map.</summary>
    FloatInterleaved,

    /// <summary><c>CV_16SC2</c> and <c>CV_16UC1</c> maps.</summary>
    FixedPoint,
}

/// <summary>
/// Parameters of one <c>initUndistortRectifyMapComputer</c> run.
/// </summary>
internal sealed class UndistortMapParams
{
    public required double[] Ir { get; init; }

    public required double[] Tilt { get; init; }

    public required double[] K { get; init; }

    public double U0 { get; init; }

    public double V0 { get; init; }

    public double Fx { get; init; }

    public double Fy { get; init; }

    /// <summary>
    /// Builds the parameters as OpenCV <c>initUndistortRectifyMap</c>: <c>iR = (Ar·R)⁻¹</c>
    /// with <c>gemm</c> and <c>invert(DECOMP_LU)</c>.
    /// </summary>
    public static UndistortMapParams Create(double[] a, double[] k, double[] r, double[] ar)
    {
        Mat<double> prod = LinearAlgebra.Gemm(CameraDistortion.ToMat(ar), CameraDistortion.ToMat(r), 1, null, 0);
        LinearAlgebra.Invert(prod, out Mat<double> inv, DecompTypes.Lu);
        var tilt = new double[9];
        CameraDistortion.TiltProjection(k[12], k[13], tilt);
        return new UndistortMapParams
        {
            Ir = CameraDistortion.Read3x3(inv, nameof(inv)),
            Tilt = tilt,
            K = k,
            U0 = a[2],
            V0 = a[5],
            Fx = a[0],
            Fy = a[4],
        };
    }
}

/// <summary>
/// Map rows of <c>initUndistortRectifyMapComputer</c>. Pixels in complete blocks of four follow
/// the 128-bit <c>v_float64</c> body (baseline SSE, no FMA): lane <c>l</c> of a block uses
/// <c>_x + ir[0]·l</c>, the block base advances by <c>4·ir[0]</c>, and the arithmetic uses the
/// <c>v_muladd</c> order. The remaining pixels follow the scalar tail order. Fixed-point blocks
/// saturate the integer parts to 16 bits; the tail truncates them.
/// </summary>
internal static class UndistortMapKernel
{
    private const int Block = 4;
    private const int InterBits = 5;
    private const int InterTabSize = 1 << InterBits;

    public static void Run(
        UndistortMapParams p,
        int width,
        int height,
        UndistortMapKind kind,
        Mat<float>? mapX,
        Mat<float>? mapY,
        Mat<short>? xy,
        Mat<ushort>? fractions)
    {
        ParallelRows.For(height, (long)width * 48, (start, end) => Stripe(p, width, start, end, kind, mapX, mapY, xy, fractions));
    }

    private static void Stripe(
        UndistortMapParams p,
        int width,
        int start,
        int end,
        UndistortMapKind kind,
        Mat<float>? mapX,
        Mat<float>? mapY,
        Mat<short>? xy,
        Mat<ushort>? fractions)
    {
        double[] buffer = ArrayPool<double>.Shared.Rent(width * 5);
        try
        {
            Span<double> bx = buffer.AsSpan(0, width);
            Span<double> by = buffer.AsSpan(width, width);
            Span<double> bw = buffer.AsSpan(2 * width, width);
            Span<double> u = buffer.AsSpan(3 * width, width);
            Span<double> v = buffer.AsSpan(4 * width, width);
            for (int i = start; i < end; i++)
            {
                int simdEnd = ComputeRow(p, i, width, bx, by, bw, u, v);
                switch (kind)
                {
                    case UndistortMapKind.FloatPair:
                        StoreFloat(u, mapX!.GetRowSpan(i));
                        StoreFloat(v, mapY!.GetRowSpan(i));
                        break;
                    case UndistortMapKind.FloatInterleaved:
                        StoreInterleaved(u, v, mapX!.GetRowSpan(i));
                        break;
                    default:
                        StoreFixed(u, v, simdEnd, xy!.GetRowSpan(i), fractions!.GetRowSpan(i));
                        break;
                }
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(buffer);
        }
    }

    /// <summary>Computes the source coordinates of row <paramref name="i"/>; returns the end of the block part.</summary>
    internal static int ComputeRow(
        UndistortMapParams p,
        int i,
        int width,
        Span<double> bx,
        Span<double> by,
        Span<double> bw,
        Span<double> u,
        Span<double> v)
    {
        double[] ir = p.Ir;
        double x0 = i * ir[1] + ir[2], y0 = i * ir[4] + ir[5], w0 = i * ir[7] + ir[8];
        double sx0 = ir[0] * 0, sy0 = ir[3] * 0, sw0 = ir[6] * 0;
        double sx1 = ir[0] * 1, sy1 = ir[3] * 1, sw1 = ir[6] * 1;
        double sx2 = ir[0] * 2, sy2 = ir[3] * 2, sw2 = ir[6] * 2;
        double sx3 = ir[0] * 3, sy3 = ir[3] * 3, sw3 = ir[6] * 3;
        double dx = Block * ir[0], dy = Block * ir[3], dw = Block * ir[6];
        int j = 0;
        if (Vector256.IsHardwareAccelerated && width >= Block)
        {
            j = FusedBlocks(
                p,
                width,
                Vector256.Create(sx0, sx1, sx2, sx3),
                Vector256.Create(sy0, sy1, sy2, sy3),
                Vector256.Create(sw0, sw1, sw2, sw3),
                dx,
                dy,
                dw,
                ref x0,
                ref y0,
                ref w0,
                u,
                v);
            for (int t = j; t < width; t++, x0 += ir[0], y0 += ir[3], w0 += ir[6])
            {
                TailPixel(p, x0, y0, w0, out u[t], out v[t]);
            }

            return j;
        }

        for (; j <= width - Block; j += Block, x0 += dx, y0 += dy, w0 += dw)
        {
            bx[j] = x0 + sx0;
            bx[j + 1] = x0 + sx1;
            bx[j + 2] = x0 + sx2;
            bx[j + 3] = x0 + sx3;
            by[j] = y0 + sy0;
            by[j + 1] = y0 + sy1;
            by[j + 2] = y0 + sy2;
            by[j + 3] = y0 + sy3;
            bw[j] = w0 + sw0;
            bw[j + 1] = w0 + sw1;
            bw[j + 2] = w0 + sw2;
            bw[j + 3] = w0 + sw3;
        }

        int simdEnd = j;
        BlockPixels(p, bx[..simdEnd], by[..simdEnd], bw[..simdEnd], u, v);

        for (; j < width; j++, x0 += ir[0], y0 += ir[3], w0 += ir[6])
        {
            TailPixel(p, x0, y0, w0, out u[j], out v[j]);
        }

        return simdEnd;
    }

    /// <summary>
    /// Computes the complete blocks of a row in registers; lane <c>l</c> of a block reads
    /// <c>x0 + s[l]</c> as the scalar block setup does. Returns the end of the block part.
    /// </summary>
    private static int FusedBlocks(
        UndistortMapParams p,
        int width,
        Vector256<double> sx,
        Vector256<double> sy,
        Vector256<double> sw,
        double dx,
        double dy,
        double dw,
        ref double x0,
        ref double y0,
        ref double w0,
        Span<double> u,
        Span<double> v)
    {
        double[] k = p.K;
        double[] t = p.Tilt;
        var one = Vector256<double>.One;
        var two = Vector256.Create(2.0);
        var k1 = Vector256.Create(k[0]);
        var k2 = Vector256.Create(k[1]);
        var p1 = Vector256.Create(k[2]);
        var p2 = Vector256.Create(k[3]);
        var k3 = Vector256.Create(k[4]);
        var k4 = Vector256.Create(k[5]);
        var k5 = Vector256.Create(k[6]);
        var k6 = Vector256.Create(k[7]);
        var s1 = Vector256.Create(k[8]);
        var s2 = Vector256.Create(k[9]);
        var s3 = Vector256.Create(k[10]);
        var s4 = Vector256.Create(k[11]);
        var t0 = Vector256.Create(t[0]);
        var t1 = Vector256.Create(t[1]);
        var t2 = Vector256.Create(t[2]);
        var t3 = Vector256.Create(t[3]);
        var t4 = Vector256.Create(t[4]);
        var t5 = Vector256.Create(t[5]);
        var t6 = Vector256.Create(t[6]);
        var t7 = Vector256.Create(t[7]);
        var t8 = Vector256.Create(t[8]);
        var fx = Vector256.Create(p.Fx);
        var fy = Vector256.Create(p.Fy);
        var u0 = Vector256.Create(p.U0);
        var v0 = Vector256.Create(p.V0);
        ref double ru = ref MemoryMarshal.GetReference(u);
        ref double rv = ref MemoryMarshal.GetReference(v);
        double bx0 = x0, by0 = y0, bw0 = w0;
        int j = 0;
        for (; j <= width - Block; j += Block, bx0 += dx, by0 += dy, bw0 += dw)
        {
            var w = one / (Vector256.Create(bw0) + sw);
            var x = (Vector256.Create(bx0) + sx) * w;
            var y = (Vector256.Create(by0) + sy) * w;
            var xd = x * x;
            var yd = y * y;
            var r2 = xd + yd;
            var num = (((k3 * r2) + k2) * r2 + k1) * r2 + one;
            var kr = num / ((((k6 * r2) + k5) * r2 + k4) * r2 + one);
            xd = two * xd + r2;
            yd = two * yd + r2;
            var m2 = (x * y) * two;
            x *= kr;
            y *= kr;
            xd = xd * p2 + x;
            yd = yd * p1 + y;
            xd = p1 * m2 + xd;
            yd = p2 * m2 + yd;
            var r4 = r2 * r2;
            xd = s1 * r2 + (s2 * r4 + xd);
            yd = s3 * r2 + (s4 * r4 + yd);
            var tx = t0 * xd + (t1 * yd + t2);
            var ty = t3 * xd + (t4 * yd + t5);
            var tz = t6 * xd + (t7 * yd + t8);
            var iz = Vector256.ConditionalSelect(Vector256.Equals(tz, Vector256<double>.Zero), one, one / tz);
            ((fx * iz) * tx + u0).StoreUnsafe(ref ru, (nuint)j);
            ((fy * iz) * ty + v0).StoreUnsafe(ref rv, (nuint)j);
        }

        x0 = bx0;
        y0 = by0;
        w0 = bw0;
        return j;
    }

    private static void BlockPixels(UndistortMapParams p, ReadOnlySpan<double> bx, ReadOnlySpan<double> by, ReadOnlySpan<double> bw, Span<double> u, Span<double> v)
    {
        int n = bx.Length;
        int j = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
        {
            double[] k = p.K;
            double[] t = p.Tilt;
            var one = Vector<double>.One;
            var two = new Vector<double>(2.0);
            var k1 = new Vector<double>(k[0]);
            var k2 = new Vector<double>(k[1]);
            var p1 = new Vector<double>(k[2]);
            var p2 = new Vector<double>(k[3]);
            var k3 = new Vector<double>(k[4]);
            var k4 = new Vector<double>(k[5]);
            var k5 = new Vector<double>(k[6]);
            var k6 = new Vector<double>(k[7]);
            var s1 = new Vector<double>(k[8]);
            var s2 = new Vector<double>(k[9]);
            var s3 = new Vector<double>(k[10]);
            var s4 = new Vector<double>(k[11]);
            var t0 = new Vector<double>(t[0]);
            var t1 = new Vector<double>(t[1]);
            var t2 = new Vector<double>(t[2]);
            var t3 = new Vector<double>(t[3]);
            var t4 = new Vector<double>(t[4]);
            var t5 = new Vector<double>(t[5]);
            var t6 = new Vector<double>(t[6]);
            var t7 = new Vector<double>(t[7]);
            var t8 = new Vector<double>(t[8]);
            var fx = new Vector<double>(p.Fx);
            var fy = new Vector<double>(p.Fy);
            var u0 = new Vector<double>(p.U0);
            var v0 = new Vector<double>(p.V0);
            ref double rx = ref MemoryMarshal.GetReference(bx);
            ref double ry = ref MemoryMarshal.GetReference(by);
            ref double rw = ref MemoryMarshal.GetReference(bw);
            ref double ru = ref MemoryMarshal.GetReference(u);
            ref double rv = ref MemoryMarshal.GetReference(v);
            int last = n - Vector<double>.Count;
            while (true)
            {
                if (j > last)
                {
                    if (j == n)
                    {
                        break;
                    }

                    j = last;
                }

                var w = one / Vector.LoadUnsafe(ref rw, (nuint)j);
                var x = Vector.LoadUnsafe(ref rx, (nuint)j) * w;
                var y = Vector.LoadUnsafe(ref ry, (nuint)j) * w;
                var xd = x * x;
                var yd = y * y;
                var r2 = xd + yd;
                var num = (((k3 * r2) + k2) * r2 + k1) * r2 + one;
                var kr = num / ((((k6 * r2) + k5) * r2 + k4) * r2 + one);
                xd = two * xd + r2;
                yd = two * yd + r2;
                var m2 = (x * y) * two;
                x *= kr;
                y *= kr;
                xd = xd * p2 + x;
                yd = yd * p1 + y;
                xd = p1 * m2 + xd;
                yd = p2 * m2 + yd;
                var r4 = r2 * r2;
                xd = s1 * r2 + (s2 * r4 + xd);
                yd = s3 * r2 + (s4 * r4 + yd);
                var tx = t0 * xd + (t1 * yd + t2);
                var ty = t3 * xd + (t4 * yd + t5);
                var tz = t6 * xd + (t7 * yd + t8);
                var iz = Vector.ConditionalSelect(Vector.Equals(tz, Vector<double>.Zero), one, one / tz);
                ((fx * iz) * tx + u0).StoreUnsafe(ref ru, (nuint)j);
                ((fy * iz) * ty + v0).StoreUnsafe(ref rv, (nuint)j);
                j += Vector<double>.Count;
            }

            return;
        }

        for (; j < n; j++)
        {
            BlockPixel(p, bx[j], by[j], bw[j], out u[j], out v[j]);
        }
    }

    /// <summary>One lane of the <c>v_float64</c> body.</summary>
    internal static void BlockPixel(UndistortMapParams p, double bx, double by, double bw, out double u, out double v)
    {
        double[] k = p.K;
        double[] t = p.Tilt;
        double w = 1.0 / bw;
        double x = bx * w, y = by * w;
        double xd = x * x, yd = y * y;
        double r2 = xd + yd;
        double num = ((k[4] * r2 + k[1]) * r2 + k[0]) * r2 + 1.0;
        double kr = num / (((k[7] * r2 + k[6]) * r2 + k[5]) * r2 + 1.0);
        xd = 2.0 * xd + r2;
        yd = 2.0 * yd + r2;
        double m2 = x * y * 2.0;
        x *= kr;
        y *= kr;
        xd = xd * k[3] + x;
        yd = yd * k[2] + y;
        xd = k[2] * m2 + xd;
        yd = k[3] * m2 + yd;
        double r4 = r2 * r2;
        xd = k[8] * r2 + (k[9] * r4 + xd);
        yd = k[10] * r2 + (k[11] * r4 + yd);
        double tx = t[0] * xd + (t[1] * yd + t[2]);
        double ty = t[3] * xd + (t[4] * yd + t[5]);
        double tz = t[6] * xd + (t[7] * yd + t[8]);
        double iz = tz == 0 ? 1.0 : 1.0 / tz;
        u = p.Fx * iz * tx + p.U0;
        v = p.Fy * iz * ty + p.V0;
    }

    /// <summary>The scalar tail of <c>initUndistortRectifyMapComputer</c>.</summary>
    private static void TailPixel(UndistortMapParams p, double x0, double y0, double w0, out double u, out double v)
    {
        double[] k = p.K;
        double[] t = p.Tilt;
        double w = 1.0 / w0, x = x0 * w, y = y0 * w;
        double x2 = x * x, y2 = y * y;
        double r2 = x2 + y2, xy2 = 2 * x * y;
        double kr = (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2) / (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2);
        double xd = x * kr + k[2] * xy2 + k[3] * (r2 + 2 * x2) + k[8] * r2 + k[9] * r2 * r2;
        double yd = y * kr + k[2] * (r2 + 2 * y2) + k[3] * xy2 + k[10] * r2 + k[11] * r2 * r2;
        double tx = CameraDistortion.MulRow(t, 0, xd, yd);
        double ty = CameraDistortion.MulRow(t, 1, xd, yd);
        double tz = CameraDistortion.MulRow(t, 2, xd, yd);
        double invProj = tz != 0 ? 1.0 / tz : 1;
        u = p.Fx * invProj * tx + p.U0;
        v = p.Fy * invProj * ty + p.V0;
    }

    private static void StoreFloat(ReadOnlySpan<double> src, Span<float> dst)
    {
        int n = src.Length;
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int step = Vector<float>.Count;
            ref double rs = ref MemoryMarshal.GetReference(src);
            ref float rd = ref MemoryMarshal.GetReference(dst);
            for (; j <= n - step; j += step)
            {
                var lo = Vector.LoadUnsafe(ref rs, (nuint)j);
                var hi = Vector.LoadUnsafe(ref rs, (nuint)(j + Vector<double>.Count));
                Vector.Narrow(lo, hi).StoreUnsafe(ref rd, (nuint)j);
            }
        }

        for (; j < n; j++)
        {
            dst[j] = (float)src[j];
        }
    }

    private static void StoreInterleaved(ReadOnlySpan<double> u, ReadOnlySpan<double> v, Span<float> dst)
    {
        for (int j = 0; j < u.Length; j++)
        {
            dst[2 * j] = (float)u[j];
            dst[2 * j + 1] = (float)v[j];
        }
    }

    private static void StoreFixed(ReadOnlySpan<double> u, ReadOnlySpan<double> v, int simdEnd, Span<short> xy, Span<ushort> fractions)
    {
        int j = 0;
        if (Avx.IsSupported && simdEnd >= Block)
        {
            ref double ru = ref MemoryMarshal.GetReference(u);
            ref double rv = ref MemoryMarshal.GetReference(v);
            ref short rxy = ref MemoryMarshal.GetReference(xy);
            ref ushort rf = ref MemoryMarshal.GetReference(fractions);
            var scale = Vector256.Create((double)InterTabSize);
            var mask = Vector128.Create(InterTabSize - 1);
            for (; j <= simdEnd - Block; j += Block)
            {
                // cvtpd2dq rounds half to even and returns int.MinValue out of range, as CvRound.
                Vector128<int> iu = Avx.ConvertToVector128Int32(Vector256.LoadUnsafe(ref ru, (nuint)j) * scale);
                Vector128<int> iv = Avx.ConvertToVector128Int32(Vector256.LoadUnsafe(ref rv, (nuint)j) * scale);
                Vector128<int> f = (iu & mask) + ((iv & mask) << InterBits);
                Unsafe.WriteUnaligned(
                    ref Unsafe.As<ushort, byte>(ref Unsafe.Add(ref rf, j)),
                    Sse41.PackUnsignedSaturate(f, f).AsUInt64().ToScalar());
                Vector128<int> su = Sse2.ShiftRightArithmetic(iu, InterBits);
                Vector128<int> sv = Sse2.ShiftRightArithmetic(iv, InterBits);
                Sse2.PackSignedSaturate(Sse2.UnpackLow(su, sv), Sse2.UnpackHigh(su, sv))
                    .StoreUnsafe(ref rxy, (nuint)(2 * j));
            }
        }

        for (; j < simdEnd; j++)
        {
            int iu = ScalarConverter.CvRound(u[j] * InterTabSize);
            int iv = ScalarConverter.CvRound(v[j] * InterTabSize);
            fractions[j] = (ushort)((iu & (InterTabSize - 1)) + ((iv & (InterTabSize - 1)) * InterTabSize));
            xy[2 * j] = SaturateShort(iu >> InterBits);
            xy[2 * j + 1] = SaturateShort(iv >> InterBits);
        }

        for (; j < u.Length; j++)
        {
            int iu = ScalarConverter.CvRound(u[j] * InterTabSize);
            int iv = ScalarConverter.CvRound(v[j] * InterTabSize);
            xy[2 * j] = unchecked((short)(iu >> InterBits));
            xy[2 * j + 1] = unchecked((short)(iv >> InterBits));
            fractions[j] = (ushort)(((iv & (InterTabSize - 1)) * InterTabSize) + (iu & (InterTabSize - 1)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static short SaturateShort(int value) => (short)Math.Clamp(value, short.MinValue, short.MaxValue);
}
