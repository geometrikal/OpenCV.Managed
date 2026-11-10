// This implementation is ported from the OpenCV 2-D affine estimator callbacks.
//
// Original source:
//   modules/geometry/src/ptsetreg.cpp (Affine2DEstimatorCallback,
//   AffinePartial2DEstimatorCallback, Translation2DEstimatorCallback)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Error loops of the 2-D kernels. The vector loops process four interleaved points per
/// <see cref="Vector256{T}"/>: even lanes hold x, odd lanes hold y. Each lane keeps the float
/// operation order of the scalar C++ code.
/// </summary>
internal static class ErrorLanes
{
    /// <summary>Swaps the x and y lanes of every point.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> SwapPairs(Vector256<float> v) => Avx.Permute(v, 0b10_11_00_01);

    /// <summary>Sets the mask bytes of four points from the even-lane compare bits.</summary>
    /// <returns>The number of inliers among the four points.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int StoreMask(Vector256<float> err, Vector256<float> t, Span<byte> mask, int i)
    {
        int bits = Avx.MoveMask(Avx.CompareLessThanOrEqual(err, t)) & 0x55;
        mask[i] = (byte)(bits & 1);
        mask[i + 1] = (byte)((bits >> 2) & 1);
        mask[i + 2] = (byte)((bits >> 4) & 1);
        mask[i + 3] = (byte)((bits >> 6) & 1);
        return BitOperations.PopCount((uint)bits);
    }

    /// <summary>Reads the point span as interleaved floats.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ref float Floats(ReadOnlySpan<Point2f> points)
        => ref Unsafe.As<Point2f, float>(ref MemoryMarshal.GetReference(points));
}

/// <summary>OpenCV <c>Affine2DEstimatorCallback</c>: a full 2 × 3 affine model from 3 pairs.</summary>
internal readonly struct Affine2DKernel : IEstimatorKernel
{
    public int ModelPoints => 3;

    public int ModelSize => 6;

    public int RunKernel(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, Span<double> m)
    {
        double x1 = from[0].X, y1 = from[0].Y;
        double x2 = from[1].X, y2 = from[1].Y;
        double x3 = from[2].X, y3 = from[2].Y;

        double X1 = to[0].X, Y1 = to[0].Y;
        double X2 = to[1].X, Y2 = to[1].Y;
        double X3 = to[2].X, Y3 = to[2].Y;

        double d = 1.0 / ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)));

        m[0] = d * ((X1 * (y2 - y3)) + (X2 * (y3 - y1)) + (X3 * (y1 - y2)));
        m[1] = d * ((X1 * (x3 - x2)) + (X2 * (x1 - x3)) + (X3 * (x2 - x1)));
        m[2] = d * ((X1 * ((x2 * y3) - (x3 * y2))) + (X2 * ((x3 * y1) - (x1 * y3))) + (X3 * ((x1 * y2) - (x2 * y1))));

        m[3] = d * ((Y1 * (y2 - y3)) + (Y2 * (y3 - y1)) + (Y3 * (y1 - y2)));
        m[4] = d * ((Y1 * (x3 - x2)) + (Y2 * (x1 - x3)) + (Y3 * (x2 - x1)));
        m[5] = d * ((Y1 * ((x2 * y3) - (x3 * y2))) + (Y2 * ((x3 * y1) - (x1 * y3))) + (Y3 * ((x1 * y2) - (x2 * y1))));
        return 1;
    }

    public void ComputeError(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> model, Span<float> err)
        => AffineErrors.Compute(from, to, model, err);

    public int FindInliers(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> model, float t, Span<byte> mask)
        => AffineErrors.Inliers(from, to, model, t, mask);

    public bool CheckSubset(ReadOnlySpan<Point2f> ms1, ReadOnlySpan<Point2f> ms2, int count)
        => !PointSetRegistrator.HaveCollinearPoints(ms1, count) && !PointSetRegistrator.HaveCollinearPoints(ms2, count);
}

/// <summary>
/// OpenCV <c>AffinePartial2DEstimatorCallback</c>: rotation, uniform scale and translation from 2
/// pairs. The error and the subset check are those of <see cref="Affine2DKernel"/>.
/// </summary>
internal readonly struct AffinePartial2DKernel : IEstimatorKernel
{
    public int ModelPoints => 2;

    public int ModelSize => 6;

    public int RunKernel(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, Span<double> m)
    {
        double x1 = from[0].X, y1 = from[0].Y;
        double x2 = from[1].X, y2 = from[1].Y;

        double X1 = to[0].X, Y1 = to[0].Y;
        double X2 = to[1].X, Y2 = to[1].Y;

        double d = 1.0 / (((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2)));

        double s0 = d * (((X1 - X2) * (x1 - x2)) + ((Y1 - Y2) * (y1 - y2)));
        double s1 = d * (((Y1 - Y2) * (x1 - x2)) - ((X1 - X2) * (y1 - y2)));
        double s2 = d * (((Y1 - Y2) * ((x1 * y2) - (x2 * y1))) - (((X1 * y2) - (X2 * y1)) * (y1 - y2)) - (((X1 * x2) - (X2 * x1)) * (x1 - x2)));
        double s3 = d * ((-(X1 - X2) * ((x1 * y2) - (x2 * y1))) - (((Y1 * x2) - (Y2 * x1)) * (x1 - x2)) - (((Y1 * y2) - (Y2 * y1)) * (y1 - y2)));

        m[0] = s0;
        m[4] = s0;
        m[1] = -s1;
        m[2] = s2;
        m[3] = s1;
        m[5] = s3;
        return 1;
    }

    public void ComputeError(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> model, Span<float> err)
        => AffineErrors.Compute(from, to, model, err);

    public int FindInliers(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> model, float t, Span<byte> mask)
        => AffineErrors.Inliers(from, to, model, t, mask);

    public bool CheckSubset(ReadOnlySpan<Point2f> ms1, ReadOnlySpan<Point2f> ms2, int count)
        => !PointSetRegistrator.HaveCollinearPoints(ms1, count) && !PointSetRegistrator.HaveCollinearPoints(ms2, count);
}

/// <summary>OpenCV <c>Translation2DEstimatorCallback</c>: a translation from 1 pair.</summary>
internal readonly struct Translation2DKernel : IEstimatorKernel
{
    public int ModelPoints => 1;

    public int ModelSize => 6;

    public int RunKernel(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, Span<double> m)
    {
        if (from.Length < 1)
        {
            return 0;
        }

        m[0] = 1.0;
        m[1] = 0.0;
        m[2] = (double)to[0].X - from[0].X;
        m[3] = 0.0;
        m[4] = 1.0;
        m[5] = (double)to[0].Y - from[0].Y;
        return 1;
    }

    public void ComputeError(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> model, Span<float> err)
    {
        float tx = (float)model[2], ty = (float)model[5];
        for (int i = 0; i < from.Length; i++)
        {
            float rx = from[i].X + tx - to[i].X;
            float ry = from[i].Y + ty - to[i].Y;
            err[i] = (rx * rx) + (ry * ry);
        }
    }

    public int FindInliers(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> model, float t, Span<byte> mask)
    {
        float tx = (float)model[2], ty = (float)model[5];
        int n = from.Length, i = 0, nz = 0;
        if (Avx.IsSupported && n >= 4)
        {
            ref float p1 = ref ErrorLanes.Floats(from);
            ref float p2 = ref ErrorLanes.Floats(to);
            var tv = Vector256.Create(tx, ty, tx, ty, tx, ty, tx, ty);
            var tt = Vector256.Create(t);
            for (; i <= n - 4; i += 4)
            {
                Vector256<float> v = Vector256.LoadUnsafe(ref p1, (nuint)(2 * i));
                Vector256<float> w = Vector256.LoadUnsafe(ref p2, (nuint)(2 * i));
                Vector256<float> r = (v + tv) - w;
                Vector256<float> sq = r * r;
                nz += ErrorLanes.StoreMask(sq + ErrorLanes.SwapPairs(sq), tt, mask, i);
            }
        }

        for (; i < n; i++)
        {
            float rx = from[i].X + tx - to[i].X;
            float ry = from[i].Y + ty - to[i].Y;
            int f = (rx * rx) + (ry * ry) <= t ? 1 : 0;
            mask[i] = (byte)f;
            nz += f;
        }

        return nz;
    }

    public bool CheckSubset(ReadOnlySpan<Point2f> ms1, ReadOnlySpan<Point2f> ms2, int count) => true;
}

/// <summary>The error of the 2 × 3 affine models in <c>float</c>, as OpenCV.</summary>
internal static class AffineErrors
{
    public static void Compute(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> f, Span<float> err)
    {
        float f0 = (float)f[0], f1 = (float)f[1], f2 = (float)f[2];
        float f3 = (float)f[3], f4 = (float)f[4], f5 = (float)f[5];
        for (int i = 0; i < from.Length; i++)
        {
            float a = (f0 * from[i].X) + (f1 * from[i].Y) + f2 - to[i].X;
            float b = (f3 * from[i].X) + (f4 * from[i].Y) + f5 - to[i].Y;
            err[i] = (a * a) + (b * b);
        }
    }

    public static int Inliers(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<double> f, float t, Span<byte> mask)
    {
        float f0 = (float)f[0], f1 = (float)f[1], f2 = (float)f[2];
        float f3 = (float)f[3], f4 = (float)f[4], f5 = (float)f[5];
        int n = from.Length, i = 0, nz = 0;
        if (Avx.IsSupported && n >= 4)
        {
            ref float p1 = ref ErrorLanes.Floats(from);
            ref float p2 = ref ErrorLanes.Floats(to);
            var ca = Vector256.Create(f0, f4, f0, f4, f0, f4, f0, f4);
            var cb = Vector256.Create(f1, f3, f1, f3, f1, f3, f1, f3);
            var cc = Vector256.Create(f2, f5, f2, f5, f2, f5, f2, f5);
            var tt = Vector256.Create(t);
            for (; i <= n - 4; i += 4)
            {
                Vector256<float> v = Vector256.LoadUnsafe(ref p1, (nuint)(2 * i));
                Vector256<float> w = Vector256.LoadUnsafe(ref p2, (nuint)(2 * i));

                // Lane y computes f4·y + f3·x; float addition is commutative, so it equals f3·x + f4·y.
                Vector256<float> r = (((v * ca) + (ErrorLanes.SwapPairs(v) * cb)) + cc) - w;
                Vector256<float> sq = r * r;
                nz += ErrorLanes.StoreMask(sq + ErrorLanes.SwapPairs(sq), tt, mask, i);
            }
        }

        for (; i < n; i++)
        {
            float a = (f0 * from[i].X) + (f1 * from[i].Y) + f2 - to[i].X;
            float b = (f3 * from[i].X) + (f4 * from[i].Y) + f5 - to[i].Y;
            int ok = (a * a) + (b * b) <= t ? 1 : 0;
            mask[i] = (byte)ok;
            nz += ok;
        }

        return nz;
    }
}
