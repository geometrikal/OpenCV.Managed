// This implementation is ported from the OpenCV core math functions.
//
// Original sources:
//   modules/core/src/mathfuncs_core.simd.hpp (atan_f32, v_atan_f32, fastAtan32f, magnitude32f, exp32f)
//   modules/core/src/mathfuncs_core.dispatch.cpp (cv::fastAtan2, hal::exp32f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.InteropServices;

/// <summary>
/// Single-precision OpenCV math helpers with the OpenCV operation order:
/// <c>cv::fastAtan2</c>, <c>hal::fastAtan32f</c>, <c>hal::magnitude32f</c> and
/// <c>hal::exp32f</c>.
/// </summary>
/// <remarks>
/// <para>The OpenCV vector and scalar forms of <c>atan_f32</c> and <c>magnitude32f</c> give
/// identical results without FMA, so every lane uses the scalar operation order.</para>
/// <para><c>hal::exp32f</c> in the Windows (MSVC) build of OpenCV is <c>std::exp(float)</c>
/// per element, which is the C runtime <c>expf</c>. <see cref="MathF.Exp"/> calls the same
/// C runtime function on Windows. The table-and-polynomial form of other compilers is not
/// ported.</para>
/// </remarks>
internal static class FastMath
{
    private const float DblEpsilonF = (float)2.220446049250313e-16;

    // Products rounded to float at run time, as the OpenCV static initializers.
    private static readonly float P1 = Product(0.9997878412794807f);
    private static readonly float P3 = Product(-0.3258083974640975f);
    private static readonly float P5 = Product(0.1555786518463281f);
    private static readonly float P7 = Product(-0.04432655554792128f);

    /// <summary>OpenCV <c>cv::fastAtan2(y, x)</c>: angle in degrees in <c>[0, 360)</c>.</summary>
    public static float FastAtan2(float y, float x)
    {
        float ax = MathF.Abs(x);
        float ay = MathF.Abs(y);
        float a;
        if (ax >= ay)
        {
            float c = ay / (ax + DblEpsilonF);
            float c2 = c * c;
            a = ((((P7 * c2) + P5) * c2 + P3) * c2 + P1) * c;
        }
        else
        {
            float c = ax / (ay + DblEpsilonF);
            float c2 = c * c;
            a = 90f - (((((P7 * c2) + P5) * c2 + P3) * c2 + P1) * c);
        }

        if (x < 0)
        {
            a = 180f - a;
        }

        if (y < 0)
        {
            a = 360f - a;
        }

        return a;
    }

    /// <summary>OpenCV <c>hal::fastAtan32f(Y, X, angle, len, true)</c>: angles in degrees.</summary>
    public static void FastAtan32fDegrees(ReadOnlySpan<float> y, ReadOnlySpan<float> x, Span<float> angle)
    {
        int len = angle.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated && len >= Vector<float>.Count)
        {
            ReadOnlySpan<Vector<float>> vy = MemoryMarshal.Cast<float, Vector<float>>(y[..len]);
            ReadOnlySpan<Vector<float>> vx = MemoryMarshal.Cast<float, Vector<float>>(x[..len]);
            Span<Vector<float>> va = MemoryMarshal.Cast<float, Vector<float>>(angle);
            var p7 = new Vector<float>(P7);
            var p5 = new Vector<float>(P5);
            var p3 = new Vector<float>(P3);
            var p1 = new Vector<float>(P1);
            var eps = new Vector<float>(DblEpsilonF);
            var v90 = new Vector<float>(90f);
            var v180 = new Vector<float>(180f);
            var v360 = new Vector<float>(360f);
            for (int k = 0; k < va.Length; k++)
            {
                Vector<float> xv = vx[k];
                Vector<float> yv = vy[k];
                Vector<float> ax = Vector.Abs(xv);
                Vector<float> ay = Vector.Abs(yv);
                Vector<int> ge = Vector.GreaterThanOrEqual(ax, ay);
                Vector<float> num = Vector.ConditionalSelect(ge, ay, ax);
                Vector<float> den = Vector.ConditionalSelect(ge, ax, ay);
                Vector<float> c = num / (den + eps);
                Vector<float> cc = c * c;
                Vector<float> a = ((((p7 * cc) + p5) * cc + p3) * cc + p1) * c;
                a = Vector.ConditionalSelect(ge, a, v90 - a);
                a = Vector.ConditionalSelect(Vector.LessThan(xv, Vector<float>.Zero), v180 - a, a);
                a = Vector.ConditionalSelect(Vector.LessThan(yv, Vector<float>.Zero), v360 - a, a);
                va[k] = a;
            }

            i = va.Length * Vector<float>.Count;
        }

        for (; i < len; i++)
        {
            angle[i] = FastAtan2(y[i], x[i]);
        }
    }

    /// <summary>OpenCV <c>hal::magnitude32f</c>: <c>sqrt(x·x + y·y)</c>.</summary>
    public static void Magnitude32f(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> mag)
    {
        int len = mag.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated && len >= Vector<float>.Count)
        {
            ReadOnlySpan<Vector<float>> vx = MemoryMarshal.Cast<float, Vector<float>>(x[..len]);
            ReadOnlySpan<Vector<float>> vy = MemoryMarshal.Cast<float, Vector<float>>(y[..len]);
            Span<Vector<float>> vm = MemoryMarshal.Cast<float, Vector<float>>(mag);
            for (int k = 0; k < vm.Length; k++)
            {
                Vector<float> a = vx[k];
                Vector<float> b = vy[k];
                vm[k] = Vector.SquareRoot((a * a) + (b * b));
            }

            i = vm.Length * Vector<float>.Count;
        }

        for (; i < len; i++)
        {
            float a = x[i];
            float b = y[i];
            mag[i] = MathF.Sqrt((a * a) + (b * b));
        }
    }

    private static float Product(float coefficient)
    {
        float degrees = (float)(180 / Math.PI);
        return (float)(coefficient * degrees);
    }

    /// <summary>OpenCV <c>hal::exp32f</c> of the MSVC build: <c>expf</c> per element.</summary>
    public static void Exp32f(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] = MathF.Exp(src[i]);
        }
    }
}
