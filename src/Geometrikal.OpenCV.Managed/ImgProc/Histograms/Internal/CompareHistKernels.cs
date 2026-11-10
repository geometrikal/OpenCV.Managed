// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (compareHist)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// <c>compareHist</c> with the OpenCV 5.0 accumulation order of a 128-bit build: blocks of four
/// floats, each block widened to two double pairs (elements 0-1, then 2-3) and added into two
/// double lanes; then <c>lane0 + lane1</c>; then a scalar tail. <see cref="Vector128{T}"/> has a
/// fixed width, so the result does not depend on the hardware vector width.
/// </summary>
internal sealed class CompareHistKernels
{
    private const double DblEpsilon = 2.220446049250313e-16;
    private const double FltEpsilon = 1.1920928955078125e-07;

    private readonly HistCompMethods _method;
    private double _result;
    private double _s1;
    private double _s2;
    private double _s11;
    private double _s12;
    private double _s22;

    public CompareHistKernels(HistCompMethods method)
    {
        if (method is < HistCompMethods.Correl or > HistCompMethods.KLDiv)
        {
            throw new ArgumentOutOfRangeException(nameof(method));
        }

        _method = method;
    }

    /// <summary>Processes one continuous plane, as one iteration of the OpenCV plane loop.</summary>
    public void Plane(ReadOnlySpan<float> h1, ReadOnlySpan<float> h2)
    {
        ref float a = ref MemoryMarshal.GetReference(h1);
        ref float b = ref MemoryMarshal.GetReference(h2);
        int len = h1.Length;
        switch (_method)
        {
            case HistCompMethods.Chisqr:
            case HistCompMethods.ChisqrAlt:
                ChiSquare(ref a, ref b, len, _method == HistCompMethods.ChisqrAlt);
                break;
            case HistCompMethods.Correl:
                Correl(ref a, ref b, len);
                break;
            case HistCompMethods.Intersect:
                Intersect(ref a, ref b, len);
                break;
            case HistCompMethods.Bhattacharyya:
                Bhattacharyya(ref a, ref b, len);
                break;
            default:
                KLDiv(ref a, ref b, len);
                break;
        }
    }

    /// <summary>Final value; <paramref name="total"/> is OpenCV <c>H1.total()</c>.</summary>
    public double Finish(long total)
    {
        switch (_method)
        {
            case HistCompMethods.ChisqrAlt:
                return _result * 2;
            case HistCompMethods.Correl:
            {
                double scale = 1.0 / total;
                double num = _s12 - (_s1 * _s2 * scale);
                double denom2 = (_s11 - (_s1 * _s1 * scale)) * (_s22 - (_s2 * _s2 * scale));
                return Math.Abs(denom2) > DblEpsilon ? num / Math.Sqrt(denom2) : 1.0;
            }

            case HistCompMethods.Bhattacharyya:
            {
                double s1 = _s1 * _s2;
                s1 = Math.Abs(s1) > FltEpsilon ? 1.0 / Math.Sqrt(s1) : 1.0;
                return Math.Sqrt(Math.Max(1.0 - (_result * s1), 0.0));
            }

            default:
                return _result;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Load4(ref float p, int j) => Vector128.LoadUnsafe(ref p, (nuint)j);

    private static double ReduceSum(Vector128<double> v) => v.GetElement(0) + v.GetElement(1);

    private void ChiSquare(ref float h1, ref float h2, int len, bool alt)
    {
        int j = 0;
        var eps = Vector128.Create(DblEpsilon);
        var one = Vector128.Create(1.0);
        Vector128<double> res = Vector128<double>.Zero;
        for (; j <= len - 4; j += 4)
        {
            Vector128<float> a = Load4(ref h1, j);
            Vector128<float> b = Load4(ref h2, j);
            Vector128<double> al = Vector128.WidenLower(a);
            Vector128<double> ah = Vector128.WidenUpper(a);
            Vector128<double> bl = Vector128.WidenLower(b);
            Vector128<double> bh = Vector128.WidenUpper(b);
            Vector128<double> dl = al - bl;
            Vector128<double> dh = ah - bh;
            Vector128<double> ql = alt ? al + bl : al;
            Vector128<double> qh = alt ? ah + bh : ah;
            Vector128<double> rl = (dl * dl) * (one / ql);
            res += Vector128.ConditionalSelect(Vector128.GreaterThan(Vector128.Abs(ql), eps), rl, Vector128<double>.Zero);
            Vector128<double> rh = (dh * dh) * (one / qh);
            res += Vector128.ConditionalSelect(Vector128.GreaterThan(Vector128.Abs(qh), eps), rh, Vector128<double>.Zero);
        }

        _result += ReduceSum(res);
        for (; j < len; j++)
        {
            float x = Unsafe.Add(ref h1, j);
            float y = Unsafe.Add(ref h2, j);
            double a = x - y;
            double b = alt ? x + y : x;
            if (Math.Abs(b) > DblEpsilon)
            {
                _result += a * a / b;
            }
        }
    }

    private void Correl(ref float h1, ref float h2, int len)
    {
        int j = 0;
        Vector128<double> s1 = Vector128<double>.Zero;
        Vector128<double> s2 = Vector128<double>.Zero;
        Vector128<double> s11 = Vector128<double>.Zero;
        Vector128<double> s12 = Vector128<double>.Zero;
        Vector128<double> s22 = Vector128<double>.Zero;
        for (; j <= len - 4; j += 4)
        {
            Vector128<float> a = Load4(ref h1, j);
            Vector128<float> b = Load4(ref h2, j);
            Vector128<double> ad = Vector128.WidenLower(a);
            Vector128<double> bd = Vector128.WidenLower(b);
            s12 = (ad * bd) + s12;
            s11 = (ad * ad) + s11;
            s22 = (bd * bd) + s22;
            s1 += ad;
            s2 += bd;
            ad = Vector128.WidenUpper(a);
            bd = Vector128.WidenUpper(b);
            s12 = (ad * bd) + s12;
            s11 = (ad * ad) + s11;
            s22 = (bd * bd) + s22;
            s1 += ad;
            s2 += bd;
        }

        _s12 += ReduceSum(s12);
        _s11 += ReduceSum(s11);
        _s22 += ReduceSum(s22);
        _s1 += ReduceSum(s1);
        _s2 += ReduceSum(s2);
        for (; j < len; j++)
        {
            double a = Unsafe.Add(ref h1, j);
            double b = Unsafe.Add(ref h2, j);
            _s12 += a * b;
            _s1 += a;
            _s11 += a * a;
            _s2 += b;
            _s22 += b * b;
        }
    }

    private void Intersect(ref float h1, ref float h2, int len)
    {
        int j = 0;
        Vector128<double> res = Vector128<double>.Zero;
        for (; j <= len - 4; j += 4)
        {
            Vector128<float> a = Load4(ref h1, j);
            Vector128<float> b = Load4(ref h2, j);

            // SSE minps: a < b ? a : b.
            Vector128<float> m = Vector128.ConditionalSelect(Vector128.LessThan(a, b), a, b);
            res += Vector128.WidenLower(m) + Vector128.WidenUpper(m);
        }

        _result += ReduceSum(res);
        for (; j < len; j++)
        {
            float x = Unsafe.Add(ref h1, j);
            float y = Unsafe.Add(ref h2, j);

            // std::min(x, y): y < x ? y : x.
            _result += y < x ? y : x;
        }
    }

    private void Bhattacharyya(ref float h1, ref float h2, int len)
    {
        int j = 0;
        Vector128<double> s1 = Vector128<double>.Zero;
        Vector128<double> s2 = Vector128<double>.Zero;
        Vector128<double> res = Vector128<double>.Zero;
        for (; j <= len - 4; j += 4)
        {
            Vector128<float> a = Load4(ref h1, j);
            Vector128<float> b = Load4(ref h2, j);
            Vector128<double> ad = Vector128.WidenLower(a);
            Vector128<double> bd = Vector128.WidenLower(b);
            s1 += ad;
            s2 += bd;
            res += Vector128.Sqrt(ad * bd);
            ad = Vector128.WidenUpper(a);
            bd = Vector128.WidenUpper(b);
            s1 += ad;
            s2 += bd;
            res += Vector128.Sqrt(ad * bd);
        }

        _s1 += ReduceSum(s1);
        _s2 += ReduceSum(s2);
        _result += ReduceSum(res);
        for (; j < len; j++)
        {
            double a = Unsafe.Add(ref h1, j);
            double b = Unsafe.Add(ref h2, j);
            _result += Math.Sqrt(a * b);
            _s1 += a;
            _s2 += b;
        }
    }

    private void KLDiv(ref float h1, ref float h2, int len)
    {
        for (int j = 0; j < len; j++)
        {
            double p = Unsafe.Add(ref h1, j);
            double q = Unsafe.Add(ref h2, j);
            if (Math.Abs(p) <= DblEpsilon)
            {
                continue;
            }

            if (Math.Abs(q) <= DblEpsilon)
            {
                q = 1e-10;
            }

            _result += p * Math.Log(p / q);
        }
    }
}
