// This implementation is ported from the OpenCV RHO homography estimator.
//
// Original sources:
//   modules/geometry/src/rho.cpp (RHO_HEST_REFC and its helper functions)
//   modules/geometry/src/rho.h (flags)
//   modules/geometry/src/fundam.cpp (createAndRunRHORegistrator)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// OpenCV <c>RHO_HEST_REFC</c>: PROSAC sampling, SPRT model evaluation, the non-randomness
/// stopping criterion and a final float Levenberg-Marquardt refinement. All arithmetic is in
/// <c>float</c> except the SPRT and stopping bounds, as in the C++ source.
/// </summary>
internal sealed class RhoEstimator
{
    private const double MinDeltaChange = 0.1;
    private const double ChiSq = 1.645;
    private const int MaxLevMarqIters = 100;
    private const int SampleSize = 4;
    private const double SprtTM = 25;
    private const double SprtMS = 1;
    private const double SprtEpsilon = 0.1;
    private const double SprtDelta = 0.01;
    private const double LmGainLo = 0.25;
    private const double LmGainHi = 0.75;
    private const float FltEpsilon = 1.1920928955078125e-07f;
    private const double Beta = 0.35;

    private readonly float[] _src;
    private readonly float[] _dst;
    private readonly int _n;
    private readonly float _maxD;
    private readonly double _cfd;
    private readonly uint[] _nrTable;
    private readonly uint[] _sample = new uint[SampleSize];
    private readonly float[] _packed = new float[16];
    private readonly float[] _jtj = new float[64];
    private readonly float[] _tmp1 = new float[64];
    private readonly float[] _jte = new float[8];

    private uint _maxI;
    private ulong _s0;
    private ulong _s1;

    private uint _i;
    private uint _phNum;
    private uint _phEndI;
    private double _phEndFpI;
    private uint _phMax;
    private uint _phNumInl;

    private float[] _currH = new float[9];
    private byte[] _currInl;
    private uint _currNumInl;
    private float[] _bestH = new float[9];
    private byte[] _bestInl;
    private uint _bestNumInl;

    private double _epsilon;
    private double _delta;
    private double _a;
    private uint _nTested;
    private bool _good;
    private double _lambdaAccept;
    private double _lambdaReject;

    private RhoEstimator(ReadOnlySpan<Point2f> src, ReadOnlySpan<Point2f> dst, float maxD, uint maxI, double cfd)
    {
        _n = src.Length;
        _src = ArrayPool<float>.Shared.Rent(2 * _n);
        _dst = ArrayPool<float>.Shared.Rent(2 * _n);
        MemoryMarshal.Cast<Point2f, float>(src).CopyTo(_src);
        MemoryMarshal.Cast<Point2f, float>(dst).CopyTo(_dst);
        _maxD = maxD;
        _maxI = maxI;
        _cfd = cfd;
        _currInl = ArrayPool<byte>.Shared.Rent(_n);
        _bestInl = ArrayPool<byte>.Shared.Rent(_n);
        _nrTable = ArrayPool<uint>.Shared.Rent(_n);
        Array.Clear(_currInl, 0, _n);
        Array.Clear(_bestInl, 0, _n);
        Array.Clear(_nrTable, 0, _n);
        InitNonRand(_nrTable, _n);
        FastSeed(ulong.MaxValue);
    }

    /// <summary>
    /// OpenCV <c>createAndRunRHORegistrator</c> with beta 0.35 and the flags
    /// <c>RHO_FLAG_ENABLE_NR | RHO_FLAG_ENABLE_FINAL_REFINEMENT</c>.
    /// </summary>
    /// <param name="src">Source points.</param>
    /// <param name="dst">Destination points.</param>
    /// <param name="threshold">Maximum reprojection distance of an inlier.</param>
    /// <param name="maxIters">Maximum number of iterations; also the PROSAC convergence parameter.</param>
    /// <param name="confidence">Confidence level.</param>
    /// <param name="h">Receives the 3 × 3 model (float values widened to double).</param>
    /// <param name="mask">Receives the inlier mask (0 or 1).</param>
    /// <returns>True when the best model has at least 4 inliers.</returns>
    public static bool Run(ReadOnlySpan<Point2f> src, ReadOnlySpan<Point2f> dst, double threshold, int maxIters, double confidence, Span<double> h, Span<byte> mask)
    {
        var rho = new RhoEstimator(src, dst, (float)threshold, unchecked((uint)maxIters), confidence);
        Span<float> finalH = stackalloc float[9];
        uint numInl;
        try
        {
            numInl = rho.Estimate(unchecked((uint)maxIters), finalH, mask);
        }
        finally
        {
            rho.ReturnBuffers();
        }

        for (int i = 0; i < 9; i++)
        {
            h[i] = finalH[i];
        }

        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = mask[i] != 0 ? (byte)1 : (byte)0;
        }

        return numInl != 0;
    }

    private void ReturnBuffers()
    {
        ArrayPool<float>.Shared.Return(_src);
        ArrayPool<float>.Shared.Return(_dst);
        ArrayPool<byte>.Shared.Return(_currInl);
        ArrayPool<byte>.Shared.Return(_bestInl);
        ArrayPool<uint>.Shared.Return(_nrTable);
    }

    /// <summary><c>rhoHest</c> without an extrinsic guess.</summary>
    private uint Estimate(uint rConvg, Span<float> finalH, Span<byte> inl)
    {
        if (_n < SampleSize || _maxD < 0 || _cfd < 0 || _cfd > 1)
        {
            finalH.Clear();
            inl.Clear();
            return 0;
        }

        _i = 0;
        _phNum = SampleSize;
        _phEndI = 1;
        _phEndFpI = InitPEndFpI(rConvg, (uint)_n, SampleSize);
        _phMax = (uint)_n;
        _phNumInl = 0;
        _currNumInl = 0;
        _bestNumInl = 0;
        _good = true;
        _epsilon = SprtEpsilon;
        _delta = SprtDelta;
        DesignSprtTest();

        for (_i = 0; _i < _maxI || _i < 100; _i++)
        {
            if (Hypothesize())
            {
                Verify();
            }
        }

        if (_bestNumInl > SampleSize)
        {
            Refine();
        }

        if (_bestNumInl >= SampleSize)
        {
            _bestH.CopyTo(finalH);
            _bestInl.AsSpan(0, _n).CopyTo(inl);
            return _bestNumInl;
        }

        finalH.Clear();
        inl.Clear();
        return 0;
    }

    private double FastRandom()
    {
        ulong x = _s0;
        ulong y = _s1;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        _s0 = y;
        _s1 = x;
        ulong s = unchecked(x + y);
        return s * 5.421010862427522e-20;
    }

    private void FastSeed(ulong seed)
    {
        _s0 = seed;
        _s1 = ~seed;
        for (int i = 0; i < 20; i++)
        {
            FastRandom();
        }
    }

    private bool Hypothesize()
    {
        if (_i >= _phEndI && _phNum < _phMax)
        {
            _phNum++;
            double next = (_phEndFpI * _phNum) / (_phNum - SampleSize);
            _phEndI = unchecked(_phEndI + ToUInt(Math.Ceiling(next - _phEndFpI)));
            _phEndFpI = next;
        }

        if (_i > _phEndI)
        {
            RndSmpl(4, _phNum);
        }
        else
        {
            RndSmpl(3, _phNum - 1);
            _sample[3] = _phNum - 1;
        }

        if (IsSampleDegenerate())
        {
            return false;
        }

        HFuncRefC(_packed, _currH);
        float f = _currH[0] + _currH[1] + _currH[2] + _currH[3] + _currH[4] + _currH[5] + _currH[6] + _currH[7];
        return !float.IsNaN(f);
    }

    private void RndSmpl(uint sampleSize, uint dataSetSize)
    {
        if (sampleSize * 2 > dataSetSize)
        {
            uint i = 0, j = 0;
            for (i = 0; i < sampleSize; j++)
            {
                double u = FastRandom();
                if ((dataSetSize - j) * u < (sampleSize - i))
                {
                    _sample[i++] = j;
                }
            }
        }
        else
        {
            for (uint i = 0; i < sampleSize; i++)
            {
                bool inList;
                do
                {
                    _sample[i] = ToUInt(dataSetSize * FastRandom());
                    inList = false;
                    for (uint j = 0; j < i; j++)
                    {
                        if (_sample[i] == _sample[j])
                        {
                            inList = true;
                            break;
                        }
                    }
                }
                while (inList);
            }
        }
    }

    private bool IsSampleDegenerate()
    {
        float[] p = _packed;
        for (int k = 0; k < 4; k++)
        {
            int si = 2 * (int)_sample[k];
            p[2 * k] = _src[si];
            p[(2 * k) + 1] = _src[si + 1];
            p[8 + (2 * k)] = _dst[si];
            p[8 + (2 * k) + 1] = _dst[si + 1];
        }

        float x0 = p[0], y0 = p[1], x1 = p[2], y1 = p[3], x2 = p[4], y2 = p[5], x3 = p[6], y3 = p[7];
        float X0 = p[8], Y0 = p[9], X1 = p[10], Y1 = p[11], X2 = p[12], Y2 = p[13], X3 = p[14], Y3 = p[15];
        if (x0 == x1 || x1 == x2 || x2 == x3 || x0 == x2 || x1 == x3 || x0 == x3
            || y0 == y1 || y1 == y2 || y2 == y3 || y0 == y2 || y1 == y3 || y0 == y3)
        {
            return true;
        }

        float c0s0 = y0 - y1, c0s1 = x1 - x0, c0s2 = (x0 * y1) - (y0 * x1);
        float dots0 = (c0s0 * x2) + (c0s1 * y2) + c0s2;
        float c0d0 = Y0 - Y1, c0d1 = X1 - X0, c0d2 = (X0 * Y1) - (Y0 * X1);
        float dotd0 = (c0d0 * X2) + (c0d1 * Y2) + c0d2;
        if ((Trunc(dots0) ^ Trunc(dotd0)) < 0)
        {
            return true;
        }

        float dots1 = (c0s0 * x3) + (c0s1 * y3) + c0s2;
        float dotd1 = (c0d0 * X3) + (c0d1 * Y3) + c0d2;
        if ((Trunc(dots1) ^ Trunc(dotd1)) < 0)
        {
            return true;
        }

        float c2s0 = y2 - y3, c2s1 = x3 - x2, c2s2 = (x2 * y3) - (y2 * x3);
        float dots2 = (c2s0 * x0) + (c2s1 * y0) + c2s2;
        float c2d0 = Y2 - Y3, c2d1 = X3 - X2, c2d2 = (X2 * Y3) - (Y2 * X3);
        float dotd2 = (c2d0 * X0) + (c2d1 * Y0) + c2d2;
        if ((Trunc(dots2) ^ Trunc(dotd2)) < 0)
        {
            return true;
        }

        float dots3 = (c2s0 * x1) + (c2s1 * y1) + c2s2;
        float dotd3 = (c2d0 * X1) + (c2d1 * Y1) + c2d2;
        return (Trunc(dots3) ^ Trunc(dotd3)) < 0;
    }

    private void Verify()
    {
        EvaluateModelSprt();

        // updateSPRT
        if (_good)
        {
            if (_currNumInl > _bestNumInl)
            {
                _epsilon = (double)_currNumInl / _n;
                DesignSprtTest();
            }
        }
        else
        {
            double newDelta = (double)_currNumInl / _nTested;
            if (newDelta > 0)
            {
                double relChange = Math.Abs(_delta - newDelta) / _delta;
                if (relChange > MinDeltaChange)
                {
                    _delta = newDelta;
                    DesignSprtTest();
                }
            }
        }

        if (_currNumInl > _bestNumInl)
        {
            (_currH, _bestH) = (_bestH, _currH);
            (_currInl, _bestInl) = (_bestInl, _currInl);
            (_currNumInl, _bestNumInl) = (_bestNumInl, _currNumInl);

            _maxI = CalcIterBound(_cfd, (double)_bestNumInl / _n, SampleSize, _maxI);
            NStarOptimize();
        }
    }

    private void EvaluateModelSprt()
    {
        double lambda = 1.0;
        float distSq = _maxD * _maxD;
        float[] h = _currH;
        byte[] inl = _currInl;
        _currNumInl = 0;
        _nTested = 0;
        _good = true;
        int i;
        for (i = 0; i < _n && _good; i++)
        {
            float x = _src[2 * i], y = _src[(2 * i) + 1];
            float bx = _dst[2 * i], by = _dst[(2 * i) + 1];
            float rx = (h[0] * x) + (h[1] * y) + h[2];
            float ry = (h[3] * x) + (h[4] * y) + h[5];
            float rz = (h[6] * x) + (h[7] * y) + 1.0f;
            rx /= rz;
            ry /= rz;
            rx -= bx;
            ry -= by;
            rx *= rx;
            ry *= ry;
            float dist = rx + ry;
            bool isInlier = dist <= distSq;
            _currNumInl += isInlier ? 1u : 0u;
            inl[i] = isInlier ? (byte)1 : (byte)0;
            lambda *= isInlier ? _lambdaAccept : _lambdaReject;
            _good = lambda <= _a;
        }

        _nTested = (uint)i;
    }

    private void DesignSprtTest()
    {
        double c = ((1 - _delta) * Math.Log((1 - _delta) / (1 - _epsilon))) + (_delta * Math.Log(_delta / _epsilon));
        double k = (SprtTM * c / SprtMS) + 1;
        double an = k, prevAn;
        uint i = 0;
        do
        {
            prevAn = an;
            an = k + Math.Log(an);
        }
        while ((an - prevAn > 1.5e-8) && (++i < 10));

        _a = an;
        _lambdaReject = (1.0 - _delta) / (1.0 - _epsilon);
        _lambdaAccept = _delta / _epsilon;
    }

    private void NStarOptimize()
    {
        const uint MinSampleLength = 10 * 2;
        uint bestN = (uint)_n;
        uint testN = bestN;
        uint bestNumInl = _bestNumInl;
        uint testNumInl = bestNumInl;
        for (; testN > MinSampleLength && testNumInl != 0; testN--)
        {
            if (unchecked(testNumInl * bestN) > unchecked(bestNumInl * testN))
            {
                if (testNumInl < _nrTable[testN])
                {
                    break;
                }

                bestN = testN;
                bestNumInl = testNumInl;
            }

            testNumInl -= _bestInl[testN - 1] != 0 ? 1u : 0u;
        }

        if (unchecked(bestNumInl * _phMax) > unchecked(_phNumInl * bestN))
        {
            _phMax = bestN;
            _phNumInl = bestNumInl;
            _maxI = CalcIterBound(_cfd, (double)_phNumInl / _phMax, SampleSize, _maxI);
        }
    }

    private static void InitNonRand(uint[] table, int n)
    {
        double betaChi = Math.Sqrt(Beta * (1.0 - Beta)) * ChiSq;
        for (int k = SampleSize + 1; k < n; k++)
        {
            double mu = k * Beta;
            double sigma = Math.Sqrt((double)k) * betaChi;
            table[k] = ToUInt(Math.Ceiling(SampleSize + mu + sigma));
        }
    }

    private static double InitPEndFpI(uint ransacConvg, uint n, uint s)
    {
        double numer = 1, denom = 1;
        for (uint i = 0; i < s; i++)
        {
            numer *= s - i;
            denom *= unchecked(n - i);
        }

        return ransacConvg * numer / denom;
    }

    private static uint CalcIterBound(double confidence, double inlierRate, uint sampleSize, uint maxIterBound)
    {
        uint retVal;
        double atLeastOneOutlierProbability = 1.0 - Math.Pow(inlierRate, sampleSize);
        if (atLeastOneOutlierProbability >= 1.0)
        {
            retVal = maxIterBound;
        }
        else if (atLeastOneOutlierProbability <= 0.0)
        {
            retVal = 1;
        }
        else
        {
            retVal = ToUInt(Math.Ceiling(Math.Log(1.0 - confidence) / Math.Log(atLeastOneOutlierProbability)));
        }

        return retVal <= maxIterBound ? retVal : maxIterBound;
    }

    /// <summary>Solves the 4-point homography with the RHO Gaussian elimination.</summary>
    private static void HFuncRefC(float[] p, float[] h)
    {
        float x0 = p[0], y0 = p[1], x1 = p[2], y1 = p[3], x2 = p[4], y2 = p[5], x3 = p[6], y3 = p[7];
        float X0 = p[8], Y0 = p[9], X1 = p[10], Y1 = p[11], X2 = p[12], Y2 = p[13], X3 = p[14], Y3 = p[15];
        float x0X0 = x0 * X0, x1X1 = x1 * X1, x2X2 = x2 * X2, x3X3 = x3 * X3;
        float x0Y0 = x0 * Y0, x1Y1 = x1 * Y1, x2Y2 = x2 * Y2, x3Y3 = x3 * Y3;
        float y0X0 = y0 * X0, y1X1 = y1 * X1, y2X2 = y2 * X2, y3X3 = y3 * X3;
        float y0Y0 = y0 * Y0, y1Y1 = y1 * Y1, y2Y2 = y2 * Y2, y3Y3 = y3 * Y3;

        Span<float> mn = stackalloc float[8];
        mn[0] = x0 - x2; mn[1] = x1 - x2; mn[2] = x2; mn[3] = x3 - x2;
        mn[4] = y0 - y2; mn[5] = y1 - y2; mn[6] = y2; mn[7] = y3 - y2;
        Span<float> mj = stackalloc float[24];
        mj[0] = x2X2 - x0X0; mj[1] = x2X2 - x1X1; mj[2] = -x2X2; mj[3] = x2X2 - x3X3;
        mj[4] = x2Y2 - x0Y0; mj[5] = x2Y2 - x1Y1; mj[6] = -x2Y2; mj[7] = x2Y2 - x3Y3;
        mj[8] = y2X2 - y0X0; mj[9] = y2X2 - y1X1; mj[10] = -y2X2; mj[11] = y2X2 - y3X3;
        mj[12] = y2Y2 - y0Y0; mj[13] = y2Y2 - y1Y1; mj[14] = -y2Y2; mj[15] = y2Y2 - y3Y3;
        mj[16] = X0 - X2; mj[17] = X1 - X2; mj[18] = X2; mj[19] = X3 - X2;
        mj[20] = Y0 - Y2; mj[21] = Y1 - Y2; mj[22] = Y2; mj[23] = Y3 - Y2;

        // minor[r][c] = mn[4r + c]; major[r][c] = mj[8r + c].
        float s1 = mn[0], s2 = mn[1];
        mn[5] = (mn[5] * s1) - (mn[4] * s2);
        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 1] = (mj[(8 * r) + 1] * s1) - (mj[8 * r] * s2);
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 5] = (mj[(8 * r) + 5] * s1) - (mj[(8 * r) + 4] * s2);
        }

        s2 = mn[3];
        mn[7] = (mn[7] * s1) - (mn[4] * s2);
        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 3] = (mj[(8 * r) + 3] * s1) - (mj[8 * r] * s2);
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 7] = (mj[(8 * r) + 7] * s1) - (mj[(8 * r) + 4] * s2);
        }

        s1 = mn[5];
        s2 = mn[7];
        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 3] = (mj[(8 * r) + 3] * s1) - (mj[(8 * r) + 1] * s2);
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 7] = (mj[(8 * r) + 7] * s1) - (mj[(8 * r) + 5] * s2);
        }

        s2 = mn[4];
        mn[0] = (mn[0] * s1) - (mn[1] * s2);
        for (int r = 0; r < 3; r++)
        {
            mj[8 * r] = (mj[8 * r] * s1) - (mj[(8 * r) + 1] * s2);
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 4] = (mj[(8 * r) + 4] * s1) - (mj[(8 * r) + 5] * s2);
        }

        s1 = 1.0f / mn[0];
        for (int r = 0; r < 3; r++)
        {
            mj[8 * r] *= s1;
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 4] *= s1;
        }

        s1 = 1.0f / mn[5];
        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 1] *= s1;
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 5] *= s1;
        }

        s1 = mn[2];
        s2 = mn[6];
        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 2] -= (mj[8 * r] * s1) + (mj[(8 * r) + 1] * s2);
        }

        for (int r = 0; r < 3; r++)
        {
            mj[(8 * r) + 6] -= (mj[(8 * r) + 4] * s1) + (mj[(8 * r) + 5] * s2);
        }

        s1 = mj[7];
        mj[15] /= s1;
        mj[23] /= s1;
        for (int c = 0; c < 7; c++)
        {
            s1 = mj[c];
            mj[8 + c] -= s1 * mj[15];
            mj[16 + c] -= s1 * mj[23];
        }

        s1 = mj[11];
        mj[19] /= s1;
        for (int c = 0; c < 8; c++)
        {
            if (c == 3)
            {
                continue;
            }

            s1 = mj[8 + c];
            mj[16 + c] -= s1 * mj[19];
        }

        h[0] = mj[16];
        h[1] = mj[17];
        h[2] = mj[18];
        h[3] = mj[20];
        h[4] = mj[21];
        h[5] = mj[22];
        h[6] = mj[23];
        h[7] = mj[19];
        h[8] = 1.0f;
    }

    private void Refine()
    {
        float lambda = 100.0f;
        Span<float> dH = stackalloc float[8];
        Span<float> newH = stackalloc float[9];
        float s = CalcJacobianErrors(_bestH, true);
        for (int i = 0; i < MaxLevMarqIters; i++)
        {
            while (!Chol8x8Damped(_jtj, lambda, _tmp1))
            {
                lambda *= 2.0f;
            }

            TrInv8x8(_tmp1);
            TriSolve8x8(_tmp1, _jte, dH);
            for (int k = 0; k < 8; k++)
            {
                newH[k] = _bestH[k] - dH[k];
            }

            newH[8] = _bestH[8];
            float newS = CalcJacobianErrors(newH, false);
            float gain = LmGain(dH, _jte, s, newS, lambda);
            if (gain < LmGainLo)
            {
                lambda *= 8;
                if (lambda > 1000.0f / FltEpsilon)
                {
                    break;
                }
            }
            else if (gain > LmGainHi)
            {
                lambda *= 0.5f;
            }

            if (gain > 0)
            {
                newH[..8].CopyTo(_bestH);
                s = CalcJacobianErrors(_bestH, true);
            }
        }
    }

    /// <summary><c>sacCalcJacobianErrors</c> over the inliers of the best model.</summary>
    /// <returns>The sum of squared errors.</returns>
    private float CalcJacobianErrors(ReadOnlySpan<float> h, bool jacobian)
    {
        float[] jtj = _jtj;
        float[] jte = _jte;
        if (jacobian)
        {
            Array.Clear(jtj);
            Array.Clear(jte);
        }

        float s = 0.0f;
        byte[] inl = _bestInl;
        for (int i = 0; i < _n; i++)
        {
            if (inl[i] == 0)
            {
                continue;
            }

            float x = _src[2 * i], y = _src[(2 * i) + 1];
            float bx = _dst[2 * i], by = _dst[(2 * i) + 1];
            float w = (h[6] * x) + (h[7] * y) + 1.0f;
            float iW = MathF.Abs(w) > FltEpsilon ? 1.0f / w : 0;
            float reprojX = ((h[0] * x) + (h[1] * y) + h[2]) * iW;
            float reprojY = ((h[3] * x) + (h[4] * y) + h[5]) * iW;
            float eX = reprojX - bx;
            float eY = reprojY - by;
            float e = (eX * eX) + (eY * eY);
            s += e;
            if (!jacobian)
            {
                continue;
            }

            float dxh11 = x * iW;
            float dxh12 = y * iW;
            float dxh13 = iW;
            float dxh31 = -reprojX * x * iW;
            float dxh32 = -reprojX * y * iW;
            float dyh21 = x * iW;
            float dyh22 = y * iW;
            float dyh23 = iW;
            float dyh31 = -reprojY * x * iW;
            float dyh32 = -reprojY * y * iW;

            jte[0] += eX * dxh11;
            jte[1] += eX * dxh12;
            jte[2] += eX * dxh13;
            jte[3] += eY * dyh21;
            jte[4] += eY * dyh22;
            jte[5] += eY * dyh23;
            jte[6] += (eX * dxh31) + (eY * dyh31);
            jte[7] += (eX * dxh32) + (eY * dyh32);

            jtj[0] += dxh11 * dxh11;
            jtj[8] += dxh11 * dxh12;
            jtj[9] += dxh12 * dxh12;
            jtj[16] += dxh11 * dxh13;
            jtj[17] += dxh12 * dxh13;
            jtj[18] += dxh13 * dxh13;
            jtj[27] += dyh21 * dyh21;
            jtj[35] += dyh21 * dyh22;
            jtj[36] += dyh22 * dyh22;
            jtj[43] += dyh21 * dyh23;
            jtj[44] += dyh22 * dyh23;
            jtj[45] += dyh23 * dyh23;
            jtj[48] += dxh11 * dxh31;
            jtj[49] += dxh12 * dxh31;
            jtj[50] += dxh13 * dxh31;
            jtj[51] += dyh21 * dyh31;
            jtj[52] += dyh22 * dyh31;
            jtj[53] += dyh23 * dyh31;
            jtj[54] += (dxh31 * dxh31) + (dyh31 * dyh31);
            jtj[56] += dxh11 * dxh32;
            jtj[57] += dxh12 * dxh32;
            jtj[58] += dxh13 * dxh32;
            jtj[59] += dyh21 * dyh32;
            jtj[60] += dyh22 * dyh32;
            jtj[61] += dyh23 * dyh32;
            jtj[62] += (dxh31 * dxh32) + (dyh31 * dyh32);
            jtj[63] += (dxh32 * dxh32) + (dyh32 * dyh32);
        }

        return s;
    }

    private static float LmGain(ReadOnlySpan<float> dH, ReadOnlySpan<float> jte, float s, float newS, float lambda)
    {
        float dS = s - newS;
        float dL = 0;
        for (int i = 0; i < 8; i++)
        {
            dL += dH[i] * dH[i];
        }

        dL *= lambda;
        for (int i = 0; i < 8; i++)
        {
            dL += dH[i] * jte[i];
        }

        dL *= 0.5f;
        return MathF.Abs(dL) < FltEpsilon ? dS : dS / dL;
    }

    private static bool Chol8x8Damped(ReadOnlySpan<float> a, float lambda, Span<float> l)
    {
        float lambdap1 = lambda + 1.0f;
        for (int i = 0; i < 8; i++)
        {
            float x;
            for (int j = 0; j < i; j++)
            {
                x = a[(8 * i) + j];
                for (int k = 0; k < j; k++)
                {
                    x -= l[(8 * i) + k] * l[(8 * j) + k];
                }

                l[(8 * i) + j] = x / l[(8 * j) + j];
            }

            x = a[(8 * i) + i] * lambdap1;
            for (int k = 0; k < i; k++)
            {
                x -= l[(8 * i) + k] * l[(8 * i) + k];
            }

            if (x < 0)
            {
                return false;
            }

            l[(8 * i) + i] = MathF.Sqrt(x);
        }

        return true;
    }

    /// <summary>
    /// <c>sacTRInv8x8(L, L)</c>: inverts the lower-triangular matrix in place. The source and the
    /// destination are the same buffer, as in the OpenCV call.
    /// </summary>
    private static void TrInv8x8(Span<float> m)
    {
        Span<float> s = stackalloc float[4];
        Span<float> t = stackalloc float[4];
        Span<float> u = stackalloc float[16];
        Span<float> v = stackalloc float[16];

        for (int i = 0; i < 8; i++)
        {
            m[(8 * i) + i] = 1.0f / m[(8 * i) + i];
        }

        m[8] = -m[9] * m[8] * m[0];
        m[26] = -m[27] * m[26] * m[18];
        m[44] = -m[45] * m[44] * m[36];
        m[62] = -m[63] * m[62] * m[54];

        for (int blk = 0; blk < 2; blk++)
        {
            // Rows r0, r0 + 1 against columns c0, c0 + 1 (blk 0: rows 2-3, columns 0-1; blk 1: rows 6-7, columns 4-5).
            int r0 = blk == 0 ? 2 : 6;
            int c0 = blk == 0 ? 0 : 4;
            float mr0r0 = m[(8 * r0) + r0];
            float mr1r0 = m[(8 * (r0 + 1)) + r0];
            float mr1r1 = m[(8 * (r0 + 1)) + r0 + 1];
            s[0] = mr0r0 * m[(8 * r0) + c0];
            s[1] = mr0r0 * m[(8 * r0) + c0 + 1];
            s[2] = (mr1r0 * m[(8 * r0) + c0]) + (mr1r1 * m[(8 * (r0 + 1)) + c0]);
            s[3] = (mr1r0 * m[(8 * r0) + c0 + 1]) + (mr1r1 * m[(8 * (r0 + 1)) + c0 + 1]);
            float mc0c0 = m[(8 * c0) + c0];
            float mc1c0 = m[(8 * (c0 + 1)) + c0];
            float mc1c1 = m[(8 * (c0 + 1)) + c0 + 1];
            t[0] = (s[0] * mc0c0) + (s[1] * mc1c0);
            t[1] = s[1] * mc1c1;
            t[2] = (s[2] * mc0c0) + (s[3] * mc1c0);
            t[3] = s[3] * mc1c1;
            m[(8 * r0) + c0] = -t[0];
            m[(8 * r0) + c0 + 1] = -t[1];
            m[(8 * (r0 + 1)) + c0] = -t[2];
            m[(8 * (r0 + 1)) + c0 + 1] = -t[3];
        }

        for (int r = 0; r < 4; r++)
        {
            int row = 4 + r;
            for (int c = 0; c < 4; c++)
            {
                float acc = m[(8 * row) + 4] * m[32 + c];
                for (int k = 1; k <= r; k++)
                {
                    acc += m[(8 * row) + 4 + k] * m[(8 * (4 + k)) + c];
                }

                u[(4 * r) + c] = acc;
            }
        }

        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                float acc = u[(4 * r) + c] * m[(8 * c) + c];
                for (int k = c + 1; k < 4; k++)
                {
                    acc += u[(4 * r) + k] * m[(8 * k) + c];
                }

                v[(4 * r) + c] = acc;
            }
        }

        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                m[(8 * (4 + r)) + c] = -v[(4 * r) + c];
            }
        }
    }

    private static void TriSolve8x8(ReadOnlySpan<float> l, ReadOnlySpan<float> jte, Span<float> dH)
    {
        Span<float> t = stackalloc float[8];
        for (int i = 0; i < 8; i++)
        {
            float acc = l[8 * i] * jte[0];
            for (int k = 1; k <= i; k++)
            {
                acc += l[(8 * i) + k] * jte[k];
            }

            t[i] = acc;
        }

        for (int i = 0; i < 8; i++)
        {
            float acc = l[(8 * i) + i] * t[i];
            for (int k = i + 1; k < 8; k++)
            {
                acc += l[(8 * k) + i] * t[k];
            }

            dH[i] = acc;
        }
    }

    /// <summary>x86 <c>cvttss2si</c>: truncation, or <c>int.MinValue</c> for NaN and out-of-range values.</summary>
    private static int Trunc(float f)
        => f > -2147483904.0f && f < 2147483648.0f ? (int)f : int.MinValue;

    /// <summary>
    /// MSVC x64 <c>(unsigned)double</c>: a 64-bit truncation, then the low 32 bits.
    /// </summary>
    private static uint ToUInt(double d)
        => d > -9.2233720368547758e18 && d < 9.2233720368547758e18 ? unchecked((uint)(long)d) : 0u;
}
