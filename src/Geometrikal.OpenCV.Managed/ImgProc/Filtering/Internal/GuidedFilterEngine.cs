// This implementation is ported from the opencv_contrib guided filter.
//
// Original sources (opencv_contrib, module ximgproc, tag 5.x):
//   modules/ximgproc/src/guided_filter.cpp (GuidedFilterImpl)
//   modules/ximgproc/src/edgeaware_filters_common.cpp (mul, add_mul, sub_mul, sub_mad,
//     det_2x2, div_det_2x2, div_1x, inv_self)
//
// opencv_contrib is not part of this repository; the sources are at
// https://github.com/opencv/opencv_contrib.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <c>GuidedFilterImpl</c>: guide statistics computed once, then applied to any number of
/// sources. All planes are float32; box filters are <c>boxFilter(CV_32F, normalize, BORDER_REFLECT)</c>.
/// </summary>
/// <remarks>
/// <para>OpenCV computes the elementwise steps with SSE intrinsics on the first
/// <c>4·⌊w/4⌋</c> columns and scalar code on the others. The two parts agree except for
/// <c>sub_mad</c> (association of the eps term) and <c>inv_self</c> (<c>RCPPS</c> against
/// <c>1/x</c>); the port applies the same split. Without SSE every column uses the scalar
/// formula, as OpenCV built without SSE does.</para>
/// <para><c>RCPPS</c> is an approximation whose bits depend on the processor.</para>
/// </remarks>
internal sealed class GuidedFilterEngine
{
    private static readonly (int A, int B)[][] WalkPatterns =
    [
        [(0, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (2, 2), (2, 1), (1, 1)],
    ];

    private readonly int _k;
    private readonly int _h, _w;
    private readonly int _hOriginal, _wOriginal;
    private readonly int _n;
    private readonly Mat<float>[] _guideOriginal;
    private readonly Mat<float>[] _guide;
    private readonly Mat<float>[] _guideMean;
    private readonly Mat<float>[] _inverse;
    private readonly List<PooledMat<float>>? _pooled;

    private GuidedFilterEngine(int radius, double eps, double scale, Mat<float>[] guideOriginal, List<PooledMat<float>>? pooled)
    {
        _pooled = pooled;
        _k = (2 * radius) + 1;
        Scale = scale;
        _n = guideOriginal.Length;
        _guideOriginal = guideOriginal;
        _hOriginal = guideOriginal[0].Height;
        _wOriginal = guideOriginal[0].Width;
        _h = (int)(_hOriginal * scale);
        _w = (int)(_wOriginal * scale);
        if (_h <= 0 || _w <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "The scaled guide size must be at least 1×1.");
        }

        if (scale < 1.0)
        {
            _guide = new Mat<float>[_n];
            for (int c = 0; c < _n; c++)
            {
                _guide[c] = Geometry.Resize(guideOriginal[c], new Size(_w, _h));
            }
        }
        else
        {
            _guide = guideOriginal;
        }

        _guideMean = new Mat<float>[_n];
        var means = new (Mat<float>, Mat<float>?, Mat<float>)[_n];
        for (int c = 0; c < _n; c++)
        {
            _guideMean[c] = NewPlane(_w, _h);
            means[c] = (_guide[c], null, _guideMean[c]);
        }

        BoxFilterEngine.FloatBoxes(means, _k, BorderType.Reflect);

        _inverse = ComputeInverse((float)eps, eps < 1e-2);
    }

    public double Scale { get; }

    public int Width => _wOriginal;

    public int Height => _hOriginal;

    /// <summary>
    /// Computes the guide statistics. With <paramref name="pooled"/> the planes come from the array
    /// pool and <see cref="Release"/> returns them; the engine must not be used after that.
    /// </summary>
    public static GuidedFilterEngine Create<TGuide>(Mat<TGuide> guide, int radius, double eps, double scale, bool pooled = false)
        where TGuide : unmanaged
    {
        int n = Math.Min(3, guide.Channels);
        List<PooledMat<float>>? pool = pooled ? new List<PooledMat<float>>(16) : null;
        var planes = new Mat<float>[n];
        for (int c = 0; c < n; c++)
        {
            planes[c] = NewPlane(pool, guide.Width, guide.Height);
        }

        ParallelRows.For(guide.Height, (long)guide.Width * n, (y0, y1) => SplitToFloat(guide, planes, y0, y1));
        return new GuidedFilterEngine(radius, eps, scale, planes, pool);
    }

    /// <summary>Returns the pooled planes of an engine created with <c>pooled: true</c>.</summary>
    public void Release()
    {
        if (_pooled is null)
        {
            return;
        }

        foreach (PooledMat<float> m in _pooled)
        {
            m.Dispose();
        }

        _pooled.Clear();
    }

    private Mat<float> NewPlane(int w, int h) => NewPlane(_pooled, w, h);

    private static Mat<float> NewPlane(List<PooledMat<float>>? pool, int w, int h)
    {
        if (pool is null)
        {
            return new Mat<float>(w, h, 1, clear: false);
        }

        PooledMat<float> m = PooledMat<float>.Rent(w, h, 1);
        pool.Add(m);
        return m.Mat;
    }

    private static void SplitToFloat<TGuide>(Mat<TGuide> guide, Mat<float>[] planes, int y0, int y1)
        where TGuide : unmanaged
    {
        int cn = guide.Channels;
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<TGuide> s = guide.GetReadOnlyRowSpan(y);
            for (int c = 0; c < planes.Length; c++)
            {
                ToFloatRow(s, cn, c, planes[c].GetRowSpan(y));
            }
        }
    }

    /// <summary>Index of the symmetric pair <c>(a, b)</c> in <c>SymArray2D</c> storage.</summary>
    private static int Sym(int a, int b) => a >= b ? (a * (a + 1) / 2) + b : (b * (b + 1) / 2) + a;

    private Mat<float>[] ComputeInverse(float epsF, bool clampDet)
    {
        // The 3-channel covariances are temporaries; the 1- and 2-channel ones become the inverse.
        var temporaries = new List<PooledMat<float>>(6);
        var cov = new Mat<float>[_n * (_n + 1) / 2];
        var boxes = new (Mat<float>, Mat<float>?, Mat<float>)[cov.Length];
        int job = 0;
        foreach ((int a, int b) in WalkPatterns[_n - 1])
        {
            Mat<float> c = _n == 3 ? NewPlane(temporaries, _w, _h) : NewPlane(_w, _h);
            boxes[job++] = (_guide[a], _guide[b], c);
            cov[Sym(a, b)] = c;
        }

        BoxFilterEngine.FloatBoxes(boxes, _k, BorderType.Reflect);

        float c0 = -epsF;
        ParallelRows.For(_h, (long)_w * cov.Length, (y0, y1) => CovarianceRows(cov, _guideMean, _n, c0, y0, y1));

        if (_n == 1)
        {
            ParallelRows.For(_h, _w, (y0, y1) => InverseSelfRows(cov[0], y0, y1));
            return cov;
        }

        if (_n == 2)
        {
            // covarsInv(0,0) = covars(1,1), covarsInv(0,1) = covars(0,1), covarsInv(1,1) = covars(0,0).
            var inv = new Mat<float>[] { cov[Sym(1, 1)], cov[Sym(0, 1)], cov[Sym(0, 0)] };
            ParallelRows.For(_h, (long)_w * 3, (y0, y1) => DivDet2x2Rows(inv[0], inv[1], inv[2], y0, y1));
            return inv;
        }

        var inv3 = new Mat<float>[6];
        for (int i = 0; i < 6; i++)
        {
            inv3[i] = NewPlane(_w, _h);
        }

        ParallelRows.For(_h, (long)_w * 30, (y0, y1) => Inverse3Rows(cov, inv3, clampDet, y0, y1));
        foreach (PooledMat<float> m in temporaries)
        {
            m.Dispose();
        }

        return inv3;
    }

    /// <summary>
    /// <c>sub_mul</c> for off-diagonal entries and <c>sub_mad(cov, m_a, m_b, -eps)</c> for the
    /// diagonal (ComputeCovGuideFromChannelsMul_ParBody).
    /// </summary>
    private static void CovarianceRows(Mat<float>[] cov, Mat<float>[] mean, int n, float c0, int y0, int y1)
    {
        foreach ((int a, int b) in WalkPatterns[n - 1])
        {
            Mat<float> c = cov[Sym(a, b)];
            for (int y = y0; y < y1; y++)
            {
                Span<float> d = c.GetRowSpan(y);
                ReadOnlySpan<float> ma = mean[a].GetReadOnlyRowSpan(y), mb = mean[b].GetReadOnlyRowSpan(y);
                if (a != b)
                {
                    SubMul(d, ma, mb);
                }
                else
                {
                    SubMad(d, ma, mb, c0);
                }
            }
        }
    }

    private static int LaneEnd(int w) => Sse.IsSupported ? w & ~3 : 0;

    /// <summary><c>sub_mad</c>: lanes <c>(d − c0) − s2·s1</c>, tail <c>d − (s1·s2 + c0)</c>.</summary>
    private static void SubMad(Span<float> d, ReadOnlySpan<float> s1, ReadOnlySpan<float> s2, float c0)
    {
        int lanes = LaneEnd(d.Length);
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var vc = new Vector<float>(c0);
            for (; j <= lanes - Vector<float>.Count; j += Vector<float>.Count)
            {
                ((new Vector<float>(d[j..]) - vc) - (new Vector<float>(s2[j..]) * new Vector<float>(s1[j..]))).CopyTo(d[j..]);
            }
        }

        for (; j < lanes; j++)
        {
            d[j] = (d[j] - c0) - (s2[j] * s1[j]);
        }

        for (; j < d.Length; j++)
        {
            d[j] -= (s1[j] * s2[j]) + c0;
        }
    }

    /// <summary><c>sub_mul</c>: <c>d − s1·s2</c>.</summary>
    private static void SubMul(Span<float> d, ReadOnlySpan<float> s1, ReadOnlySpan<float> s2)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= d.Length - Vector<float>.Count; j += Vector<float>.Count)
            {
                (new Vector<float>(d[j..]) - (new Vector<float>(s1[j..]) * new Vector<float>(s2[j..]))).CopyTo(d[j..]);
            }
        }

        for (; j < d.Length; j++)
        {
            d[j] -= s1[j] * s2[j];
        }
    }

    /// <summary><c>inv_self</c>: <c>RCPPS</c> on the lane columns, <c>1/x</c> on the others.</summary>
    private static void InverseSelfRows(Mat<float> m, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            Span<float> d = m.GetRowSpan(y);
            int lanes = LaneEnd(d.Length);
            int j = 0;
            if (Sse.IsSupported)
            {
                ref float r = ref MemoryMarshal.GetReference(d);
                for (; j < lanes; j += 4)
                {
                    Sse.Reciprocal(Vector128.LoadUnsafe(ref r, (nuint)j)).StoreUnsafe(ref r, (nuint)j);
                }
            }

            for (; j < d.Length; j++)
            {
                d[j] = 1.0f / d[j];
            }
        }
    }

    /// <summary><c>div_det_2x2</c>: <c>det = a00·a11 − a01·a01</c>; each entry divided by det, a01 negated.</summary>
    private static void DivDet2x2Rows(Mat<float> m00, Mat<float> m01, Mat<float> m11, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            Span<float> a00 = m00.GetRowSpan(y), a01 = m01.GetRowSpan(y), a11 = m11.GetRowSpan(y);
            int j = 0;
            if (Vector.IsHardwareAccelerated)
            {
                for (; j <= a00.Length - Vector<float>.Count; j += Vector<float>.Count)
                {
                    var v00 = new Vector<float>(a00[j..]);
                    var v01 = new Vector<float>(a01[j..]);
                    var v11 = new Vector<float>(a11[j..]);
                    Vector<float> det = (v00 * v11) - (v01 * v01);
                    (v00 / det).CopyTo(a00[j..]);
                    (v11 / det).CopyTo(a11[j..]);
                    (-v01 / det).CopyTo(a01[j..]);
                }
            }

            for (; j < a00.Length; j++)
            {
                float det = (a00[j] * a11[j]) - (a01[j] * a01[j]);
                a00[j] /= det;
                a11[j] /= det;
                a01[j] = -a01[j] / det;
            }
        }
    }

    /// <summary>
    /// 3×3 symmetric inverse (ComputeCovGuideInv_ParBody, gCnNum 3): cofactors with
    /// <c>det_2x2</c>, <c>det = C00·I00 + C10·I10 + C20·I20</c>, the optional clamp of
    /// <c>|det| &lt; 1e-6</c> to 1, and <c>div_1x</c>.
    /// </summary>
    private static void Inverse3Rows(Mat<float>[] cov, Mat<float>[] inv, bool clampDet, int y0, int y1)
    {
        int w = cov[0].Width;
        float[] detBuffer = System.Buffers.ArrayPool<float>.Shared.Rent(w);
        try
        {
            Span<float> det = detBuffer.AsSpan(0, w);
            for (int y = y0; y < y1; y++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l <= k; l++)
                    {
                        Det2x2(
                            inv[Sym(k, l)].GetRowSpan(y),
                            cov[Sym((k + 1) % 3, (l + 1) % 3)].GetReadOnlyRowSpan(y),
                            cov[Sym((k + 1) % 3, (l + 2) % 3)].GetReadOnlyRowSpan(y),
                            cov[Sym((k + 2) % 3, (l + 1) % 3)].GetReadOnlyRowSpan(y),
                            cov[Sym((k + 2) % 3, (l + 2) % 3)].GetReadOnlyRowSpan(y));
                    }
                }

                ReadOnlySpan<float> c00 = cov[Sym(0, 0)].GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> c10 = cov[Sym(1, 0)].GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> c20 = cov[Sym(2, 0)].GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> i00 = inv[Sym(0, 0)].GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> i10 = inv[Sym(1, 0)].GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> i20 = inv[Sym(2, 0)].GetReadOnlyRowSpan(y);
                for (int j = 0; j < w; j++)
                {
                    float dv = c00[j] * i00[j];
                    dv += c10[j] * i10[j];
                    dv += c20[j] * i20[j];
                    if (clampDet && MathF.Abs(dv) < 1e-6f)
                    {
                        dv = 1f;
                    }

                    det[j] = dv;
                }

                for (int e = 0; e < 6; e++)
                {
                    DivideBy(inv[e].GetRowSpan(y), det);
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(detBuffer);
        }
    }

    private static void Det2x2(Span<float> d, ReadOnlySpan<float> a00, ReadOnlySpan<float> a01, ReadOnlySpan<float> a10, ReadOnlySpan<float> a11)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= d.Length - Vector<float>.Count; j += Vector<float>.Count)
            {
                ((new Vector<float>(a00[j..]) * new Vector<float>(a11[j..])) - (new Vector<float>(a01[j..]) * new Vector<float>(a10[j..]))).CopyTo(d[j..]);
            }
        }

        for (; j < d.Length; j++)
        {
            d[j] = (a00[j] * a11[j]) - (a01[j] * a10[j]);
        }
    }

    private static void DivideBy(Span<float> a, ReadOnlySpan<float> b)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= a.Length - Vector<float>.Count; j += Vector<float>.Count)
            {
                (new Vector<float>(a[j..]) / new Vector<float>(b[j..])).CopyTo(a[j..]);
            }
        }

        for (; j < a.Length; j++)
        {
            a[j] /= b[j];
        }
    }

    // ------------------------------------------------------------ filtering

    public Mat<TDst> Filter<TSrc, TDst>(Mat<TSrc> src)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        int cn = src.Channels;
        var dst = new Mat<TDst>(_wOriginal, _hOriginal, cn, clear: false);
        int n = _n;
        var pooled = new List<PooledMat<float>>(3 * n + 3);
        Mat<float> Rent()
        {
            PooledMat<float> m = PooledMat<float>.Rent(_w, _h, 1);
            pooled.Add(m);
            return m.Mat;
        }

        try
        {
            Mat<float> p = Rent(), pm = Rent(), betaMean = Rent();
            var cov = new Mat<float>[n];
            var alpha = new Mat<float>[n];
            var alphaMean = new Mat<float>[n];
            for (int g = 0; g < n; g++)
            {
                cov[g] = Rent();
                alpha[g] = Rent();
                alphaMean[g] = Rent();
            }

            FilterChannels(src, dst, p, pm, betaMean, cov, alpha, alphaMean);
        }
        finally
        {
            foreach (PooledMat<float> m in pooled)
            {
                m.Dispose();
            }
        }

        return dst;
    }

    private void FilterChannels<TSrc, TDst>(
        Mat<TSrc> src, Mat<TDst> dst, Mat<float> p, Mat<float> pm, Mat<float> betaMean, Mat<float>[] cov, Mat<float>[] alpha, Mat<float>[] alphaMean)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        int cn = src.Channels;
        int n = _n;
        for (int si = 0; si < cn; si++)
        {
            LoadSourcePlane(src, si, p);
            var boxes = new (Mat<float>, Mat<float>?, Mat<float>)[n + 1];
            for (int g = 0; g < n; g++)
            {
                boxes[g] = (p, _guide[g], cov[g]);
            }

            boxes[n] = (p, null, pm);
            BoxFilterEngine.FloatBoxes(boxes, _k, BorderType.Reflect);

            // pm becomes beta in place, as srcCnMean and beta share storage in OpenCV.
            ParallelRows.For(_h, (long)_w * n * (n + 3), (y0, y1) => AlphaBetaRows(cov, pm, alpha, y0, y1));

            boxes[n] = (pm, null, betaMean);
            for (int g = 0; g < n; g++)
            {
                boxes[g] = (alpha[g], null, alphaMean[g]);
            }

            BoxFilterEngine.FloatBoxes(boxes, _k, BorderType.Reflect);

            Mat<float> beta = betaMean;
            Mat<float>[] a = alphaMean;
            if (Scale < 1.0)
            {
                beta = Geometry.Resize(betaMean, new Size(_wOriginal, _hOriginal));
                a = new Mat<float>[n];
                for (int g = 0; g < n; g++)
                {
                    a[g] = Geometry.Resize(alphaMean[g], new Size(_wOriginal, _hOriginal));
                }
            }

            int channel = si;
            ParallelRows.For(_hOriginal, (long)_wOriginal * n, (y0, y1) => ApplyRows(beta, a, _guideOriginal, dst, channel, y0, y1));
        }
    }

    /// <summary>Element <c>c</c> of every <c>cn</c>-element pixel of <paramref name="s"/> as float (exact).</summary>
    private static void ToFloatRow<T>(ReadOnlySpan<T> s, int cn, int c, Span<float> d)
        where T : unmanaged
    {
        if (cn == 1 && typeof(T) == typeof(float))
        {
            MemoryMarshal.Cast<T, float>(s)[..d.Length].CopyTo(d);
            return;
        }

        int x = 0;
        if (cn == 1 && typeof(T) == typeof(byte) && Vector.IsHardwareAccelerated)
        {
            ref byte sr = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<T, byte>(s));
            ref float dr = ref MemoryMarshal.GetReference(d);
            int vf = Vector<float>.Count;
            for (; x <= d.Length - Vector<byte>.Count; x += Vector<byte>.Count)
            {
                FilterVectors.WidenToSingle(Vector.LoadUnsafe(ref sr, (nuint)x), out Vector<float> f0, out Vector<float> f1, out Vector<float> f2, out Vector<float> f3);
                f0.StoreUnsafe(ref dr, (nuint)x);
                f1.StoreUnsafe(ref dr, (nuint)(x + vf));
                f2.StoreUnsafe(ref dr, (nuint)(x + (2 * vf)));
                f3.StoreUnsafe(ref dr, (nuint)(x + (3 * vf)));
            }
        }

        for (; x < d.Length; x++)
        {
            d[x] = Saturate.ToFloat(s[(x * cn) + c]);
        }
    }

    /// <summary>
    /// Channel <paramref name="channel"/> of the source as float. With <c>scale &lt; 1</c> the plane
    /// is resized in its own depth first, as OpenCV subsamples before the conversion.
    /// </summary>
    private void LoadSourcePlane<TSrc>(Mat<TSrc> src, int channel, Mat<float> p)
        where TSrc : unmanaged
    {
        Mat<TSrc> plane = src;
        int cn = src.Channels;
        int c = channel;
        if (Scale < 1.0)
        {
            plane = Geometry.Resize(ExtractChannel(src, channel), new Size(_w, _h));
            cn = 1;
            c = 0;
        }

        ParallelRows.For(_h, _w, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                ToFloatRow(plane.GetReadOnlyRowSpan(y), cn, c, p.GetRowSpan(y));
            }
        });
    }

    private static Mat<TSrc> ExtractChannel<TSrc>(Mat<TSrc> src, int channel)
        where TSrc : unmanaged
    {
        int cn = src.Channels;
        var plane = new Mat<TSrc>(src.Width, src.Height, 1, clear: false);
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<TSrc> s = src.GetReadOnlyRowSpan(y);
            Span<TSrc> d = plane.GetRowSpan(y);
            for (int x = 0; x < d.Length; x++)
            {
                d[x] = s[(x * cn) + channel];
            }
        }

        return plane;
    }

    /// <summary>
    /// Per element, in OpenCV's order: <c>cov_k −= pm·m_k</c> (ComputeCovFromSrcChannelsMul);
    /// <c>alpha_g = I(g,0)·cov_0 + I(g,1)·cov_1 + …</c> (ComputeAlpha);
    /// <c>beta = pm − alpha_0·m_0 − alpha_1·m_1 − …</c> (ComputeBeta), written into
    /// <paramref name="pm"/>.
    /// </summary>
    private void AlphaBetaRows(Mat<float>[] cov, Mat<float> pm, Mat<float>[] alpha, int y0, int y1)
    {
        int n = _n;
        for (int y = y0; y < y1; y++)
        {
            Span<float> beta = pm.GetRowSpan(y);
            for (int k = 0; k < n; k++)
            {
                SubMul(cov[k].GetRowSpan(y), beta, _guideMean[k].GetReadOnlyRowSpan(y));
            }

            for (int g = 0; g < n; g++)
            {
                Span<float> a = alpha[g].GetRowSpan(y);
                Mul(a, _inverse[InverseIndex(g, 0)].GetReadOnlyRowSpan(y), cov[0].GetReadOnlyRowSpan(y));
                for (int k = 1; k < n; k++)
                {
                    AddMul(a, _inverse[InverseIndex(g, k)].GetReadOnlyRowSpan(y), cov[k].GetReadOnlyRowSpan(y));
                }
            }

            for (int g = 0; g < n; g++)
            {
                SubMul(beta, alpha[g].GetReadOnlyRowSpan(y), _guideMean[g].GetReadOnlyRowSpan(y));
            }
        }
    }

    /// <summary>Storage index of <c>covarsInv(g, k)</c>.</summary>
    private int InverseIndex(int g, int k) => _n == 3 ? Sym(g, k) : _n == 2 ? (g == 0 && k == 0 ? 0 : g == 1 && k == 1 ? 2 : 1) : 0;

    private static void Mul(Span<float> d, ReadOnlySpan<float> s1, ReadOnlySpan<float> s2)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= d.Length - Vector<float>.Count; j += Vector<float>.Count)
            {
                (new Vector<float>(s1[j..]) * new Vector<float>(s2[j..])).CopyTo(d[j..]);
            }
        }

        for (; j < d.Length; j++)
        {
            d[j] = s1[j] * s2[j];
        }
    }

    /// <summary><c>add_mul</c>: <c>d + s1·s2</c>.</summary>
    private static void AddMul(Span<float> d, ReadOnlySpan<float> s1, ReadOnlySpan<float> s2)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; j <= d.Length - Vector<float>.Count; j += Vector<float>.Count)
            {
                (new Vector<float>(d[j..]) + (new Vector<float>(s1[j..]) * new Vector<float>(s2[j..]))).CopyTo(d[j..]);
            }
        }

        for (; j < d.Length; j++)
        {
            d[j] += s1[j] * s2[j];
        }
    }

    /// <summary>
    /// ApplyTransform_ParBody: <c>q = beta + alpha_0·I_0 + alpha_1·I_1 + …</c> with the
    /// full-resolution guide, then <c>convertTo(dDepth)</c> into one channel of the result.
    /// </summary>
    private static void ApplyRows<TDst>(Mat<float> beta, Mat<float>[] alpha, Mat<float>[] guide, Mat<TDst> dst, int channel, int y0, int y1)
        where TDst : unmanaged
    {
        int w = beta.Width;
        int cn = dst.Channels;
        float[] rowBuffer = System.Buffers.ArrayPool<float>.Shared.Rent(w);
        try
        {
            Span<float> q = rowBuffer.AsSpan(0, w);
            for (int y = y0; y < y1; y++)
            {
                beta.GetReadOnlyRowSpan(y).CopyTo(q);
                for (int g = 0; g < alpha.Length; g++)
                {
                    AddMul(q, alpha[g].GetReadOnlyRowSpan(y), guide[g].GetReadOnlyRowSpan(y));
                }

                Span<TDst> d = dst.GetRowSpan(y);
                if (cn == 1 && (typeof(TDst) == typeof(float) || typeof(TDst) == typeof(byte) || typeof(TDst) == typeof(short)))
                {
                    FilterCasts.StoreRow(d, q);
                    continue;
                }

                for (int x = 0; x < w; x++)
                {
                    d[(x * cn) + channel] = Saturate.FromFloat<TDst>(q[x]);
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(rowBuffer);
        }
    }
}
