// This implementation is ported from the OpenCV DFT-based 2-D correlation.
//
// Original sources:
//   modules/imgproc/src/filter.dispatch.cpp (dftFilter2D)
//   modules/imgproc/src/templmatch.cpp (crossCorr)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>dftFilter2D</c> and <c>crossCorr</c>: block-wise correlation with the ported DFT.
/// Byte sources use float32 spectra, except for double destinations; all other sources use
/// float64 spectra (<c>maxDepth</c> in <c>crossCorr</c>). The kernel is float32, or float64 for
/// double sources. Each tile extrapolates its border from the clipped source window, as OpenCV
/// does.
/// </summary>
/// <remarks>
/// OpenCV processes the tiles serially. The port processes tiles and channels in parallel; every
/// tile has the same operation sequence for any thread count.
/// </remarks>
internal static class CrossCorrEngine
{
    private const double BlockScale = 4.5;
    private const int MinBlockSize = 256;

    /// <summary>
    /// Smallest kernel area that selects the DFT path (<c>dftFilter2D</c> with SSE3 hardware).
    /// </summary>
    public static int Threshold<TSrc, TDst>()
        => (typeof(TSrc) == typeof(byte) && (typeof(TDst) == typeof(byte) || typeof(TDst) == typeof(short)))
            || (typeof(TSrc) == typeof(float) && typeof(TDst) == typeof(float))
            ? 130
            : 50;

    /// <summary>
    /// <c>dftFilter2D</c> for a whole (isolated) image with a row-major <paramref name="kw"/> ×
    /// <paramref name="kh"/> kernel.
    /// </summary>
    public static Mat<TDst> Filter2D<TSrc, TDst>(Mat<TSrc> src, double[] kernel, int kw, int kh, int ax, int ay, double delta, BorderType borderType)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        int cn = src.Channels;
        var dst = new Mat<TDst>(src.Width, src.Height, cn, clear: false);
        if (cn == 1 || delta == 0)
        {
            CrossCorr(src, kernel, kw, kh, dst, ax, ay, delta, borderType);
            return dst;
        }

        // crossCorr rejects a delta with several channels: OpenCV correlates into the destination
        // (float or double) or into a float temporary, and adds the delta afterwards.
        if (typeof(TDst) == typeof(double))
        {
            var ddst = (Mat<double>)(object)dst;
            CrossCorr(src, kernel, kw, kh, ddst, ax, ay, 0, borderType);
            ParallelRows.For(ddst.Height, ddst.Width * cn / Vector<double>.Count, (y0, y1) => AddDeltaDouble(ddst, delta, y0, y1));
            return dst;
        }

        float fdelta = (float)delta;
        if (typeof(TDst) == typeof(float))
        {
            var fdst = (Mat<float>)(object)dst;
            CrossCorr(src, kernel, kw, kh, fdst, ax, ay, 0, borderType);
            ParallelRows.For(fdst.Height, fdst.Width * cn / Vector<float>.Count, (y0, y1) => AddDelta(fdst, fdst, fdelta, y0, y1));
            return dst;
        }

        var temp = new Mat<float>(src.Width, src.Height, cn, clear: false);
        CrossCorr(src, kernel, kw, kh, temp, ax, ay, 0, borderType);
        ParallelRows.For(temp.Height, temp.Width * cn / Vector<float>.Count, (y0, y1) => AddDelta(temp, dst, fdelta, y0, y1));
        return dst;
    }

    private static void AddDelta<TDst>(Mat<float> temp, Mat<TDst> dst, float fdelta, int y0, int y1)
        where TDst : unmanaged
    {
        for (int y = y0; y < y1; y++)
        {
            ReadOnlySpan<float> t = temp.GetReadOnlyRowSpan(y);
            Span<TDst> d = dst.GetRowSpan(y);
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = Saturate.FromFloat<TDst>(t[i] + fdelta);
            }
        }
    }

    private static void AddDeltaDouble(Mat<double> dst, double delta, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            Span<double> d = dst.GetRowSpan(y);
            for (int i = 0; i < d.Length; i++)
            {
                d[i] += delta;
            }
        }
    }

    /// <summary>
    /// <c>crossCorr(img, templ, corr, anchor, delta, borderType | BORDER_ISOLATED)</c> with
    /// <paramref name="corr"/> of the image size and channel count. The spectra are float64 when
    /// the source depth is above <c>CV_8S</c> or the destination is double, float32 otherwise.
    /// </summary>
    public static void CrossCorr<TSrc, TDst>(Mat<TSrc> img, double[] templ, int kw, int kh, Mat<TDst> corr, int ax, int ay, double delta, BorderType borderType)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        if (typeof(TSrc) == typeof(byte) && typeof(TDst) != typeof(double))
        {
            new Job<TSrc, float, TDst>(img, templ, kw, kh, corr, ax, ay, delta, borderType).Run();
        }
        else
        {
            new Job<TSrc, double, TDst>(img, templ, kw, kh, corr, ax, ay, delta, borderType).Run();
        }
    }

    private sealed class Job<TSrc, TS, TDst>
        where TSrc : unmanaged
        where TS : unmanaged, IBinaryFloatingPointIeee754<TS>
        where TDst : unmanaged
    {
        private readonly Mat<TSrc> _img;
        private readonly Mat<TDst> _corr;
        private readonly int _ax, _ay;
        private readonly double _delta;
        private readonly BorderType _border;
        private readonly int _kw, _kh;
        private readonly int _bw, _bh, _dw, _dh;
        private readonly int _tilesX, _tilesY;
        private readonly Mat<TS> _templSpectrum;

        public Job(Mat<TSrc> img, double[] templ, int kw, int kh, Mat<TDst> corr, int ax, int ay, double delta, BorderType borderType)
        {
            _img = img;
            _corr = corr;
            _ax = ax;
            _ay = ay;
            _delta = delta;
            _border = borderType;
            _kw = kw;
            _kh = kh;
            int w = corr.Width, h = corr.Height;

            int bw = Math.Min(Math.Max(RoundHalfEven(_kw * BlockScale), MinBlockSize - _kw + 1), w);
            int bh = Math.Min(Math.Max(RoundHalfEven(_kh * BlockScale), MinBlockSize - _kh + 1), h);
            _dw = Math.Max(SpectralTransforms.GetOptimalDftSize(bw + _kw - 1), 2);
            _dh = SpectralTransforms.GetOptimalDftSize(bh + _kh - 1);
            if (_dw <= 0 || _dh <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(img), "The image and kernel are too big for the DFT path.");
            }

            _bw = Math.Min(_dw - _kw + 1, w);
            _bh = Math.Min(_dh - _kh + 1, h);
            _tilesX = (w + _bw - 1) / _bw;
            _tilesY = (h + _bh - 1) / _bh;

            // crossCorr converts the kernel to max(CV_32F, depth): float32 except for double sources.
            var padded = new Mat<TS>(_dw, _dh);
            for (int y = 0; y < _kh; y++)
            {
                Span<TS> d = padded.GetRowSpan(y);
                for (int x = 0; x < _kw; x++)
                {
                    double k = templ[(y * _kw) + x];
                    d[x] = TS.CreateTruncating(typeof(TSrc) == typeof(double) ? k : (float)k);
                }
            }

            _templSpectrum = new Mat<TS>(_dw, _dh, 1, clear: false);
            DftEngine<TS>.Dft(padded, _templSpectrum, inv: false, rowTransform: false, scaled: false, nonzeroRows: _kh);
        }

        public void Run()
        {
            int items = _tilesX * _tilesY * _corr.Channels;
            if (items == 1)
            {
                Tile(0);
                return;
            }

            ParallelRows.ForEachSerial(items, Tile);
        }

        private void Tile(int item)
        {
            int cn = _corr.Channels;
            int k = item % cn;
            int i = item / cn;
            int x = (i % _tilesX) * _bw;
            int y = (i / _tilesX) * _bh;
            int bsw = Math.Min(_bw, _corr.Width - x);
            int bsh = Math.Min(_bh, _corr.Height - y);
            int dsw = bsw + _kw - 1;
            int dsh = bsh + _kh - 1;
            int x0 = x - _ax, y0 = y - _ay;
            int x1 = Math.Max(0, x0), y1 = Math.Max(0, y0);
            int x2 = Math.Min(_img.Width, x0 + dsw), y2 = Math.Min(_img.Height, y0 + dsh);

            int length = _dw * _dh;
            TS[] a = ArrayPool<TS>.Shared.Rent(length);
            TS[] b = ArrayPool<TS>.Shared.Rent(length);
            try
            {
                var image = new Mat<TS>(a, _dw, _dh);
                var spectrum = new Mat<TS>(b, _dw, _dh);
                a.AsSpan(0, length).Clear();
                LoadWindow(image, k, x1 - x0, y1 - y0, x1, y1, x2 - x1, y2 - y1, dsw, dsh);

                DftEngine<TS>.Dft(image, spectrum, inv: false, rowTransform: false, scaled: false, nonzeroRows: dsh);
                SpectrumKernels.Mul(spectrum, _templSpectrum, spectrum, rowsFlag: false, conjB: true);
                DftEngine<TS>.Dft(spectrum, image, inv: true, rowTransform: false, scaled: true, nonzeroRows: bsh);

                StoreTile(image, k, x, y, bsw, bsh);
            }
            finally
            {
                ArrayPool<TS>.Shared.Return(b);
                ArrayPool<TS>.Shared.Return(a);
            }
        }

        /// <summary>
        /// Copies channel <paramref name="k"/> of the source window to <c>(left, top)</c> and, when
        /// the window is smaller than <c>dsw×dsh</c>, extrapolates the border from the window.
        /// </summary>
        private void LoadWindow(Mat<TS> image, int k, int left, int top, int sx, int sy, int ww, int wh, int dsw, int dsh)
        {
            int cn = _img.Channels;
            bool full = ww == dsw && wh == dsh;
            int[] colMap = ArrayPool<int>.Shared.Rent(dsw);
            try
            {
                for (int c = 0; c < dsw; c++)
                {
                    colMap[c] = full ? c : Borders.BorderInterpolate(c - left, ww, _border);
                }

                for (int r = 0; r < dsh; r++)
                {
                    int wr = full ? r : Borders.BorderInterpolate(r - top, wh, _border);
                    if (wr < 0)
                    {
                        continue;
                    }

                    Span<TS> d = image.GetRowSpan(r)[..dsw];
                    ReadOnlySpan<TSrc> s = _img.GetReadOnlyRowSpan(sy + wr);
                    int baseIndex = (sx * cn) + k;
                    if (typeof(TSrc) == typeof(byte))
                    {
                        ReadOnlySpan<byte> sb = MemoryMarshal.Cast<TSrc, byte>(s);
                        for (int c = 0; c < dsw; c++)
                        {
                            int wc = colMap[c];
                            d[c] = wc < 0 ? TS.Zero : TS.CreateTruncating(sb[baseIndex + (wc * cn)]);
                        }
                    }
                    else if (typeof(TSrc) == typeof(float))
                    {
                        ReadOnlySpan<float> sf = MemoryMarshal.Cast<TSrc, float>(s);
                        for (int c = 0; c < dsw; c++)
                        {
                            int wc = colMap[c];
                            d[c] = wc < 0 ? TS.Zero : TS.CreateTruncating(sf[baseIndex + (wc * cn)]);
                        }
                    }
                    else
                    {
                        for (int c = 0; c < dsw; c++)
                        {
                            int wc = colMap[c];
                            d[c] = wc < 0 ? TS.Zero : TS.CreateTruncating(Saturate.ToDouble(s[baseIndex + (wc * cn)]));
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(colMap);
            }
        }

        /// <summary>
        /// <c>convertTo(cdst, cdepth, 1, delta)</c> for one channel: float spectra add
        /// <c>(float)delta</c> in float; double spectra add <c>delta</c> in double. A zero delta
        /// copies or converts without the addition.
        /// </summary>
        private void StoreTile(Mat<TS> image, int k, int x, int y, int bsw, int bsh)
        {
            int cn = _corr.Channels;
            bool addDelta = _delta != 0;
            float fdelta = (float)_delta;
            for (int r = 0; r < bsh; r++)
            {
                ReadOnlySpan<TS> s = image.GetReadOnlyRowSpan(r);
                Span<TDst> d = _corr.GetRowSpan(y + r).Slice(x * cn);
                if (typeof(TS) == typeof(float))
                {
                    ReadOnlySpan<float> sf = MemoryMarshal.Cast<TS, float>(s);
                    for (int c = 0; c < bsw; c++)
                    {
                        d[(c * cn) + k] = Saturate.FromFloat<TDst>(addDelta ? sf[c] + fdelta : sf[c]);
                    }
                }
                else
                {
                    ReadOnlySpan<double> sd = MemoryMarshal.Cast<TS, double>(s);
                    if (addDelta && (typeof(TDst) == typeof(ushort) || typeof(TDst) == typeof(short)))
                    {
                        // cvt_32f (64F to 16-bit with scale): rows of at least 8 elements convert to
                        // float and add (float)delta in float; shorter rows compute v·1f + (float)delta in double.
                        for (int c = 0; c < bsw; c++)
                        {
                            d[(c * cn) + k] = bsw >= 8
                                ? Saturate.FromFloat<TDst>((float)sd[c] + fdelta)
                                : Saturate.FromDouble<TDst>(sd[c] + (double)fdelta);
                        }

                        continue;
                    }

                    for (int c = 0; c < bsw; c++)
                    {
                        d[(c * cn) + k] = Saturate.FromDouble<TDst>(addDelta ? sd[c] + _delta : sd[c]);
                    }
                }
            }
        }

        private static int RoundHalfEven(double v) => (int)Math.Round(v, MidpointRounding.ToEven);
    }
}
