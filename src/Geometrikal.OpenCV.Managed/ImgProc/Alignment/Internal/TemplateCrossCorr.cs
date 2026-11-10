// This implementation is ported from the OpenCV DFT-based template correlation.
//
// Original sources:
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
/// OpenCV <c>crossCorr(img, templ, corr, Point(0, 0), 0, 0)</c> as <c>matchTemplate</c> calls it:
/// the template has the image channel count, and the correlation size is
/// <c>(W - w + 1) × (H - h + 1)</c>. A 1-channel <c>corr</c> receives the sum of the channel
/// correlations in float, in channel order; a <c>corr</c> with the image channel count receives
/// each channel correlation. Byte images use float32 spectra; float images use float64 spectra.
/// </summary>
/// <remarks>
/// This is the <see cref="CrossCorrEngine"/> tile scheme with one template spectrum per channel.
/// Every tile window lies inside the image, so no border is extrapolated. Tiles run in parallel;
/// the channels of one tile run in order, so the channel sum does not depend on the thread count.
/// </remarks>
internal static class TemplateCrossCorr
{
    private const double BlockScale = 4.5;
    private const int MinBlockSize = 256;

    public static void Run<T>(Mat<T> img, Mat<T> templ, Mat<float> corr)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            using var job = new Job<T, float>(img, templ, corr);
            job.Run();
        }
        else
        {
            using var job = new Job<T, double>(img, templ, corr);
            job.Run();
        }
    }

    private sealed class Job<TSrc, TS> : IDisposable
        where TSrc : unmanaged
        where TS : unmanaged, IBinaryFloatingPointIeee754<TS>
    {
        private readonly Mat<TSrc> _img;
        private readonly Mat<float> _corr;
        private readonly int _kw, _kh, _cn;
        private readonly int _bw, _bh, _dw, _dh;
        private readonly int _tilesX, _tilesY;
        private readonly Mat<TS>[] _templSpectra;
        private readonly PooledMat<TS>[] _spectraPool;

        public Job(Mat<TSrc> img, Mat<TSrc> templ, Mat<float> corr)
        {
            _img = img;
            _corr = corr;
            _kw = templ.Width;
            _kh = templ.Height;
            _cn = img.Channels;
            int w = corr.Width, h = corr.Height;

            int bw = Math.Min(Math.Max(RoundHalfEven(_kw * BlockScale), MinBlockSize - _kw + 1), w);
            int bh = Math.Min(Math.Max(RoundHalfEven(_kh * BlockScale), MinBlockSize - _kh + 1), h);
            _dw = Math.Max(SpectralTransforms.GetOptimalDftSize(bw + _kw - 1), 2);
            _dh = SpectralTransforms.GetOptimalDftSize(bh + _kh - 1);
            if (_dw <= 0 || _dh <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(img), "The image and template are too big for the DFT path.");
            }

            _bw = Math.Min(_dw - _kw + 1, w);
            _bh = Math.Min(_dh - _kh + 1, h);
            _tilesX = (w + _bw - 1) / _bw;
            _tilesY = (h + _bh - 1) / _bh;

            _templSpectra = new Mat<TS>[_cn];
            _spectraPool = new PooledMat<TS>[_cn];
            using PooledMat<TS> paddedPool = PooledMat<TS>.Rent(_dw, _dh, 1);
            Mat<TS> padded = paddedPool.Mat;
            for (int k = 0; k < _cn; k++)
            {
                padded.Clear();
                for (int y = 0; y < _kh; y++)
                {
                    ReadOnlySpan<TSrc> s = templ.GetReadOnlyRowSpan(y);
                    Span<TS> d = padded.GetRowSpan(y);
                    for (int x = 0; x < _kw; x++)
                    {
                        d[x] = Load(s, (x * _cn) + k);
                    }
                }

                _spectraPool[k] = PooledMat<TS>.Rent(_dw, _dh, 1);
                _templSpectra[k] = _spectraPool[k].Mat;
                DftEngine<TS>.Dft(padded, _templSpectra[k], inv: false, rowTransform: false, scaled: false, nonzeroRows: _kh);
            }
        }

        public void Dispose()
        {
            foreach (PooledMat<TS> spectrum in _spectraPool)
            {
                spectrum.Dispose();
            }
        }

        public void Run()
        {
            int tiles = _tilesX * _tilesY;
            if (tiles == 1)
            {
                Tile(0);
                return;
            }

            ParallelRows.ForEachSerial(tiles, Tile);
        }

        private static TS Load(ReadOnlySpan<TSrc> s, int i)
        {
            if (typeof(TSrc) == typeof(byte))
            {
                return TS.CreateTruncating(MemoryMarshal.Cast<TSrc, byte>(s)[i]);
            }

            return TS.CreateTruncating(MemoryMarshal.Cast<TSrc, float>(s)[i]);
        }

        private void Tile(int i)
        {
            int x = (i % _tilesX) * _bw;
            int y = (i / _tilesX) * _bh;
            int bsw = Math.Min(_bw, _corr.Width - x);
            int bsh = Math.Min(_bh, _corr.Height - y);
            int dsw = bsw + _kw - 1;
            int dsh = bsh + _kh - 1;

            int length = _dw * _dh;
            TS[] a = ArrayPool<TS>.Shared.Rent(length);
            TS[] b = ArrayPool<TS>.Shared.Rent(length);
            try
            {
                var image = new Mat<TS>(a, _dw, _dh);
                var spectrum = new Mat<TS>(b, _dw, _dh);
                for (int k = 0; k < _cn; k++)
                {
                    a.AsSpan(0, length).Clear();
                    for (int r = 0; r < dsh; r++)
                    {
                        ReadOnlySpan<TSrc> s = _img.GetReadOnlyRowSpan(y + r);
                        Span<TS> d = image.GetRowSpan(r);
                        int baseIndex = (x * _cn) + k;
                        for (int c = 0; c < dsw; c++)
                        {
                            d[c] = Load(s, baseIndex + (c * _cn));
                        }
                    }

                    DftEngine<TS>.Dft(image, spectrum, inv: false, rowTransform: false, scaled: false, nonzeroRows: dsh);
                    SpectrumKernels.Mul(spectrum, _templSpectra[k], spectrum, rowsFlag: false, conjB: true);
                    DftEngine<TS>.Dft(spectrum, image, inv: true, rowTransform: false, scaled: true, nonzeroRows: bsh);
                    Store(image, k, x, y, bsw, bsh);
                }
            }
            finally
            {
                ArrayPool<TS>.Shared.Return(b);
                ArrayPool<TS>.Shared.Return(a);
            }
        }

        private void Store(Mat<TS> image, int k, int x, int y, int bsw, int bsh)
        {
            int ccn = _corr.Channels;
            bool accumulate = ccn == 1 && k > 0;
            for (int r = 0; r < bsh; r++)
            {
                ReadOnlySpan<TS> s = image.GetReadOnlyRowSpan(r);
                Span<float> d = _corr.GetRowSpan(y + r).Slice(x * ccn);
                int ch = ccn == 1 ? 0 : k;
                for (int c = 0; c < bsw; c++)
                {
                    float v = float.CreateTruncating(s[c]);
                    int di = (c * ccn) + ch;
                    d[di] = accumulate ? v + d[di] : v;
                }
            }
        }

        private static int RoundHalfEven(double v) => (int)Math.Round(v, MidpointRounding.ToEven);
    }
}
