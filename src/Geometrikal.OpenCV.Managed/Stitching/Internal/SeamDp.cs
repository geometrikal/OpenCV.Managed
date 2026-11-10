// This implementation is ported from the OpenCV dynamic-programming seam finder.
//
// Original sources:
//   modules/stitching/src/seam_finders.cpp (DpSeamFinder)
//   modules/stitching/include/opencv2/stitching/detail/seam_finders.hpp (DpSeamFinder)
//   modules/stitching/include/opencv2/stitching/detail/util_inl.hpp (normL2, sqr)
//   modules/core/include/opencv2/core/operations.hpp (partition)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// State and steps of OpenCV <c>DpSeamFinder</c> for one call of <c>find</c>.
/// </summary>
/// <typeparam name="T"><c>byte</c> or <c>float</c>.</typeparam>
internal sealed class SeamDp<T> : IDisposable
    where T : unmanaged
{
    private const int First = 1;
    private const int Second = 2;
    private const int Inters = 4;
    private const int IntersFirst = Inters | First;
    private const int IntersSecond = Inters | Second;

    /// <summary><c>normL2(Point3f(255, 255, 255), Point3f(0, 0, 0))</c> of util_inl.hpp (a squared norm).</summary>
    private const float BadRegionCost = 195075f;

    private readonly DpSeamCostFunction _costFunc;
    private readonly Mat<T>[] _images;
    private readonly (Mat<float> Dx, Mat<float> Dy)?[] _gradients;
    private readonly List<IDisposable> _pooled = [];

    private Point _unionTl;
    private int _unionW;
    private int _unionH;
    private byte[] _mask1 = [];
    private byte[] _mask2 = [];
    private byte[] _contour1Mask = [];
    private byte[] _contour2Mask = [];
    private int[] _labels = [];
    private int _ncomps;
    private readonly List<int> _states = [];
    private readonly List<Point> _tls = [];
    private readonly List<Point> _brs = [];
    private readonly List<List<Point>> _contours = [];
    private readonly SortedSet<(int, int)> _edges = [];
    private readonly Stack<int> _fillStack = new();

    private Mat<float>? _gradx1;
    private Mat<float>? _grady1;
    private Mat<float>? _gradx2;
    private Mat<float>? _grady2;

    public SeamDp(DpSeamCostFunction costFunc, Mat<T>[] images)
    {
        _costFunc = costFunc;
        _images = images;
        _gradients = new (Mat<float>, Mat<float>)?[images.Length];
    }

    public void Dispose()
    {
        foreach (IDisposable pooled in _pooled)
        {
            pooled.Dispose();
        }

        _pooled.Clear();
        Array.Clear(_gradients);
        _gradx1 = _grady1 = _gradx2 = _grady2 = null;
        ReturnBuffer(ref _mask1);
        ReturnBuffer(ref _mask2);
        ReturnBuffer(ref _contour1Mask);
        ReturnBuffer(ref _contour2Mask);
        ReturnBuffer(ref _labels);
    }

    /// <summary>OpenCV <c>DpSeamFinder::find</c> after the input checks.</summary>
    public void Find(IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks)
    {
        int n = _images.Length;
        if (n == 0)
        {
            return;
        }

        var pairs = new List<(int First, int Second)>();
        for (int i = 0; i + 1 < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                pairs.Add((i, j));
            }
        }

        (int First, int Second)[] sorted = pairs.ToArray();
        var keys = new int[sorted.Length];
        for (int k = 0; k < sorted.Length; k++)
        {
            keys[k] = PairDistance(sorted[k], corners);
        }

        SeamPairSort.Sort(sorted, keys);
        Array.Reverse(sorted);

        foreach ((int i0, int i1) in sorted)
        {
            Process(i0, i1, corners[i0], corners[i1], masks[i0], masks[i1]);
        }
    }

    private int PairDistance((int First, int Second) p, IReadOnlyList<Point> corners)
    {
        Mat<T> a = _images[p.First], b = _images[p.Second];
        int x1 = corners[p.First].X + (a.Width / 2), y1 = corners[p.First].Y + (a.Height / 2);
        int x2 = corners[p.Second].X + (b.Width / 2), y2 = corners[p.Second].Y + (b.Height / 2);
        int dx = x1 - x2, dy = y1 - y2;
        return unchecked((dx * dx) + (dy * dy));
    }

    private void Process(int i1, int i2, Point tl1, Point tl2, Mat<byte> mask1, Mat<byte> mask2)
    {
        Mat<T> image1 = _images[i1], image2 = _images[i2];
        int itlx = Math.Max(tl1.X, tl2.X), itly = Math.Max(tl1.Y, tl2.Y);
        int ibrx = Math.Min(tl1.X + image1.Width, tl2.X + image2.Width);
        int ibry = Math.Min(tl1.Y + image1.Height, tl2.Y + image2.Height);
        if (itlx >= ibrx || itly >= ibry)
        {
            return;
        }

        _unionTl = new Point(Math.Min(tl1.X, tl2.X), Math.Min(tl1.Y, tl2.Y));
        int ubrx = Math.Max(tl1.X + image1.Width, tl2.X + image2.Width);
        int ubry = Math.Max(tl1.Y + image1.Height, tl2.Y + image2.Height);
        _unionW = ubrx - _unionTl.X;
        _unionH = ubry - _unionTl.Y;
        int size = _unionW * _unionH;

        _mask1 = Rent(_mask1, size);
        _mask2 = Rent(_mask2, size);
        _contour1Mask = Rent(_contour1Mask, size);
        _contour2Mask = Rent(_contour2Mask, size);
        if (_labels.Length < size)
        {
            ReturnBuffer(ref _labels);
            _labels = ArrayPool<int>.Shared.Rent(size);
        }

        PlaceMask(mask1, tl1, _mask1);
        PlaceMask(mask2, tl2, _mask2);
        ContourMask(_mask1, _contour1Mask);
        ContourMask(_mask2, _contour2Mask);

        FindComponents();
        FindEdges();
        ResolveConflicts(i1, i2, tl1, tl2, mask1, mask2);
    }

    private static byte[] Rent(byte[] buffer, int size)
    {
        if (buffer.Length < size)
        {
            ReturnBuffer(ref buffer);
            buffer = ArrayPool<byte>.Shared.Rent(size);
        }

        Array.Clear(buffer, 0, size);
        return buffer;
    }

    private static void ReturnBuffer<TElem>(ref TElem[] buffer)
    {
        if (buffer.Length > 0)
        {
            ArrayPool<TElem>.Shared.Return(buffer);
        }

        buffer = [];
    }

    private static TElem[] RentCleared<TElem>(int length)
    {
        TElem[] buffer = ArrayPool<TElem>.Shared.Rent(length);
        Array.Clear(buffer, 0, length);
        return buffer;
    }

    private void PlaceMask(Mat<byte> mask, Point tl, byte[] dst)
    {
        int ox = tl.X - _unionTl.X, oy = tl.Y - _unionTl.Y;
        for (int y = 0; y < mask.Height; y++)
        {
            mask.GetReadOnlyRowSpan(y).CopyTo(dst.AsSpan(((oy + y) * _unionW) + ox, mask.Width));
        }
    }

    private void ContourMask(byte[] mask, byte[] contour)
    {
        int w = _unionW, h = _unionH;
        for (int y = 0; y < h; y++)
        {
            int r = y * w;
            if (y == 0 || y == h - 1 || w < Vector<byte>.Count + 2)
            {
                for (int x = 0; x < w; x++)
                {
                    int i = r + x;
                    if (mask[i] != 0 &&
                        ((x == 0 || mask[i - 1] == 0) || (x == w - 1 || mask[i + 1] == 0) ||
                         (y == 0 || mask[i - w] == 0) || (y == h - 1 || mask[i + w] == 0)))
                    {
                        contour[i] = 255;
                    }
                }

                continue;
            }

            contour[r] = mask[r] != 0 ? (byte)255 : (byte)0;
            contour[r + w - 1] = mask[r + w - 1] != 0 ? (byte)255 : (byte)0;
            ReadOnlySpan<byte> m = mask;
            Span<byte> c = contour;
            int last = w - 1 - Vector<byte>.Count;
            for (int x = 1; ; x += Vector<byte>.Count)
            {
                if (x > last)
                {
                    x = last;
                }

                int i = r + x;
                Vector<byte> centre = Vector.Equals(new Vector<byte>(m.Slice(i)), Vector<byte>.Zero);
                Vector<byte> edge = Vector.Equals(new Vector<byte>(m.Slice(i - 1)), Vector<byte>.Zero)
                    | Vector.Equals(new Vector<byte>(m.Slice(i + 1)), Vector<byte>.Zero)
                    | Vector.Equals(new Vector<byte>(m.Slice(i - w)), Vector<byte>.Zero)
                    | Vector.Equals(new Vector<byte>(m.Slice(i + w)), Vector<byte>.Zero);
                Vector.AndNot(edge, centre).CopyTo(c.Slice(i));
                if (x == last)
                {
                    break;
                }
            }
        }
    }

    private bool IsContourPixel(int x, int y, int l)
    {
        int w = _unionW, i = (y * w) + x;
        int[] lab = _labels;
        return (x == 0 || lab[i - 1] != l) || (x == w - 1 || lab[i + 1] != l) ||
               (y == 0 || lab[i - w] != l) || (y == _unionH - 1 || lab[i + w] != l);
    }

    private void FindComponents()
    {
        int w = _unionW, h = _unionH;
        int[] lab = _labels;
        _ncomps = 0;
        _states.Clear();
        _tls.Clear();
        _brs.Clear();
        _contours.Clear();

        for (int i = 0; i < w * h; i++)
        {
            bool a = _mask1[i] != 0, b = _mask2[i] != 0;
            lab[i] = a && b ? int.MaxValue : a ? int.MaxValue - 1 : b ? int.MaxValue - 2 : 0;
        }

        // The OpenCV raster loop gives each component its bounding box and its contour pixels in
        // raster order; the fill computes the box and a second raster pass collects the contours.
        for (int y = 0; y < h; y++)
        {
            int r = y * w;
            for (int x = 0; x < w; x++)
            {
                int v = lab[r + x];
                if (v >= int.MaxValue - 2)
                {
                    _states.Add(v == int.MaxValue ? Inters : v == int.MaxValue - 1 ? First : Second);
                    Rect box = FloodFillEqual(lab, w, h, x, y, ++_ncomps);
                    _tls.Add(new Point(box.X, box.Y));
                    _brs.Add(new Point(box.X + box.Width, box.Y + box.Height));
                    _contours.Add([]);
                }
            }
        }

        for (int y = 0; y < h; y++)
        {
            int r = y * w;
            for (int x = 0; x < w; x++)
            {
                int l = lab[r + x];
                if (l != 0 && IsContourPixel(x, y, l))
                {
                    _contours[l - 1].Add(new Point(x, y));
                }
            }
        }
    }

    /// <summary>
    /// OpenCV <c>floodFill(img, seed, newVal)</c> on a 32S image with zero differences and
    /// 4-connectivity: the 4-connected region of pixels equal to the seed value gets
    /// <paramref name="newVal"/>. The fill order differs from OpenCV; the region does not.
    /// </summary>
    /// <returns>The bounding box of the region.</returns>
    private Rect FloodFillEqual(int[] img, int w, int h, int x, int y, int newVal)
    {
        int old = img[(y * w) + x];
        if (old == newVal)
        {
            return new Rect(x, y, 1, 1);
        }

        int minX = x, maxX = x, minY = y, maxY = y;
        Stack<int> stack = _fillStack;
        stack.Clear();
        stack.Push((y * w) + x);
        while (stack.Count > 0)
        {
            int seed = stack.Pop();
            int sy = seed / w;
            int r = sy * w;
            int sx = seed - r;
            if (img[seed] != old)
            {
                continue;
            }

            int xl = sx, xr = sx;
            while (xl > 0 && img[r + xl - 1] == old)
            {
                xl--;
            }

            while (xr < w - 1 && img[r + xr + 1] == old)
            {
                xr++;
            }

            img.AsSpan(r + xl, xr - xl + 1).Fill(newVal);
            minX = Math.Min(minX, xl);
            maxX = Math.Max(maxX, xr);
            minY = Math.Min(minY, sy);
            maxY = Math.Max(maxY, sy);

            if (sy > 0)
            {
                PushSpans(img, r - w, xl, xr, old, stack);
            }

            if (sy < h - 1)
            {
                PushSpans(img, r + w, xl, xr, old, stack);
            }
        }

        return new Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private static void PushSpans(int[] img, int r, int xl, int xr, int old, Stack<int> stack)
    {
        bool inSpan = false;
        for (int x = xl; x <= xr; x++)
        {
            if (img[r + x] == old)
            {
                if (!inSpan)
                {
                    stack.Push(r + x);
                    inSpan = true;
                }
            }
            else
            {
                inSpan = false;
            }
        }
    }

    private void FindEdges()
    {
        int w = _unionW, h = _unionH;
        int[] lab = _labels;
        _edges.Clear();
        for (int ci = 0; ci < _ncomps; ci++)
        {
            int l = ci + 1;
            foreach (Point p in _contours[ci])
            {
                int i = (p.Y * w) + p.X;
                if (p.X > 0)
                {
                    AddEdge(ci, lab[i - 1], l);
                }

                if (p.Y > 0)
                {
                    AddEdge(ci, lab[i - w], l);
                }

                if (p.X < w - 1)
                {
                    AddEdge(ci, lab[i + 1], l);
                }

                if (p.Y < h - 1)
                {
                    AddEdge(ci, lab[i + w], l);
                }
            }
        }
    }

    private void AddEdge(int ci, int nb, int l)
    {
        if (nb != 0 && nb != l)
        {
            _edges.Add((ci, nb - 1));
            _edges.Add((nb - 1, ci));
        }
    }

    private void ResolveConflicts(int i1, int i2, Point tl1, Point tl2, Mat<byte> mask1, Mat<byte> mask2)
    {
        if (_costFunc == DpSeamCostFunction.ColorGrad)
        {
            ComputeGradients(i1, i2);
        }

        Mat<T> image1 = _images[i1], image2 = _images[i2];
        int w = _unionW;
        int[] lab = _labels;
        bool hasConflict = true;
        while (hasConflict)
        {
            int c1 = 0, c2 = 0;
            hasConflict = false;
            foreach ((int a, int b) in _edges)
            {
                c1 = a;
                c2 = b;
                if ((_states[c1] & Inters) != 0 && (_states[c1] & ~Inters) != _states[c2])
                {
                    hasConflict = true;
                    break;
                }
            }

            if (!hasConflict)
            {
                break;
            }

            int l1 = c1 + 1, l2 = c2 + 1;
            if (HasOnlyOneNeighbor(c1))
            {
                for (int y = _tls[c1].Y; y < _brs[c1].Y; y++)
                {
                    for (int x = _tls[c1].X; x < _brs[c1].X; x++)
                    {
                        if (lab[(y * w) + x] == l1)
                        {
                            lab[(y * w) + x] = l2;
                        }
                    }
                }

                _states[c1] = _states[c2] == First ? Second : First;
            }
            else
            {
                if (GetSeamTips(c1, c2, out Point p1, out Point p2) &&
                    EstimateSeam(image1, image2, tl1, tl2, c1, p1, p2, out List<Point> seam, out bool isHorizontal))
                {
                    UpdateLabelsUsingSeam(c1, c2, seam, isHorizontal);
                }

                _states[c1] = _states[c2] == First ? IntersSecond : IntersFirst;
            }

            UpdateComponent(c1, l1);
            UpdateComponent(c2, l2);

            _edges.Remove((c1, c2));
            _edges.Remove((c2, c1));
        }

        int dx1 = _unionTl.X - tl1.X, dy1 = _unionTl.Y - tl1.Y;
        int dx2 = _unionTl.X - tl2.X, dy2 = _unionTl.Y - tl2.Y;

        for (int y = 0; y < mask2.Height; y++)
        {
            Span<byte> row = mask2.GetRowSpan(y);
            int ly = y - dy2;
            int m1y = ly + dy1;
            for (int x = 0; x < mask2.Width; x++)
            {
                int l = lab[(ly * w) + x - dx2];
                if (l > 0 && (_states[l - 1] & First) != 0 && MaskAt(mask1, m1y, x - dx2 + dx1))
                {
                    row[x] = 0;
                }
            }
        }

        for (int y = 0; y < mask1.Height; y++)
        {
            Span<byte> row = mask1.GetRowSpan(y);
            int ly = y - dy1;
            int m2y = ly + dy2;
            for (int x = 0; x < mask1.Width; x++)
            {
                int l = lab[(ly * w) + x - dx1];
                if (l > 0 && (_states[l - 1] & Second) != 0 && MaskAt(mask2, m2y, x - dx1 + dx2))
                {
                    row[x] = 0;
                }
            }
        }
    }

    private static bool MaskAt(Mat<byte> mask, int y, int x)
        => (uint)y < (uint)mask.Height && (uint)x < (uint)mask.Width && mask.At(y, x) != 0;

    private void UpdateComponent(int c, int l)
    {
        int x0 = _tls[c].X, x1 = _brs[c].X;
        int y0 = _tls[c].Y, y1 = _brs[c].Y;
        int tlx = int.MaxValue, tly = int.MaxValue, brx = int.MinValue, bry = int.MinValue;
        List<Point> contour = _contours[c];
        contour.Clear();
        int w = _unionW;
        for (int y = y0; y < y1; y++)
        {
            for (int x = x0; x < x1; x++)
            {
                if (_labels[(y * w) + x] == l)
                {
                    tlx = Math.Min(tlx, x);
                    tly = Math.Min(tly, y);
                    brx = Math.Max(brx, x + 1);
                    bry = Math.Max(bry, y + 1);
                    if (IsContourPixel(x, y, l))
                    {
                        contour.Add(new Point(x, y));
                    }
                }
            }
        }

        _tls[c] = new Point(tlx, tly);
        _brs[c] = new Point(brx, bry);
    }

    private void ComputeGradients(int i1, int i2)
    {
        (_gradx1, _grady1) = Gradient(i1);
        (_gradx2, _grady2) = Gradient(i2);
    }

    private (Mat<float> Dx, Mat<float> Dy) Gradient(int i)
    {
        if (_gradients[i] is { } g)
        {
            return g;
        }

        Mat<T> image = _images[i];
        PooledMat<T> gray = PooledMat<T>.Rent(image.Width, image.Height, 1);
        _pooled.Add(gray);
        ToGray(image, gray.Mat);
        PooledMat<float> dx = PooledMat<float>.Rent(image.Width, image.Height, 1);
        _pooled.Add(dx);
        PooledMat<float> dy = PooledMat<float>.Rent(image.Width, image.Height, 1);
        _pooled.Add(dy);
        Derivatives.SobelInto(gray.Mat, 1, 0, 3, 1, 0, BorderType.Reflect101, dx.Mat);
        Derivatives.SobelInto(gray.Mat, 0, 1, 3, 1, 0, BorderType.Reflect101, dy.Mat);
        var result = (dx.Mat, dy.Mat);
        _gradients[i] = result;
        return result;
    }

    /// <summary><c>cvtColor</c> with <c>COLOR_BGR2GRAY</c> or <c>COLOR_BGRA2GRAY</c> into <paramref name="gray"/>.</summary>
    private static void ToGray(Mat<T> image, Mat<T> gray)
    {
        int scn = image.Channels;
        if (typeof(T) == typeof(float))
        {
            var row = new GrayFloatRow(scn, 0);
            var src = (Mat<float>)(object)image;
            var dst = (Mat<float>)(object)gray;
            for (int y = 0; y < src.Height; y++)
            {
                row.Apply(src.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), src.Width);
            }
        }
        else
        {
            var row = new GrayIntRow<T>(scn, 0);
            for (int y = 0; y < image.Height; y++)
            {
                row.Apply(image.GetReadOnlyRowSpan(y), gray.GetRowSpan(y), image.Width);
            }
        }
    }

    private bool HasOnlyOneNeighbor(int comp)
        => _edges.GetViewBetween((comp, int.MinValue), (comp, int.MaxValue)).Count == 1;

    private bool CloseToContour(int y, int x, byte[] contourMask)
    {
        const int Rad = 2;
        for (int dy = -Rad; dy <= Rad; dy++)
        {
            if (y + dy >= 0 && y + dy < _unionH)
            {
                for (int dx = -Rad; dx <= Rad; dx++)
                {
                    if (x + dx >= 0 && x + dx < _unionW && contourMask[((y + dy) * _unionW) + x + dx] != 0)
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private bool GetSeamTips(int comp1, int comp2, out Point p1, out Point p2)
    {
        p1 = p2 = default;
        if ((_states[comp1] & Inters) == 0)
        {
            throw new InvalidOperationException("DpSeamFinder: the component is not an intersection.");
        }

        var specialPoints = new List<Point>();
        int l2 = comp2 + 1;
        int w = _unionW, h = _unionH;
        int[] lab = _labels;
        foreach (Point pt in _contours[comp1])
        {
            int x = pt.X, y = pt.Y, i = (y * w) + x;
            if (CloseToContour(y, x, _contour1Mask) &&
                CloseToContour(y, x, _contour2Mask) &&
                ((x > 0 && lab[i - 1] == l2) ||
                 (y > 0 && lab[i - w] == l2) ||
                 (x < w - 1 && lab[i + 1] == l2) ||
                 (y < h - 1 && lab[i + w] == l2)))
            {
                specialPoints.Add(pt);
            }
        }

        if (specialPoints.Count < 2)
        {
            return false;
        }

        int[] labels = Partition(specialPoints, 10);
        int nlabels = labels.Max() + 1;
        if (nlabels < 2)
        {
            return false;
        }

        var sumX = new int[nlabels];
        var sumY = new int[nlabels];
        var points = new List<Point>[nlabels];
        for (int i = 0; i < nlabels; i++)
        {
            points[i] = [];
        }

        for (int i = 0; i < specialPoints.Count; i++)
        {
            sumX[labels[i]] += specialPoints[i].X;
            sumY[labels[i]] += specialPoints[i].Y;
            points[labels[i]].Add(specialPoints[i]);
        }

        int idx0 = -1, idx1 = -1;
        double maxDist = -double.MaxValue;
        for (int i = 0; i < nlabels - 1; i++)
        {
            for (int j = i + 1; j < nlabels; j++)
            {
                double size1 = points[i].Count, size2 = points[j].Count;
                double cx1 = ScalarConverter.CvRound(sumX[i] / size1), cy1 = ScalarConverter.CvRound(sumY[i] / size1);
                double cx2 = ScalarConverter.CvRound(sumX[j] / size2), cy2 = ScalarConverter.CvRound(sumY[j] / size2);
                double dist = ((cx1 - cx2) * (cx1 - cx2)) + ((cy1 - cy2) * (cy1 - cy2));
                if (dist > maxDist)
                {
                    maxDist = dist;
                    idx0 = i;
                    idx1 = j;
                }
            }
        }

        p1 = Closest(points[idx0], sumX[idx0], sumY[idx0]);
        p2 = Closest(points[idx1], sumX[idx1], sumY[idx1]);
        return true;
    }

    private static Point Closest(List<Point> pts, int sumX, int sumY)
    {
        double size = pts.Count;
        double cx = ScalarConverter.CvRound(sumX / size);
        double cy = ScalarConverter.CvRound(sumY / size);
        int closest = pts.Count;
        double minDist = double.MaxValue;
        for (int j = 0; j < pts.Count; j++)
        {
            double dist = ((pts[j].X - cx) * (pts[j].X - cx)) + ((pts[j].Y - cy) * (pts[j].Y - cy));
            if (dist < minDist)
            {
                minDist = dist;
                closest = j;
            }
        }

        return pts[closest];
    }

    /// <summary>OpenCV <c>cv::partition</c> with the <c>ClosePoints</c> predicate.</summary>
    internal static int[] Partition(List<Point> vec, int minDist)
    {
        int n = vec.Count;
        var parent = new int[n];
        var rank = new int[n];
        Array.Fill(parent, -1);
        int minDist2 = minDist * minDist;

        for (int i = 0; i < n; i++)
        {
            int root = i;
            while (parent[root] >= 0)
            {
                root = parent[root];
            }

            for (int j = 0; j < n; j++)
            {
                int ddx = vec[i].X - vec[j].X, ddy = vec[i].Y - vec[j].Y;
                if (i == j || !((ddx * ddx) + (ddy * ddy) < minDist2))
                {
                    continue;
                }

                int root2 = j;
                while (parent[root2] >= 0)
                {
                    root2 = parent[root2];
                }

                if (root2 != root)
                {
                    int rk = rank[root], rk2 = rank[root2];
                    if (rk > rk2)
                    {
                        parent[root2] = root;
                    }
                    else
                    {
                        parent[root] = root2;
                        rank[root2] += rk == rk2 ? 1 : 0;
                        root = root2;
                    }

                    int k = j, p;
                    while ((p = parent[k]) >= 0)
                    {
                        parent[k] = root;
                        k = p;
                    }

                    k = i;
                    while ((p = parent[k]) >= 0)
                    {
                        parent[k] = root;
                        k = p;
                    }
                }
            }
        }

        var labels = new int[n];
        int nclasses = 0;
        for (int i = 0; i < n; i++)
        {
            int root = i;
            while (parent[root] >= 0)
            {
                root = parent[root];
            }

            if (rank[root] >= 0)
            {
                rank[root] = ~nclasses++;
            }

            labels[i] = ~rank[root];
        }

        return labels;
    }

    private float Diff(Mat<T> image1, int y1, int x1, Mat<T> image2, int y2, int x2)
    {
        int cn = image1.Channels;
        if (typeof(T) == typeof(byte))
        {
            ref byte r1 = ref Unsafe.As<T, byte>(ref image1.AtElement(y1, cn * x1));
            ref byte r2 = ref Unsafe.As<T, byte>(ref image2.AtElement(y2, cn * x2));
            int d0 = r1 - r2;
            int d1 = Unsafe.Add(ref r1, 1) - Unsafe.Add(ref r2, 1);
            int d2 = Unsafe.Add(ref r1, 2) - Unsafe.Add(ref r2, 2);
            return (d0 * d0) + (d1 * d1) + (d2 * d2);
        }
        else
        {
            ref float r1 = ref Unsafe.As<T, float>(ref image1.AtElement(y1, cn * x1));
            ref float r2 = ref Unsafe.As<T, float>(ref image2.AtElement(y2, cn * x2));
            float d0 = r1 - r2;
            float d1 = Unsafe.Add(ref r1, 1) - Unsafe.Add(ref r2, 1);
            float d2 = Unsafe.Add(ref r1, 2) - Unsafe.Add(ref r2, 2);
            return (d0 * d0) + (d1 * d1) + (d2 * d2);
        }
    }

    private void ComputeCosts(
        Mat<T> image1, Mat<T> image2, Point tl1, Point tl2, int comp, Rect roi, float[] costV, float[] costH)
    {
        int l = comp + 1;
        int w = _unionW;
        int[] lab = _labels;
        int dx1 = _unionTl.X - tl1.X, dy1 = _unionTl.Y - tl1.Y;
        int dx2 = _unionTl.X - tl2.X, dy2 = _unionTl.Y - tl2.Y;
        bool grad = _costFunc == DpSeamCostFunction.ColorGrad;
        int vw = roi.Width + 1;

        for (int y = roi.Y; y < roi.Y + roi.Height; y++)
        {
            for (int x = roi.X; x < roi.X + roi.Width + 1; x++)
            {
                float c;
                if (x > 0 && x < w && lab[(y * w) + x] == l && lab[(y * w) + x - 1] == l)
                {
                    float costColor = (Diff(image1, y + dy1, x + dx1 - 1, image2, y + dy2, x + dx2) +
                                       Diff(image1, y + dy1, x + dx1, image2, y + dy2, x + dx2 - 1)) / 2;
                    if (grad)
                    {
                        float costGrad = MathF.Abs(_gradx1!.At(y + dy1, x + dx1)) + MathF.Abs(_gradx1.At(y + dy1, x + dx1 - 1)) +
                                         MathF.Abs(_gradx2!.At(y + dy2, x + dx2)) + MathF.Abs(_gradx2.At(y + dy2, x + dx2 - 1)) + 1f;
                        c = costColor / costGrad;
                    }
                    else
                    {
                        c = costColor;
                    }
                }
                else
                {
                    c = BadRegionCost;
                }

                costV[((y - roi.Y) * vw) + x - roi.X] = c;
            }
        }

        for (int y = roi.Y; y < roi.Y + roi.Height + 1; y++)
        {
            for (int x = roi.X; x < roi.X + roi.Width; x++)
            {
                float c;
                if (y > 0 && y < _unionH && lab[(y * w) + x] == l && lab[((y - 1) * w) + x] == l)
                {
                    float costColor = (Diff(image1, y + dy1 - 1, x + dx1, image2, y + dy2, x + dx2) +
                                       Diff(image1, y + dy1, x + dx1, image2, y + dy2 - 1, x + dx2)) / 2;
                    if (grad)
                    {
                        float costGrad = MathF.Abs(_grady1!.At(y + dy1, x + dx1)) + MathF.Abs(_grady1.At(y + dy1 - 1, x + dx1)) +
                                         MathF.Abs(_grady2!.At(y + dy2, x + dx2)) + MathF.Abs(_grady2.At(y + dy2 - 1, x + dx2)) + 1f;
                        c = costColor / costGrad;
                    }
                    else
                    {
                        c = costColor;
                    }
                }
                else
                {
                    c = BadRegionCost;
                }

                costH[((y - roi.Y) * roi.Width) + x - roi.X] = c;
            }
        }
    }

    private static bool PairLess(float a, int ai, float b, int bi) => a < b || (!(b < a) && ai < bi);

    private bool EstimateSeam(
        Mat<T> image1, Mat<T> image2, Point tl1, Point tl2, int comp, Point p1, Point p2, out List<Point> seam, out bool isHorizontal)
    {
        if ((_states[comp] & Inters) == 0)
        {
            throw new InvalidOperationException("DpSeamFinder: the component is not an intersection.");
        }

        Rect roi = RectFromPoints(_tls[comp], _brs[comp]);
        int rw = roi.Width, rh = roi.Height;
        float[] costV = RentCleared<float>(rh * (rw + 1));
        float[] costH = RentCleared<float>((rh + 1) * rw);
        byte[] control = RentCleared<byte>(rw * rh);
        byte[] reachable = RentCleared<byte>(rw * rh);
        float[] cost = RentCleared<float>(rw * rh);
        try
        {
            return EstimateSeamCore(image1, image2, tl1, tl2, comp, p1, p2, roi, costV, costH, control, reachable, cost, out seam, out isHorizontal);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(costV);
            ArrayPool<float>.Shared.Return(costH);
            ArrayPool<byte>.Shared.Return(control);
            ArrayPool<byte>.Shared.Return(reachable);
            ArrayPool<float>.Shared.Return(cost);
        }
    }

    private bool EstimateSeamCore(
        Mat<T> image1,
        Mat<T> image2,
        Point tl1,
        Point tl2,
        int comp,
        Point p1,
        Point p2,
        Rect roi,
        float[] costV,
        float[] costH,
        byte[] control,
        byte[] reachable,
        float[] cost,
        out List<Point> seam,
        out bool isHorizontal)
    {
        int rw = roi.Width, rh = roi.Height;
        ComputeCosts(image1, image2, tl1, tl2, comp, roi, costV, costH);
        int vw = rw + 1;

        var src = new Point(p1.X - roi.X, p1.Y - roi.Y);
        var dst = new Point(p2.X - roi.X, p2.Y - roi.Y);
        int l = comp + 1;
        int w = _unionW;
        int[] lab = _labels;

        bool swapped = false;
        isHorizontal = Math.Abs(dst.X - src.X) > Math.Abs(dst.Y - src.Y);
        if (isHorizontal)
        {
            if (src.X > dst.X)
            {
                (src, dst) = (dst, src);
                swapped = true;
            }
        }
        else if (src.Y > dst.Y)
        {
            swapped = true;
            (src, dst) = (dst, src);
        }

        reachable[(src.Y * rw) + src.X] = 1;
        cost[(src.Y * rw) + src.X] = 0f;

        Span<float> stepCost = stackalloc float[3];
        Span<int> stepId = stackalloc int[3];
        if (isHorizontal)
        {
            for (int x = src.X + 1; x <= dst.X; x++)
            {
                for (int y = 0; y < rh; y++)
                {
                    int n = 0;
                    if (lab[((y + roi.Y) * w) + x + roi.X] == l)
                    {
                        int i = (y * rw) + x - 1;
                        if (reachable[i] != 0)
                        {
                            stepCost[n] = cost[i] + costH[(y * rw) + x - 1];
                            stepId[n++] = 1;
                        }

                        if (y > 0 && reachable[i - rw] != 0)
                        {
                            stepCost[n] = cost[i - rw] + costH[((y - 1) * rw) + x - 1] + costV[((y - 1) * vw) + x];
                            stepId[n++] = 2;
                        }

                        if (y < rh - 1 && reachable[i + rw] != 0)
                        {
                            stepCost[n] = cost[i + rw] + costH[((y + 1) * rw) + x - 1] + costV[(y * vw) + x];
                            stepId[n++] = 3;
                        }
                    }

                    if (n != 0)
                    {
                        Store(stepCost, stepId, n, cost, control, reachable, (y * rw) + x);
                    }
                }
            }
        }
        else
        {
            for (int y = src.Y + 1; y <= dst.Y; y++)
            {
                for (int x = 0; x < rw; x++)
                {
                    int n = 0;
                    if (lab[((y + roi.Y) * w) + x + roi.X] == l)
                    {
                        int i = ((y - 1) * rw) + x;
                        if (reachable[i] != 0)
                        {
                            stepCost[n] = cost[i] + costV[((y - 1) * vw) + x];
                            stepId[n++] = 1;
                        }

                        if (x > 0 && reachable[i - 1] != 0)
                        {
                            stepCost[n] = cost[i - 1] + costV[((y - 1) * vw) + x - 1] + costH[(y * rw) + x - 1];
                            stepId[n++] = 2;
                        }

                        if (x < rw - 1 && reachable[i + 1] != 0)
                        {
                            stepCost[n] = cost[i + 1] + costV[((y - 1) * vw) + x + 1] + costH[(y * rw) + x];
                            stepId[n++] = 3;
                        }
                    }

                    if (n != 0)
                    {
                        Store(stepCost, stepId, n, cost, control, reachable, (y * rw) + x);
                    }
                }
            }
        }

        seam = [];
        if (reachable[(dst.Y * rw) + dst.X] == 0)
        {
            return false;
        }

        Point p = dst;
        seam.Add(new Point(p.X + roi.X, p.Y + roi.Y));
        if (isHorizontal)
        {
            for (; p.X != src.X; seam.Add(new Point(p.X + roi.X, p.Y + roi.Y)))
            {
                byte c = control[(p.Y * rw) + p.X];
                if (c == 2)
                {
                    p = p with { Y = p.Y - 1 };
                }
                else if (c == 3)
                {
                    p = p with { Y = p.Y + 1 };
                }

                p = p with { X = p.X - 1 };
            }
        }
        else
        {
            for (; p.Y != src.Y; seam.Add(new Point(p.X + roi.X, p.Y + roi.Y)))
            {
                byte c = control[(p.Y * rw) + p.X];
                if (c == 2)
                {
                    p = p with { X = p.X - 1 };
                }
                else if (c == 3)
                {
                    p = p with { X = p.X + 1 };
                }

                p = p with { Y = p.Y - 1 };
            }
        }

        if (!swapped)
        {
            seam.Reverse();
        }

        if (seam[0] != p1 || seam[^1] != p2)
        {
            throw new InvalidOperationException("DpSeamFinder: the seam does not join the seam tips.");
        }

        return true;
    }

    private static void Store(Span<float> stepCost, Span<int> stepId, int n, float[] cost, byte[] control, byte[] reachable, int i)
    {
        int best = 0;
        for (int k = 1; k < n; k++)
        {
            if (PairLess(stepCost[k], stepId[k], stepCost[best], stepId[best]))
            {
                best = k;
            }
        }

        cost[i] = stepCost[best];
        control[i] = (byte)stepId[best];
        reachable[i] = 255;
    }

    /// <summary>OpenCV <c>Rect(Point pt1, Point pt2)</c>.</summary>
    private static Rect RectFromPoints(Point a, Point b)
    {
        int x = Math.Min(a.X, b.X), y = Math.Min(a.Y, b.Y);
        return new Rect(x, y, Math.Max(a.X, b.X) - x, Math.Max(a.Y, b.Y) - y);
    }

    private void UpdateLabelsUsingSeam(int comp1, int comp2, List<Point> seam, bool isHorizontalSeam)
    {
        Point tl = _tls[comp1];
        int mw = _brs[comp1].X - tl.X, mh = _brs[comp1].Y - tl.Y;
        int[] mask = RentCleared<int>(mw * mh);
        try
        {
            UpdateLabelsUsingSeamCore(comp1, comp2, seam, isHorizontalSeam, mask, mw, mh);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(mask);
        }
    }

    private void UpdateLabelsUsingSeamCore(int comp1, int comp2, List<Point> seam, bool isHorizontalSeam, int[] mask, int mw, int mh)
    {
        Point tl = _tls[comp1];
        List<Point> contour = _contours[comp1];
        foreach (Point p in contour)
        {
            mask[((p.Y - tl.Y) * mw) + p.X - tl.X] = 255;
        }

        foreach (Point p in seam)
        {
            mask[((p.Y - tl.Y) * mw) + p.X - tl.X] = 255;
        }

        int l1 = comp1 + 1, l2 = comp2 + 1;
        int w = _unionW, h = _unionH;
        int[] lab = _labels;
        int ncomps = 0;
        for (int y = 0; y < mh; y++)
        {
            for (int x = 0; x < mw; x++)
            {
                if (mask[(y * mw) + x] == 0 && lab[((y + tl.Y) * w) + x + tl.X] == l1)
                {
                    FloodFillEqual(mask, mw, mh, x, y, ++ncomps);
                }
            }
        }

        ReadOnlySpan<int> ndx = [-1, +1, 0, 0, -1, +1, -1, +1];
        ReadOnlySpan<int> ndy = [0, 0, -1, +1, -1, -1, +1, +1];
        foreach (Point p in contour)
        {
            int x = p.X - tl.X, y = p.Y - tl.Y;
            bool ok = false;
            for (int j = 0; j < 8; j++)
            {
                int c = x + ndx[j], r = y + ndy[j];
                if (c >= 0 && c < mw && r >= 0 && r < mh)
                {
                    int v = mask[(r * mw) + c];
                    if (v != 0 && v != 255)
                    {
                        ok = true;
                        mask[(y * mw) + x] = v;
                    }
                }
            }

            if (!ok)
            {
                mask[(y * mw) + x] = 0;
            }
        }

        foreach (Point p in seam)
        {
            int x = p.X - tl.X, y = p.Y - tl.Y;
            int i = (y * mw) + x;
            if (isHorizontalSeam)
            {
                mask[i] = y < mh - 1 && mask[i + mw] != 0 && mask[i + mw] != 255 ? mask[i + mw] : 0;
            }
            else
            {
                mask[i] = x < mw - 1 && mask[i + 1] != 0 && mask[i + 1] != 255 ? mask[i + 1] : 0;
            }
        }

        // Index 0 and indexes above ncomps are map keys that exist only after an increment.
        int keyCount = Math.Max(ncomps + 1, 256);
        var connect2 = new int[keyCount];
        var connectOther = new int[keyCount];
        foreach (Point p in contour)
        {
            int x = p.X, y = p.Y, i = (y * w) + x;
            int m = mask[((y - tl.Y) * mw) + x - tl.X];
            if ((x > 0 && lab[i - 1] == l2) ||
                (y > 0 && lab[i - w] == l2) ||
                (x < w - 1 && lab[i + 1] == l2) ||
                (y < h - 1 && lab[i + w] == l2))
            {
                connect2[m]++;
            }

            if ((x > 0 && lab[i - 1] != l1 && lab[i - 1] != l2) ||
                (y > 0 && lab[i - w] != l1 && lab[i - w] != l2) ||
                (x < w - 1 && lab[i + 1] != l1 && lab[i + 1] != l2) ||
                (y < h - 1 && lab[i + w] != l1 && lab[i + w] != l2))
            {
                connectOther[m]++;
            }
        }

        var isAdjComp = new int[ncomps + 1];
        double len = contour.Count;
        for (int k = 1; k <= ncomps; k++)
        {
            isAdjComp[k] = connect2[k] / len > 0.05 && connectOther[k] / len < 0.1 ? 1 : 0;
        }

        for (int y = 0; y < mh; y++)
        {
            for (int x = 0; x < mw; x++)
            {
                int m = mask[(y * mw) + x];
                if (m != 0 && m < isAdjComp.Length && isAdjComp[m] != 0)
                {
                    lab[((y + tl.Y) * w) + x + tl.X] = l2;
                }
            }
        }
    }
}
