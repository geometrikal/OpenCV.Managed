// This implementation is ported from the OpenCV inpainting.
//
// Original source:
//   modules/photo/src/inpaint.cpp (cv::inpaint, icvTeleaInpaintFMM, icvNSInpaintFMM, icvCalcFMM)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Photo;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Inpainting method, as OpenCV <c>INPAINT_NS</c> and <c>INPAINT_TELEA</c>.</summary>
public enum InpaintMethod
{
    /// <summary>Navier-Stokes based method (<c>INPAINT_NS</c>).</summary>
    NavierStokes = 0,

    /// <summary>Fast marching method of Telea (<c>INPAINT_TELEA</c>).</summary>
    Telea = 1,
}

/// <summary>Image inpainting, as OpenCV <c>inpaint</c>.</summary>
public static class Inpainting
{
    private const byte Known = 0;
    private const byte Band = 1;
    private const byte Inside = 2;
    private const byte Change = 3;

    /// <summary>
    /// Restores the masked region of an image from its neighbourhood, as OpenCV <c>inpaint</c>.
    /// </summary>
    /// <param name="source">
    /// <c>byte</c> image with 1 or 3 channels, or a single-channel <c>ushort</c> or <c>float</c> image.
    /// </param>
    /// <param name="mask">Single-channel mask of the source size; non-zero pixels are restored.</param>
    /// <param name="inpaintRadius">Radius of the neighbourhood; <c>cvRound</c> of it is clamped to [1, 100].</param>
    /// <param name="method">Inpainting method.</param>
    /// <returns>A new image. Pixels outside the mask are copies of the source.</returns>
    /// <remarks>
    /// Pixels are restored in fast-marching order: increasing distance from the mask boundary, with
    /// equal distances in insertion order. The arithmetic follows OpenCV 5.0 in float and double
    /// steps. Integer outputs round <c>value + 0.5</c> half to even (Telea) or round the value
    /// (Navier-Stokes); float Telea outputs are not offset.
    /// </remarks>
    public static Mat<T> Inpaint<T>(Mat<T> source, Mat<byte> mask, double inpaintRadius, InpaintMethod method)
        where T : unmanaged =>
        Inpaint(source, mask, inpaintRadius, method, legacy455: false);

    // legacy455 reproduces cv2 4.5.5: neighbour values at (km, lm) and Telea outputs of (float)(value + 0.5f).
    internal static Mat<T> Inpaint<T>(Mat<T> source, Mat<byte> mask, double inpaintRadius, InpaintMethod method, bool legacy455)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(mask);
        bool supported = (typeof(T) == typeof(byte) && source.Channels is 1 or 3)
            || ((typeof(T) == typeof(ushort) || typeof(T) == typeof(float)) && source.Channels == 1);
        if (!supported)
        {
            throw new NotSupportedException("8-bit, 16-bit unsigned or 32-bit float 1-channel and 8-bit 3-channel images are supported.");
        }

        if (mask.Width != source.Width || mask.Height != source.Height || mask.Channels != 1)
        {
            throw new ArgumentException("The mask must be single-channel with the source size.", nameof(mask));
        }

        if (method is not (InpaintMethod.Telea or InpaintMethod.NavierStokes))
        {
            throw new ArgumentOutOfRangeException(nameof(method));
        }

        int range = Core.Internal.ScalarConverter.CvRound(inpaintRadius);
        range = Math.Min(Math.Max(range, 1), 100);

        var state = new State<T>(source, mask);
        var heap = new Heap();
        heap.Add(state.BandFlags, state.Cols);

        if (method == InpaintMethod.Telea)
        {
            // The ring of width range outside the band gets negative distances from an outward march.
            var ring = Dilate(state.Mask, state.Rows, state.Cols, range, range);
            Subtract(ring, state.Mask);
            var outHeap = new Heap();
            outHeap.Add(state.BandFlags, state.Cols);
            Subtract(ring, state.BandFlags);
            SetBorder(ring, state.Rows, state.Cols);
            CalcFmm(ring, state.Times, state.Rows, state.Cols, outHeap, negate: true);
            Telea(state, heap, range, legacy455);
        }
        else
        {
            NavierStokes(state, heap, range, legacy455);
        }

        return state.Output;
    }

    private static float Min4(float a, float b, float c, float d)
    {
        a = Math.Min(a, b);
        c = Math.Min(c, d);
        return Math.Min(a, c);
    }

    private static float Solve(int i1, int j1, int i2, int j2, byte[] f, float[] t, int cols)
    {
        double a11 = t[(i1 * cols) + j1], a22 = t[(i2 * cols) + j2];
        double m12 = Math.Min(a11, a22);
        double sol;
        if (f[(i1 * cols) + j1] != Inside)
        {
            if (f[(i2 * cols) + j2] != Inside)
            {
                sol = Math.Abs(a11 - a22) >= 1.0 ? 1 + m12 : (a11 + a22 + Math.Sqrt(2 - ((a11 - a22) * (a11 - a22)))) * 0.5;
            }
            else
            {
                sol = 1 + a11;
            }
        }
        else
        {
            sol = f[(i2 * cols) + j2] != Inside ? 1 + a22 : 1 + m12;
        }

        return (float)sol;
    }

    private static float Distance(int i, int j, byte[] f, float[] t, int cols) =>
        Min4(Solve(i - 1, j, i, j - 1, f, t, cols), Solve(i + 1, j, i, j - 1, f, t, cols),
            Solve(i - 1, j, i, j + 1, f, t, cols), Solve(i + 1, j, i, j + 1, f, t, cols));

    private static (int I, int J) Neighbour(int ii, int jj, int q) => q switch
    {
        0 => (ii - 1, jj),
        1 => (ii, jj - 1),
        2 => (ii + 1, jj),
        _ => (ii, jj + 1),
    };

    private static void CalcFmm(byte[] f, float[] t, int rows, int cols, Heap heap, bool negate)
    {
        while (heap.TryPop(out int ii, out int jj))
        {
            f[(ii * cols) + jj] = negate ? Change : Known;
            for (int q = 0; q < 4; q++)
            {
                var (i, j) = Neighbour(ii, jj, q);
                if (i <= 0 || j <= 0 || i > rows || j > cols)
                {
                    continue;
                }

                if (f[(i * cols) + j] == Inside)
                {
                    float dist = Distance(i, j, f, t, cols);
                    t[(i * cols) + j] = dist;
                    f[(i * cols) + j] = Band;
                    heap.Push(i, j, dist);
                }
            }
        }

        if (negate)
        {
            for (int k = 0; k < f.Length; k++)
            {
                if (f[k] == Change)
                {
                    f[k] = Known;
                    t[k] = -t[k];
                }
            }
        }
    }

    // Telea gradient of the distance map at (i, j).
    private static (float X, float Y) DistanceGradient(byte[] f, float[] t, int cols, int i, int j)
    {
        int c = (i * cols) + j;
        float gx, gy;
        if (f[c + 1] != Inside)
        {
            gx = f[c - 1] != Inside ? (t[c + 1] - t[c - 1]) * 0.5f : t[c + 1] - t[c];
        }
        else
        {
            gx = f[c - 1] != Inside ? t[c] - t[c - 1] : 0;
        }

        if (f[c + cols] != Inside)
        {
            gy = f[c - cols] != Inside ? (t[c + cols] - t[c - cols]) * 0.5f : t[c + cols] - t[c];
        }
        else
        {
            gy = f[c - cols] != Inside ? t[c] - t[c - cols] : 0;
        }

        return (gx, gy);
    }

    private static void Telea<T>(State<T> s, Heap heap, int range, bool legacy)
        where T : unmanaged
    {
        byte[] f = s.Mask;
        float[] t = s.Times;
        int rows = s.Rows, cols = s.Cols, cn = s.Channels;
        var ia = new float[cn];
        var jx = new float[cn];
        var jy = new float[cn];
        var sw = new float[cn];
        while (heap.TryPop(out int ii, out int jj))
        {
            f[(ii * cols) + jj] = Known;
            for (int q = 0; q < 4; q++)
            {
                var (i, j) = Neighbour(ii, jj, q);
                if (i <= 0 || j <= 0 || i > rows - 1 || j > cols - 1 || f[(i * cols) + j] != Inside)
                {
                    continue;
                }

                float dist = Distance(i, j, f, t, cols);
                t[(i * cols) + j] = dist;
                var gradT = DistanceGradient(f, t, cols, i, j);
                Array.Clear(ia);
                Array.Clear(jx);
                Array.Clear(jy);
                Array.Fill(sw, 1.0e-20f);

                for (int k = i - range; k <= i + range; k++)
                {
                    int km = k - 1 + (k == 1 ? 1 : 0), kp = k - 1 - (k == rows - 2 ? 1 : 0);
                    for (int l = j - range; l <= j + range; l++)
                    {
                        int lm = l - 1 + (l == 1 ? 1 : 0), lp = l - 1 - (l == cols - 2 ? 1 : 0);
                        if (!(k > 0 && l > 0 && k < rows - 1 && l < cols - 1))
                        {
                            continue;
                        }

                        int kl = (k * cols) + l;
                        if (f[kl] == Inside || ((l - j) * (l - j)) + ((k - i) * (k - i)) > range * range)
                        {
                            continue;
                        }

                        float ry = i - k, rx = j - l;
                        float len = (rx * rx) + (ry * ry);
                        float dst = cn == 3 ? (float)(1.0 / (len * Math.Sqrt(len))) : (float)(1.0 / (len * MathF.Sqrt(len)));
                        float lev = (float)(1.0 / (1 + MathF.Abs(t[kl] - t[(i * cols) + j])));
                        float dir = (rx * gradT.X) + (ry * gradT.Y);
                        if (MathF.Abs(dir) <= 0.01)
                        {
                            dir = 0.000001f;
                        }

                        float w = MathF.Abs(dst * lev * dir);
                        for (int color = 0; color < cn; color++)
                        {
                            float gix, giy;
                            if (f[kl + 1] != Inside)
                            {
                                gix = f[kl - 1] != Inside ? (float)(s.Get(km, lp + 1, color) - s.Get(km, lm - 1, color)) * 2.0f
                                    : (float)(s.Get(km, lp + 1, color) - s.Get(km, lm, color));
                            }
                            else
                            {
                                gix = f[kl - 1] != Inside ? (float)(s.Get(km, lp, color) - s.Get(km, lm - 1, color)) : 0;
                            }

                            if (f[kl + cols] != Inside)
                            {
                                giy = f[kl - cols] != Inside ? (float)(s.Get(kp + 1, lm, color) - s.Get(km - 1, lm, color)) * 2.0f
                                    : (float)(s.Get(kp + 1, lm, color) - s.Get(km, lm, color));
                            }
                            else
                            {
                                giy = f[kl - cols] != Inside ? (float)(s.Get(kp, lm, color) - s.Get(km - 1, lm, color)) : 0;
                            }

                            ia[color] += w * (float)(legacy ? s.Get(km, lm, color) : s.Get(k - 1, l - 1, color));
                            jx[color] -= w * (gix * rx);
                            jy[color] -= w * (giy * ry);
                            sw[color] += w;
                        }
                    }
                }

                for (int color = 0; color < cn; color++)
                {
                    float sat = (ia[color] / sw[color]) + ((jx[color] + jy[color]) / (MathF.Sqrt((jx[color] * jx[color]) + (jy[color] * jy[color])) + 1.0e-20f));
                    if (legacy)
                    {
                        s.SetLegacyTelea(i - 1, j - 1, color, sat + 0.5f);
                    }
                    else
                    {
                        s.SetTelea(i - 1, j - 1, color, sat);
                    }
                }

                f[(i * cols) + j] = Band;
                heap.Push(i, j, dist);
            }
        }
    }

    private static void NavierStokes<T>(State<T> s, Heap heap, int range, bool legacy)
        where T : unmanaged
    {
        byte[] f = s.Mask;
        float[] t = s.Times;
        int rows = s.Rows, cols = s.Cols, cn = s.Channels;
        var ia = new float[cn];
        var sw = new float[cn];
        while (heap.TryPop(out int ii, out int jj))
        {
            f[(ii * cols) + jj] = Known;
            for (int q = 0; q < 4; q++)
            {
                var (i, j) = Neighbour(ii, jj, q);
                if (i <= 0 || j <= 0 || i > rows - 1 || j > cols - 1 || f[(i * cols) + j] != Inside)
                {
                    continue;
                }

                float dist = Distance(i, j, f, t, cols);
                t[(i * cols) + j] = dist;
                Array.Clear(ia);
                Array.Fill(sw, 1.0e-20f);
                for (int k = i - range; k <= i + range; k++)
                {
                    int km = k - 1 + (k == 1 ? 1 : 0), kp = k - 1 - (k == rows - 2 ? 1 : 0);
                    for (int l = j - range; l <= j + range; l++)
                    {
                        int lm = l - 1 + (l == 1 ? 1 : 0), lp = l - 1 - (l == cols - 2 ? 1 : 0);
                        if (!(k > 0 && l > 0 && k < rows - 1 && l < cols - 1))
                        {
                            continue;
                        }

                        int kl = (k * cols) + l;
                        if (f[kl] == Inside || ((l - j) * (l - j)) + ((k - i) * (k - i)) > range * range)
                        {
                            continue;
                        }

                        float ry = i - k, rx = j - l;
                        float len = (rx * rx) + (ry * ry);
                        float dst = 1 / ((len * len) + 1);
                        for (int color = 0; color < cn; color++)
                        {
                            float gix, giy;
                            if (f[kl + cols] != Inside)
                            {
                                gix = f[kl - cols] != Inside
                                    ? (float)Math.Abs(s.Get(kp + 1, lm, color) - s.Get(kp, lm, color)) + (float)Math.Abs(s.Get(kp, lm, color) - s.Get(km - 1, lm, color))
                                    : (float)Math.Abs(s.Get(kp + 1, lm, color) - s.Get(kp, lm, color)) * 2.0f;
                            }
                            else
                            {
                                gix = f[kl - cols] != Inside ? (float)Math.Abs(s.Get(kp, lm, color) - s.Get(km - 1, lm, color)) * 2.0f : 0;
                            }

                            if (f[kl + 1] != Inside)
                            {
                                giy = f[kl - 1] != Inside
                                    ? (float)Math.Abs(s.Get(km, lp + 1, color) - s.Get(km, lm, color)) + (float)Math.Abs(s.Get(km, lm, color) - s.Get(km, lm - 1, color))
                                    : (float)Math.Abs(s.Get(km, lp + 1, color) - s.Get(km, lm, color)) * 2.0f;
                            }
                            else
                            {
                                giy = f[kl - 1] != Inside ? (float)Math.Abs(s.Get(km, lm, color) - s.Get(km, lm - 1, color)) * 2.0f : 0;
                            }

                            gix = -gix;
                            float dir = (rx * gix) + (ry * giy);
                            if (MathF.Abs(dir) <= 0.01)
                            {
                                dir = 0.000001f;
                            }
                            else
                            {
                                dir = MathF.Abs(((rx * gix) + (ry * giy)) / MathF.Sqrt(len * ((gix * gix) + (giy * giy))));
                            }

                            float w = dst * dir;
                            ia[color] += w * (float)(legacy ? s.Get(km, lm, color) : s.Get(k - 1, l - 1, color));
                            sw[color] += w;
                        }
                    }
                }

                for (int color = 0; color < cn; color++)
                {
                    s.SetRounded(i - 1, j - 1, color, (double)ia[color] / sw[color]);
                }

                f[(i * cols) + j] = Band;
                heap.Push(i, j, dist);
            }
        }
    }

    private static byte[] Dilate(byte[] src, int rows, int cols, int rx, int ry)
    {
        var tmp = new byte[src.Length];
        var dst = new byte[src.Length];
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                byte m = 0;
                for (int dx = Math.Max(0, x - rx); dx <= Math.Min(cols - 1, x + rx); dx++)
                {
                    m = Math.Max(m, src[(y * cols) + dx]);
                }

                tmp[(y * cols) + x] = m;
            }
        }

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                byte m = 0;
                for (int dy = Math.Max(0, y - ry); dy <= Math.Min(rows - 1, y + ry); dy++)
                {
                    m = Math.Max(m, tmp[(dy * cols) + x]);
                }

                dst[(y * cols) + x] = m;
            }
        }

        return dst;
    }

    private static byte[] DilateCross(byte[] src, int rows, int cols)
    {
        var dst = new byte[src.Length];
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int c = (y * cols) + x;
                byte m = src[c];
                if (y > 0) m = Math.Max(m, src[c - cols]);
                if (y < rows - 1) m = Math.Max(m, src[c + cols]);
                if (x > 0) m = Math.Max(m, src[c - 1]);
                if (x < cols - 1) m = Math.Max(m, src[c + 1]);
                dst[c] = m;
            }
        }

        return dst;
    }

    private static void Subtract(byte[] a, byte[] b)
    {
        for (int k = 0; k < a.Length; k++)
        {
            a[k] = (byte)Math.Max(0, a[k] - b[k]);
        }
    }

    private static void SetBorder(byte[] m, int rows, int cols)
    {
        Array.Clear(m, 0, cols);
        Array.Clear(m, (rows - 1) * cols, cols);
        for (int i = 1; i < rows - 1; i++)
        {
            m[i * cols] = 0;
            m[(i * cols) + cols - 1] = 0;
        }
    }

    private sealed class State<T>
        where T : unmanaged
    {
        public State(Mat<T> source, Mat<byte> mask)
        {
            Rows = source.Height + 2;
            Cols = source.Width + 2;
            Channels = source.Channels;
            Width = source.Width;
            Data = source.ToArray();
            Mask = new byte[Rows * Cols];
            for (int y = 0; y < source.Height; y++)
            {
                var row = mask.GetReadOnlyRowSpan(y);
                for (int x = 0; x < source.Width; x++)
                {
                    if (row[x] != 0)
                    {
                        Mask[((y + 1) * Cols) + x + 1] = Inside;
                    }
                }
            }

            SetBorder(Mask, Rows, Cols);
            BandFlags = DilateCross(Mask, Rows, Cols);
            Subtract(BandFlags, Mask);
            SetBorder(BandFlags, Rows, Cols);
            Times = new float[Rows * Cols];
            Array.Fill(Times, 1.0e6f);
            for (int k = 0; k < BandFlags.Length; k++)
            {
                if (BandFlags[k] != 0)
                {
                    Times[k] = 0;
                }
            }
        }

        public int Rows { get; }

        public int Cols { get; }

        public int Channels { get; }

        public byte[] Mask { get; }

        public byte[] BandFlags { get; }

        public float[] Times { get; }

        public Mat<T> Output => new(Data, Width, Rows - 2, Channels);

        private int Width { get; }

        private T[] Data { get; }

        public double Get(int y, int x, int c)
        {
            T v = Data[(((y * Width) + x) * Channels) + c];
            if (typeof(T) == typeof(byte))
            {
                return System.Runtime.CompilerServices.Unsafe.As<T, byte>(ref v);
            }

            if (typeof(T) == typeof(ushort))
            {
                return System.Runtime.CompilerServices.Unsafe.As<T, ushort>(ref v);
            }

            return System.Runtime.CompilerServices.Unsafe.As<T, float>(ref v);
        }

        // OpenCV round_cast: integers round value + 0.5 half to even (double); float keeps the value.
        public void SetTelea(int y, int x, int c, float sat)
        {
            if (typeof(T) == typeof(float))
            {
                Store(y, x, c, sat);
                return;
            }

            SetRounded(y, x, c, sat + 0.5);
        }

        // cv2 4.5.5: saturate_cast of the float value; integers round half to even.
        public void SetLegacyTelea(int y, int x, int c, float sat)
        {
            if (typeof(T) == typeof(float))
            {
                Store(y, x, c, sat);
                return;
            }

            SetRounded(y, x, c, sat);
        }

        public void SetRounded(int y, int x, int c, double value)
        {
            if (typeof(T) == typeof(float))
            {
                Store(y, x, c, (float)value);
                return;
            }

            double max = typeof(T) == typeof(byte) ? byte.MaxValue : ushort.MaxValue;
            double r = Math.Round(value, MidpointRounding.ToEven);
            double clamped = double.IsNaN(r) ? 0 : Math.Clamp(r, 0, max);
            int iv = (int)clamped;
            ref T slot = ref Data[(((y * Width) + x) * Channels) + c];
            if (typeof(T) == typeof(byte))
            {
                System.Runtime.CompilerServices.Unsafe.As<T, byte>(ref slot) = (byte)iv;
            }
            else
            {
                System.Runtime.CompilerServices.Unsafe.As<T, ushort>(ref slot) = (ushort)iv;
            }

        }

        private void Store(int y, int x, int c, float v)
        {
            System.Runtime.CompilerServices.Unsafe.As<T, float>(ref Data[(((y * Width) + x) * Channels) + c]) = v;
        }
    }

    // OpenCV CvPriorityQueueFloat: smallest T first, equal T in insertion order.
    private sealed class Heap
    {
        private readonly PriorityQueue<(int I, int J), (float T, int Order)> _queue = new(Comparer<(float T, int Order)>.Create(
            (a, b) => a.T != b.T ? a.T.CompareTo(b.T) : a.Order.CompareTo(b.Order)));

        private int _next;

        public void Add(byte[] flags, int cols)
        {
            for (int k = 0; k < flags.Length; k++)
            {
                if (flags[k] != 0)
                {
                    Push(k / cols, k % cols, 0);
                }
            }
        }

        public void Push(int i, int j, float t) => _queue.Enqueue((i, j), (t, _next++));

        public bool TryPop(out int i, out int j)
        {
            if (_queue.TryDequeue(out var e, out _))
            {
                (i, j) = e;
                return true;
            }

            i = j = 0;
            return false;
        }
    }
}
