// This implementation is ported from the OpenCV spatial gradient.
//
// Original sources:
//   modules/imgproc/src/spatialgradient.cpp (spatialGradient, spatialGradientKernel)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Fused 3×3 Sobel gradient: one pass reads three source rows and writes both
/// <c>dx = (v22 − v00) + (v02 − v20) + 2(v12 − v10)</c> and
/// <c>dy = (v22 − v00) + (v20 − v02) + 2(v21 − v01)</c>. The result equals
/// <c>Sobel(src, CV_16S, 1, 0, 3)</c> and <c>Sobel(src, CV_16S, 0, 1, 3)</c> on an isolated image.
/// Values stay in [−1020, 1020]; the arithmetic is exact in 16 bits.
/// </summary>
internal static class SobelGradient3x3
{
    /// <summary>Computes the gradient of the whole image into <paramref name="dx"/> and <paramref name="dy"/>.</summary>
    public static void Apply(Mat<byte> source, Mat<short> dx, Mat<short> dy, BorderType border)
    {
        long work = (long)source.RowLength * 12 / Vector<short>.Count;
        if (!ParallelRows.IsSerial(source.Height, work))
        {
            PageTouch.Rows(dx);
            PageTouch.Rows(dy);
        }

        ParallelRows.For(source.Height, work, (y0, y1) => ApplyRows(source, dx, dy, border, y0, y1));
    }

    private static void ApplyRows(Mat<byte> source, Mat<short> dx, Mat<short> dy, BorderType border, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            Row(source, y, border, dx.GetRowSpan(y), dy.GetRowSpan(y));
        }
    }

    [ThreadStatic]
    private static short[]? t_columns;

    /// <summary>
    /// Computes one gradient row (all channels). <paramref name="border"/> is
    /// <see cref="BorderType.Replicate"/> or <see cref="BorderType.Reflect101"/>.
    /// </summary>
    /// <remarks>
    /// The vector path first forms the column sums <c>S = a + 2b + c</c> and differences
    /// <c>D = c − a</c> of the three rows, then <c>dx = S[x + 1] − S[x − 1]</c> and
    /// <c>dy = D[x − 1] + 2D[x] + D[x + 1]</c>. The int16 values equal the direct formula.
    /// </remarks>
    public static void Row(Mat<byte> source, int y, BorderType border, Span<short> dx, Span<short> dy)
    {
        int height = source.Height;
        int width = source.Width;
        int cn = source.Channels;
        int n = width * cn;
        ref byte a = ref MemoryMarshal.GetReference(source.GetReadOnlyRowSpan(Borders.BorderInterpolate(y - 1, height, border)));
        ref byte b = ref MemoryMarshal.GetReference(source.GetReadOnlyRowSpan(y));
        ref byte c = ref MemoryMarshal.GetReference(source.GetReadOnlyRowSpan(Borders.BorderInterpolate(y + 1, height, border)));
        ref short gx = ref MemoryMarshal.GetReference(dx);
        ref short gy = ref MemoryMarshal.GetReference(dy);

        int vb = Vector<byte>.Count;
        if (!Vector.IsHardwareAccelerated || n < vb)
        {
            for (int i = 0; i < n; i++)
            {
                ScalarElement(ref a, ref b, ref c, ref gx, ref gy, i, width, cn, border);
            }

            return;
        }

        int span = n + (2 * cn);
        short[] columns = t_columns is { } cached && cached.Length >= 2 * span ? cached : (t_columns = new short[2 * span]);
        ref short s = ref columns[cn];
        ref short d = ref columns[span + cn];
        int vs = Vector<short>.Count;
        for (int e = 0; e < n; e += vb)
        {
            // The final block overlaps the previous one; it rewrites equal values.
            e = Math.Min(e, n - vb);
            Widen(Vector.LoadUnsafe(ref a, (nuint)e), out Vector<short> a0, out Vector<short> a1);
            Widen(Vector.LoadUnsafe(ref b, (nuint)e), out Vector<short> b0, out Vector<short> b1);
            Widen(Vector.LoadUnsafe(ref c, (nuint)e), out Vector<short> c0, out Vector<short> c1);
            (a0 + (b0 * 2) + c0).StoreUnsafe(ref s, (nuint)e);
            (a1 + (b1 * 2) + c1).StoreUnsafe(ref s, (nuint)(e + vs));
            (c0 - a0).StoreUnsafe(ref d, (nuint)e);
            (c1 - a1).StoreUnsafe(ref d, (nuint)(e + vs));
        }

        int left = Borders.BorderInterpolate(-1, width, border) * cn;
        int right = Borders.BorderInterpolate(width, width, border) * cn;
        for (int k = 0; k < cn; k++)
        {
            Unsafe.Add(ref s, k - cn) = Unsafe.Add(ref s, left + k);
            Unsafe.Add(ref d, k - cn) = Unsafe.Add(ref d, left + k);
            Unsafe.Add(ref s, n + k) = Unsafe.Add(ref s, right + k);
            Unsafe.Add(ref d, n + k) = Unsafe.Add(ref d, right + k);
        }

        ref short sl = ref columns[0];
        ref short sr = ref columns[2 * cn];
        ref short dl = ref columns[span];
        ref short dr = ref columns[span + (2 * cn)];
        for (int e = 0; e < n; e += vs)
        {
            // The final block overlaps the previous one; it rewrites equal values.
            e = Math.Min(e, n - vs);
            (Vector.LoadUnsafe(ref sr, (nuint)e) - Vector.LoadUnsafe(ref sl, (nuint)e)).StoreUnsafe(ref gx, (nuint)e);
            (Vector.LoadUnsafe(ref dl, (nuint)e) + (Vector.LoadUnsafe(ref d, (nuint)e) * 2) + Vector.LoadUnsafe(ref dr, (nuint)e))
                .StoreUnsafe(ref gy, (nuint)e);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Widen(Vector<byte> v, out Vector<short> lo, out Vector<short> hi)
    {
        Vector.Widen(v, out Vector<ushort> l, out Vector<ushort> h);
        lo = Vector.AsVectorInt16(l);
        hi = Vector.AsVectorInt16(h);
    }

    private static void ScalarElement(
        ref byte a, ref byte b, ref byte c, ref short gx, ref short gy, int i, int width, int cn, BorderType border)
    {
        int x = i / cn;
        int ch = i - (x * cn);
        int l = (Borders.BorderInterpolate(x - 1, width, border) * cn) + ch;
        int r = (Borders.BorderInterpolate(x + 1, width, border) * cn) + ch;
        int sL = Unsafe.Add(ref a, l) + (2 * Unsafe.Add(ref b, l)) + Unsafe.Add(ref c, l);
        int sR = Unsafe.Add(ref a, r) + (2 * Unsafe.Add(ref b, r)) + Unsafe.Add(ref c, r);
        int dL = Unsafe.Add(ref c, l) - Unsafe.Add(ref a, l);
        int dC = Unsafe.Add(ref c, i) - Unsafe.Add(ref a, i);
        int dR = Unsafe.Add(ref c, r) - Unsafe.Add(ref a, r);
        Unsafe.Add(ref gx, i) = (short)(sR - sL);
        Unsafe.Add(ref gy, i) = (short)(dL + (2 * dC) + dR);
    }
}
