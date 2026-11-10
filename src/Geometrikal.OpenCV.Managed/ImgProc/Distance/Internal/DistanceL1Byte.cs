// This implementation reproduces the OpenCV 8-bit city-block distance transform.
//
// Original source:
//   modules/imgproc/src/distransform.cpp (distanceATS_L1_8u, (C) 2006 by Jay Stavinzky)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/distransform.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// The result of <c>distanceATS_L1_8u</c>: <c>min(city-block distance to the nearest zero pixel, 255)</c>,
/// computed in two separable passes.
/// </summary>
/// <remarks>
/// <para>The OpenCV raster passes with <c>lut[x] = min(x + 1, 255)</c> and the 4-neighbourhood give
/// the exact city-block distance saturated at 255: <c>lut</c> is monotone, so
/// <c>lut[min(a, b)] = min(lut[a], lut[b])</c>, and saturation commutes with every step.</para>
/// <para>The city-block distance is separable: pass 1 computes the saturated distance to the
/// nearest zero pixel in the same row; pass 2 runs <c>v = min(v + 1, d)</c> down and up each
/// column. Rows are independent in pass 1 and columns are independent in pass 2, so both passes
/// run in parallel stripes, and pass 2 runs on <c>Vector&lt;byte&gt;</c> lanes. Pass 1 reads each
/// source element before it writes the same destination element and reads no other source row,
/// so the source may be the destination.</para>
/// </remarks>
internal static class DistanceL1Byte
{
    public static void Transform(Mat<byte> source, Mat<byte> destination)
    {
        int width = source.Width, height = source.Height;
        var src = new Plane(source.DangerousGetArray(), source.DangerousGetOffset(), source.Stride);
        var dst = new Plane(destination.DangerousGetArray(), destination.DangerousGetOffset(), destination.Stride);

        ParallelRows.For(height, (long)width * 4, (y0, y1) => RowPass(src, dst, width, y0, y1));

        int vc = Vector<byte>.Count;
        if (Vector.IsHardwareAccelerated && width >= vc)
        {
            int chunks = width / vc;
            ParallelRows.For(chunks, (long)height * 4, (c0, c1) => ColumnPass(dst, width, height, c0, c1, c1 == chunks));
        }
        else
        {
            ColumnPassScalar(dst, width, height);
        }
    }

    private static void RowPass(Plane src, Plane dst, int width, int y0, int y1)
    {
        for (int y = y0; y < y1; y++)
        {
            ref byte s = ref src.Row(y);
            ref byte d = ref dst.Row(y);
            ZeroTerms(ref s, ref d, width);

            int a = d;
            for (int x = 1; x < width; x++)
            {
                ref byte dx = ref Unsafe.Add(ref d, x);
                a = Math.Min(a + 1, dx);
                dx = (byte)a;
            }

            a = Unsafe.Add(ref d, width - 1);
            for (int x = width - 2; x >= 0; x--)
            {
                ref byte dx = ref Unsafe.Add(ref d, x);
                a = Math.Min(a + 1, dx);
                dx = (byte)a;
            }
        }
    }

    /// <summary>
    /// <c>d[x] = s[x] == 0 ? 0 : 255</c>. Each source element is read before the same destination
    /// element is written; the loop uses a scalar tail, not an overlapping final vector, because
    /// the source may be the destination.
    /// </summary>
    private static void ZeroTerms(ref byte s, ref byte d, int width)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<byte>.Count;
            for (; x <= width - vc; x += vc)
            {
                Vector.OnesComplement(Vector.Equals(Vector.LoadUnsafe(ref s, (nuint)x), Vector<byte>.Zero)).StoreUnsafe(ref d, (nuint)x);
            }
        }

        for (; x < width; x++)
        {
            Unsafe.Add(ref d, x) = Unsafe.Add(ref s, x) == 0 ? (byte)0 : (byte)255;
        }
    }

    /// <summary>
    /// Down and up passes over the columns of chunks <c>[c0, c1)</c> of <c>Vector&lt;byte&gt;.Count</c>
    /// columns. The stripe with the last chunk also covers the remaining columns with a final vector
    /// that overlaps its own last chunk; the update is idempotent.
    /// </summary>
    private static void ColumnPass(Plane dst, int width, int height, int c0, int c1, bool last)
    {
        int vc = Vector<byte>.Count;
        int tail = last && width % vc != 0 ? width - vc : -1;
        for (int y = 1; y < height; y++)
        {
            ref byte prev = ref dst.Row(y - 1);
            ref byte cur = ref dst.Row(y);
            for (int c = c0; c < c1; c++)
            {
                Step(ref prev, ref cur, c * vc);
            }

            if (tail >= 0)
            {
                Step(ref prev, ref cur, tail);
            }
        }

        for (int y = height - 2; y >= 0; y--)
        {
            ref byte prev = ref dst.Row(y + 1);
            ref byte cur = ref dst.Row(y);
            for (int c = c0; c < c1; c++)
            {
                Step(ref prev, ref cur, c * vc);
            }

            if (tail >= 0)
            {
                Step(ref prev, ref cur, tail);
            }
        }
    }

    /// <summary><c>cur = min(min(prev, 254) + 1, cur)</c> for one vector of columns.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Step(ref byte prev, ref byte cur, int x)
    {
        Vector<byte> v = Vector.Min(Vector.LoadUnsafe(ref prev, (nuint)x), new Vector<byte>(254)) + Vector<byte>.One;
        Vector.Min(v, Vector.LoadUnsafe(ref cur, (nuint)x)).StoreUnsafe(ref cur, (nuint)x);
    }

    private static void ColumnPassScalar(Plane dst, int width, int height)
    {
        for (int y = 1; y < height; y++)
        {
            ref byte prev = ref dst.Row(y - 1);
            ref byte cur = ref dst.Row(y);
            for (int x = 0; x < width; x++)
            {
                ref byte d = ref Unsafe.Add(ref cur, x);
                d = (byte)Math.Min(Unsafe.Add(ref prev, x) + 1, d);
            }
        }

        for (int y = height - 2; y >= 0; y--)
        {
            ref byte prev = ref dst.Row(y + 1);
            ref byte cur = ref dst.Row(y);
            for (int x = 0; x < width; x++)
            {
                ref byte d = ref Unsafe.Add(ref cur, x);
                d = (byte)Math.Min(Unsafe.Add(ref prev, x) + 1, d);
            }
        }
    }

    private readonly record struct Plane(byte[] Data, int Offset, int Stride)
    {
        public ref byte Row(int y) => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + ((nint)y * Stride));
    }
}
