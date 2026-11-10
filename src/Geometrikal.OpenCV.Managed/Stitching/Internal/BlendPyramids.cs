// This implementation is ported from the OpenCV stitching blenders.
//
// Original source:
//   modules/stitching/src/blenders.cpp (createLaplacePyr, restoreImageFromLaplacePyr)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Laplacian pyramid build and collapse of the multi-band blender.</summary>
internal static class BlendPyramids
{
    /// <summary>OpenCV <c>createLaplacePyr</c> for an 8-bit source: 8-bit Gaussian levels, <c>short</c> differences.</summary>
    /// <param name="img">Source image.</param>
    /// <param name="levels">Number of Laplacian levels.</param>
    /// <param name="pool">Receives the pooled matrices of the result and the temporaries; null allocates the result.</param>
    public static Mat<short>[] LaplaceFromBytes(Mat<byte> img, int levels, List<IDisposable>? pool = null)
    {
        var pyr = new Mat<short>[levels + 1];
        if (levels == 0)
        {
            pyr[0] = Conversions.ConvertTo<byte, short>(img);
            return pyr;
        }

        var temps = new List<IDisposable>();
        try
        {
            Mat<byte> current = img;
            Mat<byte> downNext = Down(img, temps);
            for (int i = 1; i < levels; i++)
            {
                Mat<byte> lvlDown = Down(downNext, temps);
                Mat<byte> lvlUp = New<byte>(current.Width, current.Height, current.Channels, temps);
                Pyramids.PyrUp(downNext, lvlUp);
                pyr[i - 1] = Difference(current, lvlUp, pool);
                current = downNext;
                downNext = lvlDown;
            }

            Mat<byte> up = New<byte>(current.Width, current.Height, current.Channels, temps);
            Pyramids.PyrUp(downNext, up);
            pyr[levels - 1] = Difference(current, up, pool);
            pyr[levels] = Conversions.ConvertTo<byte, short>(downNext);
            return pyr;
        }
        finally
        {
            foreach (IDisposable t in temps)
            {
                t.Dispose();
            }
        }
    }

    /// <summary>OpenCV <c>createLaplacePyr</c> for a <c>short</c> source; <paramref name="img"/> becomes level 0 and is modified.</summary>
    /// <param name="img">Source image; it becomes level 0.</param>
    /// <param name="levels">Number of Laplacian levels.</param>
    /// <param name="pool">Receives the pooled matrices of the result and the temporaries; null allocates the result.</param>
    public static Mat<short>[] LaplaceFromShorts(Mat<short> img, int levels, List<IDisposable>? pool = null)
    {
        var pyr = new Mat<short>[levels + 1];
        pyr[0] = img;
        for (int i = 0; i < levels; i++)
        {
            pyr[i + 1] = Down(pyr[i], pool);
        }

        var temps = new List<IDisposable>();
        try
        {
            for (int i = 0; i < levels; i++)
            {
                Mat<short> tmp = New<short>(pyr[i].Width, pyr[i].Height, pyr[i].Channels, temps);
                Pyramids.PyrUp(pyr[i + 1], tmp);
                Arithmetic.Subtract(pyr[i], tmp, pyr[i]);
            }
        }
        finally
        {
            foreach (IDisposable t in temps)
            {
                t.Dispose();
            }
        }

        return pyr;
    }

    /// <summary>OpenCV <c>restoreImageFromLaplacePyr</c>: saturating <c>short</c> sums from the top level down.</summary>
    public static void Restore(Mat<short>[] pyr)
    {
        for (int i = pyr.Length - 1; i > 0; i--)
        {
            using PooledMat<short> tmp = PooledMat<short>.Rent(pyr[i - 1].Width, pyr[i - 1].Height, pyr[i - 1].Channels);
            Pyramids.PyrUp(pyr[i], tmp.Mat);
            Arithmetic.Add(tmp.Mat, pyr[i - 1], pyr[i - 1]);
        }
    }

    /// <summary><c>pyrDown</c> with the default destination size into a pooled or new matrix.</summary>
    public static Mat<T> Down<T>(Mat<T> src, List<IDisposable>? pool)
        where T : unmanaged
    {
        Mat<T> dst = New<T>((src.Width + 1) / 2, (src.Height + 1) / 2, src.Channels, pool);
        Pyramids.PyrDown(src, dst);
        return dst;
    }

    /// <summary>A matrix whose elements the caller overwrites: pooled when <paramref name="pool"/> is set.</summary>
    public static Mat<T> New<T>(int width, int height, int channels, List<IDisposable>? pool)
        where T : unmanaged
    {
        if (pool is null)
        {
            return new Mat<T>(width, height, channels, clear: false);
        }

        PooledMat<T> rented = PooledMat<T>.Rent(width, height, channels);
        pool.Add(rented);
        return rented.Mat;
    }

    /// <summary><c>a − b</c> as <c>short</c> (exact for 8-bit operands).</summary>
    private static Mat<short> Difference(Mat<byte> a, Mat<byte> b, List<IDisposable>? pool)
    {
        Mat<short> d = New<short>(a.Width, a.Height, a.Channels, pool);
        ParallelRows.For(a.Height, a.RowLength / Vector<byte>.Count + 1, (y0, y1) =>
        {
            for (int y = y0; y < y1; y++)
            {
                ReadOnlySpan<byte> ra = a.GetReadOnlyRowSpan(y);
                ReadOnlySpan<byte> rb = b.GetReadOnlyRowSpan(y);
                Span<short> rd = d.GetRowSpan(y);
                int n = Vector<byte>.Count;
                int x = 0;
                for (; x <= rd.Length - n; x += n)
                {
                    Vector.Widen(new Vector<byte>(ra.Slice(x)), out Vector<ushort> a0, out Vector<ushort> a1);
                    Vector.Widen(new Vector<byte>(rb.Slice(x)), out Vector<ushort> b0, out Vector<ushort> b1);
                    Vector.AsVectorInt16(a0 - b0).CopyTo(rd.Slice(x));
                    Vector.AsVectorInt16(a1 - b1).CopyTo(rd.Slice(x + (n / 2)));
                }

                for (; x < rd.Length; x++)
                {
                    rd[x] = (short)(ra[x] - rb[x]);
                }
            }
        });
        return d;
    }
}
