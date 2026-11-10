// This implementation is ported from the OpenCV chamfer distance transforms.
//
// Original source:
//   modules/imgproc/src/distransform.cpp (initTopBottom, distanceTransform_3x3,
//   distanceTransform_5x5, distanceTransformEx_5x5, getDistanceTransformMask)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/distransform.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Fixed-point chamfer weights: <c>cvRound(metric * 65536)</c> of the OpenCV float metrics.</summary>
internal readonly record struct ChamferWeights(uint Hv, uint Diag, uint Long);

/// <summary>
/// Two-pass chamfer distance transforms over a padded <c>uint</c> buffer, as OpenCV 5.0.
/// The border value is <c>uint.MaxValue - (largest weight)</c>, so no sum overflows.
/// </summary>
internal static class ChamferDistance
{
    private const float Scale = 1f / 65536;

    /// <summary>Returns the weights of <c>getDistanceTransformMask</c> for C, L1 or L2.</summary>
    public static ChamferWeights Weights(DistanceTypes type, bool mask5) => (type, mask5) switch
    {
        (DistanceTypes.C, false) => new(65536, 65536, 0),
        (DistanceTypes.L1, false) => new(65536, 131072, 0),
        (DistanceTypes.L2, false) => new(62587, 89738, 0),
        (DistanceTypes.C, true) => new(65536, 65536, 131072),
        (DistanceTypes.L1, true) => new(65536, 131072, 196608),
        _ => new(65536, 91750, 143976),
    };

    public static void Transform3x3(Mat<byte> source, Mat<float> destination, ChamferWeights w)
    {
        int width = source.Width, height = source.Height;
        int tw = width + 2;
        int length = checked(tw * (height + 2));
        uint distMax = uint.MaxValue - w.Diag;
        uint[] buffer = ArrayPool<uint>.Shared.Rent(length);
        try
        {
            ref uint temp = ref MemoryMarshal.GetArrayDataReference(buffer);
            new Span<uint>(buffer, 0, tw).Fill(distMax);
            new Span<uint>(buffer, length - tw, tw).Fill(distMax);
            ref byte src = ref MemoryMarshal.GetArrayDataReference(source.DangerousGetArray());
            int srcOffset = source.DangerousGetOffset();
            bool vector = Vector.IsHardwareAccelerated && width >= Vector<byte>.Count;

            for (int i = 0; i < height; i++)
            {
                ref uint cur = ref Unsafe.Add(ref temp, ((i + 1) * tw) + 1);
                Unsafe.Add(ref cur, -1) = distMax;
                Unsafe.Add(ref cur, width) = distMax;
                ref byte s = ref Unsafe.Add(ref src, srcOffset + ((nint)i * source.Stride));
                ref uint top = ref Unsafe.Subtract(ref cur, tw);
                if (vector)
                {
                    Candidates3Vector(ref s, ref top, ref cur, width, w, distMax);
                }
                else
                {
                    Candidates3Scalar(ref s, ref top, ref cur, width, w, distMax);
                }

                LeftRecursion(ref cur, width, w.Hv, distMax);
            }

            ref float dst = ref MemoryMarshal.GetArrayDataReference(destination.DangerousGetArray());
            int dstOffset = destination.DangerousGetOffset();
            for (int i = height - 1; i >= 0; i--)
            {
                ref uint cur = ref Unsafe.Add(ref temp, ((i + 1) * tw) + 1);
                ref uint bottom = ref Unsafe.Add(ref cur, tw);
                if (vector)
                {
                    Bottom3Vector(ref bottom, ref cur, width, w);
                }
                else
                {
                    Bottom3Scalar(ref bottom, ref cur, width, w);
                }

                RightRecursion(ref cur, width, w.Hv, distMax);
                StoreRow(ref cur, ref Unsafe.Add(ref dst, dstOffset + ((nint)i * destination.Stride)), width);
            }
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(buffer);
        }
    }

    public static void Transform5x5(Mat<byte> source, Mat<float> destination, ChamferWeights w)
    {
        int width = source.Width, height = source.Height;
        int tw = width + 4;
        int length = checked(tw * (height + 4));
        uint distMax = uint.MaxValue - w.Long;
        uint[] buffer = ArrayPool<uint>.Shared.Rent(length);
        try
        {
            ref uint temp = ref MemoryMarshal.GetArrayDataReference(buffer);
            new Span<uint>(buffer, 0, 2 * tw).Fill(distMax);
            new Span<uint>(buffer, length - (2 * tw), 2 * tw).Fill(distMax);
            ref byte src = ref MemoryMarshal.GetArrayDataReference(source.DangerousGetArray());
            int srcOffset = source.DangerousGetOffset();
            bool vector = Vector.IsHardwareAccelerated && width >= Vector<byte>.Count;

            for (int i = 0; i < height; i++)
            {
                ref uint cur = ref Unsafe.Add(ref temp, ((i + 2) * tw) + 2);
                Unsafe.Add(ref cur, -2) = distMax;
                Unsafe.Add(ref cur, -1) = distMax;
                Unsafe.Add(ref cur, width) = distMax;
                Unsafe.Add(ref cur, width + 1) = distMax;
                ref byte s = ref Unsafe.Add(ref src, srcOffset + ((nint)i * source.Stride));
                if (vector)
                {
                    Candidates5Vector(ref s, ref cur, tw, width, w, distMax);
                }
                else
                {
                    Candidates5Scalar(ref s, ref cur, tw, width, w, distMax);
                }

                LeftRecursion(ref cur, width, w.Hv, distMax);
            }

            ref float dst = ref MemoryMarshal.GetArrayDataReference(destination.DangerousGetArray());
            int dstOffset = destination.DangerousGetOffset();
            for (int i = height - 1; i >= 0; i--)
            {
                ref uint cur = ref Unsafe.Add(ref temp, ((i + 2) * tw) + 2);
                if (vector)
                {
                    Bottom5Vector(ref cur, tw, width, w);
                }
                else
                {
                    Bottom5Scalar(ref cur, tw, width, w);
                }

                RightRecursion(ref cur, width, w.Hv, distMax);
                StoreRow(ref cur, ref Unsafe.Add(ref dst, dstOffset + ((nint)i * destination.Stride)), width);
            }
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// <c>distanceTransformEx_5x5</c>. Neighbours are tested in the OpenCV order with a strict
    /// comparison, so the first nearest neighbour gives the label.
    /// </summary>
    public static void Transform5x5Labels(
        Mat<byte> source, Mat<float> destination, Mat<int> labels, ChamferWeights w, DistanceTransformLabelTypes labelType)
    {
        int width = source.Width, height = source.Height;
        int tw = width + 4;
        int length = checked(tw * (height + 4));
        uint distMax = uint.MaxValue - w.Long;
        uint hv = w.Hv, diag = w.Diag, lng = w.Long;
        uint[] tbuf = ArrayPool<uint>.Shared.Rent(length);
        int[] lbuf = ArrayPool<int>.Shared.Rent(length);
        try
        {
            new Span<uint>(tbuf, 0, 2 * tw).Fill(distMax);
            new Span<uint>(tbuf, length - (2 * tw), 2 * tw).Fill(distMax);
            InitLabels(source, new Mat<int>(lbuf, width, height, 1, tw, (2 * tw) + 2), labelType);

            ref uint temp = ref MemoryMarshal.GetArrayDataReference(tbuf);
            ref int lab = ref MemoryMarshal.GetArrayDataReference(lbuf);
            ref byte src = ref MemoryMarshal.GetArrayDataReference(source.DangerousGetArray());
            int srcOffset = source.DangerousGetOffset();
            int u1 = -tw, u2 = -2 * tw;
            bool vector = Vector.IsHardwareAccelerated && width >= Vector<byte>.Count;

            for (int i = 0; i < height; i++)
            {
                int row = ((i + 2) * tw) + 2;
                ref uint t = ref Unsafe.Add(ref temp, row);
                ref int l = ref Unsafe.Add(ref lab, row);
                Unsafe.Add(ref t, -2) = distMax;
                Unsafe.Add(ref t, -1) = distMax;
                Unsafe.Add(ref t, width) = distMax;
                Unsafe.Add(ref t, width + 1) = distMax;
                ref byte s = ref Unsafe.Add(ref src, srcOffset + ((nint)i * source.Stride));
                if (vector)
                {
                    LabelTopVector(ref s, ref t, ref l, tw, width, w, distMax);
                }
                else
                {
                    for (int j = 0; j < width; j++)
                    {
                        if (Unsafe.Add(ref s, j) == 0)
                        {
                            Unsafe.Add(ref t, j) = 0;
                            continue;
                        }

                        uint t0 = distMax;
                        int l0 = 0;
                        Relax(ref t, ref l, j + u2 - 1, lng, ref t0, ref l0);
                        Relax(ref t, ref l, j + u2 + 1, lng, ref t0, ref l0);
                        Relax(ref t, ref l, j + u1 - 2, lng, ref t0, ref l0);
                        Relax(ref t, ref l, j + u1 - 1, diag, ref t0, ref l0);
                        Relax(ref t, ref l, j + u1, hv, ref t0, ref l0);
                        Relax(ref t, ref l, j + u1 + 1, diag, ref t0, ref l0);
                        Relax(ref t, ref l, j + u1 + 2, lng, ref t0, ref l0);
                        Unsafe.Add(ref t, j) = t0;
                        Unsafe.Add(ref l, j) = l0;
                    }
                }

                // The left neighbour is the last candidate; zero pixels hold 0 and never change.
                uint left = distMax;
                for (int j = 0; j < width; j++)
                {
                    ref uint tj = ref Unsafe.Add(ref t, j);
                    uint c = left + hv;
                    if (tj > c)
                    {
                        tj = c;
                        Unsafe.Add(ref l, j) = Unsafe.Add(ref l, j - 1);
                    }

                    left = tj;
                }
            }

            ref float dst = ref MemoryMarshal.GetArrayDataReference(destination.DangerousGetArray());
            int dstOffset = destination.DangerousGetOffset();
            ref int outLabels = ref MemoryMarshal.GetArrayDataReference(labels.DangerousGetArray());
            int labelOffset = labels.DangerousGetOffset();
            int d1 = tw, d2 = 2 * tw;
            for (int i = height - 1; i >= 0; i--)
            {
                int row = ((i + 2) * tw) + 2;
                ref uint t = ref Unsafe.Add(ref temp, row);
                ref int l = ref Unsafe.Add(ref lab, row);
                ref float d = ref Unsafe.Add(ref dst, dstOffset + ((nint)i * destination.Stride));
                ref int lo = ref Unsafe.Add(ref outLabels, labelOffset + ((nint)i * labels.Stride));

                // OpenCV tests the candidates only when t0 > HV. Every candidate is at least HV, so
                // with the strict comparison the unconditional test gives the same distance and label.
                if (vector)
                {
                    LabelBottomVector(ref t, ref l, tw, width, w);
                }
                else
                {
                    for (int j = 0; j < width; j++)
                    {
                        uint t0 = Unsafe.Add(ref t, j);
                        int l0 = Unsafe.Add(ref l, j);
                        Relax(ref t, ref l, j + d2 + 1, lng, ref t0, ref l0);
                        Relax(ref t, ref l, j + d2 - 1, lng, ref t0, ref l0);
                        Relax(ref t, ref l, j + d1 + 2, lng, ref t0, ref l0);
                        Relax(ref t, ref l, j + d1 + 1, diag, ref t0, ref l0);
                        Relax(ref t, ref l, j + d1, hv, ref t0, ref l0);
                        Relax(ref t, ref l, j + d1 - 1, diag, ref t0, ref l0);
                        Relax(ref t, ref l, j + d1 - 2, lng, ref t0, ref l0);
                        Unsafe.Add(ref t, j) = t0;
                        Unsafe.Add(ref l, j) = l0;
                    }
                }

                uint right = distMax;
                for (int j = width - 1; j >= 0; j--)
                {
                    ref uint tj = ref Unsafe.Add(ref t, j);
                    uint c = right + hv;
                    if (tj > c)
                    {
                        tj = c;
                        Unsafe.Add(ref l, j) = Unsafe.Add(ref l, j + 1);
                    }

                    right = tj;
                }

                StoreRow(ref t, ref d, width);
                MemoryMarshal.CreateReadOnlySpan(ref l, width).CopyTo(MemoryMarshal.CreateSpan(ref lo, width));
            }
        }
        finally
        {
            ArrayPool<uint>.Shared.Return(tbuf);
            ArrayPool<int>.Shared.Return(lbuf);
        }
    }

    /// <summary>
    /// Forward labelled pass over the seven neighbours in the rows above, in the OpenCV order, one
    /// pixel per lane. Zero pixels get distance 0 and keep their label. The rows above are final, so
    /// the final group may overlap the previous one.
    /// </summary>
    private static void LabelTopVector(ref byte s, ref uint t, ref int l, int tw, int width, ChamferWeights w, uint distMax)
    {
        int vb = Vector<byte>.Count, vu = Vector<uint>.Count;
        var hv = new Vector<uint>(w.Hv);
        var diag = new Vector<uint>(w.Diag);
        var lng = new Vector<uint>(w.Long);
        var max = new Vector<uint>(distMax);
        int u1 = -tw, u2 = -2 * tw;
        for (int j = 0; j < width; j += vb)
        {
            j = Math.Min(j, width - vb);
            ZeroMasks(ref s, j, out Vector<uint> z0, out Vector<uint> z1, out Vector<uint> z2, out Vector<uint> z3);
            for (int k = 0; k < 4; k++)
            {
                nuint o = (nuint)(j + (k * vu));
                Vector<uint> t0 = max;
                Vector<int> l0 = Vector<int>.Zero;
                Pick(ref t, ref l, u2 - 1, o, lng, ref t0, ref l0);
                Pick(ref t, ref l, u2 + 1, o, lng, ref t0, ref l0);
                Pick(ref t, ref l, u1 - 2, o, lng, ref t0, ref l0);
                Pick(ref t, ref l, u1 - 1, o, diag, ref t0, ref l0);
                Pick(ref t, ref l, u1, o, hv, ref t0, ref l0);
                Pick(ref t, ref l, u1 + 1, o, diag, ref t0, ref l0);
                Pick(ref t, ref l, u1 + 2, o, lng, ref t0, ref l0);
                Vector<uint> z = k switch { 0 => z0, 1 => z1, 2 => z2, _ => z3 };
                Vector.AndNot(t0, z).StoreUnsafe(ref t, o);
                Vector.ConditionalSelect(Vector.AsVectorInt32(z), Vector.LoadUnsafe(ref l, o), l0).StoreUnsafe(ref l, o);
            }
        }
    }

    /// <summary>
    /// Backward labelled pass over the seven neighbours in the rows below, in the OpenCV order. A
    /// repeated pass over the same lanes does not change the result, so the final vector may overlap.
    /// </summary>
    private static void LabelBottomVector(ref uint t, ref int l, int tw, int width, ChamferWeights w)
    {
        int vu = Vector<uint>.Count;
        var hv = new Vector<uint>(w.Hv);
        var diag = new Vector<uint>(w.Diag);
        var lng = new Vector<uint>(w.Long);
        int d1 = tw, d2 = 2 * tw;
        for (int j = 0; j < width; j += vu)
        {
            j = Math.Min(j, width - vu);
            nuint o = (nuint)j;
            Vector<uint> t0 = Vector.LoadUnsafe(ref t, o);
            Vector<int> l0 = Vector.LoadUnsafe(ref l, o);
            Pick(ref t, ref l, d2 + 1, o, lng, ref t0, ref l0);
            Pick(ref t, ref l, d2 - 1, o, lng, ref t0, ref l0);
            Pick(ref t, ref l, d1 + 2, o, lng, ref t0, ref l0);
            Pick(ref t, ref l, d1 + 1, o, diag, ref t0, ref l0);
            Pick(ref t, ref l, d1, o, hv, ref t0, ref l0);
            Pick(ref t, ref l, d1 - 1, o, diag, ref t0, ref l0);
            Pick(ref t, ref l, d1 - 2, o, lng, ref t0, ref l0);
            t0.StoreUnsafe(ref t, o);
            l0.StoreUnsafe(ref l, o);
        }
    }

    /// <summary>Lane-wise <c>if (t0 &gt; t[n] + w) { t0 = t[n] + w; l0 = l[n]; }</c> for the neighbour at <paramref name="offset"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Pick(ref uint t, ref int l, int offset, nuint o, Vector<uint> weight, ref Vector<uint> t0, ref Vector<int> l0)
    {
        Vector<uint> c = Vector.LoadUnsafe(ref Unsafe.Add(ref t, offset), o) + weight;
        Vector<uint> m = Vector.GreaterThan(t0, c);
        t0 = Vector.ConditionalSelect(m, c, t0);
        l0 = Vector.ConditionalSelect(Vector.AsVectorInt32(m), Vector.LoadUnsafe(ref Unsafe.Add(ref l, offset), o), l0);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Relax(ref uint t, ref int l, int offset, uint weight, ref uint t0, ref int l0)
    {
        uint c = Unsafe.Add(ref t, offset) + weight;
        if (t0 > c)
        {
            t0 = c;
            l0 = Unsafe.Add(ref l, offset);
        }
    }

    /// <summary>Writes the initial labels of the zero pixels; the other labels are overwritten by the forward pass.</summary>
    private static void InitLabels(Mat<byte> source, Mat<int> labels, DistanceTransformLabelTypes labelType)
    {
        int width = source.Width, height = source.Height;
        if (labelType == DistanceTransformLabelTypes.ConnectedComponent)
        {
            using PooledMat<byte> pooled = PooledMat<byte>.Rent(width, height, 1);
            Mat<byte> zero = pooled.Mat;
            for (int y = 0; y < height; y++)
            {
                ZeroMaskRow(ref MemoryMarshal.GetReference(source.GetReadOnlyRowSpan(y)), ref MemoryMarshal.GetReference(zero.GetRowSpan(y)), width);
            }

            ConnectedComponentLabeling.ConnectedComponents(zero, labels, 8, ConnectedComponentsAlgorithm.Wu);
            return;
        }

        int k = 1;
        for (int y = 0; y < height; y++)
        {
            ref byte s = ref MemoryMarshal.GetReference(source.GetReadOnlyRowSpan(y));
            ref int l = ref MemoryMarshal.GetReference(labels.GetRowSpan(y));
            for (int x = 0; x < width; x++)
            {
                if (Unsafe.Add(ref s, x) == 0)
                {
                    Unsafe.Add(ref l, x) = k++;
                }
            }
        }
    }

    private static void ZeroMaskRow(ref byte s, ref byte d, int width)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated && width >= Vector<byte>.Count)
        {
            int vc = Vector<byte>.Count;
            for (; x < width; x += vc)
            {
                x = Math.Min(x, width - vc);
                Vector.Equals(Vector.LoadUnsafe(ref s, (nuint)x), Vector<byte>.Zero).StoreUnsafe(ref d, (nuint)x);
            }

            return;
        }

        for (; x < width; x++)
        {
            Unsafe.Add(ref d, x) = Unsafe.Add(ref s, x) == 0 ? (byte)255 : (byte)0;
        }
    }

    /// <summary>Forward recursion along the row; zero pixels hold candidate 0 and stay 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LeftRecursion(ref uint cur, int width, uint hv, uint distMax)
    {
        uint left = distMax;
        for (int j = 0; j < width; j++)
        {
            ref uint c = ref Unsafe.Add(ref cur, j);
            left = Math.Min(c, left + hv);
            c = left;
        }
    }

    /// <summary>
    /// Backward recursion along the row. OpenCV skips pixels with <c>t0 &lt;= HV</c>; every
    /// candidate is at least HV, so the unconditional minimum is equal.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RightRecursion(ref uint cur, int width, uint hv, uint distMax)
    {
        uint right = distMax;
        for (int j = width - 1; j >= 0; j--)
        {
            ref uint c = ref Unsafe.Add(ref cur, j);
            right = Math.Min(c, right + hv);
            c = right;
        }
    }

    private static void Candidates3Scalar(ref byte s, ref uint top, ref uint cur, int width, ChamferWeights w, uint distMax)
    {
        for (int j = 0; j < width; j++)
        {
            uint m = Math.Min(Math.Min(Unsafe.Add(ref top, j - 1) + w.Diag, Unsafe.Add(ref top, j) + w.Hv), Unsafe.Add(ref top, j + 1) + w.Diag);
            Unsafe.Add(ref cur, j) = Unsafe.Add(ref s, j) == 0 ? 0 : Math.Min(m, distMax);
        }
    }

    private static void Candidates3Vector(ref byte s, ref uint top, ref uint cur, int width, ChamferWeights w, uint distMax)
    {
        int vb = Vector<byte>.Count, vu = Vector<uint>.Count;
        var hv = new Vector<uint>(w.Hv);
        var diag = new Vector<uint>(w.Diag);
        var max = new Vector<uint>(distMax);
        ref uint topLeft = ref Unsafe.Subtract(ref top, 1);
        ref uint topRight = ref Unsafe.Add(ref top, 1);
        for (int j = 0; j < width; j += vb)
        {
            j = Math.Min(j, width - vb);
            ZeroMasks(ref s, j, out Vector<uint> z0, out Vector<uint> z1, out Vector<uint> z2, out Vector<uint> z3);
            for (int k = 0; k < 4; k++)
            {
                nuint o = (nuint)(j + (k * vu));
                Vector<uint> m = Vector.Min(Vector.Min(Vector.LoadUnsafe(ref topLeft, o) + diag, Vector.LoadUnsafe(ref top, o) + hv), Vector.LoadUnsafe(ref topRight, o) + diag);
                Vector<uint> z = k switch { 0 => z0, 1 => z1, 2 => z2, _ => z3 };
                Vector.AndNot(Vector.Min(m, max), z).StoreUnsafe(ref cur, o);
            }
        }
    }

    private static void Bottom3Scalar(ref uint bottom, ref uint cur, int width, ChamferWeights w)
    {
        for (int j = 0; j < width; j++)
        {
            uint m = Math.Min(Math.Min(Unsafe.Add(ref bottom, j + 1) + w.Diag, Unsafe.Add(ref bottom, j) + w.Hv), Unsafe.Add(ref bottom, j - 1) + w.Diag);
            ref uint c = ref Unsafe.Add(ref cur, j);
            c = Math.Min(c, m);
        }
    }

    private static void Bottom3Vector(ref uint bottom, ref uint cur, int width, ChamferWeights w)
    {
        int vu = Vector<uint>.Count;
        var hv = new Vector<uint>(w.Hv);
        var diag = new Vector<uint>(w.Diag);
        ref uint bottomLeft = ref Unsafe.Subtract(ref bottom, 1);
        ref uint bottomRight = ref Unsafe.Add(ref bottom, 1);
        for (int j = 0; j < width; j += vu)
        {
            // The minimum is idempotent, so the final vector may overlap the previous one.
            j = Math.Min(j, width - vu);
            nuint o = (nuint)j;
            Vector<uint> m = Vector.Min(Vector.Min(Vector.LoadUnsafe(ref bottomRight, o) + diag, Vector.LoadUnsafe(ref bottom, o) + hv), Vector.LoadUnsafe(ref bottomLeft, o) + diag);
            Vector.Min(Vector.LoadUnsafe(ref cur, o), m).StoreUnsafe(ref cur, o);
        }
    }

    private static void Candidates5Scalar(ref byte s, ref uint cur, int tw, int width, ChamferWeights w, uint distMax)
    {
        ref uint top1 = ref Unsafe.Subtract(ref cur, tw);
        ref uint top2 = ref Unsafe.Subtract(ref cur, 2 * tw);
        for (int j = 0; j < width; j++)
        {
            uint m = Math.Min(Unsafe.Add(ref top2, j - 1) + w.Long, Unsafe.Add(ref top2, j + 1) + w.Long);
            m = Math.Min(m, Unsafe.Add(ref top1, j - 2) + w.Long);
            m = Math.Min(m, Unsafe.Add(ref top1, j - 1) + w.Diag);
            m = Math.Min(m, Unsafe.Add(ref top1, j) + w.Hv);
            m = Math.Min(m, Unsafe.Add(ref top1, j + 1) + w.Diag);
            m = Math.Min(m, Unsafe.Add(ref top1, j + 2) + w.Long);
            Unsafe.Add(ref cur, j) = Unsafe.Add(ref s, j) == 0 ? 0 : Math.Min(m, distMax);
        }
    }

    private static void Candidates5Vector(ref byte s, ref uint cur, int tw, int width, ChamferWeights w, uint distMax)
    {
        int vb = Vector<byte>.Count, vu = Vector<uint>.Count;
        var hv = new Vector<uint>(w.Hv);
        var diag = new Vector<uint>(w.Diag);
        var lng = new Vector<uint>(w.Long);
        var max = new Vector<uint>(distMax);
        ref uint top1 = ref Unsafe.Subtract(ref cur, tw + 2);
        ref uint top2 = ref Unsafe.Subtract(ref cur, (2 * tw) + 1);
        for (int j = 0; j < width; j += vb)
        {
            j = Math.Min(j, width - vb);
            ZeroMasks(ref s, j, out Vector<uint> z0, out Vector<uint> z1, out Vector<uint> z2, out Vector<uint> z3);
            for (int k = 0; k < 4; k++)
            {
                nuint o = (nuint)(j + (k * vu));
                Vector<uint> m = Vector.Min(Vector.LoadUnsafe(ref top2, o) + lng, Vector.LoadUnsafe(ref top2, o + 2) + lng);
                m = Vector.Min(m, Vector.LoadUnsafe(ref top1, o) + lng);
                m = Vector.Min(m, Vector.LoadUnsafe(ref top1, o + 1) + diag);
                m = Vector.Min(m, Vector.LoadUnsafe(ref top1, o + 2) + hv);
                m = Vector.Min(m, Vector.LoadUnsafe(ref top1, o + 3) + diag);
                m = Vector.Min(m, Vector.LoadUnsafe(ref top1, o + 4) + lng);
                Vector<uint> z = k switch { 0 => z0, 1 => z1, 2 => z2, _ => z3 };
                Vector.AndNot(Vector.Min(m, max), z).StoreUnsafe(ref cur, o);
            }
        }
    }

    private static void Bottom5Scalar(ref uint cur, int tw, int width, ChamferWeights w)
    {
        ref uint bot1 = ref Unsafe.Add(ref cur, tw);
        ref uint bot2 = ref Unsafe.Add(ref cur, 2 * tw);
        for (int j = 0; j < width; j++)
        {
            uint m = Math.Min(Unsafe.Add(ref bot2, j + 1) + w.Long, Unsafe.Add(ref bot2, j - 1) + w.Long);
            m = Math.Min(m, Unsafe.Add(ref bot1, j + 2) + w.Long);
            m = Math.Min(m, Unsafe.Add(ref bot1, j + 1) + w.Diag);
            m = Math.Min(m, Unsafe.Add(ref bot1, j) + w.Hv);
            m = Math.Min(m, Unsafe.Add(ref bot1, j - 1) + w.Diag);
            m = Math.Min(m, Unsafe.Add(ref bot1, j - 2) + w.Long);
            ref uint c = ref Unsafe.Add(ref cur, j);
            c = Math.Min(c, m);
        }
    }

    private static void Bottom5Vector(ref uint cur, int tw, int width, ChamferWeights w)
    {
        int vu = Vector<uint>.Count;
        var hv = new Vector<uint>(w.Hv);
        var diag = new Vector<uint>(w.Diag);
        var lng = new Vector<uint>(w.Long);
        ref uint bot1 = ref Unsafe.Add(ref cur, tw - 2);
        ref uint bot2 = ref Unsafe.Add(ref cur, (2 * tw) - 1);
        for (int j = 0; j < width; j += vu)
        {
            j = Math.Min(j, width - vu);
            nuint o = (nuint)j;
            Vector<uint> m = Vector.Min(Vector.LoadUnsafe(ref bot2, o + 2) + lng, Vector.LoadUnsafe(ref bot2, o) + lng);
            m = Vector.Min(m, Vector.LoadUnsafe(ref bot1, o + 4) + lng);
            m = Vector.Min(m, Vector.LoadUnsafe(ref bot1, o + 3) + diag);
            m = Vector.Min(m, Vector.LoadUnsafe(ref bot1, o + 2) + hv);
            m = Vector.Min(m, Vector.LoadUnsafe(ref bot1, o + 1) + diag);
            m = Vector.Min(m, Vector.LoadUnsafe(ref bot1, o) + lng);
            Vector.Min(Vector.LoadUnsafe(ref cur, o), m).StoreUnsafe(ref cur, o);
        }
    }

    /// <summary>All-ones lanes for the zero source bytes <c>s[j .. j + Vector&lt;byte&gt;.Count)</c>, in four <c>uint</c> vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ZeroMasks(ref byte s, int j, out Vector<uint> z0, out Vector<uint> z1, out Vector<uint> z2, out Vector<uint> z3)
    {
        Vector<byte> bytes = Vector.LoadUnsafe(ref s, (nuint)j);
        Vector.Widen(bytes, out Vector<ushort> lo, out Vector<ushort> hi);
        Vector.Widen(lo, out Vector<uint> a, out Vector<uint> b);
        Vector.Widen(hi, out Vector<uint> c, out Vector<uint> d);
        z0 = Vector.Equals(a, Vector<uint>.Zero);
        z1 = Vector.Equals(b, Vector<uint>.Zero);
        z2 = Vector.Equals(c, Vector<uint>.Zero);
        z3 = Vector.Equals(d, Vector<uint>.Zero);
    }

    /// <summary>
    /// <c>dst = (float)t * 2^-16</c> with the exact <c>uint</c> to <c>float</c> conversion:
    /// <c>hi + lo * 2^-16</c> is exact before the single rounding of the addition.
    /// </summary>
    internal static void StoreRow(ref uint t, ref float d, int width)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && width >= Vector<uint>.Count)
        {
            int vu = Vector<uint>.Count;
            var mask = new Vector<uint>(0xFFFF);
            var scale = new Vector<float>(Scale);
            for (; j < width; j += vu)
            {
                j = Math.Min(j, width - vu);
                Vector<uint> v = Vector.LoadUnsafe(ref t, (nuint)j);
                Vector<float> hi = Vector.ConvertToSingle(Vector.AsVectorInt32(Vector.ShiftRightLogical(v, 16)));
                Vector<float> lo = Vector.ConvertToSingle(Vector.AsVectorInt32(v & mask));
                (hi + (lo * scale)).StoreUnsafe(ref d, (nuint)j);
            }

            return;
        }

        for (; j < width; j++)
        {
            Unsafe.Add(ref d, j) = (float)Unsafe.Add(ref t, j) * Scale;
        }
    }
}
