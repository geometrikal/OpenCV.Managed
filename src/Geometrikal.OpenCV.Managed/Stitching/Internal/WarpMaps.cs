// This implementation is ported from the OpenCV stitching warpers.
//
// Original sources:
//   modules/stitching/src/warpers.cpp (PlaneWarper::buildMaps, detectResultRoi)
//   modules/stitching/include/opencv2/stitching/detail/warpers_inl.hpp
//     (RotationWarperBase::buildMaps, warpBackward, detectResultRoi, detectResultRoiByBorder)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Map building and result-ROI detection for the rotation warpers.</summary>
internal static class WarpMaps
{
    /// <summary>Accumulates a bounding box with <c>std::min</c>/<c>std::max</c> semantics (NaN keeps the old value).</summary>
    private struct Bounds
    {
        public float TlU, TlV, BrU, BrV;

        public static Bounds Empty => new() { TlU = float.MaxValue, TlV = float.MaxValue, BrU = -float.MaxValue, BrV = -float.MaxValue };

        public void Add(float u, float v)
        {
            if (u < TlU) TlU = u;
            if (v < TlV) TlV = v;
            if (BrU < u) BrU = u;
            if (BrV < v) BrV = v;
        }

        public readonly void ToPoints(out Point tl, out Point br)
        {
            tl = new Point(CvtInt(TlU), CvtInt(TlV));
            br = new Point(CvtInt(BrU), CvtInt(BrV));
        }
    }

    /// <summary>C++ <c>static_cast&lt;int&gt;(float)</c> on x64: out-of-range and NaN give <c>INT_MIN</c>.</summary>
    internal static int CvtInt(float value)
        => value is >= -2147483648f and < 2147483648f ? (int)value : int.MinValue;

    /// <summary>OpenCV <c>PlaneWarper::detectResultRoi</c>: the four source corners.</summary>
    public static void DetectRoiPlane(ProjectorParams p, Size src, out Point tl, out Point br)
    {
        var b = Bounds.Empty;
        PlaneProjector.MapForward(p, 0, 0, out float u, out float v);
        b.Add(u, v);
        PlaneProjector.MapForward(p, 0, src.Height - 1, out u, out v);
        b.Add(u, v);
        PlaneProjector.MapForward(p, src.Width - 1, 0, out u, out v);
        b.Add(u, v);
        PlaneProjector.MapForward(p, src.Width - 1, src.Height - 1, out u, out v);
        b.Add(u, v);
        b.ToPoints(out tl, out br);
    }

    /// <summary>OpenCV <c>RotationWarperBase::detectResultRoiByBorder</c>.</summary>
    public static void DetectRoiByBorder<TProj>(ProjectorParams p, Size src, out Point tl, out Point br)
        where TProj : IProjector
    {
        var b = Bounds.Empty;
        float u, v;
        for (float x = 0; x < src.Width; ++x)
        {
            TProj.MapForward(p, x, 0, out u, out v);
            b.Add(u, v);
            TProj.MapForward(p, x, src.Height - 1, out u, out v);
            b.Add(u, v);
        }

        for (int y = 0; y < src.Height; ++y)
        {
            TProj.MapForward(p, 0, y, out u, out v);
            b.Add(u, v);
            TProj.MapForward(p, src.Width - 1, y, out u, out v);
            b.Add(u, v);
        }

        b.ToPoints(out tl, out br);
    }

    /// <summary>OpenCV <c>SphericalWarper::detectResultRoi</c>: the border plus the two poles.</summary>
    public static void DetectRoiSpherical(ProjectorParams p, Size src, out Point tl, out Point br)
    {
        DetectRoiByBorder<SphericalProjector>(p, src, out Point dtl, out Point dbr);

        float tlU = dtl.X, tlV = dtl.Y, brU = dbr.X, brV = dbr.Y;
        float[] rinv = p.RInv;
        float[] k = p.K;
        float poleV = (float)(Math.PI * p.Scale);

        for (int pole = 0; pole < 2; pole++)
        {
            float x = rinv[1];
            float y = pole == 0 ? rinv[4] : -rinv[4];
            float z = rinv[7];
            if (y > 0f)
            {
                float x_ = (k[0] * x + k[1] * y) / z + k[2];
                float y_ = k[4] * y / z + k[5];
                if (x_ > 0f && x_ < src.Width && y_ > 0f && y_ < src.Height)
                {
                    float vPole = pole == 0 ? poleV : 0f;
                    tlU = StdMin(tlU, 0f);
                    tlV = StdMin(tlV, vPole);
                    brU = StdMax(brU, 0f);
                    brV = StdMax(brV, vPole);
                }
            }
        }

        tl = new Point(CvtInt(tlU), CvtInt(tlV));
        br = new Point(CvtInt(brU), CvtInt(brV));
    }

    private static float StdMin(float a, float b) => b < a ? b : a;

    private static float StdMax(float a, float b) => a < b ? b : a;

    /// <summary>Fills the backward maps for destination pixels <c>[tl, br]</c>, as OpenCV <c>buildMaps</c>.</summary>
    public static void BuildMaps<TProj>(ProjectorParams p, Point tl, Mat<float> xmap, Mat<float> ymap)
        where TProj : IProjector
    {
        int width = xmap.Width;
        long work = Math.Max(1, width / Vector<float>.Count) * 4L;
        if (typeof(TProj) == typeof(PlaneProjector))
        {
            ParallelRows.For(xmap.Height, work, (y0, y1) => PlaneRows(p, tl, xmap, ymap, y0, y1));
            return;
        }

        bool spherical = typeof(TProj) == typeof(SphericalProjector);
        if (!spherical && typeof(TProj) != typeof(CylindricalProjector))
        {
            ParallelRows.For(xmap.Height, width * 40L, (y0, y1) => GenericRows<TProj>(p, tl, xmap, ymap, y0, y1));
            return;
        }

        // sin(u) and cos(u) depend on the column only; the row terms depend on the row only.
        float[] sinU = ArrayPool<float>.Shared.Rent(width);
        float[] cosU = ArrayPool<float>.Shared.Rent(width);
        try
        {
            for (int i = 0; i < width; i++)
            {
                float u = (tl.X + i) / p.Scale;
                sinU[i] = MathF.Sin(u);
                cosU[i] = MathF.Cos(u);
            }

            ParallelRows.For(xmap.Height, work, (y0, y1) => RotationRows(p, tl, sinU, cosU, spherical, xmap, ymap, y0, y1));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(sinU);
            ArrayPool<float>.Shared.Return(cosU);
        }
    }

    private static void GenericRows<TProj>(ProjectorParams p, Point tl, Mat<float> xmap, Mat<float> ymap, int y0, int y1)
        where TProj : IProjector
    {
        for (int row = y0; row < y1; row++)
        {
            Span<float> xr = xmap.GetRowSpan(row);
            Span<float> yr = ymap.GetRowSpan(row);
            float v = tl.Y + row;
            for (int i = 0; i < xr.Length; i++)
            {
                TProj.MapBackward(p, tl.X + i, v, out float x, out float y);
                xr[i] = x;
                yr[i] = y;
            }
        }
    }

    /// <summary>
    /// Spherical and cylindrical backward mapping from per-column <c>sin(u)</c>, <c>cos(u)</c> and
    /// per-row terms, in <see cref="Vector{T}"/> lanes with the scalar operation order of
    /// <see cref="SphericalProjector.MapBackward"/> and <see cref="CylindricalProjector.MapBackward"/>.
    /// </summary>
    private static void RotationRows(ProjectorParams p, Point tl, float[] sinU, float[] cosU, bool spherical, Mat<float> xmap, Mat<float> ymap, int y0, int y1)
    {
        const float pi = (float)Math.PI;
        float[] k = p.KRInv;
        int n = Vector<float>.Count;
        var vk0 = new Vector<float>(k[0]);
        var vk2 = new Vector<float>(k[2]);
        var vk3 = new Vector<float>(k[3]);
        var vk5 = new Vector<float>(k[5]);
        var vk6 = new Vector<float>(k[6]);
        var vk8 = new Vector<float>(k[8]);
        var minusOne = new Vector<float>(-1f);
        for (int row = y0; row < y1; row++)
        {
            Span<float> xr = xmap.GetRowSpan(row);
            Span<float> yr = ymap.GetRowSpan(row);
            float v = (tl.Y + row) / p.Scale;
            float rowScale, yv;
            if (spherical)
            {
                rowScale = MathF.Sin(pi - v);
                yv = MathF.Cos(pi - v);
            }
            else
            {
                rowScale = 1f;
                yv = v;
            }

            float k1y = k[1] * yv, k4y = k[4] * yv, k7y = k[7] * yv;
            int width = xr.Length;
            int i = 0;
            if (width >= n)
            {
                var vs = new Vector<float>(rowScale);
                var vk1y = new Vector<float>(k1y);
                var vk4y = new Vector<float>(k4y);
                var vk7y = new Vector<float>(k7y);
                for (; ; i += n)
                {
                    if (i > width - n)
                    {
                        i = width - n;
                    }

                    var su = new Vector<float>(sinU, i);
                    var cu = new Vector<float>(cosU, i);
                    Vector<float> xs = spherical ? vs * su : su;
                    Vector<float> zs = spherical ? vs * cu : cu;
                    Vector<float> x = (vk0 * xs + vk1y) + (vk2 * zs);
                    Vector<float> y = (vk3 * xs + vk4y) + (vk5 * zs);
                    Vector<float> z = (vk6 * xs + vk7y) + (vk8 * zs);
                    Vector<int> positive = Vector.GreaterThan(z, Vector<float>.Zero);
                    Vector.ConditionalSelect(positive, x / z, minusOne).CopyTo(xr.Slice(i));
                    Vector.ConditionalSelect(positive, y / z, minusOne).CopyTo(yr.Slice(i));
                    if (i == width - n)
                    {
                        break;
                    }
                }
            }
            else
            {
                for (; i < width; i++)
                {
                    float xs = spherical ? rowScale * sinU[i] : sinU[i];
                    float zs = spherical ? rowScale * cosU[i] : cosU[i];
                    float x = k[0] * xs + k1y + k[2] * zs;
                    float y = k[3] * xs + k4y + k[5] * zs;
                    float z = k[6] * xs + k7y + k[8] * zs;
                    xr[i] = z > 0 ? x / z : -1f;
                    yr[i] = z > 0 ? y / z : -1f;
                }
            }
        }
    }

    /// <summary>Plane backward mapping in <see cref="Vector{T}"/> lanes with the scalar operation order.</summary>
    private static void PlaneRows(ProjectorParams p, Point tl, Mat<float> xmap, Mat<float> ymap, int y0, int y1)
    {
        float[] k = p.KRInv;
        float scale = p.Scale;
        float t0 = p.T[0];
        float oneMinusT2 = 1 - p.T[2];
        float c2 = k[2] * oneMinusT2, c5 = k[5] * oneMinusT2, c8 = k[8] * oneMinusT2;
        int n = Vector<float>.Count;
        var vk0 = new Vector<float>(k[0]);
        var vk3 = new Vector<float>(k[3]);
        var vk6 = new Vector<float>(k[6]);
        var vc2 = new Vector<float>(c2);
        var vc5 = new Vector<float>(c5);
        var vc8 = new Vector<float>(c8);
        var vScale = new Vector<float>(scale);
        var vt0 = new Vector<float>(t0);
        Span<float> ramp = stackalloc float[n];
        for (int i = 0; i < n; i++)
        {
            ramp[i] = i;
        }

        var vRamp = new Vector<float>(ramp);

        for (int row = y0; row < y1; row++)
        {
            Span<float> xr = xmap.GetRowSpan(row);
            Span<float> yr = ymap.GetRowSpan(row);
            float v = (tl.Y + row) / scale - p.T[1];
            float k1v = k[1] * v, k4v = k[4] * v, k7v = k[7] * v;
            var vk1v = new Vector<float>(k1v);
            var vk4v = new Vector<float>(k4v);
            var vk7v = new Vector<float>(k7v);
            int width = xr.Length;
            int i = 0;
            if (width >= n)
            {
                ref float xb = ref MemoryMarshal.GetReference(xr);
                ref float yb = ref MemoryMarshal.GetReference(yr);
                for (; ; i += n)
                {
                    if (i > width - n)
                    {
                        i = width - n;
                    }

                    // Integer u values below 2^24 convert to float exactly, so the ramp sum is exact.
                    var u = (new Vector<float>(tl.X + i) + vRamp) / vScale - vt0;
                    var x = (vk0 * u + vk1v) + vc2;
                    var y = (vk3 * u + vk4v) + vc5;
                    var z = (vk6 * u + vk7v) + vc8;
                    (x / z).StoreUnsafe(ref xb, (nuint)i);
                    (y / z).StoreUnsafe(ref yb, (nuint)i);
                    if (i == width - n)
                    {
                        break;
                    }
                }
            }
            else
            {
                for (; i < width; i++)
                {
                    float u = (tl.X + i) / scale - t0;
                    float z = k[6] * u + k7v + c8;
                    xr[i] = (k[0] * u + k1v + c2) / z;
                    yr[i] = (k[3] * u + k4v + c5) / z;
                }
            }
        }
    }

    /// <summary>Forward maps of <c>warpBackward</c>: destination pixel to the warped source position minus <paramref name="srcTl"/>.</summary>
    public static void BuildBackwardMaps<TProj>(ProjectorParams p, Point srcTl, Mat<float> xmap, Mat<float> ymap)
        where TProj : IProjector
    {
        ParallelRows.For(xmap.Height, xmap.Width * 40L, (y0, y1) =>
        {
            for (int row = y0; row < y1; row++)
            {
                Span<float> xr = xmap.GetRowSpan(row);
                Span<float> yr = ymap.GetRowSpan(row);
                for (int i = 0; i < xr.Length; i++)
                {
                    TProj.MapForward(p, i, row, out float u, out float v);
                    xr[i] = u - srcTl.X;
                    yr[i] = v - srcTl.Y;
                }
            }
        });
    }
}
