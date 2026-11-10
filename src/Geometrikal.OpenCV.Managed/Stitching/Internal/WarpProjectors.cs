// This implementation is ported from the OpenCV stitching warpers.
//
// Original sources:
//   modules/stitching/src/warpers.cpp (ProjectorBase::setCameraParams)
//   modules/stitching/include/opencv2/stitching/detail/warpers_inl.hpp (mapForward, mapBackward)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Camera parameters of a projector, as OpenCV <c>detail::ProjectorBase</c>.</summary>
internal sealed class ProjectorParams
{
    public float Scale;
    public readonly float[] K = new float[9];
    public readonly float[] RInv = new float[9];
    public readonly float[] RKInv = new float[9];
    public readonly float[] KRInv = new float[9];
    public readonly float[] T = new float[3];

    /// <summary>Sets K, R and T, as OpenCV <c>ProjectorBase::setCameraParams</c>.</summary>
    public void SetCameraParams(Mat<float> k, Mat<float> r, Mat<float>? t)
    {
        Validate3x3(k, nameof(k));
        Validate3x3(r, nameof(r));
        if (t is not null &&
            !(t.Channels == 1 && ((t.Width == 1 && t.Height == 3) || (t.Width == 3 && t.Height == 1))))
        {
            throw new ArgumentException("T must be a 3x1 or 1x3 single-channel float matrix.", nameof(t));
        }

        var kc = new Mat<float>(3, 3);
        var rc = new Mat<float>(3, 3);
        var rinv = new Mat<float>(3, 3);
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                kc.At(y, x) = k.At(y, x);
                rc.At(y, x) = r.At(y, x);
                rinv.At(x, y) = r.At(y, x);
            }
        }

        LinearAlgebra.Invert(kc, out Mat<float> kinv, DecompTypes.Lu);
        Mat<float> rKinv = LinearAlgebra.Gemm(rc, kinv, 1, null, 0);
        Mat<float> kRinv = LinearAlgebra.Gemm(kc, rinv, 1, null, 0);

        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                K[y * 3 + x] = kc.At(y, x);
                RInv[y * 3 + x] = rinv.At(y, x);
                RKInv[y * 3 + x] = rKinv.At(y, x);
                KRInv[y * 3 + x] = kRinv.At(y, x);
            }
        }

        if (t is null)
        {
            T[0] = T[1] = T[2] = 0f;
        }
        else if (t.Width == 1)
        {
            T[0] = t.At(0, 0); T[1] = t.At(1, 0); T[2] = t.At(2, 0);
        }
        else
        {
            T[0] = t.At(0, 0); T[1] = t.At(0, 1); T[2] = t.At(0, 2);
        }
    }

    internal static void Validate3x3(Mat<float> m, string name)
    {
        ArgumentNullException.ThrowIfNull(m, name);
        if (m.Width != 3 || m.Height != 3 || m.Channels != 1)
        {
            throw new ArgumentException("The matrix must be 3x3 with one channel.", name);
        }
    }
}

/// <summary>Forward and backward mapping of one projection type.</summary>
internal interface IProjector
{
    static abstract void MapForward(ProjectorParams p, float x, float y, out float u, out float v);

    static abstract void MapBackward(ProjectorParams p, float u, float v, out float x, out float y);
}

/// <summary>OpenCV <c>detail::PlaneProjector</c>.</summary>
internal readonly struct PlaneProjector : IProjector
{
    public static void MapForward(ProjectorParams p, float x, float y, out float u, out float v)
    {
        float[] r = p.RKInv;
        float[] t = p.T;
        float x_ = r[0] * x + r[1] * y + r[2];
        float y_ = r[3] * x + r[4] * y + r[5];
        float z_ = r[6] * x + r[7] * y + r[8];

        x_ = t[0] + x_ / z_ * (1 - t[2]);
        y_ = t[1] + y_ / z_ * (1 - t[2]);

        u = p.Scale * x_;
        v = p.Scale * y_;
    }

    public static void MapBackward(ProjectorParams p, float u, float v, out float x, out float y)
    {
        float[] k = p.KRInv;
        float[] t = p.T;
        u = u / p.Scale - t[0];
        v = v / p.Scale - t[1];

        x = k[0] * u + k[1] * v + k[2] * (1 - t[2]);
        y = k[3] * u + k[4] * v + k[5] * (1 - t[2]);
        float z = k[6] * u + k[7] * v + k[8] * (1 - t[2]);

        x /= z;
        y /= z;
    }
}

/// <summary>OpenCV <c>detail::SphericalProjector</c>.</summary>
internal readonly struct SphericalProjector : IProjector
{
    private const float Pi = (float)Math.PI;

    public static void MapForward(ProjectorParams p, float x, float y, out float u, out float v)
    {
        float[] r = p.RKInv;
        float x_ = r[0] * x + r[1] * y + r[2];
        float y_ = r[3] * x + r[4] * y + r[5];
        float z_ = r[6] * x + r[7] * y + r[8];

        u = p.Scale * MathF.Atan2(x_, z_);
        float w = y_ / MathF.Sqrt(x_ * x_ + y_ * y_ + z_ * z_);
        v = p.Scale * (Pi - MathF.Acos(float.IsNaN(w) ? 0f : w));
    }

    public static void MapBackward(ProjectorParams p, float u, float v, out float x, out float y)
    {
        float[] k = p.KRInv;
        u /= p.Scale;
        v /= p.Scale;

        float sinv = MathF.Sin(Pi - v);
        float x_ = sinv * MathF.Sin(u);
        float y_ = MathF.Cos(Pi - v);
        float z_ = sinv * MathF.Cos(u);

        x = k[0] * x_ + k[1] * y_ + k[2] * z_;
        y = k[3] * x_ + k[4] * y_ + k[5] * z_;
        float z = k[6] * x_ + k[7] * y_ + k[8] * z_;

        if (z > 0)
        {
            x /= z;
            y /= z;
        }
        else
        {
            x = y = -1;
        }
    }
}

/// <summary>OpenCV <c>detail::CylindricalProjector</c>.</summary>
internal readonly struct CylindricalProjector : IProjector
{
    public static void MapForward(ProjectorParams p, float x, float y, out float u, out float v)
    {
        float[] r = p.RKInv;
        float x_ = r[0] * x + r[1] * y + r[2];
        float y_ = r[3] * x + r[4] * y + r[5];
        float z_ = r[6] * x + r[7] * y + r[8];

        u = p.Scale * MathF.Atan2(x_, z_);
        v = p.Scale * y_ / MathF.Sqrt(x_ * x_ + z_ * z_);
    }

    public static void MapBackward(ProjectorParams p, float u, float v, out float x, out float y)
    {
        float[] k = p.KRInv;
        u /= p.Scale;
        v /= p.Scale;

        float x_ = MathF.Sin(u);
        float y_ = v;
        float z_ = MathF.Cos(u);

        x = k[0] * x_ + k[1] * y_ + k[2] * z_;
        y = k[3] * x_ + k[4] * y_ + k[5] * z_;
        float z = k[6] * x_ + k[7] * y_ + k[8] * z_;

        if (z > 0)
        {
            x /= z;
            y /= z;
        }
        else
        {
            x = y = -1;
        }
    }
}
