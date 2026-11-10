// This implementation is ported from the OpenCV 5.0 stitching wave correction.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp (WaveCorrectKind)
//   modules/stitching/src/motion_estimators.cpp (autoDetectWaveCorrectKind, waveCorrect)
//   modules/core/src/matrix_operations.cpp (Mat::cross)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>Wave correction kind, as OpenCV <c>detail::WaveCorrectKind</c>.</summary>
public enum WaveCorrectKind
{
    /// <summary>Straighten a horizontal panorama (<c>WAVE_CORRECT_HORIZ</c>).</summary>
    Horizontal = 0,

    /// <summary>Straighten a vertical panorama (<c>WAVE_CORRECT_VERT</c>).</summary>
    Vertical = 1,

    /// <summary>Select <see cref="Horizontal"/> or <see cref="Vertical"/> from the rotations (<c>WAVE_CORRECT_AUTO</c>).</summary>
    Auto = 2,
}

/// <summary>
/// Wave correction of camera rotations, as OpenCV <c>detail::waveCorrect</c>.
/// </summary>
/// <remarks>
/// The rotations are read as <c>float</c> values and the computation uses <c>float</c>, as
/// OpenCV requires <c>CV_32F</c> rotations. The corrected rotations are <c>float</c> values.
/// </remarks>
public static class WaveCorrection
{
    private const double DblMin = 2.2250738585072014e-308;

    /// <summary>
    /// Selects the wave correction kind, as OpenCV <c>autoDetectWaveCorrectKind</c>: vertical
    /// when the projected optical axes spread more along y than along x, otherwise horizontal.
    /// </summary>
    /// <param name="rmats">3 × 3 rotations. They are not modified.</param>
    /// <returns><see cref="WaveCorrectKind.Horizontal"/> or <see cref="WaveCorrectKind.Vertical"/>.</returns>
    public static WaveCorrectKind AutoDetectWaveCorrectKind(IReadOnlyList<Mat<double>> rmats)
    {
        ArgumentNullException.ThrowIfNull(rmats);
        if (rmats.Count == 0)
        {
            throw new ArgumentException("At least one rotation is required.", nameof(rmats));
        }

        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        bool first = true;
        foreach (Mat<double> r in rmats)
        {
            float z = (float)r.At(2, 2);
            float x = (float)r.At(0, 2) / z;
            float y = (float)r.At(1, 2) / z;
            if (first)
            {
                minX = maxX = x;
                minY = maxY = y;
                first = false;
                continue;
            }

            if (x < minX)
            {
                minX = x;
            }

            if (!(x < maxX))
            {
                maxX = x;
            }

            if (y < minY)
            {
                minY = y;
            }

            if (!(y < maxY))
            {
                maxY = y;
            }
        }

        double deltaX = maxX - minX;
        double deltaY = maxY - minY;
        return deltaY > deltaX ? WaveCorrectKind.Vertical : WaveCorrectKind.Horizontal;
    }

    /// <summary>
    /// Rotates all cameras so that the panorama is straight, as OpenCV <c>waveCorrect</c>. With
    /// fewer than two rotations, or when the correction axis is undefined, the rotations are not
    /// changed.
    /// </summary>
    /// <param name="rmats">3 × 3 rotations. Each element is replaced by the corrected rotation.</param>
    /// <param name="kind">Correction kind.</param>
    public static void WaveCorrect(IList<Mat<double>> rmats, WaveCorrectKind kind)
    {
        ArgumentNullException.ThrowIfNull(rmats);
        if (rmats.Count <= 1)
        {
            return;
        }

        if (kind == WaveCorrectKind.Auto)
        {
            kind = AutoDetectWaveCorrectKind((IReadOnlyList<Mat<double>>)rmats.ToList());
        }

        if (kind is not WaveCorrectKind.Horizontal and not WaveCorrectKind.Vertical)
        {
            throw new ArgumentOutOfRangeException(nameof(kind), "Unsupported kind of wave correction.");
        }

        var rf = new Mat<float>[rmats.Count];
        for (int i = 0; i < rf.Length; i++)
        {
            rf[i] = SmallMat.ToFloat(rmats[i]);
        }

        var moment = new Mat<float>(3, 3);
        Span<float> mom = moment.GetSpan();
        foreach (Mat<float> r in rf)
        {
            Mat<float> col = Column(r, 0);
            Mat<float> outer = LinearAlgebra.Gemm(col, col, 1, null, 0, GemmFlags.BTranspose);
            ReadOnlySpan<float> o = outer.GetReadOnlySpan();
            for (int k = 0; k < 9; k++)
            {
                mom[k] += o[k];
            }
        }

        LinearAlgebra.Eigen(moment, out _, out Mat<float> vecs);
        int row = kind == WaveCorrectKind.Horizontal ? 2 : 0;
        Span<float> rg1 = stackalloc float[3] { vecs.At(row, 0), vecs.At(row, 1), vecs.At(row, 2) };

        Span<float> imgK = stackalloc float[3];
        foreach (Mat<float> r in rf)
        {
            imgK[0] += r.At(0, 2);
            imgK[1] += r.At(1, 2);
            imgK[2] += r.At(2, 2);
        }

        Span<float> rg0 = stackalloc float[3];
        Cross(rg1, imgK, rg0);
        double norm = Math.Sqrt((((double)rg0[0] * rg0[0]) + ((double)rg0[1] * rg0[1])) + ((double)rg0[2] * rg0[2]));
        if (norm <= DblMin)
        {
            return;
        }

        float scale = (float)(1.0 / norm);
        for (int k = 0; k < 3; k++)
        {
            rg0[k] = (rg0[k] * scale) + 0f;
        }

        Span<float> rg2 = stackalloc float[3];
        Cross(rg0, rg1, rg2);

        double conf = 0;
        if (kind == WaveCorrectKind.Horizontal)
        {
            foreach (Mat<float> r in rf)
            {
                conf += Dot(rg0, r);
            }
        }
        else
        {
            foreach (Mat<float> r in rf)
            {
                conf -= Dot(rg1, r);
            }
        }

        if (conf < 0)
        {
            for (int k = 0; k < 3; k++)
            {
                rg0[k] = (rg0[k] * -1f) + 0f;
                rg1[k] = (rg1[k] * -1f) + 0f;
            }
        }

        var rot = new Mat<float>(3, 3);
        for (int k = 0; k < 3; k++)
        {
            rot.At(0, k) = rg0[k];
            rot.At(1, k) = rg1[k];
            rot.At(2, k) = rg2[k];
        }

        for (int i = 0; i < rf.Length; i++)
        {
            rmats[i] = SmallMat.ToDouble(SmallMat.Mul(rot, rf[i]));
        }
    }

    private static Mat<float> Column(Mat<float> r, int c)
    {
        var col = new Mat<float>(1, 3);
        for (int k = 0; k < 3; k++)
        {
            col.At(k, 0) = r.At(k, c);
        }

        return col;
    }

    private static void Cross(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
    {
        c[0] = (a[1] * b[2]) - (a[2] * b[1]);
        c[1] = (a[2] * b[0]) - (a[0] * b[2]);
        c[2] = (a[0] * b[1]) - (a[1] * b[0]);
    }

    private static double Dot(ReadOnlySpan<float> v, Mat<float> r)
    {
        double s = 0;
        for (int k = 0; k < 3; k++)
        {
            s += (double)v[k] * r.At(k, 0);
        }

        return s;
    }
}
