using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Deterministic benchmark inputs. <c>tools/bench_cv2.py</c> implements the same formulas,
/// so the managed port and cv2 process identical pixels.
/// </summary>
public static class Inputs
{
    public const int ShapeCount = 64;

    public static uint Hash(uint x, uint y)
    {
        uint h = (x * 73856093u) ^ (y * 19349663u);
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        return h;
    }

    public static Mat<byte> Gray(int size)
    {
        var mat = new Mat<byte>(size, size);
        for (int y = 0; y < size; y++)
        {
            Span<byte> row = mat.GetRowSpan(y);
            for (int x = 0; x < size; x++)
            {
                row[x] = (byte)(((x ^ y) + (int)(Hash((uint)x, (uint)y) & 63)) & 255);
            }
        }

        return mat;
    }

    public static Mat<byte> Color(int size)
    {
        var mat = new Mat<byte>(size, size, 3);
        for (int y = 0; y < size; y++)
        {
            Span<byte> row = mat.GetRowSpan(y);
            for (int x = 0; x < size; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    uint h = Hash((uint)x, (uint)(y * 3 + c));
                    row[x * 3 + c] = (byte)((x * (c + 1) + y * (3 - c) + (int)(h & 63)) & 255);
                }
            }
        }

        return mat;
    }

    public static Mat<float> GrayFloat(int size)
    {
        Mat<byte> gray = Gray(size);
        var mat = new Mat<float>(size, size);
        ReadOnlySpan<byte> src = gray.GetReadOnlySpan();
        Span<float> dst = mat.GetSpan();
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = src[i];
        }

        return mat;
    }

    /// <summary>
    /// 16x16 cells; about one third are foreground, and half of those contain a 6x6 hole.
    /// </summary>
    public static Mat<byte> Binary(int size)
    {
        var mat = new Mat<byte>(size, size);
        for (int y = 0; y < size; y++)
        {
            Span<byte> row = mat.GetRowSpan(y);
            for (int x = 0; x < size; x++)
            {
                uint h = Hash((uint)(x >> 4), (uint)(y >> 4));
                int lx = x & 15;
                int ly = y & 15;
                bool on = h % 3 == 0;
                bool hole = (h >> 8) % 2 == 0 && lx is >= 5 and <= 10 && ly is >= 5 and <= 10;
                row[x] = on && !hole ? (byte)255 : (byte)0;
            }
        }

        return mat;
    }

    /// <summary>About 0.1% zero pixels; every other pixel is 255.</summary>
    public static Mat<byte> SparseZeros(int size)
    {
        var mat = new Mat<byte>(size, size);
        for (int y = 0; y < size; y++)
        {
            Span<byte> row = mat.GetRowSpan(y);
            for (int x = 0; x < size; x++)
            {
                row[x] = Hash((uint)(x + 11), (uint)y) % 1000 == 0 ? (byte)0 : (byte)255;
            }
        }

        return mat;
    }

    /// <summary>Watershed seeds: label <c>k + 1</c> at <see cref="RandomPoint"/>(k, 3); a later seed overwrites an earlier one.</summary>
    public static Mat<int> Seeds(int size, int count)
    {
        var mat = new Mat<int>(size, size);
        for (int k = 0; k < count; k++)
        {
            Point p = RandomPoint(k, 3, size);
            mat.At(p.Y, p.X) = k + 1;
        }

        return mat;
    }

    public static Point RandomPoint(int index, int salt, int size)
        => new((int)(Hash((uint)index, (uint)(2 * salt)) % (uint)size),
               (int)(Hash((uint)index, (uint)(2 * salt + 1)) % (uint)size));
}
