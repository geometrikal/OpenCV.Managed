// This implementation is ported from the OpenCV warp and remap drivers.
//
// Original sources:
//   modules/imgproc/src/imgwarp.cpp (RemapInvoker, WarpAffineInvoker, WarpPerspectiveInvoker,
//     cv::remap kernel tables)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Row driver of the warps and remap: each destination row gets its coordinates from an
/// <see cref="IRowCoords"/> and then runs the table-driven kernel of the interpolation.
/// </summary>
internal static class WarpEngine
{
    public static void Run<T, TCoords>(
        Mat<T> src, Mat<T> dst, TCoords coords, InterpolationFlags interpolation, BorderType border, Scalar value, bool relative)
        where T : unmanaged
        where TCoords : struct, IRowCoords
    {
        var s = new RemapSource<T>(src, border, value);
        int taps = interpolation switch
        {
            InterpolationFlags.Nearest => 1,
            InterpolationFlags.Linear => 4,
            InterpolationFlags.Cubic => 16,
            _ => 64,
        };
        long work = (long)dst.Width * dst.Channels * taps;
        ParallelRows.For(dst.Height, work, (y0, y1) => Rows(s, dst, coords, interpolation, relative, y0, y1));
    }

    private static void Rows<T, TCoords>(
        RemapSource<T> s, Mat<T> dst, TCoords coords, InterpolationFlags interpolation, bool relative, int y0, int y1)
        where T : unmanaged
        where TCoords : struct, IRowCoords
    {
        int width = dst.Width;
        short[] xyArray = ArrayPool<short>.Shared.Rent(2 * width);
        ushort[] fxyArray = ArrayPool<ushort>.Shared.Rent(width);
        try
        {
            Span<short> xy = xyArray.AsSpan(0, 2 * width);
            Span<ushort> fxy = fxyArray.AsSpan(0, width);
            bool nearest = interpolation == InterpolationFlags.Nearest;
            for (int y = y0; y < y1; y++)
            {
                coords.Row(y, xy, fxy, nearest);
                Kernel(s, dst.GetRowSpan(y), xy, fxy, interpolation, y, relative);
            }
        }
        finally
        {
            ArrayPool<short>.Shared.Return(xyArray);
            ArrayPool<ushort>.Shared.Return(fxyArray);
        }
    }

    private static void Kernel<T>(
        in RemapSource<T> s, Span<T> d, ReadOnlySpan<short> xy, ReadOnlySpan<ushort> fxy, InterpolationFlags interpolation, int y, bool relative)
        where T : unmanaged
    {
        if (interpolation == InterpolationFlags.Nearest)
        {
            RemapKernels.Nearest(s, d, xy, 0, y, relative);
            return;
        }

        InterTables tables = interpolation switch
        {
            InterpolationFlags.Linear => InterTables.Linear,
            InterpolationFlags.Cubic => InterTables.Cubic,
            _ => InterTables.Lanczos4,
        };
        int k = tables.Ksize;
        if (typeof(T) == typeof(byte))
        {
            ref readonly RemapSource<byte> sb = ref Unsafe.As<RemapSource<T>, RemapSource<byte>>(ref Unsafe.AsRef(in s));
            Span<byte> db = MemoryMarshal.Cast<T, byte>(d);
            if (k == 2)
            {
                RemapKernels.Bilinear8u(sb, db, xy, fxy, tables.Fixed, 0, y, relative);
            }
            else
            {
                RemapKernels.Separable<byte, int, short, RemapOps8u>(sb, db, xy, fxy, tables.Fixed, k, 0, y, relative);
            }
        }
        else if (typeof(T) == typeof(double))
        {
            ref readonly RemapSource<double> sd = ref Unsafe.As<RemapSource<T>, RemapSource<double>>(ref Unsafe.AsRef(in s));
            Span<double> dd = MemoryMarshal.Cast<T, double>(d);
            if (k == 2)
            {
                RemapKernels.Bilinear<double, double, float, RemapOpsDouble>(sd, dd, xy, fxy, tables.Float, 0, y, relative);
            }
            else
            {
                RemapKernels.Separable<double, double, float, RemapOpsDouble>(sd, dd, xy, fxy, tables.Float, k, 0, y, relative);
            }
        }
        else if (k == 2)
        {
            RemapKernels.Bilinear<T, float, float, RemapOpsFloat<T>>(s, d, xy, fxy, tables.Float, 0, y, relative);
        }
        else
        {
            RemapKernels.Separable<T, float, float, RemapOpsFloat<T>>(s, d, xy, fxy, tables.Float, k, 0, y, relative);
        }
    }

    /// <summary>Validates the element type and channel count for an interpolation of the warps and remap.</summary>
    public static InterpolationFlags Validate<T>(InterpolationFlags interpolation, int channels)
        where T : unmanaged
    {
        if (interpolation == InterpolationFlags.Area)
        {
            interpolation = InterpolationFlags.Linear;
        }

        if (interpolation is not (InterpolationFlags.Nearest or InterpolationFlags.Linear or InterpolationFlags.Cubic or InterpolationFlags.Lanczos4))
        {
            throw new ArgumentOutOfRangeException(nameof(interpolation), interpolation, "The warps and remap support Nearest, Linear, Cubic, Area and Lanczos4.");
        }

        Saturate.EnsureSupported<T>();
        if (interpolation != InterpolationFlags.Nearest && (typeof(T) == typeof(sbyte) || typeof(T) == typeof(int)))
        {
            throw new NotSupportedException($"The element type '{typeof(T)}' supports only nearest-neighbour interpolation.");
        }

        if (interpolation is InterpolationFlags.Cubic or InterpolationFlags.Lanczos4 && channels > 4)
        {
            throw new NotSupportedException("Cubic and Lanczos4 interpolation support at most 4 channels.");
        }

        return interpolation;
    }

    public static void ValidateBorder(BorderType border)
    {
        if (border is < BorderType.Constant or > BorderType.Transparent)
        {
            throw new ArgumentOutOfRangeException(nameof(border), border, "Unknown border type.");
        }
    }
}
