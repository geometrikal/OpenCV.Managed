// This implementation is ported from the OpenCV flood fill.
//
// Original source:
//   modules/imgproc/src/floodfill.cpp (Diff8uC1, Diff8uC3, DiffC1, DiffC3)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/floodfill.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;

/// <summary>
/// The pixel test of the gradient flood fill: pixel <c>a</c> joins the region when
/// <c>b - lo &lt;= a &lt;= b + up</c> in every channel, with the arithmetic of the OpenCV functor.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
internal interface IFloodDiff<T>
    where T : unmanaged
{
    /// <summary>Values per pixel.</summary>
    static abstract int Channels { get; }

    /// <summary>Tests pixel <paramref name="a"/> against pixel <paramref name="b"/>.</summary>
    bool Within(ref T a, ref T b);
}

/// <summary>OpenCV <c>Diff8uC1</c>: unsigned wrap-around range test.</summary>
internal readonly struct FloodDiff8uC1(byte lo, byte up) : IFloodDiff<byte>
{
    private readonly uint _lo = lo;
    private readonly uint _interval = (uint)(lo + up);

    public static int Channels => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref byte a, ref byte b) => (uint)(a - b + (int)_lo) <= _interval;
}

/// <summary>OpenCV <c>Diff8uC3</c>.</summary>
internal readonly struct FloodDiff8uC3(byte lo0, byte lo1, byte lo2, byte up0, byte up1, byte up2) : IFloodDiff<byte>
{
    private readonly uint _lo0 = lo0;
    private readonly uint _lo1 = lo1;
    private readonly uint _lo2 = lo2;
    private readonly uint _interval0 = (uint)(lo0 + up0);
    private readonly uint _interval1 = (uint)(lo1 + up1);
    private readonly uint _interval2 = (uint)(lo2 + up2);

    public static int Channels => 3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref byte a, ref byte b)
        => (uint)(a - b + (int)_lo0) <= _interval0
        && (uint)(Unsafe.Add(ref a, 1) - Unsafe.Add(ref b, 1) + (int)_lo1) <= _interval1
        && (uint)(Unsafe.Add(ref a, 2) - Unsafe.Add(ref b, 2) + (int)_lo2) <= _interval2;
}

/// <summary>OpenCV <c>DiffC1&lt;int&gt;</c>: the difference wraps on overflow.</summary>
internal readonly struct FloodDiff32sC1(int lo, int up) : IFloodDiff<int>
{
    private readonly int _lo = unchecked(-lo);
    private readonly int _up = up;

    public static int Channels => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref int a, ref int b)
    {
        int d = unchecked(a - b);
        return _lo <= d && d <= _up;
    }
}

/// <summary>OpenCV <c>DiffC3&lt;Vec3i&gt;</c>: each difference wraps on overflow.</summary>
internal readonly struct FloodDiff32sC3(int lo0, int lo1, int lo2, int up0, int up1, int up2) : IFloodDiff<int>
{
    private readonly int _lo0 = unchecked(-lo0);
    private readonly int _lo1 = unchecked(-lo1);
    private readonly int _lo2 = unchecked(-lo2);
    private readonly int _up0 = up0;
    private readonly int _up1 = up1;
    private readonly int _up2 = up2;

    public static int Channels => 3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref int a, ref int b)
    {
        int d0 = unchecked(a - b);
        int d1 = unchecked(Unsafe.Add(ref a, 1) - Unsafe.Add(ref b, 1));
        int d2 = unchecked(Unsafe.Add(ref a, 2) - Unsafe.Add(ref b, 2));
        return _lo0 <= d0 && d0 <= _up0
            && _lo1 <= d1 && d1 <= _up1
            && _lo2 <= d2 && d2 <= _up2;
    }
}

/// <summary>OpenCV <c>DiffC1&lt;float&gt;</c>: a NaN difference fails the test.</summary>
internal readonly struct FloodDiff32fC1(float lo, float up) : IFloodDiff<float>
{
    private readonly float _lo = -lo;
    private readonly float _up = up;

    public static int Channels => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref float a, ref float b)
    {
        float d = a - b;
        return _lo <= d && d <= _up;
    }
}

/// <summary>OpenCV <c>DiffC3&lt;Vec3f&gt;</c>.</summary>
internal readonly struct FloodDiff32fC3(float lo0, float lo1, float lo2, float up0, float up1, float up2) : IFloodDiff<float>
{
    private readonly float _lo0 = -lo0;
    private readonly float _lo1 = -lo1;
    private readonly float _lo2 = -lo2;
    private readonly float _up0 = up0;
    private readonly float _up1 = up1;
    private readonly float _up2 = up2;

    public static int Channels => 3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref float a, ref float b)
    {
        float d0 = a - b;
        float d1 = Unsafe.Add(ref a, 1) - Unsafe.Add(ref b, 1);
        float d2 = Unsafe.Add(ref a, 2) - Unsafe.Add(ref b, 2);
        return _lo0 <= d0 && d0 <= _up0
            && _lo1 <= d1 && d1 <= _up1
            && _lo2 <= d2 && d2 <= _up2;
    }
}

/// <summary>Pixel equality of the simple fill (OpenCV <c>floodFill_CnIR</c>) for integer elements.</summary>
internal readonly struct FloodEqualC1<T> : IFloodDiff<T>
    where T : unmanaged
{
    public static int Channels => 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref T a, ref T b) => EqualityComparer<T>.Default.Equals(a, b);
}

/// <summary>Three-channel pixel equality of the simple fill for integer elements.</summary>
internal readonly struct FloodEqualC3<T> : IFloodDiff<T>
    where T : unmanaged
{
    public static int Channels => 3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Within(ref T a, ref T b)
        => EqualityComparer<T>.Default.Equals(a, b)
        && EqualityComparer<T>.Default.Equals(Unsafe.Add(ref a, 1), Unsafe.Add(ref b, 1))
        && EqualityComparer<T>.Default.Equals(Unsafe.Add(ref a, 2), Unsafe.Add(ref b, 2));
}
