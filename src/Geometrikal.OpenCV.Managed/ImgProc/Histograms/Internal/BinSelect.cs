namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;

/// <summary>
/// Branch-free bin selection. Out-of-range pixels are frequent and irregular, so conditional
/// branches in the counting loops mispredict.
/// </summary>
internal static class BinSelect
{
    /// <summary><c>min(a, limit)</c> for <c>0 &lt;= a, limit &lt; 2^30</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Min(int a, int limit)
    {
        int d = a - limit;
        return limit + (d & (d >> 31));
    }

    /// <summary><c>i</c> if <c>0 &lt;= i &lt; total</c>, else <c>total</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Clip(int i, int total)
    {
        int inside = ((i - total) >> 31) & ~(i >> 31);
        return (i & inside) | (total & ~inside);
    }

    /// <summary>-1 when the mask byte is zero, else 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Unselected(byte mask) => -(int)(((uint)mask - 1) >> 31);
}
