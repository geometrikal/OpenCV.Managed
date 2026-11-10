namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Commits the pages of a new destination on the calling thread. Page faults taken by several
/// threads at once on a fresh large-object-heap array cost more than the same faults on one thread.
/// </summary>
internal static class PageTouch
{
    private const int PageBytes = 4096;

    /// <summary>Minimum destination size that is touched.</summary>
    private const long MinBytes = 1 << 16;

    /// <summary>
    /// Writes one element per page of a continuous matrix; the caller overwrites all elements later.
    /// A matrix with row padding is not touched, because the padding can belong to other data.
    /// </summary>
    public static void Rows<T>(Mat<T> dst)
        where T : unmanaged
    {
        long bytes = (long)dst.Stride * dst.Height * Unsafe.SizeOf<T>();
        if (bytes < MinBytes || !dst.IsContinuous)
        {
            return;
        }

        int step = Math.Max(1, PageBytes / Unsafe.SizeOf<T>());
        T[] array = dst.DangerousGetArray();
        ref T first = ref MemoryMarshal.GetArrayDataReference(array);
        long start = dst.DangerousGetOffset();
        long end = start + ((long)(dst.Height - 1) * dst.Stride) + dst.RowLength;
        for (long i = start; i < end; i += step)
        {
            Unsafe.Add(ref first, (nint)i) = default;
        }
    }
}
