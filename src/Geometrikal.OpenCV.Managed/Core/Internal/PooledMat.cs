namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Buffers;

/// <summary>
/// A temporary packed matrix over an <see cref="ArrayPool{T}"/> array. <see cref="Mat"/>
/// must not be used or escape after <see cref="Dispose"/>.
/// </summary>
internal readonly struct PooledMat<T> : IDisposable
    where T : unmanaged
{
    private readonly T[] _array;

    private PooledMat(T[] array, Mat<T> mat)
    {
        _array = array;
        Mat = mat;
    }

    public Mat<T> Mat { get; }

    /// <summary>Rents an uninitialized matrix.</summary>
    public static PooledMat<T> Rent(int width, int height, int channels)
    {
        T[] array = ArrayPool<T>.Shared.Rent(checked(width * height * channels));
        return new PooledMat<T>(array, new Mat<T>(array, width, height, channels));
    }

    public void Dispose() => ArrayPool<T>.Shared.Return(_array);
}
