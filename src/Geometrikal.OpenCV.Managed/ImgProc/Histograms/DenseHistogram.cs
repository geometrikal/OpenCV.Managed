namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A dense N-dimensional histogram of <c>float</c> bins, equivalent to the OpenCV <c>CV_32F</c>
/// <c>Mat</c> that <c>calcHist</c> produces. Bins are stored in row-major order: the last
/// dimension varies fastest.
/// </summary>
/// <remarks>
/// <see cref="Mat{T}"/> has two dimensions, so histograms with three or more dimensions need this
/// type. <see cref="ToMat"/> and <see cref="FromMat"/> convert one- and two-dimensional histograms.
/// A one-dimensional histogram corresponds to the OpenCV <c>N x 1</c> matrix.
/// </remarks>
public sealed class DenseHistogram
{
    /// <summary>OpenCV <c>CV_MAX_DIM</c>.</summary>
    public const int MaxDims = 32;

    private readonly int[] _sizes;
    private readonly float[] _data;

    /// <summary>Creates a histogram with all bins set to zero.</summary>
    /// <param name="sizes">Number of bins in each dimension; 1 to <see cref="MaxDims"/> positive values.</param>
    /// <exception cref="ArgumentException">The dimension count is invalid or the bin count overflows.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A size is not positive.</exception>
    public DenseHistogram(params ReadOnlySpan<int> sizes)
    {
        if (sizes.Length is < 1 or > MaxDims)
        {
            throw new ArgumentException($"A histogram has 1 to {MaxDims} dimensions.", nameof(sizes));
        }

        long total = 1;
        foreach (int s in sizes)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(s, nameof(sizes));
            total *= s;
            if (total > Array.MaxLength)
            {
                throw new ArgumentException("The histogram has too many bins.", nameof(sizes));
            }
        }

        _sizes = sizes.ToArray();
        _data = new float[total];
    }

    private DenseHistogram(int[] sizes, float[] data)
    {
        _sizes = sizes;
        _data = data;
    }

    /// <summary>Number of dimensions.</summary>
    public int Dims => _sizes.Length;

    /// <summary>Number of bins in each dimension.</summary>
    public ReadOnlySpan<int> Sizes => _sizes;

    /// <summary>Total number of bins.</summary>
    public int Length => _data.Length;

    /// <summary>All bins in row-major order.</summary>
    public Span<float> Values => _data;

    /// <summary>Bin of a one-dimensional histogram.</summary>
    public ref float At(int i0)
    {
        EnsureDims(1);
        return ref _data[Check(i0, 0)];
    }

    /// <summary>Bin of a two-dimensional histogram.</summary>
    public ref float At(int i0, int i1)
    {
        EnsureDims(2);
        return ref _data[(Check(i0, 0) * _sizes[1]) + Check(i1, 1)];
    }

    /// <summary>Bin of a three-dimensional histogram.</summary>
    public ref float At(int i0, int i1, int i2)
    {
        EnsureDims(3);
        return ref _data[(((Check(i0, 0) * _sizes[1]) + Check(i1, 1)) * _sizes[2]) + Check(i2, 2)];
    }

    /// <summary>Bin at an index with one entry per dimension.</summary>
    public ref float At(params ReadOnlySpan<int> index)
    {
        EnsureDims(index.Length);
        int offset = 0;
        for (int i = 0; i < index.Length; i++)
        {
            offset = (offset * _sizes[i]) + Check(index[i], i);
        }

        return ref _data[offset];
    }

    /// <summary>Returns a copy with the same sizes and bins.</summary>
    public DenseHistogram Clone() => new((int[])_sizes.Clone(), (float[])_data.Clone());

    /// <summary>
    /// Copies a one- or two-dimensional histogram into a new matrix. One dimension gives an
    /// <c>N x 1</c> matrix (width 1); two dimensions give <c>Sizes[0]</c> rows and
    /// <c>Sizes[1]</c> columns.
    /// </summary>
    /// <exception cref="InvalidOperationException">The histogram has more than two dimensions.</exception>
    public Mat<float> ToMat()
    {
        if (Dims > 2)
        {
            throw new InvalidOperationException("Only one- and two-dimensional histograms convert to a matrix.");
        }

        int width = Dims == 1 ? 1 : _sizes[1];
        return new Mat<float>((float[])_data.Clone(), width, _sizes[0]);
    }

    /// <summary>
    /// Creates a two-dimensional histogram with <c>Height</c> x <c>Width</c> bins from a
    /// single-channel matrix. The bins are copied.
    /// </summary>
    /// <exception cref="ArgumentException">The matrix has more than one channel.</exception>
    public static DenseHistogram FromMat(Mat<float> matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Channels != 1)
        {
            throw new ArgumentException("The matrix must have one channel.", nameof(matrix));
        }

        return new DenseHistogram([matrix.Height, matrix.Width], matrix.ToArray());
    }

    internal float[] Data => _data;

    internal int[] SizeArray => _sizes;

    internal bool SameSizes(DenseHistogram other) => _sizes.AsSpan().SequenceEqual(other._sizes);

    private void EnsureDims(int dims)
    {
        if (dims != _sizes.Length)
        {
            throw new ArgumentException($"The histogram has {_sizes.Length} dimensions.");
        }
    }

    private int Check(int i, int dim)
    {
        if ((uint)i >= (uint)_sizes[dim])
        {
            throw new ArgumentOutOfRangeException(nameof(i));
        }

        return i;
    }
}
