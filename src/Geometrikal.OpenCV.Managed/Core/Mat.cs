namespace Geometrikal.OpenCV.Managed.Core;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// A managed, stride-aware, OpenCV-style matrix.
///
/// Storage is backed by a managed T[] array. Mat instances can either own an
/// entire allocation or represent a view into another Mat's allocation.
///
/// No disposal is required.
/// </summary>
public sealed class Mat<T>
    where T : unmanaged
{
    private readonly T[] _buffer;
    private readonly int _offset;

    /// <summary>
    /// Allocates a new continuous matrix.
    /// </summary>
    public Mat(
        int width,
        int height,
        int channels = 1,
        bool clear = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);

        Width = width;
        Height = height;
        Channels = channels;
        Stride = checked(width * channels);

        int length = checked(Stride * height);

        _buffer = clear
            ? new T[length]
            : GC.AllocateUninitializedArray<T>(length);

        _offset = 0;
    }

    /// <summary>
    /// Wraps an existing managed array without copying.
    /// </summary>
    public Mat(
        T[] buffer,
        int width,
        int height,
        int channels = 1,
        int? stride = null,
        int offset = 0)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);

        int rowLength = checked(width * channels);
        int actualStride = stride ?? rowLength;

        if (actualStride < rowLength)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stride),
                "Stride cannot be smaller than the active row length.");
        }

        int requiredLength = checked(
            offset +
            ((height - 1) * actualStride) +
            rowLength);

        if (requiredLength > buffer.Length)
        {
            throw new ArgumentException(
                "The supplied buffer is too small for the requested matrix.",
                nameof(buffer));
        }

        _buffer = buffer;
        _offset = offset;

        Width = width;
        Height = height;
        Channels = channels;
        Stride = actualStride;
    }

    private Mat(
        T[] buffer,
        int offset,
        int width,
        int height,
        int channels,
        int stride)
    {
        _buffer = buffer;
        _offset = offset;

        Width = width;
        Height = height;
        Channels = channels;
        Stride = stride;
    }

    /// <summary>
    /// Width in pixels.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Height in rows.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Number of T values per pixel.
    /// </summary>
    public int Channels { get; }

    /// <summary>
    /// Number of T elements from the start of one row to the next.
    ///
    /// This is measured in elements, not bytes.
    /// </summary>
    public int Stride { get; }

    /// <summary>
    /// Number of active T elements in each row.
    /// </summary>
    public int RowLength => checked(Width * Channels);

    /// <summary>
    /// Number of logical pixels.
    /// </summary>
    public int PixelCount => checked(Width * Height);

    /// <summary>
    /// Number of active T elements, excluding row padding.
    /// </summary>
    public int ElementCount => checked(PixelCount * Channels);

    /// <summary>
    /// Number of bytes in each active row.
    /// </summary>
    public int RowBytes => checked(RowLength * Unsafe.SizeOf<T>());

    /// <summary>
    /// Number of bytes between the starts of consecutive rows.
    /// </summary>
    public int StepBytes => checked(Stride * Unsafe.SizeOf<T>());

    /// <summary>
    /// True when all active elements are contiguous with no row padding.
    /// </summary>
    public bool IsContinuous => Stride == RowLength;

    /// <summary>
    /// True when this matrix represents the complete backing array.
    /// </summary>
    public bool OwnsWholeBuffer =>
        _offset == 0 &&
        IsContinuous &&
        ElementCount == _buffer.Length;

    /// <summary>
    /// Returns the active elements in a row.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> GetRowSpan(int y)
    {
        ValidateRow(y);

        return _buffer.AsSpan(
            checked(_offset + (y * Stride)),
            RowLength);
    }

    /// <summary>
    /// Returns the active elements in a row as a read-only span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> GetReadOnlyRowSpan(int y)
    {
        ValidateRow(y);

        return _buffer.AsSpan(
            checked(_offset + (y * Stride)),
            RowLength);
    }

    /// <summary>
    /// Returns the entire matrix as one span.
    ///
    /// This is only valid for continuous matrices.
    /// </summary>
    public Span<T> GetSpan()
    {
        EnsureContinuous();

        return _buffer.AsSpan(_offset, ElementCount);
    }

    /// <summary>
    /// Returns the entire matrix as one read-only span.
    ///
    /// This is only valid for continuous matrices.
    /// </summary>
    public ReadOnlySpan<T> GetReadOnlySpan()
    {
        EnsureContinuous();

        return _buffer.AsSpan(_offset, ElementCount);
    }

    /// <summary>
    /// Returns durable memory for a continuous matrix.
    ///
    /// Use Memory when the view must be stored or passed through async code.
    /// Use Span for synchronous processing.
    /// </summary>
    public Memory<T> GetMemory()
    {
        EnsureContinuous();

        return _buffer.AsMemory(_offset, ElementCount);
    }

    /// <summary>
    /// Returns durable read-only memory for a continuous matrix.
    /// </summary>
    public ReadOnlyMemory<T> GetReadOnlyMemory()
    {
        EnsureContinuous();

        return _buffer.AsMemory(_offset, ElementCount);
    }

    /// <summary>
    /// Gets or sets an element by row and element-column.
    ///
    /// elementX is measured in T elements, not pixels.
    /// For RGB data, the columns are R,G,B,R,G,B...
    /// </summary>
    public ref T AtElement(int y, int elementX)
    {
        ValidateRow(y);

        if ((uint)elementX >= (uint)RowLength)
        {
            throw new ArgumentOutOfRangeException(nameof(elementX));
        }

        return ref _buffer[
            checked(_offset + (y * Stride) + elementX)];
    }

    /// <summary>
    /// Gets or sets a single-channel pixel.
    /// </summary>
    public ref T At(int y, int x)
    {
        EnsureSingleChannel();
        ValidateCoordinates(x, y);

        return ref _buffer[
            checked(_offset + (y * Stride) + x)];
    }

    /// <summary>
    /// Gets or sets one channel of a multi-channel pixel.
    /// </summary>
    public ref T At(int y, int x, int channel)
    {
        ValidateCoordinates(x, y);

        if ((uint)channel >= (uint)Channels)
        {
            throw new ArgumentOutOfRangeException(nameof(channel));
        }

        int index = checked(
            _offset +
            (y * Stride) +
            (x * Channels) +
            channel);

        return ref _buffer[index];
    }

    /// <summary>
    /// Creates a non-copying rectangular view.
    ///
    /// Changes to the ROI modify the original matrix.
    /// </summary>
    public Mat<T> Roi(
        int x,
        int y,
        int width,
        int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        if (x > Width - width)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        if (y > Height - height)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        int roiOffset = checked(
            _offset +
            (y * Stride) +
            (x * Channels));

        return new Mat<T>(
            _buffer,
            roiOffset,
            width,
            height,
            Channels,
            Stride);
    }

    /// <summary>
    /// Creates a non-copying rectangular view, as OpenCV <c>Mat(Rect)</c>.
    /// </summary>
    public Mat<T> Roi(Rect rect) => Roi(rect.X, rect.Y, rect.Width, rect.Height);

    /// <summary>
    /// Returns a copied, continuous matrix.
    /// </summary>
    public Mat<T> Clone()
    {
        var result = new Mat<T>(
            Width,
            Height,
            Channels,
            clear: false);

        CopyTo(result);

        return result;
    }

    /// <summary>
    /// Copies this matrix into another matrix of the same shape.
    /// </summary>
    public void CopyTo(Mat<T> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        EnsureSameShape(destination);

        if (IsContinuous && destination.IsContinuous)
        {
            GetReadOnlySpan().CopyTo(destination.GetSpan());
            return;
        }

        for (int y = 0; y < Height; y++)
        {
            GetReadOnlyRowSpan(y)
                .CopyTo(destination.GetRowSpan(y));
        }
    }

    /// <summary>
    /// Copies the elements selected by <paramref name="mask"/> into another matrix of the same
    /// shape, as OpenCV <c>copyTo(dst, mask)</c>. Unselected destination elements keep their
    /// values.
    /// </summary>
    /// <param name="destination">Destination matrix with the shape of this matrix.</param>
    /// <param name="mask">
    /// Byte mask of the same size. With one channel, a nonzero element selects the whole pixel;
    /// with this matrix's channel count, each element selects one channel.
    /// </param>
    public void CopyTo(Mat<T> destination, Mat<byte> mask)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(mask);
        EnsureSameShape(destination);
        Internal.Elementwise.ValidateMask(this, mask, allowPerChannel: true);

        for (int y = 0; y < Height; y++)
        {
            Internal.Elementwise.StoreMasked(
                GetReadOnlyRowSpan(y), mask.GetReadOnlyRowSpan(y), destination.GetRowSpan(y), Channels);
        }
    }

    /// <summary>
    /// Sets every element, or the elements selected by <paramref name="mask"/>, to a value, as
    /// OpenCV <c>setTo(value, mask)</c>. Component <c>c</c> of <paramref name="value"/> converts
    /// to the element type with <c>saturate_cast</c> and is stored in channel <c>c</c>.
    /// </summary>
    /// <param name="value">Value per channel; at most 4 channels.</param>
    /// <param name="mask">
    /// Optional byte mask of the same size, with one channel or this matrix's channel count.
    /// </param>
    public void SetTo(Scalar value, Mat<byte>? mask = null)
    {
        if (Channels > 4)
        {
            throw new InvalidOperationException("SetTo supports at most 4 channels.");
        }

        Internal.Elementwise.ValidateMask(this, mask, allowPerChannel: true);
        T[] row = Internal.Elementwise.UnrollScalar(
            Internal.ScalarConverter.ToRawColor<T>(value, Channels), RowLength);

        for (int y = 0; y < Height; y++)
        {
            if (mask is null)
            {
                row.CopyTo(GetRowSpan(y));
            }
            else
            {
                Internal.Elementwise.StoreMasked<T>(row, mask.GetReadOnlyRowSpan(y), GetRowSpan(y), Channels);
            }
        }
    }

    /// <summary>
    /// Fills every active matrix element.
    /// </summary>
    public void Fill(T value)
    {
        if (IsContinuous)
        {
            GetSpan().Fill(value);
            return;
        }

        for (int y = 0; y < Height; y++)
        {
            GetRowSpan(y).Fill(value);
        }
    }

    /// <summary>
    /// Clears every active matrix element.
    /// </summary>
    public void Clear()
    {
        if (IsContinuous)
        {
            GetSpan().Clear();
            return;
        }

        for (int y = 0; y < Height; y++)
        {
            GetRowSpan(y).Clear();
        }
    }

    /// <summary>
    /// Copies the matrix into a new packed managed array.
    /// </summary>
    public T[] ToArray()
    {
        var result = GC.AllocateUninitializedArray<T>(ElementCount);

        if (IsContinuous)
        {
            GetReadOnlySpan().CopyTo(result);
            return result;
        }

        int destinationOffset = 0;

        for (int y = 0; y < Height; y++)
        {
            ReadOnlySpan<T> row = GetReadOnlyRowSpan(y);
            row.CopyTo(result.AsSpan(destinationOffset, row.Length));
            destinationOffset += row.Length;
        }

        return result;
    }

    /// <summary>
    /// Returns the underlying array.
    ///
    /// The returned array may contain data outside this matrix when this Mat
    /// is an ROI or wraps a padded buffer.
    /// </summary>
    public T[] DangerousGetArray() => _buffer;

    /// <summary>
    /// Offset of the first matrix element inside the backing array.
    /// </summary>
    public int DangerousGetOffset() => _offset;

    /// <summary>
    /// Reinterprets a continuous matrix as another unmanaged element type.
    ///
    /// This copies no data, but the resulting object is backed by a new
    /// managed array representation and therefore cannot directly share the
    /// original T[] array. Prefer CastSpan for temporary reinterpretation.
    /// </summary>
    public Span<TTo> CastSpan<TTo>()
        where TTo : unmanaged
    {
        return MemoryMarshal.Cast<T, TTo>(GetSpan());
    }

    /// <summary>
    /// Reinterprets a continuous read-only matrix span.
    /// </summary>
    public ReadOnlySpan<TTo> CastReadOnlySpan<TTo>()
        where TTo : unmanaged
    {
        return MemoryMarshal.Cast<T, TTo>(GetReadOnlySpan());
    }

    /// <summary>
    /// Applies an operation to each row.
    /// </summary>
    public void ProcessRows(RowSpanAction<T> action)
    {
        ArgumentNullException.ThrowIfNull(action);

        for (int y = 0; y < Height; y++)
        {
            action(y, GetRowSpan(y));
        }
    }

    /// <summary>
    /// Applies a read-only operation to each row.
    /// </summary>
    public void ProcessRows(ReadOnlyRowSpanAction<T> action)
    {
        ArgumentNullException.ThrowIfNull(action);

        for (int y = 0; y < Height; y++)
        {
            action(y, GetReadOnlyRowSpan(y));
        }
    }

    public override string ToString()
    {
        return
            $"{nameof(Mat<T>)}<{typeof(T).Name}> " +
            $"{Width}x{Height}x{Channels}, " +
            $"stride={Stride}, continuous={IsContinuous}";
    }

    private void EnsureContinuous()
    {
        if (!IsContinuous)
        {
            throw new InvalidOperationException(
                "This operation requires a continuous matrix. " +
                "Call Clone() to create a packed copy.");
        }
    }

    private void EnsureSingleChannel()
    {
        if (Channels != 1)
        {
            throw new InvalidOperationException(
                "This operation requires a single-channel matrix.");
        }
    }

    private void EnsureSameShape(Mat<T> other)
    {
        if (Width != other.Width ||
            Height != other.Height ||
            Channels != other.Channels)
        {
            throw new ArgumentException(
                "Matrix dimensions and channel counts must match.",
                nameof(other));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ValidateRow(int y)
    {
        if ((uint)y >= (uint)Height)
        {
            throw new ArgumentOutOfRangeException(nameof(y));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ValidateCoordinates(int x, int y)
    {
        if ((uint)x >= (uint)Width)
        {
            throw new ArgumentOutOfRangeException(nameof(x));
        }

        if ((uint)y >= (uint)Height)
        {
            throw new ArgumentOutOfRangeException(nameof(y));
        }
    }
}

public delegate void RowSpanAction<T>(
    int y,
    Span<T> row)
    where T : unmanaged;

public delegate void ReadOnlyRowSpanAction<T>(
    int y,
    ReadOnlySpan<T> row)
    where T : unmanaged;