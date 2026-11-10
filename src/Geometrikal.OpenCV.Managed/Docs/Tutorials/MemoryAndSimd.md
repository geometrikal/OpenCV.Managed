# Tutorial: memory handling and SIMD in the filter engines

This tutorial explains the techniques that made filtering and morphology 7× to 200× faster.
It describes each technique, shows its use in the code, and gives the measured effect.

Measurements are from `Geometrikal.OpenCV.Managed.Benchmark` at 1024×1024, single thread, on an
Intel Core i9-11900 with .NET 10. `Docs/Notes/PerformanceAnalysis.md` has the full data.

## 1. The starting point

The first port was a direct scalar translation. This is the old inner loop of erode/dilate
with an arbitrary structuring element:

```csharp
for (int y = 0; y < height; y++)
{
    Span<T> outRow = dst.GetRowSpan(y);
    for (int x = 0; x < width; x++)
    {
        for (int c = 0; c < channels; c++)
        {
            ReadOnlySpan<T> first = padded.GetReadOnlyRowSpan(y + coordY[0]);
            T m = first[((x + coordX[0]) * channels) + c];
            for (int k = 1; k < nz; k++)
            {
                ReadOnlySpan<T> src = padded.GetReadOnlyRowSpan(y + coordY[k]);
                m = MinMax(m, src[((x + coordX[k]) * channels) + c], erode);
            }

            outRow[(x * channels) + c] = m;
        }
    }
}
```

This loop has four costs for each output element:

1. It creates one row span per kernel tap. `GetReadOnlyRowSpan` validates the row index,
   does `checked` arithmetic, and validates the span range.
2. Every `src[...]` index has a bounds check.
3. `MinMax(..., erode)` tests the `erode` flag at run time for every element.
4. It processes one element per iteration. The CPU can process 32 bytes per instruction.

The function also allocated a padded copy of the source and, for rectangular kernels, a
full-size intermediate image on every call.

## 2. Memory: pooled temporaries

### What it is

`ArrayPool<T>.Shared` is a .NET cache of arrays. `Rent` returns an array of at least the
requested length. `Return` puts the array back for the next caller. After warm-up, `Rent`
does not allocate.

`PooledMat<T>` (`Core/Internal/PooledMat.cs`) wraps a rented array in a `Mat<T>`:

```csharp
internal readonly struct PooledMat<T> : IDisposable
    where T : unmanaged
{
    public static PooledMat<T> Rent(int width, int height, int channels)
    {
        T[] array = ArrayPool<T>.Shared.Rent(checked(width * height * channels));
        return new PooledMat<T>(array, new Mat<T>(array, width, height, channels));
    }

    public void Dispose() => ArrayPool<T>.Shared.Return(_array);
}
```

### How it is used

Each filter rents its padded source and returns it with `using`:

```csharp
using PooledMat<T> pooled = Borders.CopyMakeBorderPooled(source, top, bottom, left, right, borderType, constant);
Mat<T> padded = pooled.Mat;
```

Row buffers that are not matrices use `ArrayPool<T>.Shared` directly, with `try`/`finally`.

Rules:

- Use pooled memory only for temporaries. The returned result is a normal `Mat<T>` that the
  caller owns. The public API does not change.
- Do not let `pooled.Mat` escape the method. After `Dispose`, another caller can rent the
  same array.
- A rented array is not cleared. Write every element before you read it.
- A rented array can be longer than requested. Use the `Mat<T>` dimensions, not
  `array.Length`.

### Why it is faster

.NET puts every array of 85,000 bytes or more on the large object heap (LOH). The garbage
collector collects the LOH only in a generation 2 collection, which scans the full heap.
A 1024×1024 byte image is 1 MB. Before pooling, 50 calls of `Erode` caused 19 generation 2
collections. With pooling, the only large allocation is the result.

The result is still allocated. At 256×256 with three channels, the result is 192 KB and
goes on the LOH. The gen2 collections that LOH allocations trigger are now most of the cost
of `Erode` U8C3 (2.5× cv2) and `CopyMakeBorder` U8C3 (5× cv2). The application can raise
the LOH threshold (`System.GC.LOHThreshold` in `runtimeconfig.json`). With a 2 MB threshold,
`Erode` U8C3 at 256×256 drops from 85 µs to 33 µs for the same allocated bytes.

## 3. Memory: references instead of spans in inner loops

### What it is

`Span<T>` is a safe view of memory. Each indexer access checks the index against the length.
The JIT removes the check only when it can prove the index is in range, for example in
`for (int i = 0; i < span.Length; i++)`. Indices such as `i + k * step` defeat this proof.

`MemoryMarshal.GetReference(span)` returns a `ref T` to the first element.
`Unsafe.Add(ref r, i)` and `Vector.LoadUnsafe(ref r, i)` read at an offset with no check.

### How it is used

The engines get one reference per row or per image, outside the element loop. They never
create spans inside the element loop:

```csharp
ref T src = ref MemoryMarshal.GetReference(padded.GetReadOnlySpan());
for (int y = 0; y < padded.Height; y++)
{
    FoldTaps<T, TOp>(ref Unsafe.Add(ref src, y * paddedStride), cn, kw,
                     ref Unsafe.Add(ref mid, y * rowLength), rowLength);
}
```

The code that uses `Unsafe` is responsible for the range. The loop bounds must prove that
every access is inside the buffer. Padding makes this possible: the padded source has
`kw - 1` extra columns and `kh - 1` extra rows, so `i + (taps - 1) * step` stays inside
the row for every `i < length`. `Tests/Regression` runs odd widths and heights to catch
range errors at the tails.

### Why it is faster

Removing the per-tap span creation and the bounds checks gave 2× to 4× before any SIMD:

| Operation | Before | Overhead removed, scalar |
|---|---|---|
| MedianBlur 3×3 | 88.1 ms | 43.0 ms |
| Dilate ellipse 5×5 | 75.5 ms | 30.1 ms |
| Sobel dx U8→S16 | 10.8 ms | 2.9 ms |

## 4. Compile-time operation selection

### What it is

A generic method with a `struct` type parameter gets its own machine code for each struct.
The JIT knows the exact type and inlines its methods. A static abstract interface member
lets the code call the operation without an instance.

`Core/Internal/MinMaxOps.cs` defines the operation:

```csharp
internal interface IMinMaxOp
{
    static abstract T Apply<T>(T a, T b) where T : unmanaged;
    static abstract Vector<T> Apply<T>(Vector<T> a, Vector<T> b) where T : unmanaged;
}

internal readonly struct MinOp : IMinMaxOp { ... }   // std::min
internal readonly struct MaxOp : IMinMaxOp { ... }   // std::max
```

### How it is used

The engine selects the operation once, outside all loops:

```csharp
if (erode) ApplySeparableRect<T, MinOp>(padded, dst, kw, kh);
else       ApplySeparableRect<T, MaxOp>(padded, dst, kw, kh);
```

Inside, `TOp.Apply(a, b)` compiles to one `min` or `max` instruction.

The same technique removes type tests. `typeof(T) == typeof(byte)` is a JIT constant for
each `T`. The JIT deletes the branches that do not apply.

### Why it is faster

The old code tested `erode` for every element. The new code has no branch in the inner loop.
A branch-free loop is also a precondition for SIMD.

## 5. SIMD with `Vector<T>`

### What it is

SIMD (single instruction, multiple data) instructions operate on a register that holds
several elements. `System.Numerics.Vector<T>` is a register-sized vector. Its size depends
on the CPU:

| Hardware | `Vector<T>` size | Bytes per operation | Ints or floats per operation |
|---|---|---|---|
| AVX2 (this machine) | 256 bits | 32 | 8 |
| SSE2 / ARM NEON | 128 bits | 16 | 4 |
| No acceleration | software fallback | — | — |

.NET uses 256 bits on AVX-512 hardware by default. `DOTNET_PreferredVectorBitWidth=512`
selects 512 bits.

`Vector.IsHardwareAccelerated` is true when the operations map to real instructions.
`MinMaxOps.IsVectorized<T>()` combines this with `Vector<T>.IsSupported`. Both are JIT
constants, so the test costs nothing.

### How it is used

Every vector loop has the same form: a vector body for whole vectors, then a scalar tail
for the remaining elements. This is `MorphEngine.FoldTaps`:

```csharp
int i = 0;
if (MinMaxOps.IsVectorized<T>())
{
    int vc = Vector<T>.Count;
    for (; i <= length - vc; i += vc)
    {
        Vector<T> acc = Vector.LoadUnsafe(ref s, (nuint)i);
        for (int k = 1; k < taps; k++)
        {
            acc = TOp.Apply(acc, Vector.LoadUnsafe(ref s, (nuint)(i + (k * step))));
        }

        acc.StoreUnsafe(ref d, (nuint)i);
    }
}

for (; i < length; i++)
{
    T acc = Unsafe.Add(ref s, i);
    for (int k = 1; k < taps; k++)
    {
        acc = TOp.Apply(acc, Unsafe.Add(ref s, i + (k * step)));
    }

    Unsafe.Add(ref d, i) = acc;
}
```

The scalar tail does the same operations in the same order as one vector lane. The tail
also runs the whole row when SIMD is not available.

The median filter applies the same idea to a sorting network. Each of the nine inputs is a
vector of 32 neighbouring pixels. Each compare-exchange sorts 32 pixel windows at once:

```csharp
Vector<T> p0 = Vector.LoadUnsafe(ref s0, l);   // row 0, left neighbours
Vector<T> p1 = Vector.LoadUnsafe(ref s0, c);   // row 0, centre
...
Network9(ref p0, ref p1, ..., ref p8);         // 19 MinMax steps, unrolled
p4.StoreUnsafe(ref dr, c);                     // 32 medians
```

Linear filters must widen the data. A sum of 25 bytes times a coefficient does not fit in a
byte. `FilterVectors` (`ImgProc/Filtering/Internal/FilterVectors.cs`) provides:

- `WidenToInt32` / `WidenToSingle`: one `Vector<byte>` (32 bytes) to four `Vector<int>` or
  four `Vector<float>` (8 elements each), in element order.
- `CvRound`: OpenCV `cvRound` for each lane (round half to even; out-of-range and NaN give
  `int.MinValue`).
- `SaturateToByte` / `SaturateToInt16`: clamp, then narrow back, as OpenCV `saturate_cast`.

A U8 filter therefore runs: load 32 bytes, widen to four int or float vectors, multiply-add
each tap, round, saturate, narrow, store 32 bytes.

### Why it is faster

One AVX2 instruction does the work of 32 byte operations or 8 float operations. Min, max,
add and multiply are each one instruction. With the overhead from sections 3 and 4 removed,
the gain is close to the lane count:

| Operation | Scalar, overhead removed | `Vector<T>` | Gain |
|---|---|---|---|
| MedianBlur 3×3 | 43.0 ms | 2.19 ms | 20× |
| Erode rect 3×3 | 9.7 ms | 1.55 ms | 6× |
| Dilate ellipse 5×5 | 30.1 ms | 1.55 ms | 19× |

These are experiment numbers that still included the old padding and allocations. With
pooling (section 2) and span-based border copies, the library results are:

| Operation | Original | Current | cv2, 1 thread, SSE, no IPP |
|---|---|---|---|
| MedianBlur 3×3 | 99.8 ms | 1.44 ms | 0.72 ms |
| Erode rect 3×3 | 12.9 ms | 0.21 ms | 0.39 ms |
| Dilate ellipse 5×5 | 63.6 ms | 0.31 ms | 0.50 ms |
| GaussianBlur 5×5 U8 | 10.6 ms | 0.72 ms | 0.80 ms |
| Sobel dx U8→S16 | 9.53 ms | 0.70 ms | 0.73 ms |
| CopyMakeBorder 16 px C3 | 2.22 ms | 0.32 ms | 0.76 ms |

## 6. Scalar tails

### What it is

A vector loop processes `Vector<T>.Count` elements per iteration. The remaining elements at
the end of a row go through a scalar tail. With AVX2 and bytes, the tail is up to 31
elements per row.

### Why it matters

A scalar tail element can cost 50 to 100 times more than one lane of the vector body. On
narrow rows the tail dominates. The 3×3 median at 256×256 had a tail of 31 pixels on every
256-pixel row. The tail used the generic scalar network. It took more time than the vector
body for the other 225 pixels.

### How it is used

When the operation is a pure function of its inputs, the final vector overlaps the previous
one. It writes the same values again to the overlapped elements:

```csharp
int last = n - cn - vc;
for (; j <= last + vc - 1 && last >= cn; j += vc)
{
    // The final vector overlaps the previous one; it rewrites equal values.
    j = Math.Min(j, last);
    ...
}
```

The overlap is valid only when the output does not feed back into the input of the same
loop. A running sum (box filter column pass) cannot use it.

Effect, BenchmarkDotNet short run:

| Operation | Scalar tail | Overlapping final vector | cv2, 1 thread |
|---|---|---|---|
| MedianBlur 3×3, 256×256 | 158 µs | 16 µs | 72 µs |
| MedianBlur 3×3, 1024×1024 | 1.44 ms | 0.24 ms | 0.72 ms |
| MedianBlur 5×5, 1024×1024 | 4.35 ms | 1.99 ms | 3.63 ms |

The 3×3 result also includes Paeth's method for bytes: sort each column of three once, then
take `med3(max of lows, median of middles, min of highs)`. Neighbouring outputs share the
column sorts, so each output vector needs about 18 min/max operations instead of 38.

## 7. Register blocking

### What it is

A tap-major loop processes one kernel tap for the whole row, then the next tap:

```csharp
for (int k = 0; k < taps; k++)
    for (int i = 0; i < width; i += vc)
        sums[i] += coeff[k] * src[i + offset[k]];   // load sums, compute, store sums
```

Every tap loads and stores the whole sum row. A block-major loop keeps a block of
accumulators in registers and loops over the taps inside the block:

```csharp
for (int i = 0; i < width; i += 4 * vc)
{
    Vector<float> a0 = delta, a1 = delta, a2 = delta, a3 = delta;
    for (int k = 0; k < taps; k++)
    {
        var c = new Vector<float>(coeff[k]);
        a0 += c * Load(src, i + offset[k]);
        a1 += c * Load(src, i + offset[k] + vc);
        ...
    }

    Store(sums, i, a0, a1, a2, a3);
}
```

Each lane still adds the taps in the same order, so the result is bit-exact. Four independent
accumulators also hide the latency of the add instruction.

### Where it is used

`Filter2DEngine.ConvolveRows`, `SepFilterEngine.RowFloat` (generic branch) and
`SepFilterEngine.ColumnGeneric`. A byte source for `Filter2D` is converted to float once,
because widening the same bytes for every tap costs more than one conversion pass.

### Effect

`Filter2D` 11×11 U8 at 1024×1024 went from 13.2 ms to 6.1 ms (cv2: 12.6 ms). Float
`GaussianBlur` 15×15 did not change. That case is limited by memory traffic: the padded copy,
the full-size row-pass image and the 4 MB output. A row ring buffer (`PerformanceAnalysis.md`,
step E) addresses this.

## 8. Multithreading

### What it is

`CvParallel.MaxDegreeOfParallelism` (`Core/CvParallel.cs`) is the public setting, as OpenCV
`cv::setNumThreads`. The default is `Environment.ProcessorCount`. The value 1 disables
multithreading.

`ParallelRows.For(count, workPerRow, body)` (`Core/Internal/ParallelRows.cs`) splits the rows
into contiguous stripes and calls `body(start, end)` for each stripe with `Parallel.For`. It
uses one stripe for each 64 K units of estimated work, up to the thread limit. Small images
stay on the calling thread. The caller's work estimate must reflect real cost: morphology
divides its element count by `Vector<T>.Count`, because one min/max instruction covers that
many elements and the pass is memory-bound. Without this, Erode 3×3 at 1024×1024 took 338 µs
with 16 threads and 228 µs with one.

### How it is used

Each engine moves the body of its row loop into a lambda:

```csharp
ParallelRows.For(dst.Height, (long)rowLength * kh, (y0, y1) =>
{
    ref T mid = ref rowPass[0];          // ref locals are declared inside the lambda
    for (int y = y0; y < y1; y++)
    {
        ...
    }
});
```

Rules:

- Call a static method from the lambda and pass values as parameters:
  `ParallelRows.For(h, work, (y0, y1) => ColumnRows(buffer, dst, ky, width, y0, y1));`.
  A loop written inside the lambda reads every captured variable from the closure object
  on each iteration. In the separable engine this cost 25–45%.
- A lambda cannot capture a `ref` local or a `Span<T>`. Declare them inside the stripe.
- Scratch buffers (sum rows, column histograms) belong to one stripe. Rent them inside the
  lambda.
- Each output row must not depend on the processing of another output row. A running sum
  does: the box filter column pass and the median histogram carry state from row to row.
  Each stripe seeds its own state from the rows above it. This is exact for integer sums and
  for histograms. The box filter for float sources uses `double` running sums, whose rounding
  depends on the history. That path stays on one thread.

### Verification

`Tests/Core/CvParallelTests` runs every parallel path on a 517×301 image with 1 thread and with
8 threads, and requires identical bytes. A test of this kind must be able to fail: seeding the
box filter stripes from row 0 instead of the stripe start fails 6 of its cases.

## 9. Keeping the results bit-exact

The port must produce the same bytes as OpenCV. SIMD must not change any result.

- **Integer min, max, add and multiply** give the same result in any order. They are exact.
- **Float arithmetic** depends on the order of operations. Each vector lane does the same
  operations in the same order as the scalar code. The lanes are independent, so the result
  of each lane equals the scalar result.
- **Fused multiply-add.** The .NET JIT does not contract `a * b + c` into one FMA
  instruction. An FMA rounds once instead of twice and gives a different result. Code must
  not call `Vector.FusedMultiplyAdd` or `Fma` in a path that must match OpenCV.
- **Rounding.** `Vector.Round` rounds half to even, as `cvRound` does. `FilterVectors.CvRound`
  adds the out-of-range and NaN handling.
- **Float min/max.** `Vector.Min` does not have `std::min` semantics for NaN and ±0. `MinOp`
  and `MaxOp` use `ConditionalSelect(LessThan(...))` for floats, which matches `b < a ? b : a`.

`Tests/Regression` verifies this. It hashes the output of 924 cases on wide images with odd
sizes and compares the hashes with the output of the original scalar code (commit
`f613774`). The cv2 golden tests also stay green.

## 10. Checklist for a new vectorized loop

1. Get `ref T` row references outside the element loop. Do not create spans inside it.
2. Select operations and types with generic `struct` parameters or `typeof(T)` tests.
3. Test `MinMaxOps.IsVectorized<T>()` before the vector body.
4. Write the vector body for `i <= length - Vector<T>.Count`, then a scalar tail.
5. Make the tail do the same operations, in the same order, as one lane. When the loop has no
   running state, replace the tail with an overlapping final vector.
6. For multi-tap filters, keep a block of accumulators in registers across the taps.
7. Make sure padding or loop bounds keep every `Unsafe` or `LoadUnsafe` access in range.
8. Rent temporaries from `ArrayPool<T>` or `PooledMat<T>`. Do not let them escape.
9. Run the row loop through `ParallelRows.For`. Put the stripe body in a static method with
   parameters; do not write the hot loop inside the lambda. Keep `ref` locals and scratch
   buffers inside the stripe. Seed running state per stripe, or keep the loop on one thread.
   Divide the work estimate by `Vector<T>.Count` for memory-bound SIMD loops.
10. Add regression cases with widths that are not multiples of the vector size, and a
    `CvParallelTests` case for the new path. Break the code on purpose once to confirm that
    the new test fails.
11. Build every edited project before a long benchmark run, and compare numbers only between
    runs on a quiet machine. Some cases vary by up to 2× between runs while each run has a
    low coefficient of variation.
