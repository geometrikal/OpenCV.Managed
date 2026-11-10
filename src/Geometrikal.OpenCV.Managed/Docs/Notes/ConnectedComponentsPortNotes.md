# Connected components port notes

Semantics of `ConnectedComponentLabeling` (OpenCV `connectedComponents` and
`connectedComponentsWithStats`), the label numbering rules, the parallel strategy, the
deliberate differences from OpenCV 5.0, and the golden constraints. Read this before you change
a labeling scan or the connected components golden generator.

## Sources

- Port source: OpenCV 5.0, `modules/imgproc/src/connectedcomponents.cpp`.
- Golden reference: cv2 4.5.5.
- Files:
  - `ImgProc/ConnectedComponents/ConnectedComponentLabeling.cs`: public API and validation.
  - `ImgProc/ConnectedComponents/ConnectedComponentsAlgorithm.cs`,
    `ConnectedComponentsStat.cs`: enums with the OpenCV numeric values.
  - `ImgProc/ConnectedComponents/ConnectedComponentsResult.cs`: result records.
  - `ImgProc/ConnectedComponents/Internal/CclScans.cs`: first scan (pixel and block),
    union-find, stripe merge, flattening.
  - `ImgProc/ConnectedComponents/Internal/CclRelabel.cs`: second scan and statistics.
  - `ImgProc/ConnectedComponents/Internal/CclEngine.cs`: scan selection, stripes, statistics
    merge, statistics matrices.

## API

| Method | OpenCV equivalent | Result |
|---|---|---|
| `ConnectedComponents<TLabel>(image, connectivity = 8, algorithm = Default)` | `connectedComponents` | `ConnectedComponentsResult<TLabel>` with a new label image |
| `ConnectedComponents<TLabel>(image, labels, connectivity = 8, algorithm = Default)` | `connectedComponents` | Writes `labels`; returns the label count |
| `ConnectedComponentsWithStats<TLabel>(image, connectivity = 8, algorithm = Default)` | `connectedComponentsWithStats` | `ConnectedComponentsStatsResult<TLabel>` |

- `image` is `Mat<byte>` or `Mat<sbyte>`, one channel. A nonzero element is foreground. A
  negative `sbyte` is foreground.
- `TLabel` is `int` (`CV_32S`) or `ushort` (`CV_16U`). Other types throw
  `NotSupportedException`.
- `labels` must have the image size and one channel. It can be an ROI.
- `Stats` is a `Mat<int>` with 5 columns and one row per label. `ConnectedComponentsStat`
  indexes the columns (`Left`, `Top`, `Width`, `Height`, `Area`).
- `Centroids` is a `Mat<double>` with 2 columns (x, y) and one row per label.
- The label count includes the background label 0.
- `Mat<T>` cannot be empty. The OpenCV empty-image case does not apply.

## Validation

| Condition | Exception |
|---|---|
| Null image or labels | `ArgumentNullException` |
| Image or labels with more than one channel; labels of another size | `ArgumentException` |
| Connectivity other than 4 or 8 | `ArgumentOutOfRangeException` |
| Algorithm outside `-1..5` | `ArgumentOutOfRangeException` |
| `TLabel` other than `int` or `ushort` | `NotSupportedException` |
| `ushort` provisional label overflow | `OverflowException` |

## Label numbering

The partition into components is the same for every algorithm. The numbering is not.

| Connectivity | Algorithm | OpenCV scan | Numbering |
|---|---|---|---|
| 8 | `Wu`, `Sauf` | SAUF | Raster order of the first pixel |
| 8 | `Grana`, `Bbdt` | BBDT | Raster order of the first 2x2 block |
| 8 | `Default`, `Bolelli`, `Spaghetti` | Spaghetti | Raster order of the first 2x2 block |
| 4 | `Wu`, `Sauf`, `Grana`, `Bbdt` | SAUF 4C | Raster order of the first pixel |
| 4 | `Default`, `Bolelli`, `Spaghetti` | Spaghetti4C | Raster order of the first pixel |

- Block `(y / 2, x / 2)` contains pixel `(y, x)`. All foreground pixels of one block are
  8-connected.
- Example, 8-connectivity, image `[[0,0,0,0,1],[1,0,0,0,0]]`: `Wu` gives `[[0,0,0,0,1],[2,0,0,0,0]]`;
  `Default` gives `[[0,0,0,0,2],[1,0,0,0,0]]`.
- Cause: the scans create provisional labels in scan order. Union-find keeps the smallest
  provisional label as the root. `flattenL` numbers the roots in increasing order. The final
  label of a component is therefore the rank of its first pixel (pixel scan) or first block
  (block scan).

## Scans

The port implements three first scans. Each gives the OpenCV numbering and the OpenCV
provisional label count of the algorithms that it replaces.

| Kernel | Serves | New provisional label when |
|---|---|---|
| `Pixel4Row` | All algorithms with 4-connectivity | `x` is foreground; `q` (above) and `s` (left) are background |
| `Pixel8Row` | `Wu`, `Sauf` with 8-connectivity | `x` is foreground; `p`, `q`, `r` (above row) and `s` are background |
| `Block8Row` | `Default`, `Grana`, `Bolelli`, `Bbdt`, `Spaghetti` with 8-connectivity | Block X has a foreground pixel and connects to none of the blocks P, Q, R, S |

- Block names follow `LabelingBolelli`: P, Q, R are the blocks above-left, above and
  above-right; S is the block to the left; X is the current block `{o, p, s, t}`.
- Block connection rules: X~P iff `h && o`; X~Q iff `(i || j) && (o || p)`; X~R iff `k && p`;
  X~S iff `(n || r) && (o || s)`. Pixels outside the image are background.
- The port writes the rules as if-chains, not as the generated Spaghetti and BBDT forests. The
  forests take the same decisions.
- A union is skipped when an earlier step already connected the two sets. Example: with
  4-connectivity, `x` takes the label of `s` without a union when `q` and the pixel above `s`
  are foreground. A skipped union does not change the partition, the roots or the numbering.
- Union-find: `set_union` with the smaller root, path compression on both paths.
- Second scan: final label `P[provisional]` for every pixel. The block scan stores one
  provisional label per block in a separate buffer and expands it to the foreground pixels.

## Statistics

- Accumulation: per horizontal run of equal final labels, including the background label 0.
  A run `[c0, c1]` on row `y` adds `c1 - c0 + 1` to the area, `(c0 + c1) * length / 2` to the
  x sum and `y * length` to the y sum. Sums are exact `ulong` values.
- Finish, as OpenCV `CCStatsOp::finish`:
  - Area > 0: width and height from the minimum and maximum coordinates; centroid = sum / area
    in `double`.
  - Area = 0 (label 0 when no pixel is background): row `[-1, int.MaxValue, 0, 0, 0]`, centroid
    `(NaN, NaN)` with the positive quiet NaN `0x7FF8000000000000`.
- The result does not depend on the accumulation order.

## ushort overflow (OpenCV 5.0)

- OpenCV 5.0 `checkLabelTypeOverflowBeforeIncrement` throws when a scan creates provisional
  label 65535. The port throws `OverflowException` for the same provisional count (see
  [Parallel strategy](#parallel-strategy) for the stripe case).
- The check counts provisional labels, not components. A `ushort` call can throw when the
  component count is below 65535. Example: 33124 U shapes, each with two provisional labels.
- 65534 isolated components succeed with label count 65535. 65535 isolated components throw.
- cv2 4.5.5 has no check and wraps labels modulo 65536.

## Parallel strategy

- Stripes start at even rows. The stripe count is at most
  `CvParallel.MaxDegreeOfParallelism`, at most the number of row pairs, and at most one stripe
  per 32768 pixels.
- Each stripe scans its rows as if its first row were the image top. Each stripe uses a
  disjoint provisional label range that starts at `stripeFirstLabel4Connectivity` (pixel 4C)
  or `stripeFirstLabel8Connectivity` (pixel 8C and block scan). The ranges increase with the
  stripe index.
- After the parallel scan, the boundary rows are merged sequentially (`mergeLabels*`). The
  label ranges are flattened in stripe order (`flattenLParallel`).
- The second scan runs in the same stripes. Stripe 0 accumulates into the global statistics.
  Every other stripe accumulates into a local array over the final labels of its provisional
  range, plus a background accumulator. The local arrays are then merged.
- Output does not depend on the stripe count: the smallest root is the earliest provisional
  label in scan order for every stripe split.
- `ushort` labels also use the parallel scan. The overflow condition is evaluated after the
  merge with the sequential provisional count:
  - A stripe scan treats the row above its first row (pixel scans) or first row pair (block
    scan) as background. It creates a label for an element that connects only upwards; the
    sequential scan does not.
  - The merge counts these elements: pixel scans, a foreground pixel with a background left
    neighbour and a foreground neighbour above; block scan, a block that connects to P, Q or R
    and not to S.
  - Sequential count = sum of the stripe label counts - merge counts. The port throws when the
    sequential count is 65535 or more. This equals the OpenCV 5.0 condition.
  - The provisional labels are stored as `int` during the scan, so the stripe label ranges can
    exceed 65535.

## Performance

- Rows are copied into padded buffers as 0/1 bytes. The scans then need no edge tests, and
  word compares detect solid foreground.
- Background runs are skipped 32 pixels at a time with a vector compare and a leading-zero
  count.
- Fast paths: a block with the full row above and `n, o, p` foreground copies the label of Q;
  in the pixel 8C scan, eight pixels under eight foreground pixels copy the row above with one
  vector store; in the pixel 4C scan, eight pixels under foreground take the left label.
- The block expansion of the second scan uses AVX2 (`vpermd` duplicates each block label for
  two pixels, then a mask with the foreground). Other hardware uses the scalar loop.
- The relabel of pixel labels stores one broadcast value for a vector of equal labels.
- Statistics find run ends with a vector compare of each row with itself shifted by one pixel.
- The allocating overloads allocate the label image. At 256x256 with `int` labels, the 256 KB
  array is on the large object heap and the resulting generation 2 collections cost more than
  the labeling. Use the overload with a `labels` destination to reuse the label image.

## Deliberate differences from OpenCV 5.0

- Empty input cannot occur (`Mat<T>` has no empty shape). OpenCV Spaghetti has undefined
  behaviour on an empty image.
- `ushort` labels run in parallel stripes. OpenCV runs `CV_16U` sequentially. The output and
  the overflow condition are identical.
- The overflow error is `OverflowException` with a .NET message. OpenCV raises `StsError`.
- On overflow, the parallel scan throws after the first scan; the label destination is
  partly written. OpenCV throws during the scan and also leaves the destination partly
  written.

## Golden constraints (`tools/generate_cc_golden.py`)

- Environment: one thread, IPP and OpenCL off.
- cv2 4.5.5 Spaghetti4C (`Default`, `Bolelli`, `Spaghetti` with 4-connectivity) leaves
  background labels uninitialised (OpenCV issue 21366, fixed in 4.6.0). Every 4-connectivity
  case takes its expected output from `CCL_WU`. OpenCV 5.0 Spaghetti4C equals SAUF 4C. The
  field `source` records the cv2 algorithm.
- `ushort` cases use images whose provisional label bound is below 65535. The overflow rule is
  covered by analytic tests in `CcBasicTests`.
- Random images are rebuilt from a hash formula (`synthetic`); they are not stored. Cases of
  101x203 and larger store the SHA-256 of the outputs.
- Suspect cases: none.
- `CcBasicTests` also compares every algorithm with an independent flood-fill reference on
  odd sizes, including sizes that split into stripes.
