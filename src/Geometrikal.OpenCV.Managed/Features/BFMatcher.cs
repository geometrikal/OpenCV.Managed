// This implementation is ported from the OpenCV brute-force descriptor matcher.
//
// Original sources:
//   modules/features/src/matchers.cpp (DescriptorMatcher, BFMatcher::knnMatchImpl,
//     BFMatcher::radiusMatchImpl, convertMatches, checkMasks)
//   modules/features/include/opencv2/features.hpp (declarations)
//   modules/core/src/batch_distance.cpp (batchDistance)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.Features.Internal;

/// <summary>
/// Brute-force descriptor matcher, as OpenCV <c>cv::BFMatcher</c>. Each query descriptor is
/// compared with every train descriptor.
/// </summary>
/// <remarks>
/// <para><b>Descriptors.</b> One descriptor per row of a single-channel <c>Mat&lt;byte&gt;</c> or
/// <c>Mat&lt;float&gt;</c>. Query and train descriptors have the same element type and width.
/// Byte descriptors support <see cref="NormType.L1"/>, <see cref="NormType.L2"/>,
/// <see cref="NormType.L2Sqr"/>, <see cref="NormType.Hamming"/> and
/// <see cref="NormType.Hamming2"/>. Float descriptors support <see cref="NormType.L1"/>,
/// <see cref="NormType.L2"/> and <see cref="NormType.L2Sqr"/>. Other combinations throw
/// <see cref="NotSupportedException"/>.</para>
/// <para><b>Distances.</b> The values are bit-identical to OpenCV 5.0 on its SSE3 baseline. Float
/// <see cref="NormType.L2Sqr"/> and <see cref="NormType.L2"/> use the OpenCV 16-element partial
/// sum order; float <see cref="NormType.L1"/> uses one sequential sum. <see cref="NormType.L2"/>
/// is the float square root of the squared distance.</para>
/// <para><b>Order.</b> For equal distances, the k-nearest result keeps the train descriptor with
/// the lower index first (OpenCV inserts with a strict comparison). The radius result is sorted
/// by distance with the element order of MSVC <c>std::sort</c>, which the cv2 Windows wheels
/// use; this order is not stable.</para>
/// <para><b>Masks.</b> A mask is a <c>Mat&lt;byte&gt;</c> with one row per query descriptor and one
/// column per train descriptor. A zero element excludes the pair.</para>
/// <para><b>Train collection.</b> <see cref="Add{T}(Mat{T})"/> adds train descriptor sets (OpenCV
/// "images"). <see cref="DMatch.ImgIdx"/> is the set index. The overloads with a
/// <c>train</c> argument ignore the collection and give <see cref="DMatch.ImgIdx"/> 0. The
/// matcher stores references to the added matrices; it does not copy them.</para>
/// <para><b>Threads.</b> Query rows are processed in parallel. The output does not depend on the
/// thread count.</para>
/// </remarks>
public sealed class BFMatcher
{
    private const int ImgIdxShift = 18;
    private const int ImgIdxOne = 1 << ImgIdxShift;

    private readonly List<object> _train = [];

    /// <summary>Creates a matcher, as OpenCV <c>BFMatcher(normType, crossCheck)</c>.</summary>
    /// <param name="normType">Distance norm.</param>
    /// <param name="crossCheck">
    /// When true, <see cref="Match{T}(Mat{T}, Mat{T}, Mat{byte})"/> and
    /// <see cref="KnnMatch{T}(Mat{T}, Mat{T}, int, Mat{byte}, bool)"/> with <c>k = 1</c> return only
    /// pairs (i, j) where train descriptor j is the nearest to query i and query i is the nearest
    /// to train descriptor j. Cross-check needs <c>k = 1</c>, no mask and one train set.
    /// <see cref="RadiusMatch{T}(Mat{T}, Mat{T}, float, Mat{byte}, bool)"/> ignores it.
    /// </param>
    public BFMatcher(NormType normType = NormType.L2, bool crossCheck = false)
    {
        if (normType is not (NormType.L1 or NormType.L2 or NormType.L2Sqr or NormType.Hamming or NormType.Hamming2))
        {
            throw new ArgumentOutOfRangeException(nameof(normType), normType, "Unsupported norm for BFMatcher.");
        }

        NormType = normType;
        CrossCheck = crossCheck;
    }

    /// <summary>Distance norm.</summary>
    public NormType NormType { get; }

    /// <summary>True when cross-check is enabled.</summary>
    public bool CrossCheck { get; }

    /// <summary>Number of train descriptor sets in the collection.</summary>
    public int TrainSetCount => _train.Count;

    /// <summary>True when the train collection is empty.</summary>
    public bool IsEmpty => _train.Count == 0;

    /// <summary>
    /// Adds a train descriptor set to the collection, as OpenCV <c>DescriptorMatcher::add</c>.
    /// All sets have the same element type.
    /// </summary>
    public void Add<T>(Mat<T> descriptors)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(descriptors);
        BatchDistance.SelectKind<T>(NormType);
        if (descriptors.Channels != 1)
        {
            throw new ArgumentException("Descriptors must have one channel.", nameof(descriptors));
        }

        if (_train.Count > 0 && _train[0] is not Mat<T>)
        {
            throw new ArgumentException("All train descriptor sets must have the same element type.", nameof(descriptors));
        }

        if (descriptors.Height >= ImgIdxOne)
        {
            throw new ArgumentException($"A train descriptor set must have fewer than {ImgIdxOne} rows.", nameof(descriptors));
        }

        _train.Add(descriptors);
    }

    /// <summary>Removes all train descriptor sets, as OpenCV <c>DescriptorMatcher::clear</c>.</summary>
    public void Clear() => _train.Clear();

    /// <summary>
    /// Finds the best train match for each query descriptor, as OpenCV
    /// <c>match(queryDescriptors, trainDescriptors, matches, mask)</c>.
    /// </summary>
    /// <returns>One match for each query descriptor that has a match, in query order.</returns>
    public DMatch[] Match<T>(Mat<T> query, Mat<T> train, Mat<byte>? mask = null)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(train);
        return ToSingle(KnnMatchCore(query, [train], 1, [mask], true));
    }

    /// <summary>
    /// Finds the best match in the train collection for each query descriptor, as OpenCV
    /// <c>match(queryDescriptors, matches, masks)</c>.
    /// </summary>
    /// <param name="query">Query descriptors.</param>
    /// <param name="masks">Null, or one mask (or null) per train set.</param>
    public DMatch[] Match<T>(Mat<T> query, IReadOnlyList<Mat<byte>?>? masks = null)
        where T : unmanaged => ToSingle(KnnMatchCore(query, TrainSets<T>(), 1, masks, true));

    /// <summary>
    /// Finds the <paramref name="k"/> best train matches for each query descriptor, as OpenCV
    /// <c>knnMatch(queryDescriptors, trainDescriptors, matches, k, mask, compactResult)</c>.
    /// </summary>
    /// <param name="query">Query descriptors.</param>
    /// <param name="train">Train descriptors.</param>
    /// <param name="k">Number of matches for each query; at least 1.</param>
    /// <param name="mask">Optional pair mask.</param>
    /// <param name="compactResult">When true, query descriptors without a match have no entry.</param>
    /// <returns>For each query, up to <paramref name="k"/> matches by increasing distance.</returns>
    public DMatch[][] KnnMatch<T>(Mat<T> query, Mat<T> train, int k, Mat<byte>? mask = null, bool compactResult = false)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(train);
        return KnnMatchCore(query, [train], k, [mask], compactResult);
    }

    /// <summary>
    /// Finds the <paramref name="k"/> best matches in the train collection, as OpenCV
    /// <c>knnMatch(queryDescriptors, matches, k, masks, compactResult)</c>.
    /// </summary>
    public DMatch[][] KnnMatch<T>(Mat<T> query, int k, IReadOnlyList<Mat<byte>?>? masks = null, bool compactResult = false)
        where T : unmanaged => KnnMatchCore(query, TrainSets<T>(), k, masks, compactResult);

    /// <summary>
    /// Finds all train descriptors within <paramref name="maxDistance"/> of each query
    /// descriptor, as OpenCV <c>radiusMatch(queryDescriptors, trainDescriptors, matches,
    /// maxDistance, mask, compactResult)</c>.
    /// </summary>
    /// <param name="query">Query descriptors.</param>
    /// <param name="train">Train descriptors.</param>
    /// <param name="maxDistance">Inclusive distance limit; must be greater than <c>FLT_EPSILON</c> (1.1920929e-7).</param>
    /// <param name="mask">Optional pair mask.</param>
    /// <param name="compactResult">
    /// When true, the lists of the query descriptors with matches come first, in query order, and
    /// the empty lists follow. OpenCV does not remove the empty lists; the port keeps this
    /// behaviour, so the result always has one list per query descriptor.
    /// </param>
    /// <returns>For each list, the matches sorted by distance (see the class remarks for ties).</returns>
    public DMatch[][] RadiusMatch<T>(Mat<T> query, Mat<T> train, float maxDistance, Mat<byte>? mask = null, bool compactResult = false)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(train);
        return RadiusMatchCore(query, [train], maxDistance, [mask], compactResult);
    }

    /// <summary>
    /// Radius match against the train collection, as OpenCV <c>radiusMatch(queryDescriptors,
    /// matches, maxDistance, masks, compactResult)</c>.
    /// </summary>
    public DMatch[][] RadiusMatch<T>(Mat<T> query, float maxDistance, IReadOnlyList<Mat<byte>?>? masks = null, bool compactResult = false)
        where T : unmanaged => RadiusMatchCore(query, TrainSets<T>(), maxDistance, masks, compactResult);

    private List<Mat<T>> TrainSets<T>()
        where T : unmanaged
    {
        var sets = new List<Mat<T>>(_train.Count);
        foreach (object o in _train)
        {
            sets.Add(o as Mat<T> ?? throw new ArgumentException("Query and train descriptors must have the same element type."));
        }

        return sets;
    }

    private static DMatch[] ToSingle(DMatch[][] knn)
    {
        var result = new DMatch[knn.Length];
        for (int i = 0; i < knn.Length; i++)
        {
            result[i] = knn[i][0];
        }

        return result;
    }

    private DistanceKind Validate<T>(Mat<T> query, IReadOnlyList<Mat<T>> trains, IReadOnlyList<Mat<byte>?>? masks)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(query);
        DistanceKind kind = BatchDistance.SelectKind<T>(NormType);
        if (query.Channels != 1)
        {
            throw new ArgumentException("Descriptors must have one channel.", nameof(query));
        }

        foreach (Mat<T> t in trains)
        {
            if (t.Channels != 1 || t.Width != query.Width)
            {
                throw new ArgumentException("Train descriptors must have one channel and the query descriptor width.");
            }
        }

        if (masks is not null && masks.Count > 0)
        {
            if (masks.Count != trains.Count)
            {
                throw new ArgumentException("The mask count must equal the train set count.", nameof(masks));
            }

            for (int i = 0; i < masks.Count; i++)
            {
                Mat<byte>? m = masks[i];
                if (m is not null && (m.Channels != 1 || m.Height != query.Height || m.Width != trains[i].Height))
                {
                    throw new ArgumentException("A mask must have one row per query descriptor and one column per train descriptor.", nameof(masks));
                }
            }
        }

        return kind;
    }

    private DMatch[][] KnnMatchCore<T>(Mat<T> query, IReadOnlyList<Mat<T>> trains, int k, IReadOnlyList<Mat<byte>?>? masks, bool compactResult)
        where T : unmanaged
    {
        DistanceKind kind = Validate(query, trains, masks);
        if (trains.Count == 0)
        {
            return [];
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        int rows = query.Height;
        int cols = -1;
        int[] dist = [];
        int[] nidx = [];
        int update = 0;
        for (int img = 0; img < trains.Count; img++)
        {
            Mat<T> train = trains[img];
            Mat<byte>? mask = masks is null || masks.Count == 0 ? null : masks[img];
            int kk = Math.Min(k, train.Height);
            if (img == 0)
            {
                cols = kk;
                dist = new int[rows * kk];
                nidx = new int[rows * kk];
                dist.AsSpan().Fill(BatchDistanceKernels.MaskedValue(kind));
                nidx.AsSpan().Fill(-1);
            }
            else if (kk != cols)
            {
                throw new NotSupportedException("OpenCV leaves the result undefined when a later train set has fewer than k rows and fewer rows than the first set.");
            }

            if (CrossCheck)
            {
                if (kk != 1 || update != 0 || mask is not null)
                {
                    throw new NotSupportedException("Cross-check needs k = 1, one train set and no mask.");
                }

                BatchDistance.CrossCheck(kind, query, train, dist, nidx);
            }
            else
            {
                BatchDistance.Knn(kind, query, train, mask, kk, update, dist, nidx);
            }

            update += ImgIdxOne;
        }

        var matches = new List<DMatch[]>(rows);
        var mq = new List<DMatch>(cols);
        for (int q = 0; q < rows; q++)
        {
            mq.Clear();
            for (int j = 0; j < cols; j++)
            {
                int id = nidx[(q * cols) + j];
                if (id < 0)
                {
                    break;
                }

                mq.Add(new DMatch(q, id & (ImgIdxOne - 1), id >> ImgIdxShift, BatchDistance.ToFloat(kind, dist[(q * cols) + j])));
            }

            if (mq.Count > 0 || !compactResult)
            {
                matches.Add([.. mq]);
            }
        }

        return [.. matches];
    }

    private DMatch[][] RadiusMatchCore<T>(Mat<T> query, IReadOnlyList<Mat<T>> trains, float maxDistance, IReadOnlyList<Mat<byte>?>? masks, bool compactResult)
        where T : unmanaged
    {
        DistanceKind kind = Validate(query, trains, masks);
        if (trains.Count == 0)
        {
            return [];
        }

        if (!(maxDistance > 1.1920929e-07f))
        {
            throw new ArgumentOutOfRangeException(nameof(maxDistance), "maxDistance must be greater than FLT_EPSILON.");
        }

        int rows = query.Height;
        var lists = new List<DMatch>[rows];
        for (int q = 0; q < rows; q++)
        {
            lists[q] = [];
        }

        for (int img = 0; img < trains.Count; img++)
        {
            Mat<T> train = trains[img];
            Mat<byte>? mask = masks is null || masks.Count == 0 ? null : masks[img];
            int n = train.Height;
            int[] dist = new int[rows * n];
            BatchDistance.Full(kind, query, train, mask, dist);
            for (int q = 0; q < rows; q++)
            {
                List<DMatch> mq = lists[q];
                for (int j = 0; j < n; j++)
                {
                    float d = BatchDistance.ToFloat(kind, dist[(q * n) + j]);
                    if (d <= maxDistance)
                    {
                        mq.Add(new DMatch(q, j, img, d));
                    }
                }
            }
        }

        var result = new DMatch[rows][];
        int q0 = 0;
        for (int q = 0; q < rows; q++)
        {
            if (lists[q].Count == 0 && compactResult)
            {
                continue;
            }

            DMatch[] sorted = [.. lists[q]];
            MsvcSort.Sort(sorted.AsSpan(), default(DistanceLess));
            result[q0++] = sorted;
        }

        for (; q0 < rows; q0++)
        {
            result[q0] = [];
        }

        return result;
    }

    private readonly struct DistanceLess : ILess<DMatch>
    {
        public bool Less(in DMatch a, in DMatch b) => a.Distance < b.Distance;
    }
}
