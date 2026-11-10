// This implementation is ported from the OpenCV keypoint filters.
//
// Original sources:
//   modules/features/src/keypoint.cpp (KeyPointsFilter)
//   modules/features/include/opencv2/features.hpp (declaration)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

using System.Buffers;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Filters for keypoint lists, as OpenCV <c>cv::KeyPointsFilter</c>.
/// </summary>
/// <remarks>
/// <para><b>Mutating.</b> Every method changes the list in place, as the OpenCV functions change
/// their <c>std::vector</c>.</para>
/// <para><b>NaN.</b> OpenCV comparators are not strict weak orders when a compared member is NaN,
/// so the OpenCV result is implementation-defined. The port orders NaN after every number in
/// <see cref="RetainBest"/> and before every number in the sorts of
/// <see cref="RemoveDuplicated"/> and <see cref="RemoveDuplicatedSorted"/>. Keypoints with NaN
/// members are never duplicates.</para>
/// </remarks>
public static class KeyPointsFilter
{
    /// <summary>
    /// Keeps the <paramref name="nPoints"/> keypoints with the largest response and every other
    /// keypoint whose response equals the smallest kept response, as OpenCV
    /// <c>KeyPointsFilter::retainBest</c>.
    /// </summary>
    /// <remarks>
    /// OpenCV uses <c>std::nth_element</c> and <c>std::partition</c>; the order of the kept
    /// keypoints is implementation-defined. OpenCV guarantees only that element
    /// <c>nPoints − 1</c> has the boundary response, that the elements before it have a response
    /// not less than the boundary, and that the elements after it have the boundary response.
    /// The port returns the kept keypoints sorted by decreasing response, with equal responses in
    /// input order. This order satisfies the OpenCV guarantees and does not depend on the
    /// platform. The set of kept keypoints equals the OpenCV set.
    /// </remarks>
    /// <param name="keypoints">Keypoints; changed in place.</param>
    /// <param name="nPoints">Number of keypoints to keep. A negative value or a value not less than the count keeps the list unchanged.</param>
    public static void RetainBest(List<KeyPoint> keypoints, int nPoints)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        CollectionsMarshal.SetCount(keypoints, RetainBestSpan(CollectionsMarshal.AsSpan(keypoints), nPoints));
    }

    /// <summary>
    /// <see cref="RetainBest"/> on a span: the kept keypoints move to the start of
    /// <paramref name="keypoints"/>, in the order of <see cref="RetainBest"/>.
    /// </summary>
    /// <returns>The number of kept keypoints.</returns>
    internal static int RetainBestSpan(Span<KeyPoint> keypoints, int nPoints)
    {
        int n = keypoints.Length;
        if (nPoints < 0 || n <= nPoints)
        {
            return n;
        }

        if (nPoints == 0)
        {
            return 0;
        }

        if (RetainBestWithoutNaN(keypoints, nPoints, out int kept))
        {
            return kept;
        }

        int[] order = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }

        KeyPoint[] items = keypoints.ToArray();
        Array.Sort(order, (i, j) => CompareResponseDescending(items, i, j));
        float boundary = items[order[nPoints - 1]].Response;
        int count = nPoints;
        while (count < n && items[order[count]].Response >= boundary)
        {
            count++;
        }

        for (int i = 0; i < count; i++)
        {
            keypoints[i] = items[order[i]];
        }

        return count;
    }

    /// <summary>
    /// Selection form of <see cref="RetainBestSpan"/> with the same result: the boundary response
    /// is found in linear time and only the kept keypoints are sorted.
    /// </summary>
    /// <returns>False, with the keypoints unchanged, when a response is NaN.</returns>
    private static bool RetainBestWithoutNaN(Span<KeyPoint> items, int nPoints, out int count)
    {
        int n = items.Length;
        count = 0;
        float[] responses = ArrayPool<float>.Shared.Rent(n);
        (float Response, int Index)[]? keptBuffer = null;
        KeyPoint[]? resultBuffer = null;
        try
        {
            for (int i = 0; i < n; i++)
            {
                float r = items[i].Response;
                if (float.IsNaN(r))
                {
                    return false;
                }

                responses[i] = r;
            }

            float boundary = SelectAscending(responses.AsSpan(0, n), n - nPoints);
            for (int i = 0; i < n; i++)
            {
                if (items[i].Response >= boundary)
                {
                    count++;
                }
            }

            keptBuffer = ArrayPool<(float Response, int Index)>.Shared.Rent(count);
            Span<(float Response, int Index)> kept = keptBuffer.AsSpan(0, count);
            for (int i = 0, k = 0; i < n; i++)
            {
                float r = items[i].Response;
                if (r >= boundary)
                {
                    kept[k++] = (r, i);
                }
            }

            kept.Sort(static (a, b) => a.Response > b.Response ? -1 : a.Response < b.Response ? 1 : a.Index.CompareTo(b.Index));
            resultBuffer = ArrayPool<KeyPoint>.Shared.Rent(count);
            Span<KeyPoint> result = resultBuffer.AsSpan(0, count);
            for (int i = 0; i < count; i++)
            {
                result[i] = items[kept[i].Index];
            }

            result.CopyTo(items);
            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(responses);
            if (keptBuffer is not null)
            {
                ArrayPool<(float Response, int Index)>.Shared.Return(keptBuffer);
            }

            if (resultBuffer is not null)
            {
                ArrayPool<KeyPoint>.Shared.Return(resultBuffer);
            }
        }
    }

    /// <summary>Returns the element at index <paramref name="k"/> of the ascending order; reorders <paramref name="values"/>.</summary>
    private static float SelectAscending(Span<float> values, int k)
    {
        int lo = 0;
        int hi = values.Length - 1;
        while (lo < hi)
        {
            float pivot = values[lo + ((hi - lo) / 2)];
            int i = lo;
            int j = hi;
            while (i <= j)
            {
                while (values[i] < pivot)
                {
                    i++;
                }

                while (values[j] > pivot)
                {
                    j--;
                }

                if (i <= j)
                {
                    (values[i], values[j]) = (values[j], values[i]);
                    i++;
                    j--;
                }
            }

            if (k <= j)
            {
                hi = j;
            }
            else if (k >= i)
            {
                lo = i;
            }
            else
            {
                return values[k];
            }
        }

        return values[k];
    }

    /// <summary>
    /// Removes the keypoints whose rounded position is not inside the image after removal of a
    /// border of <paramref name="borderSize"/> pixels, as OpenCV
    /// <c>KeyPointsFilter::runByImageBorder</c>. The position rounds with <c>cvRound</c>
    /// (half to even), as OpenCV <c>Point_&lt;int&gt;(Point2f)</c>.
    /// </summary>
    /// <param name="keypoints">Keypoints; changed in place.</param>
    /// <param name="imageSize">Image size.</param>
    /// <param name="borderSize">Border width. Zero or less keeps the list unchanged. When the image is not larger than twice the border in each dimension, every keypoint is removed.</param>
    public static void RunByImageBorder(List<KeyPoint> keypoints, Size imageSize, int borderSize)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        CollectionsMarshal.SetCount(keypoints, RunByImageBorderSpan(CollectionsMarshal.AsSpan(keypoints), imageSize, borderSize));
    }

    /// <summary>
    /// <see cref="RunByImageBorder"/> on a span: the kept keypoints move to the start of
    /// <paramref name="keypoints"/> in input order.
    /// </summary>
    /// <returns>The number of kept keypoints.</returns>
    internal static int RunByImageBorderSpan(Span<KeyPoint> keypoints, Size imageSize, int borderSize)
    {
        if (borderSize <= 0)
        {
            return keypoints.Length;
        }

        if (imageSize.Height <= borderSize * 2 || imageSize.Width <= borderSize * 2)
        {
            return 0;
        }

        int x1 = imageSize.Width - borderSize;
        int y1 = imageSize.Height - borderSize;
        int count = 0;
        for (int i = 0; i < keypoints.Length; i++)
        {
            KeyPoint kp = keypoints[i];
            int x = ScalarConverter.CvRound(kp.Pt.X);
            int y = ScalarConverter.CvRound(kp.Pt.Y);
            if (borderSize <= x && x < x1 && borderSize <= y && y < y1)
            {
                keypoints[count++] = kp;
            }
        }

        return count;
    }

    /// <summary>
    /// Removes the keypoints whose size is outside <c>[minSize, maxSize]</c>, as OpenCV
    /// <c>KeyPointsFilter::runByKeypointSize</c>.
    /// </summary>
    /// <param name="keypoints">Keypoints; changed in place.</param>
    /// <param name="minSize">Minimum size; zero or more.</param>
    /// <param name="maxSize">Maximum size; not less than <paramref name="minSize"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">A size bound is negative, or <paramref name="minSize"/> is greater than <paramref name="maxSize"/>.</exception>
    public static void RunByKeypointSize(List<KeyPoint> keypoints, float minSize, float maxSize = float.MaxValue)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        if (!(minSize >= 0) || !(maxSize >= 0) || !(minSize <= maxSize))
        {
            throw new ArgumentOutOfRangeException(nameof(minSize), "Requires 0 <= minSize <= maxSize.");
        }

        keypoints.RemoveAll(kp => kp.Size < minSize || kp.Size > maxSize);
    }

    /// <summary>
    /// Removes the keypoints at a zero mask pixel, as OpenCV
    /// <c>KeyPointsFilter::runByPixelsMask</c>. The pixel is
    /// <c>((int)(y + 0.5f), (int)(x + 0.5f))</c> (truncation toward zero).
    /// </summary>
    /// <param name="keypoints">Keypoints; changed in place.</param>
    /// <param name="mask">Single-channel mask, or null to keep the list unchanged (OpenCV empty mask).</param>
    /// <exception cref="ArgumentException">The mask has more than one channel.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A keypoint pixel is outside the mask. OpenCV reads outside the matrix in this case.</exception>
    public static void RunByPixelsMask(List<KeyPoint> keypoints, Mat<byte>? mask)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        if (mask is null)
        {
            return;
        }

        if (mask.Channels != 1)
        {
            throw new ArgumentException("The mask must have one channel.", nameof(mask));
        }

        CollectionsMarshal.SetCount(keypoints, RunByPixelsMaskSpan(CollectionsMarshal.AsSpan(keypoints), mask));
    }

    /// <summary>
    /// <see cref="RunByPixelsMask"/> on a span with a single-channel mask: the kept keypoints move
    /// to the start of <paramref name="keypoints"/> in input order.
    /// </summary>
    /// <returns>The number of kept keypoints.</returns>
    internal static int RunByPixelsMaskSpan(Span<KeyPoint> keypoints, Mat<byte> mask)
    {
        int count = 0;
        for (int i = 0; i < keypoints.Length; i++)
        {
            KeyPoint kp = keypoints[i];
            float fx = kp.Pt.X + 0.5f;
            float fy = kp.Pt.Y + 0.5f;
            if (!(fx > -1f && fx < mask.Width) || !(fy > -1f && fy < mask.Height))
            {
                throw new ArgumentOutOfRangeException(nameof(keypoints), kp.Pt, "A keypoint is outside the mask.");
            }

            if (mask.At((int)fy, (int)fx) != 0)
            {
                keypoints[count++] = kp;
            }
        }

        return count;
    }

    /// <summary>
    /// Removes keypoints that have the position, size and angle of an earlier keypoint in the
    /// sort order of OpenCV <c>KeyPointsFilter::removeDuplicated</c>, and keeps the input order
    /// of the remaining keypoints.
    /// </summary>
    /// <remarks>
    /// The sort order is x and y ascending, size descending, angle ascending, response
    /// descending, octave descending, class descending, then input index. Of each group of
    /// duplicates, the first keypoint in that order stays.
    /// </remarks>
    /// <param name="keypoints">Keypoints; changed in place.</param>
    public static void RemoveDuplicated(List<KeyPoint> keypoints)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        int n = keypoints.Count;
        KeyPoint[] items = keypoints.ToArray();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++)
        {
            idx[i] = i;
        }

        Array.Sort(idx, (i, j) =>
        {
            int r = CompareKeyPoints(items[i], items[j]);
            return r != 0 ? r : i.CompareTo(j);
        });

        bool[] remove = new bool[n];
        for (int i = 1, j = 0; i < n; i++)
        {
            KeyPoint kp1 = items[idx[i]];
            KeyPoint kp2 = items[idx[j]];
            if (!SameGeometry(kp1, kp2))
            {
                j = i;
            }
            else
            {
                remove[idx[i]] = true;
            }
        }

        keypoints.Clear();
        for (int i = 0; i < n; i++)
        {
            if (!remove[i])
            {
                keypoints.Add(items[i]);
            }
        }
    }

    /// <summary>
    /// Sorts the keypoints in the order of OpenCV <c>KeyPointsFilter::removeDuplicatedSorted</c>
    /// and removes each keypoint that has the position, size and angle of its predecessor.
    /// </summary>
    /// <remarks>
    /// The order is x and y ascending, size descending, angle ascending, response descending,
    /// octave descending, class descending. OpenCV uses <c>std::sort</c>, which does not keep the
    /// input order of keypoints that compare equal; such keypoints differ at most in the sign of
    /// zero members. The port keeps their input order.
    /// </remarks>
    /// <param name="keypoints">Keypoints; sorted and changed in place.</param>
    public static void RemoveDuplicatedSorted(List<KeyPoint> keypoints)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        int n = keypoints.Count;
        if (n < 2)
        {
            return;
        }

        KeyPoint[] items = ArrayPool<KeyPoint>.Shared.Rent(n);
        int[] idx = ArrayPool<int>.Shared.Rent(n);
        try
        {
            CollectionsMarshal.AsSpan(keypoints).CopyTo(items);
            Span<int> order = idx.AsSpan(0, n);
            for (int i = 0; i < n; i++)
            {
                order[i] = i;
            }

            // The index tie-break makes the order total, so the result does not depend on the sort algorithm.
            order.Sort((i, j) =>
            {
                int r = CompareKeyPoints(items[i], items[j]);
                return r != 0 ? r : i.CompareTo(j);
            });

            keypoints.Clear();
            keypoints.Add(items[order[0]]);
            for (int j = 1; j < n; j++)
            {
                KeyPoint kp2 = items[order[j]];
                if (!SameGeometry(keypoints[^1], kp2))
                {
                    keypoints.Add(kp2);
                }
            }
        }
        finally
        {
            ArrayPool<KeyPoint>.Shared.Return(items);
            ArrayPool<int>.Shared.Return(idx);
        }
    }

    private static bool SameGeometry(KeyPoint a, KeyPoint b)
        => a.Pt.X == b.Pt.X && a.Pt.Y == b.Pt.Y && a.Size == b.Size && a.Angle == b.Angle;

    /// <summary>OpenCV <c>KeyPoint12_LessThan</c> as a three-way comparison.</summary>
    private static int CompareKeyPoints(KeyPoint a, KeyPoint b)
    {
        int r = a.Pt.X.CompareTo(b.Pt.X);
        if (r != 0)
        {
            return r;
        }

        r = a.Pt.Y.CompareTo(b.Pt.Y);
        if (r != 0)
        {
            return r;
        }

        r = b.Size.CompareTo(a.Size);
        if (r != 0)
        {
            return r;
        }

        r = a.Angle.CompareTo(b.Angle);
        if (r != 0)
        {
            return r;
        }

        r = b.Response.CompareTo(a.Response);
        if (r != 0)
        {
            return r;
        }

        r = b.Octave.CompareTo(a.Octave);
        return r != 0 ? r : b.ClassId.CompareTo(a.ClassId);
    }

    private static int CompareResponseDescending(KeyPoint[] items, int i, int j)
    {
        float a = items[i].Response;
        float b = items[j].Response;
        if (a > b)
        {
            return -1;
        }

        if (a < b)
        {
            return 1;
        }

        bool an = float.IsNaN(a);
        bool bn = float.IsNaN(b);
        if (an != bn)
        {
            return an ? 1 : -1;
        }

        return i.CompareTo(j);
    }
}
