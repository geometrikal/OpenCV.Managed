// This implementation reproduces the element order of the MSVC STL std::sort, which the cv2
// Windows wheels use in DpSeamFinder::find.
//
// Original sources:
//   modules/stitching/src/seam_finders.cpp (DpSeamFinder::find, ImagePairLess)
//   MSVC STL <algorithm> (_Sort_unchecked, _Partition_by_median_guess_unchecked,
//     _Guess_median_unchecked, _Med3_unchecked, _Insertion_sort_unchecked, heap helpers)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).
// The MSVC STL is licensed under the Apache License 2.0 with LLVM exception.

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>
/// Unstable sort with the MSVC STL <c>std::sort</c> algorithm: insertion sort for ranges of up
/// to 32 elements, median-guess partitions with a fat pivot, and heap sort when the division
/// budget runs out. Equal keys end in the same order as in MSVC <c>std::sort</c>.
/// </summary>
internal static class SeamPairSort
{
    private const int IsortMax = 32;

    /// <summary>Sorts <paramref name="items"/> by ascending <paramref name="keys"/>; both arrays move together.</summary>
    public static void Sort<T>(T[] items, int[] keys)
    {
        var s = new Sorter<T>(items, keys);
        s.SortRange(0, items.Length, items.Length);
    }

    private readonly struct Sorter<T>(T[] items, int[] keys)
    {
        private bool Less(int a, int b) => keys[a] < keys[b];

        private void Swap(int a, int b)
        {
            (items[a], items[b]) = (items[b], items[a]);
            (keys[a], keys[b]) = (keys[b], keys[a]);
        }

        public void SortRange(int first, int last, int ideal)
        {
            for (;;)
            {
                if (last - first <= IsortMax)
                {
                    InsertionSort(first, last);
                    return;
                }

                if (ideal <= 0)
                {
                    MakeHeap(first, last);
                    SortHeap(first, last);
                    return;
                }

                (int midFirst, int midSecond) = Partition(first, last);
                ideal = (ideal >> 1) + (ideal >> 2);
                if (midFirst - first < last - midSecond)
                {
                    SortRange(first, midFirst, ideal);
                    first = midSecond;
                }
                else
                {
                    SortRange(midSecond, last, ideal);
                    last = midFirst;
                }
            }
        }

        private void InsertionSort(int first, int last)
        {
            if (first == last)
            {
                return;
            }

            for (int mid = first + 1; mid != last; mid++)
            {
                T val = items[mid];
                int valKey = keys[mid];
                int hole = mid;
                if (valKey < keys[first])
                {
                    Array.Copy(items, first, items, first + 1, mid - first);
                    Array.Copy(keys, first, keys, first + 1, mid - first);
                    items[first] = val;
                    keys[first] = valKey;
                }
                else
                {
                    for (int prev = hole - 1; valKey < keys[prev]; hole = prev, prev--)
                    {
                        items[hole] = items[prev];
                        keys[hole] = keys[prev];
                    }

                    items[hole] = val;
                    keys[hole] = valKey;
                }
            }
        }

        private void Med3(int first, int mid, int last)
        {
            if (Less(mid, first))
            {
                Swap(mid, first);
            }

            if (Less(last, mid))
            {
                Swap(last, mid);
                if (Less(mid, first))
                {
                    Swap(mid, first);
                }
            }
        }

        private void GuessMedian(int first, int mid, int last)
        {
            int count = last - first;
            if (count > 40)
            {
                int step = (count + 1) >> 3;
                int twoStep = step << 1;
                Med3(first, first + step, first + twoStep);
                Med3(mid - step, mid, mid + step);
                Med3(last - twoStep, last - step, last);
                Med3(first + step, mid, last - step);
            }
            else
            {
                Med3(first, mid, last);
            }
        }

        private (int, int) Partition(int first, int last)
        {
            int mid = first + ((last - first) >> 1);
            GuessMedian(first, mid, last - 1);
            int pfirst = mid;
            int plast = pfirst + 1;

            while (first < pfirst && !Less(pfirst - 1, pfirst) && !Less(pfirst, pfirst - 1))
            {
                --pfirst;
            }

            while (plast < last && !Less(plast, pfirst) && !Less(pfirst, plast))
            {
                ++plast;
            }

            int gfirst = plast;
            int glast = pfirst;

            for (;;)
            {
                for (; gfirst < last; ++gfirst)
                {
                    if (Less(pfirst, gfirst))
                    {
                        continue;
                    }
                    else if (Less(gfirst, pfirst))
                    {
                        break;
                    }
                    else if (plast != gfirst)
                    {
                        Swap(plast, gfirst);
                        ++plast;
                    }
                    else
                    {
                        ++plast;
                    }
                }

                for (; first < glast; --glast)
                {
                    int prev = glast - 1;
                    if (Less(prev, pfirst))
                    {
                        continue;
                    }
                    else if (Less(pfirst, prev))
                    {
                        break;
                    }
                    else if (--pfirst != prev)
                    {
                        Swap(pfirst, prev);
                    }
                }

                if (glast == first && gfirst == last)
                {
                    return (pfirst, plast);
                }

                if (glast == first)
                {
                    if (plast != gfirst)
                    {
                        Swap(pfirst, plast);
                    }

                    ++plast;
                    Swap(pfirst, gfirst);
                    ++pfirst;
                    ++gfirst;
                }
                else if (gfirst == last)
                {
                    if (--glast != --pfirst)
                    {
                        Swap(glast, pfirst);
                    }

                    Swap(pfirst, --plast);
                }
                else
                {
                    Swap(gfirst, --glast);
                    ++gfirst;
                }
            }
        }

        private void PushHeapByIndex(int first, int hole, int top, T val, int valKey)
        {
            for (int idx = (hole - 1) >> 1; top < hole && keys[first + idx] < valKey; idx = (hole - 1) >> 1)
            {
                items[first + hole] = items[first + idx];
                keys[first + hole] = keys[first + idx];
                hole = idx;
            }

            items[first + hole] = val;
            keys[first + hole] = valKey;
        }

        private void PopHeapHoleByIndex(int first, int hole, int bottom, T val, int valKey)
        {
            int top = hole;
            int idx = hole;
            int maxNonLeaf = (bottom - 1) >> 1;
            while (idx < maxNonLeaf)
            {
                idx = (2 * idx) + 2;
                if (keys[first + idx] < keys[first + idx - 1])
                {
                    --idx;
                }

                items[first + hole] = items[first + idx];
                keys[first + hole] = keys[first + idx];
                hole = idx;
            }

            if (idx == maxNonLeaf && bottom % 2 == 0)
            {
                items[first + hole] = items[first + bottom - 1];
                keys[first + hole] = keys[first + bottom - 1];
                hole = bottom - 1;
            }

            PushHeapByIndex(first, hole, top, val, valKey);
        }

        private void MakeHeap(int first, int last)
        {
            int bottom = last - first;
            for (int hole = bottom >> 1; hole > 0;)
            {
                --hole;
                PopHeapHoleByIndex(first, hole, bottom, items[first + hole], keys[first + hole]);
            }
        }

        private void SortHeap(int first, int last)
        {
            for (; last - first >= 2; --last)
            {
                int l = last - 1;
                T val = items[l];
                int valKey = keys[l];
                items[l] = items[first];
                keys[l] = keys[first];
                PopHeapHoleByIndex(first, 0, l - first, val, valKey);
            }
        }
    }
}
