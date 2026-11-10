// This implementation reproduces the element order of std::sort in the Microsoft C++ standard
// library (MSVC STL, <algorithm>: _Sort_unchecked, _Partition_by_median_guess_unchecked,
// _Guess_median_unchecked, _Med3_unchecked, _Insertion_sort_unchecked, _Make_heap_unchecked,
// _Sort_heap_unchecked). The cv2 Windows wheels are built with MSVC, so OpenCV functions that
// call std::sort give this order for equal keys.
//
// The MSVC STL is licensed under the Apache License 2.0 with LLVM exception.

namespace Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>Strict weak ordering for <see cref="MsvcSort"/>.</summary>
internal interface ILess<T>
{
    /// <summary>True when <paramref name="a"/> orders before <paramref name="b"/>.</summary>
    bool Less(in T a, in T b);
}

/// <summary>
/// Introsort with the same comparisons and swaps as MSVC <c>std::sort</c>. Elements that
/// compare equal end in the same order as with MSVC.
/// </summary>
internal static class MsvcSort
{
    private const int InsertionSortMax = 32;

    /// <summary>Sorts <paramref name="items"/> in place.</summary>
    public static void Sort<T, TLess>(Span<T> items, TLess less)
        where TLess : struct, ILess<T>
    {
        SortRange(items, 0, items.Length, items.Length, less);
    }

    private static void SortRange<T, TLess>(Span<T> a, int first, int last, int ideal, TLess less)
        where TLess : struct, ILess<T>
    {
        for (;;)
        {
            if (last - first <= InsertionSortMax)
            {
                InsertionSort(a, first, last, less);
                return;
            }

            if (ideal <= 0)
            {
                MakeHeap(a, first, last, less);
                SortHeap(a, first, last, less);
                return;
            }

            (int midFirst, int midLast) = Partition(a, first, last, less);
            ideal = (ideal >> 1) + (ideal >> 2);

            if (midFirst - first < last - midLast)
            {
                SortRange(a, first, midFirst, ideal, less);
                first = midLast;
            }
            else
            {
                SortRange(a, midLast, last, ideal, less);
                last = midFirst;
            }
        }
    }

    private static void InsertionSort<T, TLess>(Span<T> a, int first, int last, TLess less)
        where TLess : struct, ILess<T>
    {
        if (first == last)
        {
            return;
        }

        for (int mid = first + 1; mid != last; mid++)
        {
            T val = a[mid];
            if (less.Less(val, a[first]))
            {
                a.Slice(first, mid - first).CopyTo(a.Slice(first + 1));
                a[first] = val;
            }
            else
            {
                int hole = mid;
                for (int prev = hole - 1; less.Less(val, a[prev]); prev--)
                {
                    a[hole] = a[prev];
                    hole = prev;
                }

                a[hole] = val;
            }
        }
    }

    private static void Med3<T, TLess>(Span<T> a, int first, int mid, int last, TLess less)
        where TLess : struct, ILess<T>
    {
        if (less.Less(a[mid], a[first]))
        {
            Swap(a, mid, first);
        }

        if (less.Less(a[last], a[mid]))
        {
            Swap(a, last, mid);
            if (less.Less(a[mid], a[first]))
            {
                Swap(a, mid, first);
            }
        }
    }

    private static void GuessMedian<T, TLess>(Span<T> a, int first, int mid, int last, TLess less)
        where TLess : struct, ILess<T>
    {
        int count = last - first;
        if (count > 40)
        {
            int step = (count + 1) >> 3;
            int twoStep = step << 1;
            Med3(a, first, first + step, first + twoStep, less);
            Med3(a, mid - step, mid, mid + step, less);
            Med3(a, last - twoStep, last - step, last, less);
            Med3(a, first + step, mid, last - step, less);
        }
        else
        {
            Med3(a, first, mid, last, less);
        }
    }

    private static (int, int) Partition<T, TLess>(Span<T> a, int first, int last, TLess less)
        where TLess : struct, ILess<T>
    {
        int mid = first + ((last - first) >> 1);
        GuessMedian(a, first, mid, last - 1, less);
        int pfirst = mid;
        int plast = pfirst + 1;

        while (first < pfirst && !less.Less(a[pfirst - 1], a[pfirst]) && !less.Less(a[pfirst], a[pfirst - 1]))
        {
            --pfirst;
        }

        while (plast < last && !less.Less(a[plast], a[pfirst]) && !less.Less(a[pfirst], a[plast]))
        {
            ++plast;
        }

        int gfirst = plast;
        int glast = pfirst;

        for (;;)
        {
            for (; gfirst < last; ++gfirst)
            {
                if (less.Less(a[pfirst], a[gfirst]))
                {
                    continue;
                }
                else if (less.Less(a[gfirst], a[pfirst]))
                {
                    break;
                }
                else if (plast != gfirst)
                {
                    Swap(a, plast, gfirst);
                    ++plast;
                }
                else
                {
                    ++plast;
                }
            }

            for (; first < glast; --glast)
            {
                int glastPrev = glast - 1;
                if (less.Less(a[glastPrev], a[pfirst]))
                {
                    continue;
                }
                else if (less.Less(a[pfirst], a[glastPrev]))
                {
                    break;
                }
                else if (--pfirst != glastPrev)
                {
                    Swap(a, pfirst, glastPrev);
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
                    Swap(a, pfirst, plast);
                }

                ++plast;
                Swap(a, pfirst, gfirst);
                ++pfirst;
                ++gfirst;
            }
            else if (gfirst == last)
            {
                if (--glast != --pfirst)
                {
                    Swap(a, glast, pfirst);
                }

                Swap(a, pfirst, --plast);
            }
            else
            {
                Swap(a, gfirst++, --glast);
            }
        }
    }

    private static void MakeHeap<T, TLess>(Span<T> a, int first, int last, TLess less)
        where TLess : struct, ILess<T>
    {
        int bottom = last - first;
        for (int hole = bottom >> 1; hole > 0;)
        {
            --hole;
            T val = a[first + hole];
            PopHeapHoleByIndex(a, first, hole, bottom, val, less);
        }
    }

    private static void SortHeap<T, TLess>(Span<T> a, int first, int last, TLess less)
        where TLess : struct, ILess<T>
    {
        for (; last - first >= 2; --last)
        {
            int dest = last - 1;
            T val = a[dest];
            a[dest] = a[first];
            PopHeapHoleByIndex(a, first, 0, dest - first, val, less);
        }
    }

    private static void PopHeapHoleByIndex<T, TLess>(Span<T> a, int first, int hole, int bottom, T val, TLess less)
        where TLess : struct, ILess<T>
    {
        int top = hole;
        int idx = hole;
        int maxNonLeaf = (bottom - 1) >> 1;
        while (idx < maxNonLeaf)
        {
            idx = (2 * idx) + 2;
            if (less.Less(a[first + idx], a[first + idx - 1]))
            {
                --idx;
            }

            a[first + hole] = a[first + idx];
            hole = idx;
        }

        if (idx == maxNonLeaf && bottom % 2 == 0)
        {
            a[first + hole] = a[first + bottom - 1];
            hole = bottom - 1;
        }

        for (int parent = (hole - 1) >> 1; top < hole && less.Less(a[first + parent], val); parent = (hole - 1) >> 1)
        {
            a[first + hole] = a[first + parent];
            hole = parent;
        }

        a[first + hole] = val;
    }

    private static void Swap<T>(Span<T> a, int i, int j)
    {
        (a[i], a[j]) = (a[j], a[i]);
    }
}
