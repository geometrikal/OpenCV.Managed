namespace Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Splits a row range into contiguous stripes and runs them in parallel, limited by
/// <see cref="CvParallel.MaxDegreeOfParallelism"/>.
/// </summary>
internal static class ParallelRows
{
    /// <summary>Minimum work per stripe, in element-operations, that justifies a thread.</summary>
    private const long MinWorkPerStripe = 1 << 16;

    [ThreadStatic]
    private static bool t_serial;

    /// <summary>
    /// Calls <paramref name="body"/> with disjoint ranges <c>[start, end)</c> that cover
    /// <c>[0, count)</c>. <paramref name="workPerRow"/> estimates the element-operations per row.
    /// </summary>
    public static void For(int count, long workPerRow, Action<int, int> body)
    {
        int stripes = StripeCount(count, workPerRow);
        if (stripes <= 1)
        {
            body(0, count);
            return;
        }

        Parallel.For(
            0,
            stripes,
            new ParallelOptions { MaxDegreeOfParallelism = stripes },
            s => body((int)((long)count * s / stripes), (int)((long)count * (s + 1) / stripes)));
    }

    /// <summary>
    /// Runs <paramref name="body"/> for every index in <c>[0, count)</c> on up to
    /// <see cref="CvParallel.MaxDegreeOfParallelism"/> threads. Every <see cref="For"/> call made
    /// by <paramref name="body"/> stays on the calling thread.
    /// </summary>
    public static void ForEachSerial(int count, Action<int> body)
    {
        int dop = t_serial ? 1 : Math.Min(CvParallel.MaxDegreeOfParallelism, count);
        if (dop <= 1)
        {
            bool previous = t_serial;
            t_serial = true;
            try
            {
                for (int i = 0; i < count; i++)
                {
                    body(i);
                }
            }
            finally
            {
                t_serial = previous;
            }

            return;
        }

        Parallel.For(
            0,
            count,
            new ParallelOptions { MaxDegreeOfParallelism = dop },
            i =>
            {
                t_serial = true;
                try
                {
                    body(i);
                }
                finally
                {
                    t_serial = false;
                }
            });
    }

    /// <summary>Number of stripes that <see cref="For"/> uses for these arguments.</summary>
    public static int Stripes(int count, long workPerRow) => Math.Max(1, StripeCount(count, workPerRow));

    /// <summary>True when <see cref="For"/> with these arguments runs on the calling thread only.</summary>
    public static bool IsSerial(int count, long workPerRow) => StripeCount(count, workPerRow) <= 1;

    private static int StripeCount(int count, long workPerRow)
    {
        int dop = CvParallel.MaxDegreeOfParallelism;
        if (dop <= 1 || count <= 1 || t_serial)
        {
            return 1;
        }

        long byWork = count * Math.Max(workPerRow, 1) / MinWorkPerStripe;
        return (int)Math.Min(Math.Min(dop, count), byWork);
    }
}
