namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Process-wide multithreading setting for image operations, equivalent to OpenCV
/// <c>cv::setNumThreads</c>.
/// </summary>
/// <remarks>
/// <para>Filters and morphology split the output rows into stripes and process the stripes
/// in parallel. Small images stay on the calling thread.</para>
/// <para>Results do not depend on this setting. Every output element is computed with the same
/// operations in the same order for any thread count.</para>
/// <para>Contours and drawing are always single-threaded.</para>
/// </remarks>
public static class CvParallel
{
    private static int _maxDegreeOfParallelism = Environment.ProcessorCount;

    /// <summary>
    /// Maximum number of threads one operation uses. The default is
    /// <see cref="Environment.ProcessorCount"/>. The value 1 disables multithreading.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is less than 1.</exception>
    public static int MaxDegreeOfParallelism
    {
        get => Volatile.Read(ref _maxDegreeOfParallelism);
        set
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
            Volatile.Write(ref _maxDegreeOfParallelism, value);
        }
    }
}
