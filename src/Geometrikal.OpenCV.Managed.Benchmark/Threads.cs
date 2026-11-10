using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Applies the thread count from the <c>CV_THREADS</c> environment variable ("all" or a
/// number). The default is 1, which matches the single-threaded cv2 modes.
/// </summary>
internal static class Threads
{
    public const string Variable = "CV_THREADS";

    public static void Apply()
    {
        string? value = Environment.GetEnvironmentVariable(Variable);
        CvParallel.MaxDegreeOfParallelism = value switch
        {
            null or "" => 1,
            "all" => Environment.ProcessorCount,
            _ => int.Parse(value),
        };
    }
}
