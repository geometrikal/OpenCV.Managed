using System.Runtime.InteropServices;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Shared setup for the OpenCvSharp suites: native copies of the <see cref="Inputs"/> data and
/// the native thread count.
/// </summary>
internal static class OcvInputs
{
    /// <summary>Sets the native OpenCV thread count from the <c>CV_THREADS</c> environment variable.</summary>
    public static void ApplyThreads()
    {
        string? value = Environment.GetEnvironmentVariable(Threads.Variable);
        Cv2.SetNumThreads(value switch
        {
            null or "" => 1,
            "all" => Environment.ProcessorCount,
            _ => int.Parse(value),
        });
    }

    /// <summary>Copies a managed port matrix into a new continuous native matrix of the same depth and channel count.</summary>
    public static Mat ToOcv<T>(Port.Mat<T> source)
        where T : unmanaged
    {
        int depth = default(T) switch
        {
            byte => MatType.CV_8U,
            sbyte => MatType.CV_8S,
            ushort => MatType.CV_16U,
            short => MatType.CV_16S,
            int => MatType.CV_32S,
            float => MatType.CV_32F,
            double => MatType.CV_64F,
            _ => throw new NotSupportedException(typeof(T).Name),
        };
        var mat = new Mat(source.Height, source.Width, MatType.MakeType(depth, source.Channels));
        byte[] bytes = MemoryMarshal.AsBytes(source.ToArray().AsSpan()).ToArray();
        Marshal.Copy(bytes, 0, mat.Data, bytes.Length);
        return mat;
    }

    public static Point ToOcv(Port.Point point) => new(point.X, point.Y);

    /// <summary>Returns a value that depends on the result, so that the call is not dead code.</summary>
    public static int Consume(Mat result) => result.Rows;
}
