using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using Geometrikal.OpenCV.Managed.Benchmark;
using Geometrikal.OpenCV.Managed.ImgProc;
using Perfolizer.Horology;

if (args is ["--checksums"])
{
    // Compare with `python tools/bench_cv2.py --checksums` to confirm identical inputs.
    foreach (int size in new[] { 256, 1024 })
    {
        long gray = Inputs.Gray(size).ToArray().Sum(b => (long)b);
        long color = Inputs.Color(size).ToArray().Sum(b => (long)b);
        var binary = Inputs.Binary(size);
        long white = binary.ToArray().Count(b => b != 0);
        int contours = Contours.FindContours(binary, RetrievalMode.Tree, ContourApproximationMode.Simple).Contours.Count;
        Console.WriteLine($"size={size} gray={gray} color={color} binary={white} contours={contours}");
    }

    return;
}

// --quick runs a 3-iteration job for spot checks; --fast runs 10 iterations of at least 50 ms;
// full runs use 15 iterations of at least 100 ms. --inProcess runs the job in this process.
bool quick = args.Contains("--quick");
bool fast = args.Contains("--fast");
bool inProcess = args.Contains("--inProcess");
args = args.Where(a => a is not ("--quick" or "--fast" or "--inProcess")).ToArray();

// --threads N|all sets CvParallel.MaxDegreeOfParallelism in the benchmark processes and
// writes to a separate artifacts folder.
string threads = "1";
int threadsIndex = Array.IndexOf(args, "--threads");
if (threadsIndex >= 0 && threadsIndex + 1 < args.Length)
{
    threads = args[threadsIndex + 1];
    args = args.Where((_, i) => i != threadsIndex && i != threadsIndex + 1).ToArray();
}

Environment.SetEnvironmentVariable(Threads.Variable, threads);

// Methods with loops are compiled fully optimised at once, so that short warm-ups do not measure
// tier-0 or OSR code. The csproj sets the same option for --inProcess runs.
Job job = (quick
    ? Job.ShortRun
    : Job.Default.WithWarmupCount(3)
        .WithIterationCount(fast ? 10 : 15)
        .WithIterationTime(TimeInterval.FromMilliseconds(fast ? 50 : 100)))
    .WithEnvironmentVariable("DOTNET_TC_QuickJitForLoops", "0");
if (inProcess)
{
    job = job.WithToolchain(InProcessEmitToolchain.Instance);
}

var config = DefaultConfig.Instance
    .AddJob(job)
    .AddExporter(JsonExporter.FullCompressed)
    .AddExporter(MarkdownExporter.GitHub);
// DOTNET_MaxVectorTBitWidth=512 (inherited by the benchmark processes) widens Vector<T> on AVX-512
// hardware; those runs write to folders with a "-v512" suffix.
string vector = System.Numerics.Vector<byte>.Count == 64 ? "-v512" : string.Empty;
if (threads != "1" || vector.Length > 0)
{
    string folder = threads switch
    {
        "1" => "BenchmarkDotNet.Artifacts",
        "all" => "BenchmarkDotNet.Artifacts-mt",
        _ => $"BenchmarkDotNet.Artifacts-t{threads}",
    };
    config = config.WithArtifactsPath(Path.Combine(Directory.GetCurrentDirectory(), folder + vector));
}

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
