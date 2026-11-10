// Writes the output hash of every RegressionCases case to the JSON file given as the first argument.
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Tests.Regression;

var result = new SortedDictionary<string, string>(StringComparer.Ordinal);
foreach (var (name, run) in RegressionCases.All())
{
    string hash;
    try
    {
        hash = run();
    }
    catch (Exception ex)
    {
        hash = "throws:" + ex.GetType().Name;
    }

    result.Add(name, hash);
}

File.WriteAllText(args[0], JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine($"{result.Count} cases, {result.Values.Count(v => v.StartsWith("throws", StringComparison.Ordinal))} throw");
