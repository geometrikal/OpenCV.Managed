namespace Geometrikal.OpenCV.Managed.Tests.Regression;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

public class RegressionSnapshotTests
{
    private static readonly Lazy<Dictionary<string, string>> Expected = new(() =>
        JsonSerializer.Deserialize<Dictionary<string, string>>(
            File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "regression_hashes.json")))!);

    private static readonly Lazy<Dictionary<string, Func<string>>> Cases = new(() =>
        RegressionCases.All().ToDictionary(c => c.Name, c => c.Run));

    public static IEnumerable<object[]> CaseNames() => RegressionCases.All().Select(c => new object[] { c.Name });

    [Fact]
    public void Snapshot_CoversEveryCase()
    {
        Assert.Equal(Cases.Value.Keys.Order(), Expected.Value.Keys.Order());
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Output_MatchesSnapshot(string name)
    {
        Assert.Equal(Expected.Value[name], Run(Cases.Value[name]));
    }

    /// <summary>Returns the output hash, or the exception type name when the case throws.</summary>
    public static string Run(Func<string> run)
    {
        try
        {
            return run();
        }
        catch (Exception ex)
        {
            return "throws:" + ex.GetType().Name;
        }
    }
}
