namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 5.0.0 reference outputs in <c>TestData/estimation_golden.json</c> through
/// <see cref="GeometricEstimation"/>. Inlier masks must be identical; models must be bit-exact.
/// Regenerate with <c>python tools/generate_estimation_golden.py</c>.
/// </summary>
public sealed class EstimationGoldenTests
{
    private static readonly List<GoldenCase> Cases = Load();

    public static IEnumerable<object[]> CaseNames()
        => Cases.Where(c => !c.Suspect).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_LoadsAndIsNonEmpty() => Assert.True(Cases.Count > 300);

    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void GeometricEstimation_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Cases.Single(x => x.Name == name);
        if (c.Error)
        {
            Assert.ThrowsAny<Exception>(() => Run(c));
            return;
        }

        (double[]? model, byte[] mask) = Run(c);
        Assert.Equal(c.Mask, mask);
        if (c.Model is null)
        {
            Assert.Null(model);
            return;
        }

        Assert.NotNull(model);
        Assert.Equal(c.Model.Length, model!.Length);
        for (int i = 0; i < model.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(c.Model[i]) != BitConverter.DoubleToInt64Bits(model[i]))
            {
                Assert.Fail($"{c.Name}: model[{i}] = {model[i]:R}, expected {c.Model[i]:R} " +
                    $"(difference {model[i] - c.Model[i]:E3}).");
            }
        }
    }

    private static (double[]? Model, byte[] Mask) Run(GoldenCase c)
    {
        var method = (RobustEstimationMethod)c.Method;
        if (c.Dtype == "f32")
        {
            Point2f[] s = MemoryMarshal.Cast<byte, Point2f>(c.Src).ToArray();
            Point2f[] d = MemoryMarshal.Cast<byte, Point2f>(c.Dst).ToArray();
            return c.Func switch
            {
                "affine" => FromResult(GeometricEstimation.EstimateAffine2D(s, d, method, c.Threshold, c.MaxIters, c.Confidence, c.RefineIters)),
                "partial" => FromResult(GeometricEstimation.EstimateAffinePartial2D(s, d, method, c.Threshold, c.MaxIters, c.Confidence, c.RefineIters)),
                "translation" => FromResult(GeometricEstimation.EstimateTranslation2D(s, d, method, c.Threshold, c.MaxIters, c.Confidence, c.RefineIters)),
                _ => FromResult(GeometricEstimation.FindHomography(s, d, method, c.Threshold, c.MaxIters, c.Confidence)),
            };
        }
        else
        {
            Point2d[] s = MemoryMarshal.Cast<byte, Point2d>(c.Src).ToArray();
            Point2d[] d = MemoryMarshal.Cast<byte, Point2d>(c.Dst).ToArray();
            return c.Func switch
            {
                "affine" => FromResult(GeometricEstimation.EstimateAffine2D(s, d, method, c.Threshold, c.MaxIters, c.Confidence, c.RefineIters)),
                "partial" => FromResult(GeometricEstimation.EstimateAffinePartial2D(s, d, method, c.Threshold, c.MaxIters, c.Confidence, c.RefineIters)),
                "translation" => FromResult(GeometricEstimation.EstimateTranslation2D(s, d, method, c.Threshold, c.MaxIters, c.Confidence, c.RefineIters)),
                _ => FromResult(GeometricEstimation.FindHomography(s, d, method, c.Threshold, c.MaxIters, c.Confidence)),
            };
        }
    }

    private static (double[]?, byte[]) FromResult(GeometricEstimationResult r)
        => (r.Model?.ToArray(), r.InlierMask);

    private static (double[]?, byte[]) FromResult(TranslationEstimationResult r)
        => ([r.Translation.X, r.Translation.Y], r.InlierMask);

    private static List<GoldenCase> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "estimation_golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        var list = new List<GoldenCase>();
        foreach (JsonElement e in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            bool error = e.TryGetProperty("error", out JsonElement er) && er.GetBoolean();
            double[]? model = null;
            if (e.TryGetProperty("model", out JsonElement me))
            {
                model = MemoryMarshal.Cast<byte, double>(Convert.FromBase64String(me.GetString()!)).ToArray();
            }

            list.Add(new GoldenCase(
                e.GetProperty("name").GetString()!,
                e.GetProperty("func").GetString()!,
                e.GetProperty("method").GetInt32(),
                e.GetProperty("threshold").GetDouble(),
                e.GetProperty("maxIters").GetInt32(),
                e.GetProperty("confidence").GetDouble(),
                e.GetProperty("refineIters").GetInt32(),
                e.GetProperty("dtype").GetString()!,
                Convert.FromBase64String(e.GetProperty("src").GetString()!),
                Convert.FromBase64String(e.GetProperty("dst").GetString()!),
                e.GetProperty("suspect").GetBoolean(),
                error,
                model,
                error ? [] : Convert.FromBase64String(e.GetProperty("mask").GetString()!)));
        }

        return list;
    }

    private sealed record GoldenCase(
        string Name,
        string Func,
        int Method,
        double Threshold,
        int MaxIters,
        double Confidence,
        int RefineIters,
        string Dtype,
        byte[] Src,
        byte[] Dst,
        bool Suspect,
        bool Error,
        double[]? Model,
        byte[] Mask);
}
