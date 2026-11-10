namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 <c>matchTemplate</c> cases of <c>TestData/alignment_golden.json</c>.
/// Results must match byte for byte.
/// </summary>
public sealed class TemplateMatchingGoldenTests
{
    public static IEnumerable<object[]> CaseNames() => AlignmentGolden.Names("matchTemplate");

    [Fact]
    public void GoldenFile_CoversMethodsTypesAndMasks()
    {
        var cases = AlignmentGolden.Data.Cases.Where(c => c.Op == "matchTemplate").ToList();
        Assert.True(cases.Count > 200);
        Assert.Equal(6, cases.Select(c => c.Params["method"].GetString()).Distinct().Count());
        Assert.Contains(cases, c => c.Inputs.ContainsKey("mask") && AlignmentGolden.Dtype(c, "mask") == "f32");
        Assert.Contains(cases, c => c.Inputs.ContainsKey("mask") && AlignmentGolden.Dtype(c, "mask") == "u8");
        Assert.Contains(cases, c => c.ExpectedSha is not null);
    }

    /// <summary>No golden case differs between cv2 5.0.0 and the port.</summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(AlignmentGolden.Data.Cases.Where(c => c.Suspect).Select(c => c.Name));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void MatchTemplate_MatchesOpenCvGolden(string name)
    {
        AlignmentGolden.GoldenCase c = AlignmentGolden.Case(name);
        Mat<float> actual = AlignmentGolden.Dtype(c, "image") == "u8" ? Run<byte>(c) : Run<float>(c);
        byte[] bytes = AlignmentGolden.Bytes(actual);
        if (c.ExpectedSha is not null)
        {
            Assert.Equal(c.ExpectedSha, Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());
            return;
        }

        Mat<float> exp = AlignmentGolden.LoadMat<float>(AlignmentGolden.Data.Arrays[c.Expected!.Value]);
        Assert.Equal(exp.Width, actual.Width);
        Assert.Equal(exp.Height, actual.Height);
        for (int y = 0; y < exp.Height; y++)
        {
            for (int x = 0; x < exp.Width; x++)
            {
                float e = exp.At(y, x), a = actual.At(y, x);
                if (BitConverter.SingleToInt32Bits(e) != BitConverter.SingleToInt32Bits(a))
                {
                    Assert.Fail($"Golden mismatch for '{name}' at ({x}, {y}): expected {e:R}, actual {a:R}.");
                }
            }
        }
    }

    private static Mat<float> Run<T>(AlignmentGolden.GoldenCase c)
        where T : unmanaged
    {
        Mat<T> image = AlignmentGolden.Input<T>(c, "image");
        Mat<T> templ = AlignmentGolden.Input<T>(c, "templ");
        TemplateMatchModes method = Enum.Parse<TemplateMatchModes>(c.Params["method"].GetString()!);
        if (!c.Inputs.ContainsKey("mask"))
        {
            return TemplateMatching.MatchTemplate(image, templ, method);
        }

        return AlignmentGolden.Dtype(c, "mask") == "u8"
            ? TemplateMatching.MatchTemplate(image, templ, method, AlignmentGolden.Input<byte>(c, "mask"))
            : TemplateMatching.MatchTemplate(image, templ, method, AlignmentGolden.Input<float>(c, "mask"));
    }
}
