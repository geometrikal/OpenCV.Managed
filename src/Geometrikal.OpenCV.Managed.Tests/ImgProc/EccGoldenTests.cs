namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 <c>computeECC</c> and <c>findTransformECC</c> cases of
/// <c>TestData/alignment_golden.json</c>. The correlation coefficients and the final warp
/// matrices must match bit for bit.
/// </summary>
public sealed class EccGoldenTests
{
    public static IEnumerable<object[]> ComputeNames() => AlignmentGolden.Names("computeECC");

    public static IEnumerable<object[]> FindNames() => AlignmentGolden.Names("findTransformECC");

    [Fact]
    public void GoldenFile_CoversMotionTypesAndMasks()
    {
        var cases = AlignmentGolden.Data.Cases.Where(c => c.Op == "findTransformECC").ToList();
        Assert.Equal(4, cases.Select(c => c.Params["motion"].GetString()).Distinct().Count());
        Assert.Contains(cases, c => c.Inputs.ContainsKey("templateMask"));
        Assert.Contains(cases, c => c.Inputs.ContainsKey("inputMask") && !c.Inputs.ContainsKey("templateMask"));
    }

    [Theory]
    [MemberData(nameof(ComputeNames))]
    public void ComputeEcc_MatchesOpenCvGolden(string name)
    {
        AlignmentGolden.GoldenCase c = AlignmentGolden.Case(name);
        Mat<byte>? mask = c.Inputs.ContainsKey("inputMask") ? AlignmentGolden.Input<byte>(c, "inputMask") : null;
        double ecc = AlignmentGolden.Dtype(c, "templ") == "u8"
            ? EccAlignment.ComputeEcc(AlignmentGolden.Input<byte>(c, "templ"), AlignmentGolden.Input<byte>(c, "input"), mask)
            : EccAlignment.ComputeEcc(AlignmentGolden.Input<float>(c, "templ"), AlignmentGolden.Input<float>(c, "input"), mask);
        Assert.Equal(AlignmentGolden.Bits(c.Scalars["ecc"]), ecc);
    }

    [Theory]
    [MemberData(nameof(FindNames))]
    public void FindTransformEcc_MatchesOpenCvGolden(string name)
    {
        AlignmentGolden.GoldenCase c = AlignmentGolden.Case(name);
        (double rho, float[] warp) = AlignmentGolden.Dtype(c, "templ") == "u8" ? Run<byte>(c) : Run<float>(c);
        double[] expWarp = c.Scalars["warp"].EnumerateArray().Select(AlignmentGolden.Bits).ToArray();
        double expRho = AlignmentGolden.Bits(c.Scalars["rho"]);
        double maxDiff = expWarp.Select((e, i) => Math.Abs(e - warp[i])).Max();
        Assert.True(
            maxDiff == 0 && BitConverter.DoubleToInt64Bits(rho) == BitConverter.DoubleToInt64Bits(expRho),
            $"'{name}': warp max diff {maxDiff:E3}, rho {rho:R} vs {expRho:R}.");
    }

    private static (double Rho, float[] Warp) Run<T>(AlignmentGolden.GoldenCase c)
        where T : unmanaged
    {
        MotionTypes motion = Enum.Parse<MotionTypes>(c.Params["motion"].GetString()!);
        int rows = motion == MotionTypes.Homography ? 3 : 2;
        var warp = new Mat<float>(3, rows);
        LinearAlgebra.SetIdentity(warp);
        bool countOnly = c.Params.TryGetValue("countOnly", out var co) && co.GetBoolean();
        var criteria = new TermCriteria(
            countOnly ? TermCriteriaType.MaxIter : TermCriteriaType.MaxIter | TermCriteriaType.Eps,
            c.Params["maxCount"].GetInt32(),
            c.Params["eps"].GetDouble());
        Mat<byte>? inputMask = c.Inputs.ContainsKey("inputMask") ? AlignmentGolden.Input<byte>(c, "inputMask") : null;
        Mat<byte>? templateMask = c.Inputs.ContainsKey("templateMask") ? AlignmentGolden.Input<byte>(c, "templateMask") : null;
        double rho = EccAlignment.FindTransformEccWithMask(
            AlignmentGolden.Input<T>(c, "templ"),
            AlignmentGolden.Input<T>(c, "input"),
            templateMask,
            inputMask,
            warp,
            motion,
            criteria,
            c.Params["gauss"].GetInt32());
        return (rho, warp.ToArray());
    }
}
