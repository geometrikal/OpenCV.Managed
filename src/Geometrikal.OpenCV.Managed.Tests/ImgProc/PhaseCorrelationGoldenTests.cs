namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Replays the cv2 5.0.0 <c>createHanningWindow</c>, <c>phaseCorrelate</c> and
/// <c>phaseCorrelateIterative</c> cases of <c>TestData/alignment_golden.json</c>. Windows must
/// match byte for byte; shifts and responses must match bit for bit.
/// </summary>
public sealed class PhaseCorrelationGoldenTests
{
    public static IEnumerable<object[]> HannNames() => AlignmentGolden.Names("createHanningWindow");

    public static IEnumerable<object[]> PhaseNames() => AlignmentGolden.Names("phaseCorrelate");

    public static IEnumerable<object[]> IterativeNames() => AlignmentGolden.Names("phaseCorrelateIterative");

    [Fact]
    public void GoldenFile_CoversWindowsAndTypes()
    {
        var cases = AlignmentGolden.Data.Cases.Where(c => c.Op == "phaseCorrelate").ToList();
        Assert.Contains(cases, c => c.Inputs.ContainsKey("window"));
        Assert.Contains(cases, c => !c.Inputs.ContainsKey("window"));
        Assert.Contains(cases, c => AlignmentGolden.Dtype(c, "a") == "f64");
        Assert.True(AlignmentGolden.Data.Cases.Count(c => c.Op == "phaseCorrelateIterative") >= 6);
    }

    [Theory]
    [MemberData(nameof(HannNames))]
    public void CreateHanningWindow_MatchesOpenCvGolden(string name)
    {
        AlignmentGolden.GoldenCase c = AlignmentGolden.Case(name);
        var size = new Size(c.Params["width"].GetInt32(), c.Params["height"].GetInt32());
        byte[] actual = c.Params["dtype"].GetString() == "f32"
            ? AlignmentGolden.Bytes(PhaseCorrelation.CreateHanningWindow<float>(size))
            : AlignmentGolden.Bytes(PhaseCorrelation.CreateHanningWindow<double>(size));
        if (c.ExpectedSha is not null)
        {
            Assert.Equal(c.ExpectedSha, Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(actual)).ToLowerInvariant());
            return;
        }

        byte[] expected = Convert.FromBase64String(AlignmentGolden.Data.Arrays[c.Expected!.Value].Data);
        Assert.Equal(expected.Length, actual.Length);
        int esz = c.Params["dtype"].GetString() == "f32" ? 4 : 8;
        for (int i = 0; i < expected.Length; i += esz)
        {
            if (!expected.AsSpan(i, esz).SequenceEqual(actual.AsSpan(i, esz)))
            {
                int k = i / esz;
                Assert.Fail($"Golden mismatch for '{name}' at ({k % size.Width}, {k / size.Width}).");
            }
        }
    }

    [Theory]
    [MemberData(nameof(PhaseNames))]
    public void PhaseCorrelate_MatchesOpenCvGolden(string name)
    {
        AlignmentGolden.GoldenCase c = AlignmentGolden.Case(name);
        (Point2d shift, double response) = AlignmentGolden.Dtype(c, "a") == "f32" ? Run<float>(c) : Run<double>(c);
        Assert.Equal(AlignmentGolden.Bits(c.Scalars["x"]), shift.X);
        Assert.Equal(AlignmentGolden.Bits(c.Scalars["y"]), shift.Y);
        Assert.Equal(AlignmentGolden.Bits(c.Scalars["response"]), response);
    }

    [Theory]
    [MemberData(nameof(IterativeNames))]
    public void PhaseCorrelateIterative_MatchesOpenCvGolden(string name)
    {
        AlignmentGolden.GoldenCase c = AlignmentGolden.Case(name);
        int l2 = c.Params["L2size"].GetInt32(), iters = c.Params["maxIters"].GetInt32();
        Point2d shift = AlignmentGolden.Dtype(c, "a") == "f32"
            ? PhaseCorrelation.PhaseCorrelateIterative(AlignmentGolden.Input<float>(c, "a"), AlignmentGolden.Input<float>(c, "b"), l2, iters)
            : PhaseCorrelation.PhaseCorrelateIterative(AlignmentGolden.Input<double>(c, "a"), AlignmentGolden.Input<double>(c, "b"), l2, iters);
        Assert.Equal(AlignmentGolden.Bits(c.Scalars["x"]), shift.X);
        Assert.Equal(AlignmentGolden.Bits(c.Scalars["y"]), shift.Y);
    }

    private static (Point2d Shift, double Response) Run<T>(AlignmentGolden.GoldenCase c)
        where T : unmanaged, System.Numerics.IBinaryFloatingPointIeee754<T>
    {
        Mat<T>? window = c.Inputs.ContainsKey("window") ? AlignmentGolden.Input<T>(c, "window") : null;
        Point2d shift = PhaseCorrelation.PhaseCorrelate(AlignmentGolden.Input<T>(c, "a"), AlignmentGolden.Input<T>(c, "b"), window, out double response);
        return (shift, response);
    }
}
