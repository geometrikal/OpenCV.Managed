namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins the OpenCV 5.0 behaviour of the Hough golden cases where cv2 4.5.5 differs:
/// <list type="bullet">
/// <item>PR #27992: standard <c>HoughLines</c> outputs ρ with the integer bin offset
/// <c>(numrho − 1)/2</c>, the offset of the accumulator. 4.5.5 subtracted <c>(numrho − 1)·0.5</c>,
/// so with an even <c>numrho</c> its ρ is half a step lower.</item>
/// <item>Commit 1f0bfc8d83: <c>numangle = floor((max − min)/step) + 1</c> (minus one when the range
/// is about π); 4.5.5 used <c>cvRound((max − min)/step)</c>.</item>
/// </list>
/// </summary>
public sealed class HoughOpenCv5DifferenceTests
{
    internal static readonly string[] RhoBiasCases =
    [
        "lines_a_r2_t0.0349_th10", "lines_b_r2_t0.0349_th10", "lines_c_r2_t0.0349_th10", "lines_sparse_r2_t0.0349_th10",
        "lines_b_r0.5_t0.0087_th25", "lines_c_r0.5_t0.0087_th25",
    ];

    public static TheoryData<string> RhoBias() => new(RhoBiasCases);

    [Theory]
    [MemberData(nameof(RhoBias))]
    public void HoughLines_EvenRhoBinCount_IsHalfABinAboveCv2(string name)
    {
        var c = HoughGoldenTests.Find(name);
        var p = c.GetProperty("params");
        double rho = p.GetProperty("rho").GetDouble();
        var lines = HoughTransforms.HoughLines(HoughGoldenTests.ImageOf(c), rho, p.GetProperty("theta").GetDouble(), p.GetProperty("threshold").GetInt32());
        var expected = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(c.GetProperty("result").GetString()!)).ToArray();

        Assert.Equal(c.GetProperty("count").GetInt32(), lines.Length);
        for (int i = 0; i < lines.Length; i++)
        {
            Assert.Equal(expected[2 * i] + ((float)rho / 2), lines[i].Rho);
            Assert.Equal(expected[(2 * i) + 1], lines[i].Theta);
        }
    }

    [Fact]
    public void HoughLines_AngleRange_HasTheFloorPlusOneAngles()
    {
        var c = HoughGoldenTests.Find("lines_c_range");
        double step = Math.PI / 180;
        var lines = HoughTransforms.HoughLines(HoughGoldenTests.ImageOf(c), 1, step, 12, minTheta: 0.3, maxTheta: 2.0);
        var expected = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(c.GetProperty("result").GetString()!)).ToArray();
        var actual = lines.Select(l => (l.Rho, l.Theta)).ToHashSet();
        float lastOld = (float)0.3 + (96 * (float)step);
        float added = (float)0.3 + (97 * (float)step);

        // The 98th angle exists in OpenCV 5.0 and finds lines there.
        Assert.Contains(lines, l => l.Theta == added);

        // Lines away from the old last angle are unchanged.
        for (int i = 0; i < expected.Length; i += 2)
        {
            if (expected[i + 1] < lastOld)
            {
                Assert.Contains((expected[i], expected[i + 1]), actual);
            }
        }
    }
}
