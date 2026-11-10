namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Pins the OpenCV 5.0 behaviour of the math function golden cases where cv2 4.5.5 differs (see
/// the reasons in <c>generate_mathfuncs_golden.py</c>):
/// <list type="bullet">
/// <item>64F <c>cartToPolar</c> and <c>polarToCart</c>: OpenCV 5.0 computes through <c>float</c>; the
/// golden values are double. The port must equal the golden values rounded to float precision.</item>
/// <item>64F <c>polarToCart</c> without magnitudes: cv2 4.5.5 returns uninitialised memory; the port
/// must give <c>cos</c>/<c>sin</c> to float accuracy.</item>
/// <item>32F <c>polarToCart</c>: OpenCV 5.0 uses <c>v_sincos</c> for vector blocks and a double
/// magnitude product; both forms approximate <c>cos</c>/<c>sin</c> to about 1e-7.</item>
/// <item><c>exp</c>, <c>log</c> and real <c>pow</c>: the cv2 build uses an approximate vector math
/// routine; the port uses the C runtime per element. Differences stay within a few ulps.</item>
/// </list>
/// </summary>
public sealed class MathFunctionsOpenCv5DifferenceTests
{
    /// <summary>Golden cases marked suspect in <c>generate_mathfuncs_golden.py</c>.</summary>
    internal static readonly string[] SuspectCases = BuildSuspectList();

    public static IEnumerable<object[]> Suspects() => SuspectCases.Select(n => new object[] { n });

    [Theory]
    [MemberData(nameof(Suspects))]
    public void SuspectCase_MatchesOpenCv5Behaviour(string name)
    {
        var c = MathFunctionsGoldenTests.Find(name);
        string dtype = c.Inputs.First(i => i is not null)!.Dtype;
        byte[][] actual = dtype == "f32" ? MathFunctionsGoldenTests.Run<float>(c) : MathFunctionsGoldenTests.Run<double>(c);
        bool unitDouble = c.Op == "polarToCart" && c.Inputs[0] is null && dtype == "f64";
        for (int k = 0; k < actual.Length; k++)
        {
            double[] got = ToDoubles(actual[k], dtype);
            double[] expected = unitDouble
                ? Analytic(c, k)
                : ToDoubles(Convert.FromBase64String(c.Outputs[k].Data), dtype);
            double[]? angles = c.Op == "polarToCart" ? ToDoubles(Convert.FromBase64String(c.Inputs[1]!.Data), dtype) : null;
            bool degrees = c.Params.TryGetValue("degrees", out var d) && d.GetBoolean();
            for (int i = 0; i < got.Length; i++)
            {
                if (double.IsNaN(expected[i]) || double.IsInfinity(expected[i]))
                {
                    Assert.Equal(expected[i], got[i]);
                    continue;
                }

                double tolerance = Tolerance(c.Op, expected[i], c);
                if (angles is not null)
                {
                    // OpenCV 5.0 converts the angle to radians in float before the reduction.
                    double radians = Math.Abs(angles[i]) * (degrees ? Math.PI / 180 : 1);
                    tolerance += radians * 6e-8 * Math.Max(1, MaxMagnitude(c));
                }

                Assert.True(Math.Abs(got[i] - expected[i]) <= tolerance,
                    $"{name} output {k} element {i}: expected {expected[i]:R}, actual {got[i]:R}, tolerance {tolerance:G3}");
            }
        }
    }

    [Fact]
    public void CartToPolar_Double_MagnitudeHasFloatPrecision()
    {
        var x = new Mat<double>([1.0 / 3, 2.0 / 7], 2, 1);
        var y = new Mat<double>([0.1, 0.3], 2, 1);

        var (mag, _) = MathFunctions.CartToPolar(x, y);

        Assert.Equal((double)MathF.Sqrt(((float)(1.0 / 3) * (float)(1.0 / 3)) + (0.1f * 0.1f)), mag.At(0, 0));
    }

    private static double Tolerance(string op, double expected, MathFunctionsGoldenTests.GoldenCase c)
    {
        double magnitude = Math.Abs(expected);
        return op switch
        {
            "polarToCart" => 2e-6 * Math.Max(1, MaxMagnitude(c)),
            "cartToPolar" => 1e-6 * Math.Max(1, magnitude),
            "pow" => 4e-6 * Math.Max(1e-30, magnitude),
            _ => 3e-7 * Math.Max(1e-30, magnitude),
        };
    }

    private static double MaxMagnitude(MathFunctionsGoldenTests.GoldenCase c)
    {
        if (c.Inputs[0] is not { } m)
        {
            return 1;
        }

        return ToDoubles(Convert.FromBase64String(m.Data), m.Dtype).Max(Math.Abs);
    }

    private static double[] Analytic(MathFunctionsGoldenTests.GoldenCase c, int output)
    {
        bool degrees = c.Params["degrees"].GetBoolean();
        double[] angle = ToDoubles(Convert.FromBase64String(c.Inputs[1]!.Data), "f64");
        return angle.Select(a =>
        {
            double radians = (float)a * (degrees ? Math.PI / 180 : 1);
            return output == 0 ? Math.Cos(radians) : Math.Sin(radians);
        }).ToArray();
    }

    private static double[] ToDoubles(byte[] bytes, string dtype) => dtype == "f32"
        ? MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(v => (double)v).ToArray()
        : MemoryMarshal.Cast<byte, double>(bytes).ToArray();

    private static string[] BuildSuspectList()
    {
        string[] sizes = ["1x5", "2x1030", "3x13c3", "9x37"];
        string[] vector = ["2x1030", "3x13c3", "9x37"];
        string[] units = ["deg", "rad"];
        var list = new List<string>();
        list.AddRange(sizes.SelectMany(s => units.Select(u => $"cartToPolar_f64_{s}_{u}")));
        list.AddRange(sizes.SelectMany(s => units.Select(u => $"polarToCart_f64_{s}_{u}")));
        list.AddRange(sizes.SelectMany(s => units.Select(u => $"polarToCartUnit_f64_{s}_{u}")));
        list.AddRange(vector.SelectMany(s => units.Select(u => $"polarToCart_f32_{s}_{u}")));
        list.Add("polarToCart_f32_1x5_deg");
        list.AddRange(vector.SelectMany(s => units.Select(u => $"polarToCartUnit_f32_{s}_{u}")));
        list.AddRange(
        [
            "exp_f32_1x5", "exp_f32_2x1030", "exp_f32_3x13c3", "exp_f32_9x37", "exp_f64_2x1030", "exp_f64_3x13c3", "exp_f64_9x37",
            "log_f32_2x1030", "log_f32_3x13c3", "log_f32_9x37", "log_f64_2x1030",
            "pow_f32_2x1030_p-2.3", "pow_f32_2x1030_p1.7", "pow_f32_3x13c3_p-2.3", "pow_f32_3x13c3_p1.7",
            "pow_f32_9x37_p-2.3", "pow_f32_9x37_p1.7", "pow_f64_2x1030_p-2.3",
        ]);
        return [.. list];
    }
}
