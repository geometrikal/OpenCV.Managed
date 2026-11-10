namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Stitching;
using static StitchGolden;

/// <summary>
/// Replays the cv2 5.0.0 <c>PyRotationWarper</c> reference outputs in
/// <c>TestData/stitch_warp_golden.json</c> (plane, affine, cylindrical, spherical). Points, maps
/// and warped images must match bit for bit. Regenerate with
/// <c>tools/generate_stitch_warp_golden.py</c>.
/// </summary>
public sealed class WarperGoldenTests
{
    private static readonly string[] Ops = ["warpPoint", "buildMaps", "warp", "warpBackward"];

    /// <summary>Largest 8-bit difference between the managed table-path bilinear remap and the cv2 5.0 float kernel.</summary>
    private const int LinearTolerance = 1;

    /// <summary>As <see cref="LinearTolerance"/> with a constant border; the kernels blend the border value differently at the source edge.</summary>
    private const int LinearConstantTolerance = 4;

    /// <summary>Largest 8-bit difference between the managed table-path bicubic remap and the cv2 5.0 float kernel.</summary>
    private const int CubicTolerance = 1;

    public static IEnumerable<object[]> CaseNames() => Names(Ops);

    [Fact]
    public void GoldenFile_CoversEveryWarperAndOperation()
    {
        var cases = Data.Cases.Where(c => Ops.Contains(c.Op)).ToList();
        Assert.True(cases.Count > 100);
        Assert.Equal(new[] { "affine", "cylindrical", "plane", "spherical" }, cases.Select(c => c.Params["type"].GetString()!).Distinct().OrderBy(s => s, StringComparer.Ordinal));
        Assert.Equal(Ops.OrderBy(s => s, StringComparer.Ordinal), cases.Select(c => c.Op).Distinct().OrderBy(s => s, StringComparer.Ordinal));
    }

    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Data.Cases.Where(c => c.Suspect && Ops.Contains(c.Op)).Select(c => c.Name));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Warper_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Case(name);
        RotationWarper w = Create(c);
        Mat<float> k = Input<float>(c, 0);
        Mat<float> r = Input<float>(c, 1);
        var size = new Size(c.Params["srcWidth"].GetInt32(), c.Params["srcHeight"].GetInt32());
        switch (c.Op)
        {
            case "warpPoint":
            {
                var pts = c.Params["points"].EnumerateArray().Select(p => new Point2f(Bits(p[0]), Bits(p[1]))).ToArray();
                uint[] fw = c.Scalars["forward"].EnumerateArray().Select(e => e.GetUInt32()).ToArray();
                uint[] bw = c.Scalars["backward"].EnumerateArray().Select(e => e.GetUInt32()).ToArray();
                for (int i = 0; i < pts.Length; i++)
                {
                    Point2f uv = w.WarpPoint(pts[i], k, r);
                    Assert.Equal(fw[2 * i], BitConverter.SingleToUInt32Bits(uv.X));
                    Assert.Equal(fw[(2 * i) + 1], BitConverter.SingleToUInt32Bits(uv.Y));
                    Point2f xy = w.WarpPointBackward(uv, k, r);
                    Assert.Equal(bw[2 * i], BitConverter.SingleToUInt32Bits(xy.X));
                    Assert.Equal(bw[(2 * i) + 1], BitConverter.SingleToUInt32Bits(xy.Y));
                }

                break;
            }

            case "buildMaps":
            {
                int[] roi = Ints(c.Scalars["roi"]);
                Assert.Equal(new Rect(roi[0], roi[1], roi[2], roi[3]), w.WarpRoi(size, k, r));
                int[] mr = Ints(c.Scalars["mapsRect"]);
                Rect rect = w.BuildMaps(size, k, r, out Mat<float> mx, out Mat<float> my);
                Assert.Equal(new Rect(mr[0], mr[1], mr[2], mr[3]), rect);
                AssertOutput(c, 0, mx);
                AssertOutput(c, 1, my);
                break;
            }

            case "warp":
            {
                Mat<byte> img = Input<byte>(c, 2);
                var interp = Enum.Parse<InterpolationFlags>(c.Params["interp"].GetString()!);
                var border = Enum.Parse<BorderType>(c.Params["border"].GetString()!);
                Point corner = w.Warp(img, k, r, interp, border, out Mat<byte> dst);
                int[] ec = Ints(c.Scalars["corner"]);
                Assert.Equal(new Point(ec[0], ec[1]), corner);
                switch (interp)
                {
                    case InterpolationFlags.Nearest:
                        AssertOutput(c, 0, dst);
                        break;
                    case InterpolationFlags.Linear:
                        // The managed remap follows the table path; cv2 5.0 uses its float bilinear kernel for float maps.
                        AssertOutput(c, 1, dst);
                        int tol = border == BorderType.Constant ? LinearConstantTolerance : LinearTolerance;
                        Assert.True(ByteDiff(c, 0, dst).Max <= tol, $"max diff {ByteDiff(c, 0, dst)}");
                        break;
                    default:
                        Assert.True(ByteDiff(c, 0, dst).Max <= CubicTolerance, $"max diff {ByteDiff(c, 0, dst)}");
                        break;
                }
                break;
            }

            default:
            {
                Mat<byte> warped = Input<byte>(c, 2);
                Mat<byte> back = w.WarpBackward(warped, k, r, InterpolationFlags.Linear, BorderType.Reflect, size);
                Assert.True(ByteDiff(c, 0, back).Max <= LinearTolerance, $"max diff {ByteDiff(c, 0, back)}");
                AssertOutput(c, 1, w.WarpBackward(warped, k, r, InterpolationFlags.Nearest, BorderType.Reflect, size));
                break;
            }
        }
    }

    private static RotationWarper Create(GoldenCase c)
    {
        float scale = Bits(c.Params["scale"]);
        return c.Params["type"].GetString() switch
        {
            "plane" => new PlaneWarper(scale),
            "affine" => new AffineWarper(scale),
            "cylindrical" => new CylindricalWarper(scale),
            _ => new SphericalWarper(scale),
        };
    }
}
