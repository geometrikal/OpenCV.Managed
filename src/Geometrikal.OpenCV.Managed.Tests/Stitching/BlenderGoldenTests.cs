namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;
using static StitchGolden;

/// <summary>
/// Replays the cv2 5.0.0 blender reference outputs in <c>TestData/stitch_warp_golden.json</c>
/// (FeatherBlender, MultiBandBlender, createWeightMap, createLaplacePyr,
/// restoreImageFromLaplacePyr, normalizeUsingWeightMap). Outputs must match byte for byte.
/// </summary>
public sealed class BlenderGoldenTests
{
    private static readonly string[] Ops = ["feather", "multiband", "createWeightMap", "createLaplacePyr", "restoreLaplacePyr", "normalizeUsingWeightMap"];

    public static IEnumerable<object[]> CaseNames() => Names(Ops);

    [Fact]
    public void GoldenFile_CoversEveryBlenderOperation()
        => Assert.Equal(Ops.OrderBy(s => s, StringComparer.Ordinal), Data.Cases.Where(c => Ops.Contains(c.Op)).Select(c => c.Op).Distinct().OrderBy(s => s, StringComparer.Ordinal));

    /// <summary>
    /// The cv2 binding returns the unmodified input as level 0 of <c>createLaplacePyr</c> with a
    /// <c>CV_16S</c> source (the C++ code writes into the UMat that wraps the input).
    /// <c>BlenderBasicTests.CreateLaplacePyr_Short_LevelsRestoreSource</c> pins the port.
    /// </summary>
    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Equal(
            new[] { "createLaplacePyr_i16_l1", "createLaplacePyr_i16_l3" },
            Data.Cases.Where(c => c.Suspect && Ops.Contains(c.Op)).Select(c => c.Name).OrderBy(s => s, StringComparer.Ordinal));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Blender_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Case(name);
        switch (c.Op)
        {
            case "feather":
            case "multiband":
            {
                Point[] corners = Corners(c);
                var images = new List<Mat<byte>>();
                var masks = new List<Mat<byte>>();
                for (int i = 0; i < corners.Length; i++)
                {
                    images.Add(Input<byte>(c, 2 * i));
                    masks.Add(Input<byte>(c, (2 * i) + 1));
                }

                Blender blender;
                if (c.Op == "feather")
                {
                    blender = new FeatherBlender(Bits(c.Params["sharpness"]));
                }
                else
                {
                    var type = c.Params["weightType"].GetString() == "f32" ? BlendWeightType.Float : BlendWeightType.Short;
                    blender = new MultiBandBlender(c.Params["bands"].GetInt32(), type);
                }

                blender.Prepare(StitchingUtils.ResultRoi(corners, images.Select(m => new Size(m.Width, m.Height)).ToArray()));
                bool src16 = c.Op == "feather" || c.Params["src16"].GetBoolean();
                for (int i = 0; i < corners.Length; i++)
                {
                    if (src16)
                    {
                        blender.Feed(Conversions.ConvertTo<byte, short>(images[i]), masks[i], corners[i]);
                    }
                    else
                    {
                        ((MultiBandBlender)blender).Feed(images[i], masks[i], corners[i]);
                    }
                }

                blender.Blend(out Mat<short> dst, out Mat<byte> dstMask);
                AssertOutput(c, 0, dst);
                AssertOutput(c, 1, dstMask);
                break;
            }

            case "createWeightMap":
                AssertOutput(c, 0, Blender.CreateWeightMap(Input<byte>(c, 0), Bits(c.Params["sharpness"])));
                break;

            case "createLaplacePyr":
            {
                int levels = c.Params["levels"].GetInt32();
                ArrayData a = Data.Arrays[c.Inputs[0]];
                Mat<short>[] pyr = a.Dtype == "u8" ? Blender.CreateLaplacePyr(Input<byte>(c, 0), levels) : Blender.CreateLaplacePyr(Input<short>(c, 0), levels);
                Assert.Equal(c.Outputs.Count, pyr.Length);
                for (int i = 0; i < pyr.Length; i++)
                {
                    AssertOutput(c, i, pyr[i]);
                }

                break;
            }

            case "restoreLaplacePyr":
            {
                Mat<short>[] pyr = Enumerable.Range(0, c.Inputs.Count).Select(i => Input<short>(c, i)).ToArray();
                Blender.RestoreImageFromLaplacePyr(pyr);
                AssertOutput(c, 0, pyr[0]);
                break;
            }

            default:
            {
                Mat<short> img = Input<short>(c, 1);
                if (Data.Arrays[c.Inputs[0]].Dtype == "f32")
                {
                    Blender.NormalizeUsingWeightMap(Input<float>(c, 0), img);
                }
                else
                {
                    Blender.NormalizeUsingWeightMap(Input<short>(c, 0), img);
                }

                AssertOutput(c, 0, img);
                break;
            }
        }
    }
}
