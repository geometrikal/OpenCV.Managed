namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;
using static StitchGolden;

/// <summary>
/// Replays the cv2 5.0.0 exposure-compensator reference outputs in
/// <c>TestData/stitch_warp_golden.json</c> (gain, channels, blocks gain, blocks channels). Gains,
/// gain maps and compensated images must match bit for bit.
/// </summary>
public sealed class ExposureGoldenTests
{
    public static IEnumerable<object[]> CaseNames() => Names("exposure");

    [Fact]
    public void GoldenFile_CoversEveryCompensator()
        => Assert.Equal(
            new[] { "blockschannels", "blocksgain", "channels", "gain" },
            Data.Cases.Where(c => c.Op == "exposure").Select(c => c.Params["kind"].GetString()!).Distinct().OrderBy(s => s, StringComparer.Ordinal));

    [Fact]
    public void GoldenFile_SuspectCases_MatchAdjudicatedList()
        => Assert.Empty(Data.Cases.Where(c => c.Suspect && c.Op == "exposure").Select(c => c.Name));

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Compensator_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = Case(name);
        Point[] corners = Corners(c);
        int n = corners.Length;
        var images = new Mat<byte>[n];
        var masks = new Mat<byte>[n];
        for (int i = 0; i < n; i++)
        {
            images[i] = Input<byte>(c, 2 * i);
            masks[i] = Input<byte>(c, (2 * i) + 1);
        }

        int nrFeeds = c.Params["nrFeeds"].GetInt32();
        double threshold = DBits(c.Params["threshold"]);
        string kind = c.Params["kind"].GetString()!;
        ExposureCompensator comp = kind switch
        {
            "gain" => new GainCompensator(nrFeeds) { SimilarityThreshold = threshold },
            "channels" => new ChannelsCompensator(nrFeeds) { SimilarityThreshold = threshold },
            "blocksgain" => new BlocksGainCompensator(c.Params["bw"].GetInt32(), c.Params["bh"].GetInt32(), nrFeeds)
            {
                SimilarityThreshold = threshold,
                NrGainsFilteringIterations = c.Params["iters"].GetInt32(),
            },
            _ => new BlocksChannelsCompensator(c.Params["bw"].GetInt32(), c.Params["bh"].GetInt32(), nrFeeds)
            {
                SimilarityThreshold = threshold,
                NrGainsFilteringIterations = c.Params["iters"].GetInt32(),
            },
        };

        byte[][] before = images.Select(StitchGolden.Bytes).ToArray();
        comp.Feed(corners, images, masks);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(before[i], StitchGolden.Bytes(images[i]));
        }

        int outIndex = 0;
        if (comp is BlocksCompensator blocks)
        {
            Mat<float>[] maps = blocks.GainMaps;
            for (int i = 0; i < n; i++)
            {
                AssertOutput(c, outIndex++, maps[i]);
            }
        }
        else
        {
            ulong[][] expected = c.Scalars["gains"].EnumerateArray().Select(g => g.EnumerateArray().Select(e => e.GetUInt64()).ToArray()).ToArray();
            for (int i = 0; i < n; i++)
            {
                double[] actual = comp is GainCompensator g
                    ? [g.Gains[i]]
                    : [((ChannelsCompensator)comp).Gains[i].V0, ((ChannelsCompensator)comp).Gains[i].V1, ((ChannelsCompensator)comp).Gains[i].V2, 0];
                Assert.Equal(expected[i], actual.Select(BitConverter.DoubleToUInt64Bits).ToArray());
            }
        }

        for (int i = 0; i < n; i++)
        {
            Mat<byte> img = images[i].Clone();
            comp.Apply(i, corners[i], img, masks[i]);
            AssertOutput(c, outIndex++, img);
        }
    }
}
