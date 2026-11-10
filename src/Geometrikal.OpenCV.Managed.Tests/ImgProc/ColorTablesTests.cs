namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// The Lab tables must equal OpenCV's tables bit for bit. Expected values are the first 16 hex
/// digits of the SHA-256 of each little-endian table, computed from cv2 4.5.5 reference models.
/// </summary>
public sealed class ColorTablesTests
{
    private static string Hash<T>(T[] values)
        where T : unmanaged
        => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())))[..16];

    [Fact]
    public void LabTables_MatchOpenCvHashes()
    {
        LabTables.Tables t = LabTables.Get;
        Assert.Equal("8bfeace00785402e", Hash(t.SRgbGammaTabB));
        Assert.Equal("8d9ac99d93fa1ed9", Hash(t.LabCbrtTabB));
        Assert.Equal("052b99307cb34480", Hash(t.SRgbInvGammaTabB));
        Assert.Equal("b666b964678fb9f2", Hash(t.LabToYFB));
        Assert.Equal("4068bf48e7c11939", Hash(t.AbToXzB));
        Assert.Equal("cb0ff3b32e1baf35", Hash(t.LabCbrtTab));
        Assert.Equal("bedb2c0948747752", Hash(t.SRgbGammaTab));
        Assert.Equal("2fb7afab11fcb707", Hash(t.SRgbInvGammaTab));
        Assert.Equal("3e9ac0dd633718b1", Hash(t.RgbToLabLut));
        Assert.Equal("38118073c5d471c3", Hash(t.TrilinearWeights));
    }

    [Fact]
    public void LabTables_SpotValues()
    {
        LabTables.Tables t = LabTables.Get;
        Assert.Equal(new ushort[] { 6, 440, 2040 }, new[] { t.SRgbGammaTabB[10], t.SRgbGammaTabB[128], t.SRgbGammaTabB[255] });
        Assert.Equal(new ushort[] { 4520, 11992, 17745, 32768, 32975, 37555 },
            new[] { t.LabCbrtTabB[0], t.LabCbrtTabB[100], t.LabCbrtTabB[324], t.LabCbrtTabB[2040], t.LabCbrtTabB[2079], t.LabCbrtTabB[3071] });
    }

    [Fact]
    public void LabCoefficients_MatchOpenCv()
    {
        Assert.Equal(new[] { 778, 1541, 1777, 296, 2929, 871, 3575, 448, 73 }, LabCoefficients.ForwardInt(0));
        Assert.Equal(new[] { 12615, -6296, -2223, -3773, 7684, 185, 217, -836, 4715 }, LabCoefficients.InverseInt(0));
    }
}
