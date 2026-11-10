namespace Geometrikal.OpenCV.Managed.Tests.Photo;

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Photo;

/// <summary>
/// Pins the inpaint golden cases where cv2 4.5.5 and OpenCV 5.0 can differ:
/// <list type="bullet">
/// <item>Commit 2688248df7 (2024): OpenCV 5.0 reads each neighbour value at <c>(k − 1, l − 1)</c>;
/// 4.5.5 read <c>(km, lm)</c>, which is image row or column 1 for neighbours in row or column 0.
/// Cases whose mask lies within the radius of the top or left image edge are suspect.</item>
/// <item>Commit dfb6069ad9 (2024): float Telea outputs no longer add 0.5.</item>
/// </list>
/// The internal 4.5.5 mode of <see cref="Inpainting"/> changes only these two expressions and must
/// reproduce every suspect case exactly.
/// </summary>
public sealed class InpaintOpenCv5DifferenceTests
{
    public static TheoryData<string> Suspects() => new(InpaintGoldenTests.SuspectNames());

    [Theory]
    [MemberData(nameof(Suspects))]
    public void Inpaint_Legacy455Mode_ReproducesCv2(string name)
    {
        var (img, r) = InpaintGoldenTests.Find(name);
        int w = img.GetProperty("width").GetInt32(), h = img.GetProperty("height").GetInt32(), cn = img.GetProperty("channels").GetInt32();
        byte[] data = Convert.FromBase64String(img.GetProperty("data").GetString()!);
        var mask = new Mat<byte>(Convert.FromBase64String(r.GetProperty("mask").GetString()!), w, h);
        var method = r.GetProperty("method").GetString() == "telea" ? InpaintMethod.Telea : InpaintMethod.NavierStokes;
        double radius = r.GetProperty("radius").GetDouble();

        byte[] actual = img.GetProperty("name").GetString() switch
        {
            "u16" => MemoryMarshal.AsBytes(Inpainting.Inpaint(new Mat<ushort>(MemoryMarshal.Cast<byte, ushort>(data).ToArray(), w, h), mask, radius, method, legacy455: true).ToArray().AsSpan()).ToArray(),
            "f32" => MemoryMarshal.AsBytes(Inpainting.Inpaint(new Mat<float>(MemoryMarshal.Cast<byte, float>(data).ToArray(), w, h), mask, radius, method, legacy455: true).ToArray().AsSpan()).ToArray(),
            _ => Inpainting.Inpaint(new Mat<byte>(data, w, h, cn), mask, radius, method, legacy455: true).ToArray(),
        };

        Assert.Equal(Convert.FromBase64String(r.GetProperty("result").GetString()!), actual);
    }
}
