namespace Geometrikal.OpenCV.Managed.Tests.Photo;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Photo;

public sealed class InpaintBasicTests
{
    [Theory]
    [InlineData(InpaintMethod.Telea)]
    [InlineData(InpaintMethod.NavierStokes)]
    public void Inpaint_FlatImage_FillsHoleWithTheFlatValue(InpaintMethod method)
    {
        var src = new Mat<byte>(30, 20);
        src.SetTo(new Scalar(90));
        var mask = new Mat<byte>(30, 20);
        for (int y = 8; y < 12; y++)
        {
            for (int x = 10; x < 16; x++)
            {
                src.At(y, x) = 0;
                mask.At(y, x) = 255;
            }
        }

        var result = Inpainting.Inpaint(src, mask, 3, method);

        // Telea adds a normalised gradient term of up to sqrt(2), as cv2 (89-92 on this image).
        int tolerance = method == InpaintMethod.Telea ? 2 : 0;
        Assert.All(result.ToArray(), v => Assert.InRange(v, 90 - tolerance, 90 + tolerance));
    }

    [Fact]
    public void Inpaint_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<byte>(50, 40, 3);
        var roi = parent.Roi(4, 3, 31, 23);
        var rng = new Random(3);
        for (int y = 0; y < 23; y++)
        {
            var row = roi.GetRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                row[x] = (byte)rng.Next(256);
            }
        }

        var maskParent = new Mat<byte>(45, 30);
        var mask = maskParent.Roi(2, 2, 31, 23);
        mask.At(10, 12) = 1;
        mask.At(11, 12) = 1;
        mask.At(0, 5) = 1;

        var packed = roi.Clone();
        Assert.Equal(Inpainting.Inpaint(packed, mask.Clone(), 4, InpaintMethod.Telea).ToArray(), Inpainting.Inpaint(roi, mask, 4, InpaintMethod.Telea).ToArray());
        Assert.Equal(Inpainting.Inpaint(packed, mask.Clone(), 2, InpaintMethod.NavierStokes).ToArray(), Inpainting.Inpaint(roi, mask, 2, InpaintMethod.NavierStokes).ToArray());
    }

    [Fact]
    public void Inpaint_InvalidInputs_Throw()
    {
        Assert.Throws<NotSupportedException>(() => Inpainting.Inpaint(new Mat<ushort>(5, 5, 3), new Mat<byte>(5, 5), 3, InpaintMethod.Telea));
        Assert.Throws<NotSupportedException>(() => Inpainting.Inpaint(new Mat<short>(5, 5), new Mat<byte>(5, 5), 3, InpaintMethod.Telea));
        Assert.Throws<ArgumentException>(() => Inpainting.Inpaint(new Mat<byte>(5, 5), new Mat<byte>(6, 5), 3, InpaintMethod.Telea));
        Assert.Throws<ArgumentOutOfRangeException>(() => Inpainting.Inpaint(new Mat<byte>(5, 5), new Mat<byte>(5, 5), 3, (InpaintMethod)7));
    }
}
