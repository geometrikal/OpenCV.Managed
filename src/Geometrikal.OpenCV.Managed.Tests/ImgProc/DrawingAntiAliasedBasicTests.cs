namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Analytic tests for the anti-aliased drawing paths.</summary>
public sealed class DrawingAntiAliasedBasicTests
{
    private static readonly Scalar White = new(255, 255, 255, 255);

    [Fact]
    public void NonByteImage_LineAa_EqualsLine8()
    {
        var aa = new Mat<ushort>(20, 20, 3);
        var l8 = new Mat<ushort>(20, 20, 3);
        Drawing.Circle(aa, new Point(10, 9), 6, White, 3, LineType.AntiAliased);
        Drawing.Circle(l8, new Point(10, 9), 6, White, 3, LineType.EightConnected);
        Assert.Equal(l8.ToArray(), aa.ToArray());
    }

    [Fact]
    public void RoiTarget_WritesThroughAndKeepsOutsidePixels()
    {
        var parent = new Mat<byte>(30, 30, 3);
        var whole = new Mat<byte>(20, 20, 3);
        Mat<byte> roi = parent.Roi(new Rect(5, 7, 20, 20));
        Drawing.Line(roi, new Point(-3, 2), new Point(25, 17), White, 3, LineType.AntiAliased);
        Drawing.Line(whole, new Point(-3, 2), new Point(25, 17), White, 3, LineType.AntiAliased);

        for (int y = 0; y < 20; y++)
        {
            Assert.Equal(whole.GetRowSpan(y).ToArray(), roi.GetRowSpan(y).ToArray());
        }

        Assert.All(parent.GetRowSpan(0).ToArray(), v => Assert.Equal(0, v));
    }

    [Fact]
    public void OffImageLine_IsNoOp()
    {
        var m = new Mat<byte>(10, 10);
        Drawing.Line(m, new Point(-50, -5), new Point(-2, -40), White, 1, LineType.AntiAliased);
        Assert.All(m.ToArray(), v => Assert.Equal(0, v));
    }

    [Fact]
    public void HorizontalLine_HasNearFullIntensityCentreRow()
    {
        var m = new Mat<byte>(20, 9);
        Drawing.Line(m, new Point(2, 4), new Point(17, 4), White, 1, LineType.AntiAliased);
        Assert.InRange(m.GetRowSpan(4)[10], (byte)200, (byte)255);
        Assert.True(m.GetRowSpan(4)[10] > m.GetRowSpan(3)[10]);
        Assert.Equal(0, m.GetRowSpan(0)[10]);
    }
}
