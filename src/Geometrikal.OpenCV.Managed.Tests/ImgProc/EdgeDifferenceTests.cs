namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins the deliberate difference of the edge port from OpenCV. OpenCV's L1 magnitude loop
/// (canny.cpp:459-473) takes <c>v_abs</c> of int16 lanes and reinterprets the result as int16,
/// so <c>|-32768|</c> is -32768 in its vector body and 32768 in its scalar tail; which elements
/// fall in the body depends on the vector width of the build. The port uses 32768 for every
/// element. The golden battery avoids -32768 in L1 derivative cases.
/// </summary>
public sealed class EdgeDifferenceTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(17)]
    [InlineData(38)]
    public void CannyDerivativesL1_MinShort_UsesMagnitude32768AtEveryColumn(int column)
    {
        var dx = new Mat<short>(40, 1);
        var dy = new Mat<short>(40, 1);
        dx.At(0, column) = short.MinValue;
        Mat<byte> edges = EdgeDetection.Canny(dx, dy, 1000, 30000);
        for (int x = 0; x < 40; x++)
        {
            Assert.Equal(x == column ? 255 : 0, edges.At(0, x));
        }
    }

    [Fact]
    public void CannyDerivativesL1_MinShortInThreeChannels_SelectsThatChannel()
    {
        var dx = new Mat<short>(20, 1, 3);
        var dy = new Mat<short>(20, 1, 3);
        dx.At(0, 9, 0) = 1000;
        dy.At(0, 9, 2) = short.MinValue;
        Mat<byte> edges = EdgeDetection.Canny(dx, dy, 2000, 30000);
        Assert.Equal(255, edges.At(0, 9));
    }
}
