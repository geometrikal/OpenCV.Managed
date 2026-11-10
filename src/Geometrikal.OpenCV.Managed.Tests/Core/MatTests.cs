namespace Geometrikal.OpenCV.Managed.Tests.Core;

using Geometrikal.OpenCV.Managed.Core;

public class MatTests
{
    [Fact]
    public void NewMat_IsContinuous_AndZeroInitialised()
    {
        var mat = new Mat<byte>(width: 4, height: 3);

        Assert.Equal(4, mat.Width);
        Assert.Equal(3, mat.Height);
        Assert.Equal(1, mat.Channels);
        Assert.True(mat.IsContinuous);

        foreach (byte value in mat.ToArray())
        {
            Assert.Equal(0, value);
        }
    }

    [Fact]
    public void Roi_IsNotContinuous_AndSharesStorage()
    {
        var parent = new Mat<byte>(width: 20, height: 20);
        Mat<byte> roi = parent.Roi(x: 3, y: 4, width: 10, height: 8);

        // An ROI into a wider parent keeps the parent stride, so its rows are
        // not tightly packed. This is the invariant every port must respect.
        Assert.Equal(20, roi.Stride);
        Assert.Equal(10, roi.RowLength);
        Assert.False(roi.IsContinuous);

        // Writes through the ROI are visible in the parent.
        roi.At(y: 0, x: 0) = 42;
        Assert.Equal(42, parent.At(y: 4, x: 3));
    }
}
