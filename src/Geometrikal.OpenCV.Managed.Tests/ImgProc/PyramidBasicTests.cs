namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class PyramidBasicTests
{
    private static readonly int[] Weights = [1, 4, 6, 4, 1];

    [Fact]
    public void PyrDown_DefaultSize_IsHalfRoundedUp()
    {
        Mat<byte> d = Pyramids.PyrDown(new Mat<byte>(7, 4, 3));
        Assert.Equal((4, 2, 3), (d.Width, d.Height, d.Channels));
    }

    [Fact]
    public void PyrUp_DefaultSize_IsDouble()
    {
        Mat<float> d = Pyramids.PyrUp(new Mat<float>(7, 4, 2));
        Assert.Equal((14, 8, 2), (d.Width, d.Height, d.Channels));
    }

    [Fact]
    public void ZeroImage_GivesZero()
    {
        Assert.All(Pyramids.PyrDown(new Mat<short>(9, 5, 2)).ToArray(), v => Assert.Equal(0, v));
        Assert.All(Pyramids.PyrUp(new Mat<double>(3, 5, 1)).ToArray(), v => Assert.Equal(0.0, v));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    [InlineData(37, 11)]
    public void ConstantImage_StaysConstant(int w, int h)
    {
        var b = new Mat<byte>(w, h, 3);
        b.Fill(77);
        Assert.All(Pyramids.PyrDown(b).ToArray(), v => Assert.Equal(77, v));
        Assert.All(Pyramids.PyrUp(b).ToArray(), v => Assert.Equal(77, v));
        var f = new Mat<float>(w, h, 1);
        f.Fill(3.5f);
        Assert.All(Pyramids.PyrDown(f).ToArray(), v => Assert.Equal(3.5f, v));
        Assert.All(Pyramids.PyrUp(f).ToArray(), v => Assert.Equal(3.5f, v));
    }

    /// <summary>
    /// Every width from 1 to 80 against a direct evaluation of the 5×5 kernel. Integer pyrDown is
    /// exact, so the vector body, the tails and the border tables must give the same value.
    /// </summary>
    [Theory]
    [InlineData(1, BorderType.Reflect101)]
    [InlineData(2, BorderType.Reflect)]
    [InlineData(3, BorderType.Replicate)]
    [InlineData(4, BorderType.Wrap)]
    public void PyrDown_Integer_MatchesDirectKernel(int cn, BorderType border)
    {
        var rng = new Random(cn);
        for (int w = 1; w <= 80; w++)
        {
            int h = 1 + (w % 7);
            Mat<byte> b = Random<byte>(rng, w, h, cn, 256);
            Assert.Equal(DirectDown(b, border, v => (byte)v), Pyramids.PyrDown(b, new Size(0, 0), border).ToArray());
            Mat<ushort> u = Random<ushort>(rng, w, h, cn, 65536);
            Assert.Equal(DirectDown(u, border, v => (ushort)v), Pyramids.PyrDown(u, new Size(0, 0), border).ToArray());
        }
    }

    [Fact]
    public void PyrDown_Roi_EqualsPackedCopy()
    {
        var rng = new Random(3);
        Mat<float> parent = Random<float>(rng, 50, 40, 3, 1000);
        Mat<float> roi = parent.Roi(5, 7, 31, 23);
        Assert.Equal(Pyramids.PyrDown(roi.Clone()).ToArray(), Pyramids.PyrDown(roi).ToArray());
        Assert.Equal(Pyramids.PyrUp(roi.Clone()).ToArray(), Pyramids.PyrUp(roi).ToArray());
    }

    [Fact]
    public void Source_IsUnchanged_AndResultIsDeterministic()
    {
        Mat<ushort> src = Random<ushort>(new Random(4), 33, 17, 2, 65536);
        ushort[] before = src.ToArray();
        ushort[] first = Pyramids.PyrDown(src).ToArray();
        Assert.Equal(first, Pyramids.PyrDown(src).ToArray());
        Assert.Equal(Pyramids.PyrUp(src).ToArray(), Pyramids.PyrUp(src).ToArray());
        Assert.Equal(before, src.ToArray());
    }

    [Fact]
    public void DestinationOverloads_MatchAllocatingOverloads()
    {
        Mat<short> src = Random<short>(new Random(5), 21, 14, 3, 65536, -32768);
        var down = new Mat<short>(11, 8, 3);
        Pyramids.PyrDown(src, down, BorderType.Replicate);
        Assert.Equal(Pyramids.PyrDown(src, new Size(11, 8), BorderType.Replicate).ToArray(), down.ToArray());
        var up = new Mat<short>(41, 29, 3);
        Pyramids.PyrUp(src, up);
        Assert.Equal(Pyramids.PyrUp(src, new Size(41, 29)).ToArray(), up.ToArray());
    }

    [Fact]
    public void PyrDown_DestinationSharingSourceArray_UsesOriginalSource()
    {
        var src = new Mat<byte>(1, 1);
        src.Fill(200);
        Pyramids.PyrDown(src, src);
        Assert.Equal(200, src.At(0, 0));
    }

    /// <summary>OpenCV leaves column 2 uninitialized for this size; the port repeats column 1.</summary>
    [Fact]
    public void PyrUp_OneColumnToThreeColumns_RepeatsColumnOne()
    {
        var src = new Mat<byte>(1, 2);
        src.At(0, 0) = 10;
        src.At(1, 0) = 90;
        Mat<byte> d = Pyramids.PyrUp(src, new Size(3, 4));
        for (int y = 0; y < 4; y++)
        {
            Assert.Equal(d.At(y, 1), d.At(y, 2));
        }
    }

    [Fact]
    public void PyrUp_OddHeight_CopiesRowTwoNMinusTwo()
    {
        Mat<float> src = Random<float>(new Random(6), 4, 3, 1, 100);
        Mat<float> d = Pyramids.PyrUp(src, new Size(8, 7));
        Assert.Equal(d.GetReadOnlyRowSpan(4).ToArray(), d.GetReadOnlyRowSpan(6).ToArray());
    }

    [Fact]
    public void BuildPyramid_LevelZeroIsSource_AndLevelsArePyrDown()
    {
        Mat<byte> src = Random<byte>(new Random(7), 45, 30, 1, 256);
        Mat<byte>[] levels = Pyramids.BuildPyramid(src, 3, BorderType.Wrap);
        Assert.Equal(4, levels.Length);
        Assert.Same(src, levels[0]);
        Assert.Equal((6, 4), (levels[3].Width, levels[3].Height));
        Assert.Equal(Pyramids.PyrDown(levels[1], new Size(0, 0), BorderType.Wrap).ToArray(), levels[2].ToArray());
        Assert.Single(Pyramids.BuildPyramid(src, 0));
    }

    [Fact]
    public void InvalidArguments_Throw()
    {
        var b = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentNullException>(() => Pyramids.PyrDown<byte>(null!));
        Assert.Throws<ArgumentNullException>(() => Pyramids.PyrUp<byte>(null!));
        Assert.Throws<NotSupportedException>(() => Pyramids.PyrDown(new Mat<int>(4, 4)));
        Assert.Throws<NotSupportedException>(() => Pyramids.PyrUp(new Mat<sbyte>(4, 4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pyramids.PyrDown(b, new Size(0, 0), BorderType.Constant));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pyramids.PyrDown(b, new Size(0, 0), BorderType.Transparent));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pyramids.PyrUp(b, new Size(0, 0), BorderType.Reflect));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pyramids.BuildPyramid(b, 2, BorderType.Constant));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pyramids.BuildPyramid(b, -1));
        Assert.Throws<ArgumentException>(() => Pyramids.PyrDown(b, new Size(6, 4)));
        Assert.Throws<ArgumentException>(() => Pyramids.PyrDown(b, new Size(2, 4)));
        Assert.Throws<ArgumentException>(() => Pyramids.PyrUp(b, new Size(18, 16)));
        Assert.Throws<ArgumentException>(() => Pyramids.PyrUp(b, new Size(16, 14)));
        Assert.Throws<ArgumentException>(() => Pyramids.PyrDown(b, new Mat<byte>(4, 4, 3)));
        Assert.Throws<ArgumentException>(() => Pyramids.PyrUp(b, new Mat<byte>(16, 16, 2)));
    }

    [Fact]
    public void ValidCustomSizes_AreAccepted()
    {
        var b = new Mat<byte>(8, 8);
        Assert.Equal(5, Pyramids.PyrDown(b, new Size(5, 3)).Width);
        Assert.Equal(15, Pyramids.PyrUp(b, new Size(15, 17)).Width);
        Assert.Equal(16, Pyramids.PyrUp(b, new Size(16, 0)).Width);
    }

    private static T[] DirectDown<T>(Mat<T> src, BorderType border, Func<int, T> cast)
        where T : unmanaged
    {
        int dw = (src.Width + 1) / 2;
        int dh = (src.Height + 1) / 2;
        int cn = src.Channels;
        var result = new T[dw * dh * cn];
        for (int y = 0; y < dh; y++)
        {
            for (int x = 0; x < dw; x++)
            {
                for (int c = 0; c < cn; c++)
                {
                    long s = 0;
                    for (int i = 0; i < 5; i++)
                    {
                        int sy = Borders.BorderInterpolate((2 * y) - 2 + i, src.Height, border);
                        for (int j = 0; j < 5; j++)
                        {
                            int sx = Borders.BorderInterpolate((2 * x) - 2 + j, src.Width, border);
                            s += Weights[i] * Weights[j] * Convert.ToInt64(src.At(sy, sx, c));
                        }
                    }

                    result[(((y * dw) + x) * cn) + c] = cast((int)((s + 128) >> 8));
                }
            }
        }

        return result;
    }

    private static Mat<T> Random<T>(Random rng, int w, int h, int cn, int range, int offset = 0)
        where T : unmanaged
    {
        var m = new Mat<T>(w, h, cn);
        T[] values = Enumerable.Range(0, w * h * cn).Select(_ =>
        {
            int v = rng.Next(range) + offset;
            return typeof(T) == typeof(byte) ? (T)(object)(byte)v
                : typeof(T) == typeof(ushort) ? (T)(object)(ushort)v
                : typeof(T) == typeof(short) ? (T)(object)(short)v
                : typeof(T) == typeof(float) ? (T)(object)(float)(v * 0.37)
                : (T)(object)(v * 0.37);
        }).ToArray();
        values.CopyTo(m.GetSpan());
        return m;
    }
}
