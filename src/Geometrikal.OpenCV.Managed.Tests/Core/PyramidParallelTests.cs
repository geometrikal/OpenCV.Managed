namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Tests.ImgProc;

/// <summary>
/// Row-stripe multithreading of <c>pyrDown</c>, <c>pyrUp</c> and <c>buildPyramid</c> must not
/// change any output element. The images split into several stripes.
/// </summary>
[Collection(nameof(CvParallelCollection))]
public sealed class PyramidParallelTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (string type in new[] { "u8", "u16", "i16", "f32", "f64" })
        {
            foreach (int cn in new[] { 1, 3 })
            {
                foreach (string op in new[] { "down", "up", "build" })
                {
                    yield return [type, cn, op];
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Output_DoesNotDependOnThreadCount(string type, int cn, string op)
    {
        Func<byte[]> run = type switch
        {
            "u8" => Runner<byte>(cn, op),
            "u16" => Runner<ushort>(cn, op),
            "i16" => Runner<short>(cn, op),
            "f32" => Runner<float>(cn, op),
            _ => Runner<double>(cn, op),
        };

        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            byte[] serial = run();
            CvParallel.MaxDegreeOfParallelism = 8;
            byte[] parallel = run();
            Assert.Equal(serial, parallel);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void LargeImage_Output_DoesNotDependOnThreadCount(int cn)
    {
        Mat<byte> src = PyramidGoldenTests.HashInput<byte>(2051, 1537, cn, 3);
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            byte[] down1 = PyramidGoldenTests.Bytes(Pyramids.PyrDown(src));
            byte[] up1 = PyramidGoldenTests.Bytes(Pyramids.PyrUp(src));
            CvParallel.MaxDegreeOfParallelism = 8;
            Assert.Equal(down1, PyramidGoldenTests.Bytes(Pyramids.PyrDown(src)));
            Assert.Equal(up1, PyramidGoldenTests.Bytes(Pyramids.PyrUp(src)));
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    private static Func<byte[]> Runner<T>(int cn, string op)
        where T : unmanaged
    {
        Mat<T> src = PyramidGoldenTests.HashInput<T>(517, 301, cn, 11);
        return op switch
        {
            "down" => () => PyramidGoldenTests.Bytes(Pyramids.PyrDown(src, new Size(0, 0), BorderType.Reflect)),
            "up" => () => PyramidGoldenTests.Bytes(Pyramids.PyrUp(src, new Size(1033, 603))),
            _ => () => Pyramids.BuildPyramid(src, 3).Skip(1).SelectMany(PyramidGoldenTests.Bytes).ToArray(),
        };
    }
}
