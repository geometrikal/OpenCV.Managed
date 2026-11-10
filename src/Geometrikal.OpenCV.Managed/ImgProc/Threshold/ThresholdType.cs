// This enumeration mirrors OpenCV's ThresholdTypes.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (ThresholdTypes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Threshold operation and flags, with OpenCV <c>THRESH_*</c> numeric values. The value is one
/// operation (<see cref="Binary"/> to <see cref="ToZeroInv"/>) combined with at most one of
/// <see cref="Otsu"/> and <see cref="Triangle"/>, and optionally <see cref="DryRun"/>.
/// </summary>
[Flags]
public enum ThresholdType
{
    /// <summary><c>dst = src &gt; thresh ? maxValue : 0</c> (<c>THRESH_BINARY</c>).</summary>
    Binary = 0,

    /// <summary><c>dst = src &lt;= thresh ? maxValue : 0</c> (<c>THRESH_BINARY_INV</c>).</summary>
    BinaryInv = 1,

    /// <summary><c>dst = min(src, thresh)</c> (<c>THRESH_TRUNC</c>).</summary>
    Trunc = 2,

    /// <summary><c>dst = src &gt; thresh ? src : 0</c> (<c>THRESH_TOZERO</c>).</summary>
    ToZero = 3,

    /// <summary><c>dst = src &lt;= thresh ? src : 0</c> (<c>THRESH_TOZERO_INV</c>).</summary>
    ToZeroInv = 4,

    /// <summary>Bit mask of the operation field (<c>THRESH_MASK</c>). Not an operation.</summary>
    Mask = 7,

    /// <summary>Computes the threshold with Otsu's method (<c>THRESH_OTSU</c>).</summary>
    Otsu = 8,

    /// <summary>Computes the threshold with the triangle method (<c>THRESH_TRIANGLE</c>).</summary>
    Triangle = 16,

    /// <summary>
    /// Computes and returns the threshold without writing the destination
    /// (<c>THRESH_DRYRUN</c>, OpenCV 5.0).
    /// </summary>
    DryRun = 128,
}
