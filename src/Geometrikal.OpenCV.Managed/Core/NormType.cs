namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Norm type, mirroring OpenCV <c>NormTypes</c> (same numeric values). The low 3 bits select
/// the norm; <see cref="Relative"/> combines with them, for example <c>L2 | Relative</c>.
/// </summary>
public enum NormType
{
    /// <summary>Maximum absolute value.</summary>
    Inf = 1,

    /// <summary>Sum of absolute values.</summary>
    L1 = 2,

    /// <summary>Square root of the sum of squares.</summary>
    L2 = 4,

    /// <summary>Sum of squares.</summary>
    L2Sqr = 5,

    /// <summary>
    /// Hamming distance: number of different bits. Byte descriptors only. The matrix norm
    /// functions reject it.
    /// </summary>
    Hamming = 6,

    /// <summary>
    /// Hamming distance over 2-bit cells: number of bit pairs that differ. Byte descriptors only.
    /// The matrix norm functions reject it.
    /// </summary>
    Hamming2 = 7,

    /// <summary>
    /// Flag for the two-matrix norm: divides the difference norm by the norm of the second matrix.
    /// </summary>
    Relative = 8,

    /// <summary>
    /// Range normalisation for <see cref="Conversions.Normalize{TSrc, TDst}(Mat{TSrc}, double, double, NormType, Mat{byte})"/>
    /// only; the norm functions reject it.
    /// </summary>
    MinMax = 32,
}
