namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>Flags of <see cref="SpectralTransforms.Dft{T}"/>, with the OpenCV <c>DftFlags</c> values.</summary>
[Flags]
public enum DftFlags
{
    /// <summary>Forward transform without scaling.</summary>
    None = 0,

    /// <summary><c>DFT_INVERSE</c>: inverse transform.</summary>
    Inverse = 1,

    /// <summary><c>DFT_SCALE</c>: divides the result by the number of transformed elements.</summary>
    Scale = 2,

    /// <summary><c>DFT_ROWS</c>: transforms each row independently.</summary>
    Rows = 4,

    /// <summary><c>DFT_COMPLEX_OUTPUT</c>: a forward transform of real input gives the full complex spectrum.</summary>
    ComplexOutput = 16,

    /// <summary><c>DFT_REAL_OUTPUT</c>: an inverse transform of complex input gives real output.</summary>
    RealOutput = 32,

    /// <summary><c>DFT_COMPLEX_INPUT</c>: asserts that the input has 2 channels.</summary>
    ComplexInput = 64,
}

/// <summary>Flags of <see cref="SpectralTransforms.Dct{T}"/>, with the OpenCV <c>DCT_*</c> values.</summary>
[Flags]
public enum DctFlags
{
    /// <summary>Forward 1D or 2D transform.</summary>
    None = 0,

    /// <summary><c>DCT_INVERSE</c>: inverse transform.</summary>
    Inverse = 1,

    /// <summary><c>DCT_ROWS</c>: transforms each row independently.</summary>
    Rows = 4,
}
