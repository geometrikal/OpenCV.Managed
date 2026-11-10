namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Interpolation method, mirroring the interpolation values of OpenCV <c>InterpolationFlags</c>.</summary>
public enum InterpolationFlags
{
    /// <summary>Nearest neighbour (<c>INTER_NEAREST</c>).</summary>
    Nearest = 0,

    /// <summary>Bilinear (<c>INTER_LINEAR</c>).</summary>
    Linear = 1,

    /// <summary>Bicubic over a 4×4 neighbourhood (<c>INTER_CUBIC</c>).</summary>
    Cubic = 2,

    /// <summary>
    /// Pixel-area relation for decimation; bilinear weights with area coordinates for
    /// enlargement (<c>INTER_AREA</c>). The warps and remap use <see cref="Linear"/> instead.
    /// </summary>
    Area = 3,

    /// <summary>Lanczos over an 8×8 neighbourhood (<c>INTER_LANCZOS4</c>).</summary>
    Lanczos4 = 4,

    /// <summary>Bit-exact bilinear in fixed point (<c>INTER_LINEAR_EXACT</c>). Resize only.</summary>
    LinearExact = 5,

    /// <summary>Bit-exact nearest neighbour (<c>INTER_NEAREST_EXACT</c>). Resize only.</summary>
    NearestExact = 6,
}
