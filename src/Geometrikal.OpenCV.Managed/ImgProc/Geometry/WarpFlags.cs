namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Warp and remap options, mirroring the flag values of OpenCV <c>InterpolationFlags</c>.</summary>
[Flags]
public enum WarpFlags
{
    /// <summary>No option.</summary>
    None = 0,

    /// <summary>
    /// Pixels that map outside the source are set to zero (<c>WARP_FILL_OUTLIERS</c>).
    /// <see cref="Geometry.WarpPolar{T}"/> only.
    /// </summary>
    FillOutliers = 8,

    /// <summary>
    /// The transformation matrix maps destination to source coordinates (<c>WARP_INVERSE_MAP</c>).
    /// Without this flag the warps invert the matrix first.
    /// </summary>
    InverseMap = 16,

    /// <summary>
    /// The remap maps hold offsets relative to the destination pixel (<c>WARP_RELATIVE_MAP</c>,
    /// OpenCV 5.0). Remap only.
    /// </summary>
    RelativeMap = 32,
}
