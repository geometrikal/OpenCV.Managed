// This enumeration mirrors the OpenCV floodFill flags.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (FloodFillFlags, cv::floodFill)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Flood fill flags with the OpenCV bit layout. Bits 0-7 hold the connectivity (0 or 4 for
/// 4-connectivity, 8 for 8-connectivity). Bits 8-15 hold the mask fill value
/// (<see cref="FloodFilling.MaskFillValue(byte)"/>); 0 selects the value 1. Bits 16 and 17 hold
/// <see cref="FixedRange"/> and <see cref="MaskOnly"/>.
/// </summary>
[Flags]
public enum FloodFillFlags
{
    /// <summary>Four-connected fill. The value 0 has the same effect.</summary>
    Connectivity4 = 4,

    /// <summary>Eight-connected fill. Diagonal neighbours are connected.</summary>
    Connectivity8 = 8,

    /// <summary>
    /// Compares each pixel with the seed pixel (<c>FLOODFILL_FIXED_RANGE</c>). Without this flag,
    /// each pixel is compared with the neighbour that adds it to the region.
    /// </summary>
    FixedRange = 1 << 16,

    /// <summary>Writes the mask only and does not change the image (<c>FLOODFILL_MASK_ONLY</c>).</summary>
    MaskOnly = 1 << 17,
}
