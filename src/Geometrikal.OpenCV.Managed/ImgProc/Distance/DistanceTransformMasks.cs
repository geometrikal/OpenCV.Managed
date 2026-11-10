// This enumeration mirrors OpenCV's DistanceTransformMasks.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (DistanceTransformMasks)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Distance transform mask size, with OpenCV <c>DIST_MASK_*</c> numeric values.</summary>
public enum DistanceTransformMasks
{
    /// <summary>Exact Euclidean transform (<c>DIST_MASK_PRECISE</c>).</summary>
    Precise = 0,

    /// <summary>3x3 chamfer mask (<c>DIST_MASK_3</c>).</summary>
    Mask3 = 3,

    /// <summary>5x5 chamfer mask (<c>DIST_MASK_5</c>).</summary>
    Mask5 = 5,
}
