// This enumeration mirrors OpenCV's DistanceTypes.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (DistanceTypes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Distance metric, with OpenCV <c>DIST_*</c> numeric values.</summary>
/// <remarks>
/// <see cref="DistanceTransforms"/> computes <see cref="L1"/>, <see cref="L2"/> and <see cref="C"/>
/// with the chamfer masks. The precise transform (<see cref="DistanceTransformMasks.Precise"/>)
/// computes the Euclidean distance for every value. The other values are defined for completeness.
/// </remarks>
public enum DistanceTypes
{
    /// <summary>User-defined distance (<c>DIST_USER</c>).</summary>
    User = -1,

    /// <summary><c>|x1-x2| + |y1-y2|</c> (<c>DIST_L1</c>).</summary>
    L1 = 1,

    /// <summary>Euclidean distance (<c>DIST_L2</c>).</summary>
    L2 = 2,

    /// <summary><c>max(|x1-x2|, |y1-y2|)</c> (<c>DIST_C</c>).</summary>
    C = 3,

    /// <summary><c>DIST_L12</c>.</summary>
    L12 = 4,

    /// <summary><c>DIST_FAIR</c>.</summary>
    Fair = 5,

    /// <summary><c>DIST_WELSCH</c>.</summary>
    Welsch = 6,

    /// <summary><c>DIST_HUBER</c>.</summary>
    Huber = 7,
}
