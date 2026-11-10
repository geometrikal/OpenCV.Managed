// This enumeration mirrors OpenCV's DistanceTransformLabelTypes.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (DistanceTransformLabelTypes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Label numbering of the labelled distance transform, with OpenCV <c>DIST_LABEL_*</c> numeric values.</summary>
public enum DistanceTransformLabelTypes
{
    /// <summary>
    /// Each 8-connected component of zero pixels gets one label. The labels start at 1 and follow
    /// the raster order of the first pixel of each component (<c>DIST_LABEL_CCOMP</c>).
    /// </summary>
    ConnectedComponent = 0,

    /// <summary>Each zero pixel gets its own label. The labels start at 1 and follow the raster order (<c>DIST_LABEL_PIXEL</c>).</summary>
    Pixel = 1,
}
