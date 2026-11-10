// This enumeration mirrors OpenCV's ConnectedComponentsTypes.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (ConnectedComponentsTypes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Column of <see cref="ConnectedComponentsStatsResult{TLabel}.Stats"/>, with OpenCV
/// <c>CC_STAT_*</c> numeric values.
/// </summary>
public enum ConnectedComponentsStat
{
    /// <summary>Leftmost x coordinate of the bounding box (<c>CC_STAT_LEFT</c>).</summary>
    Left = 0,

    /// <summary>Topmost y coordinate of the bounding box (<c>CC_STAT_TOP</c>).</summary>
    Top = 1,

    /// <summary>Width of the bounding box (<c>CC_STAT_WIDTH</c>).</summary>
    Width = 2,

    /// <summary>Height of the bounding box (<c>CC_STAT_HEIGHT</c>).</summary>
    Height = 3,

    /// <summary>Number of pixels of the component (<c>CC_STAT_AREA</c>).</summary>
    Area = 4,
}
