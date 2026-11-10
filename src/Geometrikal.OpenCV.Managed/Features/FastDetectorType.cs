// This implementation is ported from the OpenCV FAST detector.
//
// Original sources:
//   modules/features/include/opencv2/features.hpp (FastFeatureDetector::DetectorType)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// FAST circle pattern, as OpenCV <c>FastFeatureDetector::DetectorType</c>. The name gives the
/// number of contiguous circle pixels and the circle size.
/// </summary>
public enum FastDetectorType
{
    /// <summary>5 contiguous pixels of an 8-pixel circle of radius 1 (<c>TYPE_5_8</c>).</summary>
    Type5_8 = 0,

    /// <summary>7 contiguous pixels of a 12-pixel circle of radius 2 (<c>TYPE_7_12</c>).</summary>
    Type7_12 = 1,

    /// <summary>9 contiguous pixels of a 16-pixel circle of radius 3 (<c>TYPE_9_16</c>).</summary>
    Type9_16 = 2,
}
