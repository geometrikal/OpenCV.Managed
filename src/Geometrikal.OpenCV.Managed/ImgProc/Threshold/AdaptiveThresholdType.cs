// This enumeration mirrors OpenCV's AdaptiveThresholdTypes.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (AdaptiveThresholdTypes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Local mean used by adaptive thresholding, with OpenCV numeric values.</summary>
public enum AdaptiveThresholdType
{
    /// <summary>Box mean of the block (<c>ADAPTIVE_THRESH_MEAN_C</c>).</summary>
    MeanC = 0,

    /// <summary>Gaussian-weighted mean of the block (<c>ADAPTIVE_THRESH_GAUSSIAN_C</c>).</summary>
    GaussianC = 1,
}
