// This implementation is ported from the OpenCV ORB detector.
//
// Original sources:
//   modules/features/include/opencv2/features.hpp (ORB::ScoreType)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features;

/// <summary>
/// Keypoint ranking score of <see cref="Orb"/>, as OpenCV <c>ORB::ScoreType</c>.
/// </summary>
public enum OrbScoreType
{
    /// <summary>Harris corner response on a 7x7 block (<c>HARRIS_SCORE</c>).</summary>
    Harris = 0,

    /// <summary>FAST corner score (<c>FAST_SCORE</c>).</summary>
    Fast = 1,
}
