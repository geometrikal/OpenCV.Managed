// This implementation is ported from the OpenCV core data types.
//
// Original sources:
//   modules/core/include/opencv2/core/types.hpp (TermCriteria)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>Termination criteria kinds, as OpenCV <c>TermCriteria::Type</c> (same values).</summary>
[Flags]
public enum TermCriteriaType
{
    /// <summary>Stop after <see cref="TermCriteria.MaxCount"/> iterations.</summary>
    MaxIter = 1,

    /// <summary>Stop when the change is at most <see cref="TermCriteria.Epsilon"/>.</summary>
    Eps = 2,
}

/// <summary>
/// Termination criteria for iterative algorithms, as OpenCV <c>cv::TermCriteria</c>.
/// </summary>
/// <param name="Type">Active criteria.</param>
/// <param name="MaxCount">Maximum number of iterations.</param>
/// <param name="Epsilon">Required accuracy.</param>
public readonly record struct TermCriteria(TermCriteriaType Type, int MaxCount, double Epsilon);
