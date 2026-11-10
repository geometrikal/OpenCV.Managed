// This enumeration is based on the OpenCV robust estimation method constants.
//
// Original source:
//   modules/geometry/include/opencv2/geometry/3d.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib;

/// <summary>
/// Robust estimation methods of <see cref="GeometricEstimation"/>. The values are the OpenCV
/// constants.
/// </summary>
public enum RobustEstimationMethod
{
    /// <summary>
    /// Least squares over all points (OpenCV method 0). Only <see cref="GeometricEstimation.FindHomography(ReadOnlySpan{Core.Point2f}, ReadOnlySpan{Core.Point2f}, RobustEstimationMethod, double, int, double)"/>
    /// accepts this value.
    /// </summary>
    None = 0,

    /// <summary>Least median of squares (<c>LMEDS</c>).</summary>
    Lmeds = 4,

    /// <summary>Classic RANSAC (<c>RANSAC</c>).</summary>
    Ransac = 8,

    /// <summary>
    /// PROSAC-based RHO estimator (<c>RHO</c>). Only <see cref="GeometricEstimation.FindHomography(ReadOnlySpan{Core.Point2f}, ReadOnlySpan{Core.Point2f}, RobustEstimationMethod, double, int, double)"/>
    /// accepts this value.
    /// </summary>
    Rho = 16,

    /// <summary>USAC with default settings (<c>USAC_DEFAULT</c>). Not supported.</summary>
    UsacDefault = 32,

    /// <summary>Parallel USAC (<c>USAC_PARALLEL</c>). Not supported.</summary>
    UsacParallel = 33,

    /// <summary>USAC for the fundamental matrix with 8 points (<c>USAC_FM_8PTS</c>). Not supported.</summary>
    UsacFm8Pts = 34,

    /// <summary>USAC with fast settings (<c>USAC_FAST</c>). Not supported.</summary>
    UsacFast = 35,

    /// <summary>USAC with accurate settings (<c>USAC_ACCURATE</c>). Not supported.</summary>
    UsacAccurate = 36,

    /// <summary>USAC with PROSAC sampling (<c>USAC_PROSAC</c>). Not supported.</summary>
    UsacProsac = 37,

    /// <summary>USAC with MAGSAC++ (<c>USAC_MAGSAC</c>). Not supported.</summary>
    UsacMagsac = 38,
}
