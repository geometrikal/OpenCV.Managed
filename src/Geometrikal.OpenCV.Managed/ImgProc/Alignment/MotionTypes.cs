namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>Motion models of the ECC alignment, as OpenCV <c>MotionTypes</c> (same values).</summary>
public enum MotionTypes
{
    /// <summary>Translation: 2 parameters. The warp is a 2×3 matrix.</summary>
    Translation = 0,

    /// <summary>Rotation and translation: 3 parameters. The warp is a 2×3 matrix.</summary>
    Euclidean = 1,

    /// <summary>Affine transformation: 6 parameters. The warp is a 2×3 matrix.</summary>
    Affine = 2,

    /// <summary>Homography: 8 parameters. The warp is a 3×3 matrix.</summary>
    Homography = 3,
}
