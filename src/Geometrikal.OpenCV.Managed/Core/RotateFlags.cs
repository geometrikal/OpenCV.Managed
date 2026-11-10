namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>Rotation for <see cref="MatrixTransforms.Rotate{T}"/>, mirroring OpenCV <c>RotateFlags</c>.</summary>
public enum RotateFlags
{
    /// <summary>Rotate 90 degrees clockwise (<c>ROTATE_90_CLOCKWISE</c>).</summary>
    Rotate90Clockwise = 0,

    /// <summary>Rotate 180 degrees (<c>ROTATE_180</c>).</summary>
    Rotate180 = 1,

    /// <summary>Rotate 90 degrees counterclockwise (<c>ROTATE_90_COUNTERCLOCKWISE</c>).</summary>
    Rotate90Counterclockwise = 2,
}
