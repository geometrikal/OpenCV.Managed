namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Shape comparison method for <c>ShapeDescriptors.MatchShapes</c>, mirroring OpenCV
/// <c>ShapeMatchModes</c> (same numeric values). <c>hA</c> and <c>hB</c> are the Hu moments of the
/// two shapes, and <c>mX = sign(hX) * log10(|hX|)</c>. A Hu moment pair contributes only when both
/// absolute values are greater than 1e-5.
/// </summary>
public enum ShapeMatchModes
{
    /// <summary><c>sum(|1 / mB - 1 / mA|)</c>.</summary>
    ContoursMatchI1 = 1,

    /// <summary><c>sum(|mB - mA|)</c>.</summary>
    ContoursMatchI2 = 2,

    /// <summary><c>max(|mA - mB| / |mA|)</c>.</summary>
    ContoursMatchI3 = 3,
}
