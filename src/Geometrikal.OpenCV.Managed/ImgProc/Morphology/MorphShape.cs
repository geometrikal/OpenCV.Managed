namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Shape of a structuring element produced by
/// <see cref="Morphology.GetStructuringElement"/>. Values match OpenCV's
/// <c>MorphShapes</c> constants.
/// </summary>
public enum MorphShape
{
    /// <summary>All elements set (<c>MORPH_RECT</c>).</summary>
    Rect = 0,

    /// <summary>The anchor row and column set (<c>MORPH_CROSS</c>).</summary>
    Cross = 1,

    /// <summary>A filled ellipse inscribed in the rectangle (<c>MORPH_ELLIPSE</c>).</summary>
    Ellipse = 2,

    /// <summary>A Manhattan-distance diamond (<c>MORPH_DIAMOND</c>, new in OpenCV 5).</summary>
    Diamond = 3,
}
