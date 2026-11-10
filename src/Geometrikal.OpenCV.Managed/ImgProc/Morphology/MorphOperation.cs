namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Morphological operation performed by
/// <see cref="Morphology.MorphologyEx{T}"/>. Values match OpenCV's
/// <c>MorphTypes</c> constants.
/// </summary>
public enum MorphOperation
{
    /// <summary>Neighbourhood minimum (<c>MORPH_ERODE</c>).</summary>
    Erode = 0,

    /// <summary>Neighbourhood maximum (<c>MORPH_DILATE</c>).</summary>
    Dilate = 1,

    /// <summary>Erosion followed by dilation (<c>MORPH_OPEN</c>).</summary>
    Open = 2,

    /// <summary>Dilation followed by erosion (<c>MORPH_CLOSE</c>).</summary>
    Close = 3,

    /// <summary>Dilation minus erosion (<c>MORPH_GRADIENT</c>).</summary>
    Gradient = 4,

    /// <summary>Source minus opening (<c>MORPH_TOPHAT</c>).</summary>
    TopHat = 5,

    /// <summary>Closing minus source (<c>MORPH_BLACKHAT</c>).</summary>
    BlackHat = 6,

    /// <summary>
    /// Hit-or-miss transform (<c>MORPH_HITMISS</c>). Only supported for
    /// single-channel byte matrices; kernel cells of 1 must hit the
    /// foreground, cells of -1 must hit the background, 0 cells are ignored.
    /// </summary>
    HitMiss = 7,
}
