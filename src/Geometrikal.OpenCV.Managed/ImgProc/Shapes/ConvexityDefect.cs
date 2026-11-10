namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// One convexity defect, as the <c>Vec4i</c> elements of OpenCV <c>convexityDefects</c>.
/// </summary>
/// <param name="StartIndex">Contour index of the hull vertex where the defect starts.</param>
/// <param name="EndIndex">Contour index of the hull vertex where the defect ends.</param>
/// <param name="FarthestIndex">Contour index of the defect point farthest from the hull edge.</param>
/// <param name="FixedPointDepth">
/// Distance from the farthest point to the hull edge, multiplied by 256 and rounded to the
/// nearest integer (ties to even).
/// </param>
public readonly record struct ConvexityDefect(int StartIndex, int EndIndex, int FarthestIndex, int FixedPointDepth)
{
    /// <summary>The depth in pixels: <see cref="FixedPointDepth"/> divided by 256.</summary>
    public float Depth => FixedPointDepth / 256.0f;
}
