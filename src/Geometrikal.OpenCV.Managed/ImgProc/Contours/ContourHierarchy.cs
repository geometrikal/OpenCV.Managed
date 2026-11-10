namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Hierarchy information for a single contour, mirroring one OpenCV
/// <c>Vec4i</c> row (element order [next, previous, first_child, parent]).
///
/// Each field is an index into the <see cref="ContourResult.Contours"/> list,
/// or <c>-1</c> (the OpenCV sentinel) when the corresponding relationship does
/// not exist.
/// </summary>
/// <param name="Next">
/// Index of the next contour at the same hierarchical level, or <c>-1</c> if
/// there is no next sibling.
/// </param>
/// <param name="Previous">
/// Index of the previous contour at the same hierarchical level, or <c>-1</c>
/// if there is no previous sibling.
/// </param>
/// <param name="FirstChild">
/// Index of the first child (nested) contour, or <c>-1</c> if this contour has
/// no children.
/// </param>
/// <param name="Parent">
/// Index of the parent (enclosing) contour, or <c>-1</c> if this contour is
/// top-level.
/// </param>
public readonly record struct ContourHierarchy(int Next, int Previous, int FirstChild, int Parent);
