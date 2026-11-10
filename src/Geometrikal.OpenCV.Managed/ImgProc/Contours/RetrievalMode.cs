namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Contour retrieval mode, mirroring OpenCV's <c>cv::RetrievalModes</c>.
///
/// Determines which contours are returned and how the hierarchy is organised.
/// (<c>RETR_FLOODFILL</c> is deliberately omitted because it requires 32-bit
/// integer images, which this port does not support.)
/// </summary>
public enum RetrievalMode
{
    /// <summary>
    /// Retrieves only the extreme outer contours (OpenCV <c>RETR_EXTERNAL</c>).
    /// All returned contours are top-level; every hierarchy parent is -1.
    /// </summary>
    External = 0,

    /// <summary>
    /// Retrieves all contours without establishing any parent/child
    /// relationships (OpenCV <c>RETR_LIST</c>). Every contour is a sibling.
    /// </summary>
    List = 1,

    /// <summary>
    /// Retrieves all contours and organises them into a two-level hierarchy
    /// (OpenCV <c>RETR_CCOMP</c>): outer boundaries of connected components at
    /// the top level, and the boundaries of any holes inside them at the
    /// second level.
    /// </summary>
    ConnectedComponents = 2,

    /// <summary>
    /// Retrieves all contours and reconstructs the full nesting hierarchy
    /// (OpenCV <c>RETR_TREE</c>).
    /// </summary>
    Tree = 3,
}
