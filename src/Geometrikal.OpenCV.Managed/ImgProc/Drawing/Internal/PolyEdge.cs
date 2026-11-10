// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// A single active-edge-table entry for the general (non-convex) polygon fill,
/// ported from OpenCV's <c>PolyEdge</c> struct (drawing.cpp:51-59).
///
/// <para>An edge spans image rows <see cref="Y0"/> (inclusive) up to
/// <see cref="Y1"/> (exclusive of the fill, its lower vertex), carries the
/// current intersection abscissa <see cref="X"/> in
/// <see cref="DrawingConstants.XY_SHIFT"/> fixed-point together with the
/// per-row increment <see cref="Dx"/>, also fixed-point.</para>
///
/// <para>The original walks the active edge list through a <c>PolyEdge* next</c>
/// pointer. Managed code cannot take stable pointers into a growable list, so
/// <see cref="Next"/> holds an <em>index</em> into the working edge array built
/// by <see cref="PolygonPrimitives.FillEdgeCollection{T}"/>; <c>-1</c> denotes
/// the null pointer.</para>
/// </summary>
internal struct PolyEdge
{
    /// <summary>Upper (smaller) row of the edge; the first scanline it covers.</summary>
    public int Y0;

    /// <summary>Lower (larger) row of the edge; the scanline at which it is dropped.</summary>
    public int Y1;

    /// <summary>Current intersection abscissa, in <see cref="DrawingConstants.XY_SHIFT"/> fixed-point.</summary>
    public long X;

    /// <summary>Per-row abscissa increment, in <see cref="DrawingConstants.XY_SHIFT"/> fixed-point.</summary>
    public long Dx;

    /// <summary>
    /// Index of the next edge in the active list, or <c>-1</c> for the null
    /// pointer (the managed replacement for OpenCV's <c>PolyEdge* next</c>).
    /// </summary>
    public int Next;
}
