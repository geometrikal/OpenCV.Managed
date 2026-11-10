// This implementation is ported from the OpenCV contour implementation.
//
// Original sources:
//   modules/imgproc/src/contours_common.hpp   (in this repo: src/OpenCV/modules/imgproc/src/contours_common.hpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// The payload attached to each contour-tree node: the geometry accumulated by
/// the scanner for one border. Port of the C++ <c>Contour</c> struct
/// (contours_common.hpp:230-290).
///
/// A body holds either explicit border points (<see cref="Points"/>) or a
/// Freeman chain code (<see cref="Codes"/>), selected by <see cref="IsChain"/>.
/// The C++ <c>BlockStorage</c>-backed containers are replaced with
/// <see cref="List{T}"/> per the port scope.
/// </summary>
internal sealed class ContourBody
{
    /// <summary>Explicit border points (used when <see cref="IsChain"/> is false).</summary>
    public List<Point> Points { get; } = [];

    /// <summary>Freeman chain codes (used when <see cref="IsChain"/> is true).</summary>
    public List<sbyte> Codes { get; } = [];

    /// <summary>Bounding rectangle of the contour (C++ <c>brect</c>).</summary>
    public Rect BoundingRect { get; set; }

    /// <summary>Origin point of the contour (C++ <c>origin</c>).</summary>
    public Point Origin { get; set; }

    /// <summary>True when this contour bounds a hole rather than a filled region.</summary>
    public bool IsHole { get; set; }

    /// <summary>True when the contour is stored as a chain code rather than points.</summary>
    public bool IsChain { get; set; }

    /// <summary>
    /// Number of stored elements: chain codes when <see cref="IsChain"/>,
    /// otherwise points.
    /// </summary>
    public int Count => IsChain ? Codes.Count : Points.Count;

    /// <summary>True when neither points nor codes have been accumulated.</summary>
    public bool IsEmpty => Points.Count == 0 && Codes.Count == 0;

    /// <summary>Appends a border point (C++ <c>addPoint</c>).</summary>
    public void AddPoint(Point pt) => Points.Add(pt);
}
