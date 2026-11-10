// This implementation is ported from the OpenCV contour implementation.
//
// Original sources:
//   modules/imgproc/src/contours_common.cpp        (in this repo: src/OpenCV/modules/imgproc/src/contours_common.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Converts a completed contour <see cref="ContourTree"/> into the flat
/// <see cref="ContourResult"/> (contours plus hierarchy), ported from OpenCV's
/// <c>contourTreeToResults</c> in <c>contours_common.cpp</c>.
///
/// <para>The tree is walked twice in depth-first order using
/// <see cref="ContourTreeIterator"/> — once to emit the contour point arrays and
/// build the index remapping, and once to emit the hierarchy. Both passes visit
/// the nodes in the identical order the iterator produces, skipping the synthetic
/// root node (self index 0). OpenCV's <c>-1</c>-based index remapping is
/// reproduced with a mapping array that is shifted by one slot, so a source index
/// of <c>-1</c> maps through slot 0 (which stays <c>-1</c>).</para>
/// </summary>
internal static class ContourResultBuilder
{
    /// <summary>
    /// Builds the flattened contour result from a scanned contour tree.
    /// </summary>
    /// <param name="tree">The contour tree produced by the scanner.</param>
    /// <returns>
    /// A <see cref="ContourResult"/> whose contour and hierarchy lists have equal
    /// length; empty when the tree contains no contours.
    /// </returns>
    public static ContourResult Build(ContourTree tree)
    {
        // Check if there are no results.
        if (tree.Count == 0 ||
            (tree.Node(0).Body.IsEmpty && tree.Node(0).FirstChild == -1))
        {
            return new ContourResult(Array.Empty<Point[]>(), Array.Empty<ContourHierarchy>());
        }

        // Mapping for indexes (original -> resulting), using -1-based indexing:
        // slot (self + 1), so a source index of -1 maps through slot 0.
        int[] indexMapping = new int[tree.Count + 1];
        for (int i = 0; i < indexMapping.Length; i++)
        {
            indexMapping[i] = -1;
        }

        int total = tree.Count - 1;
        var contours = new Point[total][];
        var hierarchy = new ContourHierarchy[total];

        // First pass: emit contour points and record the index remapping.
        {
            int i = 0;
            var it = new ContourTreeIterator(tree);
            while (!it.IsDone)
            {
                ContourNode node = it.GetNext();
                if (node.Self == 0)
                {
                    continue;
                }
                indexMapping[node.Self + 1] = i;
                contours[i] = node.Body.Points.ToArray();
                i++;
            }
        }

        // Second pass: emit the hierarchy in the identical iteration order.
        {
            int i = 0;
            var it = new ContourTreeIterator(tree);
            while (!it.IsDone)
            {
                ContourNode node = it.GetNext();
                if (node.Self == 0)
                {
                    continue;
                }
                hierarchy[i] = new ContourHierarchy(
                    indexMapping[node.Next + 1],
                    indexMapping[node.Previous + 1],
                    indexMapping[node.FirstChild + 1],
                    indexMapping[node.Parent + 1]);
                i++;
            }
        }

        return new ContourResult(contours, hierarchy);
    }
}
