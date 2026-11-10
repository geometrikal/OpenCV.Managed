// This implementation is ported from the OpenCV 5.0 stitching match graph functions.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp
//   modules/stitching/src/motion_estimators.cpp (matchesGraphAsString, leaveBiggestComponent,
//   findMaxSpanningTree)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching;

using System.Globalization;
using System.Text;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.Stitching.Internal;

/// <summary>
/// Functions on the image match graph, as the OpenCV <c>detail</c> auxiliary functions.
/// </summary>
public static class MatchesGraph
{
    /// <summary>
    /// Keeps only the images of the biggest connected component of the match graph, as OpenCV
    /// <c>leaveBiggestComponent</c>. Two images are connected when the confidence of the pair is
    /// at least <paramref name="confThreshold"/>.
    /// </summary>
    /// <param name="features">Features of the n images. Replaced by the features of the kept images when images are removed.</param>
    /// <param name="pairwiseMatches">The n × n pairwise matches. Replaced by the m × m matches of the kept images (with new image indices) when images are removed.</param>
    /// <param name="confThreshold">Minimum pair confidence of an edge.</param>
    /// <returns>The original indices of the kept images, in ascending order.</returns>
    public static int[] LeaveBiggestComponent(ref ImageFeatures[] features, ref MatchesInfo[] pairwiseMatches, float confThreshold)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(pairwiseMatches);
        int n = features.Length;
        var comps = new DisjointSets(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (pairwiseMatches[(i * n) + j].Confidence < confThreshold)
                {
                    continue;
                }

                int c1 = comps.FindSetByElem(i);
                int c2 = comps.FindSetByElem(j);
                if (c1 != c2)
                {
                    comps.MergeSets(c1, c2);
                }
            }
        }

        int maxComp = 0;
        for (int i = 1; i < n; i++)
        {
            if (comps.Size[i] > comps.Size[maxComp])
            {
                maxComp = i;
            }
        }

        var indices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (comps.FindSetByElem(i) == maxComp)
            {
                indices.Add(i);
            }
        }

        if (indices.Count == n)
        {
            return indices.ToArray();
        }

        int m = indices.Count;
        var featuresSubset = new ImageFeatures[m];
        var matchesSubset = new MatchesInfo[m * m];
        for (int i = 0; i < m; i++)
        {
            featuresSubset[i] = features[indices[i]];
            for (int j = 0; j < m; j++)
            {
                MatchesInfo copy = pairwiseMatches[(indices[i] * n) + indices[j]].Clone();
                copy.SrcImgIdx = i;
                copy.DstImgIdx = j;
                matchesSubset[(i * m) + j] = copy;
            }
        }

        features = featuresSubset;
        pairwiseMatches = matchesSubset;
        return indices.ToArray();
    }

    /// <summary>
    /// Returns the maximum spanning forest of the match graph in the DOT language, as OpenCV
    /// <c>matchesGraphAsString</c>.
    /// </summary>
    /// <param name="paths">Image paths; only the file names are printed.</param>
    /// <param name="pairwiseMatches">The n × n pairwise matches.</param>
    /// <param name="confThreshold">Minimum pair confidence of an edge.</param>
    /// <returns>The DOT text.</returns>
    public static string MatchesGraphAsString(IReadOnlyList<string> paths, IReadOnlyList<MatchesInfo> pairwiseMatches, float confThreshold)
    {
        ArgumentNullException.ThrowIfNull(paths);
        ArgumentNullException.ThrowIfNull(pairwiseMatches);
        var sb = new StringBuilder("graph matches_graph{\n");
        int n = paths.Count;
        var edges = new SortedSet<(int, int)>();
        var comps = new DisjointSets(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (pairwiseMatches[(i * n) + j].Confidence < confThreshold)
                {
                    continue;
                }

                int c1 = comps.FindSetByElem(i);
                int c2 = comps.FindSetByElem(j);
                if (c1 != c2)
                {
                    comps.MergeSets(c1, c2);
                    edges.Add((i, j));
                }
            }
        }

        foreach ((int from, int to) in edges)
        {
            MatchesInfo info = pairwiseMatches[(from * n) + to];
            sb.Append('"').Append(FileName(paths[from])).Append("\" -- \"").Append(FileName(paths[to])).Append('"')
                .Append("[label=\"Nm=").Append(info.Matches.Count.ToString(CultureInfo.InvariantCulture))
                .Append(", Ni=").Append(info.NumInliers.ToString(CultureInfo.InvariantCulture))
                .Append(", C=").Append(FormatDouble(info.Confidence)).Append("\"];\n");
        }

        for (int i = 0; i < n; i++)
        {
            if (comps.Size[comps.FindSetByElem(i)] == 1)
            {
                sb.Append('"').Append(FileName(paths[i])).Append("\";\n");
            }
        }

        sb.Append('}');
        return sb.ToString();
    }

    /// <summary>
    /// OpenCV <c>findMaxSpanningTree</c>: Kruskal on the edges with a transform (weight: number
    /// of inliers, sorted by MSVC <c>std::sort</c> with <c>std::greater</c>), then the tree
    /// centres (vertices with the smallest maximum distance to a leaf).
    /// </summary>
    internal static Graph FindMaxSpanningTree(int n, IReadOnlyList<MatchesInfo> pairwiseMatches, out List<int> centers)
    {
        var edges = new List<GraphEdge>();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                MatchesInfo m = pairwiseMatches[(i * n) + j];
                if (m.H is null)
                {
                    continue;
                }

                edges.Add(new GraphEdge(i, j, m.NumInliers));
            }
        }

        var comps = new DisjointSets(n);
        var spanTree = new Graph(n);
        var powers = new int[n];
        GraphEdge[] sorted = edges.ToArray();
        MsvcSort.Sort(sorted.AsSpan(), default(GraphEdgeGreater));
        foreach (GraphEdge e in sorted)
        {
            int c1 = comps.FindSetByElem(e.From);
            int c2 = comps.FindSetByElem(e.To);
            if (c1 != c2)
            {
                comps.MergeSets(c1, c2);
                spanTree.AddEdge(e.From, e.To, e.Weight);
                spanTree.AddEdge(e.To, e.From, e.Weight);
                powers[e.From]++;
                powers[e.To]++;
            }
        }

        var maxDists = new int[n];
        var cur = new int[n];
        for (int leaf = 0; leaf < n; leaf++)
        {
            if (powers[leaf] != 1)
            {
                continue;
            }

            Array.Clear(cur);
            spanTree.WalkBreadthFirst(leaf, e => cur[e.To] = cur[e.From] + 1);
            for (int j = 0; j < n; j++)
            {
                maxDists[j] = Math.Max(maxDists[j], cur[j]);
            }
        }

        int minMax = maxDists[0];
        for (int i = 1; i < n; i++)
        {
            minMax = Math.Min(minMax, maxDists[i]);
        }

        centers = [];
        for (int i = 0; i < n; i++)
        {
            if (maxDists[i] == minMax)
            {
                centers.Add(i);
            }
        }

        if (centers.Count is 0 or > 2)
        {
            throw new InvalidOperationException("The match graph is not connected: the spanning tree has more than two centres.");
        }

        return spanTree;
    }

    private static string FileName(string path)
    {
        int p = path.LastIndexOfAny(['/', '\\']);
        return p < 0 ? path : path[(p + 1)..];
    }

    /// <summary>Formats as C++ <c>std::ostream</c> with the default precision (6, general).</summary>
    internal static string FormatDouble(double v)
    {
        string s = v.ToString("G6", CultureInfo.InvariantCulture);
        int e = s.IndexOf('E');
        if (e < 0)
        {
            return s;
        }

        string mant = s[..e];
        string exp = s[(e + 1)..];
        char sign = exp[0] == '-' ? '-' : '+';
        string digits = exp.TrimStart('+', '-');
        if (digits.Length < 2)
        {
            digits = "0" + digits;
        }

        return mant + "e" + sign + digits;
    }
}
