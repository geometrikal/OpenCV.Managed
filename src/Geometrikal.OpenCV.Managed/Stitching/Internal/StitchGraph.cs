// This implementation is ported from the OpenCV 5.0 stitching graph utilities.
//
// Original sources:
//   modules/stitching/include/opencv2/stitching/detail/util.hpp (DisjointSets, GraphEdge, Graph)
//   modules/stitching/include/opencv2/stitching/detail/util_inl.hpp (Graph::walkBreadthFirst)
//   modules/stitching/src/util.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>detail::DisjointSets</c> (union by rank, path compression).</summary>
internal sealed class DisjointSets
{
    private readonly int[] _rank;

    public DisjointSets(int count)
    {
        _rank = new int[count];
        Parent = new int[count];
        Size = new int[count];
        for (int i = 0; i < count; i++)
        {
            Parent[i] = i;
            Size[i] = 1;
        }
    }

    public int[] Parent { get; }

    public int[] Size { get; }

    public int FindSetByElem(int elem)
    {
        int set = elem;
        while (set != Parent[set])
        {
            set = Parent[set];
        }

        while (elem != Parent[elem])
        {
            int next = Parent[elem];
            Parent[elem] = set;
            elem = next;
        }

        return set;
    }

    public int MergeSets(int set1, int set2)
    {
        if (_rank[set1] < _rank[set2])
        {
            Parent[set1] = set2;
            Size[set2] += Size[set1];
            return set2;
        }

        if (_rank[set2] < _rank[set1])
        {
            Parent[set2] = set1;
            Size[set1] += Size[set2];
            return set1;
        }

        Parent[set1] = set2;
        _rank[set2]++;
        Size[set2] += Size[set1];
        return set2;
    }
}

/// <summary>OpenCV <c>detail::GraphEdge</c>.</summary>
internal readonly record struct GraphEdge(int From, int To, float Weight);

/// <summary>Strict order of <c>std::greater&lt;GraphEdge&gt;</c>.</summary>
internal readonly struct GraphEdgeGreater : ILess<GraphEdge>
{
    public bool Less(in GraphEdge a, in GraphEdge b) => a.Weight > b.Weight;
}

/// <summary>OpenCV <c>detail::Graph</c>: one edge list per vertex, in insertion order.</summary>
internal sealed class Graph
{
    private readonly List<GraphEdge>[] _edges;

    public Graph(int vertices)
    {
        _edges = new List<GraphEdge>[vertices];
        for (int i = 0; i < vertices; i++)
        {
            _edges[i] = [];
        }
    }

    public int NumVertices => _edges.Length;

    public void AddEdge(int from, int to, float weight) => _edges[from].Add(new GraphEdge(from, to, weight));

    /// <summary>Breadth-first walk from <paramref name="from"/>; calls <paramref name="body"/> once for each tree edge.</summary>
    public void WalkBreadthFirst(int from, Action<GraphEdge> body)
    {
        var was = new bool[NumVertices];
        var queue = new Queue<int>();
        was[from] = true;
        queue.Enqueue(from);
        while (queue.Count > 0)
        {
            int vertex = queue.Dequeue();
            foreach (GraphEdge edge in _edges[vertex])
            {
                if (!was[edge.To])
                {
                    body(edge);
                    was[edge.To] = true;
                    queue.Enqueue(edge.To);
                }
            }
        }
    }
}
