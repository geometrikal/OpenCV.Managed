// This implementation is ported from the OpenCV graph-cut graph.
//
// Original sources:
//   modules/geometry/include/opencv2/geometry/detail/gcgraph.hpp (GCGraph<TWeight>)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Stitching.Internal;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Port of OpenCV <c>GCGraph&lt;float&gt;</c>: a Boykov–Kolmogorov max-flow graph with the
/// OpenCV vertex queue, edge order, orphan stack and float arithmetic.
/// </summary>
/// <remarks>
/// Vertices and edges are structs in pooled arrays, as the OpenCV vectors. The OpenCV vertex
/// pointer <c>next</c> is a vertex index: <see cref="Null"/> is the null pointer and the index
/// <c>vtxCount</c> (one past the last vertex) is the queue stub <c>nilNode</c>. Call
/// <see cref="Dispose"/> to return the arrays.
/// </remarks>
internal sealed class GcGraph : IDisposable
{
    private const int Null = -1;
    private const int Terminal = -1;
    private const int Orphan = -2;

    private readonly int _vtxCapacity;
    private readonly int _edgeCapacity;
    private Vtx[] _vtcs;
    private Edge[] _edges;
    private int _vtxCount;
    private int _edgeCount;
    private float _flow;

    /// <summary>Creates a graph for at most <paramref name="vtxCount"/> vertices and <paramref name="edgeCount"/> edge pairs.</summary>
    public GcGraph(int vtxCount, int edgeCount)
    {
        _vtxCapacity = vtxCount;
        _edgeCapacity = checked((2 * edgeCount) + 2);
        _vtcs = ArrayPool<Vtx>.Shared.Rent(vtxCount + 1);
        _edges = ArrayPool<Edge>.Shared.Rent(_edgeCapacity);
    }

    public int VertexCount => _vtxCount;

    /// <summary>OpenCV <c>addVtx</c>.</summary>
    public int AddVtx()
    {
        if (_vtxCount == _vtxCapacity)
        {
            throw new InvalidOperationException("The graph vertex capacity is exhausted.");
        }

        int v = _vtxCount++;
        _vtcs[v] = new Vtx { Next = Null };
        return v;
    }

    /// <summary>OpenCV <c>addEdges</c>.</summary>
    public void AddEdges(int i, int j, float w, float revw)
    {
        if ((uint)i >= (uint)_vtxCount || (uint)j >= (uint)_vtxCount || i == j || !(w >= 0) || !(revw >= 0))
        {
            throw new ArgumentException("Invalid graph edge.");
        }

        if (_edgeCount == 0)
        {
            _edges[0] = default;
            _edges[1] = default;
            _edgeCount = 2;
        }

        if (_edgeCount + 2 > _edgeCapacity)
        {
            throw new InvalidOperationException("The graph edge capacity is exhausted.");
        }

        int e = _edgeCount;
        ref Vtx vi = ref _vtcs[i];
        ref Vtx vj = ref _vtcs[j];
        _edges[e] = new Edge { Dst = j, Next = vi.First, Weight = w };
        vi.First = e;
        _edges[e + 1] = new Edge { Dst = i, Next = vj.First, Weight = revw };
        vj.First = e + 1;
        _edgeCount = e + 2;
    }

    /// <summary>OpenCV <c>addTermWeights</c>.</summary>
    public void AddTermWeights(int i, float sourceW, float sinkW)
    {
        ref Vtx v = ref _vtcs[i];
        float dw = v.Weight;
        if (dw > 0)
        {
            sourceW += dw;
        }
        else
        {
            sinkW -= dw;
        }

        _flow += (sourceW < sinkW) ? sourceW : sinkW;
        v.Weight = sourceW - sinkW;
    }

    /// <summary>OpenCV <c>inSourceSegment</c>.</summary>
    public bool InSourceSegment(int i) => _vtcs[i].T == 0;

    /// <summary>OpenCV <c>maxFlow</c>.</summary>
    public float MaxFlow()
    {
        if (_vtxCount == 0 || _edgeCount == 0)
        {
            throw new InvalidOperationException("The graph has no vertices or no edges.");
        }

        int n = _vtxCount;
        int nil = n;
        ref Vtx vtx = ref MemoryMarshal.GetArrayDataReference(_vtcs);
        ref Edge edge = ref MemoryMarshal.GetArrayDataReference(_edges);

        int first = nil, last = nil;
        int currTs = 0;
        Unsafe.Add(ref vtx, nil).Next = nil;
        var orphans = new Stack<int>();

        for (int i = 0; i < n; i++)
        {
            ref Vtx v = ref Unsafe.Add(ref vtx, i);
            v.Ts = 0;
            if (v.Weight != 0)
            {
                Unsafe.Add(ref vtx, last).Next = i;
                last = i;
                v.Dist = 1;
                v.Parent = Terminal;
                v.T = v.Weight < 0 ? (byte)1 : (byte)0;
            }
            else
            {
                v.Parent = 0;
            }
        }

        first = Unsafe.Add(ref vtx, first).Next;
        Unsafe.Add(ref vtx, last).Next = nil;
        Unsafe.Add(ref vtx, nil).Next = Null;

        for (;;)
        {
            int e0 = -1, ei = 0;
            float minWeight;

            while (first != nil)
            {
                int vi = first;
                ref Vtx v = ref Unsafe.Add(ref vtx, vi);
                if (v.Parent != 0)
                {
                    int vt = v.T;
                    for (ei = v.First; ei != 0; ei = Unsafe.Add(ref edge, ei).Next)
                    {
                        if (Unsafe.Add(ref edge, ei ^ vt).Weight == 0)
                        {
                            continue;
                        }

                        int ui = Unsafe.Add(ref edge, ei).Dst;
                        ref Vtx u = ref Unsafe.Add(ref vtx, ui);
                        if (u.Parent == 0)
                        {
                            u.T = (byte)vt;
                            u.Parent = ei ^ 1;
                            u.Ts = v.Ts;
                            u.Dist = v.Dist + 1;
                            if (u.Next == Null)
                            {
                                u.Next = nil;
                                Unsafe.Add(ref vtx, last).Next = ui;
                                last = ui;
                            }

                            continue;
                        }

                        if (u.T != vt)
                        {
                            e0 = ei ^ vt;
                            break;
                        }

                        if (u.Dist > v.Dist + 1 && u.Ts <= v.Ts)
                        {
                            u.Parent = ei ^ 1;
                            u.Ts = v.Ts;
                            u.Dist = v.Dist + 1;
                        }
                    }

                    if (e0 > 0)
                    {
                        break;
                    }
                }

                first = v.Next;
                v.Next = Null;
            }

            if (e0 <= 0)
            {
                break;
            }

            minWeight = Unsafe.Add(ref edge, e0).Weight;
            for (int k = 1; k >= 0; k--)
            {
                int vi;
                for (vi = Unsafe.Add(ref edge, e0 ^ k).Dst; ; vi = Unsafe.Add(ref edge, ei).Dst)
                {
                    if ((ei = Unsafe.Add(ref vtx, vi).Parent) < 0)
                    {
                        break;
                    }

                    float weight = Unsafe.Add(ref edge, ei ^ k).Weight;
                    minWeight = minWeight > weight ? weight : minWeight;
                }

                float tw = MathF.Abs(Unsafe.Add(ref vtx, vi).Weight);
                minWeight = minWeight > tw ? tw : minWeight;
            }

            Unsafe.Add(ref edge, e0).Weight -= minWeight;
            Unsafe.Add(ref edge, e0 ^ 1).Weight += minWeight;
            _flow += minWeight;

            for (int k = 1; k >= 0; k--)
            {
                int vi;
                for (vi = Unsafe.Add(ref edge, e0 ^ k).Dst; ; vi = Unsafe.Add(ref edge, ei).Dst)
                {
                    ref Vtx v = ref Unsafe.Add(ref vtx, vi);
                    if ((ei = v.Parent) < 0)
                    {
                        break;
                    }

                    Unsafe.Add(ref edge, ei ^ (k ^ 1)).Weight += minWeight;
                    if ((Unsafe.Add(ref edge, ei ^ k).Weight -= minWeight) == 0)
                    {
                        orphans.Push(vi);
                        v.Parent = Orphan;
                    }
                }

                ref Vtx root = ref Unsafe.Add(ref vtx, vi);
                root.Weight = root.Weight + (minWeight * (1 - (k * 2)));
                if (root.Weight == 0)
                {
                    orphans.Push(vi);
                    root.Parent = Orphan;
                }
            }

            currTs++;
            while (orphans.Count > 0)
            {
                int v2i = orphans.Pop();
                ref Vtx v2 = ref Unsafe.Add(ref vtx, v2i);
                int d, minDist = int.MaxValue;
                e0 = 0;
                int vt = v2.T;

                for (ei = v2.First; ei != 0; ei = Unsafe.Add(ref edge, ei).Next)
                {
                    if (Unsafe.Add(ref edge, ei ^ (vt ^ 1)).Weight == 0)
                    {
                        continue;
                    }

                    int ui = Unsafe.Add(ref edge, ei).Dst;
                    ref Vtx u = ref Unsafe.Add(ref vtx, ui);
                    if (u.T != vt || u.Parent == 0)
                    {
                        continue;
                    }

                    for (d = 0; ;)
                    {
                        if (u.Ts == currTs)
                        {
                            d += u.Dist;
                            break;
                        }

                        int ej = u.Parent;
                        d++;
                        if (ej < 0)
                        {
                            if (ej == Orphan)
                            {
                                d = int.MaxValue - 1;
                            }
                            else
                            {
                                u.Ts = currTs;
                                u.Dist = 1;
                            }

                            break;
                        }

                        u = ref Unsafe.Add(ref vtx, Unsafe.Add(ref edge, ej).Dst);
                    }

                    if (++d < int.MaxValue)
                    {
                        if (d < minDist)
                        {
                            minDist = d;
                            e0 = ei;
                        }

                        for (u = ref Unsafe.Add(ref vtx, Unsafe.Add(ref edge, ei).Dst); u.Ts != currTs;
                             u = ref Unsafe.Add(ref vtx, Unsafe.Add(ref edge, u.Parent).Dst))
                        {
                            u.Ts = currTs;
                            u.Dist = --d;
                        }
                    }
                }

                if ((v2.Parent = e0) > 0)
                {
                    v2.Ts = currTs;
                    v2.Dist = minDist;
                    continue;
                }

                v2.Ts = 0;
                for (ei = v2.First; ei != 0; ei = Unsafe.Add(ref edge, ei).Next)
                {
                    int ui = Unsafe.Add(ref edge, ei).Dst;
                    ref Vtx u = ref Unsafe.Add(ref vtx, ui);
                    int ej = u.Parent;
                    if (u.T != vt || ej == 0)
                    {
                        continue;
                    }

                    if (Unsafe.Add(ref edge, ei ^ (vt ^ 1)).Weight != 0 && u.Next == Null)
                    {
                        u.Next = nil;
                        Unsafe.Add(ref vtx, last).Next = ui;
                        last = ui;
                    }

                    if (ej > 0 && Unsafe.Add(ref edge, ej).Dst == v2i)
                    {
                        orphans.Push(ui);
                        u.Parent = Orphan;
                    }
                }
            }
        }

        return _flow;
    }

    public void Dispose()
    {
        if (_vtcs.Length == 0)
        {
            return;
        }

        ArrayPool<Vtx>.Shared.Return(_vtcs);
        ArrayPool<Edge>.Shared.Return(_edges);
        _vtcs = [];
        _edges = [];
    }

    private struct Vtx
    {
        public int Next;
        public int Parent;
        public int First;
        public int Ts;
        public int Dist;
        public float Weight;
        public byte T;
    }

    private struct Edge
    {
        public int Dst;
        public int Next;
        public float Weight;
    }
}
