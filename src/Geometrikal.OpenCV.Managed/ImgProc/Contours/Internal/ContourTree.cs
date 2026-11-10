// This implementation is ported from the OpenCV contour implementation.
//
// Original sources:
//   modules/imgproc/src/contours_common.hpp   (in this repo: src/OpenCV/modules/imgproc/src/contours_common.hpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// A node in the contour tree. Port of the C++ <c>TreeNode&lt;Contour&gt;</c>
/// (contours_common.hpp:31-62).
///
/// Nodes participate in two independent linked lists: the bidirectional
/// sibling list (<see cref="Previous"/> / <see cref="Next"/>) that, together
/// with <see cref="Parent"/> / <see cref="FirstChild"/>, forms the hierarchy;
/// and the unidirectional <see cref="CTableNext"/> chain used by the scanner to
/// group candidate parents by label. All index fields use <c>-1</c> as the
/// "no node" sentinel.
/// </summary>
internal sealed class ContourNode
{
    /// <summary>Creates a node with the given self-index and body.</summary>
    public ContourNode(int self, ContourBody body)
    {
        Self = self;
        Body = body;
    }

    /// <summary>This node's own index within its <see cref="ContourTree"/>.</summary>
    public int Self { get; }

    /// <summary>The contour geometry attached to this node.</summary>
    public ContourBody Body { get; }

    /// <summary>Index of the parent node, or <c>-1</c> if this node is a root/top-level node.</summary>
    public int Parent { get; set; } = -1;

    /// <summary>Index of the first child node, or <c>-1</c> if there are no children.</summary>
    public int FirstChild { get; set; } = -1;

    /// <summary>Index of the previous sibling, or <c>-1</c> if none (C++ <c>prev</c>).</summary>
    public int Previous { get; set; } = -1;

    /// <summary>Index of the next sibling, or <c>-1</c> if none (C++ <c>next</c>).</summary>
    public int Next { get; set; } = -1;

    /// <summary>
    /// Index of the next node in the scanner's label-table chain, or <c>-1</c>
    /// if none (C++ <c>ctable_next</c>). Unrelated to the sibling list.
    /// </summary>
    public int CTableNext { get; set; } = -1;
}

/// <summary>
/// The contour tree that accumulates nodes during scanning. Port of the C++
/// <c>Tree&lt;Contour&gt;</c> (contours_common.hpp:64-144). Backed by a
/// contiguous node list so that a node's <see cref="ContourNode.Self"/> equals
/// its list index. (C++ <c>addSiblingAfter</c> is unused by this port and is
/// deliberately omitted.)
/// </summary>
internal sealed class ContourTree
{
    private readonly List<ContourNode> _nodes = [];

    /// <summary>Number of nodes in the tree (C++ <c>size()</c>).</summary>
    public int Count => _nodes.Count;

    /// <summary>
    /// Appends a new node wrapping <paramref name="body"/> and returns it. The
    /// node's <see cref="ContourNode.Self"/> is its index (C++ <c>newElem</c>).
    /// </summary>
    public ContourNode NewNode(ContourBody body)
    {
        int index = _nodes.Count;
        var node = new ContourNode(index, body);
        _nodes.Add(node);
        return node;
    }

    /// <summary>Returns the node at the given index (C++ <c>elem</c>).</summary>
    public ContourNode Node(int index) => _nodes[index];

    /// <summary>
    /// Walks the sibling <see cref="ContourNode.Next"/> chain from
    /// <paramref name="index"/> and returns the last sibling. A <c>-1</c> input
    /// returns <c>-1</c> (C++ <c>lastSibling</c>).
    /// </summary>
    public int LastSibling(int index)
    {
        if (index != -1)
        {
            while (true)
            {
                ContourNode cur = Node(index);
                if (cur.Next == -1)
                {
                    break;
                }

                index = cur.Next;
            }
        }

        return index;
    }

    /// <summary>
    /// Makes <paramref name="childIndex"/> the first child of
    /// <paramref name="parentIndex"/>, prepending it to the existing child list
    /// (C++ <c>addChild</c>). The prepend semantics are load-bearing: they
    /// determine the newest-sibling-first ordering of the output contours.
    /// </summary>
    public void AddChild(int parentIndex, int childIndex)
    {
        ContourNode parent = Node(parentIndex);
        ContourNode child = Node(childIndex);

        child.Parent = parentIndex;
        if (parent.FirstChild != -1)
        {
            Node(parent.FirstChild).Previous = childIndex;
            child.Next = parent.FirstChild;
        }

        parent.FirstChild = childIndex;
        child.Previous = -1;
    }
}

/// <summary>
/// Depth-first iterator over a <see cref="ContourTree"/>. Port of the C++
/// <c>TreeIterator&lt;Contour&gt;</c> (contours_common.hpp:146-176).
///
/// Children are pushed oldest-to-newest so that the newest sibling pops first,
/// reproducing OpenCV's contour output ordering.
/// </summary>
internal sealed class ContourTreeIterator
{
    private readonly ContourTree _tree;
    private readonly Stack<int> _stack = new();

    /// <summary>
    /// Creates an iterator positioned at the root (index 0). The tree must be
    /// non-empty.
    /// </summary>
    public ContourTreeIterator(ContourTree tree)
    {
        ArgumentNullException.ThrowIfNull(tree);
        if (tree.Count == 0)
        {
            throw new InvalidOperationException("Cannot iterate an empty contour tree.");
        }

        _tree = tree;
        _stack.Push(0);
    }

    /// <summary>True once every node has been returned (the stack is empty).</summary>
    public bool IsDone => _stack.Count == 0;

    /// <summary>
    /// Returns the next node in depth-first order and schedules its children
    /// (C++ <c>getNext_s</c>).
    /// </summary>
    public ContourNode GetNext()
    {
        int idx = _stack.Pop();
        ContourNode res = _tree.Node(idx);
        int cur = _tree.LastSibling(res.FirstChild);
        while (cur != -1)
        {
            _stack.Push(cur);
            cur = _tree.Node(cur).Previous;
        }

        return res;
    }
}
