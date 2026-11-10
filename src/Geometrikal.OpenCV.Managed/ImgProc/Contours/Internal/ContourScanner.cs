// This implementation is ported from the OpenCV contour implementation.
//
// Original sources:
//   modules/imgproc/src/contours_new.cpp      (in this repo: src/OpenCV/modules/imgproc/src/contours_new.cpp)
//   modules/imgproc/src/contours_common.hpp   (in this repo: src/OpenCV/modules/imgproc/src/contours_common.hpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Diagnostics;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Sequential contour scanner: the managed port of the C++ <c>ContourScanner_</c>
/// struct together with the free functions <c>icvTraceContour</c> and
/// <c>icvFetchContourEx</c> (contours_new.cpp:99-625).
///
/// It implements the Suzuki-Abe border-following algorithm over a pre-built,
/// zero-bordered, binarized working image. C++ pointer arithmetic on the
/// <c>schar</c> image is mapped to flat integer indexing into a packed
/// <see cref="sbyte"/> array (<c>index = y * stride + x</c>, with
/// <c>stride == paddedWidth</c>); every <c>schar</c>-to-<c>int</c> promotion in
/// the original (sign extension, right/new flag bits, unsigned wrap-around
/// comparisons) is reproduced exactly.
///
/// Only the <c>schar</c> code paths are ported; the <c>RETR_FLOODFILL</c>
/// (<c>int</c> image), <c>LINK_RUNS</c>, <c>findTRUContours</c> fast path and
/// SIMD variants of the original are out of scope.
/// </summary>
internal sealed class ContourScanner
{
    // contours_new.cpp:29-34 and contours_common.hpp:14-28.
    private const int Mask8Right = unchecked((sbyte)0x80); // -128 once promoted to int
    private const int Mask8New = 0x02;
    private const int Mask8Black = 0x01;
    private const int Mask8LVal = 0x7F;
    private const sbyte MaxSize = 16;
    private const int ChainCode = 0;

    /// <summary>Freeman chain-code direction deltas (contours_common.hpp:16-17).</summary>
    private static readonly Point[] ChainCodeDeltas =
    [
        new(1, 0), new(1, -1), new(0, -1), new(-1, -1),
        new(-1, 0), new(-1, 1), new(0, 1), new(1, 1),
    ];

    private readonly sbyte[] _working;
    private readonly int _paddedWidth;
    private readonly int _paddedHeight;
    private readonly int _stride;
    private readonly RetrievalMode _mode;
    private readonly Point _offset;
    private readonly bool _isSimple;
    private readonly int _approxMethod1; // approximation method used while tracing
    private readonly int _approxMethod2; // final approximation method
    private readonly int[] _ctable;
    private readonly int[] _deltas;

    private Point _pt;   // current scanner position
    private Point _lnbd; // position of the last met contour
    private int _nbd;    // current mark value

    /// <summary>
    /// Creates a scanner over a packed, zero-bordered, binarized working image.
    /// Port of <c>ContourScanner_::create</c> (contours_new.cpp:324-366).
    /// </summary>
    /// <param name="working">
    /// Packed working image of length <paramref name="paddedWidth"/> *
    /// <paramref name="paddedHeight"/>. Border pixels must be 0 and interior
    /// foreground pixels exactly 1. The scanner writes border/label markers into
    /// this buffer as it traces.
    /// </param>
    /// <param name="paddedWidth">Working-image width including the 1px border.</param>
    /// <param name="paddedHeight">Working-image height including the 1px border.</param>
    /// <param name="mode">Retrieval mode.</param>
    /// <param name="method">Approximation method.</param>
    /// <param name="offset">
    /// Coordinate offset added to every emitted point. It must already include
    /// the <c>(-1, -1)</c> compensation for the working-image border.
    /// </param>
    public ContourScanner(
        sbyte[] working,
        int paddedWidth,
        int paddedHeight,
        RetrievalMode mode,
        ContourApproximationMode method,
        Point offset)
    {
        _working = working;
        _paddedWidth = paddedWidth;
        _paddedHeight = paddedHeight;
        _stride = paddedWidth;
        _mode = mode;
        _offset = offset;

        _pt = new Point(1, 1);
        _lnbd = new Point(0, 1);
        _nbd = 2;

        Tree = new ContourTree();
        var rootBody = new ContourBody
        {
            IsHole = true,
            BoundingRect = new Rect(0, 0, paddedWidth, paddedHeight),
        };
        ContourNode root = Tree.NewNode(rootBody);
        Debug.Assert(root.Self == 0, "The root contour node must have index 0.");

        _ctable = new int[128];
        Array.Fill(_ctable, -1);

        _approxMethod2 = _approxMethod1 = (int)method;
        if (method is ContourApproximationMode.Tc89L1 or ContourApproximationMode.Tc89KCos)
        {
            _approxMethod1 = ChainCode;
        }

        _isSimple = mode is RetrievalMode.External or RetrievalMode.List;

        // Precompute getDelta(s, step) for s in [0, 16) — contours_common.hpp:19-24.
        _deltas = new int[16];
        for (int i = 0; i < _deltas.Length; i++)
        {
            Point d = ChainCodeDeltas[i % 8];
            _deltas[i] = d.X + (d.Y * _stride);
        }
    }

    /// <summary>The accumulated contour tree. The root is always node 0.</summary>
    public ContourTree Tree { get; }

    /// <summary>
    /// Advances the raster scan to the next unvisited border, traces it, and
    /// inserts it into the tree. Port of <c>ContourScanner_::findNext</c>
    /// (contours_new.cpp:588-625).
    /// </summary>
    /// <returns><c>true</c> if a contour was found; <c>false</c> at end of image.</returns>
    public bool FindNext()
    {
        int x = _pt.X;
        int y = _pt.Y;
        int width = _paddedWidth - 1;
        int height = _paddedHeight - 1;
        Point lastPos = _lnbd;
        int prev = _working[(y * _stride) + x - 1];

        for (; y < height; y++)
        {
            int p = 0;
            for (; x < width; x++)
            {
                x = FindNextX(x, y, prev, ref p);
                if (x >= width)
                {
                    break;
                }

                if (ContourScan(prev, p, ref lastPos, x, y))
                {
                    _lnbd = lastPos;
                    return true;
                }

                prev = p;
                if ((prev & ~1) != 0)
                {
                    lastPos = lastPos with { X = x };
                }
            }

            lastPos = new Point(0, y + 1);
            x = 1;
            prev = 0;
        }

        return false;
    }

    /// <summary>
    /// Scans a row forward while the pixel value equals <paramref name="prev"/>.
    /// Scalar port of the <c>schar</c> branch of <c>findNextX</c>
    /// (contours_new.cpp:550-586); the SIMD path is intentionally omitted.
    /// </summary>
    private int FindNextX(int x, int y, int prev, ref int p)
    {
        int width = _paddedWidth - 1;
        for (; x < width && (p = _working[(y * _stride) + x]) == prev; x++)
        {
        }

        return x;
    }

    /// <summary>
    /// Classifies the pixel transition at (<paramref name="x"/>,
    /// <paramref name="y"/>), decides the new contour's parent, traces it, and
    /// links it into the tree. Port of the <c>schar</c> path of
    /// <c>ContourScanner_::contourScan</c> (contours_new.cpp:417-511).
    /// </summary>
    private bool ContourScan(int prev, int p, ref Point lastPos, int x, int y)
    {
        bool isHole = false;

        // Not the start of an outer contour (prev == 0 && p == 1)?
        if (!(prev == 0 && p == 1))
        {
            if (p != 0 || prev < 1)
            {
                return false;
            }

            if ((prev & ~1) != 0) // prev & MASK8_FLAGS
            {
                lastPos = lastPos with { X = x - 1 };
            }

            isHole = true;
        }

        if (_mode == RetrievalMode.External &&
            (isHole || _working[(lastPos.Y * _stride) + lastPos.X] > 0))
        {
            return false;
        }

        // Find contour parent.
        int mainParent = -1;
        if (_isSimple ||
            (!isHole && _mode == RetrievalMode.ConnectedComponents) ||
            lastPos.X <= 0)
        {
            mainParent = 0;
        }
        else
        {
            int lval = _working[(lastPos.Y * _stride) + lastPos.X] & Mask8LVal;
            mainParent = FindFirstBoundingContour(lastPos, y, lval, -1);

            // If the current contour and the tentative parent have the same hole
            // flag, the true parent is the tentative parent's parent (or the root).
            ContourNode mainParentElem = Tree.Node(mainParent);
            if (mainParentElem.Body.IsHole == isHole)
            {
                mainParent = mainParentElem.Parent != -1 ? mainParentElem.Parent : 0;
            }

            // The hole flag of the parent must differ from that of the contour.
            Debug.Assert(
                Tree.Node(mainParent).Body.IsHole != isHole,
                "A contour's parent must have the opposite hole flag.");
        }

        lastPos = lastPos with { X = x - (isHole ? 1 : 0) };

        ContourNode newContour = MakeContour(ref _nbd, isHole, x, y);
        if (newContour.Parent == -1)
        {
            Tree.AddChild(mainParent, newContour.Self);
        }

        _pt = new Point(x + 1, y);
        return true;
    }

    /// <summary>
    /// Searches the label table for the innermost already-traced contour that
    /// bounds <paramref name="lastPos"/>. Port of
    /// <c>ContourScanner_::findFirstBoundingContour</c>
    /// (contours_new.cpp:513-548). It deliberately performs no lower-bound
    /// rectangle checks — the original has none.
    /// </summary>
    private int FindFirstBoundingContour(Point lastPos, int y, int lval, int par)
    {
        var endPoint = new Point(lastPos.X, y);
        int res = par;
        int cur = _ctable[lval];
        while (cur != -1)
        {
            ContourNode curElem = Tree.Node(cur);
            Rect brect = curElem.Body.BoundingRect;
            if ((lastPos.X - brect.X) < brect.Width &&
                (lastPos.Y - brect.Y) < brect.Height)
            {
                if (res != -1)
                {
                    ContourNode resElem = Tree.Node(res);
                    if (TraceContour(resElem.Body.Origin, endPoint, resElem.Body.IsHole))
                    {
                        break;
                    }
                }

                res = cur;
            }

            cur = curElem.CTableNext;
        }

        return res;
    }

    /// <summary>
    /// Creates a contour node and traces its border into the body. Port of the
    /// <c>schar</c> path of <c>ContourScanner_::makeContour</c>
    /// (contours_new.cpp:368-415).
    /// </summary>
    private ContourNode MakeContour(ref int nbd, bool isHole, int x, int y)
    {
        bool isChain = _approxMethod1 == ChainCode;
        bool isDirect = _approxMethod1 == (int)ContourApproximationMode.None;

        var startPt = new Point(x - (isHole ? 1 : 0), y);

        var body = new ContourBody
        {
            IsHole = isHole,
            IsChain = isChain,
            Origin = startPt + _offset,
        };
        ContourNode res = Tree.NewNode(body);

        if (_isSimple)
        {
            FetchContourEx(startPt, Mask8New, body, isDirect);
        }
        else
        {
            int lval = nbd;

            // Advance the mark value (contours_new.cpp:396-398).
            nbd = (nbd + 1) & Mask8LVal;
            if (nbd == 0)
            {
                nbd = Mask8Black | Mask8New; // == 3
            }

            FetchContourEx(startPt, lval, body, isDirect);
            body.BoundingRect = body.BoundingRect with
            {
                X = body.BoundingRect.X - _offset.X,
                Y = body.BoundingRect.Y - _offset.Y,
            };
            res.CTableNext = _ctable[lval];
            _ctable[lval] = res.Self;
        }

        Point prevOrigin = body.Origin;
        body.Origin = startPt;
        if (_approxMethod1 != _approxMethod2)
        {
            Debug.Assert(body.IsChain, "Approximation requires a chain-coded body.");
            ChainApproximatorTc89.Approximate(body.Codes, prevOrigin, _approxMethod2, body.Points);
            body.IsChain = false;
        }

        return res;
    }

    /// <summary>
    /// Follows a border to verify that the contour originating at
    /// <paramref name="start"/> reaches <paramref name="end"/>. Port of
    /// <c>icvTraceContour&lt;schar&gt;</c> (contours_new.cpp:99-170).
    /// </summary>
    private bool TraceContour(Point start, Point end, bool isHole)
    {
        int stopIdx = (end.Y * _stride) + end.X;
        int i0 = (start.Y * _stride) + start.X;
        int i1 = 0;
        int i3;
        int i4 = 0;
        int sEnd = isHole ? 0 : 4;

        int s = sEnd;
        do
        {
            s = (s - 1) & 7;
            i1 = i0 + _deltas[s];
        }
        while (_working[i1] == 0 && s != sEnd);

        i3 = i0;

        // Check single-pixel domain.
        if (s != sEnd)
        {
            // Follow border.
            for (;;)
            {
                s = Math.Min(s, 15); // clamp_direction
                while (s < MaxSize - 1)
                {
                    ++s;
                    i4 = i3 + _deltas[s];
                    if (_working[i4] != 0)
                    {
                        break;
                    }
                }

                if (i3 == stopIdx)
                {
                    if ((_working[i3] & Mask8Right) == 0)
                    {
                        // It's the only contour.
                        return true;
                    }

                    // Check whether this is the last contour met during the raster scan.
                    int t = s;
                    while (true)
                    {
                        t = (t - 1) & 7;
                        int i5 = i3 + _deltas[t];
                        if (_working[i5] != 0) // raw check: includes marked pixels
                        {
                            break;
                        }

                        if (t == 0)
                        {
                            return true;
                        }
                    }
                }

                if (i4 == i0 && i3 == i1)
                {
                    break;
                }

                i3 = i4;
                s = (s + 4) & 7;
            }
        }
        else
        {
            return i3 == stopIdx;
        }

        return false;
    }

    /// <summary>
    /// Traces a border, emitting points or chain codes into
    /// <paramref name="contour"/> and computing its bounding rectangle. Port of
    /// <c>icvFetchContourEx&lt;schar&gt;</c> (contours_new.cpp:172-269).
    /// </summary>
    private void FetchContourEx(Point start, int nbd, ContourBody contour, bool isDirect)
    {
        int i0 = (start.Y * _stride) + start.X;
        int i1 = 0;
        int i3;
        int i4 = 0;

        Point pt = contour.Origin;

        // C++ abuses cv::Rect(x, y, w, h) as (minX, minY, maxX, maxY).
        int minX = pt.X;
        int minY = pt.Y;
        int maxX = pt.X;
        int maxY = pt.Y;

        int sEnd = contour.IsHole ? 0 : 4;
        int s = sEnd;
        do
        {
            s = (s - 1) & 7;
            i1 = i0 + _deltas[s];
        }
        while (_working[i1] == 0 && s != sEnd);

        if (s == sEnd)
        {
            _working[i0] = (sbyte)(nbd | Mask8Right);
            if (!contour.IsChain)
            {
                contour.AddPoint(pt);
            }
        }
        else
        {
            i3 = i0;
            int prevS = s ^ 4;

            // Follow border.
            for (;;)
            {
                sEnd = s;
                s = Math.Min(s, 15); // clamp_direction
                while (s < MaxSize - 1)
                {
                    ++s;
                    i4 = i3 + _deltas[s];
                    if (_working[i4] != 0)
                    {
                        break;
                    }
                }

                s &= 7;

                // Check "right" bound.
                if ((uint)(s - 1) < (uint)sEnd)
                {
                    _working[i3] = (sbyte)(nbd | Mask8Right);
                }
                else if (_working[i3] == 1) // isVal (MASK8_BLACK)
                {
                    _working[i3] = (sbyte)nbd;
                }

                if (contour.IsChain)
                {
                    contour.Codes.Add((sbyte)s);
                }
                else if (s != prevS || isDirect)
                {
                    contour.AddPoint(pt);
                }

                if (s != prevS)
                {
                    // Update bounds.
                    if (pt.X < minX)
                    {
                        minX = pt.X;
                    }
                    else if (pt.X > maxX)
                    {
                        maxX = pt.X;
                    }

                    if (pt.Y < minY)
                    {
                        minY = pt.Y;
                    }
                    else if (pt.Y > maxY)
                    {
                        maxY = pt.Y;
                    }
                }

                prevS = s;
                pt += ChainCodeDeltas[s];

                if (i4 == i0 && i3 == i1)
                {
                    break;
                }

                i3 = i4;
                s = (s + 4) & 7;
            }
        }

        contour.BoundingRect = new Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }
}
