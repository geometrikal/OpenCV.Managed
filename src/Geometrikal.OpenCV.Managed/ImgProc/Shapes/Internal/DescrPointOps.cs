namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Coordinate arithmetic of OpenCV <c>Point_&lt;T&gt;</c> templates. Each member evaluates in the
/// coordinate type <c>T</c> (<c>int</c> wraps, <c>float</c> rounds to single precision) and
/// converts the result to <c>double</c>, as the C++ expressions do.
/// </summary>
internal interface IDescrPointOps<TP>
    where TP : unmanaged
{
    /// <summary><c>(double)p.x</c>.</summary>
    static abstract double X(TP p);

    /// <summary><c>(double)p.y</c>.</summary>
    static abstract double Y(TP p);

    /// <summary><c>(float)p.x</c>.</summary>
    static abstract float FX(TP p);

    /// <summary><c>(float)p.y</c>.</summary>
    static abstract float FY(TP p);

    /// <summary><c>a.x - b.x</c> in <c>T</c>.</summary>
    static abstract double Dx(TP a, TP b);

    /// <summary><c>a.y - b.y</c> in <c>T</c>.</summary>
    static abstract double Dy(TP a, TP b);

    /// <summary><c>(a.x - b.x)^2 + (a.y - b.y)^2</c> in <c>T</c>.</summary>
    static abstract double SquaredDistance(TP a, TP b);

    /// <summary><c>(p.x - s.x) * (e.x - p.x) + (p.y - s.y) * (e.y - p.y)</c> in <c>T</c>.</summary>
    static abstract double InnerProduct(TP s, TP p, TP e);

    /// <summary><c>a.x != b.x || a.y != b.y</c>.</summary>
    static abstract bool Differ(TP a, TP b);
}

/// <summary><see cref="IDescrPointOps{TP}"/> for <c>cv::Point</c>.</summary>
internal readonly struct DescrIntPointOps : IDescrPointOps<Point>
{
    public static double X(Point p) => p.X;

    public static double Y(Point p) => p.Y;

    public static float FX(Point p) => p.X;

    public static float FY(Point p) => p.Y;

    public static double Dx(Point a, Point b) => unchecked(a.X - b.X);

    public static double Dy(Point a, Point b) => unchecked(a.Y - b.Y);

    public static double SquaredDistance(Point a, Point b)
    {
        int dx = unchecked(a.X - b.X), dy = unchecked(a.Y - b.Y);
        return unchecked((dx * dx) + (dy * dy));
    }

    public static double InnerProduct(Point s, Point p, Point e)
        => unchecked(((p.X - s.X) * (e.X - p.X)) + ((p.Y - s.Y) * (e.Y - p.Y)));

    public static bool Differ(Point a, Point b) => a.X != b.X || a.Y != b.Y;
}

/// <summary><see cref="IDescrPointOps{TP}"/> for <c>cv::Point2f</c>.</summary>
internal readonly struct DescrFloatPointOps : IDescrPointOps<Point2f>
{
    public static double X(Point2f p) => p.X;

    public static double Y(Point2f p) => p.Y;

    public static float FX(Point2f p) => p.X;

    public static float FY(Point2f p) => p.Y;

    public static double Dx(Point2f a, Point2f b) => (float)(a.X - b.X);

    public static double Dy(Point2f a, Point2f b) => (float)(a.Y - b.Y);

    public static double SquaredDistance(Point2f a, Point2f b)
    {
        float dx = a.X - b.X, dy = a.Y - b.Y;
        return (float)((float)(dx * dx) + (float)(dy * dy));
    }

    public static double InnerProduct(Point2f s, Point2f p, Point2f e)
        => (float)((float)((p.X - s.X) * (e.X - p.X)) + (float)((p.Y - s.Y) * (e.Y - p.Y)));

    public static bool Differ(Point2f a, Point2f b) => a.X != b.X || a.Y != b.Y;
}
