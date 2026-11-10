namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>Comparison operation, mirroring OpenCV <c>CmpTypes</c> (same numeric values).</summary>
public enum CmpType
{
    /// <summary><c>a == b</c>.</summary>
    Eq = 0,

    /// <summary><c>a &gt; b</c>.</summary>
    Gt = 1,

    /// <summary><c>a &gt;= b</c>.</summary>
    Ge = 2,

    /// <summary><c>a &lt; b</c>.</summary>
    Lt = 3,

    /// <summary><c>a &lt;= b</c>.</summary>
    Le = 4,

    /// <summary><c>a != b</c>.</summary>
    Ne = 5,
}
