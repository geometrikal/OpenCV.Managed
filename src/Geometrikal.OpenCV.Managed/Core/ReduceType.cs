namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>Reduction operation, mirroring OpenCV <c>ReduceTypes</c> (same numeric values).</summary>
public enum ReduceType
{
    /// <summary>Sum.</summary>
    Sum = 0,

    /// <summary>Mean.</summary>
    Avg = 1,

    /// <summary>Maximum.</summary>
    Max = 2,

    /// <summary>Minimum.</summary>
    Min = 3,

    /// <summary>Sum of squares.</summary>
    Sum2 = 4,
}
