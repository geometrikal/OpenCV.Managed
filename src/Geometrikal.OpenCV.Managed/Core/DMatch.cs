// This implementation is ported from the OpenCV core data types.
//
// Original sources:
//   modules/core/include/opencv2/core/types.hpp (DMatch)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A match between a query descriptor and a train descriptor, as OpenCV <c>cv::DMatch</c>.
/// </summary>
/// <remarks>
/// <para>The parameterless constructor gives the OpenCV defaults: indexes −1 and
/// <see cref="Distance"/> <see cref="float.MaxValue"/>. <c>default(DMatch)</c> has all members
/// zero.</para>
/// <para>Ordering compares <see cref="Distance"/> only, as OpenCV <c>DMatch::operator&lt;</c>.
/// The <c>&lt;</c> and <c>&gt;</c> operators use IEEE comparison (false for NaN).
/// <see cref="CompareTo"/> uses <see cref="float.CompareTo(float)"/>, which orders NaN first, so
/// sorts are well defined. Equality compares all four members.</para>
/// </remarks>
public record struct DMatch : IComparable<DMatch>
{
    /// <summary>Creates an empty match with the OpenCV default values.</summary>
    public DMatch()
        : this(-1, -1, -1, float.MaxValue)
    {
    }

    /// <summary>Creates a match with <see cref="ImgIdx"/> −1.</summary>
    public DMatch(int queryIdx, int trainIdx, float distance)
        : this(queryIdx, trainIdx, -1, distance)
    {
    }

    /// <summary>Creates a match.</summary>
    /// <param name="queryIdx">Query descriptor index.</param>
    /// <param name="trainIdx">Train descriptor index.</param>
    /// <param name="imgIdx">Train image index.</param>
    /// <param name="distance">Distance between the descriptors; smaller is better.</param>
    public DMatch(int queryIdx, int trainIdx, int imgIdx, float distance)
    {
        QueryIdx = queryIdx;
        TrainIdx = trainIdx;
        ImgIdx = imgIdx;
        Distance = distance;
    }

    /// <summary>Query descriptor index.</summary>
    public int QueryIdx { get; set; }

    /// <summary>Train descriptor index.</summary>
    public int TrainIdx { get; set; }

    /// <summary>Train image index.</summary>
    public int ImgIdx { get; set; }

    /// <summary>Distance between the descriptors.</summary>
    public float Distance { get; set; }

    /// <summary>OpenCV <c>DMatch::operator&lt;</c>: <c>left.Distance &lt; right.Distance</c>.</summary>
    public static bool operator <(DMatch left, DMatch right) => left.Distance < right.Distance;

    /// <summary><c>right &lt; left</c>.</summary>
    public static bool operator >(DMatch left, DMatch right) => right.Distance < left.Distance;

    /// <summary><c>!(right &lt; left)</c>.</summary>
    public static bool operator <=(DMatch left, DMatch right) => !(right.Distance < left.Distance);

    /// <summary><c>!(left &lt; right)</c>.</summary>
    public static bool operator >=(DMatch left, DMatch right) => !(left.Distance < right.Distance);

    /// <summary>Compares <see cref="Distance"/> with <see cref="float.CompareTo(float)"/>.</summary>
    public readonly int CompareTo(DMatch other) => Distance.CompareTo(other.Distance);
}
