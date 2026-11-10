// This implementation is ported from the OpenCV core data types.
//
// Original sources:
//   modules/core/include/opencv2/core/types.hpp (KeyPoint)
//   modules/core/src/types.cpp (KeyPoint::convert, KeyPoint::overlap)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A salient point found by a keypoint detector, as OpenCV <c>cv::KeyPoint</c>.
/// </summary>
/// <remarks>
/// The parameterless constructor gives the OpenCV defaults (<see cref="Angle"/> −1,
/// <see cref="ClassId"/> −1). <c>default(KeyPoint)</c> has all members zero.
/// </remarks>
public record struct KeyPoint
{
    /// <summary>Creates a keypoint at the origin with the OpenCV default values.</summary>
    public KeyPoint()
        : this(default(Point2f), 0f)
    {
    }

    /// <summary>Creates a keypoint, as the OpenCV <c>KeyPoint(pt, size, angle, response, octave, class_id)</c> constructor.</summary>
    /// <param name="pt">Coordinates of the keypoint.</param>
    /// <param name="size">Diameter of the meaningful neighbourhood.</param>
    /// <param name="angle">Orientation in degrees, in <c>[0, 360)</c>, or −1 when not applicable.</param>
    /// <param name="response">Detector response; larger is stronger.</param>
    /// <param name="octave">Pyramid octave of the detection.</param>
    /// <param name="classId">Object class, or −1.</param>
    public KeyPoint(Point2f pt, float size, float angle = -1f, float response = 0f, int octave = 0, int classId = -1)
    {
        Pt = pt;
        Size = size;
        Angle = angle;
        Response = response;
        Octave = octave;
        ClassId = classId;
    }

    /// <summary>Creates a keypoint at (<paramref name="x"/>, <paramref name="y"/>).</summary>
    public KeyPoint(float x, float y, float size, float angle = -1f, float response = 0f, int octave = 0, int classId = -1)
        : this(new Point2f(x, y), size, angle, response, octave, classId)
    {
    }

    /// <summary>Coordinates of the keypoint.</summary>
    public Point2f Pt { get; set; }

    /// <summary>Diameter of the meaningful keypoint neighbourhood.</summary>
    public float Size { get; set; }

    /// <summary>Orientation in degrees, or −1 when not applicable.</summary>
    public float Angle { get; set; }

    /// <summary>Detector response used to select the strongest keypoints.</summary>
    public float Response { get; set; }

    /// <summary>Pyramid octave from which the keypoint was extracted.</summary>
    public int Octave { get; set; }

    /// <summary>Object class of the keypoint, or −1.</summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Returns the coordinates of the keypoints, as OpenCV <c>KeyPoint::convert</c>. When
    /// <paramref name="keypointIndexes"/> is not empty, returns the coordinates of the indexed
    /// keypoints in index order.
    /// </summary>
    /// <param name="keypoints">Keypoints.</param>
    /// <param name="keypointIndexes">Indexes into <paramref name="keypoints"/>, or null or empty for all keypoints.</param>
    /// <returns>A new array of points.</returns>
    /// <exception cref="ArgumentException">An index is negative (OpenCV <c>StsBadArg</c>).</exception>
    /// <exception cref="ArgumentOutOfRangeException">An index is not less than the keypoint count.</exception>
    public static Point2f[] Convert(IReadOnlyList<KeyPoint> keypoints, IReadOnlyList<int>? keypointIndexes = null)
    {
        ArgumentNullException.ThrowIfNull(keypoints);
        if (keypointIndexes is null || keypointIndexes.Count == 0)
        {
            var all = new Point2f[keypoints.Count];
            for (int i = 0; i < all.Length; i++)
            {
                all[i] = keypoints[i].Pt;
            }

            return all;
        }

        var points = new Point2f[keypointIndexes.Count];
        for (int i = 0; i < points.Length; i++)
        {
            int idx = keypointIndexes[i];
            if (idx < 0)
            {
                throw new ArgumentException("keypointIndexes has an element < 0.", nameof(keypointIndexes));
            }

            if (idx >= keypoints.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(keypointIndexes), idx, "Keypoint index is out of range.");
            }

            points[i] = keypoints[idx].Pt;
        }

        return points;
    }

    /// <summary>
    /// Creates one keypoint per point with the same size, response, octave and class, and angle
    /// −1, as OpenCV <c>KeyPoint::convert</c>.
    /// </summary>
    /// <param name="points">Keypoint coordinates.</param>
    /// <param name="size">Size of every keypoint.</param>
    /// <param name="response">Response of every keypoint.</param>
    /// <param name="octave">Octave of every keypoint.</param>
    /// <param name="classId">Class of every keypoint.</param>
    /// <returns>A new array of keypoints.</returns>
    public static KeyPoint[] Convert(IReadOnlyList<Point2f> points, float size = 1f, float response = 1f, int octave = 0, int classId = -1)
    {
        ArgumentNullException.ThrowIfNull(points);
        var keypoints = new KeyPoint[points.Count];
        for (int i = 0; i < keypoints.Length; i++)
        {
            keypoints[i] = new KeyPoint(points[i], size, -1f, response, octave, classId);
        }

        return keypoints;
    }

    /// <summary>
    /// Ratio of the intersection area to the union area of the two keypoint circles (diameter
    /// <see cref="Size"/>), as OpenCV <c>KeyPoint::overlap</c>. Returns 0 for disjoint circles
    /// and <c>min(r1², r2²) / max(r1², r2²)</c> when one circle contains the other.
    /// </summary>
    /// <remarks>
    /// The arithmetic is single precision, as in OpenCV; the centre distance is computed in
    /// double precision and rounded to float. Two keypoints of size 0 at the same position give
    /// NaN (0/0), as in OpenCV.
    /// </remarks>
    public static float Overlap(KeyPoint kp1, KeyPoint kp2)
    {
        float a = kp1.Size * 0.5f;
        float b = kp2.Size * 0.5f;
        float a2 = a * a;
        float b2 = b * b;
        float dx = kp1.Pt.X - kp2.Pt.X;
        float dy = kp1.Pt.Y - kp2.Pt.Y;
        float c = (float)Math.Sqrt(((double)dx * dx) + ((double)dy * dy));

        if (StdMin(a, b) + c <= StdMax(a, b))
        {
            return StdMin(a2, b2) / StdMax(a2, b2);
        }

        float ovrl = 0f;
        if (c < a + b)
        {
            float c2 = c * c;
            float cosAlpha = (b2 + c2 - a2) / (kp2.Size * c);
            float cosBeta = (a2 + c2 - b2) / (kp1.Size * c);
            float alpha = MathF.Acos(cosAlpha);
            float beta = MathF.Acos(cosBeta);
            float sinAlpha = MathF.Sin(alpha);
            float sinBeta = MathF.Sin(beta);

            float segmentAreaA = a2 * beta;
            float segmentAreaB = b2 * alpha;
            float triangleAreaA = a2 * sinBeta * cosBeta;
            float triangleAreaB = b2 * sinAlpha * cosAlpha;

            float intersectionArea = segmentAreaA + segmentAreaB - triangleAreaA - triangleAreaB;
            float unionArea = ((a2 + b2) * (float)Math.PI) - intersectionArea;
            ovrl = intersectionArea / unionArea;
        }

        return ovrl;
    }

    private static float StdMin(float a, float b) => b < a ? b : a;

    private static float StdMax(float a, float b) => a < b ? b : a;
}
