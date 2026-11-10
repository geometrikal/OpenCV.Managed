namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// A 4-element double-precision colour/value tuple, mirroring OpenCV's
/// <c>cv::Scalar</c>.
///
/// <para>Used by the drawing API to describe a colour independently of the
/// target matrix element type. Unused components default to <c>0</c>. When a
/// scalar is applied to an image, component <c>c</c> is written to channel
/// <c>c</c> after saturating conversion to the matrix element type.</para>
/// </summary>
/// <param name="V0">First component (channel 0).</param>
/// <param name="V1">Second component (channel 1).</param>
/// <param name="V2">Third component (channel 2).</param>
/// <param name="V3">Fourth component (channel 3).</param>
public readonly record struct Scalar(double V0, double V1 = 0, double V2 = 0, double V3 = 0)
{
    /// <summary>
    /// Gets the component for the given channel index.
    /// </summary>
    /// <param name="channel">Channel index in the range 0..3.</param>
    /// <returns>The component value for <paramref name="channel"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="channel"/> is outside 0..3.
    /// </exception>
    public double this[int channel] => channel switch
    {
        0 => V0,
        1 => V1,
        2 => V2,
        3 => V3,
        _ => throw new ArgumentOutOfRangeException(nameof(channel)),
    };

    /// <summary>
    /// Creates a scalar whose four components are all set to
    /// <paramref name="value"/>, mirroring OpenCV's <c>Scalar::all</c>.
    /// </summary>
    /// <param name="value">The value assigned to every component.</param>
    /// <returns>A scalar with all four components equal to <paramref name="value"/>.</returns>
    public static Scalar All(double value) => new(value, value, value, value);
}
