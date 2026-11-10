// This implementation is ported from the OpenCV random number generator.
//
// Original sources:
//   modules/core/include/opencv2/core/operations.hpp (RNG::RNG, RNG::next, RNG::uniform)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

/// <summary>
/// Minimal OpenCV <c>cv::RNG</c> (multiply-with-carry, coefficient 4164903690) for the ORB
/// patterns.
/// </summary>
internal struct OrbRng
{
    private ulong _state;

    /// <summary>OpenCV <c>RNG(uint64)</c>: a zero seed becomes <c>0xffffffff</c>.</summary>
    internal OrbRng(ulong state) => _state = state != 0 ? state : 0xffffffffUL;

    /// <summary>OpenCV <c>RNG::next</c>.</summary>
    internal uint Next()
    {
        _state = ((ulong)(uint)_state * 4164903690UL) + (uint)(_state >> 32);
        return (uint)_state;
    }

    /// <summary>OpenCV <c>RNG::uniform(int a, int b)</c> for <c>a &lt; b</c>: <c>next() % (b − a) + a</c> in unsigned arithmetic.</summary>
    internal int Uniform(int a, int b)
        => a == b ? a : unchecked((int)((Next() % (uint)(b - a)) + (uint)a));
}
