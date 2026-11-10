// This implementation is ported from OpenCV's random number generator.
//
// Original sources:
//   modules/core/include/opencv2/core/operations.hpp (RNG inline members)
//   modules/core/src/rand.cpp (theRNG, setRNGSeed)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;

/// <summary>
/// OpenCV <c>cv::RNG</c>: a multiply-with-carry generator with the coefficient 4164903690. The
/// sequence and the <c>uniform</c> results are bit-identical to OpenCV.
/// </summary>
internal struct CvRng
{
    private const ulong Coefficient = 4164903690UL;

    [ThreadStatic]
    private static CvRng t_theRng;

    /// <summary>The 64-bit generator state.</summary>
    public ulong State;

    /// <summary>Creates a generator as <c>RNG(state)</c>. State 0 becomes <c>0xffffffff</c>.</summary>
    /// <param name="state">Initial state.</param>
    public CvRng(ulong state) => State = state != 0 ? state : 0xffffffffUL;

    /// <summary>
    /// The per-thread default generator, as OpenCV <c>theRNG()</c>. Its initial state is
    /// <c>0xffffffff</c>.
    /// </summary>
    public static ref CvRng TheRng
    {
        get
        {
            ref CvRng rng = ref t_theRng;
            if (rng.State == 0)
            {
                rng.State = 0xffffffffUL;
            }

            return ref rng;
        }
    }

    /// <summary>Sets the state of <see cref="TheRng"/>, as OpenCV <c>setRNGSeed</c>.</summary>
    /// <param name="seed">Seed. A negative seed is sign-extended to 64 bits.</param>
    public static void SetTheRngSeed(int seed) => t_theRng = new CvRng(unchecked((ulong)(long)seed));

    /// <summary><c>RNG::next</c>: advances the state and returns its low 32 bits.</summary>
    /// <returns>The next 32-bit value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint Next()
    {
        State = unchecked(((ulong)(uint)State * Coefficient) + (State >> 32));
        return (uint)State;
    }

    /// <summary><c>RNG::uniform(int, int)</c>: a value in [a, b), or a when a equals b.</summary>
    /// <param name="a">Inclusive lower bound.</param>
    /// <param name="b">Exclusive upper bound.</param>
    /// <returns>The next value.</returns>
    public int Uniform(int a, int b) => a == b ? a : unchecked((int)((Next() % (uint)(b - a)) + (uint)a));

    /// <summary><c>RNG::uniform(float, float)</c>: <c>(float)rng * (b - a) + a</c>.</summary>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <returns>The next value.</returns>
    public float Uniform(float a, float b) => (NextFloat() * (b - a)) + a;

    /// <summary><c>RNG::uniform(double, double)</c>: <c>(double)rng * (b - a) + a</c>.</summary>
    /// <param name="a">Lower bound.</param>
    /// <param name="b">Upper bound.</param>
    /// <returns>The next value.</returns>
    public double Uniform(double a, double b) => (NextDouble() * (b - a)) + a;

    /// <summary><c>RNG::operator float</c>: a value in [0, 1) from one 32-bit draw.</summary>
    /// <returns>The next value.</returns>
    public float NextFloat() => Next() * 2.3283064365386962890625e-10f;

    /// <summary><c>RNG::operator double</c>: a value in [0, 1) from two 32-bit draws.</summary>
    /// <returns>The next value.</returns>
    public double NextDouble()
    {
        uint t = Next();
        return (((ulong)t << 32) | Next()) * 5.4210108624275221700372640043497e-20;
    }
}
