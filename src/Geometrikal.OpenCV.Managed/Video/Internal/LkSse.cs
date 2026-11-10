// This implementation is ported from the OpenCV sparse optical flow module.
//
// Original sources:
//   modules/video/src/lkpyramid.cpp (LKTrackerInvoker::operator(), CV_SIMD128 loops)
//   modules/core/include/opencv2/core/hal/intrin_sse.hpp (v_dotprod, v_zip, v_pack)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Video.Internal;

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// The window loops of OpenCV <c>LKTrackerInvoker</c> with SSE2 instructions. Elements in the
/// 8-element blocks <c>[0, nv)</c> of a window row go to 4-lane float accumulators; elements in
/// <c>[nv, n)</c> go to scalar float sums. Each result is the scalar sum plus the reduced lanes.
/// </summary>
internal static class LkSse
{
    public static bool IsSupported => Sse2.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> LoadExpand(ref byte p)
        => Sse2.UnpackLow(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(ref p)).AsByte(), Vector128<byte>.Zero).AsInt16();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> Load(ref short p) => Unsafe.ReadUnaligned<Vector128<short>>(ref Unsafe.As<short, byte>(ref p));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> Interp(Vector128<short> v00, Vector128<short> v01, Vector128<short> v10, Vector128<short> v11, Vector128<short> qw0, Vector128<short> qw1, Vector128<int> delta, [ConstantExpected] byte shift)
    {
        Vector128<int> t0 = Sse2.Add(Sse2.Add(Sse2.MultiplyAddAdjacent(Sse2.UnpackLow(v00, v01), qw0), delta), Sse2.MultiplyAddAdjacent(Sse2.UnpackLow(v10, v11), qw1));
        Vector128<int> t1 = Sse2.Add(Sse2.Add(Sse2.MultiplyAddAdjacent(Sse2.UnpackHigh(v00, v01), qw0), delta), Sse2.MultiplyAddAdjacent(Sse2.UnpackHigh(v10, v11), qw1));
        return Sse2.PackSignedSaturate(Sse2.ShiftRightArithmetic(t0, shift), Sse2.ShiftRightArithmetic(t1, shift));
    }

    private static Vector128<short> Weights(int a, int b) => Vector128.Create((short)a, (short)b, (short)a, (short)b, (short)a, (short)b, (short)a, (short)b);

    /// <summary>Patch extraction and gradient matrix sums (A11, A12, A22) for the whole window.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Patch(
        byte[] ib, int s0, int stepI, int cn, short[] db, int d0, int dstep, int cn2, short[] iwin, short[] dwin, int n, int nv, int wh,
        int iw00, int iw01, int iw10, int iw11, out float iA11, out float iA12, out float iA22)
    {
        Vector128<short> qw0 = Weights(iw00, iw01), qw1 = Weights(iw10, iw11);
        Vector128<int> qdelta = Vector128.Create(1 << 8), qdeltaD = Vector128.Create(1 << 13);
        ref byte src = ref MemoryMarshal.GetArrayDataReference(ib);
        ref short dsrc = ref MemoryMarshal.GetArrayDataReference(db);
        ref short iw = ref MemoryMarshal.GetArrayDataReference(iwin);
        ref short dw = ref MemoryMarshal.GetArrayDataReference(dwin);
        Vector128<float> a11 = Vector128<float>.Zero, a12 = Vector128<float>.Zero, a22 = Vector128<float>.Zero;
        float t11 = 0, t12 = 0, t22 = 0;
        for (int y = 0; y < wh; y++)
        {
            int s = s0 + (y * stepI), d = d0 + (y * dstep), w = y * n;
            for (int x = 0; x < nv; x += 8)
            {
                int q = s + x;
                Vector128<short> ival = Interp(
                    LoadExpand(ref Unsafe.Add(ref src, q)),
                    LoadExpand(ref Unsafe.Add(ref src, q + cn)),
                    LoadExpand(ref Unsafe.Add(ref src, q + stepI)),
                    LoadExpand(ref Unsafe.Add(ref src, q + stepI + cn)),
                    qw0,
                    qw1,
                    qdelta,
                    9);
                Unsafe.WriteUnaligned(ref Unsafe.As<short, byte>(ref Unsafe.Add(ref iw, w + x)), ival);

                for (int half = 0; half < 2; half++)
                {
                    int dd = d + (x * 2) + (half * 8);
                    Vector128<short> v = Interp(
                        Load(ref Unsafe.Add(ref dsrc, dd)),
                        Load(ref Unsafe.Add(ref dsrc, dd + cn2)),
                        Load(ref Unsafe.Add(ref dsrc, dd + dstep)),
                        Load(ref Unsafe.Add(ref dsrc, dd + dstep + cn2)),
                        qw0,
                        qw1,
                        qdeltaD,
                        14);
                    Unsafe.WriteUnaligned(ref Unsafe.As<short, byte>(ref Unsafe.Add(ref dw, ((w + x) * 2) + (half * 8))), v);
                    Vector128<float> fx = Sse2.ConvertToVector128Single(Sse2.ShiftRightArithmetic(Sse2.ShiftLeftLogical(v.AsInt32(), 16), 16));
                    Vector128<float> fy = Sse2.ConvertToVector128Single(Sse2.ShiftRightArithmetic(v.AsInt32(), 16));
                    a22 = Sse.Add(Sse.Multiply(fy, fy), a22);
                    a12 = Sse.Add(Sse.Multiply(fx, fy), a12);
                    a11 = Sse.Add(Sse.Multiply(fx, fx), a11);
                }
            }

            for (int x = nv; x < n; x++)
            {
                int q = s + x, dd = d + (x * 2);
                int ival = ((Unsafe.Add(ref src, q) * iw00) + (Unsafe.Add(ref src, q + cn) * iw01) + 256
                    + (Unsafe.Add(ref src, q + stepI) * iw10) + (Unsafe.Add(ref src, q + stepI + cn) * iw11)) >> 9;
                int ix = ((Unsafe.Add(ref dsrc, dd) * iw00) + (Unsafe.Add(ref dsrc, dd + cn2) * iw01) + (1 << 13)
                    + (Unsafe.Add(ref dsrc, dd + dstep) * iw10) + (Unsafe.Add(ref dsrc, dd + dstep + cn2) * iw11)) >> 14;
                int iy = ((Unsafe.Add(ref dsrc, dd + 1) * iw00) + (Unsafe.Add(ref dsrc, dd + cn2 + 1) * iw01) + (1 << 13)
                    + (Unsafe.Add(ref dsrc, dd + dstep + 1) * iw10) + (Unsafe.Add(ref dsrc, dd + dstep + cn2 + 1) * iw11)) >> 14;
                Unsafe.Add(ref iw, w + x) = (short)ival;
                Unsafe.Add(ref dw, (w + x) * 2) = (short)ix;
                Unsafe.Add(ref dw, ((w + x) * 2) + 1) = (short)iy;
                t11 += ix * ix;
                t12 += ix * iy;
                t22 += iy * iy;
            }
        }

        iA11 = t11 + ((a11.GetElement(0) + a11.GetElement(2)) + (a11.GetElement(1) + a11.GetElement(3)));
        iA12 = t12 + ((a12.GetElement(0) + a12.GetElement(2)) + (a12.GetElement(1) + a12.GetElement(3)));
        iA22 = t22 + ((a22.GetElement(0) + a22.GetElement(2)) + (a22.GetElement(1) + a22.GetElement(3)));
    }

    /// <summary>Mismatch vector sums (OpenCV <c>qb0</c>, <c>qb1</c> and the scalar tail) for the whole window.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Diff(
        byte[] jb, int s0, int stepJ, int cn, short[] iwin, short[] dwin, int n, int nv, int wh,
        int iw00, int iw01, int iw10, int iw11, out float ib1, out float ib2)
    {
        Vector128<short> qw0 = Weights(iw00, iw01), qw1 = Weights(iw10, iw11);
        Vector128<int> qdelta = Vector128.Create(1 << 8);
        ref byte src = ref MemoryMarshal.GetArrayDataReference(jb);
        ref short iw = ref MemoryMarshal.GetArrayDataReference(iwin);
        ref short dw = ref MemoryMarshal.GetArrayDataReference(dwin);
        Vector128<float> b0 = Vector128<float>.Zero, b1 = Vector128<float>.Zero;
        float t1 = 0, t2 = 0;
        for (int y = 0; y < wh; y++)
        {
            int s = s0 + (y * stepJ), w = y * n;
            for (int x = 0; x < nv; x += 8)
            {
                int q = s + x;
                Vector128<short> jv = Interp(
                    LoadExpand(ref Unsafe.Add(ref src, q)),
                    LoadExpand(ref Unsafe.Add(ref src, q + cn)),
                    LoadExpand(ref Unsafe.Add(ref src, q + stepJ)),
                    LoadExpand(ref Unsafe.Add(ref src, q + stepJ + cn)),
                    qw0,
                    qw1,
                    qdelta,
                    9);
                Vector128<short> diff0 = Sse2.Subtract(jv, Load(ref Unsafe.Add(ref iw, w + x)));
                Vector128<short> diff2 = Sse2.UnpackLow(diff0, diff0);
                Vector128<short> diff1 = Sse2.UnpackHigh(diff0, diff0);
                int e = (w + x) * 2;
                Vector128<short> v00 = Load(ref Unsafe.Add(ref dw, e));
                Vector128<short> v01 = Load(ref Unsafe.Add(ref dw, e + 8));
                Vector128<short> v10 = Sse2.UnpackLow(v00, v01);
                Vector128<short> v11 = Sse2.UnpackHigh(v00, v01);
                b0 = Sse.Add(b0, Sse2.ConvertToVector128Single(Sse2.MultiplyAddAdjacent(Sse2.UnpackLow(diff2, diff1), v10)));
                b1 = Sse.Add(b1, Sse2.ConvertToVector128Single(Sse2.MultiplyAddAdjacent(Sse2.UnpackHigh(diff2, diff1), v11)));
            }

            for (int x = nv; x < n; x++)
            {
                int q = s + x;
                int diff = (((Unsafe.Add(ref src, q) * iw00) + (Unsafe.Add(ref src, q + cn) * iw01)
                    + (Unsafe.Add(ref src, q + stepJ) * iw10) + (Unsafe.Add(ref src, q + stepJ + cn) * iw11) + 256) >> 9)
                    - Unsafe.Add(ref iw, w + x);
                t1 += diff * Unsafe.Add(ref dw, (w + x) * 2);
                t2 += diff * Unsafe.Add(ref dw, ((w + x) * 2) + 1);
            }
        }

        // Lanes 0 and 2 hold Ix products, lanes 1 and 3 hold Iy products.
        Vector128<float> sum = Sse.Add(b0, b1);
        ib1 = t1 + (sum.GetElement(0) + sum.GetElement(2));
        ib2 = t2 + (sum.GetElement(1) + sum.GetElement(3));
    }
}
