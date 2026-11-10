// This implementation is ported from the OpenCV transform-matrix helpers.
//
// Original source:
//   modules/geometry/src/geometry.cpp (getRotationMatrix2D_, getPerspectiveTransform,
//     getAffineTransform, invertAffineTransform)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <content>Transform-matrix helpers. A 2 × 3 matrix is a <see cref="Mat{T}"/> of width 3 and height 2.</content>
public static partial class Geometry
{
    /// <summary>
    /// Returns the affine matrix of a rotation about <paramref name="center"/> with uniform scaling,
    /// as OpenCV <c>getRotationMatrix2D</c>:
    /// <c>[[α, β, (1-α)·cx - β·cy], [-β, α, β·cx + (1-α)·cy]]</c> with <c>α = scale·cos(angle)</c>
    /// and <c>β = scale·sin(angle)</c>.
    /// </summary>
    /// <param name="center">Rotation centre in the source image.</param>
    /// <param name="angle">Angle in degrees. A positive angle rotates counterclockwise (the origin is the top-left corner).</param>
    /// <param name="scale">Isotropic scale factor.</param>
    /// <returns>A new 2 × 3 double matrix.</returns>
    public static Mat<double> GetRotationMatrix2D(Point2f center, double angle, double scale)
    {
        angle *= Math.PI / 180;
        double alpha = Math.Cos(angle) * scale;
        double beta = Math.Sin(angle) * scale;
        double cx = center.X, cy = center.Y;
        return new Mat<double>(
            [alpha, beta, ((1 - alpha) * cx) - (beta * cy), -beta, alpha, (beta * cx) + ((1 - alpha) * cy)], 3, 2);
    }

    /// <summary>
    /// Returns the affine matrix that maps three source points to three destination points, as
    /// OpenCV <c>getAffineTransform</c>. The 6 × 6 system is solved by LU decomposition with partial
    /// pivoting; a singular system gives the zero matrix.
    /// </summary>
    /// <param name="source">Three source points.</param>
    /// <param name="destination">Three destination points.</param>
    /// <returns>A new 2 × 3 double matrix.</returns>
    public static Mat<double> GetAffineTransform(ReadOnlySpan<Point2f> source, ReadOnlySpan<Point2f> destination)
    {
        RequirePoints(source, destination, 3);
        Span<double> a = stackalloc double[36];
        Span<double> b = stackalloc double[6];
        a.Clear();
        for (int i = 0; i < 3; i++)
        {
            int j = i * 12, k = (i * 12) + 6;
            a[j] = a[k + 3] = source[i].X;
            a[j + 1] = a[k + 4] = source[i].Y;
            a[j + 2] = a[k + 5] = 1;
            b[i * 2] = destination[i].X;
            b[(i * 2) + 1] = destination[i].Y;
        }

        if (!MatrixSolvers.LuSolve(a, b, 6))
        {
            b.Clear();
        }

        return new Mat<double>(b.ToArray(), 3, 2);
    }

    /// <summary>
    /// Returns the perspective matrix that maps four source points to four destination points, as
    /// OpenCV 5.0 <c>getPerspectiveTransform</c> with <c>DECOMP_LU</c>.
    /// </summary>
    /// <remarks>
    /// The 8 × 8 system with <c>M[2,2] = 1</c> is solved by LU decomposition. The products
    /// <c>-x·u</c> of the system are computed in single precision, as OpenCV. When the solve fails or
    /// the residual norm is not below <c>1e-8</c>, the method follows OpenCV 5.0: it solves the
    /// homogeneous 9-unknown system by a Jacobi SVD of <c>AᵀA</c> and returns the singular vector of
    /// the smallest singular value, with unit norm and without normalisation of <c>M[2,2]</c>.
    /// OpenCV 4.x instead returns <c>[0, …, 0, 1]</c> for such input.
    /// </remarks>
    /// <param name="source">Four source points.</param>
    /// <param name="destination">Four destination points.</param>
    /// <returns>A new 3 × 3 double matrix.</returns>
    public static Mat<double> GetPerspectiveTransform(ReadOnlySpan<Point2f> source, ReadOnlySpan<Point2f> destination)
    {
        RequirePoints(source, destination, 4);
        Span<double> a = stackalloc double[64];
        Span<double> b = stackalloc double[8];
        a.Clear();
        for (int i = 0; i < 4; i++)
        {
            float x = source[i].X, y = source[i].Y, u = destination[i].X, v = destination[i].Y;
            int r0 = i * 8, r1 = (i + 4) * 8;
            a[r0] = a[r1 + 3] = x;
            a[r0 + 1] = a[r1 + 4] = y;
            a[r0 + 2] = a[r1 + 5] = 1;
            a[r0 + 6] = -x * u;
            a[r0 + 7] = -y * u;
            a[r1 + 6] = -x * v;
            a[r1 + 7] = -y * v;
            b[i] = u;
            b[i + 4] = v;
        }

        Span<double> lu = stackalloc double[64];
        Span<double> x8 = stackalloc double[8];
        a.CopyTo(lu);
        b.CopyTo(x8);
        var m = new double[9];
        if (MatrixSolvers.LuSolve(lu, x8, 8) && ResidualNorm(a, x8, b) < 1e-8)
        {
            x8.CopyTo(m);
            m[8] = 1;
            return new Mat<double>(m, 3, 3);
        }

        Span<double> a9 = stackalloc double[72];
        for (int r = 0; r < 8; r++)
        {
            a.Slice(r * 8, 8).CopyTo(a9.Slice(r * 9, 8));
            a9[(r * 9) + 8] = -b[r];
        }

        Span<double> ata = stackalloc double[81];
        MatrixSolvers.MulTransposed(a9, 8, 9, ata);
        MatrixSolvers.SymmetricSvdVectors(ata, 9);
        ata.Slice(8 * 9, 9).CopyTo(m);
        return new Mat<double>(m, 3, 3);
    }

    /// <summary>
    /// Inverts an affine transformation, as OpenCV <c>invertAffineTransform</c> for a
    /// <c>CV_64F</c> matrix. A singular matrix gives zero linear terms.
    /// </summary>
    /// <param name="matrix">2 × 3 matrix (width 3, height 2, one channel).</param>
    /// <returns>A new 2 × 3 double matrix.</returns>
    public static Mat<double> InvertAffineTransform(Mat<double> matrix)
    {
        double[] mv = ReadAffine(matrix);
        var inverse = new double[6];
        InvertAffineTransform(mv, inverse);
        return new Mat<double>(inverse, 3, 2);
    }

    /// <summary><see cref="InvertAffineTransform(Mat{double})"/> of the row-major values <paramref name="mv"/> into <paramref name="inverse"/>.</summary>
    internal static void InvertAffineTransform(ReadOnlySpan<double> mv, Span<double> inverse)
    {
        double d = (mv[0] * mv[4]) - (mv[1] * mv[3]);
        d = d != 0 ? 1.0 / d : 0.0;
        double a11 = mv[4] * d, a22 = mv[0] * d, a12 = -mv[1] * d, a21 = -mv[3] * d;
        double b1 = (-a11 * mv[2]) - (a12 * mv[5]);
        double b2 = (-a21 * mv[2]) - (a22 * mv[5]);
        inverse[0] = a11;
        inverse[1] = a12;
        inverse[2] = b1;
        inverse[3] = a21;
        inverse[4] = a22;
        inverse[5] = b2;
    }

    /// <summary>
    /// Inverts an affine transformation, as OpenCV <c>invertAffineTransform</c> for a
    /// <c>CV_32F</c> matrix. The determinant and the linear terms use single-precision products, as
    /// the OpenCV softfloat operator overloads do; the translation uses double precision.
    /// </summary>
    /// <param name="matrix">2 × 3 matrix (width 3, height 2, one channel).</param>
    /// <returns>A new 2 × 3 float matrix.</returns>
    public static Mat<float> InvertAffineTransform(Mat<float> matrix)
    {
        float[] mv = ReadAffine(matrix);
        double d = (mv[0] * mv[4]) - (mv[1] * mv[3]);
        d = d != 0 ? 1.0 / d : 0.0;
        float df = (float)d;
        double a11 = mv[4] * df, a22 = mv[0] * df, a12 = -mv[1] * df, a21 = -mv[3] * df;
        double b1 = (-a11 * mv[2]) - (a12 * mv[5]);
        double b2 = (-a21 * mv[2]) - (a22 * mv[5]);
        return new Mat<float>([(float)a11, (float)a12, (float)b1, (float)a21, (float)a22, (float)b2], 3, 2);
    }

    private static double ResidualNorm(ReadOnlySpan<double> a, ReadOnlySpan<double> x, ReadOnlySpan<double> b)
    {
        double sum = 0;
        for (int r = 0; r < 8; r++)
        {
            double s = 0;
            for (int c = 0; c < 8; c++)
            {
                s += a[(r * 8) + c] * x[c];
            }

            double diff = s - b[r];
            sum += diff * diff;
        }

        return Math.Sqrt(sum);
    }

    private static void RequirePoints(ReadOnlySpan<Point2f> source, ReadOnlySpan<Point2f> destination, int count)
    {
        if (source.Length != count || destination.Length != count)
        {
            throw new ArgumentException($"Exactly {count} source and {count} destination points are required.");
        }
    }

    internal static T[] ReadAffine<T>(Mat<T> matrix)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Width != 3 || matrix.Height != 2 || matrix.Channels != 1)
        {
            throw new ArgumentException("The matrix must be 2x3 (width 3, height 2) with one channel.", nameof(matrix));
        }

        return matrix.ToArray();
    }
}
