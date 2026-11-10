// This implementation is ported from the OpenCV moments implementation.
//
// Original source:
//   modules/geometry/src/moments.cpp (cv::Moments::Moments, completeMomentState)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Spatial, central and normalized central moments up to the third order, mirroring OpenCV
/// <c>cv::Moments</c>. The property names are the OpenCV field names in Pascal case
/// (<c>m00</c> is <see cref="M00"/>, <c>mu20</c> is <see cref="Mu20"/>, <c>nu20</c> is
/// <see cref="Nu20"/>).
/// </summary>
/// <remarks>
/// <para>Spatial moments: <c>Mji = sum(I(x, y) * x^j * y^i)</c>.</para>
/// <para>Central moments: <c>muji = sum(I(x, y) * (x - cx)^j * (y - cy)^i)</c> with the mass center
/// <c>cx = M10 / M00</c>, <c>cy = M01 / M00</c>. <c>mu00 = M00</c>, <c>mu10 = mu01 = 0</c>; OpenCV
/// does not store them.</para>
/// <para>Normalized central moments: <c>nuji = muji / M00^((i + j) / 2 + 1)</c>.</para>
/// <para>When <c>|M00| &lt;= DBL_EPSILON</c>, the mass center is (0, 0) and every normalized moment
/// is 0. <c>default(Moments)</c> has all values 0.</para>
/// </remarks>
public readonly record struct Moments
{
    /// <summary>
    /// Initializes the spatial moments and computes the central and normalized central moments
    /// from them, as the OpenCV <c>cv::Moments</c> constructor.
    /// </summary>
    public Moments(double m00, double m10, double m01, double m20, double m11, double m02, double m30, double m21, double m12, double m03)
    {
        M00 = m00;
        M10 = m10;
        M01 = m01;
        M20 = m20;
        M11 = m11;
        M02 = m02;
        M30 = m30;
        M21 = m21;
        M12 = m12;
        M03 = m03;

        double cx = 0, cy = 0, invM00 = 0;
        if (Math.Abs(m00) > DblEpsilon)
        {
            invM00 = 1.0 / m00;
            cx = m10 * invM00;
            cy = m01 * invM00;
        }

        double mu20 = m20 - (m10 * cx);
        double mu11 = m11 - (m10 * cy);
        double mu02 = m02 - (m01 * cy);
        Mu20 = mu20;
        Mu11 = mu11;
        Mu02 = mu02;
        Mu30 = m30 - (cx * ((3 * mu20) + (cx * m10)));
        mu11 += mu11;
        Mu21 = m21 - (cx * (mu11 + (cx * m01))) - (cy * mu20);
        Mu12 = m12 - (cy * (mu11 + (cy * m10))) - (cx * mu02);
        Mu03 = m03 - (cy * ((3 * mu02) + (cy * m01)));

        double invSqrtM00 = Math.Sqrt(Math.Abs(invM00));
        double s2 = invM00 * invM00, s3 = s2 * invSqrtM00;
        Nu20 = Mu20 * s2;
        Nu11 = Mu11 * s2;
        Nu02 = Mu02 * s2;
        Nu30 = Mu30 * s3;
        Nu21 = Mu21 * s3;
        Nu12 = Mu12 * s3;
        Nu03 = Mu03 * s3;
    }

    /// <summary>C <c>DBL_EPSILON</c> (2^-52).</summary>
    private const double DblEpsilon = 2.220446049250313e-16;

    /// <summary>Spatial moment m00 (area or total mass).</summary>
    public double M00 { get; init; }

    /// <summary>Spatial moment m10.</summary>
    public double M10 { get; init; }

    /// <summary>Spatial moment m01.</summary>
    public double M01 { get; init; }

    /// <summary>Spatial moment m20.</summary>
    public double M20 { get; init; }

    /// <summary>Spatial moment m11.</summary>
    public double M11 { get; init; }

    /// <summary>Spatial moment m02.</summary>
    public double M02 { get; init; }

    /// <summary>Spatial moment m30.</summary>
    public double M30 { get; init; }

    /// <summary>Spatial moment m21.</summary>
    public double M21 { get; init; }

    /// <summary>Spatial moment m12.</summary>
    public double M12 { get; init; }

    /// <summary>Spatial moment m03.</summary>
    public double M03 { get; init; }

    /// <summary>Central moment mu20.</summary>
    public double Mu20 { get; init; }

    /// <summary>Central moment mu11.</summary>
    public double Mu11 { get; init; }

    /// <summary>Central moment mu02.</summary>
    public double Mu02 { get; init; }

    /// <summary>Central moment mu30.</summary>
    public double Mu30 { get; init; }

    /// <summary>Central moment mu21.</summary>
    public double Mu21 { get; init; }

    /// <summary>Central moment mu12.</summary>
    public double Mu12 { get; init; }

    /// <summary>Central moment mu03.</summary>
    public double Mu03 { get; init; }

    /// <summary>Normalized central moment nu20.</summary>
    public double Nu20 { get; init; }

    /// <summary>Normalized central moment nu11.</summary>
    public double Nu11 { get; init; }

    /// <summary>Normalized central moment nu02.</summary>
    public double Nu02 { get; init; }

    /// <summary>Normalized central moment nu30.</summary>
    public double Nu30 { get; init; }

    /// <summary>Normalized central moment nu21.</summary>
    public double Nu21 { get; init; }

    /// <summary>Normalized central moment nu12.</summary>
    public double Nu12 { get; init; }

    /// <summary>Normalized central moment nu03.</summary>
    public double Nu03 { get; init; }
}
