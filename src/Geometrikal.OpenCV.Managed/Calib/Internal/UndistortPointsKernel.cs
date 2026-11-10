// This implementation is ported from the OpenCV undistortPoints.
//
// Original source:
//   modules/geometry/src/pinhole.cpp (undistortPointsInternal)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/pinhole.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Iterative point undistortion of <c>undistortPointsInternal</c> with the OpenCV operation order.
/// </summary>
internal sealed class UndistortPointsKernel
{
    private readonly double[] _k;
    private readonly double[] _matTilt;
    private readonly double[] _invMatTilt;
    private readonly double[] _rr;
    private readonly bool _haveDist;
    private readonly bool _applyRr;
    private readonly double _fx, _fy, _ifx, _ify, _cx, _cy;
    private readonly TermCriteria _criteria;

    public UndistortPointsKernel(Mat<double> cameraMatrix, ReadOnlySpan<double> distCoeffs, Mat<double>? r, Mat<double>? p, TermCriteria criteria)
    {
        if (!IsValid(criteria))
        {
            throw new ArgumentException("The termination criteria are not valid.", nameof(criteria));
        }

        _criteria = criteria;
        double[] a = CameraDistortion.Read3x3(cameraMatrix, nameof(cameraMatrix));
        _k = CameraDistortion.Expand(distCoeffs, nameof(distCoeffs));
        _haveDist = distCoeffs.Length > 0;
        _matTilt = CameraDistortion.Identity();
        _invMatTilt = CameraDistortion.Identity();
        if (_haveDist && (_k[12] != 0 || _k[13] != 0))
        {
            CameraDistortion.TiltProjection(_k[12], _k[13], null, invMatTilt: _invMatTilt);
            CameraDistortion.TiltProjection(_k[12], _k[13], _matTilt);
        }

        double[] rr = r is null ? CameraDistortion.Identity() : CameraDistortion.Read3x3(r, nameof(r));
        if (p is not null)
        {
            double[] pp = CameraDistortion.ReadLeft3x3(p, nameof(p));
            Mat<double> prod = LinearAlgebra.Gemm(CameraDistortion.ToMat(pp), CameraDistortion.ToMat(rr), 1, null, 0);
            rr = CameraDistortion.Read3x3(prod, nameof(p));
        }

        _rr = rr;
        _applyRr = r is not null || p is not null;
        _fx = a[0];
        _fy = a[4];
        _ifx = 1.0 / _fx;
        _ify = 1.0 / _fy;
        _cx = a[2];
        _cy = a[5];
    }

    /// <summary>OpenCV <c>TermCriteria::isValid</c>.</summary>
    private static bool IsValid(TermCriteria c)
        => ((c.Type & TermCriteriaType.MaxIter) != 0 && c.MaxCount > 0)
           || ((c.Type & TermCriteriaType.Eps) != 0 && !double.IsNaN(c.Epsilon));

    public void Undistort(double sx, double sy, out double ox, out double oy)
    {
        double[] k = _k;
        double x = sx, y = sy;
        double u = x, v = y;
        x = (x - _cx) * _ifx;
        y = (y - _cy) * _ify;

        if (_haveDist)
        {
            double ux = CameraDistortion.MulRow(_invMatTilt, 0, x, y);
            double uy = CameraDistortion.MulRow(_invMatTilt, 1, x, y);
            double uz = CameraDistortion.MulRow(_invMatTilt, 2, x, y);
            double invProj = uz != 0 ? 1.0 / uz : 1;
            double x0 = x = invProj * ux;
            double y0 = y = invProj * uy;

            double error = double.MaxValue;
            double prevError = double.MaxValue;
            double alpha = 1.0;
            bool useCount = (_criteria.Type & TermCriteriaType.MaxIter) != 0;
            bool useEps = (_criteria.Type & TermCriteriaType.Eps) != 0;

            for (int j = 0; ; j++)
            {
                if (useCount && j >= _criteria.MaxCount)
                {
                    break;
                }

                if (useEps && error < _criteria.Epsilon)
                {
                    break;
                }

                double r2 = x * x + y * y;
                double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
                if (icdist < 0)
                {
                    x = (u - _cx) * _ifx;
                    y = (v - _cy) * _ify;
                    break;
                }

                double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x) + k[8] * r2 + k[9] * r2 * r2;
                double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y + k[10] * r2 + k[11] * r2 * r2;
                double newX = (1.0 - alpha) * x + alpha * (x0 - deltaX) * icdist;
                double newY = (1.0 - alpha) * y + alpha * (y0 - deltaY) * icdist;

                if (useEps)
                {
                    r2 = newX * newX + newY * newY;
                    double r4 = r2 * r2;
                    double r6 = r4 * r2;
                    double a1 = 2 * newX * newY;
                    double a2 = r2 + 2 * newX * newX;
                    double a3 = r2 + 2 * newY * newY;
                    double cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
                    double icdist2 = 1.0 / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
                    double xd0 = newX * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
                    double yd0 = newY * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;
                    double tx = CameraDistortion.MulRow(_matTilt, 0, xd0, yd0);
                    double ty = CameraDistortion.MulRow(_matTilt, 1, xd0, yd0);
                    double tz = CameraDistortion.MulRow(_matTilt, 2, xd0, yd0);
                    invProj = tz != 0 ? 1.0 / tz : 1;
                    double xd = invProj * tx;
                    double yd = invProj * ty;
                    double xProj = xd * _fx + _cx;
                    double yProj = yd * _fy + _cy;
                    error = Math.Sqrt(Math.Pow(xProj - u, 2) + Math.Pow(yProj - v, 2));
                }

                if (error > prevError)
                {
                    alpha *= .5;
                }
                else
                {
                    x = newX;
                    y = newY;
                }

                prevError = error;
            }
        }

        if (_applyRr)
        {
            double[] rr = _rr;
            double xx = rr[0] * x + rr[1] * y + rr[2];
            double yy = rr[3] * x + rr[4] * y + rr[5];
            double ww = 1.0 / (rr[6] * x + rr[7] * y + rr[8]);
            x = xx * ww;
            y = yy * ww;
        }

        ox = x;
        oy = y;
    }
}
