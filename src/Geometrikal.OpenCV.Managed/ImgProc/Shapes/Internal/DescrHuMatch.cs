// This implementation is ported from the OpenCV moments and shape matching.
//
// Original sources:
//   modules/geometry/src/moments.cpp       (cv::HuMoments)
//   modules/geometry/src/matchcontours.cpp (cv::matchShapes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>Hu invariants and the Hu-moment shape distances of OpenCV.</summary>
internal static class DescrHuMatch
{
    public static void HuMoments(in Moments m, Span<double> hu)
    {
        double t0 = m.Nu30 + m.Nu12;
        double t1 = m.Nu21 + m.Nu03;
        double q0 = t0 * t0, q1 = t1 * t1;
        double n4 = 4 * m.Nu11;
        double s = m.Nu20 + m.Nu02;
        double d = m.Nu20 - m.Nu02;

        hu[0] = s;
        hu[1] = (d * d) + (n4 * m.Nu11);
        hu[3] = q0 + q1;
        hu[5] = (d * (q0 - q1)) + (n4 * t0 * t1);

        t0 *= q0 - (3 * q1);
        t1 *= (3 * q0) - q1;

        q0 = m.Nu30 - (3 * m.Nu12);
        q1 = (3 * m.Nu21) - m.Nu03;

        hu[2] = (q0 * q0) + (q1 * q1);
        hu[4] = (q0 * t0) + (q1 * t1);
        hu[6] = (q1 * t0) - (q0 * t1);
    }

    public static double MatchShapes(in Moments a, in Moments b, ShapeMatchModes method)
    {
        Span<double> ma = stackalloc double[7];
        Span<double> mb = stackalloc double[7];
        HuMoments(a, ma);
        HuMoments(b, mb);

        const double eps = 1.0e-5;
        double result = 0;
        bool anyA = false, anyB = false;
        for (int i = 0; i < 7; i++)
        {
            double ama = Math.Abs(ma[i]);
            double amb = Math.Abs(mb[i]);
            anyA |= ama > 0;
            anyB |= amb > 0;
            int sma = ma[i] > 0 ? 1 : ma[i] < 0 ? -1 : 0;
            int smb = mb[i] > 0 ? 1 : mb[i] < 0 ? -1 : 0;
            if (!(ama > eps && amb > eps))
            {
                continue;
            }

            switch (method)
            {
                case ShapeMatchModes.ContoursMatchI1:
                    ama = 1.0 / (sma * Math.Log10(ama));
                    amb = 1.0 / (smb * Math.Log10(amb));
                    result += Math.Abs(-ama + amb);
                    break;
                case ShapeMatchModes.ContoursMatchI2:
                    ama = sma * Math.Log10(ama);
                    amb = smb * Math.Log10(amb);
                    result += Math.Abs(-ama + amb);
                    break;
                default:
                    ama = sma * Math.Log10(ama);
                    amb = smb * Math.Log10(amb);
                    double mmm = Math.Abs((ama - amb) / ama);
                    if (result < mmm)
                    {
                        result = mmm;
                    }

                    break;
            }
        }

        return anyA != anyB ? double.MaxValue : result;
    }
}
