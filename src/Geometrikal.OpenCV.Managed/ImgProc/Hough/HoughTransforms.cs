// This file is a managed port of OpenCV modules/imgproc/src/hough.cpp (HoughLines,
// HoughLinesP, HoughLinesPointSet).
// OpenCV is licensed under the Apache License 2.0. Copyright (C) 2000-2024, Intel Corporation,
// all rights reserved, and OpenCV contributors.

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>A line in polar form: <c>x·cos(θ) + y·sin(θ) = ρ</c>, with the accumulator votes.</summary>
/// <param name="Rho">Distance from the image origin, in pixels.</param>
/// <param name="Theta">Angle of the line normal, in radians.</param>
/// <param name="Votes">Accumulator value (OpenCV <c>CV_32FC3</c> output).</param>
public readonly record struct HoughLine(float Rho, float Theta, float Votes);

/// <summary>A line segment from <c>(X1, Y1)</c> to <c>(X2, Y2)</c>.</summary>
public readonly record struct LineSegment(int X1, int Y1, int X2, int Y2);

/// <summary>A line found in a point set, as OpenCV <c>HoughLinesPointSet</c> (<c>Vec3d</c>).</summary>
public readonly record struct HoughPointSetLine(double Votes, double Rho, double Theta);

/// <summary>
/// Hough transforms for lines, as OpenCV <c>HoughLines</c>, <c>HoughLinesP</c> and
/// <c>HoughLinesPointSet</c>. The functions read the inputs and do not change them.
/// </summary>
public static partial class HoughTransforms
{
    /// <summary>
    /// Finds lines in a binary image with the standard or multi-scale Hough transform, as OpenCV
    /// <c>HoughLines</c>. Lines are sorted by decreasing votes.
    /// </summary>
    /// <param name="image">8-bit single-channel image; non-zero pixels are edge points.</param>
    /// <param name="rho">Distance resolution of the accumulator, in pixels.</param>
    /// <param name="theta">Angle resolution of the accumulator, in radians.</param>
    /// <param name="threshold">Votes a line needs; a line needs more than this many.</param>
    /// <param name="srn">With <paramref name="stn"/>: divisor of <paramref name="rho"/> for the multi-scale transform; 0 for the standard transform.</param>
    /// <param name="stn">Divisor of <paramref name="theta"/> for the multi-scale transform.</param>
    /// <param name="minTheta">Smallest angle (standard transform).</param>
    /// <param name="maxTheta">Largest angle (standard transform).</param>
    /// <param name="useEdgeValue">Standard transform only: each pixel votes with its value instead of 1 (OpenCV 5.0).</param>
    /// <remarks>The multi-scale transform clamps the threshold to 255 and ignores the angle range, as OpenCV.</remarks>
    public static HoughLine[] HoughLines(
        Mat<byte> image,
        double rho,
        double theta,
        int threshold,
        double srn = 0,
        double stn = 0,
        double minTheta = 0,
        double maxTheta = Math.PI,
        bool useEdgeValue = false)
    {
        ValidateImage(image);
        if (srn == 0 && stn == 0)
        {
            return Standard(image, (float)rho, (float)theta, threshold, int.MaxValue, minTheta, maxTheta, useEdgeValue);
        }

        return MultiScale(image, (float)rho, (float)theta, threshold, ScalarConverter.CvRound(srn), ScalarConverter.CvRound(stn), int.MaxValue, minTheta, maxTheta);
    }

    /// <summary>
    /// Finds line segments with the progressive probabilistic Hough transform, as OpenCV
    /// <c>HoughLinesP</c>. The points are visited in the order of a <c>cv::RNG</c> seeded with
    /// <c>(uint64)-1</c>, so the result is deterministic and equal to OpenCV's.
    /// </summary>
    /// <param name="image">8-bit single-channel image; non-zero pixels are edge points.</param>
    /// <param name="rho">Distance resolution of the accumulator, in pixels.</param>
    /// <param name="theta">Angle resolution of the accumulator, in radians.</param>
    /// <param name="threshold">Votes a line needs.</param>
    /// <param name="minLineLength">Shortest segment kept (in X or Y extent).</param>
    /// <param name="maxLineGap">Largest gap between points of one segment.</param>
    public static LineSegment[] HoughLinesP(Mat<byte> image, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
    {
        ValidateImage(image);
        return Probabilistic(image, (float)rho, (float)theta, threshold,
            ScalarConverter.CvRound(minLineLength), ScalarConverter.CvRound(maxLineGap), int.MaxValue);
    }

    /// <summary>
    /// Finds lines through a set of points with the standard Hough transform over a given range, as
    /// OpenCV <c>HoughLinesPointSet</c>.
    /// </summary>
    /// <param name="points">Points (integer or floating point coordinates).</param>
    /// <param name="linesMax">Largest number of lines returned.</param>
    /// <param name="threshold">Votes a line needs; a line needs more than this many.</param>
    /// <param name="minRho">Smallest distance.</param>
    /// <param name="maxRho">Largest distance.</param>
    /// <param name="rhoStep">Distance resolution.</param>
    /// <param name="minTheta">Smallest angle in radians.</param>
    /// <param name="maxTheta">Largest angle in radians.</param>
    /// <param name="thetaStep">Angle resolution in radians.</param>
    public static HoughPointSetLine[] HoughLinesPointSet(
        IReadOnlyList<Point2f> points, int linesMax, int threshold,
        double minRho, double maxRho, double rhoStep, double minTheta, double maxTheta, double thetaStep)
    {
        ArgumentNullException.ThrowIfNull(points);
        if (linesMax <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(linesMax), "lines_max must be greater than 0");
        }

        if (threshold < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "threshold must be greater than 0");
        }

        if (maxRho - minRho <= 0 || maxTheta - minTheta <= 0)
        {
            throw new ArgumentException("max must be greater than min");
        }

        if (rhoStep <= 0 || thetaStep <= 0)
        {
            throw new ArgumentException("step must be greater than 0");
        }

        float irho = 1 / (float)rhoStep;
        float irhoMin = (float)minRho * irho;
        int numangle = ComputeNumAngle(minTheta, maxTheta, thetaStep);
        int numrho = ScalarConverter.CvRound((maxRho - minRho + 1) / rhoStep);
        var accum = new int[(numangle + 2) * (numrho + 2)];
        var tabSin = new float[numangle];
        var tabCos = new float[numangle];
        CreateTrigTable(numangle, minTheta, thetaStep, irho, tabSin, tabCos);

        foreach (var p in points)
        {
            for (int n = 0; n < numangle; n++)
            {
                int r = ScalarConverter.CvRound((p.X * tabCos[n]) + (p.Y * tabSin[n]) - irhoMin);
                if (r >= 0 && r <= numrho)
                {
                    accum[((n + 1) * (numrho + 2)) + r + 1]++;
                }
            }
        }

        var sorted = SortedMaximums(numrho, numangle, threshold, accum);
        int count = Math.Min(linesMax, sorted.Count);
        double scale = 1.0 / (numrho + 2);
        var lines = new HoughPointSetLine[count];
        for (int i = 0; i < count; i++)
        {
            int idx = sorted[i];
            int n = ScalarConverter.CvFloor(idx * scale) - 1;
            int r = idx - ((n + 1) * (numrho + 2)) - 1;
            float lineRho = (float)minRho + (r * (float)rhoStep);
            float lineAngle = (float)minTheta + (n * (float)thetaStep);
            lines[i] = new HoughPointSetLine(accum[idx], lineRho, lineAngle);
        }

        return lines;
    }

    private static HoughLine[] Standard(Mat<byte> img, float rho, float theta, int threshold, int linesMax, double minTheta, double maxTheta, bool useEdgeValue)
    {
        if (maxTheta < minTheta)
        {
            throw new ArgumentException("max_theta must be greater than min_theta");
        }

        float irho = 1 / rho;
        int width = img.Width, height = img.Height;
        int maxRho = width + height;
        int minRho = -maxRho;
        int numangle = ComputeNumAngle(minTheta, maxTheta, theta);
        int numrho = ScalarConverter.CvRound((maxRho - minRho + 1) / rho);
        int stride = numrho + 2;
        var accum = new int[(numangle + 2) * stride];
        var tabSin = new float[numangle];
        var tabCos = new float[numangle];
        CreateTrigTable(numangle, minTheta, theta, irho, tabSin, tabCos);

        int half = (numrho - 1) / 2;
        for (int i = 0; i < height; i++)
        {
            var row = img.GetReadOnlyRowSpan(i);
            for (int j = 0; j < width; j++)
            {
                if (row[j] == 0)
                {
                    continue;
                }

                int vote = useEdgeValue ? row[j] : 1;
                for (int n = 0; n < numangle; n++)
                {
                    int r = ScalarConverter.CvRound((j * tabCos[n]) + (i * tabSin[n])) + half;
                    accum[((n + 1) * stride) + r + 1] += vote;
                }
            }
        }

        var sorted = SortedMaximums(numrho, numangle, threshold, accum);
        int count = Math.Min(linesMax, sorted.Count);
        double scale = 1.0 / stride;
        var lines = new HoughLine[count];
        for (int i = 0; i < count; i++)
        {
            int idx = sorted[i];
            int n = ScalarConverter.CvFloor(idx * scale) - 1;
            int r = idx - ((n + 1) * stride) - 1;
            lines[i] = new HoughLine((r - half) * rho, (float)minTheta + (n * theta), accum[idx]);
        }

        return lines;
    }

    private static HoughLine[] MultiScale(Mat<byte> img, float rho, float theta, int threshold, int srn, int stn, int linesMax, double minTheta, double maxTheta)
    {
        const float d2r = (float)(Math.PI / 180);
        threshold = Math.Min(threshold, 255);
        int w = img.Width, h = img.Height;
        int sfn = srn * stn;
        float irho = 1 / rho;
        float itheta = 1 / theta;
        float srho = rho / srn;
        float stheta = theta / stn;
        float isrho = 1 / srho;
        float istheta = 1 / stheta;
        int rn = ScalarConverter.CvFloor(Math.Sqrt(((double)w * w) + ((double)h * h)) * irho);
        int tn = ScalarConverter.CvFloor(2 * Math.PI * itheta);

        var list = new List<(int Value, float Rho, float Theta)> { (threshold, -1f, 0f) };
        var sinTable = new float[5 * tn * stn];
        for (int index = 0; index < sinTable.Length; index++)
        {
            sinTable[index] = MathF.Cos(stheta * index * 0.2f);
        }

        var caccum = new byte[rn * tn];
        var xs = new List<int>();
        var ys = new List<int>();
        int cmax = 0;
        for (int row = 0; row < h; row++)
        {
            var data = img.GetReadOnlyRowSpan(row);
            for (int col = 0; col < w; col++)
            {
                if (data[col] == 0)
                {
                    continue;
                }

                xs.Add(col);
                ys.Add(row);
                float yc = row + 0.5f;
                float xc = col + 0.5f;
                float t = MathF.Abs(FastMath.FastAtan2(yc, xc) * d2r);
                float r = (float)Math.Sqrt(((double)xc * xc) + ((double)yc * yc));
                float r0 = r * irho;
                int ti0 = ScalarConverter.CvFloor((t + (Math.PI * 0.5)) * itheta);
                caccum[ti0]++;

                float thetaIt = rho / r;
                thetaIt = thetaIt < theta ? thetaIt : theta;
                float scaleFactor = thetaIt * itheta;
                int halftn = ScalarConverter.CvFloor(Math.PI / thetaIt);
                int iprev = -1;
                float phi = thetaIt - (float)(Math.PI * 0.5);
                float phi1 = (thetaIt + t) * itheta;
                for (int ti1 = 1; ti1 < halftn; ti1++, phi += thetaIt, phi1 += scaleFactor)
                {
                    float rv = r0 * MathF.Cos(phi);
                    int i = ((int)rv * tn) + ScalarConverter.CvFloor(phi1);
                    if (i < 0 || i >= rn * tn)
                    {
                        throw new InvalidOperationException("Hough accumulator index out of range.");
                    }

                    caccum[i] = (byte)(caccum[i] + ((i ^ iprev) != 0 ? 1 : 0));
                    iprev = i;
                    if (cmax < caccum[i])
                    {
                        cmax = caccum[i];
                    }
                }
            }
        }

        int count = 0;
        for (int k = 0; k < caccum.Length; k++)
        {
            if (caccum[k] > threshold)
            {
                count++;
            }
        }

        if (count * 100 > rn * tn)
        {
            return Standard(img, rho, theta, threshold, linesMax, minTheta, maxTheta, false);
        }

        // mcaccum[-1 .. sfn] lives in buffer[0 .. sfn + 1].
        var buffer = new byte[sfn + 2];
        for (int ri = 0; ri < rn; ri++)
        {
            for (int ti = 0; ti < tn; ti++)
            {
                if (caccum[(ri * tn) + ti] <= threshold)
                {
                    continue;
                }

                Array.Clear(buffer, 1, sfn);
                for (int index = 0; index < xs.Count; index++)
                {
                    float yc = ys[index] + 0.5f;
                    float xc = xs[index] + 0.5f;
                    float t = MathF.Abs(FastMath.FastAtan2(yc, xc) * d2r);
                    float r = (float)Math.Sqrt(((double)xc * xc) + ((double)yc * yc)) * isrho;
                    int ti0 = ScalarConverter.CvFloor((t + (Math.PI * 0.5)) * istheta);
                    int ti2 = ((ti * stn) - ti0) * 5;
                    float r0 = (float)ri * srn;
                    for (int ti1 = 0; ti1 < stn; ti1++, ti2 += 5)
                    {
                        float rv = (r * sinTable[Math.Abs(ti2)]) - r0;
                        int i = (ScalarConverter.CvFloor(rv) * stn) + ti1;
                        i = Math.Max(i, -1);
                        i = Math.Min(i, sfn);
                        buffer[i + 1]++;
                    }
                }

                for (int index = 0; index < sfn; index++)
                {
                    int value = buffer[index + 1];
                    int pos = list.Count - 1;
                    if (pos < 0 || list[pos].Value < value)
                    {
                        var vi = (value, ((index / stn) * srho) + (ri * rho), ((index % stn) * stheta) + (ti * theta) - (float)(Math.PI * 0.5));
                        list.Add(vi);
                        for (; pos >= 0; pos--)
                        {
                            if (list[pos].Value > vi.value)
                            {
                                break;
                            }

                            list[pos + 1] = list[pos];
                        }

                        list[pos + 1] = vi;
                        if (list.Count > linesMax)
                        {
                            list.RemoveAt(list.Count - 1);
                        }
                    }
                }
            }
        }

        if (list.Count > 0 && list[^1].Rho < 0)
        {
            list.RemoveAt(list.Count - 1);
        }

        return list.Select(l => new HoughLine(l.Rho, l.Theta, l.Value)).ToArray();
    }

    private static LineSegment[] Probabilistic(Mat<byte> image, float rho, float theta, int threshold, int lineLength, int lineGap, int linesMax)
    {
        const int shift = 16;
        float irho = 1 / rho;
        var rng = new CvRng(ulong.MaxValue);
        int width = image.Width, height = image.Height;
        int numangle = ComputeNumAngle(0.0, Math.PI, theta);
        int numrho = ScalarConverter.CvRound(((width + height) * 2 + 1) / rho);
        var accum = new int[numangle * numrho];
        var mask = new byte[width * height];
        var trig = new float[numangle * 2];
        for (int n = 0; n < numangle; n++)
        {
            trig[n * 2] = (float)(Math.Cos((double)n * theta) * irho);
            trig[(n * 2) + 1] = (float)(Math.Sin((double)n * theta) * irho);
        }

        var nzloc = new List<Point>();
        for (int y = 0; y < height; y++)
        {
            var data = image.GetReadOnlyRowSpan(y);
            for (int x = 0; x < width; x++)
            {
                if (data[x] != 0)
                {
                    mask[(y * width) + x] = 1;
                    nzloc.Add(new Point(x, y));
                }
            }
        }

        var lines = new List<LineSegment>();
        int half = (numrho - 1) / 2;
        Span<Point> lineEnd = stackalloc Point[2];
        for (int count = nzloc.Count; count > 0; count--)
        {
            int idx = rng.Uniform(0, count);
            int maxVal = threshold - 1, maxN = 0;
            var point = nzloc[idx];
            int i = point.Y, j = point.X;
            nzloc[idx] = nzloc[count - 1];

            if (mask[(i * width) + j] == 0)
            {
                continue;
            }

            for (int n = 0; n < numangle; n++)
            {
                int r = ScalarConverter.CvRound((j * trig[n * 2]) + (i * trig[(n * 2) + 1])) + half;
                int val = ++accum[(n * numrho) + r];
                if (maxVal < val)
                {
                    maxVal = val;
                    maxN = n;
                }
            }

            if (maxVal < threshold)
            {
                continue;
            }

            float a = -trig[(maxN * 2) + 1];
            float b = trig[maxN * 2];
            long x0 = j, y0 = i, dx0, dy0;
            bool xflag;
            if (MathF.Abs(a) > MathF.Abs(b))
            {
                xflag = true;
                dx0 = a > 0 ? 1 : -1;
                dy0 = ScalarConverter.CvRound(b * (1 << shift) / MathF.Abs(a));
                y0 = (y0 << shift) + (1 << (shift - 1));
            }
            else
            {
                xflag = false;
                dy0 = b > 0 ? 1 : -1;
                dx0 = ScalarConverter.CvRound(a * (1 << shift) / MathF.Abs(b));
                x0 = (x0 << shift) + (1 << (shift - 1));
            }

            lineEnd[0] = default;
            lineEnd[1] = default;
            for (int k = 0; k < 2; k++)
            {
                int gap = 0;
                long x = x0, y = y0, dx = k > 0 ? -dx0 : dx0, dy = k > 0 ? -dy0 : dy0;
                for (; ; x += dx, y += dy)
                {
                    long j1 = xflag ? x : x >> shift;
                    long i1 = xflag ? y >> shift : y;
                    if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                    {
                        break;
                    }

                    if (mask[(i1 * width) + j1] != 0)
                    {
                        gap = 0;
                        lineEnd[k] = new Point((int)j1, (int)i1);
                    }
                    else if (++gap > lineGap)
                    {
                        break;
                    }
                }
            }

            bool goodLine = Math.Abs(lineEnd[1].X - lineEnd[0].X) >= lineLength || Math.Abs(lineEnd[1].Y - lineEnd[0].Y) >= lineLength;
            for (int k = 0; k < 2; k++)
            {
                long x = x0, y = y0, dx = k > 0 ? -dx0 : dx0, dy = k > 0 ? -dy0 : dy0;
                for (; ; x += dx, y += dy)
                {
                    long j1 = xflag ? x : x >> shift;
                    long i1 = xflag ? y >> shift : y;
                    long m = (i1 * width) + j1;
                    if (mask[m] != 0)
                    {
                        if (goodLine)
                        {
                            for (int n = 0; n < numangle; n++)
                            {
                                int r = ScalarConverter.CvRound((j1 * trig[n * 2]) + (i1 * trig[(n * 2) + 1])) + half;
                                accum[(n * numrho) + r]--;
                            }
                        }

                        mask[m] = 0;
                    }

                    if (i1 == lineEnd[k].Y && j1 == lineEnd[k].X)
                    {
                        break;
                    }
                }
            }

            if (goodLine)
            {
                lines.Add(new LineSegment(lineEnd[0].X, lineEnd[0].Y, lineEnd[1].X, lineEnd[1].Y));
                if (lines.Count >= linesMax)
                {
                    break;
                }
            }
        }

        return [.. lines];
    }

    private static int ComputeNumAngle(double minTheta, double maxTheta, double thetaStep)
    {
        int numangle = ScalarConverter.CvFloor((maxTheta - minTheta) / thetaStep) + 1;

        // An angle range of about π would detect a line twice; the last angle is dropped.
        if (numangle > 1 && Math.Abs(Math.PI - ((numangle - 1) * thetaStep)) < thetaStep / 2)
        {
            --numangle;
        }

        return numangle;
    }

    private static void CreateTrigTable(int numangle, double minTheta, double thetaStep, float irho, float[] tabSin, float[] tabCos)
    {
        float angle = (float)minTheta;
        for (int n = 0; n < numangle; angle += (float)thetaStep, n++)
        {
            tabSin[n] = (float)(Math.Sin(angle) * irho);
            tabCos[n] = (float)(Math.Cos(angle) * irho);
        }
    }

    /// <summary>Local maximums above the threshold, sorted by decreasing votes, then by index.</summary>
    private static List<int> SortedMaximums(int numrho, int numangle, int threshold, int[] accum)
    {
        var maximums = new List<int>();
        for (int r = 0; r < numrho; r++)
        {
            for (int n = 0; n < numangle; n++)
            {
                int b = ((n + 1) * (numrho + 2)) + r + 1;
                if (accum[b] > threshold && accum[b] > accum[b - 1] && accum[b] >= accum[b + 1]
                    && accum[b] > accum[b - numrho - 2] && accum[b] >= accum[b + numrho + 2])
                {
                    maximums.Add(b);
                }
            }
        }

        maximums.Sort((l1, l2) => accum[l1] != accum[l2] ? accum[l2].CompareTo(accum[l1]) : l1.CompareTo(l2));
        return maximums;
    }

    private static void ValidateImage(Mat<byte> image)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels != 1)
        {
            throw new ArgumentException("The image must have one channel.", nameof(image));
        }
    }
}
