// This implementation is ported from the OpenCV sparse optical flow module.
//
// Original sources:
//   modules/video/src/lkpyramid.cpp (buildOpticalFlowPyramid, calcOpticalFlowPyrLK,
//     SparsePyrLKOpticalFlowImpl::calc, LKTrackerInvoker)
//   modules/video/src/lkpyramid.simd.hpp (ScharrDerivInvoker_SIMD)
//   modules/video/include/opencv2/video/tracking.hpp (declarations, OPTFLOW_* flags)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Video;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Video.Internal;

/// <summary>Flags of <see cref="OpticalFlow.CalcOpticalFlowPyrLK(Mat{byte}, Mat{byte}, Point2f[], Point2f[], byte[], float[], Size, int, TermCriteria, OpticalFlowFlags, double)"/> (OpenCV values).</summary>
[Flags]
public enum OpticalFlowFlags
{
    /// <summary>No flags.</summary>
    None = 0,

    /// <summary><c>OPTFLOW_USE_INITIAL_FLOW</c>: the next points hold the initial estimates.</summary>
    UseInitialFlow = 4,

    /// <summary><c>OPTFLOW_LK_GET_MIN_EIGENVALS</c>: the error is the minimum eigenvalue of the spatial gradient matrix.</summary>
    LkGetMinEigenvals = 8,
}

/// <summary>
/// Output of <see cref="OpticalFlow.BuildOpticalFlowPyramid"/>. Each level is a view into a
/// buffer padded by the window size on every side; the padding holds the requested border.
/// </summary>
public sealed class OpticalFlowPyramid
{
    internal OpticalFlowPyramid(Size winSize, Mat<byte>[] images, Mat<short>[]? derivatives)
    {
        WinSize = winSize;
        Images = images;
        Derivatives = derivatives;
    }

    /// <summary>Window size used for the padding.</summary>
    public Size WinSize { get; }

    /// <summary>Index of the last level, as the OpenCV return value.</summary>
    public int MaxLevel => Images.Length - 1;

    /// <summary>Pyramid levels; level 0 has the input size.</summary>
    public Mat<byte>[] Images { get; }

    /// <summary>
    /// Scharr derivatives per level (CV_16SC(2·cn): dx and dy per channel, interleaved), or null
    /// when the pyramid was built without derivatives.
    /// </summary>
    public Mat<short>[]? Derivatives { get; }
}

/// <summary>Result of the allocating <c>CalcOpticalFlowPyrLK</c> overloads.</summary>
/// <param name="NextPoints">Tracked positions in the next image.</param>
/// <param name="Status">1 when the point was tracked, otherwise 0.</param>
/// <param name="Error">Tracking error per point (see the flags). Entries that OpenCV leaves unset are 0.</param>
public sealed record OpticalFlowResult(Point2f[] NextPoints, byte[] Status, float[] Error);

/// <summary>
/// Managed ports of the OpenCV pyramidal Lucas-Kanade sparse optical flow:
/// <c>buildOpticalFlowPyramid</c> and <c>calcOpticalFlowPyrLK</c>.
/// </summary>
/// <remarks>
/// <para><b>Images.</b> 8-bit images with any channel count. The two images (or pyramids)
/// must have the same size and channel count.</para>
/// <para><b>Arithmetic.</b> Results equal OpenCV 5.0 on its SSE baseline bit for bit: fixed-point
/// bilinear weights with 14 bits, the 16-bit Scharr derivatives, and the float sums in the
/// lane order of the OpenCV 128-bit vector loops.</para>
/// <para><b>Regions of interest.</b> A view into a larger array is a submatrix. Pixels of the
/// array around it replace border pixels of pyramid level 0, as OpenCV <c>copyMakeBorder</c>
/// without <c>BORDER_ISOLATED</c>. All other levels treat their source as isolated.</para>
/// <para><b>Threads.</b> Points run in parallel. The output does not depend on the thread count.</para>
/// </remarks>
public static class OpticalFlow
{
    /// <summary>Default termination criteria of OpenCV: 30 iterations or a step of 0.01.</summary>
    public static readonly TermCriteria DefaultCriteria = new(TermCriteriaType.MaxIter | TermCriteriaType.Eps, 30, 0.01);

    /// <summary>
    /// Builds the image pyramid for <see cref="CalcOpticalFlowPyrLK(OpticalFlowPyramid, OpticalFlowPyramid, Point2f[], Point2f[], byte[], float[], Size, int, TermCriteria, OpticalFlowFlags, double)"/>,
    /// as OpenCV <c>buildOpticalFlowPyramid</c>.
    /// </summary>
    /// <param name="image">8-bit input image. It is not modified.</param>
    /// <param name="winSize">Window size; both sides greater than 2. Every level is padded by this size.</param>
    /// <param name="maxLevel">Maximum level index; non-negative. The pyramid stops before a level with a side not larger than the window side.</param>
    /// <param name="withDerivatives">True to compute the Scharr derivatives of every level.</param>
    /// <param name="pyrBorder">Border of the image padding. <see cref="BorderType.Transparent"/> leaves the padding 0 (OpenCV leaves it uninitialised).</param>
    /// <param name="derivBorder">Border of the derivative padding, with the same rule.</param>
    /// <param name="tryReuseInputImage">
    /// True to use <paramref name="image"/> as level 0 when it is a submatrix with at least the
    /// window size of pixels around it in its array. Level 0 then shares memory with the input.
    /// </param>
    /// <returns>The pyramid. Levels and derivatives are new matrices except a reused level 0.</returns>
    public static OpticalFlowPyramid BuildOpticalFlowPyramid(
        Mat<byte> image,
        Size winSize,
        int maxLevel,
        bool withDerivatives = true,
        BorderType pyrBorder = BorderType.Reflect101,
        BorderType derivBorder = BorderType.Constant,
        bool tryReuseInputImage = true)
    {
        ArgumentNullException.ThrowIfNull(image);
        RequireWindow(winSize);
        ArgumentOutOfRangeException.ThrowIfNegative(maxLevel);
        return LkPyramidBuilder.Build(image, winSize, maxLevel, withDerivatives, pyrBorder, derivBorder, tryReuseInputImage);
    }

    /// <summary>
    /// Tracks <paramref name="prevPts"/> from <paramref name="prevImg"/> to
    /// <paramref name="nextImg"/> with the iterative pyramidal Lucas-Kanade method, as OpenCV
    /// <c>calcOpticalFlowPyrLK</c>.
    /// </summary>
    /// <param name="prevImg">First 8-bit image.</param>
    /// <param name="nextImg">Second 8-bit image with the size and channel count of <paramref name="prevImg"/>.</param>
    /// <param name="prevPts">Points to track. Not modified.</param>
    /// <param name="nextPts">
    /// Output positions, with the length of <paramref name="prevPts"/>. With
    /// <see cref="OpticalFlowFlags.UseInitialFlow"/> it holds the initial estimates on input.
    /// </param>
    /// <param name="status">Output: 1 when the point was tracked, otherwise 0. Same length as the points.</param>
    /// <param name="err">
    /// Optional output error. Without <see cref="OpticalFlowFlags.LkGetMinEigenvals"/> it is the
    /// mean absolute difference of the windows divided by 32 for tracked points; with the flag it is
    /// the minimum eigenvalue. Entries that OpenCV leaves unset are not written.
    /// </param>
    /// <param name="winSize">Search window size per level; both sides greater than 2.</param>
    /// <param name="maxLevel">Maximum pyramid level; 0 uses only the input images.</param>
    /// <param name="criteria">Termination criteria. The count is clamped to [0, 100] (30 when absent), epsilon to [0, 10] (0.01 when absent).</param>
    /// <param name="flags">Operation flags.</param>
    /// <param name="minEigThreshold">Points with a smaller normalised minimum eigenvalue are not tracked.</param>
    public static void CalcOpticalFlowPyrLK(
        Mat<byte> prevImg,
        Mat<byte> nextImg,
        Point2f[] prevPts,
        Point2f[] nextPts,
        byte[] status,
        float[]? err,
        Size winSize,
        int maxLevel,
        TermCriteria criteria,
        OpticalFlowFlags flags = OpticalFlowFlags.None,
        double minEigThreshold = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(prevImg);
        ArgumentNullException.ThrowIfNull(nextImg);
        Calc(prevImg, null, nextImg, null, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);
    }

    /// <summary>
    /// <see cref="CalcOpticalFlowPyrLK(Mat{byte}, Mat{byte}, Point2f[], Point2f[], byte[], float[], Size, int, TermCriteria, OpticalFlowFlags, double)"/>
    /// with prebuilt pyramids. <paramref name="maxLevel"/> is limited to the levels of both
    /// pyramids. The derivatives of <paramref name="prevPyramid"/> are used when it has them.
    /// </summary>
    /// <exception cref="ArgumentException">A pyramid has less padding than <paramref name="winSize"/>, or the level sizes differ.</exception>
    public static void CalcOpticalFlowPyrLK(
        OpticalFlowPyramid prevPyramid,
        OpticalFlowPyramid nextPyramid,
        Point2f[] prevPts,
        Point2f[] nextPts,
        byte[] status,
        float[]? err,
        Size winSize,
        int maxLevel,
        TermCriteria criteria,
        OpticalFlowFlags flags = OpticalFlowFlags.None,
        double minEigThreshold = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(prevPyramid);
        ArgumentNullException.ThrowIfNull(nextPyramid);
        Calc(null, prevPyramid, null, nextPyramid, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);
    }

    /// <summary>
    /// Allocating form of <see cref="CalcOpticalFlowPyrLK(Mat{byte}, Mat{byte}, Point2f[], Point2f[], byte[], float[], Size, int, TermCriteria, OpticalFlowFlags, double)"/>
    /// with the OpenCV defaults: window 21 × 21, 3 levels, <see cref="DefaultCriteria"/>.
    /// </summary>
    /// <param name="prevImg">First 8-bit image.</param>
    /// <param name="nextImg">Second 8-bit image.</param>
    /// <param name="prevPts">Points to track.</param>
    /// <param name="initialNextPts">Initial estimates; required with <see cref="OpticalFlowFlags.UseInitialFlow"/>, otherwise ignored.</param>
    /// <param name="flags">Operation flags.</param>
    /// <param name="minEigThreshold">Minimum eigenvalue threshold.</param>
    /// <returns>New arrays of positions, status and error.</returns>
    public static OpticalFlowResult CalcOpticalFlowPyrLK(
        Mat<byte> prevImg,
        Mat<byte> nextImg,
        Point2f[] prevPts,
        Point2f[]? initialNextPts = null,
        OpticalFlowFlags flags = OpticalFlowFlags.None,
        double minEigThreshold = 1e-4)
        => CalcOpticalFlowPyrLK(prevImg, nextImg, prevPts, initialNextPts, new Size(21, 21), 3, DefaultCriteria, flags, minEigThreshold);

    /// <summary>
    /// Allocating form of <see cref="CalcOpticalFlowPyrLK(Mat{byte}, Mat{byte}, Point2f[], Point2f[], byte[], float[], Size, int, TermCriteria, OpticalFlowFlags, double)"/>.
    /// </summary>
    /// <returns>New arrays of positions, status and error.</returns>
    public static OpticalFlowResult CalcOpticalFlowPyrLK(
        Mat<byte> prevImg,
        Mat<byte> nextImg,
        Point2f[] prevPts,
        Point2f[]? initialNextPts,
        Size winSize,
        int maxLevel,
        TermCriteria criteria,
        OpticalFlowFlags flags = OpticalFlowFlags.None,
        double minEigThreshold = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(prevPts);
        var next = new Point2f[prevPts.Length];
        if ((flags & OpticalFlowFlags.UseInitialFlow) != 0)
        {
            ArgumentNullException.ThrowIfNull(initialNextPts);
            if (initialNextPts.Length != prevPts.Length)
            {
                throw new ArgumentException("The initial points must have the length of the previous points.", nameof(initialNextPts));
            }

            initialNextPts.CopyTo(next, 0);
        }

        var status = new byte[prevPts.Length];
        var err = new float[prevPts.Length];
        CalcOpticalFlowPyrLK(prevImg, nextImg, prevPts, next, status, err, winSize, maxLevel, criteria, flags, minEigThreshold);
        return new OpticalFlowResult(next, status, err);
    }

    private static void Calc(
        Mat<byte>? prevImg,
        OpticalFlowPyramid? prevPyr,
        Mat<byte>? nextImg,
        OpticalFlowPyramid? nextPyr,
        Point2f[] prevPts,
        Point2f[] nextPts,
        byte[] status,
        float[]? err,
        Size winSize,
        int maxLevel,
        TermCriteria criteria,
        OpticalFlowFlags flags,
        double minEigThreshold)
    {
        var rentals = new List<IDisposable>();
        try
        {
            CalcCore(prevImg, prevPyr, nextImg, nextPyr, prevPts, nextPts, status, err, winSize, maxLevel, criteria, flags, minEigThreshold, rentals);
        }
        finally
        {
            foreach (IDisposable rental in rentals)
            {
                rental.Dispose();
            }
        }
    }

    /// <param name="rentals">Receives the pooled buffers of the temporary pyramids and derivatives.</param>
    private static void CalcCore(
        Mat<byte>? prevImg,
        OpticalFlowPyramid? prevPyr,
        Mat<byte>? nextImg,
        OpticalFlowPyramid? nextPyr,
        Point2f[] prevPts,
        Point2f[] nextPts,
        byte[] status,
        float[]? err,
        Size winSize,
        int maxLevel,
        TermCriteria criteria,
        OpticalFlowFlags flags,
        double minEigThreshold,
        List<IDisposable> rentals)
    {
        ArgumentNullException.ThrowIfNull(prevPts);
        ArgumentNullException.ThrowIfNull(nextPts);
        ArgumentNullException.ThrowIfNull(status);
        ArgumentOutOfRangeException.ThrowIfNegative(maxLevel);
        RequireWindow(winSize);
        int npoints = prevPts.Length;
        if (nextPts.Length != npoints || status.Length != npoints || (err != null && err.Length != npoints))
        {
            throw new ArgumentException("nextPts, status and err must have the length of prevPts.");
        }

        if (npoints == 0)
        {
            return;
        }

        Array.Fill(status, (byte)1);

        Mat<byte>[] prevLevels;
        Mat<short>[]? prevDerivs = null;
        Mat<byte>[] nextLevels;
        if (prevPyr != null)
        {
            prevLevels = prevPyr.Images;
            prevDerivs = prevPyr.Derivatives;
            RequirePadding(prevPyr, winSize, nameof(prevPyr));
            maxLevel = Math.Min(maxLevel, prevLevels.Length - 1);
        }
        else
        {
            prevLevels = [];
        }

        if (nextPyr != null)
        {
            nextLevels = nextPyr.Images;
            RequirePadding(nextPyr, winSize, nameof(nextPyr));
            maxLevel = Math.Min(maxLevel, nextLevels.Length - 1);
        }
        else
        {
            nextLevels = [];
        }

        if (prevPyr == null)
        {
            prevLevels = LkPyramidBuilder.Build(prevImg!, winSize, maxLevel, false, BorderType.Reflect101, BorderType.Constant, true, rentals).Images;
            maxLevel = prevLevels.Length - 1;
        }

        if (nextPyr == null)
        {
            nextLevels = LkPyramidBuilder.Build(nextImg!, winSize, maxLevel, false, BorderType.Reflect101, BorderType.Constant, true, rentals).Images;
            maxLevel = nextLevels.Length - 1;
        }

        int maxCount = (criteria.Type & TermCriteriaType.MaxIter) == 0 ? 30 : Math.Clamp(criteria.MaxCount, 0, 100);
        double eps = (criteria.Type & TermCriteriaType.Eps) == 0 ? 0.01 : Math.Min(Math.Max(criteria.Epsilon, 0.0), 10.0);
        eps *= eps;

        for (int level = maxLevel; level >= 0; level--)
        {
            Mat<byte> pl = prevLevels[level], nl = nextLevels[level];
            if (pl.Width != nl.Width || pl.Height != nl.Height || pl.Channels != nl.Channels)
            {
                throw new ArgumentException("The previous and next levels must have the same size and channel count.");
            }

            Mat<short> derivI = prevDerivs != null
                ? prevDerivs[level]
                : LkPyramidBuilder.Derivative(pl, winSize, BorderType.Constant, rentals);
            var lv = new LkTracker.Level(
                pl,
                derivI,
                nl,
                winSize,
                maxCount,
                eps,
                level,
                maxLevel,
                (flags & OpticalFlowFlags.UseInitialFlow) != 0,
                (flags & OpticalFlowFlags.LkGetMinEigenvals) != 0,
                (float)minEigThreshold);
            LkTracker.Run(lv, prevPts, nextPts, status, err);
        }
    }

    private static void RequireWindow(Size winSize)
    {
        if (winSize.Width <= 2 || winSize.Height <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(winSize), winSize, "Both window sides must be greater than 2.");
        }
    }

    private static void RequirePadding(OpticalFlowPyramid pyr, Size winSize, string name)
    {
        for (int i = 0; i < pyr.Images.Length; i++)
        {
            if (!HasPadding(pyr.Images[i], winSize) || (pyr.Derivatives != null && !HasPadding(pyr.Derivatives[i], winSize)))
            {
                throw new ArgumentException("The pyramid padding is smaller than the window size.", name);
            }
        }
    }

    private static bool HasPadding<T>(Mat<T> m, Size winSize)
        where T : unmanaged
    {
        (Size whole, Point ofs) = LkBorders.LocateRoi(m);
        return ofs.X >= winSize.Width && ofs.Y >= winSize.Height
            && ofs.X + m.Width + winSize.Width <= whole.Width
            && ofs.Y + m.Height + winSize.Height <= whole.Height;
    }
}
