// This implementation is ported from the OpenCV text drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing_text.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing_text.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Font size functions of OpenCV 5.0 that do not need glyph data.
/// <c>putText</c> and <c>getTextSize</c> are not ported: OpenCV 5.0 renders
/// text with a TrueType engine, not with Hershey glyphs.
/// </summary>
public static class TextMetrics
{
    /// <summary>
    /// Returns the font scale for which the font size is
    /// <paramref name="pixelHeight"/> (OpenCV 5.0 <c>getFontScaleFromHeight</c>).
    /// The result is <c>pixelHeight / cvRound(100 / sf)</c>, where <c>sf</c> is the
    /// size factor of the font.
    /// </summary>
    /// <param name="fontFace">Font identifier. <see cref="HersheyFonts.Italic"/> is ignored.</param>
    /// <param name="pixelHeight">Required height in pixels.</param>
    /// <param name="thickness">Line thickness. OpenCV 5.0 uses it only to select the font weight.</param>
    /// <returns>The font scale.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="fontFace"/> is not a known font.</exception>
    public static double GetFontScaleFromHeight(HersheyFonts fontFace, int pixelHeight, int thickness = 1)
    {
        _ = thickness;
        int ttsize = TrueTypeSize(fontFace, 1.0);
        return (double)pixelHeight / ttsize;
    }

    /// <summary>
    /// Returns the TrueType pixel size of OpenCV 5.0 <c>hersheyToTruetype</c>:
    /// <c>cvRound(100 * fontScale / sf)</c>.
    /// </summary>
    internal static int TrueTypeSize(HersheyFonts fontFace, double fontScale)
    {
        double sf = (fontFace & ~HersheyFonts.Italic) switch
        {
            HersheyFonts.Plain => 6.6,
            HersheyFonts.Simplex or HersheyFonts.Duplex or HersheyFonts.Complex or HersheyFonts.Triplex => 3.7,
            HersheyFonts.ComplexSmall => 4.6,
            HersheyFonts.ScriptSimplex or HersheyFonts.ScriptComplex => 4.0,
            _ => throw new ArgumentOutOfRangeException(nameof(fontFace), fontFace, "Unknown font."),
        };

        return (int)Math.Round(100 * fontScale / sf, MidpointRounding.ToEven);
    }
}
