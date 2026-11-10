// This implementation is ported from the OpenCV text drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing_text.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing_text.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Font identifiers, mirroring OpenCV's <c>cv::HersheyFonts</c>.
///
/// <para>OpenCV 5.0 maps each identifier to a TrueType font (Rubik), a size and
/// a weight (<c>hersheyToTruetype</c>). OpenCV 5.0 does not contain Hershey glyph
/// tables.</para>
/// </summary>
[Flags]
public enum HersheyFonts
{
    /// <summary>Normal size sans-serif font (<c>FONT_HERSHEY_SIMPLEX</c>).</summary>
    Simplex = 0,

    /// <summary>Small size sans-serif font (<c>FONT_HERSHEY_PLAIN</c>).</summary>
    Plain = 1,

    /// <summary>Normal size sans-serif font with more weight (<c>FONT_HERSHEY_DUPLEX</c>).</summary>
    Duplex = 2,

    /// <summary>Normal size serif font (<c>FONT_HERSHEY_COMPLEX</c>).</summary>
    Complex = 3,

    /// <summary>Normal size serif font with more weight (<c>FONT_HERSHEY_TRIPLEX</c>).</summary>
    Triplex = 4,

    /// <summary>Small version of <see cref="Complex"/> (<c>FONT_HERSHEY_COMPLEX_SMALL</c>).</summary>
    ComplexSmall = 5,

    /// <summary>Hand-writing style font (<c>FONT_HERSHEY_SCRIPT_SIMPLEX</c>).</summary>
    ScriptSimplex = 6,

    /// <summary>Hand-writing style font with more weight (<c>FONT_HERSHEY_SCRIPT_COMPLEX</c>).</summary>
    ScriptComplex = 7,

    /// <summary>Italic flag (<c>FONT_ITALIC</c>). Combine it with one of the other values.</summary>
    Italic = 16,
}
