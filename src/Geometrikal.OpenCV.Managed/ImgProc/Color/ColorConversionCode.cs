namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Colour conversion code, mirroring OpenCV <c>ColorConversionCodes</c> (same numeric values).
/// Names that share a value are equivalent.
/// </summary>
public enum ColorConversionCode
{
    /// <summary>Adds an alpha channel (255, 65535 or 1.0).</summary>
    Bgr2Bgra = 0,

    /// <summary>Adds an alpha channel.</summary>
    Rgb2Rgba = 0,

    /// <summary>Removes the alpha channel.</summary>
    Bgra2Bgr = 1,

    /// <summary>Removes the alpha channel.</summary>
    Rgba2Rgb = 1,

    /// <summary>Swaps red and blue and adds an alpha channel.</summary>
    Bgr2Rgba = 2,

    /// <summary>Swaps red and blue and adds an alpha channel.</summary>
    Rgb2Bgra = 2,

    /// <summary>Swaps red and blue and removes the alpha channel.</summary>
    Rgba2Bgr = 3,

    /// <summary>Swaps red and blue and removes the alpha channel.</summary>
    Bgra2Rgb = 3,

    /// <summary>Swaps red and blue.</summary>
    Bgr2Rgb = 4,

    /// <summary>Swaps red and blue.</summary>
    Rgb2Bgr = 4,

    /// <summary>Swaps red and blue and keeps the alpha channel.</summary>
    Bgra2Rgba = 5,

    /// <summary>Swaps red and blue and keeps the alpha channel.</summary>
    Rgba2Bgra = 5,

    /// <summary>BGR or BGRA to gray.</summary>
    Bgr2Gray = 6,

    /// <summary>RGB or RGBA to gray.</summary>
    Rgb2Gray = 7,

    /// <summary>Gray to BGR.</summary>
    Gray2Bgr = 8,

    /// <summary>Gray to RGB.</summary>
    Gray2Rgb = 8,

    /// <summary>Gray to BGRA.</summary>
    Gray2Bgra = 9,

    /// <summary>Gray to RGBA.</summary>
    Gray2Rgba = 9,

    /// <summary>BGRA to gray.</summary>
    Bgra2Gray = 10,

    /// <summary>RGBA to gray.</summary>
    Rgba2Gray = 11,

    /// <summary>BGR to CIE XYZ (D65).</summary>
    Bgr2Xyz = 32,

    /// <summary>RGB to CIE XYZ (D65).</summary>
    Rgb2Xyz = 33,

    /// <summary>CIE XYZ to BGR.</summary>
    Xyz2Bgr = 34,

    /// <summary>CIE XYZ to RGB.</summary>
    Xyz2Rgb = 35,

    /// <summary>BGR to YCrCb.</summary>
    Bgr2YCrCb = 36,

    /// <summary>RGB to YCrCb.</summary>
    Rgb2YCrCb = 37,

    /// <summary>YCrCb to BGR.</summary>
    YCrCb2Bgr = 38,

    /// <summary>YCrCb to RGB.</summary>
    YCrCb2Rgb = 39,

    /// <summary>BGR to HSV; 8-bit hue in 0..179, float hue in degrees.</summary>
    Bgr2Hsv = 40,

    /// <summary>RGB to HSV; 8-bit hue in 0..179, float hue in degrees.</summary>
    Rgb2Hsv = 41,

    /// <summary>sRGB BGR to CIE L*a*b*.</summary>
    Bgr2Lab = 44,

    /// <summary>sRGB RGB to CIE L*a*b*.</summary>
    Rgb2Lab = 45,

    /// <summary>HSV to BGR; 8-bit hue in 0..179.</summary>
    Hsv2Bgr = 54,

    /// <summary>HSV to RGB; 8-bit hue in 0..179.</summary>
    Hsv2Rgb = 55,

    /// <summary>CIE L*a*b* to sRGB BGR.</summary>
    Lab2Bgr = 56,

    /// <summary>CIE L*a*b* to sRGB RGB.</summary>
    Lab2Rgb = 57,

    /// <summary>BGR to HSV; 8-bit hue in 0..255.</summary>
    Bgr2HsvFull = 66,

    /// <summary>RGB to HSV; 8-bit hue in 0..255.</summary>
    Rgb2HsvFull = 67,

    /// <summary>HSV to BGR; 8-bit hue range 255.</summary>
    Hsv2BgrFull = 70,

    /// <summary>HSV to RGB; 8-bit hue range 255.</summary>
    Hsv2RgbFull = 71,

    /// <summary>Linear BGR to CIE L*a*b*.</summary>
    Lbgr2Lab = 74,

    /// <summary>Linear RGB to CIE L*a*b*.</summary>
    Lrgb2Lab = 75,

    /// <summary>CIE L*a*b* to linear BGR.</summary>
    Lab2Lbgr = 78,

    /// <summary>CIE L*a*b* to linear RGB.</summary>
    Lab2Lrgb = 79,
}
