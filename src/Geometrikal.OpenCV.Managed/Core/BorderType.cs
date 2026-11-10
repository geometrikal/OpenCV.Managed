namespace Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Pixel extrapolation method used when an operation samples outside the
/// image, mirroring OpenCV's <c>BorderTypes</c>. Values match the OpenCV
/// constants. Using letters to denote pixels and <c>|</c> for the image edge:
/// <list type="bullet">
/// <item><description><see cref="Constant"/>: <c>iiiiii | abcdefgh | iiiiii</c> with a caller-supplied value <c>i</c>.</description></item>
/// <item><description><see cref="Replicate"/>: <c>aaaaaa | abcdefgh | hhhhhh</c>.</description></item>
/// <item><description><see cref="Reflect"/>: <c>fedcba | abcdefgh | hgfedc</c>.</description></item>
/// <item><description><see cref="Wrap"/>: <c>cdefgh | abcdefgh | abcdef</c>.</description></item>
/// <item><description><see cref="Reflect101"/>: <c>gfedcb | abcdefgh | gfedcb</c> (edge pixel not duplicated; OpenCV's default for most filters).</description></item>
/// </list>
/// <see cref="Transparent"/> is accepted only by the warps and remap; other operations throw.
/// The <c>BORDER_ISOLATED</c> flag is not supported; region-of-interest views are always treated
/// as isolated images.
/// </summary>
public enum BorderType
{
    /// <summary>Pad with a constant value (<c>BORDER_CONSTANT</c>).</summary>
    Constant = 0,

    /// <summary>Replicate the edge pixel (<c>BORDER_REPLICATE</c>).</summary>
    Replicate = 1,

    /// <summary>Mirror including the edge pixel (<c>BORDER_REFLECT</c>).</summary>
    Reflect = 2,

    /// <summary>Wrap around to the opposite edge (<c>BORDER_WRAP</c>).</summary>
    Wrap = 3,

    /// <summary>Mirror excluding the edge pixel (<c>BORDER_REFLECT_101</c>).</summary>
    Reflect101 = 4,

    /// <summary>
    /// Leave destination pixels that sample outside the source unchanged (<c>BORDER_TRANSPARENT</c>).
    /// Supported only by <c>Geometry.WarpAffine</c>, <c>Geometry.WarpPerspective</c> and
    /// <c>Geometry.Remap</c>.
    /// </summary>
    Transparent = 5,
}
