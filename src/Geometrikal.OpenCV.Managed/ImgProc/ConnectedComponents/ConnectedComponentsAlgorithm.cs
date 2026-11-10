// This enumeration mirrors OpenCV's ConnectedComponentsAlgorithmsTypes.
//
// Original source:
//   modules/imgproc/include/opencv2/imgproc.hpp (ConnectedComponentsAlgorithmsTypes)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Connected components labeling algorithm, with OpenCV <c>CCL_*</c> numeric values. The
/// algorithm selects the label numbering order; the partition into components is the same for
/// every value.
/// </summary>
/// <remarks>
/// <para>With 8-connectivity, <see cref="Wu"/> and <see cref="Sauf"/> number the components in
/// raster order of their first pixel. All other values number the components in raster order of
/// their first 2x2 block (block row <c>y / 2</c>, block column <c>x / 2</c>).</para>
/// <para>With 4-connectivity, every value numbers the components in raster order of their first
/// pixel.</para>
/// </remarks>
public enum ConnectedComponentsAlgorithm
{
    /// <summary>Spaghetti for 8-connectivity, Spaghetti4C for 4-connectivity (<c>CCL_DEFAULT</c>).</summary>
    Default = -1,

    /// <summary>SAUF (<c>CCL_WU</c>).</summary>
    Wu = 0,

    /// <summary>BBDT for 8-connectivity, SAUF for 4-connectivity (<c>CCL_GRANA</c>).</summary>
    Grana = 1,

    /// <summary>Spaghetti for 8-connectivity, Spaghetti4C for 4-connectivity (<c>CCL_BOLELLI</c>).</summary>
    Bolelli = 2,

    /// <summary>SAUF (<c>CCL_SAUF</c>).</summary>
    Sauf = 3,

    /// <summary>BBDT for 8-connectivity, SAUF for 4-connectivity (<c>CCL_BBDT</c>).</summary>
    Bbdt = 4,

    /// <summary>Spaghetti for 8-connectivity, Spaghetti4C for 4-connectivity (<c>CCL_SPAGHETTI</c>).</summary>
    Spaghetti = 5,
}
