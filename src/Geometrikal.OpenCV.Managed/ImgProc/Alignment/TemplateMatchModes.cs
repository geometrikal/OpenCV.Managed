namespace Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Comparison methods of <c>TemplateMatching.MatchTemplate</c>,
/// as OpenCV <c>TemplateMatchModes</c>. <c>T</c> is the template, <c>I</c> is the image, and the
/// sums run over the template window.
/// </summary>
public enum TemplateMatchModes
{
    /// <summary><c>R = Σ (T - I)²</c>.</summary>
    SqDiff = 0,

    /// <summary><c>R = Σ (T - I)² / sqrt(Σ T² · Σ I²)</c>.</summary>
    SqDiffNormed = 1,

    /// <summary><c>R = Σ T · I</c>.</summary>
    CCorr = 2,

    /// <summary><c>R = Σ T · I / sqrt(Σ T² · Σ I²)</c>.</summary>
    CCorrNormed = 3,

    /// <summary><c>R = Σ T' · I'</c>, with the window means removed.</summary>
    CCoeff = 4,

    /// <summary><c>R = Σ T' · I' / sqrt(Σ T'² · Σ I'²)</c>.</summary>
    CCoeffNormed = 5,
}
