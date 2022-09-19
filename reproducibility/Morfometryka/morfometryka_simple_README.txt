We provide fits tables for each Single-Sersic FIELD. Each field contains:

    'ID'
    'X' and 'Y' as per inputs
    'nFit2D' and 'nerr' -- Sersic index and its uncertainty
    'RnFit2D' and 'Rnerr' -- Half-light radius and its uncertainty
    'Rp' -- Petrosian Radius
    'LT' and 'LTerr' -- Total flux within the Petrosian Region and its uncertainty
    'InFit2D' and 'Inerr' -- Flux at RnFit2D and its uncertainty
    'qFit2D' and 'qerr' -- Axis ratio and its uncertainty
    'PAFit2D' and 'PAerr' -- Position Angle and its uncertainty

The tables contain also four quality flags:

    QF_CONVERGENCE -- True when Morfometryka failed to converge for a profile fit.
    QF_FAULTY_RUN -- True when Unexpected error found during the run. Possible: No segmentation region, no detectable source. 
    QF_NOT\_SEXTRACTED -- True when SExtractor did not find the source in the ASSOC catalog.
    QF_INCOMPLETE\_BADCUTOUT -- True when source size bigger than image. Faulty profile or bad cutout.
