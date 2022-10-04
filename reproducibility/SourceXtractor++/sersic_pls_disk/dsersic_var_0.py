from glob import glob
import numpy as np
from sourcextractor.config import *

top = load_fits_images(
    sorted(glob("../../FIELD0/Double_Sersic/dsersic_0_vis.fits")),
    psfs=sorted(glob("../../PSFs/psf_vis_os045_high_nu.psf")),
    weights=sorted(glob("../../FIELD0/Double_Sersic/dsersic_0_vis.rms.fits")),
    constant_background = 0.0,
    weight_absolute=1,
    weight_type='rms'
)

#top.split(ByKeyword('BAND'))
mesgroup = MeasurementGroup(top)
set_max_iterations(350)
constant_background = 0.0
MAG_ZEROPOINT = 23.9

## add the apertures
#all_apertures = []
#for img in mesgroup:
#    all_apertures.extend(add_aperture_photometry(img, [10, 30, 60] ) )
#    add_output_column('aper', all_apertures)

# load and execute the general disk+bulge model
exec(open("disk_p_sersic.py").read())
