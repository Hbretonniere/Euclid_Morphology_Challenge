from glob import glob
import numpy as np
from sourcextractor.config import *

top = load_fits_images(
                       sorted(glob('../../FIELD4/Single_Sersic/ssersic_4_vis.fits')),
                       weights=sorted(glob('../../FIELD4/Single_Sersic/ssersic_4_vis.rms.fits')),
                       weight_type='rms',
                       constant_background = 0.0,
                       weight_absolute=1,
                       psfs = sorted(glob('../../PSFs/psf_vis_os045_high_nu.psf'))
                      )

#top.split(ByKeyword('BAND'))
mesgroup = MeasurementGroup(top)
set_max_iterations(250)
set_engine('levmar')
constant_background = 0.0
MAG_ZEROPOINT = 23.9

#all_apertures = []
#for img in mesgroup:
#    all_apertures.extend(add_aperture_photometry(img, [10, 30, 60] ) )
#    add_output_column('aper', all_apertures)

rad = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.1 * v, 10*v), RangeType.EXPONENTIAL))
X_rad = DependentParameter(lambda r: np.log(r), rad)
add_prior( X_rad, 1.0, 0.4)

sersic = FreeParameter(1.0, Range((0.3, 5.5), RangeType.LINEAR))
X_sersic = DependentParameter(lambda n: np.log( (n-0.25)/(30-n) ), sersic )
add_prior( X_sersic, -3.9, 0.6)

e1 = FreeParameter( 0.0, Range((-0.9999, 0.9999), RangeType.LINEAR)) 
e2 = FreeParameter( 0.0, Range((-0.9999, 0.9999), RangeType.LINEAR)) 
emod = DependentParameter( lambda x,y: np.sqrt( x*x + y*y ), e1, e2 )
angle = DependentParameter( lambda e1,e2 : 0.5*np.arctan2( e1, e2 ), e1, e2 )
angle_deg = DependentParameter(lambda x: np.fmod(x,np.pi/2.0)/np.pi*180.0, angle)
ratio = DependentParameter( lambda e : np.abs(1-e)/(1+e), emod )
add_prior( e1, 0.0, 0.3 )
add_prior( e2, 0.0, 0.3 )


x,y = get_pos_parameters()
ra,dec = get_world_position_parameters(x, y)
flux = get_flux_parameter()
mag = DependentParameter( lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux)
add_model(mesgroup, SersicModel(x, y, flux, rad, ratio, angle, sersic))

add_output_column('x', x)
add_output_column('y', y)
add_output_column('ra', ra)
add_output_column('dec', dec)
add_output_column('mag', mag)
add_output_column('radius', rad)
add_output_column('X_rad',X_rad)
add_output_column('ratio', ratio)
add_output_column('sersic', sersic)
add_output_column('X_sersic',X_sersic)
add_output_column('angle', angle)
add_output_column('angle_deg', angle_deg)
add_output_column('e1',e1)
add_output_column('e2',e2)
add_output_column('emod',emod)
