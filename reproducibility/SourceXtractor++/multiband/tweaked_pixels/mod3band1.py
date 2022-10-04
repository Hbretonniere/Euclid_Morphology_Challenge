from glob import glob
import numpy as np
from sourcextractor.config import *

#### to be tuned by hand ############
fit_case = 'B+D'  # can be 'psf', 'B+D' or 'sersic'
s1 = 'sub_'       # could be "" or "sub_"
s2 = ''           # could be "" or "new_"
s1 = ''
#####################################

imas=['ds_u_lsst.fits', 'ds_g_lsst.fits', 'ds_r_lsst.fits', 'ds_i_lsst.fits', 'ds_z_lsst.fits', 'ds_vis.fits', 'ds_y_nir.fits', 'ds_j_nir.fits', 'ds_h_nir.fits' ]
weis=[ 'ds_u_lsst.rms.fits', 'ds_g_lsst.rms.fits', 'ds_r_lsst.rms.fits', 'ds_i_lsst.rms.fits', 'ds_z_lsst.rms.fits', 'ds_vis.rms.fits', 'ds_y_nir.rms.fits', 'ds_j_nir.rms.fits', 'ds_h_nir.rms.fits' ]
psfs=[ 'psf_u_lsst.fits', 'psf_g_lsst.fits', 'psf_r_lsst.fits', 'psf_i_lsst.fits', 'psf_z_lsst.fits', 'psf_vis.fits', 'psf_y_nir.fits', 'psf_j_nir.fits', 'psf_h_nir.fits' ]
imas = list( map( lambda x: s1+s2 + x, imas) )
weis = list( map( lambda x: s1+s2 + x, weis) )

MAG_ZEROPOINT = 23.9
top = load_fits_images( imas, psfs=psfs, weights=weis, 
                        constant_background = 0.0,
                        weight_absolute=1,
                        weight_type='rms'
                       )

top.split(ByKeyword('BAND'))

mesgroup = MeasurementGroup(top)
set_max_iterations(250)

all_apertures = []
i=1
for band,img in mesgroup:
    all_apertures.extend(add_aperture_photometry(img, [10, 30, 60] ) )
    add_output_column('aper_'+str(band), all_apertures)
    i+=1

x,y = get_pos_parameters()
ra,dec = get_world_position_parameters(x, y)

r_b = FreeParameter(lambda o: o.radius*0.5, Range(lambda v,o: (0.01*v,30*v), RangeType.EXPONENTIAL))
r_d = FreeParameter(lambda o: o.radius*2, Range(lambda v,o: (0.01*v,30*v), RangeType.EXPONENTIAL))
### from ground truth 
###       log10(r_d) ~ N(0.33,0.22)
###       log10(r_b) ~ 1.14 * log10 r_d -1.2 + N(0,0.38)  # captures the mild covariance between d and b radii
lrd =  DependentParameter( lambda y : np.log10(y), r_d )
add_prior( lrd, 0.33, 0.25 ) ## log10(rd) in pixels 
rel_size = DependentParameter( lambda x,y : np.log10(y)-(1.14*np.log10(x)-1.2), r_d, r_b ) 
add_prior( rel_size, 0.0, 0.4 )

angle = FreeParameter( lambda o: o.angle )
ratio_d = FreeParameter( 0.5, Range( (0.01, 1.0), RangeType.LINEAR) )
ratio_b = FreeParameter( 0.6, Range( (0.01, 1.0), RangeType.LINEAR) )
X_rd = DependentParameter( lambda q: np.log( (q-0.01)/(0.99-q) ), ratio_d ) ; add_prior( X_rd, 0.03, 1.0 )     
X_rb = DependentParameter( lambda q: np.log( (q-0.01)/(0.99-q) ), ratio_b ) ; add_prior( X_rb, 0.50, 1.1 )

add_output_column('x',x)
add_output_column('y',y)
add_output_column('disk_effR_px',r_d)
add_output_column('bulge_effR_px',r_b)
add_output_column('angle',angle)
add_output_column('disk_axr',ratio_d)
add_output_column('bulge_axr',ratio_b)
add_output_column('ra',ra)
add_output_column('dec',dec)
add_output_column('rel_s',rel_size)

i = 0
flux1 = {}
flux2 = {}
flux = {}
bt = {}
X_bt= {}
mag = {}
### sorted as LSST_u, g, r, i, z, VIS, Y, J, H
###  must be consistent with the way images are loaded!
X_bt_med = [ -3.23, -3.03, -2.86, -2.73, -2.65, -2.75, -2.53, -2.44, -2.36 ]
X_bt_wid = [ 1.74, 1.89, 2.08, 2.44, 2.45, 2.28, 2.62, 2.81, 2.94 ]

for band,group in mesgroup:
    flux[i] = get_flux_parameter()
    mag[i] = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux[i] )
    bt[i] = FreeParameter(0.1, Range((0.0,1.0), RangeType.LINEAR))
    flux1[i] = DependentParameter(lambda f, r: f*r, flux[i], bt[i] )
    flux2[i] = DependentParameter(lambda f, r: f*(1.0-r), flux[i], bt[i] )
    X_bt[i] = DependentParameter(lambda r: np.log( (r+0.01)/(1.01-r) ), bt[i] )
    add_prior( X_bt[i], X_bt_med[i], X_bt_wid[i] )
    add_model( group, ExponentialModel( x, y, flux2[i], r_d, ratio_d, angle) )
    add_model( group, DeVaucouleursModel( x, y, flux1[i], r_b, ratio_b, angle) )
    add_output_column('mag_'+str(band),mag[i])
    add_output_column('bt_'+str(band),bt[i])
    add_output_column('X_bt_'+str(band),X_bt[i])
    i += 1
