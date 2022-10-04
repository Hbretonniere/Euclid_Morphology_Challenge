ln -sf /raid/data/euclid_morpho_challenge/PSFs/nirHnew11jun.swarp.resamp_centered.norm.fits psf_h_nir.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/nirJnew12jun.swarp.resamp_centered.norm.fits psf_j_nir.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/nirYnew12jun.swarp.resamp_centered.norm.fits psf_y_nir.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/psf_lsst_g12jun.swarp.resamp_centered.norm.fits psf_g_lsst.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/psf_lsst_i12jun.swarp.resamp_centered.norm.fits psf_i_lsst.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/psf_lsst_r12jun.swarp.resamp_centered.norm.fits psf_r_lsst.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/psf_lsst_u12jun.swarp.resamp_centered.norm.fits psf_u_lsst.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/psf_lsst_z12jun.swarp.resamp_centered.norm.fits psf_z_lsst.fits
ln -sf /raid/data/euclid_morpho_challenge/PSFs/psf_visSC3ovs6_centered_realpixscale.norm.fits psf_vis.fits


imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_g_lsst.rms.fits" 2 mul ds_g_lsst.rms.fits
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_h_nir.rms.fits" 3 mul ds_h_nir.rms.fits
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_i_lsst.rms.fits" 2 mul ds_i_lsst.rms.fits
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_j_nir.rms.fits" 3 mul ds_j_nir.rms.fits
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_r_lsst.rms.fits" 2 mul ds_r_lsst.rms.fits
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_u_lsst.rms.fits" 2 mul ds_u_lsst.rms.fits
ln -sf "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_vis.rms.fits" ds_vis.rms.fits 
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_y_nir.rms.fits" 3 mul ds_y_nir.rms.fits
imarith "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_z_lsst.rms.fits" 2 mul ds_z_lsst.rms.fits


#for i in  `\ls ds*fits | grep -v rms` ; do rm  $i ; done
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_g_lsst.fits" ds_g_lsst.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_h_nir.fits" ds_h_nir.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_i_lsst.fits" ds_i_lsst.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_j_nir.fits" ds_j_nir.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_r_lsst.fits" ds_r_lsst.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_u_lsst.fits" ds_u_lsst.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_vis.fits" ds_vis.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_y_nir.fits" ds_y_nir.fits
cp "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic/dsersic_0_z_lsst.fits" ds_z_lsst.fits

julia add_band_key.jl
