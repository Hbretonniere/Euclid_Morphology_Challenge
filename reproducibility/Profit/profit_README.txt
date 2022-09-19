Results Information

The below just refers to the standard (original fields), i.e. those inside the euclidtest.oats.inaf.it/webdav/mer/V2/ folder. I will consider processing the DEEPFIELD data at a later date (but this might not be possible).

All runs were carried out on a 24 core (48 thread) cluster comprised of two Intel Xeon E5-2690 v3 (Haswell) 12-core CPUs. Times below depict the total processing user time (e.g. this is roughly how long the fitting would have taken in human time if carried out on a single core). The CPU time spent processing is therefore always less than this number (although in practice the code is usually using clode to 100% of all CPUs).

The fitting process involved a few steps. I run ProFound on a 500 x 500 cutout centred on the target source to create the fitting segmentation map and find nearby sources that require simultaneous modelling (Next output discussed below). The ProFound output also provides some reasonable initial guess for the profile solution. The failed cases where output are all NA reflect the times when ProFound failed to find a segment in the vicinity of the target. Since I work with the FITS images on disk (only loading the subset of data required, not the whole image) the memory footprint is very low- usually 10s MB.

The ProFit fitting is then carried out using my Highlander package that combines a genetic algorithm step with a CHARM MCMC process twice. Each is run for 100 steps (where model realisations are modified by the number of free parameters also). The "best"" solution is simply the most +ve log-posterior (LP below) combination of parameters found on any of the runs (usually the last). Errors are estimated from the final MCMC run in either linear of log space (depending on the parameter). In principle we have access to the full covariance matrix, but I have only provided the marginalised errors below (as per the request).

With the above setup, the typical Euclic single Sersic profile fit a bit under around a minute (mean), and a double Sersic fit around two minutes (mean). This includes the ProFound stage which takes about a second usually. In practice the vast majority fo the time is spent doing the ProFit optimisation.

Code Version / (Info) / [Reference] / {Source}:

R: v3.6.1 (The base programming language used) [R Development Core Team, 2016, R Foundation for Statistical Computing, Vienna] {https://www.r-project.org}
ProFit: v1.4.0 (This is the main fitting engine) [Robotham A. et al, 2017, MNRAS, 466, 1513] {GitHub ICRAR/ProFit}
ProFound: v1.14.4 (This is a generic source finder, and build the segmentation maps for ProFit) [Robotham A. et al, 2018, MNRAS, 476, 3137] {GitHub asgr/ProFound}
Highlander: v0.1.6 (This is the hybrid genetic and MCMC optimiser) [No ref] {GitHub asgr/Highlander}

The codes ProFit, ProFOund and Highlander are all written by Aaron Robotham.

File naming scheme:

combine_Run_[FIELD]_[FILTER]_[FIT].csv

FIELD = [0,1,2,3,4]
FILTER = [vis, u_lsst, g_lsst, r_lsst, i_lsst, z_lsst, y_nir, j_nir, h_nir]
FIT = [Sf, Rf, B4D1, BfD1]

Sf = Free Sersic index fit to the single Sersic mock images
Rf = Free Sersic index fit to the realistic morphology mock images
B4D1 = Fixed Sersic index bulge (nser=4), fixed index disk (nser=1)
BfD1 = Free Sersic index bulge, fixed index disk (nser=1)

Column names explained:

loc = Input row location (5 sigma catalogue)
X = X position in frame
Y = Y position in frame
time.elapsed = processing time for fit [minutes]
LP = Log posterior of the fit (more +ve is better)
Npix = Number of pixels used for the fit
RedChi2 = Reduced Chi Square of the fit (where 1 is ideal)
Next = Number of additional profiles modelled along with the target profile
par.sersic.mag? = Magnitude: ?='' for Sf/Rf / ?=1 for bulge / ?=2 for disk [ab mag]
par.sersic.re? = Re: ?='' for Sf/Rf / ?=1 for bulge / ?=2 for disk [log10(Re/asec)]
par.sersic.nser1 = Sersic index of bulge [log10(nser)]
par.sersic.ang? = Orientation: ?='' for Sf/Rf / ?=1 for bulge / ?=2 for disk [degrees]
par.sersic.axrat? = Axial ratio: ?='' for Sf/Rf / ?=1 for bulge / ?=2 for disk [log10(axrat)]
err.sersic.mag? = Magnitude error [ab mag]
err.sersic.re? = Re error [dex]
err.sersic.nser? = Sersic index error [dex]
err.sersic.ang? = Orientation error [degrees]
err.sersic.axrat? = Axial ratio error [dex]

Files present:

Single Sersic  on Single Sersic Mocks (visual, field 0-4):

combine_Run_0_vis_Sf.csv In: 77855 Out: 76572 Hrs: 1382
combine_Run_1_vis_Sf.csv In: 78099 Out: 75748 Hrs: 933
combine_Run_2_vis_Sf.csv In: 77981 Out: 76851 Hrs: 960
combine_Run_3_vis_Sf.csv In: 77522 Out: 76366 Hrs: 950
combine_Run_4_vis_Sf.csv In: 77915 Out: 76211 Hrs: 1151

Single Sersic on Realistic Morphology Mocks (visual, field 0-4):

combine_Run_0_vis_Rf.csv In: 77855 Out: 70650 Hrs: 1641
combine_Run_1_vis_Rf.csv In: 78099 Out: 69853 Hrs: 1454
combine_Run_2_vis_Rf.csv In: 77981 Out: 70862 Hrs: 1467
combine_Run_3_vis_Rf.csv In: 77522 Out: 70398 Hrs: 1582
combine_Run_4_vis_Rf.csv In: 77915 Out: 70321 Hrs: 1487

Double Sersic with Fixed Bulge (visual, field 0-4):

combine_Run_0_vis_B4D1.csv In: 77855 Out: 76785 Hrs: 2884
combine_Run_1_vis_B4D1.csv In: 78099 Out: 75996 Hrs: 2858
combine_Run_2_vis_B4D1.csv In: 77981 Out: 77119 Hrs: 2898
combine_Run_3_vis_B4D1.csv In: 77522 Out: 76606 Hrs: 2874
combine_Run_4_vis_B4D1.csv In: 77915 Out: 76483 Hrs: 2885

Double Sersic with Free Bulge (visual, field 0-4):

combine_Run_0_vis_BfD1.csv In: 77855 Out: 76777 Hrs: 3659
combine_Run_1_vis_BfD1.csv In: 78099 Out: 75995 Hrs: 3628
combine_Run_2_vis_BfD1.csv In: 77981 Out: 77112 Hrs: 3691
combine_Run_3_vis_BfD1.csv In: 77522 Out: 76600 Hrs: 3672
combine_Run_4_vis_BfD1.csv In: 77915 Out: 76485 Hrs: 3677

Double Sersic with Fixed Bulge (Multi-band, field 0):

combine_Run_0_u_lsst_B4D1.csv In: 77855 Out: 17127 Hrs: 1112
combine_Run_0_g_lsst_B4D1.csv In: 77855 Out: 49431 Hrs: 3350
combine_Run_0_r_lsst_B4D1.csv In: 77855 Out: 46432 Hrs: 2738 
combine_Run_0_i_lsst_B4D1.csv In: 77855 Out: 47648 Hrs: 2641
combine_Run_0_z_lsst_B4D1.csv In: 77855 Out: 48666 Hrs: 2650 
combine_Run_0_y_nir_B4D1.csv In: 77855 Out: 70402 Hrs: 2166 
combine_Run_0_j_nir_B4D1.csv In: 77855 Out: 72051 Hrs: 2414
combine_Run_0_h_nir_B4D1.csv In: 77855 Out: 72555 Hrs: 2688

Double Sersic with Free Bulge (Multi-band, field 0):

combine_Run_0_u_lsst_BfD1.csv In: 77855 Out: 17127 Hrs: 1476
combine_Run_0_g_lsst_BfD1.csv In: 77855 Out: 49425 Hrs: 4214
combine_Run_0_r_lsst_BfD1.csv In: 77855 Out: 46422 Hrs: 3771
combine_Run_0_i_lsst_BfD1.csv In: 77855 Out: 47639 Hrs: 3697
combine_Run_0_z_lsst_BfD1.csv In: 77855 Out: 48654 Hrs: 3702
combine_Run_0_y_nir_BfD1.csv In: 77855 Out: 70389 Hrs: 3222
combine_Run_0_j_nir_BfD1.csv In: 77855 Out: 72045 Hrs: 3524
combine_Run_0_h_nir_BfD1.csv In: 77855 Out: 72549 Hrs: 3850

Summary stats for all runs:

36 fields, 2,803,168 objects, 2,348,352 fits, 92,948 Hrs (mean of 2.4 minutes per fit)

DEEPFIELD #Don't have enough HPC time to do this
