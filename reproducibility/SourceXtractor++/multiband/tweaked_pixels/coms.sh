#!/bin/bash

### Before running this script, you should have run
### julia setup.jl first in order to generate images, symlinks, etc...


## name of SE++ input python script
pysc=mod3band1.py
## Hardcoded number of threads
Nth=32
## Basename for output files
opfx=fair

export MKL_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC="FALSE"
sourcextractor++ --detection-image ds_vis.fits  \
		 --weight-type=rms \
		 --weight-absolute 1 \
		 --weight-image ds_vis.rms.fits  \
		 --psf-filename psf_vis.fits \
		 --python-config-file $pysc \
		 --output-catalog-filename ${opfx}_vis.cat \
		 --output-properties=PixelCentroid,WorldCentroid,SourceIDs,GroupInfo,SourceFlags,FlexibleModelFitting,AperturePhotometry,AutoPhotometry,FluxRadius,SNRRatio \
		 --magnitude-zero-point 23.9 \
		 --segmentation-algorithm LUTZ \
		 --segmentation-lutz-window-size 2048 \
		 --thread-count $Nth \
		 --sampling-scale-factor 1 \
		 --background-cell-size 8192 \
		 --background-value=0.0 \
		 --core-minimum-area 10 \
		 --partition-corethreshold 1 \
		 --segmentation-filter gauss2.4.fits \
		 --detection-minimum-area 3 \
		 --detection-image-interpolation=0 \
		 --detection-image-saturation 0 \
		 --detection-image-gain 0 \
		 --use-cleaning \
		 --cleaning-minimum-area 10 \
		 --grouping-algorithm split \
		 --partition-multithreshold yes \
		 --partition-minimum-area 10 \
		 --partition-minimum-contrast 0.005 \
		 --output-flush-size 1000

### then, use Catalog_MatchCompare.jl to compare output with ground truth.
