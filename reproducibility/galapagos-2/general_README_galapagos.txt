GALFITM/Galapagos catalogues README for catalogues.
THIS file contain general fitting information and comment on all catalogues provided.


I provide multiple catalogues for each object: 
- single-sersic fits for the fields which required single-sersic fits  - catalogues that contain BOTH single-sersic as well as bulge and disk values for the fields that required B/D fits. Single-Sersic fits are always done in galapagos (they are the starting point of the B/D fits), and hence can be provided with no additional cost.  - several catalogues matched to the different sigma values that you provided. *all are obviously the most complete ones, but even they miss some objects. This will mostly be due to detection issues, e.g. the objects are too faint for my simple SExtractor setup.  - in FIELD0, DS: catalogues for EACH BAND! NOTICE: These are NOT independent fits, THEY ARE ALL DONE IN PARALLEL, I SIMPLY EXTRACTED THE VALUES FOR EACH BAND FOR YOU. I highly recommend using the fits table provided instead (Marc knows how). Especially the flags for each fit are NOT independent, as ALL BAND are checked. E.g. you will find all parameters in one band to be perfectly fine, but the component should not be used. This could be because the values in a different band have been constrained. The decision whether a component is 'bright enough' has been done in the magnitude values in the VIS band ONLY!  - fits table as created by Galapagos directly. There are NO quality flags in those catalogues. If you want, I can help you with creating some. They, however, contain a LOT more information than the ascii files, please see the GALAPAGOS cat readme provided.  

All detections have been carried out in the VIS-band!
For this, I have used a rough hot/cold setup that seemed to work ok. This is a USER INPUT, e.g. would be trivial to change.

DOF used: 
single-band catalogues:  
	- obviously none 	- profiles fit with all parameters free (apart from disk sersic index and bulge sersic index where appropriate)  
multi-band catalogues: 
	- single-seric fits  		- x,y: fixed with wavelength, e.g. the same in each band  		- magnitude: free in each band  		- r_e: second order polynomial (3 degrees of freedom)  		- n: second order polynomial (3 degrees of freedom)  		- axis ratio q: fixed with wavelength, e.g. the same in each band  		- PA: fixed with wavelength, e.g. the same in each band  	- B/D fits  		- x,y: fixed with wavelength, e.g. the same in each band (and tied to each other, e.g. no offsets between bulges and disks!)  		- magnitude: free in each band  		- r_e: fixed with wavelength, e.g. the same in each band  		- n: fixed with wavelength, e.g. the same in each band (==1 for disks, ==4 or variable for bulges)  		- axis ratio q: fixed with wavelength, e.g. the same in each band  		- PA: fixed with wavelength, e.g. the same in each band  


*****  Fitting Times *****

The machines I use have multiple CPUS with several cores, typically with ~2.3 GHz each core. 
E.g. 4 CPU a 12 cores, multi-threaded, allowing 96 threads at half the core speed. 
However, at some stage, disk access comes into play (which is why I avoid some of our servers, that use a - software based - gluster system, which I highly advice against, it is very slow).  
For simplicity, I ran all fields in parallel, so things got in each others way somewhat, I think. 

The total run times are hard to measure, because Galapagos can split up the image into areas and can run in parallel, and I make extensive use of this.  
Further, there are some parts of the codes (sub-routines), which are fully paralleled, e.g. make use of all cores on a server. 

Discounting the wrapper (Galapagos), the GalfitM times for the fits are measured and stored in the fits tables which I provide, but the ratio between fitting size and setup size varies wildly. 
E.g. CPU_TOTAL_GALFIT contains the time a fit took on one CPU. 
The total time for the fits themselves varies badly with the number of bands used (but results are improved). 
E.g.  
DS Field 1 single-band all single-sersic fits: ~160h
DS Field 0 multi-band all single-sersic fits: ~8000h
additional for B/D: 
DS Field 1 single-band all B/D fits: ~150h
DS Field 0 multi-band all B/D fits: ~10000h

This can be sped up significantly, but using: 
	- multiple cores (e.g.  here I generally used 12, due to server limitations). Galapagos (currently in IDL) can fit 16 galaxies in parallel (galfit only uses 1 core, and there's a 16 object limitation in IDL itself). This limit could be lifted when re-coding the code in a different language.  	- split fitting area up into several areas, each done on their own batch (I use 4 batches for the FIELD0 DS fits, so I can fill the servers properly)  

In total, that means the fitting time for the single-band fits is below 1 day (plus the wrapper) 
All multi-band fits in FIELD0 took around 12 days (plus the wrapper). 

Additionally, there is a further gain when running several models. The B/D fits require the single-sersic fits to be run. 
That means 
	1.	you always need both  	2.	When running a second B/D fit (e.g. with n_b free), you save that first step, as you can re-use the previous fits  

In total I find the following run-times (from starting the code, to finish). 
(some of those crashed in between. Trivial to re-start and run, but makes measuring the total time difficult) 
This step includes the SExtractor step, and creating all files necessary for the fits, e.g. postage stamps for all objects. SExtractor is a significant time. 
As mentioned, these ran in parallel, so somewhat slowed each other down. 


Single-sersic fits:
	FIELD0:
		SS: 48 hours 
		RM: 18 hours 

	FIELD1:
		SS: 20 hours 
		RM: 18 hours 

	FIELD2:
		SS: ~20 hours (crashed in between, can not measure) 
		RM: ~50h 

	FIELD3:
		SS: ~50h (but servers were particularly busy) 
		RM: ~80h (but servers were particularly busy) 

	FIELD4:
		SS: ~96h (but servers were particularly busy) 
		RM: ~80h (but servers were particularly busy) 


B/D fits
	FIELD0 Multi-band fits FIELD0
		single-sersic plus n_bulge=4 B/D fits: ~1800h 
		(This is where I used 4 fitting areas, e.g. using more cores, in parallel, getting this down to ~450h, ~3 weeks) 

		additional n_bulge=free B/D fits: ~850h 
		(This is where I used 4 fitting areas, e.g. using more cores, in parallel, getting this down to ~200h, additional 10 days) 

	single-band
		FIELD1:
			single-sersic plus n_bulge=4 B/D fits: ~100h 
			additional n_bulge=free B/D fits: ~38h 

		FIELD2:
			single-sersic plus n_bulge=4 B/D fits: ~110h 
			additional n_bulge=free B/D fits: ~38h 

		FIELD3:
			single-sersic plus n_bulge=4 B/D fits: ~110h 
			additional n_bulge=free B/D fits: ~44h 

		FIELD4:
			single-sersic plus n_bulge=4 B/D fits: ~48h  (might have crashed in between?) 
			additional n_bulge=free B/D fits: ~44h 


