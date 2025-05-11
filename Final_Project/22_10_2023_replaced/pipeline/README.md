# reduce.py

Python pipeline reduce.py calibrates CCD images taken with Milankovic telescope mounted at the Astronomical Station Vidojevica. 

Prerequisites:
1. psutil v5.9.0
2. ccdproc v2.4.2
3. astropy v5.3.4
4. photutils v1.13.0
5. scipy v1.13.1
6. twirl v0.4.2
7. padas v2.2.2

Header keywords required for the script to work: IMAGETYP, FILTER, EXPTIME, OBJCTRA, OBJCTDEC, DATE-OBS, AIRMASS, JD-HELIO.

It creates three dirs in the parent (working) dir: 

 1. calibration # To store calibrated images
 2. astrometry # To store astrometrically solved calibrated images
 3. photometry # To store output tables with measured photometry
 
 The basic usage is simple (standard) data reduction (bias, dark and flat-field correction):

    python /home/data/ reduce.py # -> calibration/*_cal.fit (/home/data/ can be absolute or relative path)

Note: if the script is called from the dir with raw images then path (/home/data/) can be omitted. Standard calibration is done and calibrated images with suffix _cal are created and stored in the 'calibration' dir. If other corrections are needed like the correction for hot/dead pixels an additional key (-c) should be set:

    python /home/data/ reduce.py -c # -> calibration/*cal_fix.fit

and the output calibrated files have the suffix _fix that adds up to the previous suffux _cal, so the end result files have _cal_fix suffix.
One more possibility is correction for large scale background variations invoked by additional -b key. We suggest the usage of -s key to store intermediate files.

    python /home/data/ reduce.py -c -b -s # -> calibration/*cal_fix_skyflat.fit

Intermediate files include: bad pixel map (badPixelMask.fits), individual mask of cosmic rays (*CRmask.fit) and individual files with prefix forSkyFlat. They are stored in the dir 'calibration'. The user should also check super-sky flat files for all the filters named mSkyFlat{FILTER}.fit, where FILTER stands for any of the Johnson standard filters including additional: L, SII, Halpha and Halpha continuum filters. The sky flat mSkyFlat{FILTER}.fit should look like noise without any artefacts and especially without light sources. If any of these additional corrections should be omitted, the user should continue without -c or/and -b keys.

Once satisfied, the user should continue with astrometry, since all the science frames have to be astrometrically solved so that target object and comparison stars can be located in each frame. By default, python package Twirl is used to determine celestial coordinates of the imaged objects, but also the external software Astrometry can be called via '--astool astrometry' keyword. For the sake of completness, we shal assume the user does want to apply both -c and -b correction, so KEEPING these keys, user should add additional -a key:

    python /home/data/ reduce.py -a # Twirl is called by default -> astrometry/*_cal_fix_skyflat_wcs.fit
    python /home/data/ reduce.py -a --astool astrometry # Astrometry is called -> astrometry/*_cal_fix_skyflat_wcs.fit

After proper calibration and obtained WCS solutions, photometry is measured using Photutils package:

    python /home/data/ reduce.py -a -p -l # -> photometry/*_cal_fix_skyflat_wcs.txt

where -p key stands for phootmetry and -l creates png images with all selected or predefined sources circled in each science frame stored in the dir 'photometry'. By default, FWHM is fixed to 3 pixels, but can be changed to some other fixed value (in the example below 4 pixels) using the -x key:

    python /home/data/ reduce.py -a -p -l -x 4

Another option is to make FWHM variable (-v) with the following call:

    python /home/data/ reduce.py -a -p -l -v 

Radius for aperture photometry is 3 x FWHM, annulus radius starting 10 pixels away from the aperture radius and spanning for 10 pixels. Three sigma clipping is done in the sky annulus. The peak flux is measured and if larger than 60 000 ADU the flagP is set to 1, meaning the corresponding star is saturated. The output ASCII files with photometric measurements for each individual science frame are created and stored in the 'photometry' dir. An example of this file is listed below:

    head -2 photometry/{target}_{FILTER}_cal_fix_skyflat_wcs.txt     
    xpix ypix ID RAJ2000 DECJ2000 flux_peak flux flux_err mag mag_err AIRMASS FWHM FILTER MJD-HELIO DATE-OBS flag
    1111.7350 1006.9145 0 1.5814427075972448 20.2030 13007.4668 367197.2191 925.5835 -13.9122 0.0027 1.5651 4.0000 V 60230.739660352 2023-10-13T17:36:20 0

Apart from these individual files, the 'master' photometry file is created in the RAW dir (/home/data/) for each target in which single line corresponds to the single image:

    head -2 {target}_photometry.txt
    FILENAME MJD-HELIO FILTER DATE-OBS RAJ2000 DECJ2000 xpix ypix flag flag_p flux_peak flux mag mag_err    
     ./astrometry/G191B2B-0017_cnt_cal_fix_skyflat_wcs.fit 60231.075227288995 HaContinuum 2023-10-14T01:44:34 76.3777321296412 52.8305282527778 1059.2327337833053 1072.937897607265 0 0 5034.44775390625 40773.788041787404 -11.525952652387069 0.006300336022649495 

Additional flag (flag = 0) informs the user if the target is found in the image within 5 arcsec from its position read from the catalog or the image header. If not its value is changed to 1. The flag 'flagP' is set to zero by default unless the source is saturated (flagP = 1). By default, the catalog file 'katalog.cat' should be provided before observations, and should list all targets that should be observed:

    cat katalog.cat
    Mrk335 00:06:19.54 +20:12:10.6
    G191B2B 05:05:30.62 +52:49:54.0

The pipeline reads this file and locate targets with the coordinates provided. If the file is missing, nominal coordinates read from the image header (OBJCTRA, OBJCTDEC) are used instead.

Differential photometry is done automatically if the file named {target}_stars.txt is provided in the main (working) dir. An example of this file is given below:

    cat {target}_stars.txt
    1.5813689 20.2029552 Mrk335
    1.6925974 20.1617019 7
    1.6119681 20.150317 8
    1.5837104 20.1806955 4
    1.6329513 20.2190228 D
    1.5749078 20.2214483 6
    1.5949117 20.2492471 2
    1.6084351 20.2789486 1

In this case, the 'master' photometry file contains differential photometry:

    head -2 {target}_photometry.txt
    FILENAME MJD-HELIO FILTER DATE-OBS TmC1 TmC2 TmC3 TmC4 TmC5 TmC6 TmC7 TmC1_err TmC2_err TmC3_err TmC4_err TmC5_err TmC6_err TmC7_err C1mC2 C1mC3 C1mC4 C1mC5 C1mC6 C1mC7 C2mC3 C2mC4 C2mC5 C2mC6 C2mC7 C3mC4 C3mC5 C3mC6 C3mC7 C4mC5 C4mC6 C4mC7 C5mC6 C5mC7 C6mC7 C1mC2_err C1mC3_err C1mC4_err C1mC5_err C1mC6_err C1mC7_err C2mC3_err C2mC4_err C2mC5_err C2mC6_err C2mC7_err C3mC4_err C3mC5_err C3mC6_err C3mC7_err C4mC5_err C4mC6_err C4mC7_err C5mC6_err C5mC7_err C6mC7_err 
    ./astrometry/Mrk335-0007_Ha_cal_fix_skyflat_wcs.fit 60230.86840787437 Ha 2023-10-13T20:40:14 2.6834339914239607 1.13103086312384 0.7356397275212618 -0.6197561120229782 -0.49532935122873667 0.4598300439866634 0.2074277292441984 0.006433976798852723 0.007017003627359047 0.007397612525340747 0.011424690215147594 0.010706193586646475 0.007753423770871061 0.008274522032931534 -1.5524031283001207 -1.9477942639026988 -3.303190103446939 -3.1787633426526973 -2.2236039474372973 -2.4760062621797623 -0.3953911356025781 -1.7507869751468181 -1.6263602143525766 -0.6712008191371766 -0.9236031338796415 -1.35539583954424 -1.2309690787499985 -0.27580968353459845 -0.5282119982770634 0.12442676079424153 1.0795861560096416 0.8271838412671766 0.9551593952154 0.7027570804729351 -0.252402314742465 0.0034240467672547446 0.0041485452188756424 0.009644133080275636 0.008781135317982964 0.00475408629798472 0.005563584387015894 0.005005268213717508 0.010042489000654256 0.009216866057989704 0.005517573650335894 0.006228623739610696 0.010312027758737366 0.0095098344412314 0.005994159670511726 0.0066544785564573135 0.0128932473231782 0.010570185693383069 0.010958145842037378 0.009789170557062587 0.01020686018803148 0.007047907061956785


In the header line T stands for target and C{1..7} for comparison stars, so that TmC1 is magnitude difference (hence m = minus) between the target and the first comparison star etc.
