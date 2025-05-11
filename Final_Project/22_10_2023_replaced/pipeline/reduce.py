import argparse, textwrap

from datetime import datetime as dt

import sys, os, glob, psutil, math

import ccdproc as ccdp

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from astropy.nddata import CCDData
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
#from astropy.io import ascii
from astropy.table import Table
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ZScaleInterval


from photutils.utils import circular_footprint
from photutils.segmentation import (detect_sources, make_2dgaussian_kernel)
from photutils.segmentation import SourceCatalog
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from photutils.centroids import centroid_sources,  centroid_2dg

import matplotlib.pylab as pl

import numpy as np

import multiprocessing as mp

import concurrent.futures

import warnings
import logging
logging.disable(logging.INFO)
warnings.simplefilter('ignore')


import tempfile

import traceback

import timeit

from scipy import ndimage
from scipy.ndimage import median_filter

import twirl
from shutil import which
#from sys import platform

import pandas as pd

##################################### RAZNE POMOCNE FUNKCIJE ##############################

class Logger(object):
  def __init__(self):
    self.terminal = sys.stdout
    self.log = open("logfile.log", "a")
  
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    # this flush method is needed for python 3 compatibility.
    # this handles the flush command by doing nothing.
    # you might want to specify some extra behavior here.
    pass  
  
# klasa za formatiranje tekstova unutar argparse   
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass  
  
  
# Funkcija za merenje vremena izvršavanja
def measure_timeit(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    start_time = timeit.default_timer()  # Pokretanje tajmera
    result = wrapped()  # Izvršavanje funkcije i dobijanje rezultata
    elapsed_time = timeit.default_timer() - start_time  # Kraj merenja vremena
    print(f"Function {func.__name__} executed in {elapsed_time:.2f} seconds")
    return result  # Vraćanje rezultata funkcije
  
def process_batch(batch_files):
  try:
    batch_data = []
    for fl in batch_files:
        with fits.open(fl) as hdul:
            data = hdul[0].data
            batch_data.append(data)
    
    batch_data = np.array(batch_data)
    batch_median = np.nanmedian(batch_data, axis=0)
    return batch_median
  
  except Exception as e:
    print(f"Error in process_batch: {e}")
    traceback.print_exc()
    return None
  
##################################### FUNKCIJE ZA KALIBRACIJU ##############################

def makeMasterBIAS(file_list):
  
  frame_list = []
  for fl in file_list:
    with fits.open(fl) as hdul:
      frame_list.append(hdul[0].data)
  frame_arr = np.array(frame_list)
  
  mFrame = np.median(frame_arr, axis=0)
  
  return mFrame



def makeMasterDARK(file_list, mBias):
  
  frame_list = []
  for fl in file_list:
    with fits.open(fl) as hdul:
      frame_list.append(hdul[0].data - mBias)
  frame_arr = np.array(frame_list)
  
  mFrame = np.median(frame_arr, axis=0)
  
  return mFrame
  
  

def makeMasterFLAT(file_list, mBias, mDark_dic):
  
  frame_list = []
  for fl in file_list:
    with fits.open(fl) as hdul:
      data = hdul[0].data
      header = hdul[0].header
      
    expt_flat = float(header['EXPTIME'])
    expt_dark_max = np.max(list(mDark_dic.keys()))
    
    
    # BIAS + DARK correction
    if expt_flat in mDark_dic: # Check the line below for mBias:
      frame_cal = data - mBias - mDark_dic[expt_flat]
    elif expt_dark_max > expt_flat:
      scale = expt_flat / expt_dark_max
      frame_cal = data - mBias - scale * mDark_dic[expt_dark_max]
    else:
      frame_cal = data - mBias
    
    # flats normalization
    frame_cal = frame_cal / np.median(frame_cal)
    
    # append
    frame_list.append(frame_cal)  
  
  # turn into array
  frame_arr = np.array(frame_list)
  
  # average over 0 axis
  mFrame = np.median(frame_arr, axis=0)
  
  return mFrame


def process_calibrate(args):
  
  fl, mBias, mDark_dic, mFlat_dic = args
  
  print(f'-----> Calibrate {fl} file')
  
  with fits.open(fl) as hdul:
    header = hdul[0].header
    data = hdul[0].data
    
  filt_light = str(header['FILTER'])
  expt_light = float(header['EXPTIME'])
  expt_dark_max = np.max(list(mDark_dic.keys()))

  # BIAS + DARK korekcija
  if expt_light in mDark_dic:
      frame_cal = data - mBias - mDark_dic[expt_light]
  elif expt_dark_max > expt_light:
      scale = expt_light / expt_dark_max
      frame_cal = data - mBias - scale * mDark_dic[expt_dark_max]
  else:
      frame_cal = data - mBias
  
  # FLAT korekcija
  if filt_light in mFlat_dic:
      frame_cal = frame_cal / mFlat_dic[filt_light]
      
  # maskiraj zvezde
  mean, _, std = sigma_clipped_stats(frame_cal)
  #mask = make_source_mask(frame_cal, nsigma=2, npixels=5, dilate_size=11)
  threshold = 3. * std + mean
  kernel = make_2dgaussian_kernel(3.0, size=3)  # FWHM = 3.
  convolved_data = convolve(frame_cal, kernel)
  segm = detect_sources(convolved_data, threshold, npixels=5)
  footprint = circular_footprint(radius=10)
  mask = segm.make_source_mask(footprint=footprint)
  # background statistics
  
  mean, median, std = sigma_clipped_stats(frame_cal, sigma=3.0, mask=mask)

  # updejtuj header
  header['MEAN'] = (mean, 'Mean value after 3 sigma clip with masked objects')
  header['MEDIAN'] = (median, 'Median value after 3 sigma clip with masked objects')
  header['STDEV'] = (std, 'Standard deviation after 3 sigma clip with masked objects')
    
  # snimi
  basename = os.path.basename(fl)
  outFlNm = basename.split(".")[0]+"_cal.fit"  
  header['BSCALE'] = 1.
  header['BZERO'] = 0.
  primHDU = fits.PrimaryHDU()
  primHDU.header = header
  primHDU.data = np.float32(frame_cal)
  hdulist = fits.HDUList([primHDU])
  hdulist.writeto(os.path.join(calibFolderNm, outFlNm), overwrite=True)
  


def calibrate_singleprocess(lightNms_list, mBias, mDark_dic, mFlat_dic):
  
  for fl in lightNms_list:
    args = fl, mBias, mDark_dic, mFlat_dic
    process_calibrate(args)

    

def calibrate_multiprocess(lightNms_list, mBias, mDark_dic, mFlat_dic):
    
  # angazuj vise procesa za posao    
  Ncores = psutil.cpu_count() - 1
  with concurrent.futures.ThreadPoolExecutor(max_workers=Ncores) as executor:
    args = ((fl, mBias, mDark_dic, mFlat_dic) for fl in lightNms_list)
    executor.map(process_calibrate, args)



def fixpix(lightNms_list, darkNms_list, gain, hotPixThresh=3.):
  
  # napravi prvo hot/dead pixel masku
  if not os.path.isfile("hotPixMask.fits"):
  
    # median scDarks
    frame_list = []
    for fl in darkNms_list:
      with fits.open(fl) as hdul:
        frame_list.append(hdul[0].data)
    frame_arr = np.array(frame_list)
    medDark = np.median(frame_arr, axis=0)
    
    
    # izracunaj hotPixThresh u odnosu na vrednosti piksela
    medBoxSz = 5
    mDark_blurred = median_filter(medDark, medBoxSz)  # scipy function
    difference = medDark - mDark_blurred
    hotPixThresh = hotPixThresh * np.std(difference)
    
    # napravi hot pixel mask 
    hotPixMask = np.zeros(medDark.shape)
    hotPixMask[(np.abs(difference)>hotPixThresh)] = 1
    
     # ispisi kao uint
    hotPixMask_ccd = CCDData(hotPixMask.astype('uint8'), unit=u.dimensionless_unscaled)
    header = fits.Header()
    header['imagetyp'] = "HOTPIXMASK"
    header['comment'] = "number of hot pixels: {}".format(int(hotPixMask.sum()))
    hotPixMask_ccd.header = header
    hotPixMask_ccd.write(os.path.join(calibFolderNm, "hotPixMask.fits"), overwrite=True)
  
  else:
    
    with fits.open(os.path.join(calibFolderNm, "hotPixMask.fits")) as hdul:
      hotPixMask = hdul[0].data  
      
  # za svaku sliku pravi CR  masku i kombinuj za hot/deadmaskom
  for fl in lightNms_list:
    
    print(f"\n-----> fixpix {os.path.basename(fl)}")
   
    with fits.open(fl) as hdul:
      data = hdul[0].data
      header = hdul[0].header
    
    _, cosRayMask = ccdp.cosmicray_lacosmic(data, sigclip=7, gain=gain, niter=3, objlim=20, cleantype="idw", satlevel=55000)
    
    # info
    print(f"    Numer of hot/dead pixels: {int(np.sum(hotPixMask))}")
    print(f"    Numer of cosmic rays: {int(np.sum(cosRayMask))}")
    
    finalMask = np.logical_or(hotPixMask.astype(bool), cosRayMask.astype(bool))
    finalmask_ccd = CCDData(cosRayMask.astype('uint8'), unit=u.dimensionless_unscaled)
    finalmask_ccd.write(fl.replace(".fit", "_CRmask.fit"), overwrite=True)
  
    # korekcija
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    
    # meshgrid
    xx, yy = np.meshgrid(x, y) # xx (2048, 2048), yy (2048, 2048)
    
    # masked data
    mdata = np.ma.masked_array(data, mask=finalMask==1) 
    
    # pozicije losih vrednosti koje treba da odredis interpolacijom
    missing_x = xx[mdata.mask]  # (1291,) x indeksi maskiranih piksela
    missing_y = yy[mdata.mask]  # (1291,) y indeksi maskiranih piksela
    
    # interpolacija
    interp_values = ndimage.map_coordinates(mdata, [missing_x, missing_y], order=1)
    
    # fiksiraj piksele
    interp_mdata = mdata.copy()
    interp_mdata[missing_y, missing_x] = interp_values  # vodi racuna da prvi ide missing_y, pa missing_x!!!
    
    # snimi
    interp_mdata_ccd = CCDData(interp_mdata.astype(float), unit='adu')
    interp_mdata_ccd.header = header
    interp_mdata_ccd.header['FIXPIX'] = (True,"Hot pixels fixed")
    interp_mdata_ccd.write(fl.replace(".fit", "_fix.fit"), overwrite=True)




def process_makeMasterSKYFLAT(fl):
  
  with fits.open(fl) as hdul:
      header = hdul[0].header
      data = hdul[0].data
      
  # uzmi median
  median = float(header['MEDIAN'])
  
  # normiraj
  data = data / median
      
  # maskiraj zvezde
  #mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=11)
  
  # maskiraj  
  #data_masked = np.ma.masked_array(data, mask=mask).filled(np.nan).astype(np.float32)
  mean, _, std = sigma_clipped_stats(data) 
  threshold = 3. * std + mean 
  kernel = make_2dgaussian_kernel(3.0, size=3)  
  convolved_data = convolve(data, kernel) 
  segm = detect_sources(convolved_data, threshold, npixels=5) 
  footprint = circular_footprint(radius=10) 
  mask = segm.make_source_mask(footprint=footprint) 
  data_masked = np.ma.masked_array(data, mask=mask).filled(np.nan).astype(np.float32)
  # snimi
  basename = os.path.basename(fl)
  outFlNm = basename.replace('.fit', '_forSkyFlat.fit')
  primHDU = fits.PrimaryHDU()
  primHDU.header = header
  primHDU.data = np.float32(data_masked)
  hdulist = fits.HDUList([primHDU])
  hdulist.writeto(os.path.join(calibFolderNm, outFlNm), overwrite=True)


def makeMasterSKYFLAT_multiprocess(skyflatNms_list):
  
  
  # uradi maskiranje 
  Ncores = psutil.cpu_count() - 1
  with concurrent.futures.ThreadPoolExecutor(max_workers=Ncores) as executor:
      executor.map(process_makeMasterSKYFLAT, skyflatNms_list)
  
  # prikupi maskirane fajlove
  tmp_flNm = [os.path.join(calibFolderNm, os.path.basename(fl).replace('.fit', '_forSkyFlat.fit')) for fl in skyflatNms_list]
  
  # Definišemo batch veličinu (npr. 100 fajlova po batchu)
  batch_size = 100
  num_batches = len(tmp_flNm) // batch_size + (1 if len(tmp_flNm) % batch_size != 0 else 0)
  #print(len(tmp_flNm), batch_size, num_batches)

  #batch_results = []
  temp_files = []

  # Procesiramo svaki batch i čuvamo rezultate na disku
  for i in range(num_batches):
      batch_files = tmp_flNm[i*batch_size:(i+1)*batch_size]
      batch_median = process_batch(batch_files)
      
      # Sačuvamo batch medijanu u privremeni fajl
      temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
      np.save(temp_file, batch_median)
      temp_files.append(temp_file.name)
      print(f"Batch {i} saved to temporary file: {temp_file.name}")  # Ispisuje putanju privremenog fajla
      temp_file.close()

  # Učitavamo sve batch medijane i računamo konačnu medijanu
  all_batch_medians = []
  for temp_file in temp_files:
      batch_median = np.load(temp_file)
      all_batch_medians.append(batch_median)
      os.remove(temp_file)  # Obrisati privremeni fajl nakon upotrebe

  all_batch_medians = np.array(all_batch_medians)
  final_median = np.nanmedian(all_batch_medians, axis=0)

  return final_median



def process_skycalibrate(args):
  
  fl, mSkyflat_dic = args
    
  print(f'-----> maskerSky calibrate {os.path.basename(fl)} file')
  
  with fits.open(fl) as hdul:
    header = hdul[0].header
    data = hdul[0].data
    
  # FILTER
  filt = header['FILTER']
    
  # SKYFLAT korekcija
  if mSkyflat_dic[filt] is not None:
    median_light = float(header['MEDIAN'])
    data = data - median_light * mSkyflat_dic[filt] + median_light
    
  # snimi
  basename = os.path.basename(fl)
  outFlNm = basename.split(".")[0]+"_skyflat.fit"  
  primHDU = fits.PrimaryHDU()
  primHDU.header = header
  primHDU.data = np.float32(data)
  hdulist = fits.HDUList([primHDU])
  hdulist.writeto(os.path.join(calibFolderNm, outFlNm), overwrite=True)

def skycalibrate_singleprocess(lightNms_list, mSkyflat_dic):
  for fl in lightNms_list:
    args = fl, mSkyflat_dic
    process_skycalibrate(args)


def skycalibrate_multiprocess(lightNms_list, mSkyflat_dic):
    Ncores = psutil.cpu_count() - 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=Ncores) as executor:
        args = ((fl, mSkyflat_dic) for fl in lightNms_list)
        executor.map(process_skycalibrate, args)


##################################### FUNKCIJE ZA ASTROMETRIJU ##############################


def getCPUtemp():
  
  temperatures = psutil.sensors_temperatures()
  temps_list = [r.current for r in temperatures['coretemp']]

  return temps_list

# def doAstrometry(fl, pixelscale):

#     hdul = fits.open(fl)
#     header = hdul[0].header 
#     FITSnm, RA, DEC = fl, header['OBJCTRA'], header['OBJCTDEC']
#     center = SkyCoord(RA, DEC, unit=["hour", "deg"])
#     FITSout=FITSnm.replace(".fit","_wcs.fit" ).replace('calibration','astrometry')
#     cmd = "solve-field "+FITSnm+" --ra "+str(center.ra.degree)+" --dec "+str(center.dec.degree)+" --radius 0.2 --scale-units arcsecperpix --scale-low 0.38 --scale-high 0.40 --crpix-center -p -N "+FITSout+" -O -I noneI.fits -M none.fits -R none.fits -B none.fits -P none -k none -U none -y -S none --axy noneaxy --wcs junk.wcs"
#     print(cmd)
#     os.system(cmd)
#     time.sleep(10)

def doAstrometry(FITSfls_list):
  
  lines = []
  
  for fl in FITSfls_list: 
    with fits.open(fl) as hdul:
      header = hdul[0].header
      if (header.get('objctra') is not None) & (header.get('objctdec') is not None):
          lines.append(f"{fl} {header['OBJCTRA'].replace(' ',':')} {header['OBJCTDEC'].replace(' ',':')} {header['FILTER']}")  
  
  # number of FITS files
  Nfits = len(lines)
  
  # define number of threades
  Ncores = psutil.cpu_count() - 1
  
  # number of loops
  Nloops = int(math.ceil(Nfits/float(Ncores))) # ceil zaokruzuje broj navise
  k = 0
  print('Nloops:', Nloops)
  for i in range(Nloops):
    with open("tmp", "w") as outFl:
      for j in range(int(Ncores)):
        if k >= Nfits:
          continue
        else:  
          FITSnm, RA, DEC, filterName = lines[k].split()
          FITSout=FITSnm.replace(".fit","_wcs.fit" ).replace('calibration','astrometry')
          # Check if astrometry is installed AND system cmd parallel
          # else use python twirl package
          scaleLow=pixelscale-0.01
          scaleHigh=pixelscale+0.01
          outFl.write("solve-field "+FITSnm+" --ra "+RA+" --dec "+DEC+" --radius 0.2 --scale-units arcsecperpix --scale-low " + str(scaleLow) + " --scale-high "+ str(scaleHigh)+ " --crpix-center -p -N "+FITSout+" -O -I noneI.fits -M none.fits -R none.fits -B none.fits -P none -k none -U none -y -S none --axy noneaxy --wcs bla.wcs\n")
          print("solve-field "+FITSnm+" --ra "+RA+" --dec "+DEC+" --radius 0.2 --scale-units arcsecperpix --scale-low" + str(scaleLow) + " --scale-high "+ str(scaleHigh)+ " --crpix-center -p -N "+FITSout+" -O -I noneI.fits -M none.fits -R none.fits -B none.fits -P none -k none -U none -y -S none --axy noneaxy --wcs bla.wcs\n")
          print("Astrometry (solve-field): ",FITSnm," --> ",FITSout)
          k += 1
    cmd = "parallel -j "+str(Ncores)+" < tmp"
    os.system(cmd)
 

def doTwirl(fl):
    
    hdul = fits.open(fl)
    header = hdul[0].header 
    FITSnm, RA, DEC = fl, header['OBJCTRA'], header['OBJCTDEC']
    center = SkyCoord(RA, DEC, unit=["hour", "deg"])
    FITSout=FITSnm.replace(".fit","_wcs.fit" ).replace('calibration','astrometry')
    data = hdul[0].data
    fov = np.max(data.shape) * pixelscale * u.arcsec.to(u.deg)
    sky_coords = twirl.gaia_radecs(center, 1.2 * fov)[0:30]
    pixel_coords = twirl.find_peaks(data)[0:30]
    wcs = twirl.compute_wcs(pixel_coords, sky_coords, tolerance=10)
    outFl = fits.PrimaryHDU(data)
    outFl.header = header
    outFl.header.update(wcs.to_header())
    outFl.writeto(FITSout,overwrite=True)
    print("Astrometry (twirl): ",FITSnm," --> ",FITSout)
        

# def astrometry_multiprocess(lightNms_list, pixelscale):
#   with concurrent.futures.ThreadPoolExecutor() as executor:
#     args = ((fl, pixelscale) for fl in lightNms_list)
#     executor.map(doAstrometry, args)

def astrometry_multiprocess(lightNms_list):
    
    if (which('solve-field') is not None) & (which('parallel') is not None) & (astool == 'astrometry'):
        doAstrometry(lightNms_list)
    else:
        Ncores = psutil.cpu_count() - 1
        p = mp.Pool(processes=Ncores, maxtasksperchild=1000)
        for fl in lightNms_list:
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
            p.apply_async(doTwirl, [fl])     
        p.close()
        p.join() 
        
##################################### FUNKCIJE ZA FOTOMETRIJU ##############################
  
def doPhotometry(fl, prefix): 
   
   #fl, prefix = args
   print(fl,' -> ',fl.replace("fit","txt").replace("astrometry","photometry"))
   fwhm = fwhm_fixed
   flagP = 0
   with fits.open(fl) as hdul:      
        data = hdul[0].data
        header = hdul[0].header
        #data = data / header['EXPTIME'] # images normalized to 1 sec exptime: ADU/sec
        wcs = WCS(header)
        object_name = header['OBJECT']
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)   
         
        targets_list=object_name + '_stars.txt'  
        # Comparison stars given in an external file [target]_stars.txt
        if os.path.exists(targets_list) and os.path.getsize(targets_list) > 0:
            sources=pd.read_csv(targets_list,names=['RAJ2000','DECJ2000','id'],sep=" ")
            flag = np.zeros(len(sources),dtype='int')
            pixel_coo = wcs.wcs_world2pix(sources['RAJ2000'], sources['DECJ2000'], 0)
            xpos, ypos = pixel_coo[0], pixel_coo[1]
            x_init, y_init = pixel_coo
            x, y = centroid_sources(data, x_init, y_init, box_size=11,
                                     centroid_func=centroid_2dg)
            pixel_coo = x, y
            flux_peak = data[np.rint(y).astype(int),np.rint(x).astype(int)]
            world = wcs.wcs_pix2world(np.transpose(pixel_coo), 0)
            if pd.isnull(sources['id']).any():
                mark = np.arange(len(sources))#list(range(0,len(sources)))
            else:
                mark = sources['id']
            
            if if_fwhm_variable:
                size=21 # size of a star image 
                gauss = fitting.LevMarLSQFitter()
                fwhm_i = []
                for i in range(0,len(x[1:])):
                    cutout = Cutout2D(data-median, [x[i+1],y[i+1]], size)
                    # Try to model each object
                    star=cutout.data/np.mean(cutout.data)
                    y0, x0 = np.unravel_index(np.argmax(star), star.shape)
                    sigma = np.std(star)
                    amp = np.max(star)
                    my_gauss = models.Gaussian2D(amp, x0, y0, sigma, sigma)
                    yi, xi = np.indices(star.shape)
                    g = gauss(my_gauss, xi, yi, star)
                    fwhm_i.append((g.x_fwhm + g.y_fwhm)/2.)
                # Optimal FWHM is a mean/median value    
                fwhm = np.mean(fwhm_i)
        else:
            # No comparison stars given -> find all the sources 
            kernel = make_2dgaussian_kernel(3.0, size=5)
            convolved_data = convolve(data, kernel)
            segment_map = detect_sources(convolved_data, median+3*std, npixels=10)      
            cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
            #FWHMllim=np.mean(cat.fwhm)-3*np.std(cat.fwhm)
            #FWHMulim=np.mean(cat.fwhm)+3*np.std(cat.fwhm)
            #cat = cat[(cat.fwhm>FWHMllim) & (cat.fwhm<FWHMulim)]
            flux_peak = cat.max_value
            flag = np.zeros(len(cat),dtype='int')
            mark = cat.labels
            x, y = cat.xcentroid, cat.ycentroid
            pixel_coo = x, y
            if if_fwhm_variable:
                fwhm = np.mean(cat.fwhm).value

        #print('# Aperture photometry on: ', fl)
        # aperture=3xFWHM, sky_annulus=3xFWHM+10 (width_of_sky_annulus=10)
        positions = np.transpose(pixel_coo)
        apertures = CircularAperture(positions, r=3*fwhm)
        annulus_aperture = CircularAnnulus(positions, r_in=3*fwhm+10, r_out=3*fwhm+20)
        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_stats = ApertureStats(data, annulus_aperture, sigma_clip=sigclip)
        aper_stats_bkgsub = ApertureStats(data, apertures, local_bkg=bkg_stats.median)
        flux=aper_stats_bkgsub.sum
        #flux_high=total_flux.sum-(bkg_stats.median-bkg_stats.std/2)*apertures.area
        flux_err = np.sqrt(flux[flux > 0] + apertures[flux > 0].area * bkg_stats[flux > 0].std**2 * (1+apertures[flux > 0].area/annulus_aperture[flux > 0].area))
        mag = -2.5*np.log10(flux[flux > 0])
        mag_err = 1.0857 * flux_err / flux[flux > 0]
        positions = positions[flux>0]
        xpos = pixel_coo[0][flux>0]
        ypos = pixel_coo[1][flux>0]
        mark = mark[flux>0]
        world = wcs.wcs_pix2world(positions, 0)
        flag[flux_peak>60000]=1  
        flag = flag[flux>0]
        flux_peak = flux_peak[flux>0]
        flux = flux[flux > 0]
        #print('Number of objects found:', len(flux))
        # WRITE OUTPUT FILES WITH APERTURE PHOTOMETRY
        photfile = prefix+'_photometry.txt'
        # Comparison stars are provided -> output differential photometry
        if os.path.exists(targets_list) and os.path.getsize(targets_list) > 0:
              with open(targets_list, "r") as f:
                 num_stars = len(f.readlines())
                 f.close()
              if not (os.path.exists(photfile) and os.path.getsize(photfile) > 0):
                   headline = ['FILENAME','MJD-HELIO','FILTER','DATE-OBS','FWHM']
                   for i in range(1,num_stars):
                       headline.append('TmC'+str(i))
                   for i in range(1,num_stars):
                       headline.append('TmC'+str(i)+'_err')
                   for i in range(1,num_stars-1):
                       for j in range(i+1,num_stars):
                           headline.append('C'+str(i)+'mC'+str(j))
                   for i in range(1,num_stars-1):
                       for j in range(i+1,num_stars):
                            headline.append('C'+str(i)+'mC'+str(j)+'_err')
                   headline.append('\n')
              else:
                   headline=[]
                   
              with open(prefix+'_photometry.txt', mode='a+') as outname:
                  magdiff=np.zeros(int((num_stars-1)*(num_stars-2)/2),dtype=float)
                  magdiff_err=np.zeros(int((num_stars-1)*(num_stars-2)/2),dtype=float)
                  k=0 
                  for i in range(1,num_stars-1):
                      for j in range(i+1,num_stars):
                           magdiff[k] = mag[i]-mag[j]
                           magdiff_err[k] = np.sqrt(mag_err[i]**2 + mag_err[j]**2)
                           k+=1
                  #print(prefix+'_photometry.txt target file exists.')
                  magnitudes=np.concatenate((mag[0]-mag[1:],np.sqrt(mag_err[0]**2 + mag_err[1:]**2),magdiff,magdiff_err))
                  txtline=[]                 
                  txtline=[fl,str(header['JD-HELIO']-2400000.5),str(header['FILTER']),str(header['DATE-OBS']),str(fwhm)]
                  magline=magnitudes.tolist()
                  outline=headline+txtline+magline
                  outline.append('\n')
                  outname.writelines("%s " % item for item in outline)
                  outname.close()
           
        else:
              # No comparison stars -> append only target photometry in one line
              #print('No comparison stars.')
              if not (os.path.exists(photfile) and os.path.getsize(photfile) > 0):
                   headline = ['FILENAME','MJD-HELIO','FILTER','DATE-OBS', 'FWHM','RAJ2000','DECJ2000','xpix','ypix','flag','flag_p','flux_peak','flux','mag','mag_err']
                   headline.append('\n')
              else:
                   headline=[]
                   
              catalog = SkyCoord(ra=world[:,0]*u.degree, dec=world[:,1]*u.degree)
              input_catalog='katalog.cat'
                       
              if os.path.exists(input_catalog) and os.path.getsize(input_catalog) > 0: # Use coo from input katalog.cat file
                   targets=np.genfromtxt(input_catalog, delimiter=' ', dtype=None, 
                                             names=['name','ra','dec'], encoding=None, usecols=(0,1,2))
                   if ((targets['name']).size == 1):
                           target_idx = 0
                           object_skycoo = SkyCoord(ra=targets['ra'].tolist(), dec=targets['dec'].tolist(), unit=(u.hourangle, u.deg))
                   else:    
                           target_idx = [n for n, x in enumerate(targets['name']) if object_name in x]
                           object_ra, object_dec = targets['ra'][target_idx][0], targets['dec'][target_idx][0]
                           object_skycoo = SkyCoord(ra=object_ra, dec=object_dec, unit=(u.hourangle, u.deg))
              else:
                   #print('katalog.cat not provided so use header coordinates to locate the object')
                   object_skycoo = SkyCoord(header['OBJCTRA'], header['OBJCTDEC'], unit=["hour", "deg"])            
              #print(object_skycoo, object_skycoo.ra, object_skycoo.dec)
              object_pixcoo = wcs.wcs_world2pix(object_skycoo.ra, object_skycoo.dec, 0)
              #print(object_pixcoo, object_pixcoo[0], object_pixcoo[1])
              # Find the object in the catalog complied from all the image sources           
              idx, d2d, d3d = object_skycoo.match_to_catalog_sky(catalog)
              max_sep = 5.0 * u.arcsec
              proximity_constraint = d2d.to('arcsec') < max_sep
              if proximity_constraint: 
                    flagP = 0
              else: 
                    flagP = 1
                    print("WARNING: Object not found within 5 arcsec from its input coordinates. Nominal header coordinates are used.")                
              with open(prefix+'_photometry.txt', mode='a+') as outname:
                     magnitude=[flux_peak[idx], flux[idx], mag[idx], mag_err[idx]]
                     txtline=[]                 
                     txtline=[fl,str(header['JD-HELIO']-2400000.5),str(header['FILTER']),str(header['DATE-OBS']),fwhm,world[:,0][idx], world[:,1][idx], positions[idx][0], positions[idx][1],flag[idx], flagP]
                     outline=headline+txtline+magnitude
                     outline.append('\n')
                     outname.writelines("%s " % item for item in outline)
                     outname.close()
        # Write single ascii file with aperture photometry for all the sources
        table = Table()
        table['xpix'] = xpos
        table['ypix'] = ypos
        for col in table.colnames:
            table[col].info.format = '%.4f'
        table['ID'] = mark
        table['RAJ2000'] = world[:,0]
        table['DECJ2000'] = world[:,1]
        table['flux_peak'] = flux_peak
        table['flux'] = flux
        table['flux_err'] = flux_err
        table['mag'] = mag
        table['mag_err'] = mag_err
        table['AIRMASS'] = header['AIRMASS']
        table['FWHM'] = fwhm
        for col in table.colnames[4:]:
            table[col].info.format = '%.4f'
        table['FILTER'] = header['FILTER']  
        table['MJD-HELIO'] = header['JD-HELIO']-2400000.5
        table['DATE-OBS'] = header['DATE-OBS']
        table['flag'] = flag
        # Output table name will include FWHM used for aperture photometry
        print('Overwriting tables')
        table.write(fl.replace("fit","txt").replace("astrometry","photometry"), format='ascii', overwrite=True)
          
        
        # plot
        if if_plot_apers:
              z = ZScaleInterval()
              zmin,zmax = z.get_limits(data)
              fig, ax = pl.subplots(1,1, figsize=(12,12))
              ax.set_title(os.path.basename(fl))
              ax.imshow(data, origin='lower', cmap='Greys_r', vmin=zmin, vmax=zmax, interpolation='nearest')
              if os.path.exists(targets_list) and os.path.getsize(targets_list) > 0:
                   idx = 0
              for i in range(len(positions)):
                  if i == idx:
                      circle = pl.Circle((positions[idx,0], positions[idx,1]), radius=20, color='tab:red', lw=1.5, alpha=1., fill=False, label=mark[idx])
                      label = ax.annotate(mark[idx], xy=(positions[idx,0], positions[idx,1]), fontsize=16, ha="center", color='tab:red', xytext=(10, 10),textcoords='offset points', fontstyle='oblique')
                  else:
                      circle = pl.Circle((positions[i,0], positions[i,1]), radius=25, color='yellow', lw=1.5, alpha=1, fill=False) #, label=mark[i])
                      label = ax.annotate(mark[i], xy=(positions[i,0], positions[i,1]), fontsize=16, ha="center", color='yellow', xytext=(10, 10),textcoords='offset points', fontstyle='oblique')
                  if flagP:
                      circleX = pl.Circle((object_pixcoo[0], object_pixcoo[1]), radius=15, color='tab:blue', lw=1.5, alpha=1, fill=False, label='?')
                      #label = ax.annotate('?', xy=(object_pixcoo[0], object_pixcoo[1]), fontsize=12, ha="center", color='tab:blue', xytext=(0, 0),textcoords='offset points', fontstyle='oblique')
                      ax.add_patch(circleX)
                  ax.add_patch(circle)
                  
              pl.savefig(fl.replace("fit", "png").replace("astrometry","photometry"), format='png')


def photometry_singleprocess(FITSfls_list, prefix):
   for fl in FITSfls_list:
       args = fl, prefix
       doPhotometry(args)
        

def photometry_multiprocess(FITSfls_list, prefix):

    Ncores = psutil.cpu_count() - 1 # leave one core
    p = mp.Pool(processes=Ncores,maxtasksperchild=1000)
    for fl in FITSfls_list:
        # launch a process for each file (ish).
        # The result will be approximately one process per CPU core available.
        p.apply_async(doPhotometry, [fl, prefix]) 
    
    p.close()
    p.join() 

##################################### GLAVNI PROGRAM ##############################  
  
# glavni program
def main(args):
    
  global if_plot_apers, if_fwhm_variable, fwhm_fixed, if_pause_proc, astool, pixelscale
    
  # definisi ulazne parametre
  FITS_file_path    = args.FITS_file_path
  pixelscale        = args.pixscale
  astool            = args.astool
  gain              = args.gain
  
  if_fixpix_corr      = args.if_fixpix_corr
  if_back_corr        = args.if_back_corr
  if_parallel         = args.if_parallel
  if_astrometry       = args.if_astrometry
  if_pause_proc       = args.if_pause_proc
  if_photometry       = args.if_photometry
  if_fwhm_variable    = args.if_fwhm_variable
  fwhm_fixed          = args.fwhm_fixed
  if_plot_apers       = args.if_plot_apers
  if_save_fls         = args.if_save_fls
  
  
  # definisi dolder sa FITS fajlovima
  if FITS_file_path == 'current':
    path = os.getcwd()
  else:
    if FITS_file_path[0] == '/':  
      path = FITS_file_path
    else:
      path = os.getcwd() + '/' + FITS_file_path
      
  os.chdir(path)
  # definisi path file
  path_file = os.path.join(path, '*fit')
  # proveri da li ima FITS fajlova u folderu  
  if len(glob.glob(path_file)) == 0:
    sys.exit('WARNING: There are no FITS files in defined directory')
    
    
  # ----- konstante koje se cesto pozivaju
  
  global calibFolderNm, astroFolderNm, photoFolderNm

  calibFolderNm = "./calibration"
  astroFolderNm = "./astrometry"
  photoFolderNm = "./photometry"
  # ----- napravi nove foldere
  
  if not os.path.isdir(calibFolderNm):
    os.makedirs(calibFolderNm)
  if not os.path.isdir(astroFolderNm):
    os.makedirs(astroFolderNm)  
  if if_photometry:
      if not os.path.isdir(photoFolderNm):
          os.makedirs(photoFolderNm)  
  

  # ------ sakupi sve FITS slike

  # kolekcija
  ifc = ccdp.ImageFileCollection(path)
  
  # lista BIAS imena
  biasNms_list = ifc.filter(regex_match=True, imagetyp="bias|zero").files
  
  # dictionary DARK imena sa key=EXPTIME i values=lista DARK fajlova
  darkNms_dic = {}
  ifc_dark = ifc.filter(regex_match=True, imagetyp='dark')
  ifc_dark_groups = ifc_dark.summary.group_by('exptime')
  for k,g in zip(ifc_dark_groups.groups.keys, ifc_dark_groups.groups):
    darkNms_dic[float(k['exptime'])] = [fl for fl in list(g['file'])]
  
  # dictionary FLAT imena sa key=FILTER i values=lista FLAT fajlova
  flatNms_dic = {}
  ifc_flat = ifc.filter(regex_match=True, imagetyp='flat')
  ifc_flat_groups = ifc_flat.summary.group_by('filter')
  for k,g in zip(ifc_flat_groups.groups.keys, ifc_flat_groups.groups):
    flatNms_dic[str(k['filter'])] = [fl for fl in list(g['file'])] 
    
  # lista LIGHT imena
  lightNms_list = ifc.filter(regex_match=True, imagetyp="light|object").files
  if len(lightNms_list) == 0:
      raise ValueError("There are no LIGHT .fit files (science images) in this working dir.")
  num_of_objects = np.unique(list(map(lambda x: os.path.basename(x).split('-')[:-1],lightNms_list))).size
  object_names = np.unique(list(map(lambda x: os.path.basename(x).split('-')[0],lightNms_list)))
  
  # zapocni logovanje
  sys.stdout = Logger()

  # pocetak pokretanja programa
  #dt = datetime.datetime.today()
  print (f"\n========= CALIBRATION START: {dt.today()} ===========\n")
  
  print(f"\n----------------------- MAKE MASTER BIAS FRAME: {dt.today()}------------------------\n")
  
  if not os.path.isfile(os.path.join(calibFolderNm, "mBias.fits")):
    # make master bias
    mBias = makeMasterBIAS(biasNms_list)
    print("Making master bias: mBias.fits\n")
    
    # SAVE
    primHDU = fits.PrimaryHDU()
    primHDU.data = np.float32(mBias)
    primHDU.header['HISTORY'] = "Master bias frame"
    hdulist = fits.HDUList([primHDU])
    hdulist.writeto(os.path.join(calibFolderNm, "mBias.fits"), overwrite=True)
    
  else:
    # load
    with fits.open(os.path.join(calibFolderNm, "mBias.fits")) as hdul:
      mBias = hdul[0].data
    print("Reading master bias: mBias.fits\n")
  
  
  print(f"\n----------------------- MAKE MASTER DARK FRAME: {dt.today()}------------------------\n")
  
  
  # output: mDark_dic = {5:np.array, 200:np.array, 600:np.array ... }
  mDark_dic = {}
  for expt,darkNms_list in darkNms_dic.items():
    
    # create name for master dark frame
    dark_outputNm = "mDark_{}sec.fits".format(str(int(expt)))
    
    if not os.path.isfile(os.path.join(calibFolderNm, dark_outputNm)):
      
      # make master dark
      mDark = makeMasterDARK(darkNms_list, mBias)
      print(f'Making master dark: {dark_outputNm}\n')
      
      # SAVE
      primHDU = fits.PrimaryHDU()
      primHDU.data = np.float32(mDark)
      primHDU.header['HISTORY'] = "Master dark frame"
      hdulist = fits.HDUList([primHDU])
      hdulist.writeto(os.path.join(calibFolderNm, dark_outputNm), overwrite=True)
      
      # put into a dictionary
      mDark_dic[expt] = mDark
      
    else:
      
      # load
      with fits.open(os.path.join(calibFolderNm, dark_outputNm)) as hdul:
        mDark = hdul[0].data
      print(f'Reading master dark: {dark_outputNm}\n')
        
      # put into a dictionary
      mDark_dic[expt] = mDark
      
    

  print(f"\n----------------------- MAKE MASTER FLAT FRAME: {dt.today()}------------------------\n")
  
  # output: mFlat_dic = {'B':np.array, 'V':np.array, ... }
  mFlat_dic = {}
  for filt,flatNms_list in flatNms_dic.items():
    
    # konstruisi ime za master dark frame
    flat_outputNm = "mFlat_{}.fits".format(filt)
    
    if not os.path.isfile(os.path.join(calibFolderNm, flat_outputNm)):
      
      # napravi master dark
      mFlat = makeMasterFLAT(flatNms_list, mBias, mDark_dic)
      print(f'Making master flat: {flat_outputNm}\n')
      
      # snimi
      primHDU = fits.PrimaryHDU()
      primHDU.data = np.float32(mFlat)
      primHDU.header['HISTORY'] = "Master flat frame"
      hdulist = fits.HDUList([primHDU])
      hdulist.writeto(os.path.join(calibFolderNm, flat_outputNm), overwrite=True)
      
      # put into a dictionary
      mFlat_dic[filt] = mFlat
      
    else:
      
      # load
      with fits.open(os.path.join(calibFolderNm, flat_outputNm)) as hdul:
        mFlat = hdul[0].data
      print(f'Reading master flat: {flat_outputNm}\n')
        
      # put into a dictionary
      mFlat_dic[filt] = mFlat
  
  
  
  print(f"\n----------------------- DO CLASSIC CALIBRATION: {dt.today()}------------------------\n")
    
  inputFls_list = lightNms_list  
  outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit"))    
  tmpInput = list(map(lambda x: os.path.basename(x),inputFls_list))
  tmpOutput = list(map(lambda x: os.path.basename(x).replace('_cal',''),outputFls_list))
 
  if len(tmpOutput) < len(tmpInput):
    
    tmp =  [x for x in tmpInput if x not in set(tmpOutput)]  
    inputFls_list_new = [os.path.join(path, name) for name in tmp]
    # paralelno izvrsavanje programa
    if if_parallel:
      measure_timeit(calibrate_multiprocess, inputFls_list_new, mBias, mDark_dic, mFlat_dic)
    else:
      measure_timeit(calibrate_singleprocess, inputFls_list_new, mBias, mDark_dic, mFlat_dic)
  else:
    print('Calibration of LIGHT frames is already done\n')
  
  
  if if_fixpix_corr:
    print(f"\n----------------------- BAD PIXEL CORRECTION: {dt.today()}------------------------\n")
    
    inputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit"))
    outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_fix.fit"))
    if len(outputFls_list) < len(inputFls_list):
      tmp = list(map(lambda x: x.replace('cal_fix','cal'),outputFls_list))
      inputFls_list_new =  [x for x in inputFls_list if x not in set(tmp)]  
      maxDarkNms_list = darkNms_dic[max(darkNms_dic)]
      # koriguj kosmicke zrake i hot/dead piksele
      
      measure_timeit(fixpix, inputFls_list_new, maxDarkNms_list, gain, hotPixThresh=3.)
      
      # ukloni ./calibration/*CRmask.fit fajlove
      if not if_save_fls:
        os.system(f'rm -f {os.path.join(calibFolderNm, "*CRmask.fit")}')
      
    else:
      print('Cosmic rays and hot/dead pixels already removed\n')
  else:
    
    outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit"))
      
      
  if if_back_corr:
    print(f"\n----------------------- SKY-FLAT CALIBRATION: {dt.today()}------------------------\n")
    
    doSkyFlat=False
    if if_fixpix_corr:
      inputFls_list = glob.glob(os.path.join(calibFolderNm, "*_fix.fit"))
      outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_fix_skyflat.fit"))
      if len(outputFls_list) < len(inputFls_list):
        tmp = list(map(lambda x: x.replace('_skyflat',''),outputFls_list))
        inputFls_list_new =  [x for x in inputFls_list if x not in set(tmp)] 
        doSkyFlat=True
    else:
      inputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit"))
      outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal_skyflat.fit"))
      if len(outputFls_list) < len(inputFls_list):
        tmp = list(map(lambda x: x.replace('_skyflat',''),outputFls_list))
        inputFls_list_new =  [x for x in inputFls_list if x not in set(tmp)] 
        doSkyFlat=True
    # ako jos nije uradjeno 
    #conda install -c conda-forge spyder=5.5.5
    if (doSkyFlat is True):  
      # dictionary 
      skyflatNms_dic = {}
      ifc = ccdp.ImageFileCollection(filenames=inputFls_list_new)
      ifc_groups = ifc.summary.group_by('filter')
      for k,g in zip(ifc_groups.groups.keys, ifc_groups.groups):
        skyflatNms_dic[str(k['filter'])] = [fl for fl in list(g['file'])] 
      

      # output: mSkyflat_dic = {'B':np.array, 'V':np.array, None, ... }
      mSkyflat_dic = {}
      for filt,skyflatNms_list in skyflatNms_dic.items():
        
        # konstruisi ime za master dark frame
        mSkyflat_outNm = "mSkyflat_{}.fits".format(filt)
        if not os.path.isfile(os.path.join(calibFolderNm, mSkyflat_outNm)):
          
          # make master dark
          mSkyflat = measure_timeit(makeMasterSKYFLAT_multiprocess, skyflatNms_list)
          
          # SAVE
          if mSkyflat is not None:
            primHDU = fits.PrimaryHDU()
            primHDU.data = np.float32(mSkyflat)
            # dodaj median u header
            if math.isnan(np.nanmedian(mSkyflat)):
                medvalue = 0
            else:
                medvalue = np.nanmedian(mSkyflat)
            print('Median to be written into header:', medvalue)
            primHDU.header['MEDIAN'] = (medvalue, 'Median value after 3 sigma clip with masked objects')
            primHDU.header['HISTORY'] = f"Master sky-flat frame in {filt} filter"
            hdulist = fits.HDUList([primHDU])
            hdulist.writeto(os.path.join(calibFolderNm, mSkyflat_outNm), overwrite=True)
            print(f'Making master sky flat: {mSkyflat_outNm}\n')
          
          # put into a dictionary
          mSkyflat_dic[filt] = mSkyflat
          
          # ukloni ./calibration/*forSkyFlat.fit fajlove
          if not if_save_fls:
            os.system(f'rm -f {os.path.join(calibFolderNm, "*forSkyFlat.fit")}')
          
        else:
      
          # load
          with fits.open(os.path.join(calibFolderNm, mSkyflat_outNm)) as hdul:
            mSkyflat = hdul[0].data
            
          # put into a dictionary
          mSkyflat_dic[filt] = mSkyflat
          
          #informer
          if mSkyflat_dic[filt] is not None:
            print(f'Reading master sky flat: {mSkyflat_outNm}\n')
    
      # uradi kalibraciju
      if if_parallel:
        measure_timeit(skycalibrate_multiprocess, inputFls_list, mSkyflat_dic)
      else:
        measure_timeit(skycalibrate_singleprocess, inputFls_list, mSkyflat_dic)
        
      # pokupi output fajlove   
      outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_skyflat.fit"))
  
    # ako je uradjeno
    else:
      
      print('Background already removed\n')
      
      # pokupi output fajlove   
      outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_skyflat.fit"))
      
  else:
    
    if if_fixpix_corr:
      outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_fix.fit"))
    else:
      outputFls_list = glob.glob(os.path.join(calibFolderNm, "*_cal.fit"))
      
  calibratedFls_list = outputFls_list    
  check_astrometry = False  
  if if_astrometry:
    print(f"\n----------------------- DO ASTROMETRY: {dt.today()}------------------------\n")
 
    if if_fixpix_corr:
        if if_back_corr:
            inputFls_list = glob.glob(os.path.join(calibFolderNm, "*fix_skyflat.fit"))
            outputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix_skyflat_wcs.fit"))            
        else:
            inputFls_list = glob.glob(os.path.join(calibFolderNm, "*fix.fit"))
            outputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix_wcs.fit"))
    else:
        if if_back_corr:
            inputFls_list = glob.glob(os.path.join(calibFolderNm, "*cal_skyflat.fit"))
            outputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_skyflat_wcs.fit"))
        else:    
            inputFls_list = glob.glob(os.path.join(calibFolderNm, "*cal.fit"))
            outputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_wcs.fit"))
    print('input:',len(inputFls_list), ' solved:',len(outputFls_list))
    if len(outputFls_list) < len(inputFls_list):
      tmp = list(map(lambda x: x.replace('_wcs','').replace('astrometry','calibration'),outputFls_list))
      inputFls_list_new =  [x for x in inputFls_list if x not in set(tmp)]     
      measure_timeit(astrometry_multiprocess, inputFls_list_new)
      print("Astrometry is done.")
      check_astrometry = True
    else:
      print('Astrometry already done.\n')
      check_astrometry = True
  
     
  # Check if astrometry is done for all the calibrated images
  # If so - do photometry
  outputFls_list = glob.glob(os.path.join(astroFolderNm, "*wcs.fit"))     
  if len(calibratedFls_list) == len(outputFls_list):
      check_astrometry = True
  
  if if_photometry:
      if check_astrometry:
        print(f"\n----------------------- DO PHOTOMETRY: {dt.today()}------------------------\n")
        #if if_fixpix_corr:
        #    inputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix.fit"))
        #    outputFls_list = glob.glob(os.path.join(photoFolderNm, "*fix_wcs.txt"))
        #else:
        #    inputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_wcs.fit"))
        #    outputFls_list = glob.glob(os.path.join(photoFolderNm, "*cal_wcs.txt"))
        if if_fixpix_corr:
            if if_back_corr:
                inputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix_skyflat_wcs.fit"))            
                outputFls_list = glob.glob(os.path.join(photoFolderNm, "*fix_skyflat_wcs.txt"))
            else:
                inputFls_list = glob.glob(os.path.join(astroFolderNm, "*fix_wcs.fit"))
                outputFls_list = glob.glob(os.path.join(photoFolderNm, "*fix_wcs.txt"))
        else:
            if if_back_corr:
                inputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_skyflat_wcs.fit"))
                outputFls_list = glob.glob(os.path.join(photoFolderNm, "*cal_skyflat_wcs.txt"))
            else:    
                inputFls_list = glob.glob(os.path.join(astroFolderNm, "*cal_wcs.fit"))
                outputFls_list = glob.glob(os.path.join(photoFolderNm, "*cal_wcs.txt"))

        inputFls_list_new = []
        for obj_num in range(0,num_of_objects):        
              print('Object::', object_names[obj_num])
              prefix = object_names[obj_num]
              targets_list = object_names[obj_num] + '_stars.txt'
              phot_list = object_names[obj_num] + '_photometry.txt'
              match_input = [s for s in inputFls_list if object_names[obj_num] in s]
              print('-------------------')
              photNms_list = []
              if (os.path.exists(targets_list)): #    and (os.path.getsize(targets_list) > 0:
                  #match_output_size = sum(1 for _ in open(phot_list)) - 2 # -2: 1 for header and 1 for last empty line
                  if os.path.exists(phot_list) and os.path.getsize(phot_list) > 0:
                      done = np.loadtxt(phot_list, usecols=(0,),dtype=str).tolist()
                      #print('Number of input frames:', len(match_input))
                      #print("Already done for", len(done)-1, ' input frames.')
                      if len(match_input) > len(done)-1:  
                          photNms_list = [x for x in match_input if x not in done]
                          #print('No. of missing files:', len(photNms_list))
                          print('No. of missing files:', len(match_input),' (input) -',len(done)-1,'(done) =',len(photNms_list))
                      else:
                          print('Photometry on ' + object_names[obj_num] + ' is done.\n')
                  else:
                      photNms_list = match_input
                      
              else:
                  match_output = [s for s in outputFls_list if object_names[obj_num] in s]
                  if len(match_output) < len(match_input):
                      tmp = list(map(lambda x: x.replace('txt','fit').replace('photometry','astrometry'),match_output))
                      photNms_list =  [x for x in match_input if x not in set(tmp)]
                      print('No. of missing files:', len(match_input),' (input) -',len(match_output),'(done) =',len(photNms_list))
                      #print(os.path.exists(targets_list) and os.path.getsize(targets_list) > 0)
                  else:
                      print('Photometry on ' + object_names[obj_num] + ' is already done.\n')
              if len(photNms_list) > 0:   
                  if if_parallel:
                      photometry_multiprocess(photNms_list, prefix)
                  else:
                      for fl in photNms_list: 
                          args = fl, prefix
                          doPhotometry(args)
      else:
          print("Photometry cannot be done without astrometric solution.")
          print("Run the code with -a flag to add astrometry.")
      print(f"\n----------------------- PHOTOMETRY DONE: {dt.today()}------------------------\n")         
  print(f"\n----------------------- CALIBRATION ENDS: {dt.today()}------------------------\n") 
  
if __name__ == "__main__":
  
    
  parser = argparse.ArgumentParser(
    
    prog = sys.argv[0],
    
    description=textwrap.dedent('''DESCRIPTION
      The program performs automatic calibration, astrometry and photometry. 
      
      Input:
        - 'catalog.txt' - Lista koordinata zvezda za fotometriju koja ima sledecu formu:
          
          RA DEC
          02:22:39.6 +43:02:08
          02:38:38.9 +16:36:59
          04:23:15.8 -01:20:33
      '''),
    
    formatter_class = MyFormatter,
    
    epilog=textwrap.dedent('''EPILOG
      The output FITS files for calibration, astrometry and photometry are stored in the 'calibration', 'astrometry' and 'photometry' subfolders of the radok directory where we want to run the pipeline.
      
      Calibration output files:
        - 'mBias.fits' - master bias frame
        - 'mDark*fits' - bias subtracted master dark frames
        - 'mFlat*fits - bias and dark corrected master flat-field frames
        - 'mSkyflat*fits' - master sky-flats for background correction
        - '*_cal.fit'- bias, dark anf flat-field calibrated FITS files
        - '*_fix.fit' - bad pixel corrected FITS files (optional)
        - '*_skyflat.fit' - background corrected FITS files (optional)
        - '*_forSkyFlat.fit' - FITS files for making master sky-flats (optional)
        - '*_CRmask.fit' - Cosmic-ray correction masks (optional)
        - 'badPixelMask.fits' - Bad pixel mask
        
      Astrometry output files:
        - '*.new' - WCS transformed FITS files
        
      Photometry output files:
      '''),
    
  )
   
  parser.add_argument("FITS_file_path", nargs="?", default='current', help="Apsolute path to FITS files")
   
  # pixel scale
  general = parser.add_argument_group('GENERAL', 'General parameters')
  general.add_argument('--pixscale', type=float, default=0.39, help='Pixel scale for WCS transformation')   
  general.add_argument('--astool', type=str, default='twirl', help='Tool for astrometry (astrometry | twirl)')
  general.add_argument('--gain', type=str, default=1., help='CCD gain')
  general.add_argument("-n", "--if-parallel", action="store_false", help="Whether to parallelize functions")
  general.add_argument("-s", "--if-save-fls", action="store_true", help="Whether to save check FITS files")
  # korekcija na lose piksele
  calibration = parser.add_argument_group('CALIBRATION', 'Parameters for calibration')
  calibration.add_argument("-c", "--if-fixpix-corr", action="store_true", help="Whether to perform bad pixel correction")
  calibration.add_argument("-b", "--if-back-corr", action="store_true", help="Whether to perform background correction")
  
  # astrometry
  astrometry = parser.add_argument_group('ASTROMETRY', 'Parameters for astrometry')
  astrometry.add_argument("-a", "--if-astrometry", action="store_true", help="Whether to solve WCS")
  astrometry.add_argument("-k", "--if-pause-proc", action="store_true", help="Whether to pause program when CPU is hot")
  
  # photometry + plotting apertures + setting FWHM
  photometry = parser.add_argument_group('PHOTOMETRY', 'Parameters for photometry')
  photometry.add_argument("-p", "--if-photometry", action="store_true", help="Whether to perform photometry: default = False")
  photometry.add_argument("-l", "--if-plot-apers", action="store_true", help="Whether to plot photometry apertures")
  photometry.add_argument("-x", "--fwhm-fixed", type=float, default=3., help="FWHM is fixed to some user defined value")
  photometry.add_argument("-v", "--if-fwhm-variable", action="store_true", help="Whether FWHM is left free to be determined")

  

  args = parser.parse_args()  
  main(args)
  
  
