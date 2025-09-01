import healpy as hp
from hdsims import Foregrounds
from pspy import so_map
import os

#################################

try:
    os.makedirs("S10_data/deconvolved_fullsky/", exist_ok=True)
    print(f"Made directory: S10_data/deconvolved_fullsky/")
except FileExistsError:
    print(f"Directory already exists: S10_data/deconvolved_fullsky/")

#################################

S10_resolution = hp.nside2resol(8192, arcmin=True) # Original S10 0.43' pixel size
S10_resolution_kappa = hp.nside2resol(4096, arcmin=True) # Original S10 0.86' pixel size

foregrounds_S10 = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12.0,
    new_res = S10_resolution,
    apod_width = 1.0,
    l_max = 24000)

component_to_name = { 'tsz': 'tSZ', 'ksz': 'kSZ' }
frequency_to_freqpath = { 30:'030',90:'090',148:'148',219:'219',277:'277',350:'350' }

#################################

component = 'tsz'
for frequency in [30,90,148,219,277,350]:

    deconvolve_pixel_window = True
    nside = 8192
    scaling_factor = 1.0
    if component == 'tsz':
        scaling_factor = 0.75       
    fullsky_res = hp.nside2resol(nside, arcmin=True)
    
    fullsky_map = foregrounds_S10.load_fullsky(frequency, f"S10_data/{frequency_to_freqpath[frequency]}_{component}_healpix.fits",
                                  scaling_factor = scaling_factor, deconvolve_pixel_window = deconvolve_pixel_window)
    fullsky_map.write_map(f"S10_data/deconvolved_fullsky/{frequency}GHz_{component_to_name[component]}_fullsky_deconvolved")

#################################

component = 'kappa'
nside = 4096
deconvolve_pixel_window = False
frequency = None

scaling_factor = 1.0
fullsky_res = hp.nside2resol(nside, arcmin=True)
        
fullsky_map = foregrounds_S10.load_fullsky(frequency, "S10_data/healpix_4096_KappaeffLSStoCMBfullsky.fits",
                              scaling_factor = scaling_factor, deconvolve_pixel_window = deconvolve_pixel_window)
fullsky_map.write_map(f"S10_data/deconvolved_fullsky/{component}_fullsky_deconvolved")