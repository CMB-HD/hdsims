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
    final_width = 2.0,
    new_res = S10_resolution,
    apod_width = 0.2,
    l_max = 24000)

component_to_datapath = {
    'kappa': "S10_data/healpix_4096_KappaeffLSStoCMBfullsky.fits",
    'tSZ': "S10_data/090_tsz_healpix.fits",
    'kSZ': "S10_data/090_ksz_healpix.fits"
}

#################################

for component in ['kappa','tSZ','kSZ']:
    
    deconvolve_pixel_window = True
    nside = 8192
    scaling_factor = 1.0
    frequency = 90
    if component == 'tSZ':
        scaling_factor = 0.75       
    if component == 'kappa':
        nside = 4096
        deconvolve_pixel_window = False
        frequency = None
    fullsky_res = hp.nside2resol(nside, arcmin=True)
    
    # Load and deconvolve the fullsky maps
    fullsky_map = foregrounds_S10.load_fullsky(frequency, component_to_datapath[component],
                                  scaling_factor = scaling_factor, deconvolve_pixel_window = deconvolve_pixel_window)
    if frequency == None:
        fullsky_map.write_map(f"S10_data/deconvolved_fullsky/{component}_fullsky_deconvolved")
    else:
        fullsky_map.write_map(f"S10_data/deconvolved_fullsky/{frequency}GHz_{component}_fullsky_deconvolved")


#################################