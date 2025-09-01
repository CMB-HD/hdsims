from hdsims import Foregrounds, Spectra, CMB

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os
import gc
from matplotlib.colors import LinearSegmentedColormap
from pixell import enmap, enplot, colorize, utils, wcsutils, curvedsky as cs
from pspy import so_map, so_mcm, so_spectra, pspy_utils
import useful_plots as upl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.dpi'] = 250
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.xmargin'] = 0.025
plt.rcParams['axes.ymargin'] = 0.025
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['figure.figsize'] = (5, 3)
if 'planck' not in mpl.colormaps:
    colorize.mpl_setdefault('planck')

planck_cmap = plt.get_cmap('planck')
pos_cmap = upl.truncate_colormap(planck_cmap, minval=0.5, maxval=1.0, n=100)
neg_cmap = upl.truncate_colormap(planck_cmap, minval=0.0, maxval=0.5, n=100)

overall_path_example = 'hdsims_output/'

S10_resolution = hp.nside2resol(8192, arcmin=True) # Original S10 0.43' pixel size
S10_resolution_kappa = hp.nside2resol(4096, arcmin=True) # Original S10 0.86' pixel size

###################################################################

spectra_HD = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = 0.04,
    apod_width = 1.0,
    l_max = 24000)

foregrounds_HD = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12.0,
    new_res = 0.04,
    apod_width = 1.0,
    l_max = 24000)

################################################################### CMB (lensed)

CMB_patch = CMB(
    ra = 6,
    dec = 6,
    final_width = 12.0,
    res = 0.04,
    apod_width = 1.0,
    l_max = 24000)

cmb_unlensed = CMB_patch.make_unlensed_patch(theory_path = f"{overall_path_example}camb_full_output.dat", 
                                  cmb_seed = 58)
HD_kappa_patch = so_map.read_map(f"{overall_path_example}kappa_14x14deg_ra=6_dec=6")

cmb_lensed = CMB_patch.do_lensing(cmb_unlensed, HD_kappa_patch)

cmb_lensed.write_map(f"{overall_path_example}CMB_Lensed_12x12deg_ra=6_dec=6")

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", spin0and2 = True)
HD_lensedCMB_dls, HD_lensedCMB_ells = spectra_HD.get_CMB_power(cmb_lensed, mbb_inv, binning_file, deconvolve_pw = True)
np.save(f"{overall_path_example}CMB_Lensed_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_lensedCMB_ells, "dl": HD_lensedCMB_dls})

################################################################### CMB (theory)

data = CMB_patch.make_lensed_theory(HD_kappa_spectrum = np.load(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()])

np.savetxt(f'{overall_path_example}camb_full_output_lensed.dat', data, fmt='%.6f', delimiter='\t')