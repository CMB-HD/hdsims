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

spectra_S10 = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = S10_resolution,
    apod_width = 1.0,
    l_max = 12574)
spectra_S10.make_binning_files(binning_output_path = f"{overall_path_example}binning_files/")

spectra_S10_kappa = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = S10_resolution_kappa,
    apod_width = 1.0,
    l_max = 6287)
spectra_S10_kappa.make_binning_files(binning_output_path = f"{overall_path_example}binning_files/", type_Cl = True)
spectra_S10_kappa.make_binning_files(binning_output_path = f"{overall_path_example}binning_files/")

spectra_HD = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = 0.04,
    apod_width = 1.0,
    l_max = 24000)
spectra_HD.make_binning_files(binning_output_path = f"{overall_path_example}binning_files/")
spectra_HD.make_binning_files(binning_output_path = f"{overall_path_example}binning_files/", type_Cl = True)
spectra_HD.make_binning_files(binning_output_path = f"{overall_path_example}binning_files/", spin0and2 = True)
