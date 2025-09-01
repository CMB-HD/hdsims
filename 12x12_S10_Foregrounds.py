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

spectra_S10_kappa = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = S10_resolution_kappa,
    apod_width = 1.0,
    l_max = 6287)

foregrounds_S10 = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12.0,
    new_res = S10_resolution,
    apod_width = 1.0,
    l_max = 12574)

foregrounds_S10_kappa = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 14.0,
    new_res = S10_resolution_kappa,
    apod_width = 1.0,
    l_max = 6287)

################################################################### S10

mbb_inv, binning_file, Bbl = spectra_S10.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    S10_kSZ_patch = foregrounds_S10.generate_diffuse_foreground(
                    component = 'kSZ',
                    frequency = frequency,
                    fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/{frequency}GHz_kSZ_fullsky_deconvolved")
    S10_tSZ_patch = foregrounds_S10.generate_diffuse_foreground(
                    component = 'tSZ',
                    frequency = frequency,
                    fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/{frequency}GHz_tSZ_fullsky_deconvolved")
    
    S10_radio_patch = foregrounds_S10.generate_discrete_foreground(
                                        frequency = frequency,
                                        catalog = pd.read_csv(f"{overall_path_example}S10_patches/radio_14x14deg_source_catalog_ra=6_dec=6.csv"))
    S10_CIB_patch = foregrounds_S10.generate_discrete_foreground(
                                        frequency = frequency,
                                        catalog = pd.read_csv(f"{overall_path_example}S10_patches/CIB_14x14deg_source_catalog_ra=6_dec=6.csv"),
                                        scaling_factor = 0.75)
    S10_tSZ_dls, S10_tSZ_ells = spectra_S10.get_foreground_power(S10_tSZ_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}S10_patches/tSZ_{frequency}GHz_10x10deg_ra=6_dec=6_S10spectra.npy", {"l": S10_tSZ_ells, "dl": S10_tSZ_dls})
    S10_kSZ_dls, S10_kSZ_ells = spectra_S10.get_foreground_power(S10_kSZ_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}S10_patches/kSZ_{frequency}GHz_10x10deg_ra=6_dec=6_S10spectra.npy", {"l": S10_kSZ_ells, "dl": S10_kSZ_dls})
    S10_radio_dls, S10_radio_ells = spectra_S10.get_foreground_power(S10_radio_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}S10_patches/radio_{frequency}GHz_10x10deg_ra=6_dec=6_S10spectra.npy", {"l": S10_radio_ells, "dl": S10_radio_dls})
    S10_CIB_dls, S10_CIB_ells = spectra_S10.get_foreground_power(S10_CIB_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}S10_patches/CIB_{frequency}GHz_10x10deg_ra=6_dec=6_S10spectra.npy", {"l": S10_CIB_ells, "dl": S10_CIB_dls})
    

S10_kappa_patch = foregrounds_S10_kappa.generate_diffuse_foreground(
                    component = 'kappa',
                    frequency = None,
                    fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/kappa_fullsky_deconvolved")

mbb_inv, binning_file, Bbl = spectra_S10_kappa.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
S10_kappa_cls, S10_kappa_ells = spectra_S10_kappa.get_foreground_power(S10_kappa_patch, mbb_inv, binning_file, deconvolve_pw = False, type_Cl = True)
np.save(f"{overall_path_example}S10_patches/kappa_10x10deg_ra=6_dec=6_S10spectra.npy", {"l": S10_kappa_ells, "cl": S10_kappa_cls})