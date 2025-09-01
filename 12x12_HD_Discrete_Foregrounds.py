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

################################################################### radio

radio_catalog = pd.read_csv(f"{overall_path_example}S10_patches/radio_14x14deg_source_catalog_ra=6_dec=6.csv")
mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    HD_radio_patch = foregrounds_HD.generate_discrete_foreground(
                                    frequency = frequency,
                                    catalog = radio_catalog)
    HD_radio_patch.write_map(f"{overall_path_example}radio_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_radio_dls, HD_radio_ells = spectra_HD.get_foreground_power(HD_radio_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}radio_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_radio_ells, "dl": HD_radio_dls})

################################################################### CIB Model 1

CIB_model = 1
CIB_catalog_original = pd.read_csv(f"{overall_path_example}S10_patches/CIB_14x14deg_source_catalog_ra=6_dec=6.csv")

CIB_resolutions = [S10_resolution, 0.25]
CIB_lmaxs = [12574, 24000]

foregrounds_CIB = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12.0,
    new_res = CIB_resolutions[CIB_model-1],
    apod_width = 1.0,
    l_max = CIB_lmaxs[CIB_model-1])

CIB_sim = {
    30: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=30, scaling_factor=0.75),
    90: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=90, scaling_factor=0.75),
    148: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=148, scaling_factor=0.75),
    219: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=219, scaling_factor=0.75),
    277: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=277, scaling_factor=0.75),
    350: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=350, scaling_factor=0.75)
}

CIB_catalog = foregrounds_HD.make_catalog_from_sims(sims = CIB_sim, sigma_pix_frac=0.2, seed=0)
mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    HD_CIB_patch = foregrounds_HD.generate_discrete_foreground(
                                    frequency = frequency,
                                    catalog = CIB_catalog)
    HD_CIB_patch.write_map(f"{overall_path_example}CIB_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_CIB_dls, HD_CIB_ells = spectra_HD.get_foreground_power(HD_CIB_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}CIB_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_CIB_ells, "dl": HD_CIB_dls})

################################################################### CIB Model 2

CIB_model = 2
CIB_catalog_original = pd.read_csv(f"{overall_path_example}S10_patches/CIB_14x14deg_source_catalog_ra=6_dec=6.csv")

CIB_resolutions = [S10_resolution, 0.25]
CIB_lmaxs = [12574, 24000]

foregrounds_CIB = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12.0,
    new_res = CIB_resolutions[CIB_model-1],
    apod_width = 1.0,
    l_max = CIB_lmaxs[CIB_model-1])

CIB_sim = {
    30: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=30, scaling_factor=0.75),
    90: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=90, scaling_factor=0.75),
    148: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=148, scaling_factor=0.75),
    219: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=219, scaling_factor=0.75),
    277: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=277, scaling_factor=0.75),
    350: foregrounds_CIB.place_sources_in_largerPatch(catalog = CIB_catalog_original, frequency=350, scaling_factor=0.75)
}

CIB_catalog = foregrounds_HD.make_catalog_from_sims(sims = CIB_sim, sigma_pix_frac=0.2, seed=0)
mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    HD_CIB_patch = foregrounds_HD.generate_discrete_foreground(
                                    frequency = frequency,
                                    catalog = CIB_catalog)
    HD_CIB_patch.write_map(f"{overall_path_example}CIB2_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_CIB_dls, HD_CIB_ells = spectra_HD.get_foreground_power(HD_CIB_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}CIB2_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_CIB_ells, "dl": HD_CIB_dls})
