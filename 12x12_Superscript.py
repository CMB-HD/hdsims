# Note: this does not include EVERYTHING. Before using this, you should run 12x12_Binning_Files.py, 12x12_deconvolve_all_S10_fullsky.py, and 12x12_reduce_S10_catalogs.py
# These create the binning files, deconvolve the Lambda fullsky maps, and reduce the original S10 catalogs to just those including the patch we care about, respectively.

# This file then just uses the results from those to generate the actual patches we want.

from hdsims import Foregrounds, Spectra, CMB

import time
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
'''
################################################################### kappa

start_time = time.time()

foregrounds_HD_kappa = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 14.0,
    new_res = 0.04,
    apod_width = 1.0,
    l_max = 24000)

S10_kappa_patch = foregrounds_HD_kappa.generate_diffuse_foreground(
                  component = 'kappa',
                  frequency = None,
                  fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/kappa_fullsky_deconvolved")

main_data_text, main_data_dat = foregrounds_HD_kappa.get_kappa_theory()

np.savetxt(f'{overall_path_example}camb_full_output.txt', main_data_text,
    header='ell    tt_lensed   ee_lensed   bb_lensed   te_lensed   '
           'tt_unlensed ee_unlensed bb_unlensed te_unlensed kk',
    fmt='%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e')
np.savetxt(f'{overall_path_example}/camb_full_output.dat', main_data_dat, fmt='%.6f', delimiter='\t')

hddata = np.loadtxt(f"{overall_path_example}camb_full_output.txt", comments="#")
template_ells =  hddata[:,0]
template_cls = hddata[:,9]
template_cls *= 2*np.pi/(template_ells * (template_ells + 1))**2

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
HD_kappa_patch, kappa_theory_power = foregrounds_HD_kappa.extend_to_small_scales(S10_patch = S10_kappa_patch, patch_type='kappa', template_ells=template_ells, template_cls=template_cls, mbb_inv=mbb_inv, binning_file=binning_file)

HD_kappa_patch.write_map(f"{overall_path_example}kappa_14x14deg_ra=6_dec=6")
np.save(f"{overall_path_example}kappa_smallscale_theory_spectra.npy", kappa_theory_power)

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
HD_kappa_cls, HD_kappa_ells = spectra_HD.get_foreground_power(HD_kappa_patch, mbb_inv, binning_file, deconvolve_pw = False, type_Cl = True)
np.save(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_kappa_ells, "cl": HD_kappa_cls})

print("--- Kappa: %s seconds ---" % (time.time() - start_time))

################################################################### tSZ

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    start_time = time.time()
    HD_tSZ_patch = foregrounds_HD.generate_diffuse_foreground(
                    component = 'tSZ',
                    frequency = frequency,
                    fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/{frequency}GHz_tSZ_fullsky_deconvolved")
    HD_tSZ_patch.write_map(f"{overall_path_example}tSZ_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_tSZ_dls, HD_tSZ_ells = spectra_HD.get_foreground_power(HD_tSZ_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}tSZ_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_tSZ_ells, "dl": HD_tSZ_dls})
    print(f"--- tSZ $frequency$GHz: %s seconds ---" % (time.time() - start_time))

################################################################### kSZ

ksz_template = pd.read_csv("S10_data/cmbhd_mockdata_ksz_cls_v1.1.txt", delimiter=' ')
template_ells = ksz_template['ell']
template_cls = ksz_template['C_ell^kSZ']

for frequency in [30,90,148,219,277,350]:
    start_time = time.time()
    S10_kSZ_patch = foregrounds_HD.generate_diffuse_foreground(
                component = 'kSZ',
                frequency = frequency,
                fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/{frequency}GHz_kSZ_fullsky_deconvolved")
    
    mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
    HD_kSZ_patch, kSZ_theory_power = foregrounds_HD_kappa.extend_to_small_scales(S10_patch = S10_kSZ_patch, patch_type='kSZ', template_ells=template_ells, template_cls=template_cls, mbb_inv=mbb_inv, binning_file=binning_file)
    
    HD_kSZ_patch.write_map(f"{overall_path_example}kSZ_{frequency}GHz_12x12deg_ra=6_dec=6")
    np.save(f"{overall_path_example}kSZ_{frequency}GHz_smallscale_theory_spectra.npy", kSZ_theory_power)

    mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")
    HD_kSZ_dls, HD_kSZ_ells = spectra_HD.get_foreground_power(HD_kSZ_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}kSZ_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_kSZ_ells, "dl": HD_kSZ_dls})
    print(f"--- kSZ $frequency$GHz: %s seconds ---" % (time.time() - start_time))
'''
################################################################### radio

radio_catalog = pd.read_csv(f"{overall_path_example}S10_patches/radio_14x14deg_source_catalog_ra=6_dec=6.csv")
mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    start_time = time.time()
    HD_radio_patch = foregrounds_HD.generate_discrete_foreground(
                                    frequency = frequency,
                                    catalog = radio_catalog)
    HD_radio_patch.write_map(f"{overall_path_example}radio_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_radio_dls, HD_radio_ells = spectra_HD.get_foreground_power(HD_radio_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}radio_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_radio_ells, "dl": HD_radio_dls})
    print(f"--- radio $frequency$GHz: %s seconds ---" % (time.time() - start_time))

################################################################### CIB Model 1

CIB_catalog = foregrounds_HD.make_CIB_model_catalog(CIB_model=1, CIB_catalog_original = pd.read_csv(f"{overall_path_example}S10_patches/CIB_14x14deg_source_catalog_ra=6_dec=6.csv"))

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    start_time = time.time()
    HD_CIB_patch = foregrounds_HD.generate_discrete_foreground(
                                    frequency = frequency,
                                    catalog = CIB_catalog)
    HD_CIB_patch.write_map(f"{overall_path_example}CIB_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_CIB_dls, HD_CIB_ells = spectra_HD.get_foreground_power(HD_CIB_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}CIB_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_CIB_ells, "dl": HD_CIB_dls})
    print(f"--- CIB1 $frequency$GHz: %s seconds ---" % (time.time() - start_time))

################################################################### CIB Model 2
#
#CIB_catalog = foregrounds_HD.make_CIB_model_catalog(CIB_model=2, CIB_catalog_original = pd.read_csv(f"{overall_path_example}S10_patches/CIB_14x14deg_source_catalog_ra=6_dec=6.csv"))
#
#mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")
#
#for frequency in [30,90,148,219,277,350]:
#    start_time = time.time()
#    HD_CIB_patch = foregrounds_HD.generate_discrete_foreground(
#                                    frequency = frequency,
#                                    catalog = CIB_catalog)
#    HD_CIB_patch.write_map(f"{overall_path_example}CIB_{frequency}GHz_12x12deg_ra=6_dec=6")
#
#    HD_CIB_dls, HD_CIB_ells = spectra_HD.get_foreground_power(HD_CIB_patch, mbb_inv, binning_file, deconvolve_pw = True)
#    np.save(f"{overall_path_example}CIB_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_CIB_ells, "dl": HD_CIB_dls})
#    print(f"--- CIB2 $frequency$GHz: %s seconds ---" % (time.time() - start_time))
#
################################################################### CMB (lensed)

start_time = time.time()

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

print(f"--- CMB: %s seconds ---" % (time.time() - start_time))

################################################################### CMB (theory)

data = CMB_patch.make_lensed_theory(HD_kappa_spectrum = np.load(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()])

np.savetxt(f'{overall_path_example}camb_full_output_lensed.dat', data, fmt='%.6f', delimiter='\t')