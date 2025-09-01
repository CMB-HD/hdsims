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

################################################################### kappa

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
HD_kappa_patch, kappa_theory_power = foregrounds_HD_kappa.extend_to_small_scales(S10_patch = S10_kappa_patch, patch_type='kappa', mbb_inv=mbb_inv, binning_file=binning_file)

HD_kappa_patch.write_map(f"{overall_path_example}kappa_14x14deg_ra=6_dec=6")
np.save(f"{overall_path_example}kappa_smallscale_theory_spectra.npy", kappa_theory_power)

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
HD_kappa_cls, HD_kappa_ells = spectra_HD.get_foreground_power(HD_kappa_patch, mbb_inv, binning_file, deconvolve_pw = False, type_Cl = True)
np.save(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_kappa_ells, "cl": HD_kappa_cls})

################################################################### tSZ

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")

for frequency in [30,90,148,219,277,350]:
    HD_tSZ_patch = foregrounds_HD.generate_diffuse_foreground(
                    component = 'tSZ',
                    frequency = frequency,
                    fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/{frequency}GHz_tSZ_fullsky_deconvolved")
    HD_tSZ_patch.write_map(f"{overall_path_example}tSZ_{frequency}GHz_12x12deg_ra=6_dec=6")

    HD_tSZ_dls, HD_tSZ_ells = spectra_HD.get_foreground_power(HD_tSZ_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}tSZ_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_tSZ_ells, "dl": HD_tSZ_dls})

################################################################### kSZ

ksz_template = pd.read_csv("S10_data/cmbhd_mockdata_ksz_cls_v1.1.txt", delimiter=' ')
template_ells = ksz_template['ell']
template_cls = ksz_template['C_ell^kSZ']

for frequency in [30,90,148,219,277,350]:
    S10_kSZ_patch = foregrounds_HD.generate_diffuse_foreground(
                component = 'kSZ',
                frequency = frequency,
                fullsky_deconvolved_path = f"S10_data/deconvolved_fullsky/{frequency}GHz_kSZ_fullsky_deconvolved")
    
    mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
    HD_kSZ_patch, kSZ_theory_power = foregrounds_HD_kappa.extend_to_small_scales(S10_patch = S10_kSZ_patch, patch_type='kSZ', mbb_inv=mbb_inv, binning_file=binning_file)
    
    HD_kSZ_patch.write_map(f"{overall_path_example}kSZ_{frequency}GHz_12x12deg_ra=6_dec=6")
    np.save(f"{overall_path_example}kSZ_{frequency}GHz_smallscale_theory_spectra.npy", kSZ_theory_power)

    mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")
    HD_kSZ_dls, HD_kSZ_ells = spectra_HD.get_foreground_power(HD_kSZ_patch, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}kSZ_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_kSZ_ells, "dl": HD_kSZ_dls})
