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

import camb
ini_file = 'S10_data/bode_almost_wmap5_params_highKeta.ini'
pars = camb.read_ini(ini_file)
# high-accuracy settings
pars.set_matter_power(kmax=10, k_per_logint=130)
pars.set_for_lmax (24000, \
    lens_potential_accuracy =30 , \
    lens_margin =2050)
pars.set_accuracy ( AccuracyBoost =1.1 , \
    lSampleBoost =3.0 , lAccuracyBoost =3.0 , \
    DoLateRadTruncation = False, min_l_logl_sampling=10000 )
pars.NonLinear = camb.model.NonLinear_both
pars.NonLinearModel.set_params("mead2016")
results = camb.get_results(pars)

lensed = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
unlensed = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['unlensed_scalar']

ells = np.arange(0, lensed.shape[0])
lensCL = results.get_lens_potential_cls(lmax=lensed.shape[0] + 2)
kk = (ells * (ells + 1))**2 * lensCL[2:len(ells)+2, 0] / 4.0
main_data = np.column_stack([
    ells,
    lensed[:,0], lensed[:,1], lensed[:,2], lensed[:,3],
    unlensed[:,0], unlensed[:,1], unlensed[:,2], unlensed[:,3],
    kk
])

np.savetxt(
    f'{overall_path_example}camb_full_output.txt',
    main_data,
    header='ell    tt_lensed   ee_lensed   bb_lensed   te_lensed   '
           'tt_unlensed ee_unlensed bb_unlensed te_unlensed kk',
    fmt='%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e'
)
main_data = np.column_stack([
    ells, 
    unlensed[:,0] * (ells * (ells+1)) / (2*np.pi), 
    unlensed[:,1] * (ells * (ells+1)) / (2*np.pi), 
    unlensed[:,2] * (ells * (ells+1)) / (2*np.pi), 
    unlensed[:,3] * (ells * (ells+1)) / (2*np.pi)
])

np.savetxt(f'{overall_path_example}/camb_full_output.dat', main_data, fmt='%.6f', delimiter='\t')

hddata = np.loadtxt(f"{overall_path_example}camb_full_output.txt", comments="#")
template_ells =  hddata[:,0]
template_cls = hddata[:,9]
template_cls *= 2*np.pi/(template_ells * (template_ells + 1))**2

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
S10_kappa_cls, S10_kappa_ells = spectra_HD.get_foreground_power(S10_kappa_patch, mbb_inv, binning_file, deconvolve_pw = False, type_Cl = True)
np.save(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_S10upsampledspectra.npy", {"l": S10_kappa_ells, "cl": S10_kappa_cls})

smallScale_kappa_alms, smallScale_kappa_ells, smallScale_kappa_cls = foregrounds_HD_kappa.get_theory_for_stitching(template_cls,
                                                                              template_ells,
                                                                              S10_kappa_cls, 
                                                                              S10_kappa_ells, \
                                                                              template_minimization_index = 3902, 
                                                                              patch_minimization_index = 19)
np.save(f"{overall_path_example}kappa_smallscale_theory_spectra.npy", {"l": smallScale_kappa_ells, "cl": smallScale_kappa_cls})

kappa_alms_for_stitching_resized = foregrounds_HD_kappa.get_S10_for_stitching(S10_kappa_patch)
HD_kappa_patch = foregrounds_HD_kappa.stitch_alms(alms_theory = smallScale_kappa_alms, alms_S10 = kappa_alms_for_stitching_resized, 
                                                   l_cutoff = 4000)
HD_kappa_patch.write_map(f"{overall_path_example}kappa_14x14deg_ra=6_dec=6")

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
    S10_kSZ_cls, S10_kSZ_ells = spectra_HD.get_foreground_power(S10_kSZ_patch, mbb_inv, binning_file, deconvolve_pw = True, type_Cl = True)
    smallScale_kSZ_alms, smallScale_kSZ_ells, smallScale_kSZ_cls = foregrounds_HD.get_theory_for_stitching(template_cls,
                                                                          template_ells,
                                                                          S10_kSZ_cls, 
                                                                          S10_kSZ_ells, \
                                                                          template_minimization_index = 8102, 
                                                                          patch_minimization_index = 40)
    np.save(f"{overall_path_example}kSZ_{frequency}GHz_smallscale_theory_spectra.npy", {"l": smallScale_kSZ_ells, "cl": smallScale_kSZ_cls})
    
    kSZ_alms_for_stitching_resized = foregrounds_HD.get_S10_for_stitching(S10_kSZ_patch)
    HD_kSZ_patch_extended = foregrounds_HD.stitch_alms(alms_theory = smallScale_kSZ_alms, alms_S10 = kSZ_alms_for_stitching_resized, 
                                                   l_cutoff = 8000)
    HD_kSZ_patch_extended.write_map(f"{overall_path_example}kSZ_{frequency}GHz_12x12deg_ra=6_dec=6")

    mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/")
    HD_kSZ_dls, HD_kSZ_ells = spectra_HD.get_foreground_power(HD_kSZ_patch_extended, mbb_inv, binning_file, deconvolve_pw = True)
    np.save(f"{overall_path_example}kSZ_{frequency}GHz_10x10deg_ra=6_dec=6_spectra.npy", {"l": HD_kSZ_ells, "dl": HD_kSZ_dls})
