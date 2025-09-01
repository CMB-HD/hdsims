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

from scipy.interpolate import interp1d
import camb
lmax = 24000

# original CAMB `.ini` file used in S10 sims :
ini_file = f'S10_data/bode_almost_wmap5_params_highKeta.ini'

# power of the high-resolution kappa sim in 10x10 patch w/ ctr at RA=6, dec=6:
sim_clkk = np.load(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]['cl']
sim_Lbin = np.load(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]['l']

# read in the original CAMB `.ini` file used in S10 sims & update accuracy:
pars = camb.read_ini(ini_file)
pars.set_matter_power(kmax=10, k_per_logint=130)
pars.set_for_lmax(lmax+500, lens_potential_accuracy=30, lens_margin=2050)
pars.set_accuracy( AccuracyBoost =1.1 , \
    lSampleBoost =3.0 , lAccuracyBoost =3.0 , \
    DoLateRadTruncation = False, min_l_logl_sampling=10000 )
pars.NonLinear = camb.model.NonLinear_both
pars.NonLinearModel.set_params("mead2016")

# get the CAMB unlensed CMB & lensing convergence theory:
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit='muK', raw_cl=True)
clkk = results.get_lens_potential_cls(lmax=lmax)[:,0] * 2 * np.pi / 4
ells = np.arange(lmax+1)
# lens the unlensed theory using the power of the kappa sim for L > 300 & the CAMB clkk theory for L < 300:
# CAMB needs clkk at each L, so we need to interpolate the binned sim power

sim_Lmin_interp = 30 # use sim power above this L
sim_Lbin_to_interp = sim_Lbin[sim_Lbin >= sim_Lmin_interp]
sim_clkk_to_interp = sim_clkk[sim_Lbin >= sim_Lmin_interp]
theo_Lmax_interp = round(sim_Lbin_to_interp[0]) - 1 # use CAMB theory below this L

Ls_to_interp = np.concatenate([ells[:theo_Lmax_interp], sim_Lbin_to_interp])
clkk_to_interp = np.concatenate([clkk[:theo_Lmax_interp], sim_clkk_to_interp])

camb_ells = np.arange(pars.max_l+1) # CAMB needs a clkk curve to higher Lmax than it will output
clkk_interp = interp1d(Ls_to_interp, clkk_to_interp, bounds_error=False, fill_value=clkk_to_interp[-1])(camb_ells)

# get the lensed CMB theory using the kappa power in the patch:
lensed_powers = results.get_lensed_cls_with_spectrum(clkk_interp * 4 / (2 * np.pi), lmax=lmax, CMB_unit='muK', raw_cl=True)
lensed_theory = {'ells': ells.copy()}
for i, s in enumerate(['tt', 'ee', 'bb', 'te']):
    lensed_theory[s] = lensed_powers[:,i].copy()
    lensed_theory[s][:2] = 0

data = np.column_stack([
    lensed_theory['ells'],
    lensed_theory['tt'] * lensed_theory['ells'] * (lensed_theory['ells']+1)/(2*np.pi), 
    lensed_theory['ee'] * lensed_theory['ells'] * (lensed_theory['ells']+1)/(2*np.pi), 
    lensed_theory['bb'] * lensed_theory['ells'] * (lensed_theory['ells']+1)/(2*np.pi), 
    lensed_theory['te'] * lensed_theory['ells'] * (lensed_theory['ells']+1)/(2*np.pi),
])

np.savetxt(f'{overall_path_example}camb_full_output_lensed.dat', data, fmt='%.6f', delimiter='\t')