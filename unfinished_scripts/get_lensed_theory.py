import camb
import numpy as np
from scipy.interpolate import interp1d

lmax = 24000
overall_path_full = '/gpfs/projects/SehgalGroup/jange/hdsims/output/'

# original CAMB `.ini` file used in S10 sims :
ini_file = 'raw_data/bode_almost_wmap5_params_highKeta.ini'

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os
from matplotlib.colors import LinearSegmentedColormap
from pixell import enmap, enplot, colorize, utils, wcsutils, curvedsky as cs

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.dpi'] = 250
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.xmargin'] = 0.025
plt.rcParams['axes.ymargin'] = 0.025
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['figure.figsize'] = (5, 3)
if 'planck' not in mpl.colormaps:
    colorize.mpl_setdefault('planck')

# power of the high-resolution kappa sim in 10x10 patch w/ ctr at RA=6, dec=6:
sim_Lbin = hp.read_cl(f"{overall_path_full}kappa_stitched/none/0.04/ells.fits")
sim_clkk = hp.read_cl(f"{overall_path_full}kappa_stitched/none/0.04/cls.fits")

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

sim_Lmin_interp = 300 # use sim power above this L
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

main_data = np.column_stack([
    lensed_theory['ells'],
    lensed_theory['tt'], lensed_theory['ee'], lensed_theory['bb'], lensed_theory['te'],
])

np.savetxt(
    'output/camb_full_output_lensed.txt',
    main_data,
    header='ell    tt_lensed   ee_lensed   bb_lensed   te_lensed   ',
    fmt='%d %.6e %.6e %.6e %.6e'
)