"""Define constants and default values used for the simulations."""

import numpy as np
import healpy as hp
import camb

# list of map frequencies (GHz):
freqs = [30, 90, 148, 219, 277, 350]
# list of all kinds of maps that can be generated:
all_components = ['ksz', 'tsz', 'cib', 'radio', 'kappa', 'cmb', 'unlensed_cmb']
# list of just lensed CMB (`'cmb'`) + foregrounds:
map_components = ['ksz', 'tsz', 'cib', 'radio', 'cmb']
# default list of components to generate:
components = [*map_components, 'kappa']

# HD sims
hd_res = 0.04
beam_fwhm = {30: 1.25, 90: 0.42, 148: 0.25, 219: 0.17, 277: 0.13, 350: 0.11} # arcmin
noise_level = {30: 6.5, 90: 0.7, 148: 0.8, 219: 2.0, 277: 2.7, 350: 100} # uK-arcmin

# defaults for sims:
ra_ctr = 6  # degrees
dec_ctr = 6 # degrees
width = 10  # degrees
height = 10 # degrees
apod_width = 0.5 # degrees

cmb_seed = 58 # for unlensed CMB
# define default noise seeds for temperature maps 
# NOTE : by default, when generating noise for TQU maps, we add 1000 or 2000 to get Q or U noise seed
noise_seeds = {freq: i+1 for (i, freq) in enumerate(freqs)}

lmax4alms = 24000 # lmax used when taking alms of high-res maps
lmax4theo = 40000 # lmax used when generating sim realization from theory spectrum (e.g. unlensed cmb)
lmax4spectra = 21000 # we want accurate power up to 20,000 , so add a few extra bins
bin_width = 200

spectra_col_names = ['ells', 'tt', 'ee', 'bb', 'te', 'kk'] # when saving CMB / lensing spectra (sim or theory) to file

# info needed to fit small-scale theory power to large-scale sim power:
ell_to_fit_amp = {'ksz': 3102, 'kappa': 3102}
ell_to_fit_slope = {'ksz': 8102, 'kappa': 3902}
smallscale_alms_seed = 3
lmin_for_smallscale_alms = {'ksz': 8000, 'kappa': 4000} # use theory alms above this ell

# HD accuracy settings for CAMB:
camb_version = int(camb.__version__.split('.')[0])
lens_margin_name = 'lens_output_margin' if (camb_version >= 2) else 'lens_margin'
hd_camb_accuracy_params = {'AccuracyBoost': 1.1,
                           'lAccuracyBoost': 3.0,
                           'lSampleBoost': 3.0,
                           'DoLateRadTruncation': False,
                           'min_l_logl_sampling': 10000,
                           lens_margin_name: 2050,
                           'lens_potential_accuracy': 30,
                           'halofit_version': 'mead2016',
                           'NonLinear': 'NonLinear_both',
                           'kmax': 100,
                           'k_per_logint': 130,
                          }
# define dicts to pass to each `camb.model.CAMBparams` method
# when updating accuracy of an existing `CAMBparams` instance:
camb_accuracy_params = {}
camb_matter_power_params = {}
camb_lmax_params = {}
for param in ['AccuracyBoost', 'lAccuracyBoost', 'lSampleBoost',
              'DoLateRadTruncation', 'min_l_logl_sampling']:
    camb_accuracy_params[param] = hd_camb_accuracy_params[param]
for param in ['kmax', 'k_per_logint']:
    camb_matter_power_params[param] = hd_camb_accuracy_params[param]
for param in [lens_margin_name, 'lens_potential_accuracy']:
    camb_lmax_params[param] = hd_camb_accuracy_params[param]


# S10 sims:
s10_nside = 8192
s10_kappa_nside = 4096
s10_res = hp.nside2resol(s10_nside, arcmin=True) 
s10_kappa_res = hp.nside2resol(s10_kappa_nside, arcmin=True)
s10_sim_components = ['kappa', 'ksz', 'tsz', 'cib', 'radio']
s10_ksz_map_freq = 90 # frequency (GHz) of the full-sky S10 kSZ map used (only need one b/c kSZ is frequency-independent)

# CIB models for HD sims:
baseline_cib_model_name = 'baseline'
alternative_cib_model_name = 'alternative'
s10_cib_model_name = 's10'
cib_model_names = [baseline_cib_model_name, alternative_cib_model_name, s10_cib_model_name]
cib_model_pixel_res = {baseline_cib_model_name: 0.25, alternative_cib_model_name: s10_res, s10_cib_model_name: hd_res}
cib_gauss_sigma_pix_frac = 0.2  # add Gaussian scatter to positions w/ sigma = 20% of CIB model pixel size 
cib_catalog_seed = 0 # seed used to add scatter

# S10 catalogs:
s10_catalog_col_names = {'cib': ['halo_id', 'RADeg', 'decDeg', 'z',
                                 'fluxmJy_30GHz', 'fluxmJy_90GHz', 'fluxmJy_148GHz',
                                 'fluxmJy_219GHz', 'fluxmJy_277GHz', 'fluxmJy_350GHz'],
                         'radio': ['RADeg', 'decDeg', 'z', 'fluxmJy_1.4GHz',
                                   'fluxmJy_30GHz', 'fluxmJy_90GHz', 'fluxmJy_148GHz',
                                   'fluxmJy_219GHz', 'fluxmJy_277GHz', 'fluxmJy_350GHz'],
                         'sz': ['z', 'RADeg', 'decDeg',
                                # comoving position of halo potential minimum in Mpc :
                                'comoving_pos_x_Mpc', 'comoving_pos_y_Mpc', 'comoving_pos_z_Mpc',
                                # proper peculiar velocity in km/s (for x, y, z?) :
                                'pec_vel_x_km_s', 'pec_vel_y_km_s', 'pec_vel_z_km_s',
                                'Mfof', # Mfof in Msolar

                                'Mvir', 'Mgas_vir', # in Msolar
                                'Rvir', # in proper Mpc
                                'int_tSZ_in_Rvir', # Integrated TSZ within Rvir in arcminute^2
                                'int_kSZ_in_Rvir', # Integrated KSZ within Rvir in arcminute^2
                                # Integrated SZ within Rvir at 148,219,277,30,90,350 GHz:
                                'int_SZ_in_Rvir_148GHz', 'int_SZ_in_Rvir_219GHz', 'int_SZ_in_Rvir_277GHz',
                                'int_SZ_in_Rvir_30GHz', 'int_SZ_in_Rvir_90GHz', 'int_SZ_in_Rvir_350GHz',

                                'M200', 'Mgas_200', # in Msolar
                                'R200', # in proper Mpc
                                'int_tSZ_in_R200', # Integrated TSZ within R200 in arcminute^2
                                'int_kSZ_in_R200', # Integrated KSZ within R200 in arcminute^2
                                # Integrated SZ within R200  at 148,219,277,30,90,350 GHz:
                                'int_SZ_in_R200_148GHz', 'int_SZ_in_R200_219GHz', 'int_SZ_in_R200_277GHz',
                                'int_SZ_in_R200_30GHz', 'int_SZ_in_R200_90GHz', 'int_SZ_in_R200_350GHz',

                                'M500', 'Mgas_500', # in Msolar
                                'R500', # in proper Mpc
                                'int_tSZ_in_R500', # Integrated TSZ within R500 in arcminute^2
                                'int_kSZ_in_R500', # Integrated KSZ within R500 in arcminute^2
                                # Integrated SZ within R500  at 148,219,277,30,90,350 GHz:
                                'int_SZ_in_R500_148GHz', 'int_SZ_in_R500_219GHz', 'int_SZ_in_R500_277GHz',
                                'int_SZ_in_R500_30GHz', 'int_SZ_in_R500_90GHz', 'int_SZ_in_R500_350GHz',

                                'Mstar', # in Msolar
                                'rho_gas_c', # central gas density in Msolar/Mpc^3
                                'T_c', # central temperature in keV
                                'P_c', # central pressure    in Msolar/Mpc/Gyr^2
                                'Phi_c', # central potential   in (Mpc/Gyr)^2
                               ],
                        }

s10_catalog_ra_col_index = {'cib': 1, 'radio': 0, 'sz': 1}
s10_catalog_dec_col_index = {'cib': 2, 'radio': 1, 'sz': 2}
