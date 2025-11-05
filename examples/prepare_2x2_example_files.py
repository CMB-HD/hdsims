import os
import argparse
import numpy as np
from hdsims import hdsims, utils, siminfo as si, simutils


# define command-line args and parse them:
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('hd_sims_dir', help='path to your hdsims output directory')
parser.add_argument('--lowres_sims_dir', default=None, help='path to the full-sky S10 sims & catalogs')
parser.add_argument('--freqs', default=[90], type=int, nargs='*', help='sim frequencies (in GHz)')
parser.add_argument('--all', action='store_true', help='generate all HD sims and calculate their power spectra')
parser.add_argument('--intermediate_maps', action='store_true', help='generate the intermediate, S10-resolution maps and catalogs for our patch of sky, beginning from the full-sky S10 sims')
parser.add_argument('--inv_mcm', action='store_true', help='calculate the inverse mode-coupling matrices')
parser.add_argument('--cmb', action='store_true', help='generate the HD lensed CMB sim')
args = parser.parse_args()

# initialize the HDSims class for this example:
width = 2          # width (in degrees) of region in sim maps that will ultimately be used for power spectra
height = width     # height (in degrees) of region in sim maps that will ultimately be used for power spectra
apod_width = 0.25  # apodization width (in degrees) used to apodize sims before taking their power
components = si.components
freqs = args.freqs
log = utils.get_logger(name='example', fmt="{message:s}") # use logging to print out messages as they are logged
simlib = hdsims.HDSims(args.hd_sims_dir, lowres_sims_dir=args.lowres_sims_dir, verbose=True, log=log, 
                       freqs=freqs, components=components, width=width, height=height, apod_width=apod_width)

if args.intermediate_maps:
    # save CAR maps at S10 resolution of tSZ/kSZ/lensing convergence on our patch of sky :
    # for the tSZ/kSZ maps, we need to deconvolve the pixel window from the full-sky healpix map
    #  and convert to uK units ; for the tSZ, we also multiply the map by 0.75
    diffuse_s10_components = [c for c in components if (c in ['tsz', 'ksz', 'kappa'])]
    for component in diffuse_s10_components:
        sim_freqs = [None] if (component in ['ksz', 'kappa']) else freqs
        for freq in sim_freqs:
            simlib.cutout_car_patch_from_fullsky_healpix(component, freq=freq, save=True)
    # save catalogs of all S10 sources/clusters in our patch of sky:
    simlib.save_s10_catalogs_for_patch()
    # generate cib/radio sims at the S10 resolution:
    discrete_s10_components = [c for c in components if (c in ['cib', 'radio'])]
    for component in discrete_s10_components:
        simlib.generate_intermediate_point_source_maps(component, save_intermediate_maps=True)

if args.inv_mcm:
    # for intermediate sims at S10 resolution:
    fg_components = [c for c in components if (c in ['tsz', 'ksz', 'cib', 'radio'])]
    if len(fg_components) > 0:
        for bin_dl in [True, False]:
            simlib.get_intermediate_inv_mcm('tsz', bin_dl=bin_dl) # 'tsz' just sets the map resolution to the S10 resoluton (0.43 arcmin)
    if 'kappa' in components: # S10 kappa maps have 0.86 arcmin resolution, so we need to calculate a separate inv mcm
        simlib.get_intermediate_inv_mcm('kappa', bin_dl=False)
    # for HD sims:
    if (len(fg_components) > 0) or ('cmb' in components):
        simlib.get_mode_coupling(bin_dl=True)
    if ('kappa' in components) or ('cmb' in components):
        simlib.get_mode_coupling(bin_dl=False)

if args.cmb:
    # generate the lensed cmb sim:
    simlib.generate_lensed_cmb_sim()
    # calculate the lensed cmb sim theory:
    inv_mcm_fname, _ = simlib.get_mode_coupling_fnames(bin_dl=False)
    if not os.path.exists(inv_mcm_fname): # inv mcm hasn't been saved yet, so use the precomputed kappa sim power
        sim_clkk_fname = simlib.get_signal_sim_power_fname('kappa')
        example_sim_clkk_fname = sim_clkk_fname.replace(args.hd_sims_dir, simutils.precomputed_hdsims_output_dir())
        sim_clkk = utils.load_dict_from_file(example_sim_clkk_fname, ['ells', 'kk'])
        utils.save_dict_to_file(sim_clkk_fname, sim_clkk, keys=['ells', 'kk'])
    simlib.get_sim_theory('cmb')

if args.all:
    # generate the sims and take their power, but make the plots later in the example notebook
    simlib.generate_and_powerspectra_hd_sims(save_intermediate_maps=True, save_intermediate_map_power=True, make_plots=False)


