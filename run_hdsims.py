import os
import argparse
from hdsims import hdsims, utils, siminfo as si


# --- set up to parse command-line arguments ---

# define descriptions that will be printed out in the 'help' message:
description = "Generate ultrahigh-resolution microwave sky simulations and, by default, calculate their power spectra and plot the results."
epilog = "See the documentation of the `hdsims.hdsims.HDSims` class for additional options that are not included above."

help_text = {'hd_sims_dir': ("The path to the directory where all of the output files will be saved. This directory"
                             " will be created if it does not already exist. A new sub-directory within the"
                             " `hd_sims_dir` will be created for the simulations on a given patch of sky."),
             'lowres_sims_dir': ("The path to the directory where the full-sky lower-resolution simulations and"
                                 " catalogs have been saved. This is required if the ultrahigh-resolution simulations"
                                 " have not already been generated and saved; otherwise, it is not used."),
             'ra': "The right ascension (in degrees) of the map center.",
             'dec': "The declination (in degrees) of the map center.",
             'width': ("The width (in degrees) of the maps. The `width` and `height` define the final useable area"
                       " of the maps (i.e., the inner un-apodized region) after apodizing and convolving them with"
                       " the beam; the maps will be saved on a slightly larger patch of sky to leave room for this"
                       " apodization."),
             'height': "The height (in degrees) of the maps.",
             'freqs': ("The map frequencies (in GHz) to generate simulations for."
                       " By default, all allowed frequencies will be used."),
             'components': ("A list of individual map components to generate. By default, all available components are"
                            " generated. If `'cmb'` (lensed CMB) is in the list, then unlensed CMB (`'unlensed_cmb'`)"
                            " and lensing convergence (`'kappa'`) simulations will also be generated and saved. "),
             'apod': ("The width (in degrees) of the region along each edge of the map that will be apodized when"
                      " calculating power spectra."),
             'cmbseed': ("The random seed to use when generating a realization of the unlensed CMB from"
                         " theory power spectra."),
             'nopol': ("By default, CMB temperature and polarization (T, Q, and U) maps will be generated."
                       " If `nopol` is passed, only the temperature map will be generated."),
             'nospectra': ("Pass `nospectra` if you do not want to calculate the power spectra of the simulations;"
                           " by default, the power spectra will be calculated."),
             'noplots': ("Pass `noplots` if you do not want to save plots of the simulations and their power spectra."
                         "By default, the plots will be saved."),
             'noverbose': ("Pass `noverbose` if you do not want to print any messages about the progress of the"
                           " calculations."),
            }

# define command-line args and parse them:
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description, epilog=epilog)
parser.add_argument('hd_sims_dir', help=help_text['hd_sims_dir'])
parser.add_argument('--lowres-sims-dir', default=None, help=help_text['lowres_sims_dir'])
# options for the sims that will be generated:
parser.add_argument('--ra', default=si.ra_ctr, type=float, help=help_text['ra'])
parser.add_argument('--dec', default=si.dec_ctr, type=float, help=help_text['dec'])
parser.add_argument('--width', default=si.width, type=float, help=help_text['width'])
parser.add_argument('--height', default=si.height, type=float, help=help_text['height'])
parser.add_argument('--freqs', default=si.freqs, type=int, nargs='*', help=help_text['freqs'])
parser.add_argument('--components', default=si.components, type=str, nargs='*', help=help_text['components'])
parser.add_argument('--apod', default=si.apod_width, type=float, help=help_text['apod'])
parser.add_argument('--cmbseed', default=si.cmb_seed, type=int, help=help_text['cmbseed'])
parser.add_argument('--nopol', action="store_true", help=help_text['nopol'])
# options for what will be calculated/saved in addition to the HD sims:
parser.add_argument('--nospectra', action="store_true", help=help_text['nospectra'])
parser.add_argument('--noplots', action="store_true", help=help_text['noplots'])
parser.add_argument('--noverbose', action="store_true", help=help_text['noverbose'])

args = parser.parse_args()
calc_spectra = not args.nospectra
make_plots = not args.noplots


# --- run hdsims: ---

# initialize the HDSims class: 
log = utils.get_logger(name='hdsims', fmt="{message:s}") # use logging to print out messages as they are logged
simlib = hdsims.HDSims(os.path.abspath(args.hd_sims_dir), lowres_sims_dir=args.lowres_sims_dir,
                       freqs=args.freqs, components=args.components, 
                       ra_ctr=args.ra, dec_ctr=args.dec, width=args.width, height=args.height,
                       apod_width=args.apod, cmb_seed=args.cmbseed, pol=(not args.nopol), 
                       verbose=(not args.noverbose), log=log) 


if calc_spectra: # can do everything in one line: 
    simlib.generate_hd_sims_and_calculate_powerspectra(save_intermediate_maps=True, save_intermediate_map_power=True, make_plots=make_plots)

else: # only generate the simulations (and plot them, if requested):
    simlib.generate_hd_sims(save_intermediate_maps=True)
    # save total CMB + FG maps:
    simlib.infomsg(f"getting total lensed CMB temperature + FG sims for {simlib.freqs} GHz")
    for freq in simlib.freqs:
        simlib.get_total_signal_sim(freq=freq, save=True, pol=False) 
    
    if make_plots: # just plot the maps
        freqs_to_plot = [freq for freq in [90, 148] if (freq in simlib.freqs)] # default
        if len(freqs_to_plot) == 0: # then we can't use the default ...
            freqs_to_plot = simlib.freqs # ... so just use all available freqs
        for freq in freqs_to_plot:
            simlib.plot_sim_maps(freq, save=True)

