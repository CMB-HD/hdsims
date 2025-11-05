import os
import argparse
import numpy as np
from hdsims import hdsims, utils

# define command-line args and parse them:
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('hd_sims_dir', help='path to your hdsims output directory')
parser.add_argument('lowres_sims_dir', help='path to the full-sky S10 sims & catalogs')
args = parser.parse_args()

# initialize the HDSims class: 
log = utils.get_logger(name='example', fmt="{message:s}") # use logging to print out messages as they are logged
simlib = hdsims.HDSims(args.hd_sims_dir, lowres_sims_dir=args.lowres_sims_dir, verbose=True, log=log) 
# run everything and reproduce figures:
simlib.generate_and_powerspectra_hd_sims(save_intermediate_maps=True, save_intermediate_map_power=True, make_plots=True, plot_fdiff=True, use_fig3_settings=True)



