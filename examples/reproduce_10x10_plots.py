import os
import argparse
import numpy as np
from hdsims import hdsims, utils

# define command-line args and parse them:
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('hd_sims_dir', help='path to your hdsims output directory')
parser.add_argument('--fig2fname', default='fig2.png', help='file name for the figure 2 plot')
parser.add_argument('--fig3fname', default='fig3.pdf', help='file name for the figure 3 plot')
parser.add_argument('--fig4fname', default='fig4.pdf', help='file name for the figure 4 plot')
args = parser.parse_args()

# initialize the HDSims class: 
log = utils.get_logger(name='example', fmt="{message:s}") # use logging to print out messages as they are logged
simlib = hdsims.HDSims(args.hd_sims_dir, verbose=True, log=log) 
# fig. 2 (90 GHz sims):
simlib.plot_sim_maps(90, save=True, fname=args.fig2fname)
# fig 3:
hdsims.reproduce_sim_spectra_comparison_plot(fname=args.fig3fname)
# fig 4:
hdsims.reproduce_ksz_kappa_sim_spectra_plot(fname=args.fig4fname)

