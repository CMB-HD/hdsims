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

################################################################### Fig. 2

map_path = lambda x: os.path.join(overall_path_example, x)

map_names = ['tSZ', 'kSZ', 'CIB', 'radio', 'kappa', 'CMB_T']

map_fnames = {'tSZ': map_path(f'tSZ_90GHz_12x12deg_ra=6_dec=6'),
              'kSZ': map_path(f'kSZ_90GHz_12x12deg_ra=6_dec=6'),
              'CIB': map_path(f'CIB_90GHz_12x12deg_ra=6_dec=6'),
              'radio': map_path(f'radio_90GHz_12x12deg_ra=6_dec=6'),
              'kappa': map_path(f'kappa_14x14deg_ra=6_dec=6'),
              'CMB_T': map_path(f'CMB_Lensed_12x12deg_ra=6_dec=6')}
maps = {name: so_map.read_map(map_fnames[name]).data for name in map_names}
maps['CMB_T'] = maps['CMB_T'][0]

map_labels = {'tSZ': 'tSZ',
              'kSZ': 'kSZ',
              'CIB': 'CIB',
              'radio': 'Radio Sources',
              'kappa': 'Lensing Convergence',
              'CMB_T': 'Lensed CMB'}

map_cbar_lims = {'tSZ': [-100, 0],
              'kSZ': [-15, 15],
              'CIB': [0, 25],
              'radio': [0, 10],
              'kappa': [-0.5, 0.5],
              'CMB_T': [-350, 350]}

cmaps = {'tSZ': neg_cmap, 'kSZ': planck_cmap, 'CIB': pos_cmap, 'radio': pos_cmap, 'kappa': planck_cmap, 'CMB_T': planck_cmap}

nrow = 3
ncol = 2

figsize = (4.25*ncol, 3.75*nrow)

fig = plt.figure(dpi=300, figsize=figsize)
axs = []
for i, map_name in enumerate(map_names):
    cbar_label = '' if (map_name == 'kappa') else r'$\mu$K'
    ax = fig.add_subplot(nrow, ncol, i+1)
    fig, ax = upl.plot_map(maps[map_name], fig=fig, ax=ax, show=False, grid=False,
                       cbar_shrink=0.8,
                       cmap=cmaps[map_name], clim=map_cbar_lims[map_name][1], vmin=map_cbar_lims[map_name][0], cbar_label=cbar_label,
                       plt_ra_min=1, plt_ra_max=11, plt_dec_min=1, plt_dec_max=11)
    ax.text(10, 10, map_labels[map_name], bbox=dict(facecolor='w', alpha=0.75), 
            horizontalalignment='left', verticalalignment='top',
            fontsize=12)
    ax.grid(False)
    axs.append(ax)

plt.subplots_adjust(wspace=0.35)
plt.tight_layout()
plt.show()

plt.savefig("Fig2_12x12.pdf")

################################################################### Fig. 3

hd_spectra = {
    'tSZ': {
        30: np.load(f"{overall_path_example}tSZ_30GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}tSZ_90GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}tSZ_148GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}tSZ_219GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}tSZ_277GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}tSZ_350GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
    },
    'kSZ': {
        30: np.load(f"{overall_path_example}kSZ_30GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}kSZ_90GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}kSZ_148GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}kSZ_219GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}kSZ_277GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}kSZ_350GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
    },
    'CIB': {
        30: np.load(f"{overall_path_example}CIB_30GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}CIB_90GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}CIB_148GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}CIB_219GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}CIB_277GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}CIB_350GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
    },
    'radio': {
        30: np.load(f"{overall_path_example}radio_30GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}radio_90GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}radio_148GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}radio_219GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}radio_277GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}radio_350GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
    },
    'kappa': {
        90: np.load(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
    },
    'CMB_T': {
        90: {'l': np.load(f"{overall_path_example}CMB_Lensed_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]['l'], 
             'dl': np.load(f"{overall_path_example}CMB_Lensed_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]['dl']['TT']}
    }
}
s10_spectra = {
    'tSZ': {
        30: np.load(f"{overall_path_example}S10_patches/tSZ_30GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}S10_patches/tSZ_90GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}S10_patches/tSZ_148GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}S10_patches/tSZ_219GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}S10_patches/tSZ_277GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}S10_patches/tSZ_350GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
    },
    'kSZ': {
        30: np.load(f"{overall_path_example}S10_patches/kSZ_30GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}S10_patches/kSZ_90GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}S10_patches/kSZ_148GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}S10_patches/kSZ_219GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}S10_patches/kSZ_277GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}S10_patches/kSZ_350GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
    },
    'CIB': {
        30: np.load(f"{overall_path_example}S10_patches/CIB_30GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}S10_patches/CIB_90GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}S10_patches/CIB_148GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}S10_patches/CIB_219GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}S10_patches/CIB_277GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}S10_patches/CIB_350GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
    },
    'radio': {
        30: np.load(f"{overall_path_example}S10_patches/radio_30GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        90: np.load(f"{overall_path_example}S10_patches/radio_90GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        148: np.load(f"{overall_path_example}S10_patches/radio_148GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        219: np.load(f"{overall_path_example}S10_patches/radio_219GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        277: np.load(f"{overall_path_example}S10_patches/radio_277GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()],
        350: np.load(f"{overall_path_example}S10_patches/radio_350GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
    },
    'kappa': {
        90: np.load(f"{overall_path_example}S10_patches/kappa_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
    }
}

spectra_HD = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = 0.04,
    apod_width = 1.0,
    l_max = 24000)

mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", spin0and2=True)
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
ℓ, ps_theory = pspy_utils.ps_lensed_theory_to_dict(f"{overall_path_example}camb_full_output_lensed.dat", "Dl", lmax=24000)
ps_theory_b = so_mcm.apply_Bbl(Bbl, ps_theory, spectra=spectra)
theo = {
    'CMB_T': {
        'l': hd_spectra['CMB_T'][90]['l'], 
        'dl': ps_theory_b['TT']
    }
}

freqs = [30,90,148,219,277,350]

plt_components = ['tSZ', 'CIB', 'kappa',
                  'kSZ', 'radio', 'CMB_T']
components = plt_components.copy()

hd_fdiff_spectra, s10_fdiff_spectra = upl.unify_ells(hd_spectra, s10_spectra)

s10_kwargs = {'ls': ':', 'lw': 2, 'alpha': 0.95}
hd_kwargs = {'ls': '-', 'lw': 1.75, 'alpha': 0.7, }
fdiff_kwargs = {'ls': '-', 'lw': 1}#, 'alpha': 0.7}
freq_label_kwargs = {'ls': '-', 'lw': 1.5}


hd_colors = ['tab:pink', 'tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:blue']
hd_colors = hd_colors[::-1]
s10_colors = ['k'] * len(hd_colors) #dim_colors#lightcolors
s10_label_color = 'k'#'tab:gray'
#hd_colors = colors#darkcolors



hd_label_color = 'k'
fdiff_colors = hd_colors

plt_spec_types = {component: 'dl' for component in plt_components}
plt_spec_types['kappa'] = 'cl'

plt_lmin = 200#30
plt_lmax = 20000
s10_lmaxs = {component: 8000 for component in components}
s10_lmaxs['kappa'] = 4000 #5800
s10_lmaxs['kSZ'] = 7700
s10_lmaxs['CMB_T'] = plt_lmax # (this is really for theory, just easier to use same dict)

ylims = {'tSZ': [-0.5, 22], 'kSZ': [-0.05, 2.5], 'radio': [5e-1, 1.5e8], 'CIB': [5e-3, 1.5e5], 'kappa': None, 'CMB_T': [7e7, 1.2e10]}
yticks = {'tSZ': None,
          'kSZ': None,
          'CIB': [0.1, 10, 1e3, 1e5],
          'radio': [10, 1e3, 1e5, 1e7],
          'kappa': None,
          'CMB_T': None,
         }

fdiff_ylims = {'tSZ': [-0.1, 0.05], 
               'kSZ': [-0.06, 0.09], 
               'radio': [-2, 2], 
               'CIB': [-1, 3],
               'kappa': None,
               'CMB_T': None
              }
logy = {'CIB': True, 'radio': True, 
        'tSZ': False, 'kSZ': False,
        #'tsz': True, 'ksz': True,
        'kappa': True, 'CMB_T': True,
       }
logx = {'CIB': False, 'radio': False, 
        'tSZ': False, 'kSZ': False,
        #'kappa': True, 'cmb': True,
        'kappa': False, 'CMB_T': False,
       }


s10_label = "Original S10"# (0.43')"
kappa_s10_label = "Original S10"# (0.86')"
hd_label = "This work"# (0.04')"
xlabels = {component: r'Multipole, $\ell$' for component in plt_components}
ylabels = {component: r'$\frac{\ell(\ell+1)}{2\pi} C_\ell~$ [$\mu$K$^2$]' for component in plt_components}
xlabels['kappa'] = r'Multipole, $L$'
ylabels['kappa'] = r'$\frac{L^2 (L+1)^2}{4} C_L^{\phi\phi}$'
ylabels['CMB_T'] =  r'$\ell^4 C_\ell~$ [$\mu$K$^2$]'
fdiff_ylabel = '% Diff.'#erence'
fdiff_labels = {component: 'This work vs. S10' for component in plt_components}
fdiff_labels['CMB_T'] = 'This work vs. theory'
labels = {'tSZ': 'tSZ', 'kSZ': 'kSZ', 'CIB': 'CIB', 'radio': 'Radio', 'kappa': 'Lensing convergence', 'CMB_T': 'Lensed CMB'}

axis_labelsize = 12
component_labelsize = 14


nrow = int(len(plt_components) / 2)
ncol = 2
figsize = (4*ncol, 3.5*nrow)

fdiff_plt_frac = 0.25 # fraction of plot used for frac. diff., for each component

ksz_legend_kwargs = {'ncol': 2,
                 #'borderpad': 0.35,
                 'handlelength': 1.5,
                 'handletextpad': 0.5,
                 'columnspacing': 1,
                }
kappa_legend_kwargs = {'ncol': 1,
                        # 'borderpad': 0.35,
                         'handlelength': 1.5,
                         'handletextpad': 0.5,
                         'columnspacing': 1,
                       'loc': 'center right',
                        }
fdiff_legend_kwargs = {'ncol': 1,
                        # 'borderpad': 0.35,
                         'handlelength': 1.5,
                         'handletextpad': 0.5,
                         'columnspacing': 1,
                       #'loc': 'center right',
                        }
legend_kwargs = {'kSZ': ksz_legend_kwargs, 'kappa': kappa_legend_kwargs, 'CMB_T': kappa_legend_kwargs}


fig = plt.figure(figsize=figsize, dpi=500)#, layout='tight')#'constrained')
subfigs = fig.subfigures(nrow, ncol, 
                         wspace=0.025,
                         hspace=-0.075, 
                         #squeeze=False,
                        )

for i, component in enumerate(plt_components):
    spec_type = plt_spec_types[component]
    
    i1 = i
    i2 = 0
    if i1 >= nrow:
        i1 -= nrow
        i2 += 1
    cfig = subfigs[i1, i2]#[i]
    
    ax1, ax2 = cfig.subplots(2, 1, sharex=True, height_ratios=[1-fdiff_plt_frac, fdiff_plt_frac])
    
    
    
    
    #"""
    if component == 'kSZ':
        ax1.plot([], [], label=s10_label, color=s10_label_color, **{**s10_kwargs, 'lw': 2.5})
        ax1.plot([], [], label=hd_label, color=hd_label_color, **hd_kwargs)
    elif component == 'kappa':
        ax1.plot([], [], label=s10_label, color=s10_label_color, **{**s10_kwargs, 'lw': 2.5})
        ax1.plot([], [], label=hd_label, color=hd_label_color, **hd_kwargs)
    elif component == 'CMB_T':
        ax1.plot([], [], label='Theory', color=s10_label_color, **{**s10_kwargs, 'lw': 2.5})
        ax1.plot([], [], label=hd_label, color=hd_label_color, **hd_kwargs)
        
    #"""
    for j, freq in enumerate(freqs):
        
        
        hd_color = '#bc8ae6' if (component in ['CMB_T', 'kappa']) else hd_colors[j]
        hd_lw = 1.25 if (component == 'CMB_T') else hd_kwargs['lw']
        hd_alpha = 0.35 if (component == 'kSZ') else hd_kwargs['alpha']
        if freq not in hd_spectra[component]:
            continue  # skip unavailable frequency (useful mostly for kappa)
        hd_loc = np.where((hd_spectra[component][freq]['l'] >= plt_lmin) & (hd_spectra[component][freq]['l'] <= plt_lmax))
        if component == 'CMB_T': 
            ax1.plot(hd_spectra[component][freq]['l'][hd_loc], 
                     hd_spectra[component][freq]['l'][hd_loc]**3 * hd_spectra[component][freq][spec_type][hd_loc] * (2*np.pi) / (hd_spectra[component][freq]['l'][hd_loc] + 1), 
                     color=hd_color, 
                 **{**hd_kwargs, 'lw': hd_lw, 'alpha': hd_alpha})
        else:
            ax1.plot(hd_spectra[component][freq]['l'][hd_loc], hd_spectra[component][freq][spec_type][hd_loc], color=hd_color, 
                 **{**hd_kwargs, 'lw': hd_lw, 'alpha': hd_alpha})
        
        if (component == 'kSZ'):
            ax1.plot([], [], color=hd_colors[j], label=f'{freq} GHz', **freq_label_kwargs)
        
        if (component in s10_spectra.keys()) and (freq in s10_spectra[component].keys()):
            s10_loc = np.where((s10_spectra[component][freq]['l'] >= plt_lmin) & (s10_spectra[component][freq]['l'] <= s10_lmaxs[component]))
            ax1.plot(s10_spectra[component][freq]['l'][s10_loc], s10_spectra[component][freq][spec_type][s10_loc], color=s10_colors[j], **s10_kwargs)
        elif component == 'CMB_T': 
            loc = np.where((theo[component]['l'] >= plt_lmin) & (theo[component]['l'] <= s10_lmaxs[component]))
            ax1.plot(theo[component]['l'][loc], theo[component]['l'][loc]**3 * theo[component][spec_type][loc] * (2*np.pi) / (theo[component]['l'][loc] + 1), color=s10_colors[0], **s10_kwargs)

        
        if (component in s10_spectra.keys()) and (freq in s10_spectra[component].keys()):
            loc = np.where((hd_spectra[component][freq]['l'] >= plt_lmin) & (hd_spectra[component][freq]['l'] <= s10_lmaxs[component]))
            fdiff = 100 * (hd_fdiff_spectra[component][freq][spec_type][loc] - s10_fdiff_spectra[component][freq][spec_type][loc]) / s10_fdiff_spectra[component][freq][spec_type][loc]
            if component in ['tSZ', 'kSZ']:
                ax2.plot(s10_spectra[component][freq]['l'][loc] + j*25, fdiff, color=hd_color,#fdiff_colors[j], 
                         **fdiff_kwargs)
            elif 'kappa' in component:
                
                ax2.plot(s10_spectra[component][freq]['l'][loc][:-1], fdiff[:-1], color=hd_color,#fdiff_colors[j], 
                         **fdiff_kwargs)
            else:
                ax2.plot(s10_spectra[component][freq]['l'][loc], fdiff, color=hd_color,#fdiff_colors[j], 
                         **fdiff_kwargs)
        #"""
        elif component == 'CMB_T': 
            loc = np.where((hd_spectra[component][freq]['l'] >= plt_lmin) & (hd_spectra[component][freq]['l'] <= s10_lmaxs[component]))
            fdiff = 100 * (hd_spectra[component][freq]['dl'][loc] - theo[component]['dl'][loc]) / theo[component]['dl'][loc]
            ax2.plot(theo[component]['l'][loc], fdiff, color=hd_color,#fdiff_colors[j], 
                     **fdiff_kwargs)
        #"""
        
    #ax1.set_title(component)
    if ylims[component] is not None:
        ax1.set_ylim(ylims[component])
    if logy[component]:
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', which='minor', left=False)
    if logx[component]:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    if yticks[component] is not None:
        ax1.set_yticks(yticks[component])
        
    ax2.axhline(color='k', lw=0.5)
    if fdiff_ylims[component] is not None:
        ax2.set_ylim(fdiff_ylims[component])
        
    
    ax1.set_ylabel(ylabels[component], fontsize=axis_labelsize)
    ax2.set_ylabel(fdiff_ylabel, fontsize=axis_labelsize)
    ax2.set_xlabel(xlabels[component], fontsize=axis_labelsize)
    
    
    if (component in ['CIB', 'radio']):
        
        ax1.text(0.04, 0.96, labels[component],
                 transform=ax1.transAxes, verticalalignment='top', horizontalalignment='left',
                 fontsize=component_labelsize,
                 #bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle=None),
                 #bbox={'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': 'round', 'pad': 0.2, 'alpha': 0.85},
                )
        
    else:
        ax1.text(0.96, 0.96, labels[component],
                 transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=component_labelsize,
                 #bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle=None),
                 #bbox={'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': 'round', 'pad': 0.2, 'alpha': 0.85},
                )
    
    
    if component in ['kSZ', 'CMB_T']:#, 'kappa', 'cmb']:
        ax1.legend(**legend_kwargs[component])
        #ax2.plot([], [], label=fdiff_labels[component], color='k', lw=1.25)
        #ax2.legend(**fdiff_legend_kwargs)
        
    plt.subplots_adjust(hspace=0)

plt.show()

plt.savefig("Fig3_12x12.pdf", bbox_inches='tight')

################################################################### Fig. 4

HD_kappa_patch = np.load(f"{overall_path_example}kappa_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
HD_kSZ_patch = np.load(f"{overall_path_example}kSZ_90GHz_10x10deg_ra=6_dec=6_spectra.npy", allow_pickle=True)[()]
S10_kappa_patch = np.load(f"{overall_path_example}S10_patches/kappa_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
S10_kSZ_patch = np.load(f"{overall_path_example}S10_patches/kSZ_90GHz_10x10deg_ra=6_dec=6_S10spectra.npy", allow_pickle=True)[()]
smallScale_kappa = np.load(f"{overall_path_example}kappa_smallscale_theory_spectra.npy", allow_pickle=True)[()]
smallScale_kSZ = np.load(f"{overall_path_example}kSZ_90GHz_smallscale_theory_spectra.npy", allow_pickle=True)[()]

spectra_HD = Spectra(
    ra = 6,
    dec = 6,
    final_width = 10.0,
    res = 0.04,
    apod_width = 1.0,
    l_max = 24000)

# Let's also take the small-scale extension theories we made before:
mbb_inv, binning_file, Bbl = spectra_HD.get_binning_files(path = f"{overall_path_example}binning_files/", type_Cl = True)
binned_theory_kappa_ells, binned_theory_kappa_cls = so_spectra.bin_spectra(
    np.array(smallScale_kappa['l']), np.array(smallScale_kappa['cl']), binning_file, lmax=24000, type="Cl", mbb_inv=None
)
binned_theory_kappa_dls = binned_theory_kappa_cls * binned_theory_kappa_ells * (binned_theory_kappa_ells+1) / (2*np.pi)
HD_kappa_patch['dl'] = HD_kappa_patch['cl'] * (HD_kappa_patch['l'] * (HD_kappa_patch['l']+1)) / (2*np.pi)
S10_kappa_patch['dl'] = S10_kappa_patch['cl'] * (S10_kappa_patch['l'] * (S10_kappa_patch['l']+1)) / (2*np.pi)

binned_theory_kSZ_ells, binned_theory_kSZ_cls = so_spectra.bin_spectra(
    np.array(smallScale_kSZ['l']), np.array(smallScale_kSZ['cl']), binning_file, lmax=24000, type="Cl", mbb_inv=None
)
binned_theory_kSZ_dls = binned_theory_kSZ_cls * binned_theory_kSZ_ells * (binned_theory_kSZ_ells+1) / (2*np.pi)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(10, 5),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex='col'
)
deltaEll = 200

# kappa
l_cutoff = 4000
ax1.plot(binned_theory_kappa_ells, binned_theory_kappa_dls, color="gray", label="Theory", linewidth=3, alpha=0.75)
ax1.plot(S10_kappa_patch['l'][:int((l_cutoff/deltaEll))], S10_kappa_patch['dl'][:int((l_cutoff/deltaEll))], color="black", label="S10 (0.86')", linewidth=2)
ax1.plot(HD_kappa_patch['l'], HD_kappa_patch['dl'], color="red", linestyle='--', label="This Work (0.04')", linewidth=2, alpha=0.75)
ax1.axvspan(0, l_cutoff, color='lightgray', alpha=0.5, hatch='/')

ax1.text(
    x=0.02 + 0.05 - 0.65 - 0.05,
    y=3.85 - 0.15,
    s="Lensing Convergence",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle=None)
)

ax1.set_ylabel(r"$D_\ell$ [$\mu K^2$]")
ax1.set_xlim(500, 20000)
ax1.set_ylim(0, 0.005)
ax1.legend(loc='lower right')

theory_interp_S10 = np.interp(S10_kappa_patch['l'], binned_theory_kappa_ells, binned_theory_kappa_dls)
theory_interp_HD = np.interp(HD_kappa_patch['l'], binned_theory_kappa_ells, binned_theory_kappa_dls)
S10_percent_diff = 100 * (S10_kappa_patch['dl'] - theory_interp_S10) / theory_interp_S10
HD_percent_diff = 100 * (HD_kappa_patch['dl'] - theory_interp_HD) / theory_interp_HD

ax3.plot(S10_kappa_patch['l'][:int((l_cutoff/deltaEll))], S10_percent_diff[:int((l_cutoff/deltaEll))], color='black', linewidth=2)
ax3.plot(HD_kappa_patch['l'], HD_percent_diff, color='red', linestyle='--', linewidth=2)
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_xlabel(r"$\ell$")
ax3.set_ylim(-10, 10)

# ksz
l_cutoff = 8000
ax2.plot(binned_theory_kSZ_ells, binned_theory_kSZ_dls, color="gray", label="Theory", linewidth=3, alpha=0.75)
ax2.plot(S10_kSZ_patch['l'][:int((l_cutoff/deltaEll))], S10_kSZ_patch['dl'][:int((l_cutoff/deltaEll))], color="black", label="S10 (0.43')", linewidth=2)
ax2.plot(HD_kSZ_patch['l'], HD_kSZ_patch['dl'], color="red", linestyle='--', label="This Work (0.04')", linewidth=2, alpha=0.75)
ax2.axvspan(0, l_cutoff, color='lightgray', alpha=0.5, hatch='/')

ax2.text(
    x=0.02 + 0.05 + 0.75 - 0.05,
    y=3.85 - 0.15,
    s="kSZ",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle=None)
)

ax2.set_xlim(500, 20000)
ax2.set_ylim(1, 2.5)
ax2.legend(loc='lower right')

theory_interp_S10 = np.interp(S10_kSZ_patch['l'], binned_theory_kSZ_ells, binned_theory_kSZ_dls)
theory_interp_HD = np.interp(HD_kSZ_patch['l'], binned_theory_kSZ_ells, binned_theory_kSZ_dls)
S10_percent_diff = 100 * (S10_kSZ_patch['dl'] - theory_interp_S10) / theory_interp_S10
HD_percent_diff = 100 * (HD_kSZ_patch['dl'] - theory_interp_HD) / theory_interp_HD

ax4.plot(S10_kSZ_patch['l'][:int((l_cutoff/deltaEll))], S10_percent_diff[:int((l_cutoff/deltaEll))], color='black', linewidth=2)
ax4.plot(HD_kSZ_patch['l'], HD_percent_diff, color='red', linestyle='--', linewidth=2)
ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
ax4.set_xlabel(r"$\ell$")
ax4.set_ylim(-50, 50)

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, linestyle='-', alpha=0.5)

plt.tight_layout()
plt.suptitle("Extending to Small Scales", fontsize=14)
plt.subplots_adjust(top=0.92)
plt.gcf().set_dpi(300)
plt.show()

plt.savefig("Fig4_12x12.pdf")