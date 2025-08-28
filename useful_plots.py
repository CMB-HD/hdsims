import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os
from matplotlib.colors import LinearSegmentedColormap
from pixell import enmap, enplot, colorize, utils, wcsutils, curvedsky as cs

def plot_map(imap, fig=None, ax=None, show=True, clim=None, vmin=None, cmap=None, 
             grid=True, grid_color=None, arcmin=False, title=None, figsize=(5,5), dpi=300,
             colorbar=True, cbar_label=r'$\mu$K', rotate_cbar_label=True, cbar_labelpad=0,
             cbar_shrink=0.85, cbar_pad=0.02, return_cbar=False,
             plt_ra_max=None, plt_ra_min=None,  plt_dec_min=None, plt_dec_max=None):
    
    # units for the axes: degrees or arcmin
    unit = 'arcmin' if arcmin else 'degrees'
    unit_conversion = 60 if arcmin else 1
    # axis limits: get the coordinates for edges of map to plot
    corners = np.rad2deg(enmap.corners(imap.shape, imap.wcs))
    plt_dec_min = corners[0][0] if (plt_dec_min is None) else plt_dec_min
    plt_ra_max = corners[0][1] if (plt_ra_max is None) else plt_ra_max
    plt_dec_max = corners[1][0] if (plt_dec_max is None) else plt_dec_max
    plt_ra_min = corners[1][1] if (plt_ra_min is None) else plt_ra_min
    sim_box = np.array([[plt_dec_min, plt_ra_max], [plt_dec_max, plt_ra_min]]) * utils.degree
    plt_extent = (plt_ra_max * unit_conversion, plt_ra_min * unit_conversion,  
                  plt_dec_min * unit_conversion, plt_dec_max * unit_conversion)
    
    # set up the color bar range
    vmax = clim
    if 'sym' in str(clim).lower():
        vmax = np.max(np.abs(imap.submap(sim_box)))
        vmin = -vmax
    elif (vmin is None) and (clim is not None):
        vmin = -clim
    cmap = 'planck' if (cmap is None) else cmap
    grid_color = 'tab:gray' if (grid_color is None) else grid_color
    
    # create the figure and axis if necessary and make the plot:    
    if (fig is None) or (ax is None):
        fig = plt.figure(dpi=dpi, figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    if title is not None:
        ax.set_title(title)
    im = ax.imshow(imap.submap(sim_box), origin='lower', aspect='equal',
                   extent=plt_extent,
                   vmax=vmax, vmin=vmin, cmap=cmap)
    ax.set_xlabel(f'RA [{unit}]')
    ax.set_ylabel(f'dec [{unit}]')
    #ax.set_ylim([plt_dec_min * unit_conversion, plt_dec_max * unit_conversion])
    #ax.set_xlim([plt_ra_max * unit_conversion, plt_ra_min * unit_conversion])
    if grid:
        ax.grid(alpha=0.1, color=grid_color)
    if colorbar:
        cbar = fig.colorbar(im, #ax=ax, #label=cbar_label, 
                            pad=cbar_pad, shrink=cbar_shrink)
        cbar_label_rot = 0 if rotate_cbar_label else 90#None
        cbar.set_label(cbar_label, rotation=cbar_label_rot, labelpad=cbar_labelpad)
    # by default, just display the plot:
    if show:
        plt.show()
    # otherwise, return the figure and axes to modify the plot:
    else:
        if return_cbar:
            return fig, ax, cbar
        else:
            return fig, ax

#https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def unify_ells(hd_spectra, s10_spectra):
    hd_fdiff_spectra = {}
    s10_fdiff_spectra = {}

    for comp in hd_spectra.keys():
        if comp == 'CMB_T':
            continue
        hd_fdiff_spectra[comp] = {}
        s10_fdiff_spectra[comp] = {}
        
        freqs = hd_spectra[comp].keys()
        for freq in freqs:
            hd_ells = hd_spectra[comp][freq]['l']
            s10_ells = s10_spectra[comp][freq]['l']
            
            dl_key = 'dl' if 'dl' in hd_spectra[comp][freq] else 'cl'
            
            hd_vals = hd_spectra[comp][freq][dl_key]
            s10_vals = s10_spectra[comp][freq][dl_key]
            
            shared_ells = np.unique(np.sort(np.concatenate([hd_ells, s10_ells])))
            
            hd_interp = np.interp(shared_ells, hd_ells, hd_vals)
            s10_interp = np.interp(shared_ells, s10_ells, s10_vals)
            
            hd_fdiff_spectra[comp][freq] = {'l': shared_ells, dl_key: hd_interp}
            s10_fdiff_spectra[comp][freq] = {'l': shared_ells, dl_key: s10_interp}

    return hd_fdiff_spectra, s10_fdiff_spectra