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

def plot_power(s10_spectra, hd_spectra, components, freqs):
    red = '#e81015'
    orange = '#ff6908'
    yellow = '#e0c004'
    green = '#02ad02'
    blue = '#005fc9'
    purple = '#8100f7'

    dim_red = '#de9294'
    dim_orange = '#e3a47b'
    dim_yellow = '#e8dea5'
    dim_green = '#a8e3a8'
    dim_blue = '#95b6db'
    dim_purple = '#d3b7eb'

    base_colors = [red, orange, yellow, green, blue, purple]
    dim_colors = [dim_red, dim_orange, dim_yellow, dim_green, dim_blue, dim_purple]

    # Assign colors based on input freq length
    hd_colors = base_colors[:len(freqs)]
    fdiff_colors = hd_colors
    s10_colors = dim_colors[:len(freqs)]

    s10_label_color = 'tab:gray'
    hd_label_color = 'k'

    # Plot configs
    s10_kwargs = {'ls': '-', 'lw': 5, 'alpha': 0.5}
    hd_kwargs = {'ls': '-', 'lw': 1, 'alpha': 0.9}
    hd2_kwargs = {'ls': ':', 'lw': 1, 'alpha': 0.9}
    fdiff_kwargs = {'ls': '-', 'lw': 1, 'alpha': 0.7}
    fdiff2_kwargs = {'ls': ':', 'lw': 1, 'alpha': 0.7}
    freq_label_kwargs = {'ls': '-', 'lw': 1.5}

    s10_label = "Original S10 (0.43'/0.86')"
    hd_label = "Upgraded (0.04')"

    plt_spec_types = {c: ('cl' if c == 'kappa' else 'dl') for c in components}

    plt_lmin = 30
    plt_lmax = 20000
    s10_lmaxs = {'tSZ': 8000, 'kSZ': 8000, 'radio': 8000, 'CIB': 8000, 'kappa': 4000}

    logy = {'tSZ': False, 'kSZ': False, 'radio': True, 'CIB': True, 'kappa': True}
    logx = {c: False for c in components}

    ylims = {
        'tSZ': [-0.5, 22], 'kSZ': [0, 3], 'radio': [5e-1, 1.5e8],
        'CIB': [5e-3, 1.5e5], 'kappa': [10**-10, 10**-7]
    }
    yticks = {
        'CIB': [0.1, 10, 1e3, 1e5],
        'radio': [10, 1e3, 1e5, 1e7]
    }
    fdiff_ylims = {
        'tSZ': [-0.1, 0.05], 'kSZ': [-0.05, 0.09], 'radio': [-2, 2],
        'CIB': [-0.5, 2.5], 'kappa': [-0.1, 0.05]
    }

    labels = {
        'tSZ': 'tSZ', 'kSZ': 'kSZ', 'CIB': 'CIB', 'radio': 'Radio', 'kappa': 'Lensing Convergence'
    }
    xlabels = {c: r'Multipole, $\ell$' for c in components}
    if 'kappa' in components:
        xlabels['kappa'] = r'Multipole, $L$'

    ylabels = {c: r'$\frac{\ell(\ell+1)}{2\pi} C_\ell~$ [$\mu$K$^2$]' for c in components}
    if 'kappa' in components:
        ylabels['kappa'] = r'$\frac{L^2 (L+1)^2}{4} C_L^{\phi\phi}$'

    fdiff_ylabel = '% Difference'

    # Plotting setup
    fig, axes = plt.subplots(2, len(components), figsize=(4 * len(components), 5), sharex='col',
                             gridspec_kw={'height_ratios': [3, 1]}, dpi=500)

    if len(components) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, component in enumerate(components):
        ax1, ax2 = axes[0, i], axes[1, i]
        spec_type = plt_spec_types[component]

        if component == 'kappa':
            freq = 'none'
            if freq in s10_spectra[component] and freq in hd_spectra[component]:
                s10_data = s10_spectra[component][freq]
                hd_data = hd_spectra[component][freq]

                loc_s10 = (s10_data['ells'] >= plt_lmin) & (s10_data['ells'] <= s10_lmaxs[component])
                loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= plt_lmax)

                ax1.plot(s10_data['ells'][loc_s10], s10_data['cl'][loc_s10], color=s10_label_color, **s10_kwargs)
                ax1.plot(hd_data['ells'][loc_hd], hd_data['cl'][loc_hd], color=hd_label_color, **hd_kwargs)

                common_ells = np.intersect1d(hd_data['ells'][loc_hd], s10_data['ells'][loc_s10])
                if len(common_ells) > 0:
                    s10_interp = np.interp(common_ells, s10_data['ells'], s10_data['cl'])
                    hd_interp = np.interp(common_ells, hd_data['ells'], hd_data['cl'])
                    fdiff = 100 * (hd_interp - s10_interp) / s10_interp
                    ax2.plot(common_ells, fdiff, color=hd_label_color, **fdiff_kwargs)
        elif component == "CIB":
            for j, freq in enumerate(freqs):
                if freq in s10_spectra[component]:
                    s10_data = s10_spectra[component][freq]
                    loc_s10 = (s10_data['ells'] >= plt_lmin) & (s10_data['ells'] <= s10_lmaxs[component])
                    ax1.plot(s10_data['ells'][loc_s10], s10_data[spec_type][loc_s10], color=s10_colors[j], **s10_kwargs)

                if freq in hd_spectra["CIB1"]:
                    hd_data = hd_spectra["CIB1"][freq]
                    loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= plt_lmax)
                    ax1.plot(hd_data['ells'][loc_hd], hd_data[spec_type][loc_hd], color=hd_colors[j], **hd_kwargs)
                if freq in hd_spectra["CIB2"]:
                    hd_data = hd_spectra["CIB2"][freq]
                    loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= plt_lmax)
                    ax1.plot(hd_data['ells'][loc_hd], hd_data[spec_type][loc_hd], color=hd_colors[j], **hd2_kwargs)

                if freq in s10_spectra[component] and freq in hd_spectra["CIB1"]:
                    s10_data = s10_spectra[component][freq]
                    hd_data = hd_spectra["CIB1"][freq]

                    loc_s10 = (s10_data['ells'] >= plt_lmin) & (s10_data['ells'] <= s10_lmaxs[component])
                    loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= s10_lmaxs[component])

                    common_ells = np.intersect1d(hd_data['ells'][loc_hd], s10_data['ells'][loc_s10])
                    if len(common_ells) > 0:
                        s10_interp = np.interp(common_ells, s10_data['ells'], s10_data['dl'])
                        hd_interp = np.interp(common_ells, hd_data['ells'], hd_data['dl'])
                        fdiff = 100 * (hd_interp - s10_interp) / s10_interp
                        ax2.plot(common_ells, fdiff, color=fdiff_colors[j], **fdiff_kwargs)
                if freq in s10_spectra[component] and freq in hd_spectra["CIB2"]:
                    s10_data = s10_spectra[component][freq]
                    hd_data = hd_spectra["CIB2"][freq]

                    loc_s10 = (s10_data['ells'] >= plt_lmin) & (s10_data['ells'] <= s10_lmaxs[component])
                    loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= s10_lmaxs[component])

                    common_ells = np.intersect1d(hd_data['ells'][loc_hd], s10_data['ells'][loc_s10])
                    if len(common_ells) > 0:
                        s10_interp = np.interp(common_ells, s10_data['ells'], s10_data['dl'])
                        hd_interp = np.interp(common_ells, hd_data['ells'], hd_data['dl'])
                        fdiff = 100 * (hd_interp - s10_interp) / s10_interp
                        ax2.plot(common_ells, fdiff, color=fdiff_colors[j], **fdiff2_kwargs)

            cib_handles = [
                plt.Line2D([0], [0], color=hd_label_color, **hd_kwargs, label='CIB Model 1'),
                plt.Line2D([0], [0], color=hd_label_color, **hd2_kwargs, label='CIB Model 2'),
            ]
            ax1.legend(handles=cib_handles, loc='upper right', fontsize=10, frameon=True, ncol=1)
        else:
            for j, freq in enumerate(freqs):
                if freq in s10_spectra[component]:
                    s10_data = s10_spectra[component][freq]
                    loc_s10 = (s10_data['ells'] >= plt_lmin) & (s10_data['ells'] <= s10_lmaxs[component])
                    ax1.plot(s10_data['ells'][loc_s10], s10_data[spec_type][loc_s10], color=s10_colors[j], **s10_kwargs)

                if freq in hd_spectra[component]:
                    hd_data = hd_spectra[component][freq]
                    loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= plt_lmax)
                    ax1.plot(hd_data['ells'][loc_hd], hd_data[spec_type][loc_hd], color=hd_colors[j], **hd_kwargs)

                if freq in s10_spectra[component] and freq in hd_spectra[component]:
                    s10_data = s10_spectra[component][freq]
                    hd_data = hd_spectra[component][freq]

                    loc_s10 = (s10_data['ells'] >= plt_lmin) & (s10_data['ells'] <= s10_lmaxs[component])
                    loc_hd = (hd_data['ells'] >= plt_lmin) & (hd_data['ells'] <= s10_lmaxs[component])

                    common_ells = np.intersect1d(hd_data['ells'][loc_hd], s10_data['ells'][loc_s10])
                    if len(common_ells) > 0:
                        s10_interp = np.interp(common_ells, s10_data['ells'], s10_data['dl'])
                        hd_interp = np.interp(common_ells, hd_data['ells'], hd_data['dl'])
                        fdiff = 100 * (hd_interp - s10_interp) / s10_interp
                        ax2.plot(common_ells, fdiff, color=fdiff_colors[j], **fdiff_kwargs)

        if ylims[component] is not None:
            ax1.set_ylim(ylims[component])
        if fdiff_ylims[component] is not None:
            ax2.set_ylim(fdiff_ylims[component])
        if component in yticks:
            ax1.set_yticks(yticks[component])
        if logy[component]:
            ax1.set_yscale('log')
        if logx[component]:
            ax1.set_xscale('log')
            ax2.set_xscale('log')

        ax1.set_title(labels[component], fontsize=14)
        ax1.set_ylabel(ylabels[component], fontsize=12)
        ax2.set_ylabel(fdiff_ylabel, fontsize=12)
        ax2.set_xlabel(xlabels[component], fontsize=12)
        ax2.axhline(0, color='k', lw=0.75)

    handles = [
        plt.Line2D([0], [0], color=s10_label_color, **s10_kwargs, label=s10_label),
        plt.Line2D([0], [0], color=hd_label_color, **hd_kwargs, label=hd_label),
    ]
    for j, freq in enumerate(freqs):
        handles.append(plt.Line2D([0], [0], color=hd_colors[j], **freq_label_kwargs, label=f'{int(freq)} GHz'))

    axes[0, -1].legend(handles=handles, loc='upper right', fontsize=10, frameon=True, ncol=1)

    fig.tight_layout()
    plt.show()

def cmb_plots(cmb_spectra, components=['TT', 'EE', 'BB', 'TE'], plt_lmin=30, plt_lmax=20000, fdiff_ylim = [-50, 50], fdiff_ylim_TE = [-100, 100], 
              patch_kwargs = {'ls': '-', 'lw': 1, 'alpha': 0.7}, theory_kwargs = {'ls': '--', 'lw': 1, 'alpha': 1}, patch_color = '#005fc9', theory_color = '#02ad02', 
              labels = {'lensed_patch':'Lensed Patch', 'lensed_theory':'Lensed Theory'}):
    fdiff_color = patch_color
    
    fdiff_kwargs = patch_kwargs

    ylabels = {
        'TT': r'$D_\ell^{TT}~[\mu K^2]$', 'EE': r'$D_\ell^{EE}~[\mu K^2]$',
        'BB': r'$D_\ell^{BB}~[\mu K^2]$', 'TE': r'$D_\ell^{TE}~[\mu K^2]$'
    }
    xlabels = {c: r'Multipole, $\ell$' for c in components}
    fdiff_ylabel = '% Difference'

    fig, axes = plt.subplots(2, len(components), figsize=(4 * len(components), 5),
                             sharex='col', gridspec_kw={'height_ratios': [3, 1]}, dpi=500)
    if len(components) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, comp in enumerate(components):
        ax1, ax2 = axes[0, i], axes[1, i]

        for key, color, kwargs in zip(['lensed_patch', 'lensed_theory'],
                                      [patch_color, theory_color],
                                      [patch_kwargs, theory_kwargs]):
            if key in cmb_spectra.get(comp, {}):
                data = cmb_spectra[comp][key]
                loc = (data['ells'] >= plt_lmin) & (data['ells'] <= plt_lmax)
                ax1.plot(data['ells'][loc], data['dl'][loc], color=color,
                         label=labels[key], **kwargs)

        patch_data = cmb_spectra[comp].get('lensed_patch')
        theory_data = cmb_spectra[comp].get('lensed_theory')
        if patch_data and theory_data:
            patch_ells = patch_data['ells']
            theory_ells = theory_data['ells']
            common_ells = np.intersect1d(
                patch_ells[(patch_ells >= plt_lmin) & (patch_ells <= plt_lmax)],
                theory_ells[(theory_ells >= plt_lmin) & (theory_ells <= plt_lmax)]
            )
            if len(common_ells) > 0:
                patch_interp = np.interp(common_ells, patch_ells, patch_data['dl'])
                theory_interp = np.interp(common_ells, theory_ells, theory_data['dl'])
                mask = theory_interp != 0
                fdiff = np.zeros_like(common_ells, dtype=float)
                fdiff[mask] = 100 * (patch_interp[mask] - theory_interp[mask]) / theory_interp[mask]
                ax2.plot(common_ells, fdiff, color=fdiff_color, **fdiff_kwargs)

        ax1.set_title(comp, fontsize=14)
        ax1.set_ylabel(ylabels[comp], fontsize=12)
        ax2.set_xlabel(xlabels[comp], fontsize=12)
        ax2.set_ylabel(fdiff_ylabel, fontsize=10)
        ax2.axhline(0, color='k', lw=0.75)
        if comp == 'TE':
            ax2.set_ylim(fdiff_ylim_TE)
        else:
            ax1.set_yscale('log')
            ax2.set_ylim(fdiff_ylim)

        ax1.set_xlim([plt_lmin, plt_lmax])
        ax2.set_xlim([plt_lmin, plt_lmax])

        if i == len(components) - 1:
            ax1.legend(fontsize=9, loc='upper right', frameon=True)

    fig.tight_layout()
    plt.show()