"""Contains functions for plotting."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pixell import enmap, colorize
from . import utils, siminfo as si, maps, simutils

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.dpi'] = 250
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.xmargin'] = 0.025
plt.rcParams['axes.ymargin'] = 0.025
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['figure.figsize'] = (5, 3)
if 'planck' not in mpl.colormaps:
    colorize.mpl_setdefault('planck')
plt.rcParams['image.cmap'] = 'planck'


def get_rcParams(dpi=250):
    """Return a dictionary of `matplotlib` configuration settings.

    Parameters
    ----------
    dpi : int, default=250
        The resolution of plots in dots-per-inch.

    Returns
    -------
    params : dict
        A dictionary of `matplotlib` configuration settings.
    """
    params = {'figure.dpi': dpi, 'axes.grid': True, 'grid.alpha': 0.2,
              'axes.xmargin': 0.025, 'axes.ymargin': 0.025, 'figure.figsize': (5, 3)}
    if 'planck' not in mpl.colormaps:
        colorize.mpl_setdefault('planck')
    params['image.cmap'] = 'planck'
    return params


def plot_map(imap, show=True, fname=None,
             ra_ctr=None, dec_ctr=None, plt_radius=None,
             ra_min=None, ra_max=None, dec_min=None, dec_max=None,
             title=None, colorbar=True, vmin=None, vmax=None, cmap='planck',
             symmetric_cbar=False, lognorm=False, linthresh=0.01,
             cbar_label=r'$\mu$K', rotate_cbar_label=True,
             grid=True, grid_color='tab:gray', arcmin=False, axis_labelpad=None,
             figsize=(5,5), dpi=300,
             fig=None, ax=None, cax=None,
             return_cbar=False, return_im=False, **kwargs):
    """Plot a map.

    The axes for right ascension (x-axis) and declination (y-axis) are
    labeled (by default, in units of degrees). The right ascension
    increases from right to left, and the declination increases from the
    bottom to the top.

    By default, `plot_map(imap)` will display a plot of the input map
    (e.g., as the output of a Jupyter notebook cell). The plot can be
    saved by passing a file name.

    You may also pass a figure (`matplotlib.figure.Figure`) and its axes
    (`matplotlib.axes.Axes`), and/or return these instances, to further
    modify the plot.

    There are also parameters to specify a region within the map to plot,
    and other parameters for the appearance of the plot.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        The input map to plot. It should have a shape `(Ny, Nx)` for the
        number of rows and columns, respectively.
    show : bool, default=True
        Whether to display the plot by calling `matplotlib.pyplot.show()`.
    fname : str or None, default=None
        A file name to use when saving the plot. By default, the plot will
        not be saved.
    ra_ctr, dec_ctr, plt_radius : float or None, default=None
        Parameters to specify a region within the map to plot. All must be
        in units of degrees. The width of the (square) region is given by
        `2 * plt_radius`, and the R.A. and dec. coordinates of its center
        are given by `ra_ctr` and `dec_ctr`, respectively.
        If `plt_radius=None`, `ra_ctr` and `dec_ctr` are ignored.
        If `plt_radius` is passed but `ra_ctr` or `dec_ctr` is `None`, the
        coordinates of the center of the map will be used.
        By default, the full map is plotted.
    ra_min, ra_max, dec_min, dec_max : float or None, default=None
        The minimum and maximum R.A. and dec. coordinates that define a
        region within the map to plot. All must be in units of degrees.
        By default, the minimum and maximum R.A. and dec. coordinates in
        the map are used. Ignored if a `plt_radius` is passed.
    title : str or None, default=None
        A title for the plot.
    vmin, vmax : float or None, default=None
        The minimum and maximum range of the color map, respectively.
        By default, the minimum and maximum values in the map are used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The `matplotlib` figure used to make the plot. Only returned if
        `show=False`.
    ax, cax : matplotlib.axes.Axes
        The `matplotlib` axes used for the map and its color bar,
        respectively. Only returned if `show=False`.
    cbar : matplotlib.colorbar.Colorbar
        The `matplotlib` colorbar instance. Only returned if `show=False`
        and `return_cbar=True`.
    im : matplotlib.image.AxesImage
        The `matplotlib` image instance. Only returned if `show=False`
        and `return_im=True`.

    Other Parameters
    ----------------
    colorbar : bool, default=True
        Whether to draw the color bar.
    cmap : str or matplotlib.colors.Colormap, optional
        A color map.
    symmetric_cbar : bool, default=False
        If `symmetric_cbar=True`, the range of the color map will be
        symmetric about zero.
    lognorm : bool, default=False
        Whether to use a logarithmic scale for the color map.
    linthresh : float, default=0.01
        The value below which the to use a linear, instead of logarithmic,
        color scale. Ignored if `lognorm=False`.
    cbar_label : str, default=r'$\mu$K'
        A label for the color bar.
    rotate_cbar_label : bool, default=True
        If `rotate_cbar_label=True`, the color bar label is rotated so it
        is the same orientation as the x-axis label.
    grid : bool, default=True
        Whether to draw a grid on the plot.
    grid_color : str, default='tab:gray'
        The color to use for the grid lines if `grid=True`.
    arcmin : bool, default=False
        Whether to label the x- and y-axis ticks in units of arcminutes
        instead of degrees.
    axis_labelpad : float or None, default=None
        If passed, the amount of padding to add between the axes and their
        labels.
    figsize : tuple of int, default=(5, 5)
        The width and height of the figure, in inches.
    dpi : int, default=300
        The resolution of the figure, in dots-per-inch.
    fig : matplotlib.figure.Figure or None, default=None
        A `matplotlib` figure to use when making the plot, instead of
        initializing a new figure.
    ax : matplotlib.axes.Axes or None, default=None
        The `matplotlib` axes to use for the plot of the map, instead of
        initializing a new set of axes.
    cax : matplotlib.axes.Axes or None, default=None
        The `matplotlib` axes to use for the color bar, instead of
        initializing a new set of axes. If `cax=None` but `ax` was passed,
        `ax` will be used.
    return_cbar : bool, default=False
        Whether to return the `matplotlib.colorbar.Colorbar` instance.
    return_im : bool, default=False
        Whether to return the `matplotlib.image.AxesImage` instance.
    **kwargs : dict
        Other keyword arguments are ignored.
    """
    # cut out a smaller patch from the map, if necessary:
    if plt_radius is not None:
        if (ra_ctr is None) or (dec_ctr is None):
            imap_ra_ctr, imap_dec_ctr, _, _ = maps.get_map_ctr_extent(imap.shape, imap.wcs)
            ra_ctr = imap_ra_ctr if (ra_ctr is None) else ra_ctr
            dec_ctr = imap_dec_ctr if (dec_ctr is None) else dec_ctr
        res = maps.get_map_resolution(imap.shape, imap.wcs)
        plt_shape, plt_wcs = maps.get_shape_wcs(res, ra_ctr, dec_ctr, 2 * plt_radius)
        plt_map = enmap.project(imap.copy(), plt_shape, plt_wcs)
    elif any([(coord is not None) for coord in [ra_min, ra_max, dec_min, dec_max]]):
        imap_ra_min, imap_ra_max, imap_dec_min, imap_dec_max = maps.get_map_corner_coords(imap.shape, imap.wcs)
        ra_min = imap_ra_min if (ra_min is None) else ra_min
        ra_max = imap_ra_max if (ra_max is None) else ra_max
        dec_min = imap_dec_min if (dec_min is None) else dec_min
        dec_max = imap_dec_max if (dec_max is None) else dec_max
        ra_ctr, dec_ctr, plt_width, plt_height = maps.get_patch_ctr_extent(ra_min, ra_max, dec_min, dec_max)
        res = maps.get_map_resolution(imap.shape, imap.wcs)
        plt_shape, plt_wcs = maps.get_shape_wcs(res, ra_ctr, dec_ctr, plt_width, height=plt_height)
        plt_map = enmap.project(imap.copy(), plt_shape, plt_wcs)
    else:
        plt_map = imap.copy()

    # get edges of map/patch to correctly label axes:
    ra_min, ra_max, dec_min, dec_max = maps.get_map_corner_coords(plt_map.shape, plt_map.wcs)
    plt_extent = (ra_max, ra_min, dec_min, dec_max)
    units = 'arcmin' if arcmin else 'degrees'

    # set up the color bar range
    if symmetric_cbar:
        if (vmin is None) and (vmax is None):
            clim = np.max(np.abs(plt_map))
        elif vmin is None:
            clim = vmax
        elif vmax is None:
            clim = abs(vmin)
        else:
            clim = max([abs(vmin), abs(vmax)])
        vmin = -clim
        vmax = clim
    else:
        vmin = np.min(plt_map) if (vmin is None) else vmin
        vmax = np.max(plt_map) if (vmax is None) else vmax

    if lognorm:
        if vmin > 0:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # create the figure and axis if necessary and make the plot:
    plt_height = figsize[1]
    cax_frac = 0.05
    plt_wspace = 0.025
    plt_width = plt_height * (1 + cax_frac) * (1 + plt_wspace)
    if ax is None:
        if fig is None:
            fig, (ax, cax) = plt.subplots(nrows=1, ncols=2, width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)], figsize=(plt_width, plt_height), dpi=dpi)
        else:
            ax, cax = fig.subplots(nrows=1, ncols=2, width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)])
    if cax is None:
        cax = ax
    
    im = ax.imshow(plt_map, origin='lower', aspect='equal', extent=plt_extent, norm=norm, cmap=cmap)
    if title is not None:
        ax.set_title(title)
    if arcmin:
        xticks = ax.get_xticks()
        ax.set_xticklabels([round(utils.deg2arcmin(tick), 2) for tick in xticks])
        yticks = ax.get_yticks()
        ax.set_yticklabels([round(utils.deg2arcmin(tick), 2) for tick in yticks])
    ax.set_xlabel(f'RA [{units}]', labelpad=axis_labelpad)
    ax.set_ylabel(f'dec [{units}]', labelpad=axis_labelpad)
    if grid:
        ax.grid(alpha=0.1, color=grid_color)
    if colorbar:
        cbar = fig.colorbar(im, cax=cax)#, pad=cbar_pad, shrink=cbar_shrink)
        cbar_label_rot = 0 if rotate_cbar_label else 90
        cbar.set_label(cbar_label, rotation=cbar_label_rot)
        cax.grid(visible=False)
    plt.subplots_adjust(wspace=plt_wspace)

    # by default, just display the plot:
    if show:
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', dpi=dpi)
        plt.show()
    # otherwise, return the figure and axes to modify the plot:
    else:
        if return_cbar and return_im:
            return fig, ax, cax, cbar, im
        if return_cbar:
            return fig, ax, cax, cbar
        elif return_im:
            return fig, ax, cax, im
        else:
            return fig, ax, cax



def plot_maps(imaps, labels=None, ncol=2, colorbar_ranges=None, dpi=300, show=True, fname=None, **kwargs):
    """Plot each map in a list.

    Parameters
    ----------
    imaps : list of pixell.enmap.ndmap
        The list of maps to plot.
    labels : list of str or None, default=None
        A list of labels for each map.
    ncol : int, default=2
        The number of columns in the plot.
    colorbar_ranges : list of list of float
        A list of ranges `[vmin, vmax]` to use for the colorbar for each
        map.
    dpi : int, default=300
        The resolution of the figure, in dots-per-inch.
    show : bool, default=True
        Whether to display the plot by calling `matplotlib.pyplot.show()`.
    fname : str or None, default=None
        A file name to use when saving the plot. By default, the plot will
        not be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The `matplotlib` figure used to make the plot. Only returned if
        `show=False`.
    subfigs_list : list of matplotlib.figure.SubFigure
        The `matplotlib` subfigure for each map in the list. Only returned
        if `show=False`.
    axs_list, caxs_list : list of matplotlib.axes.Axes
        The `matplotlib` axes used for each map and its color bar,
        respectively. Only returned if `show=False`.

    Other Parameters
    ----------------
    **kwargs : dict
        Additional keyword arguments passed to `plot_map`.

    See Also
    --------
    plot_map : Plot a single map.
    """
    nplt = len(imaps)
            
    if labels is None:
        labels = [None for i in range(nplt)]
    elif len(labels) != nplt: # need a different range for each map
        raise ValueError(f"You passed {nplt} `imaps` but {len(labels)} `labels`. "
                         "The list of `labels` must have the same length as the list of `imaps`.")
    
    if colorbar_ranges is None:
        colorbar_ranges = [[None, None] for i in range(nplt)]
    else:
        cbar_ranges = np.atleast_1d(colorbar_ranges.copy())
        if len(cbar_ranges.shape) == 1: # use same range for all maps
            colorbar_ranges = [[cbar_ranges[0], cbar_ranges[1]] for i in range(nplt)]
        elif len(colorbar_ranges) != nplt: # need a different range for each map
            raise ValueError(f"You passed {nplt} `imaps` but {len(colorbar_ranges)} `colorbar_ranges`. "
                             "The list of `colorbar_ranges` must have the same length as the list of `imaps`.")

    nrow = int(nplt / ncol)
    while nrow * ncol < nplt:
        nrow += 1
    cax_frac = 0.05
    plt_wspace = 0.03
    fig_wspace = 0.1
    fig_hspace = -0.1
    plt_height = 4
    plt_width = plt_height * (1 + cax_frac + plt_wspace)
    fig_width = plt_width + (ncol - 1) * plt_width * (1+fig_wspace)
    fig_height = plt_height + (nrow - 1) * plt_height * (1+fig_hspace)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    subfigs = fig.subfigures(nrows=nrow, ncols=ncol, wspace=fig_wspace, hspace=fig_hspace)
    
    # if `show=False`, return the fig/subfigs/axes/etc:
    subfigs_list = []
    axs_list = []
    caxs_list = []
    for i in range(nrow):
        for j in range(ncol):
            if (nrow > 1) and (ncol > 1):
                idx = i * ncol + j
                sfig = subfigs[i, j]
            else:
                idx = i + j
                sfig = subfigs[idx]
            if idx < nplt:
                vmin, vmax = colorbar_ranges[idx]
                plt_kwargs = {**kwargs, 'dpi': dpi, 'vmin': vmin, 'vmax': vmax, 'fname': None, 'axis_labelpad': 2}
                
                ax, cax = sfig.subplots(nrows=1, ncols=2, width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)])
                sfig, ax, cax = plot_map(imaps[idx], fig=sfig, ax=ax, cax=cax, show=False, **plt_kwargs)
                if labels[idx] is not None:
                    ax.text(0.05, 0.95, labels[idx], bbox=dict(facecolor='w', alpha=0.85), fontsize=12,
                            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
                cax.grid(visible=False)
                plt.subplots_adjust(wspace=plt_wspace)
                
                subfigs_list.append(sfig)
                axs_list.append(ax)
                caxs_list.append(cax)
                
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, subfigs_list, axs_list, caxs_list



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Return part of a color map."""
    # this was taken from:
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_colorbar_range(components, freq=None, pol=False, noise=False, symmetric=False):
    """Attempt to set a reasonable range for the color map used to plot a
    map.

    Parameters
    ----------
    components : list of str
        A list of individual components in the map.
    freq : int, default=None
        The map frequency (in GHz). Required if any of the components in
        the map (including white noise) are frequency-dependent.
    pol : bool, default=False
        Whether the map is a polarization (Q or U) map.
    noise : bool, default=False
        Whether white noise has been added to the map.
    symmetric : bool, default=False
        Whether the range should be symmetric.

    Returns
    -------
    list of float
        The range `[vmin, vmax]`.

    Notes
    -----
    The ranges used for the white noise maps are based on the CMB-HD
    instrumental noise levels.
    """
    if noise or simutils.has_freq_dependent_component(components):
        freq = simutils.validate_sim_freq(freq)
    # remove foregrounds from the list if the map is for polarization:
    plt_components = [c for c in components if ('cmb' in c)] if pol else components.copy()
    if noise:
        plt_components.append('noise')
    # define defaults for each frequency and component:
    default_clims = {'tsz': {30: [-120, 0], 90: [-100, 0], 148: [-60, 0], 219: [0, 5], 277: [0, 60], 350: [0, 140]},
                     'cib': {30: [0, 10], 90: [0, 25], 148: [0, 60], 219: [0, 150], 277: [0, 350], 350: [0, 1000]},
                     'radio': {30: [0, 200], 90: [0, 10], 148: [0, 5], 219: [0, 2.5], 277: [0, 2.5], 350: [0, 2.5]},
                     'ksz': {freq: [-15, 15]},
                     'kappa': {freq: [-0.5, 0.5]}}
    if pol:
        default_clims['cmb'] = {freq: [-50, 50]}
        default_clims['noise'] = {30: [-500, 500], 90: [-60, 60], 148: [-65, 65],
                                  219: [-150, 150], 277: [-225, 225], 350: [-2500, 2500]}
    else:
        default_clims['cmb'] = {freq: [-300, 300]}
        default_clims['noise'] = {30: [-375, 375], 90: [-40, 40], 148: [-45, 45],
                                  219: [-115, 115], 277: [-150, 150], 350: [-1000, 1000]}
    # choose the range based on which components are in the map:
    if simutils.has_cmb(components):
        # always use the same range for maps with the CMB:
        vmin, vmax = default_clims['cmb'][freq]
    else:
        vmin = -0.1
        vmax = 0.1
        for component in plt_components:
            vmin = min([vmin, default_clims[component][freq][0]])
            vmax = max([vmax, default_clims[component][freq][1]])
    if symmetric:
        clim = max([abs(vmin), abs(vmax)])
        vmin = -clim
        vmax = clim
    return [vmin, vmax]


def plot_sims(sims, freq, show=True, plot_fname=None,
              map_width=None, dpi=300, map_cbar_lims=None, ncol=2):
    """Plot a set of simulations at a single frequency.

    Parameters
    ----------
    sims : dict of pixell.enmap.ndmap
        A dictionary of the simulations. Each key must be one of `'tsz'`,
        `'ksz'`, `'cib'`, `'radio'`, `'kappa'`, or `'cmb'` for the tSZ,
        kSZ, CIB, radio, lensing convergence, or CMB maps, respectively.
    freq : int
        The frequency (GHz) of the simulations. Must be `30`, `90`, `148`,
        `219`, `277`, or `350`.
    show : bool, default=True
        Whether to display the plot by calling `matplotlib.pyplot.show()`.
    plot_fname : str or None, default=None
        The file name used to save the plot. By default, the plot will not
        be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The `matplotlib` figure used to make the plot. Only returned if
        `show=False`.
    subfigs_dict : dict of matplotlib.figure.SubFigure
        A dictionary with a key (`str`) for each map component in `sims`
        and the corresponding `matplotlib` subfigure for the plot of that
        component as the values. Only returned if `show=False`.
    axs, caxs : dict of matplotlib.axes.Axes
        Dictionaries with a key (`str`) for each map component in `sims`.
        The values in `axs` are the corresponding subplots (of the
        corresponding subfigure) used to plot each component, and the
        values in `caxs` are the subplots used for the color bar. Only
        returned if `show=False`.

    Other Parameters
    ----------------
    map_width : float or None, default=None
        The width of the region (centered on the map center) in the maps
        to plot. By default, the full maps are plotted.
    dpi : int, default=300
        The resolution of the plot, in dots-per-inch.
    map_cbar_lims : dict of list of float or None, default=None
        A dictionary of color map ranges to use for each simulation. The
        keys should be the same as the keys in `sims`, and the range
        should be given as a range `[vmin, vmax]`. Otherwise, a default
        range is used.
    ncol : int, default=2
        The number of columns in the plot.

    Notes
    -----
    This function is used to produce Figure 2 in arXiv:XXXX.XXXX (!! TODO !!).
    """
    components = list(sims.keys())
    all_map_names = ['tsz', 'ksz', 'cib', 'radio', 'kappa', 'cmb', 'cmbQ', 'cmbU'] # order to plot them in
    map_labels = {'tsz': 'tSZ', 'ksz': 'kSZ', 'cib': 'CIB', 'radio': 'Radio', 'kappa': 'Lensing Convergence',
                  'cmb': 'Lensed T CMB',  'cmbQ': 'Lensed Q CMB', 'cmbU': 'Lensed U CMB'}
    map_names = []
    pol = False
    for component in all_map_names[:-2]:
        if component in components:
            map_names.append(component)
            if component == 'cmb':
                pol = True if (len(sims['cmb'].shape) > 2) else False
                if pol:
                    map_names.append('cmbQ')
                    map_names.append('cmbU')

    # set the color map & color bar range for each component based on the frequency
    tsz_clims = {30: [-120, 0], 90: [-100, 0], 148: [-60, 0], 219: [0, 5], 277: [0, 60], 350: [0, 140]}
    cib_clims = {30: [0, 10], 90: [0, 25], 148: [0, 60], 219: [0, 150], 277: [0, 350], 350: [0, 1000]}
    radio_clims = {30: [0, 200], 90: [0, 10], 148: [0, 5], 219: [0, 2.5], 277: [0, 2.5], 350: [0, 2.5]}
    default_clims = {'tsz': tsz_clims[freq], 'ksz': [-15, 15], 'cib': cib_clims[freq],  'radio': radio_clims[freq],
                     'kappa': [-0.5, 0.5],  'cmb': [-350, 350], 'cmbQ': [-50, 50], 'cmbU': [-50, 50]}
    map_cbar_lims = {} if (map_cbar_lims is None) else map_cbar_lims
    for component in map_names:
        if component not in map_cbar_lims:
            map_cbar_lims[component] = default_clims[component]
    cmaps = {}
    full_cmap = plt.get_cmap('planck')
    pos_cmap = truncate_colormap(full_cmap, minval=0.5, maxval=1.0, n=100)
    neg_cmap = truncate_colormap(full_cmap, minval=0.0, maxval=0.5, n=100)
    for component in map_names:
        vmin = map_cbar_lims[component][0]
        vmax = map_cbar_lims[component][1]
        if (vmin < 0) and (vmax > 0):
            cmaps[component] = full_cmap
        elif vmin < 0:
            cmaps[component] = neg_cmap
        else:
            cmaps[component] = pos_cmap

    if map_width is None:
        plt_radius = None
    else:
        plt_radius = map_width / 2

    nplt = len(map_names)
    nrow = int(nplt / ncol)
    while nrow * ncol < nplt:
        nrow += 1
    cax_frac = 0.05
    plt_wspace = 0.03
    fig_wspace = 0.1
    fig_hspace = -0.1
    plt_height = 4
    plt_width = plt_height * (1 + cax_frac + plt_wspace)
    fig_width = plt_width + (ncol - 1) * plt_width * (1+fig_wspace)
    fig_height = plt_height + (nrow - 1) * plt_height * (1+fig_hspace)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    subfigs = fig.subfigures(nrows=nrow, ncols=ncol, wspace=fig_wspace, hspace=fig_hspace)

    # if `show=False`, return the fig/subfigs/axes/etc:
    subfigs_dict = {}
    axs = {}
    caxs = {}
    for i in range(nrow):
        for j in range(ncol):
            if (i * ncol + j) < len(map_names):
                if (nrow > 1) and (ncol > 1):
                    sfig = subfigs[i, j]
                else:
                    sfig = subfigs[i+j]
                component = map_names[i * ncol + j]
                vmin, vmax = map_cbar_lims[component]
                cmap = cmaps[component]
                cbar_label = '' if (component == 'kappa') else r'$\mu$K'
                if ('cmb' in component) and pol:
                    if 'Q' in component:
                        map_to_plot = sims['cmb'][1]
                    elif 'U' in component:
                        map_to_plot = sims['cmb'][2]
                    else:
                        map_to_plot = sims['cmb'][0]
                else:
                    map_to_plot = sims[component]

                ax, cax = sfig.subplots(nrows=1, ncols=2, width_ratios=[1/(1+cax_frac), cax_frac/(1+cax_frac)])
                sfig, ax, cax = plot_map(map_to_plot, fig=sfig, ax=ax, cax=cax, show=False, grid=False, vmin=vmin, vmax=vmax, cmap=cmap, cbar_label=cbar_label, plt_radius=plt_radius, axis_labelpad=2)
                ax.text(0.05, 0.95, map_labels[component], bbox=dict(facecolor='w', alpha=0.85), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=12)
                ax.grid(visible=False)
                cax.grid(visible=False)
                plt.subplots_adjust(wspace=plt_wspace)

                subfigs_dict[component] = sfig
                axs[component] = ax
                caxs[component] = cax

    if plot_fname is not None:
        plt.savefig(plot_fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        return fig, subfigs_dict, axs, caxs


def plot_sim_spectra_comparison(spectra1, spectra2, label1, label2, show=True, plot_fname=None,
                                lmin1=None, lmax1=None, lmin2=None, lmax2=None, cmb_label2=None, 
                                ncol=2, dpi=500, add_legends=True, ylims=None, yticks=None, 
                                plot_fdiff=False, spectra1_for_fdiff=None, spectra2_for_fdiff=None, fdiff_ylims=None):
    """
    Plot a comparison between two sets of power spectra.

    Compares the power spectra in `spectra1` to the corresponding power
    spectra in `spectra2`.

    Parameters
    ----------
    spectra1 : dict of dict of dict of array_like of float
        A (very) nested dictionary of the power spectra for different map
        components.
        - The first set of keys is the name of the map components. Each
          name must be one of `'tsz'`, `'ksz'`, `'cib'`, `'radio'`,
          `'kappa'`, or `'cmb'` for the tSZ, kSZ, CIB, radio, lensing
          convergence, or CMB maps, respectively.
        - The second set of keys is either given by the list of map
          frequencies (in GHz) for frequency-dependent components
          (`'tsz'`, `'cib'`, `'radio'`), or a single key of `None` for
          frequency-independent components (`'ksz'`, `'kappa'`, and
          `'cmb'` or `'unlensed_cmb'`).
        - The inner-most dictionary contains the multipoles (key `'ells'`)
          and power spectra for each component and frequency.
          - For `'tsz'`, `'ksz'`, `'cib'`, `'radio'`: The key `'tt'`
            holds the power spectrum of each map in units of uK^2,
            multiplied by `ell * (ell + 1) / (2 * pi)`.
          - For `'kappa'`: The key `'kk'` holds the dimensionless
            power spectrum of the lensing convergence map.
          - For `'cmb'` or `'unlensed_cmb'`: The key `'tt'` holds the
            power spectrum of the CMB temperature map in units of
            uK^2, multiplied by a factor of `ell^4`. If `pol=True`,
            there are also keys `'te'` (not plotted), `'ee'`, `'bb'`
            for the TE, EE, BB power spectra, respectively.
    spectra2 : dict of dict of dict of array_like of float
        The set of power spectra to compare against, with the same keys
        as `spectra1`.
    label1, label2 : str
        Labels for the set of power spectra in `spectra1` and `spectra2`,
        respectively.
    show : bool, default=True
        Whether to display the plot by calling `matplotlib.pyplot.show()`.
    plot_fname : str or None, default=None
        The file name used to save the plot. By default, the plot will not
        be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The `matplotlib` figure used to make the plot. Only returned if
        `show=False`.
    subfigs_dict : dict of matplotlib.figure.SubFigure
        A dictionary with a key (`str`) for each key in `spectra1` and the
        corresponding `matplotlib` subfigure for the plot of that
        component as the values. Only returned if `show=False`.
    axs : dict of matplotlib.axes.Axes
        A dictionary with a key (`str`) for each key in `spectra1`. The
        values are the corresponding subplots (of the corresponding
        subfigure) used to plot the power spectra of that component. Only
        returned if `show=False`.
    fdiff_axs : dict of matplotlib.axes.Axes
        A dictionary with a key (`str`) for for each key in `spectra1`.
        The values are the corresponding subplots (of the corresponding
        subfigure) used to plot the fractional difference of the power
        spectra of each component. Only returned if `show=False` and
        `plot_fdiff=True`.

    Other Parameters
    ----------------
    lmin1, lmax1 : int or None, default=None
        The multipole range to plot for the power spectra in `spectra1`.
    lmin2, lmax2 : int or None, default=None
        The multipole range to plot for the power spectra in `spectra2`.
    cmb_label2 : str or None, default=None
        The label to use for the CMB power spectra in `spectra2`.
    ncol : int, default=2
        The number of columns in the plot.
    dpi : int, default=500
        The resolution of the plot, in dots-per-inch.
    add_legends : bool, default=True
        Whether to add the legends to the plot.
    ylims : dict of array_like of float or None, default=None
        The y-axis limits for any of the power spectra plots. By default,
        no limits will be set. Each dictionary key should be one of the
        keys in `spectra1`, and the value should be a list `[ymin, ymax]`
        of the y-axis limits for that component.
    yticks : dict of array_like of float or None, default=None
        The y-axis ticks for any of the power spectra plots. By default,
        the ticks are not explicitly set. Each dictionary key should be
        one of the keys in `spectra1`, and the value should be the array
        of ticks to use for its y-axis.
    plot_fdiff : bool, default=False
        Whether to add a lower panel to the plot for each component showing
        the fractional difference between the power spectra in `spectra1`
        and `spectra2`.
    fdiff_ylims : dict of array_like of float or None, default=None
        The y-axis limits for any of the fractional difference plots. By
        default, no limits will be set. Each dictionary key should be one
        of the keys in `spectra1`, and the value should be a list
        `[ymin, ymax]` of the y-axis limits for that component. Only used
        if `plot_fdiff=True`.
    spectra1_for_fdiff, spectra2_for_fdiff : default=None
        Dictionaries of power spectra like `spectra1` and `spectra2` used
        for the fractional difference panel, if `plot_fdiff=True`. By
        default, `spectra1` and `spectra2` are used.

    Notes
    -----
    This function is used to produce Figure 3 in arXiv:XXXX.XXXX (!! TODO !!).
    """
    all_plt_components = ['tsz', 'ksz', 'cib', 'radio', 'kappa', 'cmb'] # in correct order for subplots
    plt_components = []
    plt_freqs = [None]
    for component in all_plt_components:
        if component in spectra1.keys():
            plt_components.append(component)
            if simutils.has_freq_dependent_component(component):
                plt_freqs = sorted(list(spectra1[component].keys())) 
    component_for_legend = 'ksz' if ('ksz' in plt_components) else plt_components[0]
    
    # check if we have pol.:
    if simutils.has_cmb(plt_components):
        pol = all([key in spectra1['cmb'][None].keys() for key in ['tt', 'ee', 'bb']])
    else:
        pol = False
    cmb_spec_keys = ['tt', 'ee', 'bb'] if pol else ['tt']
    default_cmb_ylims = [7e4, 2e10] if pol else [7e7, 1.2e10]
    cmb_label2 = label2 if (cmb_label2 is None) else cmb_label2

    # define size of plot and subplots:
    nplt = len(plt_components)
    nrow = int(nplt / ncol)
    while nrow * ncol < nplt:
        nrow += 1
    figsize = (4*ncol, 3.5*nrow)
    fdiff_plt_frac = 0.25 # fraction of plot used for frac. diff., for each component

    # settings for the plots (line styles, labels, etc.):
    kwargs1 = {'lw': 1.75, 'alpha': 0.7, }
    kwargs2 = {'ls': ':', 'lw': 2, 'alpha': 0.95}
    plt_colors = {30: 'tab:blue', 90: 'tab:green', 148: 'tab:olive', 
                  219: 'tab:orange', 277: 'tab:red', 350: 'tab:pink'}
    purple = '#bc8ae6' # for freq-independent components
    cmb_colors = {'tt': purple, 'ee': 'tab:cyan', 'bb': '#5cc46f'}

    axis_labelsize = 12
    component_labelsize = 14
    xlabels = {component: r'Multipole, $\ell$' for component in plt_components}
    ylabels = {component: r'$\frac{\ell(\ell+1)}{2\pi} C_\ell~$ [$\mu$K$^2$]' for component in plt_components}
    xlabels['kappa'] = r'Multipole, $L$'
    ylabels['kappa'] = r'$\frac{L^2 (L+1)^2}{4} C_L^{\phi\phi}$'
    ylabels['cmb'] =  r'$\ell^4 C_\ell~$ [$\mu$K$^2$]'
    fdiff_ylabel = '% Diff.'
    labels = {'tsz': 'tSZ', 'ksz': 'kSZ', 'cib': 'CIB', 'radio': 'Radio', 
              'kappa': 'Lensing convergence', 'cmb': 'Lensed CMB'}
    legend_kwargs = {'handlelength': 1.5, 'handletextpad': 0.5, 'columnspacing': 1}

    logy = {'cib': True, 'radio': True, 'tsz': False, 'ksz': False, 'kappa': True, 'cmb': True}
    ylims = {} if (ylims is None) else ylims
    yticks = {} if (yticks is None) else yticks
    fdiff_ylims = {} if (fdiff_ylims is None) else fdiff_ylims
        
    if plot_fdiff:
        if spectra1_for_fdiff is None:
            spectra1_for_fdiff = spectra1
        else:
            for component in plt_components:
                if component not in spectra1_for_fdiff:
                    spectra1_for_fdiff[component] = spectra1[component]

        if spectra2_for_fdiff is None:
            spectra2_for_fdiff = spectra2
        else:
            for component in plt_components:
                if component not in spectra2_for_fdiff:
                    spectra2_for_fdiff[component] = spectra2[component]

    # ----- make the plot : -----

    # if `show=False`, return the fig/subfigs/axes/etc:
    subfigs_dict = {}
    axs = {}
    fdiff_axs = {}
    fig = plt.figure(figsize=figsize, dpi=dpi)
    subfigs = fig.subfigures(nrow, ncol, wspace=0.025, hspace=-0.075)
    for i, component in enumerate(plt_components):
        # get the subfigure
        if (nrow > 1) and (ncol > 1):
            irow = i // ncol
            icol = i % ncol
            cfig = subfigs[irow, icol]
        else:
            cfig = subfigs[i]
        # get the axes
        if plot_fdiff:
            ax1, ax2 = cfig.subplots(2, 1, sharex=True, height_ratios=[1-fdiff_plt_frac, fdiff_plt_frac])
        else:
            ax1 = cfig.subplots()
            
        if 'cmb' in component:
            freq = None
            for spec_key in cmb_spec_keys:
                # plot spectra of HD sim:
                ells1, spec1 = utils.trim_spectrum_ell_range(spectra1[component][freq]['ells'], 
                                                             spectra1[component][freq][spec_key], 
                                                             lmin=lmin1, lmax=lmax1)
                ax1.plot(ells1, spec1 * ells1**4, color=cmb_colors[spec_key], **kwargs1)
                # plot theory
                ells2, spec2 = utils.trim_spectrum_ell_range(spectra2[component][freq]['ells'], 
                                                             spectra2[component][freq][spec_key], 
                                                             lmin=lmin2, lmax=lmax2)
                ax1.plot(ells2, spec2 * ells2**4, color='k', **kwargs2)
                # plot frac. diff.:
                if plot_fdiff:
                    ells1, spec1 = utils.trim_spectrum_ell_range(spectra1_for_fdiff[component][freq]['ells'], 
                                                                 spectra1_for_fdiff[component][freq][spec_key], 
                                                                 lmin=lmin1, lmax=lmax1)
                    ells2, spec2 = utils.trim_spectrum_ell_range(spectra2_for_fdiff[component][freq]['ells'], 
                                                                 spectra2_for_fdiff[component][freq][spec_key], 
                                                                 lmin=lmin2, lmax=lmax2)
                    plt_ells, plt_fdiff = utils.get_frac_diff(ells1, spec1, ells2, spec2, percent=True)
                    ax2.plot(plt_ells, plt_fdiff, color=cmb_colors[spec_key], lw=1)
            if add_legends:
                line2, = ax1.plot([], [], label=cmb_label2, color='k', **{**kwargs2, 'lw': 2.5})
                line1, = ax1.plot([], [], label=label1, color='k', **kwargs1)
                if pol:
                    # add an empty label to make legend nicer:
                    empty_line, = ax1.plot([], [], label=r'$~$', color='w') 
                    tt_line, = ax1.plot([], [], color=cmb_colors['tt'], label=r'$TT$', **kwargs1)
                    ee_line, = ax1.plot([], [], color=cmb_colors['ee'], label=r'$EE$', **kwargs1)
                    bb_line, = ax1.plot([], [], color=cmb_colors['bb'], label=r'$BB$', **kwargs1)
                    legend_ypos = 0.59 if plot_fdiff else 0.65
                    cmb_legend1_kwargs = {**legend_kwargs, 'ncol': 2, 'loc': 'lower right', 
                                          'bbox_to_anchor': (1, legend_ypos)}
                    cmb_legend2_kwargs = {**legend_kwargs, 'ncol': 3, 'loc': 'lower right', 
                                          'bbox_to_anchor': (0.975, legend_ypos), 'frameon': False}
                    first_legend = ax1.legend(handles=[line2, empty_line, line1], **cmb_legend1_kwargs)
                    ax1.add_artist(first_legend)
                    ax1.legend(handles=[tt_line, ee_line, bb_line], **cmb_legend2_kwargs)
                else:
                    ax1.legend(handles=[line2, line1], loc='center right', **legend_kwargs)
        
        else:
            freqs = plt_freqs if simutils.has_freq_dependent_component(component) else [None]
            for j, freq in enumerate(freqs):
                spec_key = 'kk' if (component == 'kappa') else 'tt'
                plt_color = purple if (component in ['cmb', 'kappa', 'ksz']) else plt_colors[freq]
                # plot spectra of HD sim:
                ells1, spec1 = utils.trim_spectrum_ell_range(spectra1[component][freq]['ells'], 
                                                             spectra1[component][freq][spec_key], 
                                                             lmin=lmin1, lmax=lmax1)
                ax1.plot(ells1, spec1, color=plt_color, **kwargs1)
                # plot spectra of s10-resolution sim:
                ells2, spec2 = utils.trim_spectrum_ell_range(spectra2[component][freq]['ells'], 
                                                             spectra2[component][freq][spec_key], 
                                                             lmin=lmin2, lmax=lmax2)
                ax1.plot(ells2, spec2, color='k', **kwargs2)
                # plot frac. diff.:
                if plot_fdiff:
                    ells1, spec1 = utils.trim_spectrum_ell_range(spectra1_for_fdiff[component][freq]['ells'], 
                                                                 spectra1_for_fdiff[component][freq][spec_key], 
                                                                 lmin=lmin1, lmax=lmax1)
                    ells2, spec2 = utils.trim_spectrum_ell_range(spectra2_for_fdiff[component][freq]['ells'], 
                                                                 spectra2_for_fdiff[component][freq][spec_key], 
                                                                 lmin=lmin2, lmax=lmax2)
                    plt_ells, plt_fdiff = utils.get_frac_diff(ells1, spec1, ells2, spec2, percent=True)
                    if component == 'tsz': 
                        # add a slight horizontal offset, since all curves are on top of each other
                        plt_ells += j*25
                    ax2.plot(plt_ells, plt_fdiff, color=plt_color, lw=1)
            if add_legends and (component == component_for_legend): # make the legend
                # plot empty arrays w/ label for the legends:
                ax1.plot([], [], label=label2, color='k', **{**kwargs2, 'lw': 2.5})
                ax1.plot([], [], label=label1, color='k', **kwargs1)
                for freq in plt_freqs:
                    ax1.plot([], [], color=plt_colors[freq], label=f'{freq} GHz', lw=1.5)
                ax1.legend(ncol=2, **legend_kwargs)

        # label the component
        xpos = 0.04 if (component in ['cib', 'radio']) else 0.96
        horizontalalignment = 'left' if (component in ['cib', 'radio']) else 'right'
        ax1.text(xpos, 0.96, labels[component], fontsize=component_labelsize, transform=ax1.transAxes, 
                 verticalalignment='top', horizontalalignment=horizontalalignment)

        # set axis limits, labels, etc.:
        if (component in ylims) and (ylims[component] is not None):
            ax1.set_ylim(ylims[component])
        if logy[component]:
            ax1.set_yscale('log')
            ax1.tick_params(axis='y', which='minor', left=False)
        if (component in yticks) and (yticks[component] is not None):
            ax1.set_yticks(yticks[component])
        ax1.set_ylabel(ylabels[component], fontsize=axis_labelsize)

        if plot_fdiff:
            ax2.axhline(color='k', lw=0.5)
            if (component in fdiff_ylims) and (fdiff_ylims[component] is not None):
                ax2.set_ylim(fdiff_ylims[component])
            ax2.set_ylabel(fdiff_ylabel, fontsize=axis_labelsize)
            ax2.set_xlabel(xlabels[component], fontsize=axis_labelsize)
        else:
            ax1.set_xlabel(xlabels[component], fontsize=axis_labelsize)

        plt.subplots_adjust(hspace=0)
        subfigs_dict[component] = cfig
        axs[component] = ax1
        if plot_fdiff:
            fdiff_axs[component] = ax2
    
    if plot_fname is not None:
        plt.savefig(plot_fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    elif plot_fdiff:
        return fig, subfigs_dict, axs, fdiff_axs
    else:
        return fig, subfigs_dict, axs
    
    
def plot_smallscale_ksz_kappa_spectra(hd_spectra, s10_spectra, theo_spectra, plot_fdiff=False,
                                      show=True, plot_fname=None, dpi=500, lmin=1000, lmax=20000):
    """Compare the kSZ and lensing convergence simulation power spectra
    to their lower-resolution counterparts and to theory power spectra.

    Parameters
    ----------
    hd_spectra : dict of dict of array_like of float
        A nested dictionary of the power spectra of the kSZ and lensing
        convergence simulations. It has the following keys and values:
        - `hd_spectra['ksz']` is a dictionary with a key `'tt'` for
            the binned kSZ simulation power spectrum in units of uK^2
            and multiplied by `ell * (ell + 1) / (2 * pi)`, and a key
            `'ells'` for the bin centers.
        - `hd_spectra['kappa']` is a dictionary with a key `'kk'` for
            the binned lensing convergence simulation power spectrum,
            and a key `'ells'` for the bin centers.
    s10_spectra : dict of dict of array_like of float
        Same as `hd_spectra`,  but for the power spectra of the
        corresponding lower-resolution simulations on the same patch
        of the sky.
    theo_spectra : dict of dict of array_like of float
        Same as `hd_spectra`,  but for the corresponding binned theory
        power spectra.
    show : bool, default=True
        Whether to display the plot by calling `matplotlib.pyplot.show()`.
    plot_fname : str or None, default=None
        The file name used to save the plot. By default, the plot will not
        be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The `matplotlib` figure used to make the plot. Only returned
        if `show=False`.
    axs : dict of matplotlib.axes.Axes
        A dictionary with a key (`str`) for each key (`'ksz'` or
        `'kappa'`) in `hd_spectra`. The values are the corresponding
        subplots used to plot the power spectra of that component. Only
        returned if `show=False`.
    fdiff_axs : dict of matplotlib.axes.Axes
        A dictionary with a key (`str`) for each key (`'ksz'` or
        `'kappa'`) in `hd_spectra`. The values are the corresponding
        subplots used to plot the fractional  difference of the power
        spectra of each component. Only returned if `show=False` and
        `plot_fdiff=True`.

    Other Parameters
    ----------------
    plot_fdiff : bool, default=False
        Whether to add a lower panel to the plot for each component showing
        the fractional difference between the power spectra in `spectra1`
        and `spectra2`.
    dpi : int, default=500
        The resolution of the plot, in dots-per-inch.
    lmin : int, default=1000
        The minimum multipole of the power spectra to plot.
    lmax : int, default=20000
        The maximum multipole of the power spectra to plot.

    Notes
    -----
    This function is used to produce Figure 4 in arXiv:XXXX.XXXX (!! TODO !!).
    """
    # settings for the plot:
    labelsize = 12  # axis labels
    fontsize = 14 # text for name of component
    fdiff_ymax = 10
    fdiff_ymin = -fdiff_ymax
    fdiff_yticks = [-10, -5, 0, 5, 10]
    xticks = np.array([0, 5000, 10000, 15000, 20000])
    xticks = xticks[xticks >= lmin]
    theo_plt_kwargs = {'lw': 3, 'alpha': 0.75, 'color': 'gray'} # theory curves
    s10_plt_kwargs = {'lw': 2, 'color': 'k'} # s10 sim spectra
    hd_plt_kwargs = {'lw': 2, 'color': 'red', 'ls': '--'} # hd sim spectra
    legend_kwargs = {'handlelength': 1.5, 'columnspacing': 1}
    shading_kwargs = {'color': 'lightgray', 'alpha': 0.5, 'hatch': '/'} # for shading region of s10 sim power
    txt_kwargs = {'x': 0.95, 'y': 0.95, 'verticalalignment': 'top', 'horizontalalignment': 'right',
                  'fontsize': fontsize, 'bbox': {'facecolor': 'w', 'edgecolor': 'k', 'boxstyle': None}}

    # make the plot:
    axs = {}
    fdiff_axs = {}
    # determine if we have spectra for both kappa and ksz:
    components = []
    if 'kappa' in hd_spectra.keys():
        components.append('kappa')
    if 'ksz' in hd_spectra.keys():
        components.append('ksz')
    ncol = len(components)
    # this could be done more nicely, but it works ...
    if ncol > 1:
        if plot_fdiff:
            fig, ((axs['kappa'], axs['ksz']), (fdiff_axs['kappa'], fdiff_axs['ksz'])) = plt.subplots(2, ncol, figsize=(9, 4.5), height_ratios=[3, 1], dpi=dpi)
            axs_list = [axs['kappa'], axs['ksz'], fdiff_axs['kappa'], fdiff_axs['ksz']]
            fdiff_axs_list = [fdiff_axs['kappa'], fdiff_axs['ksz']]
        else:
            fig, (axs['kappa'], axs['ksz']) = plt.subplots(1, ncol, figsize=(9, 4), dpi=dpi)
            axs_list = [axs['kappa'], axs['ksz']]
            fdiff_axs_list = []
    else:
        component = components[0]
        if plot_fdiff:
            fig, (axs[component], fdiff_axs[component]) = plt.subplots(2, ncol, figsize=(4.5, 4.5), height_ratios=[3, 1], dpi=dpi)
            axs_list = [axs[component], fdiff_axs[component]]
            fdiff_axs_list = [fdiff_axs[component]]
        else:
            fig, axs[component] = plt.subplots(1, ncol, figsize=(4.5, 4), dpi=dpi)
            axs_list = [axs[component]]
            fdiff_axs_list = []

    if 'kappa' in components:
        component = 'kappa'
        s10_lmax = 4000
        spec_key = 'kk'

        theo_ells, theo_spec = utils.trim_spectrum_ell_range(theo_spectra[component]['ells'], theo_spectra[component][spec_key], lmin=lmin, lmax=lmax)
        axs['kappa'].plot(theo_ells, theo_spec, label="Small-scale Theory", **theo_plt_kwargs)
        s10_ells, s10_spec = utils.trim_spectrum_ell_range(s10_spectra[component]['ells'], s10_spectra[component][spec_key], lmin=lmin, lmax=s10_lmax)
        axs['kappa'].plot(s10_ells, s10_spec, label="S10 (0.86')", **s10_plt_kwargs)
        hd_ells, hd_spec = utils.trim_spectrum_ell_range(hd_spectra[component]['ells'], hd_spectra[component][spec_key], lmin=lmin, lmax=lmax)
        axs['kappa'].plot(hd_ells, hd_spec, label="This Work (0.04')", alpha=0.75, **hd_plt_kwargs)

        axs['kappa'].axvspan(0, s10_lmax, **shading_kwargs)
        axs['kappa'].text(s="Lensing Convergence", transform=axs['kappa'].transAxes, **txt_kwargs)
        axs['kappa'].set_ylabel(r"$\frac{L^2 (L+1)^2}{4} C_L^{\phi\phi}$", fontsize=labelsize)
        axs['kappa'].set_yscale('log')
        axs['kappa'].legend(bbox_to_anchor=(1, 0.75), loc='upper right', **legend_kwargs)

        if plot_fdiff:
            s10_fdiff_ells, s10_fdiff = utils.get_frac_diff(s10_ells, s10_spec, theo_ells, theo_spec, percent=True)
            hd_fdiff_ells, hd_fdiff = utils.get_frac_diff(hd_ells, hd_spec, theo_ells, theo_spec, percent=True)
            fdiff_axs['kappa'].plot(s10_fdiff_ells, s10_fdiff, **s10_plt_kwargs)
            fdiff_axs['kappa'].plot(hd_fdiff_ells, hd_fdiff, **hd_plt_kwargs)
            fdiff_axs['kappa'].axhline(0, color='gray', linestyle='--', linewidth=1)
            fdiff_axs['kappa'].set_xlabel(r"Multipole, $L$", fontsize=labelsize)
            fdiff_axs['kappa'].axvspan(0, s10_lmax, **shading_kwargs)
        else:
            axs['kappa'].set_xlabel(r"Multipole, $L$", fontsize=labelsize)

    if 'ksz' in components:
        component = 'ksz'
        s10_lmax = 8000
        spec_key = 'tt'

        theo_ells, theo_spec = utils.trim_spectrum_ell_range(theo_spectra[component]['ells'], theo_spectra[component][spec_key], lmin=lmin, lmax=lmax)
        axs['ksz'].plot(theo_ells, theo_spec, label="Small-scale Theory", **theo_plt_kwargs)
        s10_ells, s10_spec = utils.trim_spectrum_ell_range(s10_spectra[component]['ells'], s10_spectra[component][spec_key], lmin=lmin, lmax=s10_lmax)
        axs['ksz'].plot(s10_ells, s10_spec, label="S10 (0.43')", **s10_plt_kwargs)
        hd_ells, hd_spec = utils.trim_spectrum_ell_range(hd_spectra[component]['ells'], hd_spectra[component][spec_key], lmin=lmin, lmax=lmax)
        axs['ksz'].plot(hd_ells, hd_spec, label="This Work (0.04')", alpha=0.75, **hd_plt_kwargs)

        axs['ksz'].axvspan(0, s10_lmax, **shading_kwargs)
        axs['ksz'].text(s="kSZ", transform=axs['ksz'].transAxes, **txt_kwargs)
        axs['ksz'].set_ylabel(r"$\frac{\ell(\ell+1)}{2\pi}~C_\ell$ [$\mu K^2$]", fontsize=labelsize)
        axs['ksz'].legend(bbox_to_anchor=(1, 0.45), loc='upper right', **legend_kwargs)

        if plot_fdiff:
            s10_fdiff_ells, s10_fdiff = utils.get_frac_diff(s10_ells, s10_spec, theo_ells, theo_spec, percent=True)
            hd_fdiff_ells, hd_fdiff = utils.get_frac_diff(hd_ells, hd_spec, theo_ells, theo_spec, percent=True)
            fdiff_axs['ksz'].plot(s10_fdiff_ells, s10_fdiff, **s10_plt_kwargs)
            fdiff_axs['ksz'].plot(hd_fdiff_ells, hd_fdiff, **hd_plt_kwargs)
            fdiff_axs['ksz'].axhline(0, color='gray', linestyle='--', linewidth=1)
            fdiff_axs['ksz'].set_xlabel(r"Multipole, $\ell$", fontsize=labelsize)
            fdiff_axs['ksz'].axvspan(0, s10_lmax, **shading_kwargs)
        else:
            axs['ksz'].set_xlabel(r"Multipole, $\ell$", fontsize=labelsize)

    # finish the plot:
    for ax in axs_list:
        ax.margins(0.05)
        ax.set_xlim([lmin, lmax])
        ax.set_xticks(xticks)
    if plot_fdiff:
        for ax in fdiff_axs_list:
            ax.set_ylabel('% Diff.', fontsize=labelsize)
            ax.set_ylim([fdiff_ymin, fdiff_ymax])
            ax.set_yticks(fdiff_yticks)
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    if plot_fname is not None:
        plt.savefig(plot_fname, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        if plot_fdiff:
            return fig, axs, fdiff_axs
        else:
            return fig, axs

