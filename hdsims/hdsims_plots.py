"""Plot the simulations and their power spectra."""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pixell import enmap, colorize
from . import utils, siminfo as si, maps, fgcatalogs, simutils, plots, hdsims_spectra


class HDSimsPlots(hdsims_spectra.HDSimsSpectra):
    """Plot the ultrahigh-resolution simulations and their power spectra.

    The simulation power spectra are also compared to either the power
    spectra of the corresponding lower-resolution simulations on the same
    patch of the sky, or the corresponding theory power spectra.

    Attributes
    ----------
    default_components_for_plot : list of str
        A default list of individual map components to plot: includes
        `'ksz'`, `'tsz'`, `'cib'`, `'radio'`, `'cmb'`, `'kappa'` for the
        kSZ, tSZ, CIB, radio, lensed CMB, and lensing convergence maps,
        respectively.

    Notes
    -----
    Attributes listed above without a description will have the same value
    as the corresponding parameter passed when initializing the class, or
    the default value if it is not passed. Inherited attributes are not
    listed above.
    """

    def __init__(self, hd_sims_dir, lowres_sims_dir=None,
                 freqs=si.freqs, components=si.components,
                 ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr, width=si.width, height=si.height,
                 apod_width=si.apod_width, map_apod_width=None, res=si.hd_res,
                 cmb_seed=si.cmb_seed, pol=True, noise_seeds=si.noise_seeds,
                 lmax=si.lmax4spectra, bin_edges=None, bin_info=None,
                 lmax4alms=si.lmax4alms, lmax4theo=si.lmax4theo,
                 verbose=False, log=None, make_output_dirs=True, **kwargs):
        """Initialize the `HDSimsPlots` class for a given patch on the sky.

        Parameters
        ----------
        hd_sims_dir : str
            The path to the directory where all of the output files will
            be saved. This directory will be created if it does not
            already exist.
        lowres_sims_dir : str or None, default=None
            The path to the directory where the full-sky lower-resolution
            simulations and catalogs have been saved. This is required if
            the ultrahigh-resolution simulations have not already been
            generated and saved; otherwise, it is not used.
        freqs : list of int, default=[30, 90, 148, 219, 277, 350]
            A list of map frequencies (in GHz). Each frequency in the list
            must be one of `30`, `90`, `148`, `219`, `277`, or `350`.
        components : list of str, optional
            A list of map components to generate. Each component in the
            list must be one of: `'tsz'` for the thermal SZ (tSZ); `'ksz'`
            for the kinetic SZ (kSZ); `'cib'` for the cosmic infrared
            background (CIB); `'radio'` for radio galaxies; and either
            `'cmb'` or `'unlensed_cmb'` for the lensed or unlensed CMB,
            respectively. If `'cmb'` is in the list, then unlensed CMB and
            lensing convergence (`'kappa'`) simulations will also be
            generated and saved. By default, the list includes all
            components except the `'unlensed_cmb'`.
        ra_ctr, dec_ctr : int or float, optional
            The right ascension (R.A.) and declination (dec.), in degrees,
            of the center of the patch of sky. The defaults are `ra_ctr=6`
            and `dec_ctr=6`.
        width : int or float, default=10
            The width (in degrees) of the region of the maps to be used
            for analysis, e.g. when taking the power spectrum of the
            simulations.
        height : int or float, optional
            The height (in degrees) of the region of the maps to be used
            for analysis. If the `height` is not provided, it is assumed
            to be equal to the `width`.
        apod_width : int or float, default=0.5
            The width (in degrees) of the region along each edge of the
            map that will be apodized before calculating its power
            spectrum.
        cmb_seed : int, default=58
            The random seed to use when generating a realization of the
            unlensed CMB from a theory power spectra.
        pol : bool, default=True
            If `pol=True`, CMB temperature and polarization (T, Q, and U)
            maps will be generated. Otherwise, only the temperature map
            will be generated.
        lmax : int, default=21000
            The maximum multipole to use when calculating the power
            spectra of the simulations.

        Other Parameters
        ----------------
        bin_edges : array_like of int or array_like of float, optional
            An array of bin edges used to bin the power spectra.
            The first element should be the lower edge of the first bin
            (typically, `bin_edges[0] = 2`, which should be the minimum
            value), and the remaining elements are the upper edges of each
            bin. By default, uniform binning with a bin width of `200`
            is used. See the 'Notes' section for additional information.
        bin_info : str or None, default=None
            If you use non-default `bin_edges`, you must also pass a short
            name (no special characters) that describes the binning. This
            will be used in file names for binned power spectra (and other
            files that are needed to calculate the binned power spectra).
            The `bin_info` parameter is ignored when the default
            `bin_edges` are used. See the 'Notes' section for additional
            information.
        noise_seeds : dict of int, optional
            A dictionary of integer random seeds used to generate
            realizations of the white noise at each frequency. The keys
            should be the integer frequencies (GHz), and each value
            should be a single `seed` for the temperature map.
            If `pol=True`, the noise seeds for the polarization Q and U
            noise maps are `seed+1000` and `seed+2000`, respectively.
            The default seeds are `1`, `2`, `3`, `4`, `5`, and `6` for 30,
            90, 148, 219, 277, and 350 GHz, respectively.
        lmax4alms : int, default=24000
            The maximum multipole to use when taking the spherical
            harmonic transform of a map.
        lmax4theo : int, default=40000
            The maximum multipole of any theory curves used to generate
            the simulations.
        map_apod_width : int or float, optional
            The width (in degrees) of the region along each edge of the
            map that will be apodized before taking any Fourier or
            spherical harmonic transforms. By default, the `apod_width`
            is used.
        res : int or float, default=0.04
            The resolution of the maps, in arcminutes.
        verbose : bool, default=False
            Whether to print messages describing the progress of some
            calculations.
        log : logging.Logger, optional
            A `logging.Logger` instance to use when `verbose=True`. If a
            `log` is passed, any messages will be passed to `log.info`.
            Otherwise, messages will be passed to the `print` function.
        make_output_dirs : bool, default=True
            Whether to create the sub-directories under the `hd_sims_dir`
            where the output files will be saved. This should not be
            changed, but it is provided to, e.g., allow you to check where
            the files will be saved before generating the simulations.
        **kwargs : dict
            Any keyword arguments that are needed to initialize the parent
            class for the lower-resolution, full-sky simulations.
        """
        super().__init__(hd_sims_dir, lowres_sims_dir=lowres_sims_dir, freqs=freqs, components=components,
                         ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height,
                         apod_width=apod_width, map_apod_width=map_apod_width, res=res,
                         cmb_seed=cmb_seed, pol=pol, noise_seeds=noise_seeds, lmax4alms=lmax4alms, lmax4theo=lmax4theo,
                         verbose=verbose, log=log, make_output_dirs=make_output_dirs, **kwargs)
        self.default_components_for_plot = si.components


    def components_for_plot(self, **kwargs):
        """Return a list of map components to plot.

        Parameters
        ----------
        **kwargs : dict
            There is a single recognized keyword argument: the
            `components` (`list` of `str`) to plot. The default list is
            given by the `components` attribute. If the default list
            contains the lensed CMB (`'cmb'`), then the lensing
            convergence (`'kappa'`) will also be added to the list, if it
            is not there already. If both `'cmb'` and `'unlensed_cmb'` are
            in the default list, `'unlensed_cmb'` will be removed.

        Returns
        -------
        components : list of str
            The list of map components to plot.
        """
        if 'components' in kwargs: # don't modify the list
            components = simutils.validate_sim_component_names(kwargs['components'])
        else: # use the default list
            components = self.components.copy()
            if ('cmb' in components) and ('kappa' not in components):
                # if default includes the lensed cmb, add kappa:
                components.append('kappa')
            if ('cmb' in components) and ('unlensed_cmb' in components):
                components.remove('unlensed_cmb')
        return components


    def plot_sim(self, show=True, fname=None, plot_tqu=True, **kwargs):
        """Plot a simulation.

        The simulation may be a single map, or a set of temperature and
        polarization (T, Q, U) maps.

        Parameters
        ----------
        show : bool, default=True
            Whether to display the plot (by calling
            `matplotlib.pyplot.show()`).
        fname : str or None, default=None
            To save the plot, pass the file name to `fname`. By default,
            the plot will not be saved.
        plot_tqu : bool, default=True
            Plot the temperature and polarization maps.
            If `plot_tqu=False`, only the temperature map will be plotted.
            Only used if the simulation has both temperature and
            polarization and `pol` is `True`.
        **kwargs : dict
            Additional keyword arguments passed to the `get_sim` method
            for the simulation to plot (including the map `freq` and
            `components`, and whether to convolve the `beam` or add
            `noise`), and to the `hdsims.plots.plot_map` and
            `hdsims.plots.plot_maps` functions.

        See Also
        --------
        get_sim : The simulation that will be plotted.
        hdsims.plots.plot_map, hdsims.plots.plot_maps :
            Plot a single map or a list of maps.

        Notes
        -----
        The range of the colorbar and the color map used will be
        automatically set based on the map frequency and the individual
        components that it contains.

        If the `fname` passed is not an absolute path, the plot will be
        saved in the directory returned by the `plot_dir` method.
        """
        sim = self.get_sim(**kwargs)
        components = self.get_kwarg('components', default=self.map_components, **kwargs)
        pol = self.get_kwarg('pol', **kwargs)
        freq = self.get_kwarg('freq', default=None, **kwargs)
        noise = self.get_kwarg('noise', default=False, **kwargs)
        if (fname is not None) and (not os.path.isabs(fname)):
            fname = os.path.join(self.plots_dir(), fname)

        if pol and (simutils.has_cmb(components) or noise) and plot_tqu:
            plt_maps = [sim[0], sim[1], sim[2]] # T, Q, U
            temp_clims = plots.get_colorbar_range(components, freq=freq, pol=False, noise=noise, symmetric=True)
            pol_clims = plots.get_colorbar_range(components, freq=freq, pol=True, noise=noise, symmetric=True)
            colorbar_ranges = [temp_clims, pol_clims, pol_clims]
            labels = ['T', 'Q', 'U']
            plots.plot_maps(plt_maps, labels=labels, colorbar_ranges=colorbar_ranges,
                            ncol=3, fname=fname, show=show, **kwargs)
        else:
            if len(sim.shape) > 2:
                sim = sim[0] # only T
            vmin, vmax = plots.get_colorbar_range(components, freq=freq, pol=False, noise=noise, symmetric=False)
            full_cmap = plt.get_cmap('planck')
            if (vmin < 0) and (vmax > 0): # use full color map
                cmap = full_cmap
                # use a symmetric range for colorbar:
                clim = max([abs(vmin), vmax])
                vmin = -clim
                vmax = clim
            elif vmin < 0: # use 'negative' half of color map
                cmap = plots.truncate_colormap(full_cmap, minval=0.0, maxval=0.5)
            else: # use 'positive' half of color map
                cmap = plots.truncate_colormap(full_cmap, minval=0.5, maxval=1.0)
            plt_kwargs = {**kwargs, 'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
            if 'kappa' in components:
                plt_kwargs['cbar_label'] = '' # kappa map is dimensionless
            plots.plot_map(sim, show=show, fname=fname, **plt_kwargs)


    def plot_sim_maps(self, freq, show=True, save=True, fname=None, dpi=300, **kwargs):
        """Plot the simulations at a single frequency.

        All simulations except the lensing convergence map will be
        convolved with the CAR pixel window function and the CMB-HD beam.

        Parameters
        ----------
        freq : int
            The frequency (GHz) of the simulations. Must be `30`, `90`,
            `148`, `219`, `277`, or `350`.
        show : bool, default=True
            Whether to display the plot (by calling
            `matplotlib.pyplot.show()`).
        save : bool, default=True
            Whether to save the plot.
        fname : str or None, default=None
            The file name used to save the plot if `save=True`. The
            default file name is returned by the
            `get_sim_map_plot_fname` method.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `matplotlib` figure used to make the plot. Only returned
            if `show=False`.
        subfigs_dict : dict of matplotlib.figure.SubFigure
            A dictionary with a key (`str`) for each map component in the
            list of `components` to plot and the corresponding
            `matplotlib` subfigure for the plot of that component as the
            values. Only returned if `show=False`.
        axs, caxs : dict of matplotlib.axes.Axes
            Dictionaries with a key (`str`) for each map component in the
            list of `components` to plot. The values in `axs` are the
            corresponding subplots (of the corresponding subfigure) used
            to plot each component, and the values in `caxs` are the
            subplots used for the color bar. Only returned if
            `show=False`.

        Other Parameters
        ----------------
        dpi : int, default=300
            The resolution of the plot in dots-per-inch, passed to
            `matplotlib.pyplot.figure`.
        **kwargs : dict
            The additional keyword arguments are:
            - `components` (`list` of `str`) : A list of individual map
                components to plot. The default list is returned by the
                `components_for_plot` method.
            - `cmb_seed` (`int`) : The random seed used to generate the
                unlensed CMB realization. The default is given by the
                `cmb_seed` attribute. Only used if `'cmb'` or
                `'unlensed_cmb'` is in the list of components.
            - `pol` (`bool`) : Whether to plot the CMB polarization
                (Q and U) maps. The default is given by the `pol`
                attribute. Only used if `'cmb'` or `'unlensed_cmb'` is in
                the list of components.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_sim` and
                `get_component_name` methods.

        See Also
        --------
        hdsims.plots.plot_maps

        Notes
        -----
        This method is used to produce Figure 2 in arXiv:XXXX.XXXX (!! TODO !!).
        """
        components = self.components_for_plot(**kwargs)
        self.infomsg(f"getting the {freq} GHz sims to plot for {components = }")
        sims = {}
        for component in components:
            beam = False if (component == 'kappa') else True
            sim_kwargs = {**kwargs, 'noise': False, 'components': [component]}
            sims[component] = self.get_sim(freq=freq, beam=beam, **sim_kwargs)
        self.infomsg("making the plot")
        if save:
            if fname is None:
                fname = self.get_sim_map_plot_fname(freq, **kwargs)
        else:
            fname = None
        plt_output = plots.plot_sims(sims, freq, show=show, plot_fname=fname, dpi=dpi)
        if save:
            self.infomsg(f'saved {fname}')
        if not show:
            return plt_output


    def get_sim_map_plot_fname(self, freq, **kwargs):
        """Return the default file name for the plot of the simulations at
        a single frequency.

        Parameters
        ----------
        freq : int
            The frequency (GHz) of the simulations. Must be `30`, `90`,
            `148`, `219`, `277`, or `350`.
        **kwargs : dict
            The additional keyword arguments are:
            - `components` (`list` of `str`) : A list of individual map
                components to plot. The default list is returned by the
                `components_for_plot` method.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_component_name` method.

        Returns
        -------
        fname : str
            The file name.

        Notes
        -----
        By default, the plot will be saved as a PNG file in the directory
        returned by the `plot_dir` method.
        """
        components = simutils.validate_sim_component_names(self.components_for_plot(**kwargs))
        use_default_components = set(components) == set(self.default_components_for_plot)
        fname_info = [f'hd_sims']
        if simutils.has_freq_dependent_component(components): # only include freq if it's needed
            fname_info.append(f'{freq:03d}GHz')
        if use_default_components: # using default (all components) ; add any info about other non-default kwargs
            for component in components:
                component_name = self.get_component_name(component, **kwargs)
                default_component_name = self.get_component_name(component)
                if component_name != default_component_name:
                    fname_info.append(component_name)
        else: # get list of the components in the plot
            component_info = self._map_component_list2str(**kwargs)
            fname_info.append(component_info)
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.plots_dir(), f'{fname_root}.png')
        return fname


    def plot_sim_spectra_comparison(self, show=True, save=True, fname=None, save_intermediate_maps=False,
                                    dpi=500, plot_fdiff=False, use_fig_settings=False, **kwargs):
        """Plot the power spectra of the simulations, and compare with the
        power spectra of their lower-resolution counterparts (for tSZ,
        kSZ, CIB, radio, lensing convergence) or to the corresponding
        theory power spectra (lensed or unlensed CMB).

        Parameters
        ----------
        show : bool, default=True
            Whether to display the plot (by calling
            `matplotlib.pyplot.show()`).
        save : bool, default=True
            Whether to save the plot.
        fname : str or None, default=None
            The file name used to save the plot if `save=True`. The
            default file name is returned by the
            `get_sim_spectra_plot_fname` method.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `matplotlib` figure used to make the plot. Only returned
            if `show=False`.
        subfigs_dict : dict of matplotlib.figure.SubFigure
            A dictionary with a key (`str`) for each map component in the
            list of `components` to plot and the corresponding
            `matplotlib` subfigure for the plot of that component as the
            values. Only returned if `show=False`.
        axs : dict of matplotlib.axes.Axes
            A dictionary with a key (`str`) for each map component in the
            list of `components` to plot. The values are the corresponding
            subplots (of the corresponding subfigure) used to plot the
            power spectra of that component. Only returned if `show=False`.
        fdiff_axs : dict of matplotlib.axes.Axes
            A dictionary with a key (`str`) for each map component in the
            list of `components` to plot. The values are the corresponding
            subplots (of the corresponding subfigure) used to plot the
            fractional difference of the power spectra of each component.
            Only returned if `show=False` and `plot_fdiff=True`.

        Other Parameters
        ----------------
        save_intermediate_maps : bool, default=False
            Whether to save the intermediate, lower-resolution maps. It is
            recommended to pass `save_intermediate_maps=True` to avoid
            needing to obtain the intermediate, lower-resolution maps
            multiple times.
        dpi : int, default=500
            The resolution of the plot in dots-per-inch, passed to
            `matplotlib.pyplot.figure`.
        plot_fdiff : bool, default=False
            Whether to add a lower panel to the plot for each component
            showing the fractional difference between the two sets of
            power spectra being compared.
        use_fig_settings : bool, default=False
            Whether to use the settings that produced Figure 3 in
            arXiv:XXXX.XXXX (!! TODO !!). This will fix the y-axis limits
            and tick labels.
        **kwargs : dict
            The additional keyword arguments for the power spectra to plot
            are:
            - `components` (`list` of `str`) : A list of the simulation
                components (extragalactic foregrounds, lensing
                convergence, lensed or unlensed CMB) to plot. Each element
                of the list must be a valid `component` that can be passed
                to the `get_signal_sim_power` method. The default is the
                list returned by the `components_for_plot` method.
            - `freqs` (`list` of `int`) : A list of frequencies (GHz) for
                the plot. Each element in the list must be a valid
                numerical frequency that can be passed to the
                `get_signal_sim_power` method. The default is given by the
                `freqs` attribute.
            - `cmb_seed` (`int`) : The random seed used to generate the
                unlensed CMB realization. The default is given by the
                `cmb_seed` attribute. Only used if `'cmb'` or
                `'unlensed_cmb'` is in the list of `components`.
            - `pol` (`bool`) : Whether to plot the CMB polarization power
                spectra. The default is given by the `pol` attribute. Only
                used if `'cmb'` or `'unlensed_cmb'` is in the list of
                `components`.
            - `lmax` (`int`): The maximum multipole used to calculate the
                power spectra of the simulations. The default is given by
                the `lmax` attribute.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_intermediate_sim_power`
                method.
            The additional keyword arguments for the plot itself are only
            used if `use_fig_settings=False`; they are:
            - `ylims` (`dict` of `list` of `float`) : The y-axis limits
                for any of the power spectra plots. By default, no limits
                will be set. Each dictionary key should be one of the
                names in `components`, and the value should be a list
                `[ymin, ymax]` of the y-axis limits for that component.
            - `yticks` (`dict` of `array_like` of `float`) : The y-axis
                ticks for any of the power spectra plots. By default, the
                ticks are not explicitly set. Each dictionary key should
                be one of the names in `components`, and the value should
                be the ticks to use for its y-axis.
            - `fdiff_ylims` (`dict` of `list` of `float`) : The y-axis
                limits for any of the fractional difference plots. By
                default, no limits will be set. Each dictionary key should
                be one of the names in `components`, and the value should
                be a list `[ymin, ymax]` of the y-axis limits for that
                component. Only used if `plot_fdiff=True`.

        Notes
        -----
        This method is used to produce Figure 3 in arXiv:XXXX.XXXX (!! TODO !!).

        If `plot_fdiff=True` and `'radio'` is in the list of `components`,
        we will remove very bright radio sources near the edges of both
        the HD and lower-resolution maps before taking and comparing their
        power spectra. This is only done for the spectra in the fractional
        difference panel ; the spectra plotted in the upper panel are not
        changed in any way. In particular, for the maps used in the
        fractional difference panel, we remove any radio sources with flux
        greater than 100 mJy at 90 GHz that are located within a distance
        given by the `apod_width` attribute from the map edges. These
        bright radio sources will fall under the apodization mask applied
        to the maps before taking the power spectra; due to the difference
        in resolution, the apodization will reduce their amplitude by a
        different amounts in the lower-resolution vs. HD maps, leading to
        about a 1% difference between their power spectra.
        """
        self.infomsg(f"plotting comparison between HD sim spectra and either"
                     " lower-resolution sim spectra (for FGs + kappa) or theory (for lensed CMB)")
        # spectra1 = HD sim spectra;
        # spectra2 = lower-resolution sim spectra (FGs or kappa)
        #            or theory spectra (CMB)
        spectra_kwargs = {**kwargs, 'save_intermediate_maps': save_intermediate_maps}
        spectra1, spectra2 = self._get_spectra_for_comparison_plot(**spectra_kwargs)
        if plot_fdiff and ('radio' in spectra1.keys()):
            spectra1_for_fdiff, spectra2_for_fdiff = self._get_radio_spectra_for_comparison_plot(**kwargs)
        else:
            spectra1_for_fdiff = None
            spectra2_for_fdiff = None

        # settings for the plot (labels, axis limits, etc.):
        label1 = 'This Work'
        label2 = f'Original {self.lowres_name}'
        cmb_label2 = 'Theory'
        lmin = 200
        lmax = min([20000, self.lmax])
        if 'cmb' in spectra1.keys():
            pol = len(spectra1['cmb'][None].keys()) > 2
        else:
            pol = False
        if use_fig_settings:
            cmb_ylims = [7e4, 2e10] if pol else [7e7, 1.2e10]
            ylims = {'tsz': [-0.5, 22], 
                     'ksz': [-0.05, 1.25],
                     'radio': [5e-1, 1.5e8],
                     'cib': [5e-3, 1.5e5], 'kappa': None, 'cmb': cmb_ylims}
            yticks = {component: None for component in spectra1.keys()}
            yticks['cib'] = [0.1, 10, 1e3, 1e5]
            yticks['radio'] = [10, 1e3, 1e5, 1e7]
            fdiff_ylims = {'tsz': [-0.2, 0.05],  'ksz': [-0.06, 0.06], 'radio': [-0.75, 0.75],
                           'cib': [-1, 2], 'kappa': None, 'cmb': [-7.5, 7.5]}
        else:
            ylims = None if ('ylims' not in kwargs) else kwargs['ylims']
            yticks = None if ('yticks' not in kwargs) else kwargs['yticks']
            fdiff_ylims = None if ('fdiff_ylims' not in kwargs) else kwargs['fdiff_ylims']

        # make the plot
        if save:
            if fname is None:
                fname_kwargs = {**kwargs, 'plot_fdiff': plot_fdiff}
                fname = self.get_sim_spectra_plot_fname(**fname_kwargs)
        else:
            fname = None
        plt_output = plots.plot_sim_spectra_comparison(spectra1, spectra2, label1, label2, cmb_label2=cmb_label2,
                                                       lmin1=lmin, lmax1=lmax, lmin2=lmin, lmax2=lmax,
                                                       ylims=ylims, yticks=yticks,
                                                       plot_fdiff=plot_fdiff, fdiff_ylims=fdiff_ylims,
                                                       spectra1_for_fdiff=spectra1_for_fdiff,
                                                       spectra2_for_fdiff=spectra2_for_fdiff,
                                                       dpi=dpi, show=show, plot_fname=fname)
        if save:
            self.infomsg(f'saved {fname}')
        if not show:
            return plt_output


    def get_sim_spectra_plot_fname(self, plot_fdiff=False, **kwargs):
        """Return the default file name for the plot comparing the HD
        simulation power spectra to the corresponding power spectra of
        the lower-resolution simulations or the theory power spectra.

        Parameters
        ----------
        plot_fdiff : bool, default=False
            Whether to add a lower panel to the plot for each component
            showing the fractional difference between the two sets of
            power spectra being compared.
        **kwargs : dict
            The additional keyword arguments are:
            - `components` (`list` of `str`) : A list of the simulation
                components (extragalactic foregrounds, lensing
                convergence, lensed or unlensed CMB) to plot. Each element
                of the list must be a valid `component` that can be passed
                to the `get_signal_sim_power` method. The default is the
                list returned by the `components_for_plot` method.
            - `freqs` (`list` of `int`) : A list of frequencies (GHz) for
                the plot. Each element in the list must be a valid
                numerical frequency that can be passed to the
                `get_signal_sim_power` method. The default is given by the
                `freqs` attribute.
            - `cmb_seed` (`int`) : The random seed used to generate the
                unlensed CMB realization. The default is given by the
                `cmb_seed` attribute. Only used if `'cmb'` or
                `'unlensed_cmb'` is in the list of `components`.
            - `pol` (`bool`) : Whether to plot the CMB polarization power
                spectra. The default is given by the `pol` attribute. Only
                used if `'cmb'` or `'unlensed_cmb'` is in the list of
                `components`.
            - `lmax` (`int`): The maximum multipole used to calculate the
                power spectra of the simulations. The default is given by
                the `lmax` attribute.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_component_name`
                method.

        Returns
        -------
        fname : str
            The file name for the plot.

        Notes
        -----
        By default, the plot will be saved as a PDF file in the directory
        returned by the `plot_dir` method.
        """
        components = simutils.validate_sim_component_names(self.components_for_plot(**kwargs))
        use_default_components = set(components) == set(self.default_components_for_plot)
        fname_info = [f'hd_vs_{self.lowres_name.lower()}_spectra']
        if use_default_components:
            # add any info about non-default kwargs:
            for component in components:
                component_name = self.get_component_name(component, **kwargs)
                default_component_name = self.get_component_name(component)
                if component_name != default_component_name:
                    fname_info.append(component_name)
        else:
            # add any info about components in the plot:
            component_info = self._map_component_list2str(**kwargs)
            fname_info.append(component_info)
        # if not plotting all freqs, add list of freqs to filename:
        if simutils.has_freq_dependent_component(components):
            freqs = simutils.validate_sim_freqs(self.get_kwarg('freqs', **kwargs))
            if set(freqs) != set(si.freqs):
                if len(freqs) > 1:
                    freq_info = '_'.join([f'{freq:03d}' for freq in freqs])
                    fname_info.append(f'freqs_{freq_info}_GHz')
                else:
                    fname_info.append(f'freq{freqs[0]:03d}GHz')
        if plot_fdiff:
            fname_info.append('withdiff')
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.plots_dir(), f'{fname_root}.pdf')
        return fname



    def _get_spectra_for_comparison_plot(self, save_intermediate_maps=False, **kwargs):
        """Return the power spectra of the simulations, and the power
        spectra of their lower-resolution counterparts (for tSZ, kSZ, CIB,
        radio, lensing convergence) or the corresponding theory power
        spectra (lensed or unlensed CMB).

        Parameters
        ----------
        save_intermediate_maps : bool, default=False
            Whether to save the intermediate, lower-resolution maps. It is
            recommended to pass `save_intermediate_maps=True` to avoid
            needing to obtain the intermediate, lower-resolution maps
            multiple times.

        Returns
        -------
        hd_spectra : dict of dict of dict of array_like of float
            A (very) nested dictionary of the binned power spectra of the
            ultrahigh-resolution simulations.
            - The first set of keys is given be the list of `components`.
            - The second set of keys is either given by the list of `freqs`
              for frequency-dependent components (`'tsz'`, `'cib'`,
              `'radio'`), or a single key of `None` for
              frequency-independent components (`'ksz'`, `'kappa'`, and
              `'cmb'` or `'unlensed_cmb'`).
            - The inner-most dictionary contains the multipoles (key
              `'ells'`) and power spectra for each component and
              frequency.
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
        spectra_for_comparison : dict of dict of dict of array_like of float
            A nested dictionary with the same structure as `hd_spectra`.
            For `'tsz'`, `'ksz'`, `'cib'`, `'radio'`, and `'kappa'`, the
            power spectra of the corresponding lower-resolution maps on
            the same patch of the sky are used. For `'cmb'` or
            `'unlensed_cmb'`, the corresponding binned CMB theory power
            spectra are used.

        Other Parameters
        ----------------
        **kwargs : dict
            The additional keyword arguments for the power spectra to plot
            are:
            - `components` (`list` of `str`) : A list of the simulation
                components (extragalactic foregrounds, lensing
                convergence, lensed or unlensed CMB) to plot. Each element
                of the list must be a valid `component` that can be passed
                to the `get_signal_sim_power` method. The default is the
                list returned by the `components_for_plot` method.
            - `freqs` (`list` of `int`) : A list of frequencies (GHz) for
                the plot. Each element in the list must be a valid
                numerical frequency that can be passed to the
                `get_signal_sim_power` method. The default is given by the
                `freqs` attribute.
            - `cmb_seed` (`int`) : The random seed used to generate the
                unlensed CMB realization. The default is given by the
                `cmb_seed` attribute. Only used if `'cmb'` or
                `'unlensed_cmb'` is in the list of `components`.
            - `pol` (`bool`) : Whether to plot the CMB polarization power
                spectra. The default is given by the `pol` attribute. Only
                used if `'cmb'` or `'unlensed_cmb'` is in the list of
                `components`.
            - `lmax` (`int`): The maximum multipole used to calculate the
                power spectra of the simulations. The default is given by
                the `lmax` attribute.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_intermediate_sim_power`
                method.
        """
        plt_components = self.components_for_plot(**kwargs)
        freqs = simutils.validate_sim_freqs(self.get_kwarg('freqs', **kwargs))
        # spectra of the HD sims:
        hd_spectra = {}
        # spectra of the lower-res sims (on patch of sky), or CMB theory:
        spectra_for_comparison = {}
        lowres_lmax = {'tsz': 8000, 'cib': 8000, 'radio': 8000, 'ksz': 7700, 'kappa': 3700}
        for component in plt_components:
            hd_spectra[component] = {}
            spectra_for_comparison[component] = {}
            sim_freqs = freqs if simutils.has_freq_dependent_component(component) else [None]
            bin_dl = False if (component in ['kappa', 'cmb']) else True
            spec_type = 'dl' if bin_dl else 'cl'
            for freq in sim_freqs:
                sim_spectra = self.get_signal_sim_power(component, freq=freq, bin_dl=bin_dl, **kwargs)
                hd_spectra[component][freq] = {}
                for key in sim_spectra.keys():
                    # remove `'cl'` or `'dl'` from the key:
                    spec_key = 'ells' if (key == 'ells') else key[2:]
                    hd_spectra[component][freq][spec_key] = sim_spectra[key].copy()
                if component in self.lowres_sim_components:
                    # get power of the intermediate, lower-resolution sim:
                    key = 'kk' if (component == 'kappa') else 'tt'
                    lowres_ells, lowres_spectra = self.get_intermediate_sim_power(component, freq=freq, bin_dl=bin_dl,
                                                                                  save=save_intermediate_maps,
                                                                                  save_sims=save_intermediate_maps,
                                                                                  **kwargs)
                    lowres_ells, lowres_spectra = utils.trim_spectrum_ell_range(lowres_ells, lowres_spectra,
                                                                                lmax=lowres_lmax[component])
                    spectra_for_comparison[component][freq] = {'ells': lowres_ells, key: lowres_spectra.copy()}
                else: # get the lensed cmb theory:
                    theo = self.get_sim_theory(component, binned=True, dl=bin_dl)
                    spectra_for_comparison[component][freq] = {}
                    for key in theo.keys():
                        spec_key = 'ells' if (key == 'ells') else key[2:]
                        spectra_for_comparison[component][freq][spec_key] = theo[key].copy()
        return hd_spectra, spectra_for_comparison


    def _get_radio_spectra_for_comparison_plot(self, max_flux=100, freq_for_flux_cut=90, **kwargs):
        """Remove very bright radio sources near the edges of the
        simulations and their lower-resolution counterparts, and return
        their power spectra.

        A source 'near the edges' means is located within a distance given
        by the `apod_width` attribute from each edge of the maps. The
        `max_flux` and `freq_for_flux_cut` define which sources are
        considered 'bright'.

        Parameters
        ----------
        max_flux : int or float, default=100
            The maximum radio flux in mJy (at the frequency
            `freq_for_flux_cut`) allowed to remain in the maps near the
            edges; i.e., sources near the edges with flux above `max_flux`
            at `freq_for_flux_cut` are removed.
        freq_for_flux_cut : int, default=90
            The frequency (in GHz) at which the `max_flux` is measured.

        Returns
        -------
        hd_spectra4fdiff : dict of dict of array_like of float, or None
            If there are no radio sources that need to be removed, `None`
            is returned. Otherwise, a nested dictionary of the (binned)
            power spectra of the radio simulations (after removing bright
            sources near the edges) is returned. The first set of keys is
            given by the `freqs` list. The inner-most dictionary for each
            frequency has a key `'ells'` for the multipoles and a key
            `'tt'` for the power spectrum, in units of uK^2 and multiplied
            by `ell * (ell + 1) / (2 * pi)`.
        lowres_spectra4fdiff : dict of dict of array_like of float, or None
            Same as `hd_spectra4fdiff`, but for the corresponding
            lower-resolution simulations.

        Other Parameters
        ----------------
        **kwargs : dict
            The additional keyword arguments are:
            - `freqs` (`list` of `int`): A list of map frequencies (GHz).
                Each frequency in the list must be one of `30`, `90`,
                `148`, `219`, `277`, or `350`.
            - Any other keyword arguments for the set of lower-resolution
                simulations passed to the `get_catalog`, `get_signal_sim`,
                and `get_intermediate_sim` methods.

        See Also
        --------
        plot_sim_spectra_comparison :
            The plot where these spectra are used.

        Notes
        -----
        See `plot_sim_spectra_comparison` for more details.
        """
        component = 'radio'
        freqs = simutils.validate_sim_freqs(self.get_kwarg('freqs', **kwargs))

        # check if the spectra are already saved:
        spectra_dir = os.path.join(self.plots_dir(), 'files_for_radio_spectra_diff_plot')
        hd_fnames = {}
        lowres_fnames = {}
        for freq in freqs:
            hd_fname = self.get_signal_sim_power_fname(component, freq=freq, bin_dl=True, **kwargs)
            hd_fname = hd_fname.replace(self.spectra_dir(), spectra_dir)
            hd_fnames[freq] = hd_fname.replace('dls', 'hd_dls')
            lowres_fnames[freq] = hd_fnames[freq].replace('hd_dls.txt', f'{self.lowres_name.lower()}_dls.txt')
        hd_saved = all([os.path.exists(fname) for fname in hd_fnames.values()])
        lowres_saved = all([os.path.exists(fname) for fname in lowres_fnames.values()])

        # get catalog of bright sources that fall into the apodized region:
        cat = self.get_catalog(component, include_padded_area=False, **kwargs)
        cat = cat.sort_values(f'fluxmJy_{freq_for_flux_cut}GHz', ascending=False)
        cat['idx'] = list(range(len(cat)))
        # find sources that are not apodized and remove them from catalog
        unapod_srcs = fgcatalogs.trim_catalog_positions(cat.copy(), ra_ctr=self.ra_ctr, dec_ctr=self.dec_ctr,
                                                        width=self.width-2*self.apod_width,
                                                        height=self.height-2*self.apod_width)
        apod_srcs = cat.copy()
        apod_srcs = apod_srcs[~apod_srcs['idx'].isin(unapod_srcs['idx'].values)]
        # keep only the brightest sources in the catalog
        bright_apod_srcs = apod_srcs[apod_srcs[f'fluxmJy_{freq_for_flux_cut}GHz'].gt(max_flux)].copy()

        if len(bright_apod_srcs) > 0:
            if not (hd_saved and lowres_saved):
                self.infomsg(f"removing radio sources within the apodized region with {freq_for_flux_cut} GHz"
                             f" flux above {round(max_flux,2)} mJy from both the HD and the corresponding"
                             " lower-resolution sims, and taking their power")
            spectra_dir = utils.mkdir(spectra_dir) # make the directory
            hd_spectra = {}
            lowres_spectra = {}

            # make maps of only the sources to remove from both HD and
            # lower-resolution maps, and then subtract them from the
            # corresponding sims:
            if not hd_saved:
                srcs_to_remove = maps.make_src_maps(self.padded2x_shape, self.padded2x_wcs, bright_apod_srcs, freqs)
            if not lowres_saved:
                # load in a lower-res radio sim to get its shape and wcs
                lowres_radio_sim = self.get_intermediate_sim(component, freq=self.freqs[0], **kwargs)
                srcs_to_remove_lowres = maps.make_src_maps(lowres_radio_sim.shape, lowres_radio_sim.wcs,
                                                           bright_apod_srcs, freqs)
            for freq in freqs:
                hd_spectra[freq] = {}
                if os.path.exists(hd_fnames[freq]):
                    hd_spectra[freq]['ells'], hd_spectra[freq]['tt'] = np.loadtxt(hd_fnames[freq], unpack=True)
                else:
                    sim = self.get_signal_sim(component, freq=freq, **kwargs)
                    # make the map to subtract in the same way that the
                    # sim was generated: put sources on the same initial,
                    # larger patch of sky that was used to generate the
                    # sims, apodize, convolve pixwin, and cut out the
                    # inner un-apodized region:
                    srcs_to_remove[freq] = self.convolve_pixwin(srcs_to_remove[freq], save_apod_window=True)
                    srcs_to_remove[freq] = enmap.project(srcs_to_remove[freq], sim.shape, sim.wcs)
                    sim -= srcs_to_remove[freq] # subtract the srcs
                    sim = enmap.project(sim, self.shape, self.wcs)
                    # take its power:
                    sim_spectra = self.take_power(sim, freq=freq, bin_dl=True, bin_cl=False)
                    sim_power = {'ells': sim_spectra['ells'], 'tt': sim_spectra['dltt'].copy()}
                    utils.save_dict_to_file(hd_fnames[freq], sim_power, keys=simutils.get_spectra_keys(component))
                    hd_spectra[freq] = sim_power.copy()

                lowres_spectra[freq] = {}
                if os.path.exists(lowres_fnames[freq]):
                    lowres_spectra[freq]['ells'], lowres_spectra[freq]['tt'] = np.loadtxt(lowres_fnames[freq], unpack=True)
                else:
                    # the intermediate maps are saved on the initial,
                    # larger patch of sky and were never convolved with a
                    # pixel window, so we only need to place the sources
                    # on the same patch of sky at the lower resolution and
                    # subtract it from the lower-resolution simulation:
                    sim = self.get_intermediate_sim(component, freq=freq, **kwargs) - srcs_to_remove_lowres[freq]
                    # take its power:
                    sim_power = {}
                    sim_power['ells'], sim_power['tt'] = self.take_intermediate_map_power(sim, bin_dl=True)
                    utils.save_dict_to_file(lowres_fnames[freq], sim_power, keys=simutils.get_spectra_keys(component))
                    lowres_spectra[freq] = sim_power.copy()
                lowres_spectra[freq]['ells'], lowres_spectra[freq]['tt'] = utils.trim_spectrum_ell_range(lowres_spectra[freq]['ells'],
                                                                                                         lowres_spectra[freq]['tt'],
                                                                                                         lmax=8000)

            hd_spectra4fdiff = {component: hd_spectra}
            lowres_spectra4fdiff = {component: lowres_spectra}

        else:
            hd_spectra4fdiff = None
            lowres_spectra4fdiff = None

        return hd_spectra4fdiff, lowres_spectra4fdiff


    def plot_smallscale_ksz_kappa_spectra(self, show=True, save=True, fname=None,
                                          save_intermediate_maps=False, plot_fdiff=False, dpi=500, **kwargs):
        """Plot the power spectra of the kSZ and lensing convergence
        simulations, and compare it with the power spectra of the
        corresponding lower-resolution maps on the same patch of sky and
        with the theory power spectra.

        Parameters
        ----------
        show : bool, default=True
            Whether to display the plot (by calling
            `matplotlib.pyplot.show()`).
        save : bool, default=True
            Whether to save the plot.
        fname : str or None, default=None
            The file name used to save the plot if `save=True`. The
            default file name is returned by the
            `get_ksz_kappa_spectra_plot_fname` method.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `matplotlib` figure used to make the plot. Only returned
            if `show=False`.
        axs : dict of matplotlib.axes.Axes
            A dictionary with a key (`str`) for each component (`'ksz'` or
            `'kappa'`) in the list of `components` to plot. The values are
            the corresponding subplots used to plot the power spectra of
            that component. Only returned if `show=False`.
        fdiff_axs : dict of matplotlib.axes.Axes
            A dictionary with a key (`str`) for each component (`'ksz'` or
            `'kappa'`) in the list of `components` to plot. The values are
            the corresponding subplots used to plot the fractional
            difference of the power spectra of each component. Only
            returned if `show=False` and `plot_fdiff=True`.

        Other Parameters
        ----------------
        save_intermediate_maps : bool, default=False
            Whether to save the intermediate, lower-resolution maps. It is
            recommended to pass `save_intermediate_maps=True` to avoid
            needing to obtain the intermediate, lower-resolution maps
            multiple times.
        dpi : int, default=500
            The resolution of the plot in dots-per-inch, passed to
            `matplotlib.pyplot.figure`.
        plot_fdiff : bool, default=False
            Whether to add a lower panel to the plot for each component
            showing the fractional difference between the two sets of
            power spectra being compared.
        **kwargs : dict
            The additional keyword arguments are:
            - `components` (`list` of `str`) : A list of the simulation
                components (kSZ and/or lensing convergence) to plot. The
                allowed elements of the list are `'ksz'` and `'kappa'`.
                The default list is determined by the `components`
                attribute: it includes `'ksz'` if `'ksz'` is in
                `components`, and includes `'kappa'` if either `'kappa'`
                or `'cmb'` are in `components`.
            - `lmax` (`int`): The maximum multipole used to calculate the
                power spectra of the simulations. The default is given by
                the `lmax` attribute.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_signal_sim_power`,
                `get_intermediate_sim_power`, and `get_sim_theory`
                methods.

        Notes
        -----
        This method is used to produce Figure 4 in arXiv:XXXX.XXXX (!! TODO !!).
        """
        components = simutils.validate_sim_component_names(self.get_kwarg('components', **kwargs))
        if any([component in components for component in ['ksz', 'kappa', 'cmb']]):
            self.infomsg(f"plotting comparison between small-scale HD kSZ and kappa sim spectra and the theory curves")
            # get the spectra
            spectra_kwargs = {**kwargs, 'save_intermediate_maps': save_intermediate_maps}
            hd_spectra, lowres_spectra, theo_spectra = self._get_spectra_for_ksz_kappa_plot(**spectra_kwargs)
            # make the plot
            if save:
                if fname is None:
                    fname_kwargs = {**kwargs, 'plot_fdiff': plot_fdiff}
                    fname = self.get_ksz_kappa_spectra_plot_fname(**fname_kwargs)
            else:
                fname = None
            plt_output = plots.plot_smallscale_ksz_kappa_spectra(hd_spectra, lowres_spectra, theo_spectra, show=show,
                                                                 plot_fname=fname, plot_fdiff=plot_fdiff, dpi=dpi)
            if save:
                self.infomsg(f'saved {fname}')
            if not show:
                return plt_output


    def get_ksz_kappa_spectra_plot_fname(self, plot_fdiff=False, **kwargs):
        """Return the default file name for the plot comparing the HD
        kSZ and lensing convergence simulation power spectra to the
        corresponding power spectra of the lower-resolution simulations
        or the theory power spectra.

        Parameters
        ----------
        plot_fdiff : bool, default=False
            Whether to add a lower panel to the plot for each component
            showing the fractional difference between the power spectra.
        **kwargs : dict
            The additional keyword arguments are:
            - `components` (`list` of `str`) : A list of the simulation
                components (kSZ and/or lensing convergence) to plot. The
                allowed elements of the list are `'ksz'` and `'kappa'`.
                The default list is determined by the `components`
                attribute: it includes `'ksz'` if `'ksz'` is in
                `components`, and includes `'kappa'` if either `'kappa'`
                or `'cmb'` are in `components`.

        Returns
        -------
        fname : str
            The file name for the plot.

        Notes
        -----
        By default, the plot will be saved as a PDF file in the directory
        returned by the `plot_dir` method.
        """
        components = simutils.validate_sim_component_names(self.get_kwarg('components', **kwargs))
        plt_components = []
        if 'ksz' in components:
            plt_components.append('ksz')
        if ('cmb' in components) or ('kappa' in components):
            plt_components.append('kappa')
        component_info = '_'.join(plt_components)
        fname_root = f'hd_{component_info}_vs_theory_spectra'
        if plot_fdiff:
            fname_root = f'{fname_root}_withdiff'
        fname = os.path.join(self.plots_dir(), f'{fname_root}.pdf')
        return fname


    def _get_spectra_for_ksz_kappa_plot(self, save_intermediate_maps=False, **kwargs):
        """Return the power spectra of the kSZ and lensing convergence
        simulations, and the power spectra of the corresponding
        lower-resolution maps on the same patch of sky and theory power
        spectra.

        Parameters
        ----------
        save_intermediate_maps : bool, default=False
            Whether to save the intermediate, lower-resolution maps. It is
            recommended to pass `save_intermediate_maps=True` to avoid
            needing to obtain the intermediate, lower-resolution maps
            multiple times.

        Returns
        -------
        hd_spectra : dict of dict of array_like of float
            A nested dictionary of the power spectra of the kSZ (if
            `'ksz'` is in the list of `components`) and lensing
            convergence (if `'kappa'` or `'cmb'` is in `components`)
            simulations. It has the following keys and values:
            - `hd_spectra['ksz']` is a dictionary with a key `'tt'` for
                the binned kSZ simulation power spectrum in units of uK^2
                and multiplied by `ell * (ell + 1) / (2 * pi)`, and a key
                `'ells'` for the bin centers.
            - `hd_spectra['kappa']` is a dictionary with a key `'kk'` for
                the binned lensing convergence simulation power spectrum,
                and a key `'ells'` for the bin centers.
        lowres_spectra : dict of dict of array_like of float
            Same as `hd_spectra`,  but for the power spectra of the
            corresponding lower-resolution simulations on the same patch
            of the sky.
        theo_spectra : dict of dict of array_like of float
            Same as `hd_spectra`,  but for the corresponding binned theory
            power spectra.

        Other Parameters
        ----------------
        **kwargs : dict
            The additional keyword arguments are:
            - `components` (`list` of `str`) : A list of the simulation
                components (kSZ and/or lensing convergence) to plot. The
                allowed elements of the list are `'ksz'` and `'kappa'`.
                The default list is determined by the `components`
                attribute: it includes `'ksz'` if `'ksz'` is in
                `components`, and includes `'kappa'` if either `'kappa'`
                or `'cmb'` are in `components`.
            - `lmax` (`int`): The maximum multipole used to calculate the
                power spectra of the simulations. The default is given by
                the `lmax` attribute.
            - Any other keyword arguments for the set of lower-resolution
                simulations, passed to the `get_signal_sim_power`,
                `get_intermediate_sim_power`, and `get_sim_theory`
                methods.
        """
        components = simutils.validate_sim_component_names(self.get_kwarg('components', **kwargs))
        plt_components = []
        if 'ksz' in components:
            plt_components.append('ksz')
        if ('cmb' in components) or ('kappa' in components):
            plt_components.append('kappa')
        hd_spectra = {}
        lowres_spectra = {}
        theo_spectra = {}
        for component in plt_components:
            bin_dl = False if (component == 'kappa') else True
            spec_type = 'dl' if bin_dl else 'cl'
            key = 'kk' if (component == 'kappa') else 'tt'
            # hd sim:
            hd_power = self.get_signal_sim_power(component, bin_dl=bin_dl, **kwargs)
            hd_spectra[component] = {'ells': hd_power['ells'].copy(), key: hd_power[f'{spec_type}{key}'].copy()}

            # lower-resolution sim:
            lowres_ells, lowres_power = self.get_intermediate_sim_power(component, bin_dl=bin_dl, save=True,
                                                                        save_sim=save_intermediate_maps, **kwargs)
            lowres_spectra[component] = {'ells': lowres_ells, key: lowres_power.copy()}
            # theory:
            theo = self.get_sim_theory(component, binned=True, dl=bin_dl)
            theo_spectra[component] = {}
            for theo_key in theo.keys(): # rename keys
                spec_key = 'ells' if (theo_key == 'ells') else theo_key[2:]
                theo_spectra[component][spec_key] = theo[theo_key].copy()
        return hd_spectra, lowres_spectra, theo_spectra




