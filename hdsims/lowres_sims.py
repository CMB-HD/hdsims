"""General class for the lower-resolution, full-sky simulations."""

from . import siminfo as si, simutils


class LowResSims(simutils.Sims):
    """Cut out maps and catalogs for a patch of sky from a set of 
    lower-resolution full-sky simulations.
    
    The intermediate, lower-resolution simulations of diffuse components
    (tSZ, kSZ, and lensing convergence maps) are projected from the full
    sky to CAR maps on a patch of the sky. The catalogs of CIB and radio
    galaxies within the patch of sky can be used to generate a set of
    lower-resolution CAR maps of the CIB and radio sources.
    
    This class is intended to be a base/parent class, used to create 
    new classes for a specific set of simulations. See the 'Notes' section
    below for instructions on how to use a non-default set of 
    lower-resolution, full-sky simulations.
    
    Attributes
    ----------
    `lowres_sims_dir` : str
    `freqs` : list of int
    `lowres_sim_components` : list of str
    `lowres_name` : str
    
    Notes
    -----
    Attributes listed above without a description will have the same value
    as the corresponding parameter passed when initializing the class, or
    the default value if it is not passed. 
    
    The `freqs` attribute will be added to the dictionary of 
    `default_kwargs` (inherited from `hdsims.simutils.Sims`). Otherwise, 
    see the `hdsims.simutils.Sims` class for more information about the 
    inherited attributes and methods.
    """
    
    def __init__(self, hd_sims_dir, lowres_sims_dir,
                 freqs=si.freqs, lowres_sim_components=si.s10_sim_components, lowres_name='lowres',
                 ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr, width=si.width, height=si.height,
                 apod_width=si.apod_width, map_apod_width=None, lowres_apod_width=None,
                 res=si.hd_res, verbose=False, log=None, make_output_dirs=True):
        """Initialization for the intermediate, lower-resolution
        simulations on a patch of sky.

        Parameters
        ----------
        hd_sims_dir : str
            The path to the directory where all of the output files for
            this patch of sky will be saved. This directory will be
            created if it does not already exist.
        lowres_sims_dir : str
            The path to the directory where the lower-resolution, full-sky
            simulations and catalogs have been saved.
        lowres_name : str, default='lowres'
            A short name to label the set of full-sky, lower-resolution
            simulations in file names. Cannot contain any special
            characters.
        freqs : list of int, default=[30, 90, 148, 219, 277, 350]
            A list of available frequencies (in GHz) for the simulations
            and catalogs.
        lowres_sim_components : list of str, optional
            A list of available lower-resolution map or catalog
            components. The default list includes `'ksz'`, `'tsz'`,
            `'kappa'` for the kSZ, tSZ and lensing convergence maps,
            respectively, and `'cib'`, `'radio'` for the CIB and radio
            catalogs, respectively.
        ra_ctr, dec_ctr : int or float, optional
            The right ascension (R.A.) and declination (dec.), in degrees,
            of the center of the patch of sky. The defaults are `ra_ctr=6`
            and `dec_ctr=6`.
        width : int or float, default=10
            The width (in degrees) of the region of the final,
            ultrahigh-resolution map to be used for analysis, e.g. when
            taking its power spectrum.
        height : int or float, optional
            The height (in degrees) of the region of the final,
            ultrahigh-resolution map to be used for analysis. By default,
            the `height` is assumed to be equal to the `width`.
        apod_width : int or float, default=0.5
            The width (in degrees) of the region along each edge of the
            map that will be apodized before calculating its power
            spectrum.

        Other Parameters
        ----------------
        map_apod_width : int or float, optional
            The width (in degrees) of the region along each edge of the
            map that will be apodized before taking any Fourier or
            spherical harmonic transforms. By default, the `apod_width`
            will be used.
        lowres_apod_width : int or float, optional
            The width (in degrees) of the region along each edge of the
            initial, lower-resolution maps that will be apodized before
            taking any Fourier or spherical harmonic transforms. By default,
            the `map_apod_width` will be used.
        res : int or float, default=0.04
            The resolution of the ultrahigh-resolution maps, in arcminutes.
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
        """
        super().__init__(hd_sims_dir, ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height,
                         apod_width=apod_width, map_apod_width=map_apod_width, lowres_apod_width=lowres_apod_width,
                         res=res, verbose=verbose, log=log, make_output_dirs=make_output_dirs)
        self.lowres_sims_dir = lowres_sims_dir
        self.lowres_name = lowres_name
        self.lowres_sim_components = lowres_sim_components
        self.freqs = freqs
        self.default_kwargs['freqs'] = self.freqs
        
    
    # the following methods must be defined by a derived class:
    
    def get_sim_catalog(self, component, **kwargs):
        """Return the catalog of objects in the simulations for a patch of
        sky, cut out from a full-sky catalog.
        
        This catalog contains all objects (e.g. CIB or radio galaxies, SZ
        clusters) within the region defined by the `padded2x_width` and
        `padded2x_height` attributes inherited from the 
        `hdsims.simutils.Sims` class.
        
        Parameters
        ----------
        component : str
            The name of the map component, e.g. `'cib'` for the CIB.
        **kwargs : dict
            Any additional keyword arguments.
            
        Returns
        -------
        pandas.DataFrame
            A catalog of objects in the simulation of the `component` on
            the full patch of sky defined by the `padded2x_width` and
            `padded2x_height` attributes. 
            
            The catalog has columns named `'RADeg'` and `'decDeg'` for the
            R.A. and dec. coordinates (in degrees), respectively, of each
            object. For CIB and radio galaxies, the columns for the flux
            (in mJy) of each source at each `freq` (in GHz) in the list of
            `freqs` are named `'fluxmJy_{freq}GHz'`; e.g., the column name
            is `'fluxmJy_90GHz'` for 90 GHz.
        """
        raise NotImplementedError
        
        
    def get_intermediate_sim(self, component, freq=None, save=False, **kwargs):
        """Return the lower-resolution simulation for single map component
        and frequency on a patch of the sky.
        
        The simulation will have the same resolution as the full-sky maps,
        and an area defined by the `padded2x_width` and `padded2x_height`
        attributes inherited from the `hdsims.simutils.Sims` class. Maps 
        of diffuse components (e.g. the tSZ, kSZ, or lensing convergence)
        are cut out from the full-sky maps, and maps of discrete
        components (e.g., CIB or radio galaxies) are generated from their
        corresponding catalog.
        
        Parameters
        ----------
        component : str
            The name of the map component, e.g. `'cib'` for the CIB.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components.
        save : bool, default=False
            Whether to save the map.
        **kwargs : dict
            Any additional keyword arguments.
        
        Returns
        -------
        pixell.enmap.ndmap
            The lower-resolution map of the given component and frequency
            on the patch of sky.
            
        See Also
        --------
        get_sim_catalog : 
            The catalog used to generate maps of discrete components.
        """
        raise NotImplementedError
        
        
    def get_intermediate_sim_power(self, component, freq=None, bin_dl=False, save=False, save_sim=False, **kwargs):
        """Return the power spectrum of the lower-resolution simulation of
        a single map component and frequency on a patch of the sky.
        
        The maps are trimmed to the region defined by the `width` and
        `height` attributes (inherited from `hdsims.simutils.Sims`) and
        apodized before calculating their power spectra. An inverse 
        mode-coupling matrix is applied to correct for the apodization and
        bin the spectra.
        
        Parameters
        ----------
        component : str
            The name of the map component, e.g. `'cib'` for the CIB.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components.
        bin_dl : bool, default=False
            If `bin_dl=True`, the power spectra are multiplied by a factor
            of `ell * (ell + 1) / (2 * pi)` at each multipole `ell`. 
        save : bool, default=False
            Whether to save the power spectra.
        save_sim : bool, default=False
            Whether to save the simulation.
        **kwargs : dict
            Any additional keyword arguments.
            
        Returns
        -------
        ells, power_spectrum : array_like of float
            One-dimensional arrays holding the binned multipoles and the
            binned power spectrum of the simulation, respectively.
        """
        raise NotImplementedError
    
        
    def apodize_intermediate_map(self, imap):
        """Apodize an intermediate, lower-resolution map.

        If the map has the same area as the area defined by the `width`
        and `height` attributes, it will be apodized over a region with a
        width given by the `apod_width` attribute along each edge.
        Otherwise, the `lowres_apod_width` attribute will be used instead.

        Parameters
        ----------
        imap : pixell.enmap.ndmap
            The input map. It must be the same resolution as the
            lower-resolution simulations, and it must contain the region
            defined by the `width` and `height` attributes.

        Returns
        -------
        pixell.enmap.ndmap
            The apodized map.
        """
        raise NotImplementedError
        
        
    def take_intermediate_map_power(self, imap, bin_dl=False):
        """Return the power spectrum of a map at the resolution of the
        lower-resolution simulations on a patch of the sky.
        
        The map will be trimmed to the region defined by the `width` and
        `height` attributes (inherited from `hdsims.simutils.Sims`) and 
        apodized before calculating its power spectrum. An inverse 
        mode-coupling matrix is applied to correct for the apodization and
        bin the spectrum.
        
        Parameters
        ----------
        imap : pixell.enmap.ndmap
            The input map. It must be the same resolution as the 
            lower-resolution simulations, and it must contain the region
            defined by the `width` and `height` attributes.
        bin_dl : bool, default=False
            If `bin_dl=True`, the power spectra are multiplied by a factor
            of `ell * (ell + 1) / (2 * pi)` at each multipole `ell`. 
            
        Returns
        -------
        ells, power_spectrum : array_like of float
            One-dimensional arrays holding the binned multipoles and the
            binned power spectrum of the map, respectively.
        """
        raise NotImplementedError
        
        
    # the following methods that may optionally be defined by a derived
    # class; otherwise, the default behavior defined here is used:
        
    def get_camb_cosmo_params(self):
        """Return a dictionary of cosmological parameter names and values
        that can be passed to CAMB.
        
        The dictionary must contain a set of cosmological parameters that
        can be passed to the `camb.set_params` function to initialize a
        `camb.model.CAMBparams` instance. Alternatively, if no dictionary
        is defined, a default cosmology will be used with CAMB.
        
        Returns
        -------
        dict or None
            Either a dictionary of parameter names and values recognized by
            CAMB, or `None` to use a default set of CAMB parameters.
        
        See Also
        --------
        camb.set_params
        hdsims.siminfo.hd_camb_accuracy_params : 
            A dict of CAMB accuracy parameters used for CMB-HD.
        hdsims.simutils.get_cambparams_for_sim :
            An instance of `camb.model.CAMBparams` using the default set
            of CAMB parameters.
        
        Notes
        -----
        If no dictionary of parameters is provided, the default will be
        used. The default CAMB parameters are based on the cosmology used
        in Sehgal et. al. 2010 (arXiv:0908.0540), with the CMB-HD accuracy
        settings in the `siminfo.hd_camb_accuracy_params` dictionary.
        
        If the CAMB parameters are used to generate ultrahigh-resolution
        simulations (e.g. in the `hdsims.hdsimsgen.HDSimsGen` class), then 
        the CMB-HD CAMB accuracy settings will be added to the dictionary. 
        The CAMB unlensed CMB and lensing convergence theory power spectra 
        will be used when generating the ultrahigh-resolution (lensed or 
        unlensed) CMB and lensing convergence maps.
        """
        return None # default
    
    
    def load_cl_ksz_template(self):
        """Return a template theory kSZ power spectrum.
        
        The power spectrum `C_ell` is in units of uK^2 without any
        multiplicative factors, i.e., not multiplied by a factor of 
        `ell * (ell + 1) / (2*pi)` at each multipole `ell`. It is defined
        at each multipole starting from `ell=0`, and normalized such that
        `ell * (ell + 1) * C_ell / (2*pi) = 1` at `ell = 3000`.
        
        Returns
        -------
        template_ells, template_cls : array_like of float
            The multipoles and power spectrum, respectively.
        
        See Also
        --------
        hdsims.simutils.load_cl_ksz_template : The default template.
        """
        return simutils.load_cl_ksz_template() # default
    
    
    def get_lowres_sim_component_name(self, component, **kwargs):
        """Return a name used as a label for each map component in file
        names.
        
        Parameters
        ----------
        component : str
            The name of the map component, e.g. `'cib'` for the CIB.
        **kwargs : dict
            Any additional keyword arguments.
            
        Returns
        -------
        str
            A short name for the map `component` that will be used in file
            names.
        """
        return component # default
    

