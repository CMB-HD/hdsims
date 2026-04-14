"""Contains functions and the `S10Sims` class for the set of full-sky
Sehgal et. al. 2010 ('S10', arXiv:0908.0540) simulations.

"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import healpy as hp
from pixell import enmap
from . import utils, siminfo as si, maps, fgcatalogs, simpower, simutils, lowres_sims


def get_fullsky_s10_sim_fname(component, freq=None, s10_sims_dir=None, pixwin_deconvolved_uK=False):
    """Return the path to the full-sky S10 HEALPix map of the given
    component and frequency.

    Parameters
    ----------
    component : str
        The map component. Must be one of `'ksz'`, `'tsz'`, or `'kappa'`
        for the kSZ, tSZ and lensing convergence maps, respectively.
    freq : int or None, default=None
        The map frequency (in GHz). If provided, it must be one of `30`,
        `90`, `148`, `219`, `277`, `350`. Only required if
        `component='tsz'`.
    s10_sims_dir : str or None, default=None
        The path to the directory where the full-sky maps are saved.
        If `s10_sims_dir` is passed, the absolute path is returned;
        otherwise, a relative path is returned.
    pixwin_deconvolved_uK : bool, default=False
        Only used for `component='tsz'` or `component='ksz'`.
        If `pixwin_deconvolved_uK=True`, returns the path to the
        pixel-window-deconvolved map in units of uK; in this case, the tSZ
        map has also been multiplied by 0.75. Otherwise, returns the path
        to the original, unmodified S10 map.

    Returns
    -------
    str
        The path to the map.
    """
    component = simutils.validate_sim_component_name(component, valid_components=['tsz', 'ksz', 'kappa'])
    if 'sz' in component:
        if (component == 'ksz') and (freq is None):
            freq = si.s10_ksz_map_freq
        else:
            freq = simutils.validate_sim_freq(freq)
        fname_info = [f'{int(freq):03d}', f'{component.lower()}']
        if pixwin_deconvolved_uK:
            fname_info.append('nopixwin_uK')
        fname_root = '_'.join(fname_info)
        fname = f'{fname_root}_healpix.fits'
    else: # kappa
        fname = 'healpix_4096_KappaeffLSStoCMBfullsky.fits'
    if s10_sims_dir is not None:
        fname = os.path.join(s10_sims_dir, fname)
    return fname


def get_fullsky_s10_catalog_fnames(component, s10_sims_dir=None):
    """Return the path(s) to the full-sky S10 catalog file(s).

    Parameters
    ----------
    component : str
        The name of the map component. The allowed values are: `'radio'`
        for radio galaxies, `'cib'` for the CIB, or one of `'sz'`,
        `'tsz'`, or `'ksz'` for SZ clusters.
    s10_sims_dir : str or None, default=None
        The path to the directory where the full-sky maps are saved.
        If `s10_sims_dir` is passed, the absolute path is returned;
        otherwise, a relative path is returned.

    Returns
    -------
    str or list of str
        The path to the catalog, or for `component='cib'`, or a list of
        paths to each catalog file.

    Notes
    -----
    The full-sky S10 CIB catalog is broken up into multiple files; the
    radio and SZ cluster catalogs are saved in a single file.
    """
    if 'sz' in component.lower():
        component = 'sz'
    else:
        component = simutils.validate_sim_component_name(component, valid_components=['cib', 'radio', 'sz'])
    if component == 'cib':
        fnames = ["IRgal_S_1.dat", "IRgal_S_2.dat", "IRgal_S_3.dat", "IRgal_S_4.dat",
                  "IRgal_S_5.dat", "IRgal_S_6.dat", "IRgal_S_7.dat", "IRgal_S_8.dat",
                  "IRgal_S_9.dat", "IRgal_S_10.dat", "IRBlastPop.dat"]
        if s10_sims_dir is not None:
            fnames = [os.path.join(s10_sims_dir, fname) for fname in fnames]
        return fnames
    else:
        fname = 'radio.cat' if (component.lower() == 'radio') else 'halo_sz.ascii'
        if s10_sims_dir is not None:
            fname = os.path.join(s10_sims_dir, fname)
        return fname


# the following functions are used to translate the coordinates in the
# S10 catalogs (which contain coordinates for R.A. between 0 and 
# 90 degrees, dec. between 0 and 90 degrees) to the full sky:

def get_coords_in_s10_octant(ra, dec):
    """Given a pair of right ascension and declination coordinates, return
    the corresponding pair translated into the original S10 octant (R.A.
    between 0 and 90 degrees, dec. between 0 and 90 degrees).

    See arXiv:0908.0540 for more details.

    Parameters
    ----------
    ra, dec : float
        The right ascension and declination, in degrees.

    Returns
    -------
    ra0, dec0 : float
        The corresponding pair of coordinates in the original S10 octant.
    """
    # make sure -90 <= dec <= 90 degrees:
    if (dec < -90) or (dec > 90):
        raise ValueError(f"`{dec = }`. The declination `dec` should be in units of degrees,"
                         " and must have a value between -90 and 90.")
    # make sure 0 <= ra < 360
    orig_ra = ra
    if ra < 0:
        ra += 360
    if (ra < 0) or (ra > 360):
        raise ValueError(f"`ra = {orig_ra}`. The right ascension `ra` should be in units of degrees, "
                         "and must have a value between 0 and 360.")
    # get the corresponding spherical coordinates:
    theta, phi = utils.dec_ra_to_theta_phi(dec, ra)
    # determine which octant the RA is in:
    if ra <= 90:
        n = 0
    elif ra < 180:
        n = 1
    elif ra < 270:
        n = 2
    elif ra < 360:
        n = 3
    # determine which hemisphere the dec is in & calculate the corresponding
    #  spherical coordinates in the original S10 octant:
    if dec >= 0: # northern hemisphere
        theta0 = theta
        phi0 = phi - n * np.pi / 2
    else: # southern hemisphere
        theta0 = np.pi - theta
        phi0 = (n+1) * np.pi/2 - phi
    # return the corresponding RA & dec in the original S10 octant:
    dec0, ra0 = utils.theta_phi_to_dec_ra(theta0, phi0)
    return ra0, dec0


def get_s10_octant(ra, dec):
    """Determine which octant of the sky contains the pair of right 
    ascension and declination coordinates (in degrees).
    """
    if ra < 0: # make sure 0 <= ra < 360
        ra += 360
    hemisphere = 'N' if (dec >= 0) else 'S'
    if ra <= 90:
        n = 0
    elif ra < 180:
        n = 1
    elif ra < 270:
        n = 2
    elif ra < 360:
        n = 3
    return hemisphere, n


def get_s10_octants_in_patch(ra_ctr, dec_ctr, width, height):
    """Determine which octants are contained in a given patch on the sky.
    """
    # look at center and at corners
    octants_in_patch = [get_s10_octant(ra_ctr, dec_ctr)]
    ra_min, ra_max, dec_min, dec_max = maps.get_patch_corner_coords(ra_ctr, dec_ctr, width, height=height)
    # loop through pairs of coordinates in the patch & add the octant of those coords
    for ra in np.linspace(ra_min, ra_max, 6):
        for dec in np.linspace(dec_min, dec_max, 4):
            octants_in_patch.append(get_s10_octant(ra, dec))
    return list(set(octants_in_patch))


def get_theta_phi_coords_in_octant(theta0, phi0, north=True, n=0):
    """Given a pair (`theta0`, `phi0`) of spherical coordinates (in
    radians) within the original S10 octant, return the pair of
    coordinates mirrored into the given octant.
    """
    if n not in range(4):
        raise ValueError(f"`{n = }`. The value of `n` must be 0, 1, 2, or 3.")
    if north:
        theta = theta0
        phi = phi0 + (n * np.pi / 2)
    else:
        theta = np.pi - theta0
        phi = (np.pi / 2) - phi0 + (n * np.pi / 2)
    return theta, phi


def get_mirrored_theta_phi_coords(theta0, phi0, octants=None):
    """Given a pair (`theta0`, `phi0`) of spherical coordinates (in
    radians) within the original S10 octant, return all eight pairs of
    coordinates mirrored onto the full sky, or mirrored into the given
    `octants`.
    """
    all_s10_octants = [('N', 0), ('N', 1), ('N', 2), ('N', 3), ('S', 0), ('S', 1), ('S', 2), ('S', 3)]
    octants = all_s10_octants if (octants is None) else octants
    theta_vals = np.zeros(len(octants))
    phi_vals = np.zeros(len(octants))
    for i, (hemisphere, n) in enumerate(octants):
        theta_vals[i], phi_vals[i] = get_theta_phi_coords_in_octant(theta0, phi0, north=(hemisphere == 'N'), n=n)
    return theta_vals, phi_vals


def get_mirrored_dec_ra_coords(dec0, ra0, octants=None):
    """Given a pair (`dec0`, `ra0`) of declination and right ascension (in
    degrees) within the original S10 octant, return all eight pairs of
    coordinates mirrored onto the full sky, or mirrored into the given
    `octants`.
    """
    theta0, phi0 = utils.dec_ra_to_theta_phi(dec0, ra0)
    theta, phi = get_mirrored_theta_phi_coords(theta0, phi0, octants=octants)
    dec, ra = utils.theta_phi_to_dec_ra(theta, phi)
    return dec, ra


def get_mirrored_coords_in_patch(ra, dec, ra_min, ra_max, dec_min, dec_max, octants=None):
    """Given a pair (`ra`, `dec`) of right ascension and declination
    (in degrees), return all coordinate pairs that fall into the given
    patch of sky, after mirroring the input coordinate pair to the full
    sky.
    """
    # first, get the corresponding RA and dec within the original S10 octant:
    ra0, dec0 = (ra, dec) if utils.ra_dec_are_in_patch(ra, dec, 0, 0, 90, 90) else get_coords_in_s10_octant(ra, dec)
    # now get pairs of (ra, dec), mirrored onto other octants of sky:
    decs, ras = get_mirrored_dec_ra_coords(dec0, ra0, octants=octants)
    # only keep the pairs that are within the original patch
    ras_in_patch = [r for (r, d) in zip(ras, decs) if utils.ra_dec_are_in_patch(r, d, ra_min, dec_min, ra_max, dec_max)]
    decs_in_patch = [d for (r, d) in zip(ras, decs) if utils.ra_dec_are_in_patch(r, d, ra_min, dec_min, ra_max, dec_max)]
    return ras_in_patch, decs_in_patch


# for the different CIB models, based on the S10 CIB catalog:

def validate_cib_model_name(cib_model, valid_model_names=si.cib_model_names):
    """Verify that the name of the given CIB model is a valid name.

    The 'CIB model' refers to the the CIB catalog for the simulations is
    generated from the original full-sky catalog of the Sehgal et. al.
    (arXiv:0908.0540) simulations. See arXiv:XXXX.XXXXX (!! TODO:LINK2PAPER !!)
    for further details.

    Parameters
    ----------
    cib_model : str
        The name of the CIB model.
    valid_model_names : list of str, optional
        A list of valid CIB model names. The default is the list of CIB
        model names in `hdsims.siminfo.cib_model_names`.

    Returns
    -------
    cib_model : str
        The name of the CIB model. Will be all lowercase.

    Raises
    ------
    ValueError
        If the CIB model name is not in the list of `valid_model_names`.

    See Also
    --------
    hdsims.siminfo.cib_model_names : The default valid CIB model names.
    """
    if cib_model.lower() not in valid_model_names:
        raise ValueError(f"Unknown `{cib_model = }`. The CIB model name must be one of {valid_model_names}.")
    return cib_model.lower()



class S10Sims(lowres_sims.LowResSims):
    """Cut out maps and catalogs for a patch of sky from the full-sky
    Sehgal et. al. 2010 ('S10', arXiv:0908.0540) simulations.
    
    The intermediate, lower-resolution simulations of diffuse components
    (tSZ, kSZ, and lensing convergence maps) are projected from the full
    sky HEALPix maps to CAR maps on a patch of the sky. The catalogs of
    CIB and radio galaxies within the patch of sky are used to generate a
    set of lower-resolution CAR maps of the CIB and radio sources. 
    
    Attributes
    ----------
    `cib_model` : str or None
    `bin_info` : None
        If the default `hdsims` binning is not used  for power spectra,
        `bin_info` is a string used in filenames that describes the
        binning being used. Here, the default binning is used.
    
    Notes
    -----
    Attributes listed above without a description will have the same value
    as the corresponding parameter passed when initializing the class, or
    the default value if it is not passed. 
    
    The `cib_model` attribute will be added to the dictionary of 
    `default_kwargs` (inherited from `hdsims.simutils.Sims`). The 
    following methods of `hdsims.lowres_sims.LowResSims` are overridden 
    here:
        - `get_sim_catalog`
        - `get_intermediate_sim`
        - `get_intermediate_sim_power`
        - `apodize_intermediate_map`
        - `take_intermediate_map_power`
        - `get_lowres_sim_component_name`
    The `lowres_name` attribute is also overridden. Otherwise, see the 
    `hdsims.simutils.Sims` and `hdsims.lowres_sims.LowResSims` classes for
    more information about the inherited attributes and methods. 
    
    The methods defined in `hdsims.lowres_sims.LowResSims` (some of which
    are overridden here) are intended to be the only ones that a user may
    need to use this class.
    
    The full-sky S10 simulations are available on LAMBDA 
    (https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html). 
    The maps are in HEALPix format with an `nside` parameter of `8192`
    (approximately 0.43 arcminute resolution) for the tSZ, kSZ, CIB, and
    radio galaxies, or `4096` (approximately 0.86 arcminute resolution)
    for the lensing convergence map. The HEALPix maps of the tSZ, kSZ,
    CIB, and radio galaxies have been convolved with the pixel window
    function and are in units of Jy/steradian.

    The HEALPix pixel window is deconvolved from the full-sky S10 tSZ and
    kSZ maps, and they are converted to units of uK before cutting them
    out as CAR maps for the patch of sky. The S10 tSZ sims are also
    multiplied by 0.75 (as was done in arXiv:1808.07445). Since the S10
    catalog of SZ clusters is not used to generate any simulations, we do
    not modify any of its values (i.e., nothing in the SZ catalog will be
    multiplied by 0.75).

    We also multiply all CIB fluxes at each frequency in the original
    full-sky S10 catalog by 0.75 (as was done in arXiv:1808.07445) before
    saving the CIB catalog on the patch of sky. If the `cib_model` is not
    `None`, then this catalog is modified before being used to generate
    CIB simulations; see arXiv:XXXX.XXXXX (!! TODO !!) for further details.
    
    A minimum `map_apod_width` of 0.3 degrees is imposed due to the 0.86
    arcminute resolution of the full-sky S10 lensing convergence map; this
    ensures that the apodization width is at least 20 pixels in that map.
    """
    
    def __init__(self, hd_sims_dir, lowres_sims_dir,
                 freqs=si.freqs, components=si.components,
                 ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr, width=si.width, height=si.height, 
                 apod_width=si.apod_width, map_apod_width=None, res=si.hd_res,
                 make_output_dirs=True, verbose=False, log=None, cib_model=si.baseline_cib_model_name):
        """Initialization for the intermediate S10-resolution simulations
        on a patch of sky.
        
        Parameters
        ----------
        hd_sims_dir : str
            The path to the directory where all of the output files for
            this patch of sky will be saved. This directory will be 
            created if it does not already exist.
        lowres_sims_dir : str
            The path to the directory where the full-sky S10 simulations
            and catalogs have been saved. 
        freqs : list of int, default=[30, 90, 148, 219, 277, 350]
            A list of available frequencies (in GHz) for the simulations
            and catalogs. Each frequency in the list must be one of `30`,
            `90`, `148`, `219`, `277`, `350`.
        components : list of str, optional
            A list of available map or catalog components. Any components
            in the list that do not have a corresponding S10 map or
            catalog will be ignored. The available S10 components are
            `'ksz'`, `'tsz'`, `'kappa'` for the kSZ, tSZ and lensing
            convergence maps, respectively, and `'cib'`, `'radio'` for the
            CIB and radio catalogs, respectively.
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
        apod_width : int or float, default=1
            The width (in degrees) of the region along each edge of the
            map that will be apodized before calculating its power
            spectrum.

        Other Parameters
        ----------------
        cib_model : str, default='baseline'
            The name of the HD CIB model to use. The options are 
            `'baseline'` or `'alternative`'. The default is the 
            `'baseline'` CIB model used in arXiv:XXXX.XXXXX (!! TODO !!).
        map_apod_width : int or float, optional
            The width (in degrees) of the region along each edge of the 
            map that will be apodized before taking any Fourier or
            spherical harmonic transforms. By default, the value is either
            `apod_width/2`; however, if `apod_width/2` is less than 0.3, a
            default of `map_apod_width=0.3` is used. 
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
            
        Warns
        -----
        UserWarning
            If the chosen patch of sky contains dec. = 0 degrees or
            R.A. = 0 or 90 degrees, where the full-sky kSZ and lensing
            convergence maps have discontinuities.
        """
        # set `map_apod_width` to be at least ~20 pixels for S10 kappa map:
        min_map_apod_width = 0.3 # degrees
        if map_apod_width is None:
            map_apod_width = apod_width / 2
            if map_apod_width < min_map_apod_width:
                map_apod_width = min_map_apod_width
        super().__init__(hd_sims_dir, lowres_sims_dir, lowres_name='S10',
                         lowres_sim_components=si.s10_sim_components, freqs=simutils.validate_sim_freqs(freqs),
                         ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height, 
                         apod_width=apod_width, map_apod_width=map_apod_width, res=res,
                         verbose=verbose, log=log, make_output_dirs=make_output_dirs,)
        self.cib_model = validate_cib_model_name(cib_model)
        self.default_kwargs['cib_model'] = self.cib_model
        self.bin_info = None # use default binning if sim spectra are needed
        
        # warn the user if their patch crosses dec = 0 or RA = 0, 90, 180,
        # or 270, because the S10 kSZ & kappa maps have discontinuities 
        # at these locations:
        ra_min, ra_max, dec_min, dec_max = maps.get_patch_corner_coords(self.ra_ctr, self.dec_ctr, 
                                                                        self.width, height=self.height)
        ras = [0, 90, 180, 270]
        coord_info = [f'R.A. = {round(ra, 2)} deg.' for ra in ras if utils.ra_is_in_patch(ra, ra_min, ra_max)]
        if utils.dec_is_in_patch(0, dec_min, dec_max):
            coord_info.append('dec. = 0 degrees')
        coord_info = ', '.join(coord_info)
        map_ctr_info = f"R.A. = {round(self.ra_ctr,2)} deg., dec. = {round(self.dec_ctr,2)} deg."
        map_info = f"{self._map_area_info} map centered at {map_ctr_info}"
        if len(coord_info) > 0:
            warnings.warn(f"The {map_info} crosses {coord_info}, where there is a discontinuity "
                          "in the full-sky S10 kSZ and lensing convergence maps. These discontinuities "
                          "occur along dec. = 0 degrees and R.A. = 0, 90, 180, and 270 degrees."
                          "You may proceed, but just be aware of this.")

    
    # ---------------------------------------------------------------
    #  full-sky S10 healpix maps and catalogs:
    # ---------------------------------------------------------------
    
    def get_fullsky_s10_catalog_fnames(self, component):
        """Return the path(s) to the full-sky S10 catalog file(s).
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are: 
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.
        
        Returns
        -------
        str or list of str
            The absolute path to the catalog, or for `component='cib'`, a
            list of paths to each catalog file.
            
        Raises
        ------
        ValueError
            If the `lowres_sims_dir` attribute (defined during 
            initialization) is `None` or an empty string.
            
        Notes
        -----
        The full-sky S10 CIB catalog is broken up into multiple files;
        the radio and SZ cluster catalogs are saved in a single file.
        """
        if self.lowres_sims_dir in [None, '']:
            raise ValueError("You must provide the path to the `lowres_sims_dir`"
                             " during initialization in order to use the S10 data.")
        return get_fullsky_s10_catalog_fnames(component, s10_sims_dir=self.lowres_sims_dir)
    
    
    def get_fullsky_s10_sim_fname(self, component, freq=None, pixwin_deconvolved_uK=False):
        """Return the path to the full-sky S10 HEALPix map of the given
        component and frequency.
        
        Parameters
        ----------
        component : str
            The map component. Must be one of `'ksz'`, `'tsz'`, or
            `'kappa'` for the kSZ, tSZ and lensing convergence maps,
            respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If provided, it must be one of 
            `30`, `90`, `148`, `219`, `277`, `350`. Only required if 
            `component='tsz'`.
        pixwin_deconvolved_uK : bool, default=False
            Only used for `component='tsz'` or `component='ksz'`. 
            If `pixwin_deconvolved_uK=True`, returns the path to the 
            pixel-window-deconvolved map in units of uK; in this case, the
            tSZ map has also been multiplied by 0.75. Otherwise, returns
            the path to the original, unmodified S10 map.
        
        Returns
        -------
        str
            The absolute path to the map.
            
        Raises
        ------
        ValueError
            If the `lowres_sims_dir` attribute (defined during 
            initialization) is `None` or an empty string.
        """
        if self.lowres_sims_dir in [None, '']:
            raise ValueError("You must provide the path to the `lowres_sims_dir`"
                             " during initialization in order to use the S10 data.")
        if (component.lower() == 'ksz') and (freq is None): 
            # by default, we use the 90 GHz S10 kSZ map
            freq = si.s10_ksz_map_freq
        return get_fullsky_s10_sim_fname(component, freq=freq, s10_sims_dir=self.lowres_sims_dir, 
                                         pixwin_deconvolved_uK=pixwin_deconvolved_uK)
    
    
    def get_fullsky_s10sim(self, component, freq=None, save=True):
        """Return the full-sky S10 HEALPix map of the given component and
        frequency.
        
        The HEALPix pixel window will be deconvolved from the full-sky S10
        tSZ and kSZ maps, and they will be converted to units of uK. The
        S10 tSZ maps are also multiplied by 0.75.
        
        Parameters
        ----------
        component : str
            The map component. Must be one of `'ksz'`, `'tsz'`, or
            `'kappa'` for the kSZ, tSZ and lensing convergence maps,
            respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If provided, it must be one of
            `30`, `90`, `148`, `219`, `277`, or `350`. Only required if
            `component='tsz'`.
        save : bool, default=True
            Only used for `component='tsz'` or `component='ksz'`. 
            If `save=True`, the full-sky pixel-window-deconvolved map in
            units of uK (with tSZ multiplied by 0.75) will be saved. This
            may be useful when cutting out different patches of sky from
            the full-sky maps, since we need to take a spherical harmonic
            transform of the maps to deconvolve the pixel window.
        
        Returns
        -------
        fullsky_sim : array_like of float
            The full-sky HEALPix map.
            
        Raises
        ------
        ValueError
            If the `lowres_sims_dir` attribute (defined during 
            initialization) is `None` or an empty string.
            
        See Also
        --------
        healpy : Python package for HEALPix maps.
        """
        fullsky_s10_fname = self.get_fullsky_s10_sim_fname(component, freq=freq, pixwin_deconvolved_uK=True)
        if ('sz' in component) and (not os.path.exists(fullsky_s10_fname)):
            # only need one freq. for kSZ:
            freq = si.s10_ksz_map_freq if (component == 'ksz') else simutils.validate_sim_freq(freq) 
            # deconvolve pixel window from full-sky, convert to uK, 
            # and multiply by 0.75 for tSZ:
            scaling_factor = 0.75 if (component == 'tsz') else 1 
            orig_s10_fname = self.get_fullsky_s10_sim_fname(component, freq=freq, pixwin_deconvolved_uK=False)
            self.infomsg(f"deconvolving pixel window from full-sky healpix {freq} GHz {component} map")
            t = time.time()
            fullsky_sim = hp.read_map(orig_s10_fname)
            fullsky_sim = utils.Jy_per_str_to_uK(fullsky_sim, freq) * scaling_factor 
            fullsky_sim = maps.deconvolve_healpix_pixel_window(fullsky_sim, si.s10_nside)
            self.infomsg(f"{utils.tmsg(time.time() - t)} to deconvolve pixel window")
            if save:
                hp.write_map(fullsky_s10_fname, fullsky_sim)
                self.infomsg(f"saved {fullsky_s10_fname}")
        else:
            fullsky_sim = hp.read_map(fullsky_s10_fname)
        fullsky_sim = fullsky_sim.astype(np.float64)
        return fullsky_sim
        
    
    # ---------------------------------------------------------------
    #  lower-resolution CAR maps & catalogs for our patch of sky:
    # ---------------------------------------------------------------
    
    def get_intermediate_map_res(self, component):
        """Return the pixel resolution of the intermediate,
        S10-resolution CAR map.

        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for
            the kSZ, tSZ, lensing convergence, CIB, and radio maps,
            respectively.

        Returns
        -------
        res : float
            The map resolution, in arcminutes.

        See Also
        --------
        healpy.nside2resol : Map resolution from HEALPix `nside` parameter.

        Notes
        -----
        The resolution of the tSZ, kSZ, CIB, and radio maps is
        approximately 0.43 arcminutes, and the resolution of the
        lensing convergence map (`component='kappa'`) is approximately
        0.86 arcminutes.
        """
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        res = si.s10_kappa_res if (component == 'kappa') else si.s10_res
        return res


    def get_intermediate_map_shape_wcs(self, component, padded_ntimes=2):
        """Return the geometry of the intermediate, lower-resolution map
        on a patch of the sky.

        The geometry of a `pixell.enmap.ndmap` CAR map for a given map
        size, location, and resolution is defined by its `shape` and `wcs`
        attributes.

        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for
            the kSZ, tSZ, lensing convergence, CIB, and radio maps,
            respectively.
        padded_ntimes : int, default=2
            The number of times the patch of sky has been 'padded', i.e.
            increased by the `map_apod_width` on each side. The allowed
            values are:
            - `padded_ntimes=0` for patch with an area given by the `width`
              and `height` attributes defined during initialization,
            - `padded_ntimes=1` for patch with an area given by the
              `padded_width` and `padded_height` attributes,
            - `padded_ntimes=2` for patch with an area given by the
              `padded2x_width` and `padded2x_height` attributes.

        Returns
        -------
        shape : tuple of int
            The shape `(Ny, Nx)` of the array of map pixels on a given
            patch of sky at the lower resolution. `Ny` and `Nx` are the
            number of pixels along the dec. and R.A. directions,
            respectively.
        wcs : astropy.wcs.wcs.WCS
            Specifies the astropy World Coordinate System for the
            pixelization of the map at the lower resolution.

        See Also
        --------
        pixell.enmap.geometry, hdsims.maps.get_shape_wcs :
            Constructs the `shape`, `wcs` pair for a map.
        """
        if padded_ntimes not in [0, 1, 2]:
            raise ValueError(f"`{padded_ntimes = }`. Pass `padded_ntimes=0` for a {self._map_area_info} map,"
                             f" `padded_ntimes=1` for a {self._padded_map_area_info} map, or"
                             f" `padded_ntimes=2` for a {self._padded2x_map_area_info} map.")
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        res = self.get_intermediate_map_res(component)
        if padded_ntimes == 2:
            width = self.padded2x_width
            height = self.padded2x_height
        elif padded_ntimes == 1:
            width = self.padded_width
            height = self.padded_height
        else:
            width = self.width
            height = self.height
        shape, wcs = maps.get_shape_wcs(res, self.ra_ctr, self.dec_ctr, width, height=height)
        return shape, wcs
    

    def get_intermediate_map_apod_window_fname(self, component, **kwargs):
        """Return the path to the apodization window for a 
        S10-resolution CAR map.
        
        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for 
            the kSZ, tSZ, lensing convergence, CIB, and radio maps, 
            respectively.
        
        Returns
        -------
        fname : str
            The path to the apodization window.
        
        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments for the apodization window:
            - The `width` and `height` (int or float), in degrees, of the
              map. By default, the values of the `padded2x_width` and 
              `padded2x_height` attributes are used.
            - The `apod_width` (degrees) to use. The default is given by
              the `map_apod_width` attribute. 
        """
        defaults = {'width': self.padded2x_width, 'height': self.padded2x_height, 'apod_width': self.map_apod_width}
        kwargs = self.get_kwargs_with_defaults(defaults=defaults, **kwargs)
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        fname = simutils.get_apod_window_fname(kwargs['apod_width'], kwargs['width'], kwargs['height'], 
                                               maps_dir=self.intermediate_maps_dir())
        if component == 'kappa':
            fname = fname.replace('.fits', '_s10kappa.fits')
        else:
            fname = fname.replace('.fits', '_s10.fits')
        return fname


    def get_intermediate_map_apod_window(self, component, save=False, **kwargs):
        """Return the apodization window for a S10-resolution CAR map.

        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for
            the kSZ, tSZ, lensing convergence, CIB, and radio maps,
            respectively.
        save : bool, default=False
            If `save=True`, the apodization window will be saved.

        Returns
        -------
        window : pixell.enmap.ndmap
            The apodization window.

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments for the apodization window:
            - The `width` and `height` (int or float), in degrees, of the
              map. By default, the values of the `padded2x_width` and
              `padded2x_height` attributes are used.
            - The `apod_width` (degrees) to use. The default is given by
              the `map_apod_width` attribute.

        See Also
        --------
        get_intermediate_map_apod_window_fname :
            The filename of the window used if `save=True`.
        apodize_intermediate_map :
            Applies the apodization to a map at the S10 resolution.
        """
        fname = self.get_intermediate_map_apod_window_fname(component, **kwargs)
        if os.path.exists(fname):
            window = enmap.read_map(fname)
        else:
            defaults = {'width': self.padded2x_width, 'height': self.padded2x_height, 'apod_width': self.map_apod_width}
            kwargs = self.get_kwargs_with_defaults(defaults=defaults, **kwargs)
            res = self.get_intermediate_map_res(component)
            shape, wcs = maps.get_shape_wcs(res, self.ra_ctr, self.dec_ctr, kwargs['width'], height=kwargs['height'])
            self.infomsg("making apodization window for intermediate "
                         f"{round(kwargs['width'],2)} deg. x {round(kwargs['height'],2)} deg. map")
            t = time.time()
            window = maps.make_apod_window(shape, wcs, kwargs['apod_width'])
            if save:
                enmap.write_map(fname, window)
                self.infomsg(f"saved {fname}")
            self.infomsg(f"{utils.tmsg(time.time() - t)} to make the window")
        return window

    
    # ---------------------------------------------------------------
    #  power spectra of the S10 sims on our patch of sky:
    # ---------------------------------------------------------------
        
    def binning_file(self):
        """Returns the path to the binning file used to bin the power
        spectra of the simulations. 
        """
        fname = simutils.get_binning_file_name(binning_dir=self.binning_dir())
        if not os.path.exists(fname):
            simpower.save_binning_file(fname)
        return fname
    

    def get_intermediate_map_power_lmax(self, component):
        """Return the maximum multipole used when calculating the power
        spectrum of an intermediate, S10-resolution CAR map.

        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for
            the kSZ, tSZ, lensing convergence, CIB, and radio maps,
            respectively.

        Returns
        -------
        lmax : int
            The maximum multipole.

        See Also
        --------
        pspy.so_map.so_map.get_lmax_limit
        """
        shape, wcs = self.get_intermediate_map_shape_wcs(component, padded_ntimes=0)
        lmax = int(maps.enmap2pspy(enmap.ones(shape, wcs)).get_lmax_limit())
        return lmax


    def get_intermediate_mode_coupling_fnames(self, component, bin_dl=False):
        """Return the path to the inverse mode-coupling matrix and the
        binning matrix for the intermediate, S10-resolution CAR maps.
        
        The inverse mode-coupling matrix bins the power spectrum of the
        S10-resolution CAR maps and applies a correction to the power
        spectrum for the effect of apodizing the maps.
        
        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for 
            the kSZ, tSZ, lensing convergence, CIB, and radio maps, 
            respectively.
        bin_dl : bool, default=False
            If `bin_dl=True`, the power spectra are multiplied by a factor
            of `ell * (ell + 1) / (2 * pi)` at each multipole `ell`. 
            
        Returns
        -------
        mbb_inv_fname, bbl_fname : str
            The paths to the inverse mode-coupling matrix and the binning
            matrix, respectively.
        """
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        lmax = self.get_intermediate_map_power_lmax(component)
        fname_info = 's10kappa' if (component == 'kappa') else 's10'
        if self.bin_info is not None:
            fname_info = f'{self.bin_info}_{fname_info}'
        mbb_inv_fname, bbl_fname = simutils.get_mode_coupling_fnames(self.apod_width, lmax, bin_dl=bin_dl, 
                                                                     pol=False, fname_info=fname_info, 
                                                                     binning_dir=self.intermediate_maps_dir())
        return mbb_inv_fname, bbl_fname


    def get_intermediate_inv_mcm(self, component, bin_dl=False):
        """Return the path the inverse mode-coupling matrix for the
        S10-resolution CAR maps.
        
        The inverse mode-coupling matrix bins the power spectrum of the
        S10-resolution CAR maps and applies a correction to the power
        spectrum for the effect of apodizing the maps.
        
        Parameters
        ----------
        component : str
            The name of the map component. The available S10 components
            are `'ksz'`, `'tsz'`, `'kappa'`, `'cib'`, or `'radio'` for 
            the kSZ, tSZ, lensing convergence, CIB, and radio maps, 
            respectively.
        bin_dl : bool, default=False
            If `bin_dl=True`, the power spectra are multiplied by a factor
            of `ell * (ell + 1) / (2 * pi)` at each multipole `ell`. 
            
        Returns
        -------
        mbb_inv : array_like of float
            The inverse mode-coupling matrix with shape `(nbin, nbin)`,
            where `nbin` is the number of power spectrum bins (determined
            by the bin edges and the maximum multipole).
            
        See Also
        --------
        pspy.so_mcm.mcm_and_bbl_spin0, hdsims.simpower.calc_mode_coupling :
            Calculate an inverse mode-coupling matrix.
        binning_file : 
            The binning file used to bin the power spectrum.
        get_intermediate_map_power_lmax : 
            The maximum multipole used for the power spectrum.
        """
        fname, bbl_fname = self.get_intermediate_mode_coupling_fnames(component, bin_dl=bin_dl)
        if os.path.exists(fname):
            mbb_inv = np.loadtxt(fname)
        else:
            component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
            lmax = self.get_intermediate_map_power_lmax(component)
            binning_file = self.binning_file()
            # get window for the final `width` x `height` sim
            window = self.get_intermediate_map_apod_window(component, save=True, width=self.width, height=self.height, apod_width=self.apod_width)
            self.infomsg(f"calculating inverse mode-coupling matrix to take power of the intermediate low-resolution sim")
            t = time.time()
            mbb_inv, bbl = simpower.calc_mode_coupling(window, lmax, binning_file, bin_dl=bin_dl, pol=False) # don't need pol here
            simpower.save_mode_coupling_files(mbb_inv, fname, bbl=bbl, bbl_fname=bbl_fname)
            self.infomsg(f"{utils.tmsg(time.time() - t)} to calculate inverse mode-coupling matrix ; saved to {fname}")
        return mbb_inv


    def get_intermediate_sim_power_fname(self, component, freq=None, bin_dl=False, **kwargs):
        """Return the path to the saved power spectrum of the
        intermediate, S10-resolution map.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'ksz'`,
            `'tsz'`, `'cib'`, `'radio'`, or `'kappa'` for the kSZ, tSZ,
            CIB, radio, or lensing convergence maps, respectively.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components. The allowed frequencies are
            30, 90, 148, 219, 277, or 350 GHz.
        bin_dl : bool, default=False
            If `bin_dl=True`, the power spectra are multiplied by a factor
            of `ell * (ell + 1) / (2 * pi)` at each multipole `ell`.

        Returns
        -------
        fname : str
            The path to the power spectrum file.

        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the
            `cib_model` attribute defined during initialization.
        """
        spec_type = 'dl' if bin_dl else 'cl'
        fname_info = []
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        if simutils.has_freq_dependent_component(component):
            freq = simutils.validate_sim_freq(freq)
            fname_info.append(f'{freq:03d}')
        fname_info.append(self.get_lowres_sim_component_name(component, **kwargs))
        if self.bin_info is not None: # for non-default binning
            fname_info.append(self.bin_info)
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.intermediate_maps_dir(), f'{fname_root}_s10_{spec_type}s.txt')
        return fname
    
    
    # ---------------------------------------------------------------
    #  going from the full-sky to the patch of sky:
    # ---------------------------------------------------------------
    
    def cutout_car_patch_from_fullsky_healpix(self, component, freq=None, save=False):
        """Project from the full-sky S10 HEALPix map to a CAR map at the
        same resolution on a patch of sky.
        
        The size of the patch is given by the `padded2x_width` and
        `padded2x_height` attributes (inherited from the 
        `hdsims.simutils.Sims` class).
        
        Parameters
        ----------
        component : str
            The map component. Must be one of `'ksz'`, `'tsz'`, or
            `'kappa'` for the kSZ, tSZ and lensing convergence maps,
            respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If provided, it must be one of
            `30`, `90`, `148`, `219`, `277`, or `350`. Only required if
            `component='tsz'`.
        save : bool, default=False
            Whether to save the CAR map.
            
        Returns
        -------
        pixell.enmap.ndmap
            The S10 CAR map on the patch of sky.
        
        See Also
        --------
        pixell.reproject.healpix2map
        hdsims.maps.cutout_car_patch_from_fullsky_healpix
        """
        component = simutils.validate_sim_component_name(component)
        # we only have full-sky maps saved for tsz, ksz, and kappa
        if component not in ['tsz', 'ksz', 'kappa']:
            raise ValueError(f"`{component = }`. The full-sky S10 sims are only saved for `'tsz'`, `'ksz'`, and `'kappa'`.")
        if component == 'tsz': # make sure a valid freq. was passed
            freq = simutils.validate_sim_freq(freq)
        # get the filename for the CAR map and check if it is saved:
        cutout_fname = self.get_intermediate_sim_fname(component, freq=freq)
        if os.path.exists(cutout_fname):
            cutout = enmap.read_map(cutout_fname)
        else:
            map_info = f'{freq} GHz {component}' if (freq is not None) else component
            self.infomsg(f"{map_info} : cutting out a CAR map from the full-sky healpix map")
            t = time.time()
            shape, wcs = self.get_intermediate_map_shape_wcs(component, padded_ntimes=2)
            fullsky_map = self.get_fullsky_s10sim(component, freq=freq)
            cutout = maps.cutout_car_patch_from_fullsky_healpix(fullsky_map, shape, wcs)
            self.infomsg(f"{utils.tmsg(time.time() - t)} to cut out CAR map from full-sky healpix map")
            if save:
                enmap.write_map(cutout_fname, cutout)
                self.infomsg(f"saved {cutout_fname}")
        return cutout
    
    
    def get_patch_catalog_fname(self, component):
        """Return the path to the catalog that contains all S10 sources
        within the patch of sky.
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are: 
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.
            
        Returns
        -------
        str
            The path to the catalog file.
            
        See Also
        --------
        trim_fullsky_s10_catalog_for_patch : Saves the catalog.
        """
        return simutils.get_catalog_fname(component, self.padded2x_width, height=self.padded2x_height, 
                                          cib_model='s10', catalog_dir=self.intermediate_maps_dir(), file_ext='txt')
    
    
    def load_patch_catalog(self, component):
        """Load the catalog that contains all S10 sources within the patch
        of sky.
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are: 
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.
        
        Returns
        -------
        catalog : pandas.DataFrame
            A catalog of objects in the simulation of the `component` on
            the full patch of sky defined by the `padded2x_width` and
            `padded2x_height` attributes. 
            
            The catalog has columns named `'RADeg'` and `'decDeg'` for the
            R.A. and dec. coordinates (in degrees), respectively, of each
            object. For CIB and radio galaxies, the columns for the flux
            (in mJy) of each source at each `freq` (in GHz) in the list of
            `freqs` are named `'fluxmJy_{freq}GHz'`; e.g., the column name
            is `'fluxmJy_90GHz'` for 90 GHz.
        
        See Also
        --------
        trim_fullsky_s10_catalog_for_patch : Saves the catalog.
        get_patch_catalog_fname : Path to the catalog file.
        """
        catalog_fname = self.get_patch_catalog_fname(component)
        if 'sz' in component.lower():
            catalog_cols = si.s10_catalog_col_names['sz']
        else:
            catalog_cols = si.s10_catalog_col_names[component.lower()]
        catalog = pd.DataFrame(utils.load_dict_from_file(catalog_fname, catalog_cols))
        return catalog
    
    
    def get_patch_catalog(self, component):
        """Return the catalog that contains all S10 sources within the
        patch of sky.
        
        The size of the patch is given by the `padded2x_width` and
        `padded2x_height` attributes (inherited from the 
        `hdsims.simutils.Sims` class).
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are: 
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.
        
        Returns
        -------
        catalog : pandas.DataFrame
            A catalog of objects in the simulation of the `component` on
            the full patch of sky defined by the `padded2x_width` and
            `padded2x_height` attributes. 
            
            The catalog has columns named `'RADeg'` and `'decDeg'` for the
            R.A. and dec. coordinates (in degrees), respectively, of each
            object. For CIB and radio galaxies, the columns for the flux
            (in mJy) of each source at each `freq` (in GHz) in the list of
            `freqs` are named `'fluxmJy_{freq}GHz'`; e.g., the column name
            is `'fluxmJy_90GHz'` for 90 GHz.
        
        See Also
        --------
        trim_fullsky_s10_catalog_for_patch : Saves the catalog.
        get_patch_catalog_fname : Path to the catalog file.
        
        Notes
        -----
        The CIB and radio catalogs are used to generate the corresponding
        simulations at each frequency. For the CIB, we multiply all fluxes
        at each frequency in the original S10 catalog by 0.75 before
        saving and returning the catalog on the patch of sky. 
        
        The full-sky S10 tSZ sims are also multiplied by 0.75 before 
        cutting them out as CAR maps for the patch of sky. Because the SZ
        catalog is not used to generate any simulations, we do not modify
        any of its values (i.e., nothing in the SZ catalog will be 
        multiplied by 0.75).
        """
        catalog_fname = self.get_patch_catalog_fname(component)
        if not os.path.exists(catalog_fname):
            self.trim_fullsky_s10_catalog_for_patch(component)
        catalog = self.load_patch_catalog(component)
        return catalog
    
    
    def trim_fullsky_s10_catalog_for_patch(self, component):
        """Load in the full-sky S10 catalog(s) and save a copy that
        contains only the rows within a patch of sky.
        
        The size of the patch is given by the `padded2x_width` and
        `padded2x_height` attributes (inherited from the 
        `hdsims.simutils.Sims` class).
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are: 
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.
        
        See Also
        --------
        get_patch_catalog_fname : Path to the saved catalog.
        get_patch_catalog : Loads the catalog after saving it.
        
        Notes
        -----
        The combined size of the full-sky S10 CIB catalogs is about 81 GB,
        so it can take a very long time to loop through each row of each 
        catalog file.
        
        The CIB and radio catalogs are used to generate the corresponding
        simulations at each frequency. For the CIB, we multiply all fluxes
        at each frequency in the original S10 catalog by 0.75 before
        saving and returning the catalog on the patch of sky. 
        
        The full-sky S10 tSZ sims are also multiplied by 0.75 before 
        cutting them out as CAR maps for the patch of sky. Because the SZ
        catalog is not used to generate any simulations, we do not modify
        any of its values (i.e., nothing in the SZ catalog will be 
        multiplied by 0.75).
        """
        # get the correct `component`:
        if 'sz' in component.lower():
            component = 'sz'
        elif component.lower() not in ['cib', 'radio']:
            raise ValueError(f"`{component = }`. The valid `component` values for the "
                             "full-sky S10 catalogs are `'cib'`, `'radio'`, or `'sz'`.")
        else:
            component = component.lower()
        # we need to know if we're dealing with the CIB, since it has
        # multiple full-sky catalog files and we will multiply all CIB
        # fluxes by 0.75:
        cib = (component == 'cib')

        # if the catalog file for the patch of sky does not exist, save it:
        fname = self.get_patch_catalog_fname(component)
        if not os.path.exists(fname):
            self.infomsg(f"loading full-sky catalog for {component} and saving a trimmed version for our patch of sky")
            t = time.time()

            # get the min./max. R.A. and dec for the patch of sky:
            corners = maps.get_patch_corner_coords(self.ra_ctr, self.dec_ctr, 
                                                   self.padded2x_width, height=self.padded2x_height)
            # the 'full-sky' S10 catalog actually only contains objects
            # within an octant of the sky, which is then mirrored to the
            # full sky (see arXiv:0908.0540 for details); below, we calculate
            # which octants our patch of sky corresponds to:
            octants = get_s10_octants_in_patch(self.ra_ctr, self.dec_ctr, self.padded2x_width, self.padded2x_height)
            
            # get a list of full-sky S10 catalog files:
            catalog_files = self.get_fullsky_s10_catalog_fnames(component)
            if not cib:
                catalog_files = [catalog_files]
            # get list of column names, and column indices for RA and dec:
            cols = si.s10_catalog_col_names[component]
            ra_index = si.s10_catalog_ra_col_index[component]
            dec_index = si.s10_catalog_dec_col_index[component]
            
            # loop through the catalogs, keeping only rows that have 
            # R.A. and dec. positions within our patch of sky:
            header = ' '.join(cols)
            header = f'# {header}\n'
            lines = [header] # lines to save in catalog file
            for catalog_file in catalog_files:
                self.infomsg(f"loading full-sky S10 catalog from {catalog_file}")
                with open(catalog_file, 'r') as f:
                    for line in f:
                        values_in_row = line.strip('\n').split()
                        ra = float(values_in_row[ra_index])
                        dec = float(values_in_row[dec_index])
                        # mirror this pair of coordinates into the other 
                        # octants, and add a row for each pair that falls
                        # within our patch of sky:
                        ras, decs = get_mirrored_coords_in_patch(ra, dec, *corners, octants=octants)
                        for (r, d) in zip(ras, decs): 
                            values_in_row[ra_index] = str(r)
                            values_in_row[dec_index] = str(d)
                            new_line = ' '.join(values_in_row)
                            lines.append(f'{new_line}\n')

            # save the trimmed catalog
            txt_to_save = ''.join(lines)
            with open(fname, 'w') as f:
                f.write(txt_to_save)
            if cib: # multiply the fluxes by 0.75
                self.infomsg("multiplying all fluxes in trimmed CIB catalog by 0.75")
                flux_cols = [col for col in cols if ('flux' in col.lower())]
                # load data into a dict (as float, instead of the str we just saved)
                trimmed_catalog = utils.load_dict_from_file(fname, cols)
                for col in flux_cols:
                    trimmed_catalog[col] *= 0.75
                # re-save the catalog
                utils.save_dict_to_file(fname, trimmed_catalog, keys=cols)

            self.infomsg(f"{utils.tmsg(time.time()-t)} to get catalog ({len(lines)-1} rows); saved to {fname}")
    
    
    def generate_intermediate_point_source_maps(self, component, save=False, **kwargs):
        """Generate a set of lower-resolution maps of point sources from
        the catalog of sources in the patch of sky.

        The size of the patch is given by the `padded2x_width` and
        `padded2x_height` attributes (inherited from the
        `hdsims.simutils.Sims` class).

        Parameters
        ----------
        component : str
            The name of the point source component. The allowed values are
            `'cib'` or `'radio'` for CIB or radio galaxies, respectively.
        save : bool, default=False
            Whether to save the maps.

        Returns
        -------
        sims : dict of pixell.enmap.ndmap
            A dictionary with a key for each frequency (`int`) holding the
            map at that frequency.

        Other Parameters
        ----------------
        **kwargs : dict
            The additional, optional keyword arguments are:
            - `freqs` (`list` of `int`): A list of map frequencies (GHz)
              to generate. The default is given by the `freqs` attribute
              defined during initialization. Each frequency in the list
              must be one of `30`, `90`, `148`, `219`, `277`, or `350`.
            - `cib_model` (`str` or `None`): A name for the CIB model to
              use when generating the maps. The default is given by the
              `cib_model` attribute defined during initialization. This
              will only be used if `component='cib'`.

        See Also
        --------
        get_intermediate_sim_fname :
            The filename used when saving the maps at each frequency.
        """
        freqs = simutils.validate_sim_freqs(self.get_kwarg('freqs', **kwargs))
        component = simutils.validate_sim_component_name(component, valid_components=['cib', 'radio'])
        cib_model = validate_cib_model_name(self.get_kwarg('cib_model', **kwargs))
        fnames = {freq: self.get_intermediate_sim_fname(component, freq=freq, cib_model=cib_model) for freq in freqs}
        # check if all sims are already saved:
        sims_are_saved = all([os.path.exists(fnames[freq]) for freq in freqs])
        if sims_are_saved:
            sims = {freq: enmap.read_map(fnames[freq]) for freq in freqs}
        else:
            t = time.time()
            catalog = self.get_sim_catalog(component, cib_model=cib_model)
            # put the sources on the intermediate-resolution maps:
            self.infomsg(f"generating intermediate {component} maps for {freqs = } GHz "
                         f"from the catalog of {len(catalog)} {component} sources")
            shape, wcs = self.get_intermediate_map_shape_wcs(component)
            sims = maps.make_src_maps(shape, wcs, catalog, freqs=freqs)
            if save:
                for freq in freqs:
                    enmap.write_map(fnames[freq], sims[freq])
                    self.infomsg(f"saved intermediate {freq} GHz {component} map to {fnames[freq]}")
            self.infomsg(f"{utils.tmsg(time.time() - t)} to generate intermediate {component} maps")
        return sims

    
    def get_intermediate_sim_fname(self, component, freq=None, **kwargs):
        """Return the path to the S10-resolution CAR map of the given
        component and frequency.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'ksz'`,
            `'tsz'`, `'cib'`, `'radio'`, or `'kappa'` for the kSZ, tSZ,
            CIB, radio, or lensing convergence maps, respectively.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components. The allowed frequencies are
            30, 90, 148, 219, 277, or 350 GHz.

        Returns
        -------
        fname : str
            The path to the map file.

        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the
            `cib_model` attribute defined during initialization.
        """
        fname_info = []
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        if simutils.has_freq_dependent_component(component):
            fname_info.append(f'{simutils.validate_sim_freq(freq):03d}')
        fname_info.append(self.get_lowres_sim_component_name(component, **kwargs))
        fname_info.append(f'{simutils.round_str(self.padded2x_width)}x{simutils.round_str(self.padded2x_height)}deg')
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.intermediate_maps_dir(), f'{fname_root}_s10.fits')
        return fname


    def load_intermediate_sim(self, component, freq=None, **kwargs):
        """Return the S10-resolution CAR map of the given component and
        frequency.
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'ksz'`,
            `'tsz'`, `'cib'`, `'radio'`, or `'kappa'` for the kSZ, tSZ,
            CIB, radio, or lensing convergence maps, respectively.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components. The allowed frequencies are
            30, 90, 148, 219, 277, or 350 GHz.
        
        Returns
        -------
        pixell.enmap.ndmap
            The lower-resolution map of the given component and frequency
            on the patch of sky.
        
        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the 
            `cib_model` attribute defined during initialization.
        """
        fname = self.get_intermediate_sim_fname(component, freq=freq, **kwargs)
        return enmap.read_map(fname)
    
    
    def generate_cib_sim_catalog(self, **kwargs):
        """Generate the CIB catalog used to make CIB simulations for the
        given CIB model.

        The catalog will include CIB sources within an area given by the
        `padded2x_width` and `padded2x_height` attributes (inherited from
        the `hdsims.simutils.Sims` class).

        Returns
        -------
        catalog : pandas.DataFrame
            A catalog of CIB sources on the patch of sky defined by the
            `padded2x_width` and `padded2x_height` attributes. The catalog
            has columns named `'RADeg'` and `'decDeg'` for the  R.A. and
            dec. coordinates (in degrees), respectively, of each source.
            The columns for the flux (in mJy) of each source at each
            `freq` (in GHz) are named `'fluxmJy_{freq}GHz'`; e.g., the
            column name for the 90 GHz flux is `'fluxmJy_90GHz'`. The
            catalog has columns for the flux at 30, 90, 148, 219, 277, and
            350 GHz.

        Other Parameters
        ----------------
        **kwargs : dict
            The name of the `cib_model` (`str` or `None`) to use. The
            default is given by the `cib_model` attribute defined during
            initialization.

        See Also
        --------
        trim_fullsky_s10_catalog_for_patch :
            The catalog of all S10 sources within the patch of sky.

        Notes
        -----
        Each CIB model has an associated `pixel_size` (map resolution),
        which may be different from the S10 resolution. To generate the
        catalog for the CIB model, we:
        1. Obtain the catalog of all CIB sources from the full-sky S10
           catalog that are located within the patch of sky.
        2. Calculate which pixel each source would fall in to when placed
           on a map with the `pixel_size` resolution. (As the map pixels
           get larger, more sources will fall into a given pixel,
           increasing the total flux in that pixel.)
        3. Make a new catalog where each non-zero pixel in the
           `pixel_size`-resolution map corresponds to a single source.
           The flux of each source is given by the total flux of all
           sources in that pixel, and the location of each source is
           given by the R.A. and dec. of the pixel center. (As a result,
           the coordinates in this catalog will all fall onto a grid.)
        4. Add some scatter, randomly drawn from a Gaussian distribution,
           to the position of each source. The standard deviation of the
           distribution is 20% of the `pixel_size`. This is only done if
           the `pixel_size` is larger than the size of the
           ultrahigh-resolution map pixels.

        Since there are many CIB sources in the full-sky S10 catalogs (the
        combined size of the catalog files is about 81 GB), it may take a
        very long time to generate the catalog for the CIB model.
        """
        component = 'cib'
        freqs = si.freqs # do this once for all freqs in S10 catalog
        cib_model = validate_cib_model_name(self.get_kwarg('cib_model', **kwargs),
                                            valid_model_names=[*si.cib_model_names, None])
        fname = self.get_sim_catalog_fname(component, cib_model=cib_model)
        if not os.path.exists(fname):
            # get the catalog of all CIB sources in our patch of sky from the full-sky catalog:
            catalog = self.get_patch_catalog(component)
            if cib_model is not None:
                self.infomsg(f"making catalog for CIB sims using the '{cib_model}' CIB model")
                t = time.time()
                # make a catalog with (at most) 1 source per pixel (for this CIB model pixel size),
                # placed at the pixel center:
                pixel_size = si.cib_model_pixel_res[cib_model]
                shape, wcs = maps.get_shape_wcs(pixel_size, self.ra_ctr, self.dec_ctr,
                                                self.padded2x_width, height=self.padded2x_height)
                catalog = maps.make_catalog_for_map_geometry(catalog, shape, wcs)
                if pixel_size - self.res > self.res: # add some scatter to the catalog positions:
                    catalog['RADeg'], catalog['decDeg'] = fgcatalogs.add_gauss_scatter_to_coords(catalog['RADeg'].values,
                                                                                                 catalog['decDeg'].values,
                                                                                                 shape, wcs)
                catalog.to_csv(fname)
                self.infomsg(f"{utils.tmsg(time.time()-t)} to make CIB sim catalog ({len(catalog)} rows); saved {fname}")
        else:
            self.infomsg(f"loading catalog for CIB sims ('{cib_model}' CIB model) from {fname}")
            catalog = fgcatalogs.load_catalog(fname)
        return catalog
    
    
    def get_sim_catalog_fname(self, component, **kwargs):
        """Return the path to the catalog of objects in the simulations
        for a patch of sky.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are:
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.

        Returns
        -------
        fname : str
            The path to the catalog file.

        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the
            `cib_model` attribute defined during initialization.
        """
        # get file name for the catalog that was "cut out" for the patch
        # of sky from the full-sky catalog:
        fname = self.get_patch_catalog_fname(component)
        # for the CIB catalog, include information about the `cib_model`:
        cib_model = self.get_kwarg('cib_model', **kwargs)
        if ('cib' in component.lower()) and (cib_model is not None):
            # for clarity, add `'hd'` to the filename to distinguish the
            # catalog for this CIB model from the catalog that was
            # 'cut out' from the full-sky catalog:
            if cib_model.lower() == si.baseline_cib_model_name:
                cib_model_name = 'hd'
            elif cib_model.lower() == si.s10_cib_model_name:
                cib_model_name = 'hd_s10model'
            else:
                cib_model_name = f'hd_{cib_model}'
            catalog_path, catalog_fname = os.path.split(fname)
            if 's10' in catalog_fname:
                catalog_fname = catalog_fname.replace('s10', cib_model_name)
            else:
                catalog_fname = catalog_fname.replace('cib_', f'cib_{cib_model_name}')
            catalog_fname = catalog_fname.replace('.txt', '.csv')
            fname = os.path.join(catalog_path, catalog_fname)
        return fname
    
    
    def load_sim_catalog(self, component, **kwargs):
        """Load the catalog of objects in the simulations for a patch of
        sky, cut out from a full-sky catalog.
        
        This catalog contains all objects (e.g. CIB or radio galaxies, SZ
        clusters) within the region defined by the `padded2x_width` and
        `padded2x_height` attributes inherited from the 
        `hdsims.simutils.Sims` class.
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are: 
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.
            
        Returns
        -------
        catalog : pandas.DataFrame
            A catalog of objects in the simulation of the `component` on
            the full patch of sky defined by the `padded2x_width` and
            `padded2x_height` attributes. 
            
            The catalog has columns named `'RADeg'` and `'decDeg'` for the
            R.A. and dec. coordinates (in degrees), respectively, of each
            object. For CIB and radio galaxies, the columns for the flux
            (in mJy) of each source at each `freq` (in GHz) in the list of
            `freqs` are named `'fluxmJy_{freq}GHz'`; e.g., the column name
            is `'fluxmJy_90GHz'` for 90 GHz.
            
        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the 
            `cib_model` attribute defined during initialization.
        
        See Also
        --------
        get_sim_catalog : Makes the catalog, if necessary.
        """
        catalog_fname = self.get_sim_catalog_fname(component, **kwargs)
        if catalog_fname.endswith('txt'):
            if 'sz' in component.lower():
                catalog_cols = si.s10_catalog_col_names['sz']
            else:
                catalog_cols = si.s10_catalog_col_names[component.lower()]
            catalog = pd.DataFrame(utils.load_dict_from_file(catalog_fname, catalog_cols))
        else:
            catalog = fgcatalogs.load_catalog(catalog_fname)
        return catalog
    
    
    
    # ===============================================================
    #  methods that override `hdsims.lowres_sims.LowResSims` methods:
    # ===============================================================
    
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
            The name of the map component. The allowed values are:
            `'radio'` for radio galaxies, `'cib'` for the CIB, or one of
            `'sz'`, `'tsz'`, or `'ksz'` for SZ clusters.

        Returns
        -------
        catalog : pandas.DataFrame
            A catalog of objects in the simulation of the `component` on
            the full patch of sky defined by the `padded2x_width` and
            `padded2x_height` attributes.

            The catalog has columns named `'RADeg'` and `'decDeg'` for the
            R.A. and dec. coordinates (in degrees), respectively, of each
            object. For CIB and radio galaxies, the columns for the flux
            (in mJy) of each source at each `freq` (in GHz) in the list of
            `freqs` are named `'fluxmJy_{freq}GHz'`; e.g., the column name
            is `'fluxmJy_90GHz'` for 90 GHz.

        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass a `cib_model` (`str` or
            `None`) for the CIB model to use. The default is given by the
            `cib_model` attribute defined during initialization. If
            `cib_model=None`, the catalog of all S10 sources within the
            patch of sky is returned.

        Notes
        -----
        The CIB and radio catalogs are used to generate the corresponding
        simulations at each frequency. For the CIB, we multiply all fluxes
        at each frequency in the original S10 catalog by 0.75 before
        saving and returning the catalog on the patch of sky.

        The full-sky S10 tSZ sims are also multiplied by 0.75 before
        cutting them out as CAR maps for the patch of sky. Because the SZ
        catalog is not used to generate any simulations, we do not modify
        any of its values (i.e., nothing in the SZ catalog will be
        multiplied by 0.75).
        """
        # if the catalog was saved, load it; otherwise, need to make it:
        catalog_fname = self.get_sim_catalog_fname(component, **kwargs)
        if os.path.exists(catalog_fname):
            catalog = self.load_sim_catalog(component, **kwargs)
        else:
            cib_model = self.get_kwarg('cib_model', **kwargs)
            if ('cib' in component.lower()) and (cib_model is not None):
                # generate the catalog for this `cib_model`:
                catalog = self.generate_cib_sim_catalog(**kwargs)
            else:
                # cut out the catalog from the full sky:
                catalog = self.get_patch_catalog(component)
        return catalog
    
    
    def get_intermediate_sim(self, component, freq=None, save=False, **kwargs):
        """Return the lower-resolution simulation for single map component
        and frequency on a patch of the sky.
        
        The simulation will have the same resolution as the full-sky maps,
        and an area defined by the `padded2x_width` and `padded2x_height`
        attributes inherited from the `hdsims.simutils.Sims` class. Maps 
        of diffuse components (tSZ, kSZ, or lensing convergence) are cut
        out from the full-sky maps, and maps of discrete components (CIB
        or radio galaxies) are generated from their corresponding catalog.
        
        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'ksz'`,
            `'tsz'`, `'cib'`, `'radio'`, or `'kappa'` for the kSZ, tSZ,
            CIB, radio, or lensing convergence maps, respectively.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components. The allowed frequencies are
            30, 90, 148, 219, 277, or 350 GHz.
        save : bool, default=False
            Whether to save the map.
        
        Returns
        -------
        sim : pixell.enmap.ndmap
            The lower-resolution map of the given component and frequency
            on the patch of sky.
        
        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the 
            `cib_model` attribute defined during initialization.
            
        See Also
        --------
        get_sim_catalog : 
            The catalog used to generate maps of discrete components.
        """
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        if simutils.has_freq_dependent_component(component):
            freq = simutils.validate_sim_freq(freq)
        cib_model = self.get_kwarg('cib_model', **kwargs)
        # if the sim was saved, load it; otherwise, need to make it:
        fname = self.get_intermediate_sim_fname(component, freq=freq, cib_model=cib_model)
        if os.path.exists(fname): # load the sim
            sim = enmap.read_map(fname)
        else: # need to generate the sim
            if component in ['tsz', 'ksz', 'kappa']: # diffuse components
                # reproject full-sky healpix to CAR for a patch of sky:
                sim = self.cutout_car_patch_from_fullsky_healpix(component, freq=freq, save=save)
            else: # generate maps of discrete components from the catalog:
                # if the sims are being saved, it is quicker to loop 
                # through the catalog once and generate sims for all 
                # frequencies simultaneously ; if they're not being saved,
                # then it's quicker to just make a single sim:
                freqs = self.freqs if save else [freq]
                sim = self.generate_intermediate_point_source_maps(component, freqs=freqs, 
                                                                   save=save, cib_model=cib_model)[freq]
        return sim
        
        


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
            The name of the map component. The allowed values are `'ksz'`,
            `'tsz'`, `'cib'`, `'radio'`, or `'kappa'` for the kSZ, tSZ,
            CIB, radio, or lensing convergence maps, respectively.
        freq : int, optional
            The frequency (in GHz) of the map; must be provided for
            frequency-dependent components. The allowed frequencies are
            30, 90, 148, 219, 277, or 350 GHz.
        bin_dl : bool, default=False
            If `bin_dl=True`, the power spectra are multiplied by a factor
            of `ell * (ell + 1) / (2 * pi)` at each multipole `ell`. 
        save : bool, default=False
            Whether to save the power spectra.
        save_sim : bool, default=False
            Whether to save the simulation.
            
        Returns
        -------
        ells, power_spectrum : array_like of float
            One-dimensional arrays holding the binned multipoles and the
            binned power spectrum of the simulation, respectively.
        
        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) to use. The default is given by the 
            `cib_model` attribute defined during initialization.
        """
        # if the power was saved, load it; otherwise, calculate it:
        fname = self.get_intermediate_sim_power_fname(component, freq=freq, bin_dl=bin_dl, **kwargs)
        if os.path.exists(fname):
            lbin, binned_spectrum = np.loadtxt(fname, unpack=True)
        else:
            sim = self.get_intermediate_sim(component, freq=freq, save=save_sim, **kwargs)
            lbin, binned_spectrum = self.take_intermediate_map_power(sim, bin_dl=bin_dl)
            if save:
                np.savetxt(fname, np.column_stack([lbin, binned_spectrum]))
        return lbin, binned_spectrum
    
    
    
    
    def take_intermediate_map_power(self, imap, bin_dl=False):
        """Return the power spectrum of a map at the S10 resolution on a
        patch of the sky.
        
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
        # for the s10 sims, the kappa map has a lower resolution than the
        # others, so we usually use the `component` name to determine the
        # resolution of the map. here we don't know which `component` we
        # have, so check the resolution of the `imap`: 
        res = maps.get_map_resolution(imap.shape, imap.wcs)
        if np.isclose(res, si.s10_res):
            component = 'tsz' # specific name doesn't matter
        elif np.isclose(res, si.s10_kappa_res):
            component = 'kappa'
        else:
            raise ValueError("The resolution of your map (in arcminutes) must be the same as the S10 sims, which have"
                             f" healpix `nside` parameters of either {si.s10_nside} (approximately"
                             f" {round(si.s10_res,2)} arcminute resolution) or {si.s10_kappa_nside} (for the lensing"
                             f" convergence map; approximately {round(si.s10_kappa_res,2)} arcminute resolution)."
                             f" The exact resolution can be obtained using the `healpy.nside2resol` function.")
        # prepare to take power of the map:
        lmax = self.get_intermediate_map_power_lmax(component)
        mbb_inv = self.get_intermediate_inv_mcm(component, bin_dl=bin_dl)
        window = self.get_intermediate_map_apod_window(component, width=self.width, height=self.height, 
                                                       apod_width=self.apod_width)
        sim = enmap.project(imap.copy(), window.shape, window.wcs) # cut out inner region of map, if necessary
        # take its power:
        self.infomsg(f"taking power of the intermediate lower-resolution map")
        t = time.time()
        sim_power = simpower.take_sim_power(sim, window, lmax, self.binning_file(), mbb_inv, 
                                            bin_dl=bin_dl, deconvolve_pixwin=False)
        lbin = sim_power['ells']
        binned_spectrum = sim_power['tt']
        self.infomsg(f"{utils.tmsg(time.time() - t)} to take power")
        return sim_power['ells'], sim_power['tt'] # no polarization here
    
    
    
    def apodize_intermediate_map(self, imap):
        """Apodize an intermediate, lower-resolution map.
        
        If the map has the same area as the area defined by the `width`
        and `height` attributes, it will be apodized over a region with a
        width given by the `apod_width` attribute along each edge.
        Otherwise, the `map_apod_width` attribute will be used instead.
        
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
        # for the s10 sims, the kappa map has a lower resolution than the
        # others, so we usually use the `component` name to determine the
        # resolution of the map. here we don't know which `component` we
        # have, so check the resolution of the `imap`: 
        imap_res = maps.get_map_resolution(imap.shape, imap.wcs)
        if np.isclose(imap_res, si.s10_res):
            component = 'tsz' # specific name doesn't matter
            res = si.s10_res
        elif np.isclose(imap_res, si.s10_kappa_res):
            component = 'kappa'
            res = si.s10_kappa_res
        else:
            raise ValueError("The resolution of your map (in arcminutes) must be the same as the S10 sims, which have"
                             f" healpix `nside` parameters of either {si.s10_nside} (approximately"
                             f" {round(si.s10_res,2)} arcminute resolution) or {si.s10_kappa_nside} (for the lensing"
                             f" convergence map; approximately {round(si.s10_kappa_res,2)} arcminute resolution)."
                             f" The exact resolution can be obtained using the `healpy.nside2resol` function.")
        # get the width and height of the map:
        _, _, imap_width, imap_height = maps.get_map_ctr_extent(imap.shape, imap.wcs)
        if abs(utils.deg2arcmin(imap_width - self.width)) < imap_res:
            width = self.width
        elif abs(utils.deg2arcmin(imap_width - self.padded2x_width)) < imap_res:
            width = self.padded2x_width
        else:
            width = imap_width
        if abs(utils.deg2arcmin(imap_height - self.height)) < imap_res:
            height = self.height
        elif abs(utils.deg2arcmin(imap_height - self.padded2x_height)) < imap_res:
            height = self.padded2x_height
        else:
            height = imap_height
        # apodization width depends on the width and height:
        if (width > self.width) or (height > self.height): 
            apod_width = self.map_apod_width # for maps used to generate the HD sims
        else:
            apod_width = self.apod_width # for power spectra
        window = self.get_intermediate_map_apod_window(component, width=width, height=height, apod_width=apod_width)
        return imap * window
        
        
    def get_lowres_sim_component_name(self, component, **kwargs):
        """Return a name used as a label for each map component in file
        names.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'ksz'`,
            `'tsz'`, `'cib'`, `'radio'`, or `'kappa'` for the kSZ, tSZ,
            CIB, radio, or lensing convergence maps, respectively.

        Returns
        -------
        component_name : str
            A short name for the map `component` that will be used in file
            names.

        Other Parameters
        ----------------
        **kwargs : dict
            If `component='cib'`, you may pass the name of a `cib_model`
            (`str` or `None`) used. The default is given by the
            `cib_model` attribute defined during initialization.
        """
        component = simutils.validate_sim_component_name(component, valid_components=si.s10_sim_components)
        component_name = component
        if component == 'cib':
            cib_model = validate_cib_model_name(self.get_kwarg('cib_model', **kwargs))
            if cib_model == si.s10_cib_model_name:
                component_name = f'cib_{cib_model}model'
            elif cib_model != si.baseline_cib_model_name:
                component_name = f'cib_{cib_model}'
        return component_name

