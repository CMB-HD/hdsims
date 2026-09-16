"""Calculate power spectra of the simulations."""

import os
import time
import camb
import numpy as np
from scipy import interpolate
from pixell import enmap
from . import utils, siminfo as si, simpower, simutils, hdsimsgen, maps


class HDSimsSpectra(hdsimsgen.HDSimsMaps):
    """Measure the power spectrum of the ultrahigh-resolution simulations.

    The simulations include the lensing convergence map, or temperature and
    polarization maps of the CMB and extragalactic foregrounds, optionally
    convolved with the CMB-HD instrumental beam and/or including white noise
    at the CMB-HD instrumental noise level.

    The measured power spectra are binned and corrected for the
    mode-coupling that is induced due to the apodization window applied to
    the maps before taking their power.

    Attributes
    ----------
    lmax : int
    bin_edges : array_like of int or array_like of float
    bin_info : str or None

    See Also
    --------
    pspy : Calculates the power spectra of simulations.

    Notes
    -----
    Attributes listed above without a description will have the same value
    as the corresponding parameter passed when initializing the class, or
    the default value if it is not passed. Inherited attributes are not
    listed above.

    The `lmax` attribute will be added to the dictionary of
    `default_kwargs` (defined in the `hdsims.simutils.Sims` class).
    """

    def __init__(self, hd_sims_dir, lowres_sims_dir=None,
                 freqs=si.freqs, components=si.components,
                 ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr, width=si.width, height=si.height,
                 apod_width=si.apod_width, map_apod_width=None, res=si.hd_res,
                 cmb_seed=si.cmb_seed, pol=True, noise_seeds=si.noise_seeds,
                 lmax=si.lmax4spectra, bin_edges=None, bin_info=None,
                 lmax4alms=si.lmax4alms, lmax4theo=si.lmax4theo,
                 verbose=False, log=None, make_output_dirs=True, **kwargs):
        """Initialize the `HDSimsSpectra` class for a given patch on the sky.

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

        Notes
        -----
        If you would like to use non-default binning, you must initially
        pass both an array of `bin_edges` and the `bin_info` describing
        the binning. The array of `bin_edges` will be saved under the
        `hd_sims_dir` to a file that contains the `bin_info` in its name.
        The next time you initialize this class, you may just pass the
        same `bin_info` (and leave `bin_edges=None`), and your bin edges
        will be automatically loaded. If both `bin_edges=None` and
        `bin_info=None`, the default uniform binning will be used.
        """
        super().__init__(hd_sims_dir, lowres_sims_dir=lowres_sims_dir, freqs=freqs, components=components,
                         ra_ctr=ra_ctr, dec_ctr=dec_ctr, width=width, height=height,
                         apod_width=apod_width, map_apod_width=map_apod_width, res=res,
                         cmb_seed=cmb_seed, pol=pol, noise_seeds=noise_seeds, lmax4alms=lmax4alms, lmax4theo=lmax4theo,
                         verbose=verbose, log=log, make_output_dirs=make_output_dirs, **kwargs)

        self.lmax = int(round(lmax))
        self.default_kwargs['lmax'] = self.lmax

        self.bin_info = bin_info
        if bin_edges is not None:
            self.bin_edges = bin_edges.copy()
            if self.bin_info is None:
                raise ValueError(f"You passed an array of `bin_edges` but `{bin_info = }`; "
                                 "you must also pass a string to `bin_info` to use in the file names of "
                                 "the binning files and binned power spectra. Note that `bin_info` "
                                 "cannot contain any special characters. When using the default "
                                 "binning (by leaving `bin_edges=None`), the `bin_info` is not used.")
            self.save_bin_edges()
        elif bin_info is not None: # binning file must already be saved
            if not os.path.exists(self.bin_edges_fname()):
                raise ValueError(f"You passed `{bin_info = }` but `{bin_edges = }`. "
                                 "To use the default binning, pass `bin_info=None` and `bin_edges=None`. "
                                 "Otherwise, you must pass an array of `bin_edges` along with `bin_info` "
                                 "to save the binning file. Once the binning file has been saved, you no "
                                 "longer need to pass the `bin_edges` (but you must still pass `bin_info` "
                                 "to use those bin edges).")
            self.bin_edges = self.load_bin_edges()
        else:
            self.bin_edges = simutils.load_default_bin_edges()
            self.save_bin_edges()


    def binning_file(self, save=True):
        """Returns the path to the binning file used to bin the power
        spectra of the simulations.
        """
        fname = simutils.get_binning_file_name(bin_info=self.bin_info, binning_dir=self.binning_dir())
        if save and not os.path.exists(fname):
            simpower.save_binning_file(fname, bin_edges=self.bin_edges)
        return fname


    def bin_edges_fname(self, save=True):
        """Return the path to the file of the bin edges."""
        fname_root = 'bin_edges'
        if self.bin_info is not None:
            fname_root = f'{fname_root}_{self.bin_info}'
        fname = os.path.join(self.binning_dir(), f'{fname_root}.txt')
        return fname


    def save_bin_edges(self):
        """Save the bin edges."""
        if os.path.exists(self.binning_dir()) and not os.path.exists(self.bin_edges_fname()):
            np.savetxt(self.bin_edges_fname(), self.bin_edges)


    def load_bin_edges(self):
        """Load the bin edges."""
        return np.loadtxt(self.bin_edges_fname())


    def _patch_info_for_spectra(self, include_ra_ctr=True, **kwargs):
        """Return a string describing a patch of sky that is smaller than
        the patch of sky defined during initialization.

        Parameters
        ----------
        include_ra_ctr : bool, default=True
            Include information about the R.A. coordinate of the map
            center in the returned `patch_info` string.
        **kwargs : dict
            Optional keyword arguments used to specify the location and
            size of a patch of sky. By default, the `shape` and `wcs`
            attributes defined during initialization are used (so the
            returned `patch_info` will be `None`).
            Otherwise, the options are:
            (1) The `width` and `height` (`int` or `float`) of the patch
                of sky (centered at R.A. and dec. given by the `ra_ctr`
                and `dec_ctr` attributes); or
            (2) A `shape` (`tuple` of `int`) and `wcs` (instance of
                `astropy.wcs.wcs.WCS`) pair to define the geometry of the
                patch of sky.

        Returns
        -------
        patch_info : str or None
            A string describing the location and size of the patch of sky,
            or `None` if the map geometry is the same as the geometry
            defined during initialization.

        Notes
        -----
        The string is used in filenames for the power spectra (and the
        mode-decouling matrices used to calculate the spectra) of maps
        cut out from the region defined during initialization.

        The option `include_ra_ctr` is provided because the pixel size
        does not change with R.A., so the mode-decoupling matrices only
        depend on the map size and center dec. (and apodization width,
        binning file, etc).
        """
        kwargs = self.get_kwargs_with_defaults(**kwargs)
        shape = kwargs['shape']
        wcs = kwargs['wcs']
        if not maps.map_geometry_is_equal(shape, wcs, self.shape, self.wcs):
            if not maps.map_resolution_is_equal(shape, wcs, self.shape, self.wcs):
                res = maps.get_map_resolution(shape, wcs)
                raise ValueError(f"All maps must have the same resolution of {simutils.round_str(self.res)} arcminutes;"
                                 f" the map with `{shape = }` and `{wcs = }` has a resolution of"
                                 f" {simutils.round_str(res)} arcminutes")
            ra_ctr, dec_ctr, width, height = maps.get_map_ctr_extent(shape, wcs)
            size_info = f'{simutils.round_str(width,n=3)}x{simutils.round_str(height,n=3)}deg'
            if include_ra_ctr:
                ctr_info = f'ra{simutils.round_str(ra_ctr,n=3)}dec{simutils.round_str(dec_ctr,n=3)}'
            else:
                ctr_info = f'dec{simutils.round_str(dec_ctr,n=3)}'
            patch_info = f'{size_info}_{ctr_info}'
        else:
            patch_info = None
        return patch_info


    def get_mode_coupling_fnames(self, bin_dl=False, beam=False, freq=None, **kwargs):
        """Return the file name(s) of the inverse mode-coupling matrix
        and the corresponding binning matrix.

        Parameters
        ----------
        bin_dl : bool, default=False
            Whether the power spectra being binned should be multiplied
            by a factor of  `ell * (ell + 1) / (2 pi)` at each multipole
            `ell`.
        beam : bool, default=False
            Whether the inverse mode-coupling matrix should also correct
            the power spectra for the effect of the beam. If `beam=True`,
            you must also pass the `freq`.
        freq : int or None, default=None
            The map frequency (GHz). Must be passed if `beam=True`.

        Returns
        -------
        mcm_fname, bbl_fname : str or dict of str
            If `pol=False`, `mcm_fname` and `bbl_fname` are the file names
            of the inverse mode-coupling matrix and the corresponding
            binning matrix, respectively. The mode-coupling matrix in this
            case is applied to the power spectrum of a single map
            (as opposed to T, Q, and U maps).

            If `pol=True`, `mcm_fname` and `bbl_fname` are both
            dictionaries of file names. The keys are `'spin0xspin0`,
            `'spin0xspin2'`,  `'spin2xspin0`, `'spin2xspin2'`, where
            `'spin0'` and `'spin2'` refer to temperature and polarization
            maps, respectively.

        Other Parameters
        ----------------
        **kwargs : dict
            The other optional keyword arguments are:
            - `lmax` (`int`): The maximum multipole to use for the power
              spectra calculations.
            - `pol` (`bool`): Whether the map whose power spectrum is
              being measured has both temperature and polarization.
            - `apod_width` (`float`): The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`): The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            The default values are given by the corresponding attributes
            defined during initialization.

        See Also
        --------
        get_mode_coupling : Calculate or load the mode-coupling and
                            binning matrices.
        hdsims.simpower.load_mode_coupling_files : Load the files
        """
        if beam:
            if freq is None:
                raise ValueError(f"`{beam = }` and `{freq = }`. To correct for the beam,"
                                 " you must pass a frequency (in GHz).")
            freq = simutils.validate_sim_freq(freq)
            beam_fwhm = si.beam_fwhm[freq]
        else:
            beam_fwhm = None
        patch_info = self._patch_info_for_spectra(include_ra_ctr=False, **kwargs)
        if (self.bin_info is not None) or (patch_info is not None):
            fname_info = '_'.join([finfo for finfo in [self.bin_info, patch_info] if (finfo is not None)])
        else:
            fname_info = None
        kwargs = self.get_kwargs_with_defaults(**kwargs)
        mcm_fname, bbl_fname = simutils.get_mode_coupling_fnames(kwargs['apod_width'], kwargs['lmax'], bin_dl=bin_dl,
                                                                 pol=kwargs['pol'], beam_fwhm=beam_fwhm,
                                                                 fname_info=fname_info, binning_dir=self.binning_dir())
        return mcm_fname, bbl_fname


    def calc_mode_coupling(self, bin_dl=False, beam=False, freq=None, **kwargs):
        """Calculate the inverse mode-coupling and corresponding binning
        matrices.

        Parameters
        ----------
        bin_dl : bool, default=False
            Whether the power spectra being binned should be multiplied
            by a factor of  `ell * (ell + 1) / (2 pi)` at each multipole
            `ell`.
        beam : bool, default=False
            Whether the inverse mode-coupling matrix should also correct
            the power spectra for the effect of the beam. If `beam=True`,
            you must also pass the `freq`.
        freq : int or None, default=None
            The map frequency (GHz). Must be passed if `beam=True`.

        Returns
        -------
        mbb_inv, bbl : array_like of float, or dict of array_like of float
            If `pol=False`, `mbb_inv` and `bbl` are the inverse
            mode-coupling matrix and the corresponding binning matrix,
            respectively. The mode-coupling matrix in this case is
            applied to the power spectrum of a single map (as opposed
            to T, Q, and U maps). The mode-coupling matrix has shape
            `(nbin, nbin)` and the binning matrix has shape `(nbin, nl)`,
            where `nbin` and `nl` are the number of bins or multipoles,
            respectively, up to `lmax`.

            If `pol=True`, `mbb_inv` and `bbl` are both dictionaries of
            arrays. The keys are `'spin0xspin0`, `'spin0xspin2'`,
            `'spin2xspin0`, `'spin2xspin2'`, where `'spin0'` and `'spin2'`
            refer to temperature and polarization maps, respectively.

        Other Parameters
        ----------------
        **kwargs : dict
            The other optional keyword arguments are:
            - `lmax` (`int`): The maximum multipole to use for the power
              spectra calculations.
            - `pol` (`bool`): Whether the map whose power spectrum is
              being measured has both temperature and polarization.
            - `apod_width` (`float`): The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`): The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            The default values are given by the corresponding attributes
            defined during initialization.

        See Also
        --------
        pspy.so_mcm.mcm_and_bbl_spin0and2, pspy.so_mcm.mcm_and_bbl_spin0
        hdsims.simpower.calc_mode_coupling
        """
        kwargs = self.get_kwargs_with_defaults(**kwargs)
        window = self.get_apod_window(save=True, **kwargs)
        binning_file = self.binning_file()
        pol = kwargs['pol']
        lmax = kwargs['lmax']
        if beam:
            if freq is None:
                raise ValueError(f"`{beam = }` and `{freq = }`. "
                                 "To correct for the beam, you must pass a frequency (in GHz).")
            freq = simutils.validate_sim_freq(freq)
            beam_fwhm = si.beam_fwhm[freq]
        else:
            beam_fwhm = None
        self.infomsg(f"calculating inv. mode-coupling matrix for {lmax = }, {beam_fwhm = }, {pol = }, {bin_dl = }")
        t = time.time()
        mbb_inv, bbl = simpower.calc_mode_coupling(window, lmax, binning_file, bin_dl=bin_dl, beam_fwhm=beam_fwhm, pol=pol)
        self.infomsg(f"{utils.tmsg(time.time() - t)} for inv. mode-coupling matrix")
        return mbb_inv, bbl


    def get_mode_coupling(self, bin_dl=False, beam=False, freq=None, binning_matrix=True, **kwargs):
        """Load or calculate the inverse mode-coupling and corresponding
        binning matrices.

        Parameters
        ----------
        bin_dl : bool, default=False
            Whether the power spectra being binned should be multiplied
            by a factor of  `ell * (ell + 1) / (2 pi)` at each multipole
            `ell`.
        beam : bool, default=False
            Whether the inverse mode-coupling matrix should also correct
            the power spectra for the effect of the beam. If `beam=True`,
            you must also pass the `freq`.
        freq : int or None, default=None
            The map frequency (GHz). Must be passed if `beam=True`.
        binning_matrix : bool, default=True
            Whether to also return the binning matrix.
            If `binning_matrix=False`, only the inverse mode-coupling
            matrix is returned.

        Returns
        -------
        mbb_inv : array_like of float, or dict of array_like of float
            If `pol=False`, `mbb_inv` is the inverse mode-coupling matrix
            with shape `(nbin, nbin)` where `nbin` is the number of bins
            up to `lmax`. The mode-coupling matrix in this case is applied
            to the power spectrum of a single map (as opposed  to T, Q, and
            U maps).
            If `pol=True`, `mbb_inv` is a dictionary of inverse
            mode-coupling matrices.  The keys are `'spin0xspin0`,
            `'spin0xspin2'`,  `'spin2xspin0`, `'spin2xspin2'`, where
            `'spin0'` and `'spin2'` refer to temperature and polarization
            maps, respectively.
        bbl : array_like of float, or dict of array_like of float
            The binning matrix (if `pol=False`) or dictionary of binning
            matrices (if `pol=True`) with the same keys as `mbb_inv`.
            Each binning matrix has a shape `(nbin, lmax)`.
            Only returned if `binning_matrix=True`.

        Other Parameters
        ----------------
        **kwargs : dict
            The other optional keyword arguments are:
            - `lmax` (`int`): The maximum multipole to use for the power
              spectra calculations.
            - `pol` (`bool`): Whether the map whose power spectrum is
              being measured has both temperature and polarization.
            - `apod_width` (`float`): The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`): The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            The default values are given by the corresponding attributes
            defined during initialization.

        See Also
        --------
        pspy.so_mcm.mcm_and_bbl_spin0and2, pspy.so_mcm.mcm_and_bbl_spin0
        calc_mode_coupling, hdsims.simpower.calc_mode_coupling

        Notes
        -----
        The binning matrix will always be saved along with the inverse
        mode-coupling matrix.
        """
        mcm_fname, bbl_fname = self.get_mode_coupling_fnames(bin_dl=bin_dl, beam=beam, freq=freq, **kwargs)
        # check if files exist:
        if self.get_kwarg('pol', **kwargs):
            mcm_saved = all([os.path.exists(mcm_fname[key]) for key in mcm_fname])
            bbl_saved = all([os.path.exists(bbl_fname[key]) for key in bbl_fname])
        else:
            mcm_saved = os.path.exists(mcm_fname)
            bbl_saved = os.path.exists(bbl_fname)
        files_saved = mcm_saved and bbl_saved
        if files_saved:
            mbb_inv, bbl = simpower.load_mode_coupling_files(mcm_fname, bbl_fname=bbl_fname)
        else:
            mbb_inv, bbl = self.calc_mode_coupling(bin_dl=bin_dl, beam=beam, freq=freq, **kwargs)
            simpower.save_mode_coupling_files(mbb_inv, mcm_fname, bbl=bbl, bbl_fname=bbl_fname)
        if binning_matrix:
            return mbb_inv, bbl
        else:
            return mbb_inv


    def take_power(self, imap, pixwin=True, beam=False, freq=None, bin_cl=True, bin_dl=False, **kwargs):
        """Take the power of a map and return binned power spectra that
        have been corrected for the effect of apodization and, optionally,
        the beam convoluion.

        Parameters
        ----------
        imap : pixell.enmap.ndmap
            The input map. It must have the same resolution and be located
            on the same patch of the sky (or a patch containing the inner
            region with area given by the `width` and `height` attributes)
            as the ultrahigh-resolution simulations.
        pixwin : bool, default=True
            Whether the map has been convolved with the CAR pixel window
            function. If `pixwin=True`, the pixel window will be
            deconvolved from the map before taking its power.
        beam : bool, default=False
            Whether the map has been convolved with a beam.
            If `beam=True`, the inverse mode-coupling matrix will also
            correct the power spectra for the effect of the beam. In this
            case, the `freq` must also be passed.
        freq : int or None, default=None
            The map frequency (GHz). Must be passed if `beam=True`.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether to return power spectra that have been multiplied
            by a factor of  `ell * (ell + 1) / (2 pi)` at each multipole
            `ell`.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization).
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if the `imap` includes both temperature and
                polarization (T, Q, U maps).

        Other Parameters
        ----------------
        **kwargs : dict
            The other optional keyword arguments are:
            - `lmax` (`int`): The maximum multipole to use for the power
              spectra calculations.
            - `pol` (`bool`): Whether the map whose power spectrum is
              being measured has both temperature and polarization.
            - `apod_width` (`float`): The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`): The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            The default values are given by the corresponding attributes
            defined during initialization.

        See Also
        --------
        hdsims.simpower.calc_sim_power
        pspy.so_spectra.get_spectra, pspy.so_spectra.bin_spectra

        Notes
        -----
        The inverse mode-coupling matrix (or matrices) will be calculated
        if necessary. This may use a lot of memory or take a long time,
        depending on how much memory is available.

        The units of the power spectrum depends on the units of the `imap`.
        """
        if not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")
        lmax = int(round(self.get_kwarg('lmax', **kwargs)))
        has_pol = len(imap.shape) > 2
        window = self.get_apod_window(save=True, **kwargs)
        mbb_inv_dict = {}
        spec_types = []
        if bin_cl:
            spec_types.append('cl')
            mbb_inv_dict['cl'] = self.get_mode_coupling(bin_dl=False, beam=beam, freq=freq,
                                                        binning_matrix=False, **kwargs)
            if (not has_pol) and ('spin0xspin0' in mbb_inv_dict['cl']):
                mbb_inv_dict['cl'] = mbb_inv_dict['cl']['spin0xspin0']
        if bin_dl:
            spec_types.append('dl')
            mbb_inv_dict['dl'] = self.get_mode_coupling(bin_dl=True, beam=beam, freq=freq,
                                                        binning_matrix=False, **kwargs)
            if (not has_pol) and ('spin0xspin0' in mbb_inv_dict['dl']):
                mbb_inv_dict['dl'] = mbb_inv_dict['dl']['spin0xspin0']
        imap = enmap.project(imap.copy(), window.shape, window.wcs)
        sim_power = simpower.calc_sim_power(imap, window, lmax, self.binning_file(), mbb_inv_dict,
                                            bin_cl=bin_cl, bin_dl=bin_dl, deconvolve_pixwin=pixwin)
        return sim_power


    def get_signal_sim_power_fname(self, component, freq=None, bin_dl=False, **kwargs):
        """Return the path to the file where the power spectrum of a
        signal-only simulation is saved.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'tsz'`,
            `'ksz'`, `'cib'`, `'radio'`, `'kappa'`, `'cmb'`, or
            `'unlensed_cmb'` for the tSZ, kSZ, CIB, radio, lensing
            convergence, lensed CMB, or unlensed CMB map(s), respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If the `component` is `'tsz'`,
            `'cib'`, or `'radio'`, the frequency must be `30`, `90`,
            `148`, `219`, `277`, or `350`. Otherwise, the `freq` is not
            used.
        bin_dl : bool, default=False
            Whether the power spectra have been multiplied by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        fname : str
            The path to the file.

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the
            `get_lowres_sim_component_name` method.

            The other keyword arguments are:
            - `lmax` (`int`) : The maximum multipole used when calculating
              the power spectrum.
            - `apod_width` (`float`): The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`): The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            - If the `component` is `'cmb'` or `'unlensed_cmb'`, you may
              also pass keyword arguments for `cmb_seed` (`int`) and
              `pol` (`bool`).

            The defaults are given by the corresponding attributes, defined
            during initialization.

        Notes
        -----
        For the lensing convergence simulation (`component='kappa'`),
        `bin_dl` is always `False`.
        """
        component = simutils.validate_sim_component_name(component)
        lmax = int(round(self.get_kwarg('lmax', **kwargs)))
        component_info = self.get_component_name(component, **kwargs)
        patch_info = self._patch_info_for_spectra(**kwargs)
        apod_width = self.get_kwarg('apod_width', **kwargs)
        spec_type = 'dl' if bin_dl else 'cl'
        fname_info = []
        # map frequency and compoments:
        if simutils.has_freq_dependent_component(component):
            freq = simutils.validate_sim_freq(freq)
            fname_info.append(f'{freq:03d}')
        fname_info.append(component_info)
        # map geometry:
        if patch_info is not None:
            fname_info.append(patch_info)
        # apodization, binning, etc:
        if not np.isclose(apod_width, self.apod_width):
            fname_info.append(f'apod{simutils.round_str(apod_width)}deg')
        if lmax != self.lmax:
            fname_info.append(f"lmax{lmax}")
        if self.bin_info is not None:
            fname_info.append(self.bin_info)
        fname_root = '_'.join(fname_info)
        fname = os.path.join(self.spectra_dir(), f'{fname_root}_{spec_type}s.txt')
        return fname


    def load_signal_sim_power(self, component, freq=None, bin_cl=True, bin_dl=False, **kwargs):
        """Load the power spectra of a signal-only simulation, if it has
        been saved.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'tsz'`,
            `'ksz'`, `'cib'`, `'radio'`, `'kappa'`, `'cmb'`, or
            `'unlensed_cmb'` for the tSZ, kSZ, CIB, radio, lensing
            convergence, lensed CMB, or unlensed CMB map(s), respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If the `component` is `'tsz'`,
            `'cib'`, or `'radio'`, the frequency must be `30`, `90`,
            `148`, `219`, `277`, or `350`. Otherwise, the `freq` is not
            used.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether the power spectra have been multiplied by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization). Only returned if
                the map `component` is not `'kappa'`.
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if the `component` is either `'cmb'` or
                `'unlensed_cmb'` and `pol=True`.
            - `'clkk'` : The power spectrum of the lensing convergence map
                (only returned for `component='kappa'`).

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the
            `get_signal_sim_power_fname` method.

        See Also
        --------
        get_signal_sim_power : Try to load the power spectra before
                               calculating it.

        Notes
        -----
        For the lensing convergence simulation (`component='kappa'`),
        `bin_dl` is always `False`.
        """
        component = simutils.validate_sim_component_name(component)
        if component == 'kappa':
            bin_cl = True
            bin_dl = False
        elif not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")

        pol = self.get_kwarg('pol', **kwargs)
        cols = simutils.get_spectra_keys(component, pol=pol)
        spec_types = [] # cl or dl
        fnames = {} # cl and dl saved separately
        if bin_cl:
            spec_types.append('cl')
            fnames['cl'] = self.get_signal_sim_power_fname(component, freq=freq, bin_dl=False, **kwargs)
        if bin_dl:
            spec_types.append('dl')
            fnames['dl'] = self.get_signal_sim_power_fname(component, freq=freq, bin_dl=True, **kwargs)

        sim_power = {}
        for spec_type in spec_types:
            # for T-only CMB, if the file doesn't exist, check if file for
            # TQU was saved ; if so, load that file & save a T-only copy:
            if (not os.path.exists(fnames[spec_type])) and simutils.has_cmb(component) and (not pol):
                kwargs_with_pol = {**kwargs, 'pol': True}
                fname_with_pol = self.get_signal_sim_power_fname(component, freq=freq,
                                                                 bin_dl=('dl' in spec_type), **kwargs_with_pol)
                if os.path.exists(fname_with_pol):
                    cols_with_pol = simutils.get_spectra_keys(component, pol=True)
                    tqu_sim_power = utils.load_dict_from_file(fname_with_pol, cols_with_pol)
                    utils.save_dict_to_file(fnames[spec_type], tqu_sim_power, keys=cols)
            # otherwise, just try to load the file:
            sim_spectra = utils.load_dict_from_file(fnames[spec_type], cols)
            sim_power['ells'] = sim_spectra['ells']
            for key in cols[1:]: # re-name keys to include `'cl'` or `'dl'`
                sim_power[f'{spec_type}{key}'] = sim_spectra[key].copy()
        return sim_power


    def take_signal_sim_power(self, component, freq=None, bin_cl=True, bin_dl=False, save=True, **kwargs):
        """Calculate the power spectra of a signal-only simulation.

        Note that this calculation will always be done, regardless of
        whether the power spectra of the same map has already been saved.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'tsz'`,
            `'ksz'`, `'cib'`, `'radio'`, `'kappa'`, `'cmb'`, or
            `'unlensed_cmb'` for the tSZ, kSZ, CIB, radio, lensing
            convergence, lensed CMB, or unlensed CMB map(s), respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If the `component` is `'tsz'`,
            `'cib'`, or `'radio'`, the frequency must be `30`, `90`,
            `148`, `219`, `277`, or `350`. Otherwise, the `freq` is not
            used.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether the power spectra have been multiplied by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.
        save : bool, default=True
            Whether to save the power spectra.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization). Only returned if
                the map `component` is not `'kappa'`.
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if the `component` is either `'cmb'` or
                `'unlensed_cmb'` and `pol=True`.
            - `'clkk'` : The power spectrum of the lensing convergence map
                (only returned for `component='kappa'`).

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the `take_power`,
            `get_signal_sim`, and `get_signal_sim_power_fname` methods.

        See Also
        --------
        get_signal_sim_power : Try to load the power spectra before
                               calculating it.

        Notes
        -----
        For the lensing convergence simulation (`component='kappa'`),
        `bin_dl` is always `False`.
        """
        component = simutils.validate_sim_component_name(component)
        if component == 'kappa':
            bin_cl = True
            bin_dl = False
        elif not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")

        spec_types = [] # cl or dl
        fnames = {} # save cl and dl separately
        if bin_cl:
            spec_types.append('cl')
            fnames['cl'] = self.get_signal_sim_power_fname(component, freq=freq, bin_dl=False, **kwargs)
        if bin_dl:
            spec_types.append('dl')
            fnames['dl'] = self.get_signal_sim_power_fname(component, freq=freq, bin_dl=True, **kwargs)

        # load the sim & cut out inner region :
        sim = enmap.project(self.get_signal_sim(component, freq=freq, **kwargs), self.shape, self.wcs)
        # take its power:
        self.infomsg(f'taking power of {component} sim for {freq = }')
        t = time.time()
        sim_power = self.take_power(sim, pixwin=simutils.has_pixwin(component), bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        if component == 'kappa':
            # rename the key for lensing convergence power spectrum
            sim_power = {'ells': sim_power['ells'], 'clkk': sim_power['cltt']}
        self.infomsg(f'{utils.tmsg(time.time() - t)} to take power')

        if save:
            pol = self.get_kwarg('pol', **kwargs)
            for spec_type in spec_types:
                col_names = simutils.get_spectra_keys(component, pol=pol)
                keys = simutils.get_spectra_keys(component, cl=('cl' in spec_type), dl=('dl' in spec_type), pol=pol)
                utils.save_dict_to_file(fnames[spec_type], sim_power, keys=keys, col_names=col_names)
                self.infomsg(f"saved {fnames[spec_type]}")

        return sim_power


    def get_signal_sim_power(self, component, freq=None, bin_cl=True, bin_dl=False, save=True, **kwargs):
        """Return the power spectra of a signal-only simulation.

        The power spectra will be binned and corrected for the effect of
        apodization. The power spectrum of the lensing convergence
        simulation is dimensionless; otherwise, the power spectra have
        units of uK^2.

        Parameters
        ----------
        component : str
            The name of the map component. The allowed values are `'tsz'`,
            `'ksz'`, `'cib'`, `'radio'`, `'kappa'`, `'cmb'`, or
            `'unlensed_cmb'` for the tSZ, kSZ, CIB, radio, lensing
            convergence, lensed CMB, or unlensed CMB map(s), respectively.
        freq : int or None, default=None
            The map frequency (in GHz). If the `component` is `'tsz'`,
            `'cib'`, or `'radio'`, the frequency must be `30`, `90`,
            `148`, `219`, `277`, or `350`. Otherwise, the `freq` is not
            used.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether the power spectra have been multiplied by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.
        save : bool, default=True
            Whether to save the power spectra.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization). Only returned if
                the map `component` is not `'kappa'`.
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if the `component` is either `'cmb'` or
                `'unlensed_cmb'` and `pol=True`.
            - `'clkk'` : The power spectrum of the lensing convergence map
                (only returned for `component='kappa'`).

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the `take_power`,
            `get_signal_sim`, and `get_signal_sim_power_fname` methods.

            In addition to any keyword arguments for the initial set of
            full-sky, lower-resolution simulations (see
            `hdsims.lowres_sims.LowResSims` and its derived classes), the
            recognized keyword arguments are:
            - `lmax` (`int`) : The maximum multipole used when calculating
              the power spectrum.
            - `apod_width` (`float`): The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`): The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            - If the `component` is `'cmb'` or `'unlensed_cmb'`, you may
              also pass keyword arguments for `cmb_seed` (`int`) and
              `pol` (`bool`).
            The defaults are given by the corresponding attributes, defined
            during initialization.

        See Also
        --------
        get_signal_sim : The signal-only map.

        Notes
        -----
        For the lensing convergence simulation (`component='kappa'`),
        `bin_dl` is always `False`.
        """
        try:
            sim_power = self.load_signal_sim_power(component, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        except FileNotFoundError:
            sim_power = self.take_signal_sim_power(component, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, save=save, **kwargs)
        return sim_power


    def get_sim_power_fname(self, freq=None, beam=False, noise=False, bin_dl=False, **kwargs):
        """Return the path to the file where the power spectrum of the
        simulation is saved.

        Parameters
        ----------
        freq : int or None, default=None
            The map frequency (GHz). If (1) the map contains the tSZ
            (`'tsz'`), CIB (`'cib'`), or radio galaxies (`'radio'`),
            (2) the map is convolved with the beam (`beam=True`), or
            (3) noise has been added (`noise=True`), then the `freq`
            must be `30`, `90`, `148`, `219`, `277`, or `350`.
            Otherwise, the `freq` is not used.
        beam : bool, default=False
            Whether the simulation was convolved with a Gaussian beam
            for CMB-HD. If `beam=True`, the `freq` must be passed.
        noise : bool, default=False
            Whether white noise for CMB-HD has been added to the map.
            If `noise=True`, the `freq` must be passed.
        bin_dl : bool, default=False
            Whether the power spectra have been multiplied by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        fname : str
            The path to the file.

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the `get_component_name`
            method, including:
            - Any keyword arguments for the initial set of full-sky,
              lower-resolution simulations (passed to
              `get_lowres_sim_component_name`).
            - `cmb_seed` (`int`) and `pol` (`bool`) for the CMB map(s).
              The defaults are given by the corresponding attributes.

            The other allowed keyword arguments are:
            - A list of individual `components` (`list` of `str`) in the
              map. The default is given by the `map_components` attribute.
            - Either a single `noise_seed` (`int`) or a dictionary of
              `noise_seeds` at each frequency. Only used if `noise=True`.
              The default is given by the `noise_seeds` (using the `freq`
              as the key) attribute defined during initialization.
            - `lmax` (`int`) : The maximum multipole used when calculating
              the power spectra.
            - `apod_width` (`float`) : The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`) : The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).

        Notes
        -----
        For the lensing convergence simulation (`components=['kappa']`),
        `bin_dl` is always `False`.
        """
        components = self._get_map_components(**kwargs)
        if (len(components) == 1) and (not (beam or noise)):
            # this is just the power of signal-only, single-component map:
            fname = self.get_signal_sim_power_fname(components[0], freq=freq, bin_dl=bin_dl, **kwargs)
        else:
            lmax = int(round(self.get_kwarg('lmax', **kwargs)))
            patch_info = self._patch_info_for_spectra(**kwargs)
            apod_width = self.get_kwarg('apod_width', **kwargs)
            fname_info = []
            # check if we need the frequency; if so, add it to file name:
            if simutils.has_freq_dependent_component(components) or beam or noise:
                freq = simutils.validate_sim_freq(freq)
                fname_info.append(f'{freq:03d}')
            # add info about the map components:
            fname_info.append(self._map_component_list2str(**kwargs))
            # add other info about the map & its power:
            if beam:
                fname_info.append(f'beam{simutils.round_str(si.beam_fwhm[freq])}arcmin')
            if noise:
                noise_seed = self._get_noise_seed(freq, **kwargs)
                fname_info.append(f'noise{simutils.round_str(si.noise_level[freq])}uKarcmin{noise_seed:04d}')
            if patch_info is not None:
                fname_info.append(patch_info)
            if not np.isclose(apod_width, self.apod_width):
                fname_info.append(f'apod{simutils.round_str(apod_width)}deg')
            if lmax != self.lmax:
                fname_info.append(f"lmax{lmax}")
            if self.bin_info is not None:
                fname_info.append(self.bin_info)
            spec_type = 'dl' if bin_dl else 'cl'
            fname_root = '_'.join(fname_info)
            fname = os.path.join(self.spectra_dir(), f'{fname_root}_{spec_type}s.txt')
        return fname


    def load_sim_power(self, freq=None, beam=False, noise=False, bin_cl=True, bin_dl=False, **kwargs):
        """Load the power spectra of the simulation, if it has been saved.

        Parameters
        ----------
        freq : int or None, default=None
            The map frequency (GHz). If (1) the map contains the tSZ
            (`'tsz'`), CIB (`'cib'`), or radio galaxies (`'radio'`),
            (2) the map is convolved with the beam (`beam=True`), or
            (3) noise has been added (`noise=True`), then the `freq`
            must be `30`, `90`, `148`, `219`, `277`, or `350`.
            Otherwise, the `freq` is not used.
        beam : bool, default=False
            Whether the simulation was convolved with a Gaussian beam
            for CMB-HD. If `beam=True`, the `freq` must be passed.
        noise : bool, default=False
            Whether white noise for CMB-HD has been added to the map.
            If `noise=True`, the `freq` must be passed.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether the power spectra have been multiplied by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization). Only returned if
                `'kappa'` is not the only component in the map.
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if either `'cmb'` or  `'unlensed_cmb'` is in
                the list of map `components` and `pol=True`.
            - `'clkk'` : The power spectrum of the lensing convergence map
                (only returned for `components=['kappa']`).

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the `get_sim_power_fname`
            method.

        See Also
        --------
        get_sim_power :
            Try to load the power spectra before calculating it.

        Notes
        -----
        For the lensing convergence simulation (`components=['kappa']`),
        `bin_dl` is always `False`.
        """
        components = self._get_map_components(**kwargs)
        if 'kappa' in components:
            bin_cl = True
            bin_dl = False
        elif not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")

        pol = self.get_kwarg('pol', **kwargs)
        spec_types = [] # cl or dl
        fnames = {} # cl and dl saved separately
        cols = simutils.get_spectra_keys(components, pol=pol)
        if bin_cl:
            spec_types.append('cl')
            fnames['cl'] = self.get_sim_power_fname(freq=freq, beam=beam, noise=noise, bin_dl=False, **kwargs)
        if bin_dl:
            spec_types.append('dl')
            fnames['dl'] = self.get_sim_power_fname(freq=freq, beam=beam, noise=noise, bin_dl=True, **kwargs)

        sim_power = {}
        for spec_type in spec_types:
            # for T-only maps w/ CMB, check if file for TQU was saved;
            # if so, load that file & save a T-only copy:
            if (not os.path.exists(fnames[spec_type])) and simutils.has_cmb(components) and (not pol):
                kwargs_with_pol = {**kwargs, 'pol': True}
                fname_with_pol = self.get_sim_power_fname(freq=freq, beam=beam, noise=noise,
                                                          bin_dl=('dl' in spec_type), **kwargs_with_pol)
                if os.path.exists(fname_with_pol):
                    cols_with_pol = simutils.get_spectra_keys(components, pol=True)
                    tqu_sim_power = utils.load_dict_from_file(fname_with_pol, cols_with_pol)
                    utils.save_dict_to_file(fnames[spec_type], tqu_sim_power, keys=cols)
            sim_spectra = utils.load_dict_from_file(fnames[spec_type], cols)
            sim_power['ells'] = sim_spectra['ells']
            for key in cols[1:]: # update key names
                sim_power[f'{spec_type}{key}'] = sim_spectra[key].copy()

        return sim_power


    def take_sim_power(self, freq=None, beam=False, noise=False, bin_cl=True, bin_dl=False,
                       save=True, save_sim=False, **kwargs):
        """Calculate the power spectra of the simulation.

        Note that this calculation will always be done, regardless of
        whether the power spectra of the same map has already been saved.

        Parameters
        ----------
        freq : int or None, default=None
            The map frequency (GHz). If (1) the map contains the tSZ
            (`'tsz'`), CIB (`'cib'`), or radio galaxies (`'radio'`),
            (2) the map is convolved with the beam (`beam=True`), or
            (3) noise has been added (`noise=True`), then the `freq`
            must be `30`, `90`, `148`, `219`, `277`, or `350`.
            Otherwise, the `freq` is not used.
        beam : bool, default=False
            Whether the simulation was convolved with a Gaussian beam
            for CMB-HD. If `beam=True`, the `freq` must be passed.
        noise : bool, default=False
            Whether white noise for CMB-HD has been added to the map.
            If `noise=True`, the `freq` must be passed.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.
        save : bool, default=True
            Whether to save the power spectra.
        save_sim : bool, default=False
            Whether to save the map.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization). Only returned if
                `'kappa'` is not the only component in the map.
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if either `'cmb'` or  `'unlensed_cmb'` is in
                the list of map `components` and `pol=True`.
            - `'clkk'` : The power spectrum of the lensing convergence map
                (only returned for `components=['kappa']`).

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the `get_sim`,
            `take_power`, and `get_sim_power_fname` methods.

        See Also
        --------
        get_sim_power :
            Try to load the power spectra before calculating it.

        Notes
        -----
        For the lensing convergence simulation (`components=['kappa']`),
        `bin_dl` is always `False`.
        """
        components = self._get_map_components(**kwargs)
        if 'kappa' in components:
            bin_cl = True
            bin_dl = False
        elif not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")

        spec_types = [] # cl or dl
        fnames = {} # save cl and dl separately
        if bin_cl:
            spec_types.append('cl')
            fnames['cl'] = self.get_sim_power_fname(freq=freq, beam=beam, noise=noise, bin_dl=False, **kwargs)
        if bin_dl:
            spec_types.append('dl')
            fnames['dl'] = self.get_sim_power_fname(freq=freq, beam=beam, noise=noise, bin_dl=True, **kwargs)

        sim = self.get_sim(freq=freq, beam=beam, noise=noise, save=save_sim, **kwargs)
        self.infomsg(f'taking power of sim for {freq = }, {components = }, {beam = }, {noise = }')
        t = time.time()
        sim_power = self.take_power(sim, pixwin=simutils.has_pixwin(components),
                                    beam=beam, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        if 'kappa' in components:
            # rename the key for lensing convergence power spectrum:
            sim_power = {'ells': sim_power['ells'], 'clkk': sim_power['cltt']}
        self.infomsg(f'{utils.tmsg(time.time() - t)} to take power')

        if save:
            for spec_type in spec_types:
                pol = self.get_kwarg('pol', **kwargs)
                col_names = simutils.get_spectra_keys(components, pol=pol)
                keys = simutils.get_spectra_keys(components, cl=('cl' in spec_type), dl=('dl' in spec_type), pol=pol)
                utils.save_dict_to_file(fnames[spec_type], sim_power, keys=keys, col_names=col_names)
                self.infomsg(f"saved {fnames[spec_type]}")

        return sim_power


    def get_sim_power(self, freq=None, beam=False, noise=False, bin_cl=True, bin_dl=False,
                      save=True, save_sim=False, **kwargs):
        """Return the power spectra of the simulation.

        The power spectra will be binned and corrected for the effect of
        apodization. The power spectrum of the lensing convergence
        simulation is dimensionless; otherwise, the power spectra have
        units of uK^2.

        Parameters
        ----------
        freq : int or None, default=None
            The map frequency (GHz). If (1) the map contains the tSZ
            (`'tsz'`), CIB (`'cib'`), or radio galaxies (`'radio'`),
            (2) the map is convolved with the beam (`beam=True`), or
            (3) noise has been added (`noise=True`), then the `freq`
            must be `30`, `90`, `148`, `219`, `277`, or `350`.
            Otherwise, the `freq` is not used.
        beam : bool, default=False
            Whether the simulation was convolved with a Gaussian beam
            for CMB-HD. If `beam=True`, the `freq` must be passed.
        noise : bool, default=False
            Whether white noise for CMB-HD has been added to the map.
            If `noise=True`, the `freq` must be passed.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.
        save : bool, default=True
            Whether to save the power spectra.
        save_sim : bool, default=False
            Whether to save the map.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization). Only returned if
                `'kappa'` is not the only component in the map.
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if either `'cmb'` or  `'unlensed_cmb'` is in
                the list of map `components` and `pol=True`.
            - `'clkk'` : The power spectrum of the lensing convergence map
                (only returned for `components=['kappa']`).

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the `get_sim`,
            `take_power`, and `get_sim_power_fname` methods.

            In addition to any keyword arguments for the initial set of
            full-sky, lower-resolution simulations (see
            `hdsims.lowres_sims.LowResSims` and its derived classes), the
            recognized keyword arguments are:

            - A list of individual `components` (`list` of `str`) in the
              map. The default is given by the `map_components` attribute.
            - `lmax` (`int`) : The maximum multipole used when calculating
              the power spectra.
            - `apod_width` (`float`) : The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`) : The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            - If the map contains the CMB (either `'cmb'` or `'unlensed_cmb'`
              are in the list of `components`), you may also pass keyword
              arguments for `cmb_seed` (`int`) and `pol` (`bool`).
            - Either a single `noise_seed` (`int`) or a dictionary of
              `noise_seeds` at each frequency. Only used if `noise=True`.
              The default is given by the `noise_seeds` (using the `freq`
              as the key) attribute defined during initialization.

            The defaults are given by the corresponding attributes defined
            during initialization.

        See Also
        --------
        get_sim : The simulated map.
        take_power : Measure the power spectrum of a map.

        Notes
        -----
        For the lensing convergence simulation (`components=['kappa']`),
        `bin_dl` is always `False`.
        """
        try:
            sim_power = self.load_sim_power(freq=freq, beam=beam, noise=noise, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        except FileNotFoundError:
            sim_power = self.take_sim_power(freq=freq, beam=beam, noise=noise, bin_cl=bin_cl, bin_dl=bin_dl,
                                            save=save, save_sim=save_sim, **kwargs)
        return sim_power


    def get_noise_sim_power_fname(self, freq, pixwin=True, beam=False, bin_dl=False, **kwargs):
        """Return the path to the file where the power spectrum of the
        white noise simulation is saved.

        Parameters
        ----------
        freq : int
            The map frequency (GHz). Must be `30`, `90`, `148`, `219`,
            `277`, or `350`.
        pixwin : bool, default=True
            Whether to deconvolve the pixel window function before
            calculating the power spectra.
        beam : bool, default=False
            Whether to correct the power spectra for the effect of the
            beam.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        fname : str
            The file name.

        Other Parameters
        ----------------
        **kwargs : dict
            The optional keyword arguments are:
            - Either a single `noise_seed` (`int`) or a dictionary of
              `noise_seeds` at each frequency.
            - `pol` (`bool`) : If `pol=True`, the noise simulation
              includes both temperature and polarization (T, Q, U) maps.
              Otherwise it only includes the temperature map.
            - `lmax` (`int`) : The maximum multipole used when calculating
              the power spectra.
            - `apod_width` (`float`) : The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`) : The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            The defaults are given by the corresponding attributes, defined
            during initialization.
        """
        freq = simutils.validate_sim_freq(freq)
        spec_type = 'dl' if bin_dl else 'cl'
        lmax = int(round(self.get_kwarg('lmax', **kwargs)))
        noise_seed = self._get_noise_seed(freq, **kwargs)
        apod_width = self.get_kwarg('apod_width', **kwargs)
        patch_info = self._patch_info_for_spectra(**kwargs)
        pol_info = 'TQU' if self.get_kwarg('pol', **kwargs) else 'T'
        noise_info = f'noise{simutils.round_str(si.noise_level[freq])}uKarcmin{noise_seed:04d}{pol_info}'
        fname_parts = [f'{freq:03d}', noise_info]
        if pixwin or beam:
            fname_parts.append('deconvolved')
            if pixwin:
                fname_parts.append('pixwin')
            if beam:
                fname_parts.append(f'beam{simutils.round_str(si.beam_fwhm[freq])}arcmin')
        if patch_info is not None:
            fname_parts.append(patch_info)
        if not np.isclose(apod_width, self.apod_width):
            fname_parts.append(f'apod{simutils.round_str(apod_width)}deg')
        if lmax != self.lmax:
            fname_parts.append(f"lmax{lmax}")
        if self.bin_info is not None:
            fname_parts.append(self.bin_info)
        fname_root = '_'.join(fname_parts)
        fname = os.path.join(self.spectra_dir(), f'{fname_root}_{spec_type}s.txt')
        return fname


    def load_noise_sim_power(self, freq, pixwin=True, beam=False, bin_cl=True, bin_dl=False, **kwargs):
        """Load the power spectra of the noise simulation, if it has been
        saved.

        Parameters
        ----------
        freq : int
            The map frequency (GHz). Must be `30`, `90`, `148`, `219`,
            `277`, or `350`.
        pixwin : bool, default=True
            Whether to deconvolve the pixel window function before
            calculating the power spectra.
        beam : bool, default=False
            Whether to correct the power spectra for the effect of the
            beam.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization).
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if `pol=True`.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword arguments passed to
            `get_noise_sim_power_fname`.

        See Also
        --------
        get_noise_sim_power :
            Try to load the power spectra before calculating it.
        """
        pol = self.get_kwarg('pol', **kwargs)
        if not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")

        spec_types = [] # cl or dl
        fnames = {} # cl and dl saved separately
        cols = simutils.get_spectra_keys(['cmb'], pol=pol) # same as cmb
        if bin_cl:
            spec_types.append('cl')
            fnames['cl'] = self.get_noise_sim_power_fname(freq, pixwin=pixwin, beam=beam, bin_dl=False, **kwargs)
        if bin_dl:
            spec_types.append('dl')
            fnames['dl'] = self.get_noise_sim_power_fname(freq, pixwin=pixwin, beam=beam, bin_dl=True, **kwargs)

        sim_power = {}
        for spec_type in spec_types:
            # for T-only maps, check if file for TQU was saved ;
            # if so, load that file & save a T-only copy:
            if (not os.path.exists(fnames[spec_type])) and (not pol):
                kwargs_with_pol = {**kwargs, 'pol': True}
                fname_with_pol = self.get_noise_sim_power_fname(freq, pixwin=pixwin, beam=beam,
                                                                bin_dl=('dl' in spec_type), **kwargs_with_pol)
                if os.path.exists(fname_with_pol):
                    cols_with_pol = simutils.get_spectra_keys(['cmb'], pol=True)
                    tqu_sim_power = utils.load_dict_from_file(fname_with_pol, cols_with_pol)
                    utils.save_dict_to_file(fnames[spec_type], tqu_sim_power, keys=cols)
            sim_spectra = utils.load_dict_from_file(fnames[spec_type], cols)
            sim_power['ells'] = sim_spectra['ells']
            for key in cols[1:]:
                sim_power[f'{spec_type}{key}'] = sim_spectra[key].copy()
        return sim_power


    def take_noise_sim_power(self, freq, pixwin=True, beam=False, bin_cl=True, bin_dl=False,
                             save=True, save_sim=False, **kwargs):
        """Calculate the power spectra of the noise simulation.

        Note that this calculation will always be done, regardless of
        whether the power spectra of the same map has already been saved.

        Parameters
        ----------
        freq : int
            The map frequency (GHz). Must be `30`, `90`, `148`, `219`,
            `277`, or `350`.
        pixwin : bool, default=True
            Whether to deconvolve the pixel window function before
            calculating the power spectra.
        beam : bool, default=False
            Whether to correct the power spectra for the effect of the
            beam.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.
        save : bool, default=True
            Whether to save the power spectra.
        save_sim : bool, default=False
            Whether to save the map.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization).
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if `pol=True`.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword arguments passed to `get_noise_sim`,
            `take_power`, and `get_noise_sim_power_fname`.

        See Also
        --------
        get_noise_sim_power :
            Try to load the power spectra before calculating it.
        """
        if not (bin_cl or bin_dl):
            raise ValueError(f"`{bin_cl = }` and `{bin_dl = }`. At least one of `bin_cl` or `bin_dl` must be `True`.")
        spec_types = [] # cl or dl
        fnames = {} # cl and dl saved separately
        if bin_cl:
            spec_types.append('cl')
            fnames['cl'] = self.get_noise_sim_power_fname(freq, pixwin=pixwin, beam=beam, bin_dl=False, **kwargs)
        if bin_dl:
            spec_types.append('dl')
            fnames['dl'] = self.get_noise_sim_power_fname(freq, pixwin=pixwin, beam=beam, bin_dl=True, **kwargs)

        sim = self.get_noise_sim(freq, save=save_sim, **kwargs)
        self.infomsg(f"taking power of {freq} GHz noise sim")
        t = time.time()
        sim_power = self.take_power(sim, pixwin=pixwin, beam=beam, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        self.infomsg(f'{utils.tmsg(time.time() - t)} to take power')

        if save:
            for spec_type in spec_types:
                pol = self.get_kwarg('pol', **kwargs)
                col_names = simutils.get_spectra_keys(['cmb'], pol=pol)
                keys = simutils.get_spectra_keys(['cmb'], cl=('cl' in spec_type), dl=('dl' in spec_type), pol=pol)
                utils.save_dict_to_file(fnames[spec_type], sim_power, keys=keys, col_names=col_names)
                self.infomsg(f"saved {fnames[spec_type]}")
        return sim_power


    def get_noise_sim_power(self, freq, pixwin=True, beam=False, bin_cl=True, bin_dl=False,
                            save=True, save_sim=False, **kwargs):
        """Return the power spectra of the noise simulation.

        The power spectra (units of uK^2) will be binned and corrected for
        the effect of apodization (and optionally a beam; see the 'Notes'
        section below).

        Parameters
        ----------
        freq : int
            The map frequency (GHz). Must be `30`, `90`, `148`, `219`,
            `277`, or `350`.
        pixwin : bool, default=True
            Whether to deconvolve the pixel window function before
            calculating the power spectra.
        beam : bool, default=False
            Whether to correct the power spectra for the effect of the
            beam.
        bin_cl : bool, default=True
            Whether to return power spectra that have not been multiplied
            by any factors.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.
        save : bool, default=True
            Whether to save the power spectra.
        save_sim : bool, default=False
            Whether to save the map.

        Returns
        -------
        sim_power : dict of array_like of float
            A dictionary of the binned power spectra with the following
            keys and values:
            - `'ells'`: array of the binned multipoles.
            - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`):
                The TT power spectrum (i.e., the power spectrum of a
                single map without any polarization).
            - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
              `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`):
                The TE, EE, and BB power spectra, respectively. Only
                calculated if `pol=True`.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword arguments are passed to `get_noise_sim`,
            `take_power`, and `get_noise_sim_power_fname`:
            - Either a single `noise_seed` (`int`) or a dictionary of
              `noise_seeds` at each frequency.
            - `pol` (`bool`) : If `pol=True`, the noise simulation
              includes both temperature and polarization (T, Q, U) maps.
              Otherwise it only includes the temperature map.
            - `lmax` (`int`) : The maximum multipole used when calculating
              the power spectra.
            - `apod_width` (`float`) : The width (in degrees) of the region
              along each edge of the map that will be apodized before
              calculating its power spectrum.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
              `astropy.wcs.wcs.WCS`) : The geometry of the patch of sky
              used for power spectra (e.g., if taking power of a map cut
              out from the larger simulated region).
            The defaults are given by the corresponding attributes, defined
            during initialization.

        See Also
        --------
        get_noise_sim : The noise simulation.
        take_power : Calculate power spectra of a map.

        Notes
        -----
        The noise simulations are never convolved with a pixel window
        or a beam, but the map the noise is added to may have been; that
        map would then be pixel-window-deconvolved before taking its
        power, and its power spectrum would be corrected for the beam.
        To mimic the effect of this procedure on the power spectrum of
        the noise map, you may pass  `pixwin=True` and/or `beam=True`.
        These parameters are named to be consistent with the parameter
        names in the `take_power` method.
        """
        try:
            sim_power = self.load_noise_sim_power(freq, pixwin=pixwin, beam=beam, bin_cl=bin_cl, bin_dl=bin_dl, **kwargs)
        except FileNotFoundError:
            sim_power = self.take_noise_sim_power(freq, pixwin=pixwin, beam=beam, bin_cl=bin_cl, bin_dl=bin_dl,
                                                  save=save, save_sim=save_sim, **kwargs)
        return sim_power


    def bin_theory(self, theory_dict, bin_dl=False, **kwargs):
        """Bin a set of theory power spectra using the same binning that
        was used to bin the power spectra of the simulations.

        Parameters
        ----------
        theory_dict : dict of array_like of float
            A dictionary of the theory power spectra. The power spectra
            should have a value for each integer multipole from `2` up
            to the maximum multipole `lmax`. The power spectra should
            not be multiplied by any multipole factors.
            - The dictionary must have a key `'ells'` for the array of
              multipoles.
            - It must also have either a key `'tt'` or `'kk'` for the
              temperature (TT) or lensing convergence (kappa) power
              spectrum, respectively.
            - If `pol=True`, it may also have keys `'te'`, `'ee'`, `'bb'`
              for the TE, EE, and BB power spectra. You must either
              provide all of these spectra, or none of them (e.g., `'bb'`
              must also be included if `'te'` and `'ee'` are).
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        binned_theory_dict : dict of array_like of float
            A dictionary of binned power spectra, with the same keys as
            the input `theory_dict`.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword arguments passed to `get_mode_coupling`.

        See Also
        --------
        hdsims.simpower.bin_theory
        pspy.so_mcm.apply_Bbl

        Notes
        -----
        If the input theory spectra begin at a multipole `ell < 2` or
        end at a multipole `ell > lmax`, the arrays will be trimmed before
        binning them.
        """
        _, bbl = self.get_mode_coupling(bin_dl=bin_dl, binning_matrix=True, **kwargs)
        bin_edges = self.load_bin_edges()
        lmax = int(round(self.get_kwarg('lmax', **kwargs)))
        if not self.get_kwarg('pol', **kwargs):
            # only pass one theory power spectrum (temperature or lensing)
            theo = {key: theory for (key, theory) in theory_dict.items() if (key not in ['te', 'ee', 'bb'])}
        binned_theory_dict = simpower.bin_theory(theory_dict, bbl, bin_edges, bin_dl=bin_dl, lmax=lmax)
        return binned_theory_dict


    def calc_lensed_cmb_sim_theory(self, save=True, kappa_sim_power_Lmin=30, **kwargs):
        """Calculate the lensed CMB theory power spectra for the specific
        realization of the lensing convergence field on the simulated
        patch of sky.

        Note that the power spectra will always be calculated, even if it has
        already been saved.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the power spectra.

        Returns
        -------
        lensed_theory : dict of array_like of float
            A dictionary of the theory lensed CMB power spectra with the
            following keys and values:
            - `'ells'`: array of each integer multipole between zero and
                  the maximum given by the `lmax4theo` attribute.
            - `'tt'`, `'te'`, `'ee'`, `'bb'`: lensed CMB TT, TE, EE, and
                  BB power spectra (in units of uK^2 as C_ell's, i.e.
                  without any multiplicative multipole factors).

        Other Parameters
        ----------------
        kappa_sim_power_Lmin : int, default=30
            The minimum multpole of the measured lensing convergence
            simulation power spectrum used in the calculation.
        **kwargs : dict
            Optional keyword arguments for the lower-resolution, full-sky
            simulations passed to the `get_sim_smallscale_theory` and
            `get_signal_sim_power` methods.

        See Also
        --------
        get_signal_sim :
            The lensed CMB or lensing convergence simulation(s).
        get_sim_theory :
            Load the theory power spectra, or calculate it if necessary.
        camb.results.CAMBdata.get_lensed_cls_with_spectrum :
            Calculate lensed CMB power spectra from a given lensing
            convergence power spectrum.

        Notes
        -----
        The lensed CMB simulation is generated by lensing the unlensed
        CMB map(s) by the lensing convergence map. The lensed CMB theory
        power spectra are calculated by CAMB, using the same cosmology and
        accuracy as the unlensed CMB theory (used to generate the unlensed
        CMB simulation), but using the measured power spectrum of the
        lensing convergence simulation to do the lensing, as opposed to
        the CAMB theory lensing spectrum. Unlike the theory kSZ, unlensed
        CMB, or lensing convergence power spectra, the lensed CMB theory
        spectra are not used to generate the lensed CMB simulations.

        Since the lensing convergence simulation power spectrum is not
        measured on the full sky, it is only used down to a minimum
        multipole `kappa_sim_power_Lmin`; since it is binned, we
        interpolate it to each multipole between `kappa_sim_power_Lmin`
        and the maximum given by the `lmax` attribute. Below or above the
        bin centers corresponding to `kappa_sim_power_Lmin` or `lmax`,
        respectively, we the theory lensing convergence power spectrum
        that was fit to the power in the simulation is used.
        """
        self.infomsg('calculating lensed CMB sim theory')
        t = time.time()

        # get the CAMB params:
        pars = self.get_cambparams_for_sim()

        # take power of the kappa map
        kappa_power_kwargs = {**kwargs, 'shape': self.shape, 'wcs': self.wcs, 'apod_width': self.apod_width}
        kappa_power = self.get_signal_sim_power('kappa', save=save, **kappa_power_kwargs)
        lbin = kappa_power['ells']
        sim_clkk = kappa_power['clkk']
        # on scales that are not measured by the kappa sim power spectrum,
        # use the theory power spectrum that was fit to the kappa sim:
        ells, camb_clkk = self.get_sim_smallscale_theory('kappa', **kwargs)

        # we will use the power in the kappa map to lens the unlensed
        # theory cmb spectra, so we need a clkk curve at each multipole,
        # up to CAMB's lmax for the calculation; because we're working on
        # a patch of sky, we'll use the theory clkk fit to the sim for
        # L < `kappa_sim_power_Lmin` and L > `lmax`:
        sim_Lbin_to_interp = lbin[lbin >= kappa_sim_power_Lmin]
        sim_clkk_to_interp = sim_clkk[lbin >= kappa_sim_power_Lmin]
        sim_Lmin_interp = round(sim_Lbin_to_interp[0]) # use theory below this L
        sim_Lmax_interp = round(sim_Lbin_to_interp[-1]) + 1 # use theory above this L
        theo_loc = np.where(ells > sim_Lmax_interp)
        # interpolate to each L:
        Ls_to_interp = np.concatenate([ells[:sim_Lmin_interp - 1], sim_Lbin_to_interp, ells[theo_loc]])
        clkk_to_interp = np.concatenate([camb_clkk[:sim_Lmin_interp - 1], sim_clkk_to_interp, camb_clkk[theo_loc]])
        camb_ells = np.arange(pars.max_l+1) # CAMB needs a clkk curve to higher Lmax than it will output
        clkk_interp = interpolate.interp1d(Ls_to_interp, clkk_to_interp,
                                           bounds_error=False, fill_value=clkk_to_interp[-1])(camb_ells)

        # get the CAMB results instance:
        self.infomsg('getting CAMB results')
        results = camb.get_results(pars)
        # calculate the lensed CMB theory:
        lensed_powers = results.get_lensed_cls_with_spectrum(clkk_interp * 4 / (2 * np.pi),
                                                             lmax=self.lmax4theo, CMB_unit='muK', raw_cl=True)
        lensed_theory = {'ells': ells.copy()}
        for i, s in enumerate(['tt', 'ee', 'bb', 'te']):
            lensed_theory[s] = lensed_powers[:,i].copy()
            lensed_theory[s][:2] = 0

        if save:
            fname = self.get_sim_theory_fname('cmb')
            extra_header_info = ("spectra are C_ell (no multiplicative ell-factors) in uK^2")
            utils.save_dict_to_file(fname, lensed_theory, keys=si.spectra_col_names[:-1],
                                    extra_header_info=extra_header_info)
            self.infomsg(f'saved {fname}')

        self.infomsg(f'{utils.tmsg(time.time() - t)} for lensed CMB sim theory')
        return lensed_theory



    def get_binned_sim_theory_fname(self, component, bin_dl=False, **kwargs):
        """Return the path to the binned theory power spectra.

        Parameters
        ----------
        component : str
            The name of the map component corresponding to the theory. The
            `component` must be one of `'ksz'`, `'kappa'`, `'cmb'`, or
            `'unlensed_cmb'`.
        bin_dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`.

        Returns
        -------
        fname : str
            The path to the binned theory spectra.

        Other Parameters
        ----------------
        **kwargs : dict
            The additional optional keyword argument is `lmax` (`int`), the
            maximum multipole used when calculating the binning matrix. The
            default is given by the `lmax` attribute defined during
            initialization.
        """
        component = simutils.validate_sim_component_name(component, valid_components=self.theo_components)
        bin_dl = False if (component == 'kappa') else bin_dl
        spec_type = 'dl' if bin_dl else 'cl'
        lmax = int(round(self.get_kwarg('lmax', **kwargs)))
        unbinned_fname = self.get_sim_theory_fname(component)
        fname_info = f'binned_{spec_type}_lmax{lmax}'
        if self.bin_info is not None:
            fname_info = f'{fname_info}_{self.bin_info}'
        fname = unbinned_fname.replace('.txt', f'_{fname_info}.txt')
        return fname


    def get_sim_theory(self, component, binned=False, dl=False, save_intermediate_maps=False, **kwargs):
        """Return the binned or unbinned theory power spectra for the
        simulations.

        Parameters
        ----------
        component : str
            The name of the map component corresponding to the theory. The
            `component` must be one of `'ksz'`, `'kappa'`, `'cmb'`, or
            `'unlensed_cmb'`.
        binned : bool, default=False
            Whether to bin the power spectra.
        dl : bool, default=False
            Whether to multiply the power spectra by a factor of
            `ell * (ell + 1) / (2 pi)` at each multipole `ell`. This is
            ignored if the `component` is `'kappa'`.

        Returns
        -------
        theory : dict of array_like of float
            A dictionary of the theory power spectra with the following
            keys are values:
            - `'ells'` : The multipoles. If `binned=False`, the array has
                elements for each (integer) multipole between zero and the
                maximum given by the `lmax4theo` attribute.
                If `binned=True`, the array has elements for each bin
                center up to the bin corresponding to the maximum given by
                the `lmax` attribute.
            - `'cltt'` (if `dl=False`) or `'dltt'` (if `dl=True`): The TT
                power spectrum (units of uK^2). Only returned if the
                `component` is `'ksz'`, `'cmb'`, or `'unlensed_cmb'`.
            - `'clte'`, `'clee'`, `'clbb'` (if `dl=False`) or
              `'dlte'`, `'dlee'`, `'dlbb'` (if `dl=True`): The CMB TE, EE,
                and BB power spectra (units of uK^2). Only returned if the
                `component` is `'cmb'`, or `'unlensed_cmb'`.
            - `'clkk'`: The (dimensionless) lensing convergence power
                spectrum. Only returned if the `component` is `'kappa'`.

        Other Parameters
        ----------------
        save_intermediate_maps : bool, default=False
            Whether to save the intermediate lower-resolution maps if they
            are needed to calculate the theory power spectra.
        **kwargs : dict
            Optional keyword arguments passed to `bin_theory`,
            `get_binned_sim_theory_fname`, `calc_lensed_cmb_sim_theory`,
            and `get_sim_smallscale_theory`:
            - Any keyword arguments for the lower-resolution, full-sky
              simulations accepted by `get_intermediate_sim_power`.
            - `lmax` (`int`) : The maximum multipole used when calculating
              the binning matrix.
            The `pol`, `apod_width`, `shape`, and `wcs` keyword arguments
            (used by the `bin_theory` method) will be ignored if passed.

        Notes
        -----
        The unlensed CMB theory power spectra are calculated by CAMB and
        used to generate the unlensed CMB simulation.

        The kSZ and lensing convergence theory power spectra are obtained
        by fitting parameters that are used scale a template power
        spectrum. The parameters are fit to the large-scale power spectrum
        of the simulation. On small scales, we use a Gaussian realization
        of the theory kSZ or lensing convergence power spectrum to
        generate the corresponding simulation. The lensing convergence
        template power spectrum is calculated by CAMB, using the same
        cosmology and accuracy as the unlensed CMB theory. See 
        arXiv:2609.16128 for details about the kSZ template power spectrum.

        The lensed CMB theory power spectra are also calculated by CAMB,
        using the same cosmology and accuracy as the unlensed CMB theory,
        but using the measured power spectrum of the lensing convergence
        simulation to do the lensing, as opposed to the CAMB theory
        lensing spectrum. Unlike the other theory power spectra, the
        lensed CMB theory spectra are not used to generate the lensed CMB
        simulations.

        The kSZ, lensing convergence, and lensed CMB theory power spectra
        are unique to a given patch of sky, while the unlensed CMB power
        spectra are fixed for a given set of cosmological and accuracy
        parameters.
        """
        component = simutils.validate_sim_component_name(component, valid_components=self.theo_components)
        dl = False if (component == 'kappa') else dl
        spec_type = 'dl' if dl else 'cl'
        kwargs = {**kwargs, 'shape': self.shape, 'wcs': self.wcs, 'apod_width': self.apod_width}
        if simutils.has_cmb(component) and binned:
            # we want to bin temperaure & polarization theory spectra:
            kwargs['pol'] = True

        # try to load the theory before calculating it:
        if binned:
            fname = self.get_binned_sim_theory_fname(component, bin_dl=dl, **kwargs)
        else:
            fname = self.get_sim_theory_fname(component)

        if os.path.exists(fname):
            if binned:
                theo = utils.load_dict_from_file(fname, simutils.get_spectra_keys(component))
            else:
                theo = self.load_sim_theory(component)

        elif binned and os.path.exists(self.get_sim_theory_fname(component)):
            # load the unbinned theory, bin it, and save it:
            theo = self.load_sim_theory(component)
            theo = self.bin_theory(theo, bin_dl=dl, **kwargs)
            utils.save_dict_to_file(fname, theo, keys=simutils.get_spectra_keys(component))

        else:
            if simutils.has_cmb(component): # lensed or unlensed
                if 'unlensed' in component:
                    theo = self.get_camb_theory()
                else:
                    theo = self.calc_lensed_cmb_sim_theory(save=True, **kwargs)
            else: # ksz or kappa
                theo_kwargs = {**kwargs, 'save_intermediate_maps': save_intermediate_maps}
                theo_key = 'kk' if (component == 'kappa') else 'tt'
                theo = {}
                theo['ells'], theo[theo_key] = self.get_sim_smallscale_theory(component, **theo_kwargs)
            if binned:
                theo = self.bin_theory(theo, bin_dl=dl, **kwargs)
            utils.save_dict_to_file(fname, theo, keys=simutils.get_spectra_keys(component))

        # update key names to include `'cl'` or `'dl'`, and multiply by
        #`ells * (ells + 1) / 2pi` if `dl=True` and theory is not binned:
        theory = {'ells': theo['ells']}
        lfact = theo['ells'] * (theo['ells'] + 1) / (2 * np.pi) if (dl and (not binned)) else 1
        for key in theo.keys():
            if key != 'ells':
                theory[f'{spec_type}{key}'] = theo[key] * lfact

        return theory


    def calculate_hd_sims_powerspectra(self, save_intermediate_map_power=False, save_intermediate_maps=False,  **kwargs):
        """Calculate the power spectra of the ultrahigh-resolution
        signal-only simulations.

        If the power spectra of the kSZ, lensing convergence, lensed CMB,
        or unlensed CMB simulations are calculated, their corresponding
        theory power spectra will also be calculated.

        If necessary, the simulations will be generated, and the inverse
        mode-coupling matrices will be calculated.

        Parameters
        ----------
        save_intermediate_map_power : bool, default=False
            Whether to also calculate and save the power spectra of the
            corresponding lower-resolution maps on the patch of sky.
        save_intermediate_maps : bool, default=False
            Whether to also save the intermediate, lower-resolution
            maps on the patch of sky. Only used if the maps are needed,
            e.g., if `save_intermediate_map_power=True` or if the
            corresponding ultrahigh-resolution simulation needs to be
            generated.

        Other Parameters
        ----------------
        **kwargs : dict
            Optional keyword arguments passed to the
            `get_signal_sim_power`, `get_sim_theory`, and
            `get_intermediate_sim_power` methods.

            In addition to any keyword arguments for the initial set of
            full-sky, lower-resolution simulations (see
            `hdsims.lowres_sims.LowResSims` and its derived classes), the
            recognized keyword arguments are:

            - `components` (`list` of `str`): A list of map components.
                Each `component` in the list must be a valid name that can
                be passed to `get_signal_sim_power`.
            - `freqs` (`list` of `int`): A list of map frequencies (GHz).
                Each `freq` in the list must be a valid frequency that can
                be passed to `get_signal_sim_power`.
            - `pol` (`bool`): Whether to include CMB polarization. Only
                used if `'cmb'` or `'unlensed_cmb'` is in the list of
                `components`.
            - `cmb_seed` (`int`): The random seed used to generate the
                unlensed CMB simulation. Only used if `'cmb'` or
                `'unlensed_cmb'` is in the list of `components`.
            - `lmax` (`int`): The maximum multipole used when calculating
                the power spectra.
            - `apod_width` (`float`) : The width (in degrees) of the
                region along each edge of the map that will be apodized
                before calculating its power spectrum. Ignored when
                calculating theory power spectra.
            - A pair of `shape` (`tuple` of `int`) and `wcs` (instance of
                `astropy.wcs.wcs.WCS`) : The geometry of the patch of sky
                used for power spectra (e.g., if taking power of a map cut
                out from the larger simulated region). Ignored when
                calculating theory power spectra.

            The defaults are given by the corresponding attributes, defined
            during initialization.

        See Also
        --------
        generate_hd_sims : Generate the ultrahigh-resolution simulations.
        get_mode_coupling : The inverse mode-coupling and binning matrices.
        get_sim_theory : Theory power spectra for the simulations.
        get_intermediate_sim, get_intermediate_sim_power :
            The intermediate, lower-resolution map on the patch of sky and
            its power spectrum, respectively.

        Notes
        -----
        It is recommended that you also pass `save_intermediate_maps=True`
        if `save_intermediate_map_power=True`; otherwise, the intermediate
        lower-resolution maps will need to be obtained twice.
        """
        if 'components' in kwargs:
            # if user passed list of components, don't modify it
            components = simutils.validate_sim_component_names(kwargs['components'])
        else:
            # use default list, and add kappa if list includes lensed cmb
            components = self.components.copy()
            if ('cmb' in components) and ('kappa' not in components):
                components.append('kappa')
        freqs = simutils.validate_sim_freqs(self.get_kwarg('freqs', **kwargs))

        self.infomsg(f"getting power spectra of HD sims for {components = }")
        lowres_sim_components = [c for c in components if (c in self.lowres_sim_components)]
        if save_intermediate_map_power and (len(lowres_sim_components) > 0):
            self.infomsg(f"will also get power spectra of the intermediate, lower-resolution"
                         f" sims for {lowres_sim_components = }")
        theo_components = [c for c in components if (c in self.theo_components)]
        if len(theo_components) > 0:
            self.infomsg(f"will also save binned sim theory curves for {theo_components}")

        t_start = time.time()
        for component in components:
            # by default, we always bin cl's; for everything but kappa,
            # also bin dl's (since it's quick to do both):
            bin_dl = False if (component == 'kappa') else True
            sim_freqs = freqs if simutils.has_freq_dependent_component(component) else [None]
            for freq in sim_freqs:
                self.get_signal_sim_power(component, freq=freq, bin_dl=bin_dl, **kwargs)
                if save_intermediate_map_power and (component in self.lowres_sim_components):
                    self.get_intermediate_sim_power(component, freq=freq, bin_dl=bin_dl, save=True,
                                                    save_sim=save_intermediate_maps, **kwargs)
            # also save the binned theory:
            if component in self.theo_components:
                self.get_sim_theory(component, binned=True, dl=bin_dl,
                                    save_intermediate_maps=save_intermediate_maps, **kwargs)
        self.infomsg(f'{utils.tmsg(time.time() - t_start)} total for sim power spectra')


