import camb
import numpy as np
import logging
import os
from pixell import enmap, utils, curvedsky, lensing, powspec
from pspy import sph_tools, so_map, so_window
from scipy.interpolate import interp1d
import camb

class CMB:
    """
    Class for simulating and lensing CMB patches using CAMB theory, pixell, and pspy tools.
    Provides utilities for geometry setup, window/apodization, harmonic transforms, and
    lensing of unlensed CMB maps by kappa.
    """

    #############################################
    
    def __init__(self, ra, dec, final_width, apod_width, 
                 res, l_max, log_level=logging.INFO):
        """
        Initialize a CMB object.
        
        
        Parameters
        ----------
        ra : float
            Right ascension of the patch center in degrees.
        dec : float
            Declination of the patch center in degrees.
        final_width : float
            Desired angular width of the final patch in degrees (applies in both RA and DEC).
        apod_width : float
            Width of the apodization region applied to patch edges, in degrees.
        res : float
            Pixel resolution of final maps, in arcminutes.
        l_max : int
            Maximum multipole moment (ℓ) for spherical harmonic transforms.
        log_level : int, optional
            Logging verbosity level (default: logging.INFO).
        """
        
        self.ra = ra
        self.dec = dec
        self.final_width = final_width
        self.apod_width = apod_width
        self.res = res
        self.l_max = l_max

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        return None

    #############################################

    def enmap2pspy(self, imap):
        """
        Convert a pixell enmap to a pspy so_map.
        
        
        Parameters
        ----------
        imap : enmap
            Input pixell enmap array, with WCS and geometrys.
        
        
        Returns
        -------
        omap : so_map
            Equivalent pspy so_map containing the same data and geometry.
        """
        
        if len(imap.shape) > 2:
            ncomp = imap.shape[-3]
        else:
            ncomp = 1
        omap = so_map.car_template_from_shape_wcs(ncomp, imap.shape, imap.wcs)
        omap.data[:] = imap.copy()
        return omap

    def get_coord_box(self, ra_ctr, dec_ctr, width, height=None):
        """
        Compute coordinate box for a square or rectangular patch.
        
        
        Parameters
        ----------
        ra_ctr : float
            Center RA in degrees.
        dec_ctr : float
            Center DEC in degrees.
        width : float
            Patch width in degrees (RA extent).
        height : float, optional
            Patch height in degrees (DEC extent). If None, set equal to width.
        
        
        Returns
        -------
        coord_box : ndarray of shape (2, 2)
            [[dec_min, ra_max], [dec_max, ra_min]] in radians.
        """
        
        height = width if (height is None) else height
        ra_min = ra_ctr - (width / 2)
        ra_max = ra_ctr + (width / 2)
        dec_min = dec_ctr - (height / 2)
        dec_max = dec_ctr + (height / 2)
        coord_box = np.array([[dec_min, ra_max], [dec_max, ra_min]])
        return np.deg2rad(coord_box)
    
    def get_shape_wcs(self, res, ra_ctr, dec_ctr, width, height=None):
        """
        Compute shape and WCS for a pixell enmap patch.
        
        
        Parameters
        ----------
        res : float
            Pixel resolution in arcminutes.
        ra_ctr : float
            Center RA in degrees.
        dec_ctr : float
            Center DEC in degrees.
        width : float
            Patch width in degrees.
        height : float, optional
            Patch height in degrees. If None, set equal to width.
        
        
        Returns
        -------
        shape : tuple of int
            Shape of the enmap array (ny, nx).
        wcs : WCS
            World coordinate system for the patch.
        """
        
        coord_box = self.get_coord_box(ra_ctr, dec_ctr, width, height=height)
        shape, wcs = enmap.geometry(pos=coord_box, res=res * utils.arcmin, proj='car')
        return shape, wcs
    
    def make_apod_window(self, shape, wcs, apod_width_deg, map_type='pixell'):
        """
        Create an apodization window for a given patch.
        
        
        Parameters
        ----------
        shape : tuple of int
            Shape of the patch array (ny, nx).
        wcs : WCS
            World coordinate system.
        apod_width_deg : float
            Radius of the apodization taper, in degrees.
        map_type : str, optional
            Desired output type: "so_map", "pspy", "pspipe", "pixell", or "enmap". Default "so_map".
        
        
        Returns
        -------
        window : so_map or enmap
            Apodization window array in requested format, with values between 0 and 1.
        """
        
        # options for output map format:
        map_type = map_type.lower()
        options = ['so_map', 'pspy', 'pspipe', 'pixell', 'enmap']
        if map_type not in options:
            errmsg = (f"Invalid `map_type`: '{map_type}'. The options are: {options}.")
        # calculate the window
        window = so_map.car_template_from_shape_wcs(1, shape, wcs)
        window.data[:] = 0
        window.data[1:-1, 1:-1] = 1
        window = so_window.create_apodization(window, apo_type="C1", apo_radius_degree=apod_width_deg)
        if map_type in ['pixell', 'enmap']:
            window = window.data
        return window
    
    def make_unlensed_patch(self, theory_path, cmb_seed):
        """
        Generate an unlensed CMB patch from CAMB theory.
        
        
        Parameters
        ----------
        theory_path : str
            Path to the CAMB theory power spectrum file (.dat).
        cmb_seed : int
            Random seed for reproducibility.
        
        
        Returns
        -------
        cmb : so_map
            Unlensed 3-component (T, Q, U) CMB patch.
        """

        paddedwidth = self.final_width + 2*self.apod_width
        self.logger.info(f"making unlensed CMB sim for {round(self.res,2)}' {round(paddedwidth,2)} x {round(paddedwidth,2)} deg patch with seed = {cmb_seed}")
        cmb = so_map.car_template(3, self.ra-paddedwidth/2, self.ra+paddedwidth/2, self.dec-paddedwidth/2, self.dec+paddedwidth/2, self.res)
        cl_file = os.path.join(theory_path)
        cmb.data = curvedsky.rand_map(cmb.data.shape, cmb.data.wcs, powspec.read_spectrum(cl_file)[: 3, : 3], seed = cmb_seed)
        return cmb

    #############################################

    def kappa_to_phi(self, kappa_alm, kappa_ainfo=None):
        """
        Convert kappa alm to phi alm.
        
        
        Parameters
        ----------
        kappa_alm : ndarray
            Convergence alm array for kappa.
        kappa_ainfo : optional
            Alm info object from pixell. Default is None.
        
        
        Returns
        -------
        oalm : ndarray
            Lensing potential alm array.
        """
        
        oalm = curvedsky.almxfl(
            alm=kappa_alm,
            lfilter=lambda x: 2. / (x * (x + 1)),
            ainfo=kappa_ainfo
        )
        oalm[~np.isfinite(oalm)] = 0
        return oalm

    def save_CMB_alms(self, cmb_unlensed, kappa_map, apodize_for_alms):
        """
        Compute spherical harmonic alms for unlensed CMB components and phi.
        
        
        Parameters
        ----------
        cmb_unlensed : so_map
            Unlensed CMB patch (3 components: T, Q, U).
        kappa_map : so_map
            Kappa map aligned with the patch.
        apodize_for_alms : bool
            Whether to apply apodization before alm computation.
        
        
        Returns
        -------
        alms_unlensed : list of ndarray
            List of alms for T, Q, U.
        alms_phi : ndarray
            Phi alms computed from kappa.
        """
        
        padded_width = self.final_width + 2*self.apod_width
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, padded_width, height=padded_width)
        if apodize_for_alms:
            window_apod = self.make_apod_window(shape, wcs, self.apod_width, map_type='pixell')
        else:
            window_apod = self.enmap2pspy(enmap.ones(shape, wcs)).data
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        lmax_raw = min(self.l_max,int(window_ones.get_lmax_limit()))

        alms_unlensed = []
        for type_unlensed in [0,1,2]:
            self.logger.info(f"Computing CMB-{type_unlensed} alms")
            unlensed_map = enmap.project(cmb_unlensed.data[type_unlensed], shape, wcs) * window_apod
            alms_unlensed.append(sph_tools.get_alms(self.enmap2pspy(unlensed_map), 
                                                    window_ones, niter=0, lmax=lmax_raw))

        self.logger.info(f"Computing Kappa alms")
        kappa_map.data = enmap.project(kappa_map.data, shape, wcs) * window_apod
        alms_kappa = sph_tools.get_alms(kappa_map, window_ones, niter=0, lmax=lmax_raw)
        self.logger.info(f"Computing Phi alms")
        alms_phi = self.kappa_to_phi(alms_kappa)
        
        return alms_unlensed, alms_phi

    def get_innerPatch(self, patch):
        """
        Extract the central region of a patch with the desired final width.
        
        
        Parameters
        ----------
        patch : so_map
            Patch of width larger than self.final_width, at resolution self.new_res.
        
        
        Returns
        -------
        inner_patch : so_map
            Central square patch of side length self.final_width, at resolution self.new_res.
        """
        
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)
        inner_patch = enmap.project(patch.data, shape, wcs)
        self.logger.info(f"Cut inner patch down to {self.final_width}x{self.final_width}")
        
        return self.enmap2pspy(inner_patch)

    def do_lensing(self, cmb_unlensed, kappa_map, apodize_for_alms=True):
        """
        Lens an unlensed CMB patch using a kappa map.
        
        
        Parameters
        ----------
        cmb_unlensed : so_map
            Unlensed CMB patch (T, Q, U components).
        kappa_map : so_map
            Convergence map.
        apodize_for_alms : bool, optional
            Whether to apodize maps before computing alms. Default is True.
        
        
        Returns
        -------
        cmb : so_map
            Lensed CMB patch cut to final width.
        """
        
        padded_width = self.final_width + 2*self.apod_width
        alms_unlensed, alms_phi = self.save_CMB_alms(cmb_unlensed, kappa_map, apodize_for_alms)
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, padded_width, height=padded_width)

        cmb = so_map.car_template(3, self.ra-padded_width/2, self.ra+padded_width/2, self.dec-padded_width/2, self.dec+padded_width/2, self.res)
        for type_unlensed in [0,1,2]:
            self.logger.info(f"Lensing CMB-{type_unlensed} map")
            lensed_map = enmap.ones(shape, wcs)
            lensed_map.data *= lensing.lens_map_curved(shape, wcs, alms_phi, alms_unlensed[type_unlensed])[0]
            self.logger.info(f"Did Lensing")
            lensed_map = enmap.apply_window(lensed_map)
            self.logger.info(f"Convolved")
            cmb.data[type_unlensed] = lensed_map.copy()

        return self.get_innerPatch(cmb)

    def make_lensed_theory(self, HD_kappa_spectrum,
                            ini_file = f'S10_data/bode_almost_wmap5_params_highKeta.ini',
                            lmax = 24000, sim_Lmin_interp = 30):
        """
        Compute lensed CMB theory spectra using CAMB and patch kappa power.
        
        
        Parameters
        ----------
        HD_kappa_spectrum : dict
            Dictionary with keys 'cl' and 'l' containing kappa power spectrum and multipoles.
        ini_file : str, optional
            Path to CAMB parameter ini file. Default is 'S10_data/bode_almost_wmap5_params_highKeta.ini'.
        lmax : int, optional
            Maximum multipole for CAMB outputs. Default is 24000.
        sim_Lmin_interp : int, optional
            Minimum multipole above which to use simulation kappa spectrum instead of CAMB theory. Default is 30.
        
        
        Returns
        -------
        data : ndarray of shape (lmax+1, 5)
            Columns: ell, C_ell^TT, C_ell^EE, C_ell^BB, C_ell^TE.
        """
        
        # power of the high-resolution kappa sim in 10x10 patch w/ ctr at RA=6, dec=6:
        sim_clkk = HD_kappa_spectrum['cl']
        sim_Lbin = HD_kappa_spectrum['l']
        
        # read in the original CAMB `.ini` file used in S10 sims & update accuracy:
        pars = camb.read_ini(ini_file)
        pars.set_matter_power(kmax=10, k_per_logint=130)
        pars.set_for_lmax(lmax+500, lens_potential_accuracy=30, lens_margin=2050)
        pars.set_accuracy( AccuracyBoost =1.1 , \
            lSampleBoost =3.0 , lAccuracyBoost =3.0 , \
            DoLateRadTruncation = False, min_l_logl_sampling=10000 )
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params("mead2016")
        
        # get the CAMB unlensed CMB & lensing convergence theory:
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit='muK', raw_cl=True)
        clkk = results.get_lens_potential_cls(lmax=lmax)[:,0] * 2 * np.pi / 4
        ells = np.arange(lmax+1)
        # lens the unlensed theory using the power of the kappa sim for L > 300 & the CAMB clkk theory for L < 300:
        # CAMB needs clkk at each L, so we need to interpolate the binned sim power
        
        #sim_Lmin_interp is sim power above this L
        sim_Lbin_to_interp = sim_Lbin[sim_Lbin >= sim_Lmin_interp]
        sim_clkk_to_interp = sim_clkk[sim_Lbin >= sim_Lmin_interp]
        theo_Lmax_interp = round(sim_Lbin_to_interp[0]) - 1 # use CAMB theory below this L
        
        Ls_to_interp = np.concatenate([ells[:theo_Lmax_interp], sim_Lbin_to_interp])
        clkk_to_interp = np.concatenate([clkk[:theo_Lmax_interp], sim_clkk_to_interp])
        
        camb_ells = np.arange(pars.max_l+1) # CAMB needs a clkk curve to higher Lmax than it will output
        clkk_interp = interp1d(Ls_to_interp, clkk_to_interp, bounds_error=False, fill_value=clkk_to_interp[-1])(camb_ells)
        
        # get the lensed CMB theory using the kappa power in the patch:
        lensed_powers = results.get_lensed_cls_with_spectrum(clkk_interp * 4 / (2 * np.pi), lmax=lmax, CMB_unit='muK', raw_cl=False)
        lensed_theory = {'ells': ells.copy()}
        for i, s in enumerate(['tt', 'ee', 'bb', 'te']):
            lensed_theory[s] = lensed_powers[:,i].copy()
            lensed_theory[s][:2] = 0
        
        data = np.column_stack([
            lensed_theory['ells'],
            lensed_theory['tt'], 
            lensed_theory['ee'], 
            lensed_theory['bb'], 
            lensed_theory['te'],
        ])
        
        return data