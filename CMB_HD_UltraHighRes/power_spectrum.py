import logging
import os
import numpy as np
from pspy import so_map, so_window, so_mcm, pspy_utils, so_spectra, sph_tools
from pixell import enmap, utils
import healpy as hp

class Spectra:
    def __init__(self, ra, dec, final_width, apod_width, 
                 res, l_max, log_level=logging.INFO):
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

    def enmap2pspy(self, imap):
        if len(imap.shape) > 2:
            ncomp = imap.shape[-3]
        else:
            ncomp = 1
        omap = so_map.car_template_from_shape_wcs(ncomp, imap.shape, imap.wcs)
        omap.data[:] = imap.copy()
        return omap
    
    def get_coord_box(self, ra_ctr, dec_ctr, width, height=None):
        """
        Returns an array of [[dec_min, ra_max], [dec_max, ra_min]] for a map of
        shape `width` x `height` (or a square map if the height is not provided),
        centered at RA, dec = (`ra_ctr`, `dec_ctr`)
    
        The `ra_ctr`, `dec_ctr`, `width`, and `height` should be in units of degrees;
        the returned `coord_box` is in units of radians
        """
        height = width if (height is None) else height
        ra_min = ra_ctr - (width / 2)
        ra_max = ra_ctr + (width / 2)
        dec_min = dec_ctr - (height / 2)
        dec_max = dec_ctr + (height / 2)
        coord_box = np.array([[dec_min, ra_max], [dec_max, ra_min]])
        return np.deg2rad(coord_box)
        
    def get_shape_wcs(self, res, ra_ctr, dec_ctr, width, height=None):
        """Returns the `shape` and `wcs` for a `pixell.enmap.ndmap` of the
        given resolution and map geometry.
    
        `res` should be in units of arcminutes
        `ra_ctr`, `dec_ctr`, `width`, and `height` should each be in units of degrees
        """
        coord_box = self.get_coord_box(ra_ctr, dec_ctr, width, height=height)
        shape, wcs = enmap.geometry(pos=coord_box, res=res * utils.arcmin, proj='car')
        return shape, wcs
    
    def make_apod_window(self, shape, wcs, apod_width_deg, map_type='so_map'):
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
    
    def make_binning_files(self, delta_ell, spin = 0, binning_output_path = None):
        try:
            os.makedirs(binning_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {binning_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {binning_output_path}")
        
        width = self.final_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

        window = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.res)
        window.data[:] = 0
        window.data[1:-1, 1:-1] = 1
        window = so_window.create_apodization(window, apo_type="C1", apo_radius_degree=self.apod_width)

        self.logger.info(f"Making binning file")
        binning_file = f"lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}"
        nbins = self.l_max / delta_ell
        pspy_utils.create_binning_file(bin_size=delta_ell, n_bins=nbins, file_name=binning_output_path + binning_file)
    
        self.logger.info(f"Doing mode decoupling, without beam")
        if spin == 0:
            mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window.copy(), binning_output_path + binning_file, niter=0, lmax=self.l_max, type="Cl")
            self.logger.info(f"Saving mode decoupling")
            np.save(binning_output_path + "mbb_inv_"+binning_file+"_spin0", mbb_inv)
            np.save(binning_output_path + "Bbl_"+binning_file+"_spin0", Bbl)
        elif spin == 2:
            mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2((window,window), binning_output_path + binning_file, niter=0, lmax=self.l_max, type="Cl")
            self.logger.info(f"Saving mode decoupling")
            np.save(binning_output_path + "mbb_inv_"+binning_file+"_spin0and2", mbb_inv)
            np.save(binning_output_path + "Bbl_"+binning_file+"_spin0and2", Bbl)
        
        return mbb_inv, Bbl

    def get_foreground_power(self, patch, mbb_inv, binning_file, deconvolve_pw = True, output_path = None):
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)
        window = self.make_apod_window(shape, wcs, self.apod_width, map_type='pixell')
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        
        self.logger.info(f"Projecting")
        imap = enmap.project( patch, shape, wcs) * window
        
        if deconvolve_pw:
            self.logger.info(f"Deconvolving")
            imap = enmap.unapply_window(imap)

        self.logger.info(f"Get alms")
        alms = sph_tools.get_alms(self.enmap2pspy(imap), window_ones, niter=0, lmax=self.l_max)
        np.save(output_path + "alms", alms)
        
        self.logger.info(f"Get spectra")
        ell_patch, cl_patch = so_spectra.get_spectra(alms)
        patch_ells, patch_cls = so_spectra.bin_spectra(ell_patch, cl_patch, binning_file, self.l_max, type="Cl", mbb_inv=mbb_inv)
        hp.write_cl(output_path + "cls.fits", patch_cls, overwrite=True)
        hp.write_cl(output_path + "ells.fits", patch_ells, overwrite=True)
        
        return patch_cls, patch_ells

    def get_CMB_power(self, patch_T, patch_Q, patch_U, mbb_inv, binning_file, output_path, deconvolve_pw = False,
                      spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]):
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width)
        window = self.make_apod_window(shape, wcs, self.apod_width, map_type='pixell')
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        
        self.logger.info(f"Projecting")
        imap_T = enmap.project( patch_T, shape, wcs) * window
        imap_Q = enmap.project( patch_Q, shape, wcs) * window
        imap_U = enmap.project( patch_U, shape, wcs) * window

        if deconvolve_pw:
            self.logger.info(f"Deconvolving")
            imap_T = enmap.unapply_window(imap_T)
            imap_Q = enmap.unapply_window(imap_Q)
            imap_U = enmap.unapply_window(imap_U)

        self.logger.info(f"Stacking")
        imap = enmap.ndmap(
            np.stack([imap_T, imap_Q, imap_U]),
            imap_T.wcs
        )
        self.logger.info(f"Get alms")
        alms = sph_tools.get_alms(self.enmap2pspy(imap), np.stack([window_ones, window_ones, window_ones]), niter=0, lmax=self.l_max)
        np.save(output_path + "alms",alms)
        
        self.logger.info(f"Get spectra")
        #alms = np.load(output_path + "alms.npy")
        ell, ps = so_spectra.get_spectra(alms, alms, spectra=spectra)
        ellb, Cb = so_spectra.bin_spectra(
            ell, ps, binning_file, self.l_max, type="Cl", mbb_inv=mbb_inv[()], spectra=spectra
        )

        np.save(output_path + "cls",Cb)
        np.save(output_path + "ell",ellb)

        return Cb, ellb