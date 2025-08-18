import logging
import os
import numpy as np
from pspy import so_map, so_window, so_mcm, pspy_utils, so_spectra, sph_tools
from pixell import enmap, utils
import healpy as hp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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

    def save_sparse_thresholded(self, filename, mat, tol=1e-12):
        M = sp.csr_matrix(mat)
        M.data[np.abs(M.data) < tol] = 0.0
        M.eliminate_zeros() 
        sp.save_npz(filename, M)

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
    
    def make_binning_files(self, binning_output_path, delta_ell = 200, spin0and2 = False, type_Cl = False, pre_existing_binning_edges = None, saveBblSparsely = True):
        """
        Generates mode decoupling matrix and binning files for a chosen patch and resolution.
    
        Parameters
        ----------
        binning_output_path : str
            Path to save binning files to
        delta_ell : int
            Desired bin width in final power spectra. Default 200
        spin0and2 : bool, optional
            If True, makes binning files for spin0and2 patch of sky (i.e. CMB T,Q,U maps). Default False
        type_Cl : bool, optional
            If True, makes binning files for Cl rather than Dl. Default False
        """
        
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
        nbins = self.l_max / delta_ell
        if pre_existing_binning_edges == None:
            binning_file = f"lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}"
            pspy_utils.create_binning_file(bin_size=delta_ell, n_bins=nbins, file_name=binning_output_path + binning_file)
        else:
            with open(pre_existing_binning_edges, "r") as f:
                edges = [float(line.strip()) for line in f]
                
            with open(binning_output_path + binning_file, "w") as f_out:
                for i in range(len(edges)-1):
                    left = edges[i]
                    right = edges[i+1] - 1
                    center = (left + right) / 2
                    f_out.write(f"{left:.2f} {right:.2f} {center:.2f}\n")

    
        self.logger.info(f"Doing mode decoupling, without beam")
        if spin0and2:
            if type_Cl:
                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2((window,window), binning_output_path + binning_file, niter=0, lmax=self.l_max, type="Cl")
                self.logger.info(f"Saving mode decoupling")
                np.save(binning_output_path + "mbb_inv_"+binning_file+"_spin0and2_Cl", mbb_inv)
                if saveBblSparsely:
                    for comp in ['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']:
                        self.save_sparse_thresholded(binning_output_path + f"Bbl{comp}_"+binning_file+"_spin0and2_Cl.npz", sp.csr_matrix(Bbl[comp]))
                else:
                    np.save(binning_output_path + "Bbl_"+binning_file+"_spin0and2_Cl", Bbl)
            else:
                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2((window,window), binning_output_path + binning_file, niter=0, lmax=self.l_max, type="Dl")
                self.logger.info(f"Saving mode decoupling")
                np.save(binning_output_path + "mbb_inv_"+binning_file+"_spin0and2", mbb_inv)
                if saveBblSparsely:
                    for comp in ['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']:
                        self.save_sparse_thresholded(binning_output_path + f"Bbl{comp}_"+binning_file+"_spin0and2.npz", sp.csr_matrix(Bbl[comp]))
                else:
                    np.save(binning_output_path + "Bbl_"+binning_file+"_spin0and2", Bbl)
        else:
            if type_Cl:
                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window.copy(), binning_output_path + binning_file, niter=0, lmax=self.l_max, type="Cl")
                self.logger.info(f"Saving mode decoupling")
                np.save(binning_output_path + "mbb_inv_"+binning_file+"_spin0_Cl", mbb_inv)
                if saveBblSparsely:
                    self.save_sparse_thresholded(binning_output_path + f"Bbl_"+binning_file+"_spin0_Cl.npz", sp.csr_matrix(Bbl))
                else:
                    np.save(binning_output_path + "Bbl_"+binning_file+"_spin0_Cl", Bbl)
            else:
                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0(window.copy(), binning_output_path + binning_file, niter=0, lmax=self.l_max, type="Dl")
                self.logger.info(f"Saving mode decoupling")
                np.save(binning_output_path + "mbb_inv_"+binning_file+"_spin0", mbb_inv)
                if saveBblSparsely:
                    self.save_sparse_thresholded(binning_output_path + f"Bbl_"+binning_file+"_spin0.npz", sp.csr_matrix(Bbl))
                else:
                    np.save(binning_output_path + "Bbl_"+binning_file+"_spin0", Bbl)
        
        return None

    def get_binning_files(self, path, delta_ell = 200, spin0and2 = False, type_Cl = False, sparseBbl = True):
        """
        Returns mode decoupling matrix and binning files for use in power spectra.
    
        Parameters
        ----------
        path : str
            Path with binning files
        delta_ell : int
            Desired bin width in final power spectra. Default 200
        spin0and2 : bool, optional
            If True, finds binning files for spin0and2 patch of sky (i.e. CMB T,Q,U maps). Default False
        type_Cl : bool, optional
            If True, finds binning files for Cl rather than Dl. Default False
        """
        if spin0and2:
            if type_Cl:
                mbb_inv = np.load(f"{path}mbb_inv_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2_Cl.npy", allow_pickle=True)
                if sparseBbl:
                    Bbl = {
                        'spin0xspin0' : sp.load_npz(f"{path}Bblspin0xspin0_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2_Cl.npz").toarray(),
                        'spin0xspin2' : sp.load_npz(f"{path}Bblspin0xspin2_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2_Cl.npz").toarray(),
                        'spin2xspin0' : sp.load_npz(f"{path}Bblspin2xspin0_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2_Cl.npz").toarray(),
                        'spin2xspin2' : sp.load_npz(f"{path}Bblspin2xspin2_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2_Cl.npz").toarray()
                    }
                else:
                    Bbl = np.load(f"{path}Bbl_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2_Cl.npy", allow_pickle=True)[()]
            else:
                mbb_inv = np.load(f"{path}mbb_inv_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2.npy", allow_pickle=True)
                if sparseBbl:
                    Bbl = {
                        'spin0xspin0' : sp.load_npz(f"{path}Bblspin0xspin0_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2.npz").toarray(),
                        'spin0xspin2' : sp.load_npz(f"{path}Bblspin0xspin2_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2.npz").toarray(),
                        'spin2xspin0' : sp.load_npz(f"{path}Bblspin2xspin0_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2.npz").toarray(),
                        'spin2xspin2' : sp.load_npz(f"{path}Bblspin2xspin2_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2.npz").toarray()
                    }
                else:
                    Bbl = np.load(f"{path}Bbl_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0and2.npy", allow_pickle=True)[()]
        else:
            if type_Cl:
                mbb_inv = np.load(f"{path}mbb_inv_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0_Cl.npy", allow_pickle=True)
                if sparseBbl:
                    Bbl = sp.load_npz(f"{path}Bbl_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0_Cl.npz").toarray()
                else:
                    Bbl = np.load(f"{path}Bbl_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0_Cl.npy", allow_pickle=True)[()]
            else:
                mbb_inv = np.load(f"{path}mbb_inv_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0.npy", allow_pickle=True)
                if sparseBbl:
                    Bbl = sp.load_npz(f"{path}Bbl_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0.npz").toarray()
                else:
                    Bbl = np.load(f"{path}Bbl_lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}_spin0.npy", allow_pickle=True)[()]
        binning_file = f"{path}lmax{self.l_max}_deltaEll{delta_ell}_{self.ra},{self.dec}_{self.final_width}_{self.apod_width}_{self.res}"

        return mbb_inv, binning_file, Bbl

    def get_foreground_power(self, patch, mbb_inv, binning_file, deconvolve_pw = True, type_Cl = False):
        """
        Saves power spectrum of foreground (spin0) patch.
    
        Parameters
        ----------
        patch : enmap
            Foreground patch
        binning_output_path : str
            Path to binning file
        mbb_inv : numpy.ndarray
            Mode decoupling matrix as ndarray
        deconvolve_pw : bool
            If True, deconvolves foreground patch before taking power. Default True
        type_Cl : bool, optional
            If True, solves for spectra in terms of Cl rather than Dl. Default False
        """
        
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)
        window = self.make_apod_window(shape, wcs, self.apod_width, map_type='pixell')
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        
        self.logger.info(f"Projecting")
        imap = enmap.project( patch.data, shape, wcs) * window
        
        if deconvolve_pw:
            self.logger.info(f"Deconvolving")
            imap = enmap.unapply_window(imap)

        self.logger.info(f"Get alms")
        alms = sph_tools.get_alms(self.enmap2pspy(imap), window_ones, niter=0, lmax=self.l_max)
        
        self.logger.info(f"Get spectra")
        ell_patch, power_patch = so_spectra.get_spectra(alms)
        if type_Cl:
            patch_ells, patch_cls = so_spectra.bin_spectra(ell_patch, power_patch, binning_file, self.l_max, type="Cl", mbb_inv=mbb_inv)
            return patch_cls, patch_ells
        else:
            patch_ells, patch_dls = so_spectra.bin_spectra(ell_patch, power_patch, binning_file, self.l_max, type="Dl", mbb_inv=mbb_inv)
            return patch_dls, patch_ells

    def get_CMB_power(self, cmb, mbb_inv, binning_file, deconvolve_pw = False,
                      spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]):
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width)
        window = self.make_apod_window(shape, wcs, self.apod_width, map_type='pixell')
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        
        self.logger.info(f"Projecting")
        imap_T = enmap.project( cmb.data[0], shape, wcs) * window
        imap_Q = enmap.project( cmb.data[1], shape, wcs) * window
        imap_U = enmap.project( cmb.data[2], shape, wcs) * window

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
        
        self.logger.info(f"Get spectra")
        ell, ps = so_spectra.get_spectra(alms, alms, spectra=spectra)
        ellb, Db = so_spectra.bin_spectra(
            ell, ps, binning_file, self.l_max, type="Dl", mbb_inv=mbb_inv[()], spectra=spectra
        )
        return Db, ellb