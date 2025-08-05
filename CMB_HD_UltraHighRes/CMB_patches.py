import camb
import numpy as np
import logging
import os
from pixell import enmap, utils, curvedsky, lensing
from pspy import sph_tools, so_map, so_window

class CMB:

    #############################################
    
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

    #############################################

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
    
    def make_apod_window(self, shape, wcs, apod_width_deg, map_type='pixell'):
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
    
    def make_cmb_signal_sim(self, fileps, shape, wcs, seed=None):
        hddata = np.loadtxt(fileps, comments="#")
        l = hddata[:,0]
        
        ps_square = np.zeros((3, 3, np.size(l)))
        print(ps_square.shape)
        
        ps_square[0, 0] = hddata[:,5]
        ps_square[0, 1] = hddata[:,8]
        ps_square[1, 0] = hddata[:,8]
        ps_square[1, 1] = hddata[:,6]
        ps_square[2, 2] = hddata[:,7]
        
        cmbmap_pol = enmap.rand_map((3, *shape), wcs, cov=ps_square, seed=seed)
        return cmbmap_pol

    def make_unlensed_patch(self, theory_path, cmb_seed, unlensed_output, convolve_pw = True):
        try:
            os.makedirs(unlensed_output, exist_ok=True)
            self.logger.info(f"Made directory: {unlensed_output}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {unlensed_output}")
        
        padded_shape, padded_wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)

        # file name of sim:
        sim_info = [f'polarization_unlensedcmb{cmb_seed:03d}', f'{round(self.res,2)}arcmin', f'ra{round(self.ra,2)}dec{round(self.dec,2)}', f'{round(self.final_width,2)}x{round(self.final_width,2)}deg']
        if convolve_pw:
            sim_info.append('pwConvolved')
        sim_fname_info = '_'.join(sim_info)
        sim_fname = os.path.join(unlensed_output, f'{sim_fname_info}.fits')

        cutout_width = self.final_width + 2*self.apod_width
        cutout_shape, cutout_wcs = self.get_shape_wcs(self.res, self.ra, self.dec, cutout_width, height=cutout_width)
        
        self.logger.info(f"making unlensed CMB sim for {round(self.res,2)}' {round(cutout_width,2)} x {round(cutout_width,2)} deg patch with seed = {cmb_seed}")
        cmb_sim = self.make_cmb_signal_sim(theory_path, cutout_shape, cutout_wcs, seed=cmb_seed)
        
        if convolve_pw:
            self.logger.info(f"apodizing & convolving pixel window")
            apod_window = self.make_apod_window(cutout_shape, cutout_wcs, self.apod_width)
            cmb_sim = enmap.apply_window(cmb_sim * apod_window)
            
        self.logger.info(f"cutting out {round(self.final_width,2)} x {round(self.final_width,2)} deg patch & saving")
        cmb_sim = enmap.project(cmb_sim, padded_shape, padded_wcs)
        enmap.write_map(sim_fname, cmb_sim)
        self.logger.info(f"saved {sim_fname}")

        return cmb_sim

    #############################################

    def kappa_to_phi(self,kappa_alm,kappa_ainfo=None):
    	oalm = curvedsky.almxfl(alm=kappa_alm,lfilter=lambda x: 2./(x*(x+1)) ,ainfo=kappa_ainfo)
    	oalm[~np.isfinite(oalm)] = 0
    	return oalm

    def save_CMB_alms(self, unlensed_map_path, kappa_map_path, unlensed_output):
        try:
            os.makedirs(unlensed_output, exist_ok=True)
            self.logger.info(f"Made directory: {unlensed_output}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {unlensed_output}")
            
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        lmax_raw = min(self.l_max,int(window_ones.get_lmax_limit()))

        for type_unlensed in [0,1,2]:
            self.logger.info(f"Computing CMB-{type_unlensed} alms")
            unlensed_map = enmap.read_map(unlensed_map_path)[type_unlensed]
            alms_unlensed = sph_tools.get_alms(self.enmap2pspy(unlensed_map), window_ones, niter=0, lmax=lmax_raw)
            enmap.write_map(f"{unlensed_output}alms_unlensed_{type_unlensed}", alms_unlensed)

        self.logger.info(f"Computing Kappa alms")
        kappa_map = enmap.read_map(kappa_map_path)
        alms_kappa = sph_tools.get_alms(self.enmap2pspy(kappa_map), window_ones, niter=0, lmax=lmax_raw)
        np.save(f"{unlensed_output}alms_kappa", alms_kappa)

        self.logger.info(f"Computing Phi alms")
        alms_phi = self.kappa_to_phi(alms_kappa)
        np.save(f"{unlensed_output}alms_phi", alms_phi)
        
        return None

    def do_lensing(self, lensing_output_path):
        try:
            os.makedirs(lensing_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {lensing_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {lensing_output_path}")
        
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)
        alms_phi = np.load(f"{lensing_output_path}alms_phi.npy")
        
        for type_unlensed in [0,1,2]:
            self.logger.info(f"Lensing CMB-{type_unlensed} map")
            alms_unlensed = np.load(f"{lensing_output_path}alms_unlensed_{type_unlensed}.npy")
            lensed_map = enmap.ones(shape, wcs)
            lensed_map.data = lensing.lens_map_curved(shape, wcs, alms_phi, alms_unlensed)[0]
            self.logger.info(f"Did Lensing")
            lensed_map = enmap.apply_window(lensed_map)
            self.logger.info(f"Convolved")
            enmap.write_map(f"{lensing_output_path}lensed_{type_unlensed}output",lensed_map)
        
        return None
