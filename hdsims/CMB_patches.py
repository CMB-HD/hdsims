import camb
import numpy as np
import logging
import os
from pixell import enmap, utils, curvedsky, lensing, powspec
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
    
    def make_unlensed_patch(self, theory_path, cmb_seed):

        self.logger.info(f"making unlensed CMB sim for {round(self.res,2)}' {round(self.final_width,2)} x {round(self.final_width,2)} deg patch with seed = {cmb_seed}")
        cmb = so_map.car_template(3, self.ra-self.final_width/2, self.ra+self.final_width/2, self.dec-self.final_width/2, self.dec+self.final_width/2, self.res)
        cl_file = os.path.join(theory_path)
        cmb.data = curvedsky.rand_map(cmb.data.shape, cmb.data.wcs, powspec.read_spectrum(cl_file)[: 3, : 3], seed = cmb_seed)
        return cmb

    #############################################

    def kappa_to_phi(self,kappa_alm,kappa_ainfo=None):
    	oalm = curvedsky.almxfl(alm=kappa_alm,lfilter=lambda x: 2./(x*(x+1)) ,ainfo=kappa_ainfo)
    	oalm[~np.isfinite(oalm)] = 0
    	return oalm

    def save_CMB_alms(self, cmb_unlensed, kappa_map):
            
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        lmax_raw = min(self.l_max,int(window_ones.get_lmax_limit()))

        alms_unlensed = []
        for type_unlensed in [0,1,2]:
            self.logger.info(f"Computing CMB-{type_unlensed} alms")
            unlensed_map = cmb_unlensed.data[type_unlensed]
            alms_unlensed.append(sph_tools.get_alms(self.enmap2pspy(unlensed_map), window_ones, niter=0, lmax=lmax_raw))

        self.logger.info(f"Computing Kappa alms")
        alms_kappa = sph_tools.get_alms(kappa_map, window_ones, niter=0, lmax=lmax_raw)
        self.logger.info(f"Computing Phi alms")
        alms_phi = self.kappa_to_phi(alms_kappa)
        
        return alms_unlensed, alms_phi

    def do_lensing(self, cmb_unlensed, kappa_map):
        alms_unlensed, alms_phi = self.save_CMB_alms(cmb_unlensed, kappa_map)
        shape, wcs = self.get_shape_wcs(self.res, self.ra, self.dec, self.final_width, height=self.final_width)

        cmb = so_map.car_template(3, self.ra-self.final_width/2, self.ra+self.final_width/2, self.dec-self.final_width/2, self.dec+self.final_width/2, self.res)
        for type_unlensed in [0,1,2]:
            self.logger.info(f"Lensing CMB-{type_unlensed} map")
            lensed_map = enmap.ones(shape, wcs)
            lensed_map.data *= lensing.lens_map_curved(shape, wcs, alms_phi, alms_unlensed[type_unlensed])[0]
            self.logger.info(f"Did Lensing")
            lensed_map = enmap.apply_window(lensed_map)
            self.logger.info(f"Convolved")
            cmb.data[type_unlensed] = lensed_map.copy()
        
        return cmb
