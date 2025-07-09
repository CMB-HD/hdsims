import logging
import os
import numpy as np
import healpy as hp
from pspy import so_map, so_window, sph_tools
from pixell import reproject, enmap, curvedsky, utils
import pandas as pd
import scipy

class Foregrounds:
    T_CMB = 2.7255e6
    freq_to_conversion = {
        "030": 7.364967e7,
        "090": 5.526540e8,
        "148": 1.072480e9,
        "219": 1.318837e9,
        "277": 1.182877e9,
        "350": 8.247628e8,
        "none": T_CMB
    }

    #############################################
    
    def __init__(self, ra, dec, final_width, apod_width, 
                 new_res, l_max, log_level=logging.INFO):
        self.ra = ra
        self.dec = dec
        self.final_width = final_width
        self.apod_width = apod_width
        self.new_res = new_res
        self.l_max = l_max

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        return None

    ############################################# Diffuse Components

    def load_fullsky(self, frequency, data_path, fullsky_res, fullsky_output_path,
                     scaling_factor = 1.0, deconvolve_pixel_window = True):    
        
        conversion_factor = self.freq_to_conversion[frequency]
        self.logger.info(f"Using data from {data_path} at res={fullsky_res}")
        try:
            os.makedirs(fullsky_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {fullsky_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {fullsky_output_path}")
        
    
        nside = hp.get_nside(hp.read_map(data_path))
        fullsky_map = so_map.healpix_template(1, nside)
        self.logger.info(f"Using conversion factor {conversion_factor} and scaled map by {scaling_factor}, where nside={nside}")
        fullsky_map.data[:] = hp.read_map(data_path) * self.T_CMB * (1/conversion_factor)
        fullsky_map.data[:] *= scaling_factor
        fullsky_map.write_map(fullsky_output_path + "fullsky")
        self.logger.info(f"Saved fullsky")

        if deconvolve_pixel_window:
            self.logger.info(f"Deconvolving fullsky") 
            sim_pixwin = fullsky_map.get_pixwin()
            fullsky_map.data[:] = fullsky_map.data.astype(np.float64)
            fullsky_map = fullsky_map.convolve_with_pixwin(pixwin = sim_pixwin**-1)
            fullsky_map.write_map(fullsky_output_path + "fullsky_deconvolved")
            self.logger.info(f"Saved fullsky (deconvolved)")
            return fullsky_map

        return fullsky_map

    def extract_largerPatch_from_fullsky(self, fullsky_map, fullsky_res, patch_output_path):
        try:
            os.makedirs(patch_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {patch_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {patch_output_path}")

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

        self.logger.info(f"Getting S10 resolution patch (sources_patch)")
        template_car = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, fullsky_res)
        sim_patch = reproject.healpix2map(fullsky_map.data, template_car.data.shape, template_car.data.wcs, spin=[0], lmax=self.l_max)
        enmap.write_map(patch_output_path + "sources_patch", sim_patch)

        self.logger.info(f"Apodizing S10 patch (apodized_patch)")
        so_patch_larger = so_map.from_enmap(sim_patch)
        apodized_patch = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, fullsky_res)
        binary_car_larger_highRes = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, fullsky_res)
        binary_car_larger_highRes.data[:] = 0
        binary_car_larger_highRes.data[1:-1, 1:-1] = 1
        so_taper_larger_highRes = so_window.create_apodization(binary_car_larger_highRes, apo_type="C1", apo_radius_degree=self.apod_width)
        apodized_patch.data[:] = so_patch_larger.data[:] * so_taper_larger_highRes.data[:]
        apodized_patch.write_map(patch_output_path + "apodized_patch")

        return apodized_patch

    def upsample_largerPatch(self, apodized_patch_old, fullsky_res, patch_output_path):
        try:
            os.makedirs(patch_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {patch_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {patch_output_path}")

        if fullsky_res == self.new_res:
            self.logger.info(f"No need to upsample, already at the desired resolution")
            return apodized_patch_old

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

        self.logger.info(f"Upsampling to {self.new_res} as {patch_output_path}apodized_patch")
        template_car_larger_highRes = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        so_patch_larger_upsampled = so_map.from_enmap(
                                        enmap.resample(apodized_patch_old.data, 
                                                    template_car_larger_highRes.data.shape,
                                                    order=3)
                                    )
        enmap.write_map(patch_output_path + "apodized_patch", so_patch_larger_upsampled.data)

        return so_patch_larger_upsampled

    def convolve_largerPatch_with_beam(self, patch, FWHM, patch_output_path):
        if FWHM == 0:
            self.logger.info(f"FWHM = 0, no convolution")
            return patch

        try:
            os.makedirs(patch_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {patch_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {patch_output_path}")

        FWHM_converted = FWHM/60 * np.pi/180 # radians
        sigma = FWHM_converted/np.sqrt(8*np.log(2))
        beamConvolved_patch = so_map.from_enmap(enmap.smooth_gauss(enmap.enmap(patch.data, patch.data.wcs), sigma))
        beamConvolved_patch.write_map(patch_output_path + "beamConvolved_patch")
        self.logger.info(f"FWHM = {FWHM_converted}, convolved")

        return beamConvolved_patch

    def convolve_largerPatch_with_pw(self, patch, patch_output_path):
        try:
            os.makedirs(patch_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {patch_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {patch_output_path}")

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2
        
        pixwin = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res).get_pixwin()
        pwConvolved_patch = patch.convolve_with_pixwin(pixwin = pixwin)
        pwConvolved_patch.write_map(patch_output_path + "pwConvolved_patch")
        self.logger.info(f"Convolved with pixel window")

        return pwConvolved_patch

    def get_innerPatch(self, patch, patch_output_path):
        try:
            os.makedirs(patch_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {patch_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {patch_output_path}")
        
        width = self.final_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2
        
        inner_patch = so_map.get_submap_car(patch, 
                                      so_map.bounding_box_from_map(so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)))    
        inner_patch.write_map(patch_output_path + "inner_patch")
        self.logger.info(f"Cut inner patch down")

        return inner_patch

    def generate_diffuse_foreground(self, component, frequency, data_path, do_fullsky_part = True):

        deconvolve_pixel_window = True
        nside = 8192
        scaling_factor = 1.0
        
        if component == 'tSZ':
            scaling_factor = 0.75        
        if component == 'kappa':
            nside = 4096
            deconvolve_pixel_window = False
            
        fullsky_res = hp.nside2resol(nside, arcmin=True)

        new_res_path = self.output_path + component + "/" + frequency + "/" + str(self.new_res) + "/"
        old_res_path = self.output_path + component + "/" + frequency + "/" + str(fullsky_res) + "/"
        
        if do_fullsky_part:
            self.load_fullsky(frequency, data_path, fullsky_res, fullsky_output_path = old_res_path,
                              scaling_factor = scaling_factor, deconvolve_pixel_window = deconvolve_pixel_window) 

        fullsky_map = so_map.healpix_template(1, nside)
        if deconvolve_pixel_window:
            fullsky_map.data[:] = hp.read_map(old_res_path + "fullsky_deconvolved")
        else:
            fullsky_map.data[:] = hp.read_map(old_res_path + "fullsky")
        largerPatch = self.extract_largerPatch_from_fullsky(fullsky_map, fullsky_res, patch_output_path = old_res_path)
        largerPatch_UHR = self.upsample_largerPatch(largerPatch, fullsky_res, patch_output_path = new_res_path)
        if component != 'kappa':
            largerPatch_UHR_pwConvolved = self.convolve_largerPatch_with_pw(largerPatch_UHR, patch_output_path = new_res_path)
        else:
            largerPatch_UHR_pwConvolved = largerPatch_UHR.copy()
        innerPatch = self.get_innerPatch(largerPatch_UHR_pwConvolved, patch_output_path = new_res_path)

        return innerPatch

    ############################################# Discrete Components

    def make_catalog(self, data_path, component):
        try:
            os.makedirs(self.output_path + component, exist_ok=True)
            self.logger.info(f"Made directory: {self.output_path + component}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {self.output_path + component}")
        
        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2
        output_catalog_name = self.output_path + component + f"/sources_in_{width}x{width}_{self.ra},{self.dec}.csv"

        if component == 'CIB':
            self.logger.info(f"Making CIB catalog in the patch")
            catalog_files = ["IRBlastPop","IRgal_S_5","IRgal_S_6","IRgal_S_7",
                     "IRgal_S_8","IRgal_S_9","IRgal_S_10","IRgal_S_4","IRgal_S_1","IRgal_S_2","IRgal_S_3"]
    
            full_df = pd.DataFrame(columns=['Halo_ID', 'ra_deg', 'dec_deg', 'redshift', '30GHz_flux', '90GHz_flux', '148GHz_flux', '219GHz_flux', '277GHz_flux', '350GHz_flux'])
            for file in catalog_files:
                self.logger.info(f"Dealing with file: {file}")
                input_df = pd.read_csv(data_path + file + ".dat", delim_whitespace=True, header=None)
            
                input_df.rename(columns={0: 'Halo_ID', 
                                         1: 'ra_deg',
                                         2: 'dec_deg',
                                         3: 'redshift',
                                         4: '30GHz_flux',
                                         5: '90GHz_flux',
                                         6: '148GHz_flux',
                                         7: '219GHz_flux',
                                         8: '277GHz_flux',
                                         9: '350GHz_flux'}, inplace=True)
                
                input_df = input_df.drop(input_df[input_df.dec_deg < dec_min].index)
                input_df = input_df.drop(input_df[input_df.dec_deg > dec_max].index)
                input_df = input_df.drop(input_df[input_df.ra_deg < ra_min].index)
                input_df = input_df.drop(input_df[input_df.ra_deg > ra_max].index)
                input_df = input_df.reset_index()
            
                full_df = pd.concat([full_df,input_df],axis=0)
            full_df.to_csv(output_catalog_name)
        elif component == 'radio':
            with open(data_path, 'r') as file:
                lines = file.readlines()
                
            self.logger.info(f"Making radio catalog in the patch")
            delimiter = ' '
            within_patch = []
            for line in lines:
                fields = line.strip().split(delimiter)
                fields = [element for element in fields if element != '']
                if (ra_min <= float(fields[0]) <= ra_max) and (dec_min <= float(fields[1]) <= dec_max):
                    within_patch.append(fields)
                    
            sources = pd.DataFrame(within_patch, columns=['ra_deg','dec_deg','redshift','1.4GHz_flux','30GHz_flux','90GHz_flux','148GHz_flux','219GHz_flux','277GHz_flux','350GHz_flux'])
            
            input_df = sources.copy()
            input_df = input_df.drop(input_df[(input_df.dec_deg).astype(np.float64) < dec_min].index)
            input_df = input_df.drop(input_df[(input_df.dec_deg).astype(np.float64) > dec_max].index)
            input_df = input_df.drop(input_df[(input_df.ra_deg).astype(np.float64) < ra_min].index)
            input_df = input_df.drop(input_df[(input_df.ra_deg).astype(np.float64) > ra_max].index)
            input_df = input_df.reset_index()
            
            input_df.to_csv(output_catalog_name)
        
        return None

    def generate_largerPatch_from_catalog(self, catalog, frequency, scaling_factor, patch_output_path):
        try:
            os.makedirs(patch_output_path, exist_ok=True)
            self.logger.info(f"Made directory: {patch_output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {patch_output_path}")

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

        initial_patch = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        pixsizemap = enmap.pixsizemap(initial_patch.data.shape, initial_patch.data.wcs)

        def get_pixel_ra(ra, shape=initial_patch.data.shape, wcs=initial_patch.data.wcs):
            coord_pos = [[np.deg2rad(0)], [np.deg2rad(ra)]]
            pixel = enmap.sky2pix(shape, wcs, coord_pos)
            pixel_ra = int(round(pixel[1][0]))
            if pixel_ra >= initial_patch.data.shape[1]:
                return pixel_ra-1
            return pixel_ra
        def get_pixel_dec(dec, shape=initial_patch.data.shape, wcs=initial_patch.data.wcs):
            coord_pos = [[np.deg2rad(dec)], [np.deg2rad(0)]]
            pixel = enmap.sky2pix(shape, wcs, coord_pos)
            pixel_dec = int(round(pixel[0][0]))
            if pixel_dec >= initial_patch.data.shape[0]:
                return pixel_dec-1
            return pixel_dec

        flux_feature = str(int(frequency)) + 'GHz_flux'
        conversion_factor = self.freq_to_conversion[frequency]

        self.logger.info(f"Adding sources")
        for i in (np.arange(np.shape(catalog)[0])):      
            # Convert to Jy/steradian
            Jysteradian = (0.001*catalog[flux_feature][i])
            # Convert to delta T / T_CMB (given conversion factor)
            dTTcmb = Jysteradian/conversion_factor
            # Convert to delta T
            deltaT = dTTcmb * self.T_CMB

            initial_patch.data[get_pixel_dec(catalog['dec_deg'][i]),get_pixel_ra(catalog['ra_deg'][i])] += deltaT

        initial_patch.data[:] *= scaling_factor
        initial_patch.data[:] /= pixsizemap.data[:]
        initial_patch.write_map(patch_output_path + "initial_patch")

        self.logger.info(f"Apodizing S10 patch (apodized_patch)")
        apodized_patch = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        binary_car_larger_highRes = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        binary_car_larger_highRes.data[:] = 0
        binary_car_larger_highRes.data[1:-1, 1:-1] = 1
        so_taper_larger_highRes = so_window.create_apodization(binary_car_larger_highRes, apo_type="C1", apo_radius_degree=self.apod_width)
        apodized_patch.data[:] = initial_patch.data[:] * so_taper_larger_highRes.data[:]
        apodized_patch.write_map(patch_output_path + "apodized_patch")
        
        return apodized_patch

    def generate_discrete_foreground(self, component, frequency, data_path, make_catalog = True):
        old_res_path = self.output_path + component + "/" + frequency + "/" + str(self.new_res) + "/"
        new_res_path = self.output_path + component + "/" + frequency + "/" + str(self.new_res) + "/"

        scaling_factor = 1.0
        if component == 'CIB':
            scaling_factor = 0.75

        if make_catalog:
            self.make_catalog(data_path, component) 
        else:
            self.logger.info(f"Not cutting down the catalog, as it's already been done")
        catalog = pd.read_csv(self.output_path + component + f"/sources_in_{self.final_width + 2*self.apod_width}x{self.final_width + 2*self.apod_width}_{self.ra},{self.dec}.csv")

        largerPatch_UHR = self.generate_largerPatch_from_catalog(catalog, frequency, scaling_factor, patch_output_path = new_res_path)
        largerPatch_UHR_pwConvolved = self.convolve_largerPatch_with_pw(largerPatch_UHR, patch_output_path = new_res_path)
        innerPatch = self.get_innerPatch(largerPatch_UHR_pwConvolved, patch_output_path = new_res_path)

        return innerPatch

    def generate_discrete_foreground_from_custom_catalog(self, component, frequency, data_path, catalog):
        old_res_path = self.output_path + component + "/" + frequency + "/" + str(self.new_res) + "/"
        new_res_path = self.output_path + component + "/" + frequency + "/" + str(self.new_res) + "/"

        largerPatch_UHR = self.generate_largerPatch_from_catalog(catalog, frequency, scaling_factor = 1.0, patch_output_path = new_res_path)
        largerPatch_UHR_pwConvolved = self.convolve_largerPatch_with_pw(largerPatch_UHR, patch_output_path = new_res_path)
        innerPatch = self.get_innerPatch(largerPatch_UHR_pwConvolved, patch_output_path = new_res_path)

        return innerPatch

    ############################################# Stitching

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

    def get_theory_for_stitching(self, output_path,
                                      template_cls, template_ells, patch_cls, patch_ells, \
                                      template_minimization_index, patch_minimization_index, \
                                      template_ell0_index=3101, patch_ell0_index=15,
                                      seed = 3):
        try:
            os.makedirs(output_path, exist_ok=True)
            self.logger.info(f"Made directory: {output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {output_path}")
            
        self.logger.info(f"Saving to path {output_path}")
        self.logger.info(f"Matching template at ell={template_ells[template_minimization_index]} to patch at ell={patch_ells[patch_minimization_index]}")
        self.logger.info(f"template ell0={template_ells[template_ell0_index]} w/ patch ell0={patch_ells[patch_ell0_index]}")
        # A fixes the point at ell0
        
        # This sets A based on the patch at ell0
        A_val = patch_cls[patch_ell0_index]/template_cls[template_ell0_index]
        self.logger.info(f"% Difference at ~ell0 is now {100*(patch_cls[patch_ell0_index] - A_val * template_cls[template_ell0_index])/(A_val * template_cls[template_ell0_index])}")
        
        # Sets n do minimize the distance between theory and patch at minimization_index
        initial_guess = [0.0]
        bounds = [(-5, 5)]
        def func_to_minimize(n_kappa):
            theory_cls = ((template_ells/template_ell0_index)**n_kappa) * A_val * template_cls
            return (theory_cls[template_minimization_index] - patch_cls[patch_minimization_index])**2
        result = scipy.optimize.minimize(func_to_minimize, initial_guess, method='Nelder-Mead', bounds=bounds)
    
        n_val = result.x[0]
        self.logger.info(f"n = {n_val:.10e}")
        self.logger.info(f"A = {A_val:.10e}")
    
        theory_cls = ((template_ells/template_ell0_index)**n_val) * A_val * template_cls
        np.save(output_path + "cls_stitchingTheory",theory_cls)
        np.save(output_path + "ells_stitchingTheory",template_ells)

        self.logger.info(f"Now generating alms from theory")
        alms_theory = curvedsky.rand_alm(theory_cls,lmax=self.l_max,seed=seed)
        np.save(output_path + "alms_stitchingTheory",alms_theory)

        return theory_cls, template_ells, alms_theory

    def get_S10_for_stitching(self, output_path, upsampled_patch):
        try:
            os.makedirs(output_path, exist_ok=True)
            self.logger.info(f"Made directory: {output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {output_path}")
        
        shape, wcs = self.get_shape_wcs(self.new_res, self.ra, self.dec, self.final_width, height=self.final_width)
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        lmax_raw = min(self.l_max,int(window_ones.get_lmax_limit()))

        self.logger.info(f"Getting alms from upsampled S10 patches")
        imap = enmap.project(upsampled_patch, shape, wcs)
        alms_for_stitching = sph_tools.get_alms(self.enmap2pspy(imap), window_ones, niter=0, lmax=lmax_raw)
        self.logger.info(f"Saving alms_for_stitching")
        np.save(f"{output_path}alms_for_stitching", alms_for_stitching)

        self.logger.info(f"Resizing as alms_for_stitching_resized from {lmax_raw} to {self.l_max}")
        alms_for_stitching_resized = hp.sphtfunc.resize_alm(alms_for_stitching, 
                                                            lmax=lmax_raw, lmax_out=self.l_max,
                                                            mmax=lmax_raw, mmax_out=self.l_max)
        self.logger.info(f"Saving alms_for_stitching_resized")
        np.save(f"{output_path}alms_for_stitching_resized", alms_for_stitching_resized)
        
        return alms_for_stitching_resized

    def stitch_alms(self, alms_theory, alms_S10, output_path, l_cutoff):
        try:
            os.makedirs(output_path, exist_ok=True)
            self.logger.info(f"Made directory: {output_path}")
        except FileExistsError:
            self.logger.info(f"Directory already exists: {output_path}")
            
        self.logger.info(f"Stitching alms")
        l, m = hp.Alm.getlm(self.l_max)
        alms_combined = np.where(l >= l_cutoff, alms_theory, alms_S10)

        self.logger.info(f"Generating patch from alms")
        shape, wcs = self.get_shape_wcs(self.new_res, self.ra, self.dec, self.final_width, height=self.final_width)
        window_ones = enmap.ones(shape, wcs)
        
        patch_map = curvedsky.alm2map(alm=alms_combined, map=window_ones)
        patch_map_so = self.enmap2pspy(patch_map)
        patch_map_so.write_map(output_path + "inner_patch_stitched")
        
        return patch_map_so
