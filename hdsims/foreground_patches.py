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
    freq_to_freqpath = {
        30: "030",
        90: "090",
        148: "148",
        219: "219",
        277: "277",
        350: "350",
        None: "none"
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

    def load_fullsky(self, frequency, data_path,
                     scaling_factor = 1.0, deconvolve_pixel_window = True):    
        """
        Loads and processess full-sky input at a given frequency.
    
        Parameters
        ----------
        frequency : str
            Frequency label used in pathing (e.g. '090').
        data_path : str
            Path to the input full-sky data.
        scaling_factor : float, optional
            Optional multiplicative factor applied to the input map. Default is 1.0.
        deconvolve_pixel_window : bool, optional
            If True, pixel window deconvolution is applied on full-sky data. Default is True
    
        Returns
        -------
        fullsky_map : so_map
            Final processed full-sky map in units of micro-K.
        """

        conversion_factor = self.freq_to_conversion[self.freq_to_freqpath[frequency]]
        self.logger.info(f"Using data from {data_path}")
    
        nside = hp.get_nside(hp.read_map(data_path))
        fullsky_map = so_map.healpix_template(1, nside)
        self.logger.info(f"Using conversion factor {conversion_factor} and scaled map by {scaling_factor}, where nside={nside}")
        fullsky_map.data[:] = hp.read_map(data_path) * self.T_CMB * (1/conversion_factor)
        fullsky_map.data[:] *= scaling_factor

        if deconvolve_pixel_window:
            self.logger.info(f"Deconvolving fullsky") 
            sim_pixwin = fullsky_map.get_pixwin()
            fullsky_map.data[:] = fullsky_map.data.astype(np.float64)
            fullsky_map = fullsky_map.convolve_with_pixwin(pixwin = sim_pixwin**-1)
            return fullsky_map

        return fullsky_map

    def extract_largerPatch_from_fullsky(self, fullsky_map, fullsky_res, lmax_to_extract):
        """
        Cuts out patch of sky from full-sky data
    
        Parameters
        ----------
        fullsky_map : so_map
            Final processed full-sky map in units of micro-K without pixel window convolution.
        fullsky_res : float
            Resolution of full-sky map in units of arcmin.
    
        Returns
        -------
        apodized_patch : so_map
            Final patch of sky at resolution fullsky_res centered at self.ra, self.dec of width (self.final_width + 2*self.apod_width) apodized on each side by self.apod_width.
        """

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

        self.logger.info(f"Getting S10 resolution patch (sources_patch)")
        template_car = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, fullsky_res)
        sim_patch = reproject.healpix2map(fullsky_map.data, template_car.data.shape, template_car.data.wcs, spin=[0], lmax=lmax_to_extract)

        self.logger.info(f"Apodizing S10 patch (apodized_patch)")
        so_patch_larger = so_map.from_enmap(sim_patch)
        apodized_patch = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, fullsky_res)
        binary_car_larger_highRes = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, fullsky_res)
        binary_car_larger_highRes.data[:] = 0
        binary_car_larger_highRes.data[1:-1, 1:-1] = 1
        so_taper_larger_highRes = so_window.create_apodization(binary_car_larger_highRes, apo_type="C1", apo_radius_degree=self.apod_width)
        apodized_patch.data[:] = so_patch_larger.data[:] * so_taper_larger_highRes.data[:]

        return apodized_patch

    def upsample_largerPatch(self, apodized_patch_old, fullsky_res):
        """
        Cuts out patch of sky from full-sky data
    
        Parameters
        ----------
        apodized_patch_old : so_map
            Patch of sky at resolution fullsky_res centered at self.ra, self.dec of width (self.final_width + 2*self.apod_width) apodized on each side by self.apod_width.
        fullsky_res : float
            Resolution of full-sky map in units of arcmin.
    
        Returns
        -------
        so_patch_larger_upsampled : so_map
            Final patch of sky at resolution self.new_res centered at self.ra, self.dec of width (self.final_width + 2*self.apod_width) apodized on each side by self.apod_width.
        """

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

        self.logger.info(f"Upsampling to {self.new_res}")
        template_car_larger_highRes = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        so_patch_larger_upsampled = so_map.from_enmap(
                                        enmap.resample(apodized_patch_old.data, 
                                                    template_car_larger_highRes.data.shape,
                                                    order=3)
                                    )
        return so_patch_larger_upsampled

    def convolve_largerPatch_with_pw(self, patch):
        """
        Convolves apodized patch of sky with pixel window function.
    
        Parameters
        ----------
        patch : so_map
            Final patch of sky at resolution self.new_res centered at self.ra, self.dec of width (self.final_width + 2*self.apod_width) apodized on each side by self.apod_width.
    
        Returns
        -------
        pwConvolved_patch : so_map
            Final patch of sky at resolution self.new_res centered at self.ra, self.dec of width (self.final_width + 2*self.apod_width) apodized on each side by self.apod_width, convolved with pixel window function.
        """

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2
        
        pixwin = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res).get_pixwin()
        pwConvolved_patch = patch.convolve_with_pixwin(pixwin = pixwin)
        self.logger.info(f"Convolved with pixel window")

        return pwConvolved_patch

    def get_innerPatch(self, patch):
        """
        Cuts out inner patch of desired width from a larger patch of sky.
    
        Parameters
        ----------
        patch : so_map
            Final patch of sky at resolution self.new_res centered at self.ra, self.dec of width (self.final_width + 2*self.apod_width) apodized on each side by self.apod_width, convolved with pixel window function.
    
        Returns
        -------
        inner_patch : so_map
            Final patch of sky at resolution self.new_res centered at self.ra, self.dec of width self.final_width, convolved with pixel window function.
        """
        

        shape, wcs = self.get_shape_wcs(self.new_res, self.ra, self.dec, self.final_width, height=self.final_width)
        inner_patch = enmap.project(patch.data, shape, wcs)
        self.logger.info(f"Cut inner patch down")
        
        return self.enmap2pspy(inner_patch)

    def generate_diffuse_foreground(self, component, frequency, fullsky_deconvolved_path = None, S10_largeApodized_path = None):
        """
        Generates a patch of a diffuse foreground (e.g. tSZ, kappa) at a given frequency.
    
        Parameters
        ----------
        component : str
            Foreground type ('tSZ', 'kappa', or 'kSZ').
        frequency : int or None
            Frequency of desired foreground (30, 90, 148, 219, 277, 350) or None for lensing convergence.
        do_fullsky_part : bool, optional
            If True, loads and processes full-sky input. Default is False.
        data_path : str or None
            Path to the input full-sky data. Can be None if do_fullsky_part is False. Default is None
    
        Returns
        -------
        innerPatch : so_map
            Final processed diffuse foreground patch at resolution of self.new_res.
        """

        # Specific foreground considerations
        nside = 8192
        if component == 'kappa':
            nside = 4096
            
        fullsky_res = hp.nside2resol(nside, arcmin=True)

        if fullsky_deconvolved_path != None:
            fullsky_map = so_map.healpix_template(1, nside)
            fullsky_map.data[:] = hp.read_map(fullsky_deconvolved_path)

            if component == 'kappa':
                # nside = 4096 for kappa is not big enough to extract a full lmax=24000 patch without error. Luckily we don't use the high-ell
                # values anyway, so we can just make sure we extract at the fullsky lmax for kappa
                largerPatch = self.extract_largerPatch_from_fullsky(fullsky_map, fullsky_res, lmax_to_extract = 6287)
            else:
                largerPatch = self.extract_largerPatch_from_fullsky(fullsky_map, fullsky_res, lmax_to_extract = self.l_max)
        elif S10_largeApodized_path != None:
            largerPatch = so_map.read_map(S10_largeApodized_path)
        else:
            self.logger.info(f"No input data included")
        
        largerPatch_HD = self.upsample_largerPatch(largerPatch, fullsky_res)
        if component != 'kappa':
            largerPatch_HD_pwConvolved = self.convolve_largerPatch_with_pw(largerPatch_HD)
        else:
            largerPatch_HD_pwConvolved = largerPatch_HD.copy()
        innerPatch = self.get_innerPatch(largerPatch_HD_pwConvolved)

        return innerPatch

    ############################################# Discrete Components

    def make_catalog(self, data_path, component):
        
        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2

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
            return full_df

        elif component == 'radio':
            data_path = data_path + "radio.cat"
            
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
        
            return input_df

    def place_sources_in_largerPatch(self, catalog, frequency, scaling_factor):

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
                return initial_patch.data.shape[1]-1
            return pixel_ra
        def get_pixel_dec(dec, shape=initial_patch.data.shape, wcs=initial_patch.data.wcs):
            coord_pos = [[np.deg2rad(dec)], [np.deg2rad(0)]]
            pixel = enmap.sky2pix(shape, wcs, coord_pos)
            pixel_dec = int(round(pixel[0][0]))
            if pixel_dec >= initial_patch.data.shape[0]:
                return initial_patch.data.shape[0]-1
            return pixel_dec

        flux_feature = str(frequency) + 'GHz_flux'
        conversion_factor = self.freq_to_conversion[self.freq_to_freqpath[frequency]]

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
        
        return initial_patch

    def generate_discrete_foreground(self, frequency, catalog, scaling_factor = 1.0):
        """
        Generates a patch of a discrete foregrounds (e.g. radio, CIB) at a given frequency from a custom catalog.
    
        Parameters
        ----------
        frequency : int or None
            Frequency of desired foreground (30, 90, 148, 219, 277, 350) or None for lensing convergence.
        catalog : 
            Input catalog of point sources.
    
        Returns
        -------
        innerPatch : so_map
            Final processed diffuse foreground patch at resolution of self.new_res.
        """

        initial_patch = self.place_sources_in_largerPatch(catalog, frequency, scaling_factor)

        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2
        self.logger.info(f"Apodizing S10 patch (apodized_patch)")
        apodized_patch = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        binary_car_larger_highRes = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, self.new_res)
        binary_car_larger_highRes.data[:] = 0
        binary_car_larger_highRes.data[1:-1, 1:-1] = 1
        so_taper_larger_highRes = so_window.create_apodization(binary_car_larger_highRes, apo_type="C1", apo_radius_degree=self.apod_width)
        apodized_patch.data[:] = initial_patch.data[:] * so_taper_larger_highRes.data[:]
        
        largerPatch_HD_pwConvolved = self.convolve_largerPatch_with_pw(apodized_patch)
        innerPatch = self.get_innerPatch(largerPatch_HD_pwConvolved)

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

    def get_theory_for_stitching(self,template_cls, template_ells, patch_cls, patch_ells, \
                                      template_minimization_index, patch_minimization_index, \
                                      template_ell0_index=3102, patch_ell0_index=15,
                                      seed = 3):
        """
        Generates the alms for the small-scale extension of particular patch of sky.
    
        Parameters
        ----------
        output_path : str
            Path to the output saved files for the small-scale extension Fourier components.
        template_cls : ndarray
            Input cls for template theory we want to match.
        template_ells : ndarray
            Input ells for template theory we want to match.
        patch_cls : ndarray
            Input cls for patch theory we want to match.
        patch_ells : ndarray
            Input ells for patch theory we want to match.
        template_minimization_index : int
            The index in template_ells that matches the ell value we want to minimize the distance to
        patch_minimization_index : int
            The index in patch_ells that matches the ell value we want to minimize the distance to
        template_ell0_index : int
            The index in template_ells that matches ell_0. Default is 3102.
        patch_ell0_index : int
            The index in patch_ells that matches ell_0. Default is 15, which corresponds to ell = 3100 assuming delta_ell = 200.
        seed : int
            The seed used to generate the alms
    
        Returns
        -------
        alms_theory : ndarray
            Final alms for small-scale extension fit to the two ell values provided.
        """
            
        self.logger.info(f"Matching template at ell={template_ells[template_minimization_index]} to patch at ell={patch_ells[patch_minimization_index]}")
        self.logger.info(f"template ell0={template_ells[template_ell0_index]} w/ patch ell0={patch_ells[patch_ell0_index]}")
        # A fixes the point at ell0

        patch_cls_avg_ell0 = (patch_cls[patch_ell0_index+1]+patch_cls[patch_ell0_index]+patch_cls[patch_ell0_index-1])/3
        template_cls_avg_ell0 = (template_cls[template_ell0_index+1]+template_cls[template_ell0_index]+template_cls[template_ell0_index-1])/3
        # This sets A based on the patch at ell0
        A_val = patch_cls_avg_ell0/template_cls_avg_ell0
        self.logger.info(f"% Difference at ~ell0 is now {100*(patch_cls_avg_ell0 - A_val * template_cls_avg_ell0)/(A_val * template_cls_avg_ell0)}")
        
        # Sets n do minimize the distance between theory and patch at minimization_index
        initial_guess = [0.0]
        bounds = [(-5, 5)]
        def func_to_minimize(n_kappa):
            theory_cls = ((template_ells/template_ell0_index)**n_kappa) * A_val * template_cls
            patch_cls_avg_minimization = (patch_cls[patch_minimization_index+1]+patch_cls[patch_minimization_index]+patch_cls[patch_minimization_index-1])/3
            theory_cls_avg_minimization = (theory_cls[template_minimization_index+1]+theory_cls[template_minimization_index]+theory_cls[template_minimization_index-1])/3
            return (theory_cls_avg_minimization - patch_cls_avg_minimization)**2
        result = scipy.optimize.minimize(func_to_minimize, initial_guess, method='Nelder-Mead', bounds=bounds)
    
        n_val = result.x[0]
        self.logger.info(f"n = {n_val:.10e}")
        self.logger.info(f"A = {A_val:.10e}")
    
        theory_cls = ((template_ells/template_ell0_index)**n_val) * A_val * template_cls

        self.logger.info(f"Now generating alms from theory")
        alms_theory = curvedsky.rand_alm(theory_cls,lmax=self.l_max,seed=seed)

        return alms_theory, template_ells, theory_cls

    def get_S10_for_stitching(self, upsampled_patch):
        """
        Generates the alms for the S10 patch of sky.
    
        Parameters
        ----------
        output_path : str
            Path to the output saved files for the patch Fourier components.
        upsampled_patch : enmap
            Input foreground patch of sky
    
        Returns
        -------
        alms_for_stitching_resized : ndarray
            Final alms for original patch resized to match desired output.
        """
        
        shape, wcs = self.get_shape_wcs(self.new_res, self.ra, self.dec, self.final_width, height=self.final_width)
        window_ones = self.enmap2pspy(enmap.ones(shape, wcs))
        lmax_raw = min(self.l_max,int(window_ones.get_lmax_limit()))

        self.logger.info(f"Getting alms from upsampled S10 patches")
        imap = enmap.project(upsampled_patch.data, shape, wcs)
        alms_for_stitching = sph_tools.get_alms(self.enmap2pspy(imap), window_ones, niter=0, lmax=lmax_raw)

        self.logger.info(f"Resizing as alms_for_stitching_resized from {lmax_raw} to {self.l_max}")
        alms_for_stitching_resized = hp.sphtfunc.resize_alm(alms_for_stitching, 
                                                            lmax=lmax_raw, lmax_out=self.l_max,
                                                            mmax=lmax_raw, mmax_out=self.l_max)
        
        return alms_for_stitching_resized

    def stitch_alms(self, alms_theory, alms_S10, l_cutoff):
        """
        Combines two sets of alms to generate the desired patch of sky with Fourier components from one until some ell, at which point the Fourier components are from the other set.
    
        Parameters
        ----------
        alms_theory : ndarray
            The Fourier components used at high-ell.
        alms_S10 : ndarray
            The Fourier components used at low-ell.
        output_path : str
            Path to save real-space patch coming from both sets of Fourier components.
        l_cutoff : int
            Value of ell beyond which alms_S10 is replaced by values from alms_theory.
    
        Returns
        -------
        patch_map_so : so_map
            Real-space patch coming from both sets of Fourier components.
        """
            
        self.logger.info(f"Stitching alms")
        l, m = hp.Alm.getlm(self.l_max)
        alms_combined = np.where(l >= l_cutoff, alms_theory, alms_S10)

        self.logger.info(f"Generating patch from alms")
        shape, wcs = self.get_shape_wcs(self.new_res, self.ra, self.dec, self.final_width, height=self.final_width)
        window_ones = enmap.ones(shape, wcs)
        
        patch_map = curvedsky.alm2map(alm=alms_combined, map=window_ones)
        patch_map_so = self.enmap2pspy(patch_map)
        
        return patch_map_so

    ############################################# CIB

    def get_flux2temp_unit_conversion(self, freq):
        # divide map in Jy/str by this factor to get in delta T / T units:
        unit_conversions = {30: 7.364967e7, 90: 5.526540e8, 148: 1.072480e9, 219: 1.318837e9, 277: 1.182877e9, 350: 8.247628e8}
        return unit_conversions[freq]
    
    def uK_to_mJy_per_str(self, x, freq, TCMB=2.7255e6):
        return (x / TCMB) * self.get_flux2temp_unit_conversion(freq) * 1e3
    
    def make_catalog_from_sims(self, sims, sigma_pix_frac=0.2, seed=0):
        """
        Extracts sources from foreground patch with point sources located at center of pixels
    
        Parameters
        ----------
        sims : dict
            Dictionary of enmaps for each frequency
        component : str
            Foreground component
    
        Returns
        -------
        catalog : pd.df
            Source catalog
        """
        
        width = self.final_width + 2*self.apod_width
        
        freqs = sorted(list(sims.keys()))
        max_freq = np.max(freqs)
        shape = sims[max_freq].data.shape
        wcs = sims[max_freq].data.wcs
        posmap = np.rad2deg(enmap.posmap(shape, wcs))
        ra_map = posmap[1]
        dec_map = posmap[0]
    
        flux_sims = {}
        for freq in freqs:
            flux_sims[freq] = self.uK_to_mJy_per_str(sims[freq].data.copy(), freq) * sims[freq].data.pixsizemap()
        nonzero_mask = np.greater(flux_sims[max_freq], 0)
    
        catalog_dict = {'ra_deg': ra_map[nonzero_mask], 'dec_deg': dec_map[nonzero_mask]}
        for freq in freqs:
            catalog_dict[f'{freq}GHz_flux'] = flux_sims[freq][nonzero_mask]
        catalog = pd.DataFrame(catalog_dict)

        catalog['ra_deg'], catalog['dec_deg'] = self.add_gauss_scatter_to_coords(catalog['ra_deg'].values, catalog['dec_deg'].values, shape, wcs, sigma_pix_frac, seed)
        
        return catalog

    def add_gauss_scatter_to_coords(self, ras, decs, shape, wcs, sigma_pix_frac, seed):
        """
        Adds scatter to the location of point sources
    
        Parameters
        ----------
        ras : ndarray
            RA values of point sources in catalog
        decs : ndarray
            DEC values of point sources in catalog
        shape : 
            shape for enmap of input resolution and map geometry
        wcs : 
            wcs for enmap of input resolution and map geometry
        sigma_pix_frac : float
            Sigma pixel fraction of normal distribution used to add scatter to point sources
        seed : int
            The seed used to perform the scatter
    
        Returns
        -------
        catalog : pd.df
            Source catalog
        """
        
        # get avg width and height of each pixel
        avg_pixel_width, avg_pixel_height = np.rad2deg(enmap.pixshape(shape, wcs))
        # add some scatter around ra/dec within width/height of pixel:
        np.random.seed(seed)
        ra_scatter = np.random.normal(loc=0, scale=avg_pixel_width * sigma_pix_frac, size=len(ras))
        dec_scatter = np.random.normal(loc=0, scale=avg_pixel_height * sigma_pix_frac, size=len(decs))
        np.random.seed(None) # un-set the seed
        random_ras = ras + ra_scatter
        random_decs = decs + dec_scatter
        return random_ras, random_decs
