import logging
import os
import numpy as np
import healpy as hp
from pspy import so_map, so_window, sph_tools
from pixell import reproject, enmap, curvedsky, utils
import pandas as pd
import scipy
import camb
from .power_spectrum import Spectra

class Foregrounds:
    """
    A class for generating diffuse and discrete foreground sky patches from full-sky maps,
    and applying the appropriate catalog fixes for CIB and small-scale extensions for kSZ
    and lensing convergence.
    """
    
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
    
    def __init__(self, ra, dec, final_width, apod_width, 
                 new_res, l_max, log_level=logging.INFO):
        """
        Initialize a foreground generator.
        
        
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
        new_res : float
            Target pixel resolution of final maps, in arcminutes.
        l_max : int
            Maximum multipole moment (ℓ) for spherical harmonic transforms.
        log_level : int, optional
            Logging verbosity level (default: logging.INFO).
        """
        
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

    def load_fullsky(self, frequency, data_path,
                     scaling_factor = 1.0, deconvolve_pixel_window = True):    
        """
        Load and process a full-sky foreground map at a given frequency.

    
        Parameters
        ----------
        frequency : int or None
            Frequency of the foreground (30, 90, 148, 219, 277, 350 GHz), or None for lensing convergence (kappa).
        data_path : str
            Path to the input full-sky HEALPix map file.
        scaling_factor : float, optional
            Multiplicative factor applied to the map values. Default is 1.0.
        deconvolve_pixel_window : bool, optional
            If True, deconvolves the pixel window function. Default is True.
        
        
        Returns
        -------
        fullsky_map : so_map
            Processed full-sky map in μK.
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
        """"
        Extract an apodized larger patch from the full-sky map.

        
        Parameters
        ----------
        fullsky_map : so_map
            Input full-sky HEALPix map.
        fullsky_res : float
            Resolution of the full-sky map, in arcminutes.
        lmax_to_extract : int
            Maximum multipole used for spherical harmonic extraction.
        
        
        Returns
        -------
        apodized_patch : so_map
            Square patch centered at (self.ra, self.dec), with width (self.final_width + 2*self.apod_width) in μK, apodized by self.apod_width.
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
        Upsample an apodized patch to the target resolution.
        
        
        Parameters
        ----------
        apodized_patch_old : so_map
            Input apodized patch at fullsky_res arcminutes per pixel.
        fullsky_res : float
            Resolution of the input map in arcminutes.
        
        
        Returns
        -------
        so_patch_larger_upsampled : so_map
            Patch of width (self.final_width + 2*self.apod_width) resampled to self.new_res arcminutes per pixel.
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
        Convolve a patch with its pixel window function.
        
        
        Parameters
        ----------
        patch : so_map
            Input patch at resolution self.new_res.
        
        
        Returns
        -------
        pwConvolved_patch : so_map
            Patch convolved with pixel window function.
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
        

        shape, wcs = self.get_shape_wcs(self.new_res, self.ra, self.dec, self.final_width, height=self.final_width)
        inner_patch = enmap.project(patch.data, shape, wcs)
        self.logger.info(f"Cut inner patch down")
        
        return self.enmap2pspy(inner_patch)

    def generate_diffuse_foreground(self, component, frequency, fullsky_deconvolved_path = None, S10_largeApodized_path = None, return_Apodized_patch = False):
        """
        Generate a diffuse foreground patch (tSZ, kSZ, or kappa).
        
        
        Parameters
        ----------
        component : str
            Foreground type: "tSZ", "kSZ", or "kappa".
        frequency : int or None
            Frequency of foreground (30–350 GHz), or None for kappa (lensing convergence).
        fullsky_deconvolved_path : str, optional
            Path to a deconvolved full-sky HEALPix map. Default is None.
        S10_largeApodized_path : str, optional
            Path to an S10 patch of width (self.final_width + 2*self.apod_width) apodized by self.apod_width on each side. Default is None.
        return_Apodized_patch : bool, optional
            If True, also return the larger apodized patch. Default is False.
        
        
        Returns
        -------
        innerPatch : so_map or (so_map, so_map)
            If return_Apodized_patch is False, returns the final square patch of width self.final_width degrees at self.new_res resolution.
            If return_Apodized_patch is True, returns a tuple (largerPatch, innerPatch) of S10 patch of width (self.final_width + 2*self.apod_width) apodized by self.apod_width on each side (able to later be used as the input data in S10_largeApodized_path) and the final square patch of width self.final_width degrees at self.new_res resolution. 
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

        if return_Apodized_patch:
            return largerPatch, innerPatch

        return innerPatch

    def make_catalog(self, data_path, component):
        """
        Construct a source catalog for a given component from S10 full-sky catalogs.
        
        
        Parameters
        ----------
        data_path : str
            Directory containing component CSV or binary files from Lambda.
        component : str
            Component type: "CIB", "radio", or "SZ".
        
        
        Returns
        -------
        catalog : pandas.DataFrame
            Source catalog containing RA [deg], DEC [deg], and fluxes [mJy] at available frequencies.
        """
        
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
        elif component == 'SZ':            
            ncols = 48
            dtype = np.float32
            filtered = []
            
            with open(data_path + "halo_sz.binary", "rb") as f:
                while True:
                    record = np.fromfile(f, dtype=dtype, count=ncols)
                    if record.size < ncols:
                        break
                    if ra_min <= record[1] <= ra_max and dec_min <= record[2] <= dec_max:
                        filtered.append(record)
            
            filtered = np.array(filtered)
            
            columns = [
                "redshift", "ra_deg", "dec_deg",
                "x_mpc", "y_mpc", "z_mpc",
                "vx_kms", "vy_kms", "vz_kms",
                "Mfof",
                "Mvir", "Mgas_vir", "Rvir", "TSZ_Rvir", "KSZ_Rvir",
                "SZ_148_Rvir", "SZ_219_Rvir", "SZ_277_Rvir", "SZ_30_Rvir", "SZ_90_Rvir", "SZ_350_Rvir",
                "Mvir_R200", "Mgas_vir_R200", "R200", "TSZ_R200", "KSZ_R200",
                "SZ_148_R200", "SZ_219_R200", "SZ_277_R200", "SZ_30_R200", "SZ_90_R200", "SZ_350_R200",
                "Mvir_R500", "Mgas_vir_R500", "R500", "TSZ_R500", "KSZ_R500",
                "SZ_148_R500", "SZ_219_R500", "SZ_277_R500", "SZ_30_R500", "SZ_90_R500", "SZ_350_R500",
                "Mstar", "rho_gas_central", "T_central", "P_central", "phi_central"
            ]
            df = pd.DataFrame(filtered, columns=columns)
            df = df.reset_index(drop=True)
            
            return df

    def place_sources_in_largerPatch(self, catalog, frequency, scaling_factor):
        """
        Place catalog sources into an empty sky patch.
        
        
        Parameters
        ----------
        catalog : pandas.DataFrame
            Table of sources with RA, DEC (degrees), and flux densities [mJy].
        frequency : int
            Frequency of interest (30–350 GHz).
        scaling_factor : float
            Multiplicative factor applied to flux densities.
        
        
        Returns
        -------
        initial_patch : so_map
            Square patch of width (final_width + 2*apod_width), at self.new_res resolution, containing simulated point sources.
        """

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

    def generate_discrete_foreground(self, frequency, catalog = None, scaling_factor = 1.0, return_Apodized_patch = False, S10_largeApodized_path = None):
        """
        Generate a discrete foreground patch from a catalog or file.
        
        
        Parameters
        ----------
        frequency : int
            Frequency of the foreground (30–350 GHz).
        catalog : pandas.DataFrame, optional
            Source catalog with RA, DEC, and fluxes. Default is None.
        scaling_factor : float, optional
            Multiplicative factor applied to source amplitudes. Default is 1.0.
        return_Apodized_patch : bool, optional
            If True, also return the larger apodized patch. Default is False.
        S10_largeApodized_path : str, optional
            Path to an S10 patch of width (self.final_width + 2*self.apod_width) apodized by self.apod_width on each side. Default is None.
        
        
        Returns
        -------
        innerPatch : so_map or (so_map, so_map)
            If return_Apodized_patch is False, returns the final square patch of width self.final_width degrees at self.new_res resolution.
            If return_Apodized_patch is True, returns a tuple (apodized_patch, innerPatch) of S10 patch of width (self.final_width + 2*self.apod_width) apodized by self.apod_width on each side (able to later be used as the input data in S10_largeApodized_path) and the final square patch of width self.final_width degrees at self.new_res resolution. 
        """
        
        if catalog is not None and not catalog.empty:
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
        elif S10_largeApodized_path != None:
            apodized_patch = so_map.read_map(S10_largeApodized_path)
        else:
            self.logger.info(f"No input data included")
        
        largerPatch_HD_pwConvolved = self.convolve_largerPatch_with_pw(apodized_patch)
        innerPatch = self.get_innerPatch(largerPatch_HD_pwConvolved)

        if return_Apodized_patch:
            return apodized_patch, innerPatch

        return innerPatch

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
    
    def make_apod_window(self, shape, wcs, apod_width_deg, map_type='so_map'):
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

    def get_theory_for_stitching(self,template_cls, template_ells, patch_cls, patch_ells, \
                                      template_minimization_index, patch_minimization_index, \
                                      template_ell0_index=3102, patch_ell0_index=15,
                                      seed = 3):
        """
        Fits a template theory spectrum to a patch spectrum and generates theory alms for the new theory spectrum.
        
        
        Parameters
        ----------
        template_cls : ndarray
            Power spectrum values of the template theory.
        template_ells : ndarray
            Multipoles corresponding to template_cls.
        patch_cls : ndarray
            Power spectrum values of the extracted patch.
        patch_ells : ndarray
            Multipoles corresponding to patch_cls.
        template_minimization_index : int
            Index in template_ells used for normalization. Corresponds to the value of ℓ used to optimize n in Equations 1 and 2.
        patch_minimization_index : int
            Index in patch_ells used for normalization. Corresponds to the value of ℓ used to optimize n in Equations 1 and 2.
        template_ell0_index : int, optional
            Index in template_ells used for normalization. Corresponds to the value of ℓ_0 used to define A in Equations 1 and 2. Default is 3102.
        patch_ell0_index : int, optional
            Index in patch_ells used for normalization. Corresponds to the value of ℓ_0 used to define A in Equations 1 and 2. Default is 15.
        seed : int, optional
            Random seed used in alm generation. Default is 3.
        
        
        Returns
        -------
        alms_theory : ndarray
            Complex spherical harmonic coefficients up to self.l_max.
        template_ells : ndarray
            Ells associated with rescaled template power spectrum matching the patch.
        theory_cls : ndarray
            Rescaled template power spectrum matching the patch.
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
        Compute alms from an upsampled S10 patch.
        
        
        Parameters
        ----------
        upsampled_patch : so_map
            Input S10 patch at self.new_res arcminutes resolution.
        
        
        Returns
        -------
        alms_for_stitching_resized : ndarray
            Spherical harmonic coefficients of S10 input patch resized to match self.l_max.
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
        Stitch two sets of alms into a combined map.
        
        
        Parameters
        ----------
        alms_theory : ndarray
            Spherical harmonic coefficients containing high-ℓ modes.
        alms_S10 : ndarray
            Spherical harmonic coefficients containing low-ℓ modes.
        l_cutoff : int
            Multipole cutoff. Modes with ℓ < l_cutoff are taken from alms_S10, while modes with ℓ ≥ l_cutoff are taken from alms_theory.
        
        
        Returns
        -------
        patch_map_so : so_map
            Real-space square patch of width self.final_width degrees at resolution self.new_res.
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
        """
        Get unit conversion factor from Jy/sr to ΔT/T.
        
        
        Parameters
        ----------
        freq : int
            Frequency in GHz.
        
        
        Returns
        -------
        factor : float
            Conversion factor (multiplicative).
        """
        # divide map in Jy/str by this factor to get in delta T / T units:
        unit_conversions = {30: 7.364967e7, 90: 5.526540e8, 148: 1.072480e9, 219: 1.318837e9, 277: 1.182877e9, 350: 8.247628e8}
        return unit_conversions[freq]
    
    def uK_to_mJy_per_str(self, x, freq):
        """
        Convert a map from μK to mJy/sr.
        
        
        Parameters
        ----------
        x : ndarray
            Input map in μK.
        freq : int
            Frequency in GHz.
        
        
        Returns
        -------
        flux_map : ndarray
            Map in mJy/sr.
        """
        return (x / self.T_CMB) * self.get_flux2temp_unit_conversion(freq) * 1e3
    
    def make_catalog_from_sims(self, sims, sigma_pix_frac=0.2, seed=0):
        """
        Build a source catalog with source positions from a simulated CIB maps.
        
        
        Parameters
        ----------
        sims : dict
            Dictionary mapping frequency (GHz) to pixell enmap of sky patch.
        sigma_pix_frac : float, optional
            Fraction of pixel size used as Gaussian scatter in source positions. Default is 0.2.
        seed : int, optional
            Random seed for reproducibility. Default is 0.
        
        
        Returns
        -------
        catalog : pandas.DataFrame
            Source catalog with scattered RA, DEC (degrees) and flux densities (mJy).
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
        Apply Gaussian scatter to input sky coordinates.
        
        
        Parameters
        ----------
        ras : ndarray, shape (N,)
            RA coordinates in degrees (0..360) for N sources.
        decs : ndarray, shape (N,)
            Dec coordinates in degrees (-90..90) for N sources.
        shape : tuple of int
            Shape of the enmap array (ny, nx).
        wcs : WCS
            World coordinate system.
        sigma_pix_frac : float
            Fraction of pixel size used as Gaussian scatter in source positions.
        seed : int or None
            Random seed for reproducibility.
        
        
        Returns
        -------
        random_ras : ndarray, shape (N,)
            RA coordinates after scatter is implemented (degrees). Same length and ordering as input ras.
        random_decs : ndarray, shape (N,)
            Dec coordinates after scatter is implemented (degrees). Same length and ordering as input decs.
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

    def get_kappa_theory(self, ini_file = 'S10_data/bode_almost_wmap5_params_highKeta.ini'):
        """
        Compute CAMB with high-accuracy settings for unlensed CMB power and lensing potential power spectrum.
    
        Parameters
        ----------
        ini_file : str, optional
            Path to a CAMB-style .ini parameter file. Default is 'S10_data/bode_almost_wmap5_params_highKeta.ini'.
    
        Returns
        -------
        main_data_text : ndarray, shape (N_ell, 10)
            Table that stacks (per-ℓ) the following columns:
                [ell,
                 lensed_TT, lensed_EE, lensed_BB, lensed_TE,
                 unlensed_TT, unlensed_EE, unlensed_BB, unlensed_TE,
                 kappa_cl]
            - `ell` are integer multipoles (0..N-1).
            - lensed_* and unlensed_* are CAMB Cl arrays (units: μK²).
            - `kappa_cl` is the κ-κ power spectrum computed from the lensing potential
              (dimensionless), computed as (ℓ(ℓ+1))^2 * C_ψψ(ℓ) / 4.
    
        main_data_dat : ndarray, shape (N_ell, 5)
            Table for commonly-plotted D_ell quantities:
                [ell,
                 D_ell_unlensed_TT, D_ell_unlensed_EE, D_ell_unlensed_BB, D_ell_unlensed_TE]
            where D_ell = ℓ(ℓ+1) C_ell / (2π). Units are μK².
        """
        
        pars = camb.read_ini(ini_file)
        # high-accuracy settings
        pars.set_matter_power(kmax=10, k_per_logint=130)
        pars.set_for_lmax (self.l_max, \
            lens_potential_accuracy =30 , \
            lens_margin =2050)
        pars.set_accuracy ( AccuracyBoost =1.1 , \
            lSampleBoost =3.0 , lAccuracyBoost =3.0 , \
            DoLateRadTruncation = False, min_l_logl_sampling=10000 )
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params("mead2016")
        results = camb.get_results(pars)
        
        lensed = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
        unlensed = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['unlensed_scalar']
        
        ells = np.arange(0, lensed.shape[0])
        lensCL = results.get_lens_potential_cls(lmax=lensed.shape[0] + 2)
        kk = (ells * (ells + 1))**2 * lensCL[2:len(ells)+2, 0] / 4.0
        main_data_text = np.column_stack([
            ells,
            lensed[:,0], lensed[:,1], lensed[:,2], lensed[:,3],
            unlensed[:,0], unlensed[:,1], unlensed[:,2], unlensed[:,3],
            kk
        ])
        
        main_data_dat = np.column_stack([
            ells, 
            unlensed[:,0] * (ells * (ells+1)) / (2*np.pi), 
            unlensed[:,1] * (ells * (ells+1)) / (2*np.pi), 
            unlensed[:,2] * (ells * (ells+1)) / (2*np.pi), 
            unlensed[:,3] * (ells * (ells+1)) / (2*np.pi)
        ])
        
        return main_data_text, main_data_dat

    def extend_to_small_scales(self, S10_patch, patch_type, template_ells, template_cls, mbb_inv, binning_file, spectra_apod_width=1.0, spectra_final_width=10.0):
        """
        Extend an S10-scale patch to higher-resolution small-scale modes using a template theory.
    
        Parameters
        ----------
        S10_patch : enmap
            The original S10 patch to be extended.
        patch_type : str
            Type of patch to extend. Determines which indices, deconvolution flags and ℓ-cutoffs are used:
              - 'kappa'  : uses template_minimization_index=3902, patch_minimization_index=19, l_cutoff=4000
              - 'kSZ'    : uses template_minimization_index=8102, patch_minimization_index=40, l_cutoff=8000
        template_ells : ndarray
            Multipoles corresponding to the high-resolution template spectrum.
        template_cls : ndarray
            High-resolution template power spectrum values to be matched
        mbb_inv : array-like
            Inverse of the mode-coupling / binning matrix for deconvolving binned estimates.
        binning_file : str
            Path to a binning specification file.
        spectra_apod_width : float, optional
            Apodization radius (degrees) to use when computing power spectra for the S10 patch. Default is 1.0.
        spectra_final_width : float, optional
            Angular width (degrees) used when computing S10 spectra. Default is 10.0 deg.
    
        Returns
        -------
        HD_patch : so_map
            The stitched, high-resolution patch at self.new_res with width self.final_width.
        dict : dictionary with keys:
            'l' : ndarray
                Multipoles of the matched small-scale theory.
            'cl': ndarray
                The matched theory Cl values.
        """
            
        spectra_HD = Spectra(
            ra = self.ra,
            dec = self.dec,
            final_width = spectra_final_width,
            res = self.new_res,
            apod_width = spectra_apod_width,
            l_max = self.l_max)
        
        if patch_type == 'kappa':
            S10_kappa_cls, S10_kappa_ells = spectra_HD.get_foreground_power(S10_patch, mbb_inv, binning_file, deconvolve_pw = False, type_Cl = True)
            
            smallScale_kappa_alms, smallScale_kappa_ells, smallScale_kappa_cls = self.get_theory_for_stitching(template_cls,
                                                                                          template_ells,
                                                                                          S10_kappa_cls, 
                                                                                          S10_kappa_ells, \
                                                                                          template_minimization_index = 3902, 
                                                                                          patch_minimization_index = 19)
        
            kappa_alms_for_stitching_resized = self.get_S10_for_stitching(S10_patch)
            HD_kappa_patch = self.stitch_alms(alms_theory = smallScale_kappa_alms, alms_S10 = kappa_alms_for_stitching_resized, 
                                                               l_cutoff = 4000)
            return HD_kappa_patch, {"l": smallScale_kappa_ells, "cl": smallScale_kappa_cls}
            
        elif patch_type == 'kSZ':
            S10_kSZ_cls, S10_kSZ_ells = spectra_HD.get_foreground_power(S10_patch, mbb_inv, binning_file, deconvolve_pw = True, type_Cl = True)
            
            smallScale_kSZ_alms, smallScale_kSZ_ells, smallScale_kSZ_cls = self.get_theory_for_stitching(template_cls,
                                                                                  template_ells,
                                                                                  S10_kSZ_cls, 
                                                                                  S10_kSZ_ells, \
                                                                                  template_minimization_index = 8102, 
                                                                                  patch_minimization_index = 40)
            
            kSZ_alms_for_stitching_resized = self.get_S10_for_stitching(S10_patch)
            HD_kSZ_patch_extended = self.stitch_alms(alms_theory = smallScale_kSZ_alms, alms_S10 = kSZ_alms_for_stitching_resized, 
                                                           l_cutoff = 8000)
            return HD_kSZ_patch_extended, {"l": smallScale_kSZ_ells, "cl": smallScale_kSZ_cls}

    def place_CIB_sources_in_largerPatch(self, catalog, frequency, scaling_factor, CIB_resolution):
        """
        Place CIB source catalog entries into a CAR patch at a specified resolution, based on CIB model.
    
        Parameters
        ----------
        catalog : pandas.DataFrame
            Catalog containing at least columns ['ra_deg', 'dec_deg', '<freq>GHz_flux'], with flux in mJy.
        frequency : int
            Frequency in GHz (e.g. 30, 90, 148, 219, 277, 350).
        scaling_factor : float
            Multiplicative scaling applied to the map after placing sources.
        CIB_resolution : float
            Pixel resolution used to place patches in CAR patch.
    
        Returns
        -------
        initial_patch : so_map
            The sky patch at CIB_resolution of width (self.final_width + 2*self.apod_width) in μK units for the given frequency.
        """
        
        width = self.final_width + 2*self.apod_width
        ra_min = self.ra - width/2
        ra_max = self.ra + width/2
        dec_min = self.dec - width/2
        dec_max = self.dec + width/2
    
        initial_patch = so_map.car_template(1, ra_min, ra_max, dec_min, dec_max, CIB_resolution)
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

    def make_CIB_model_catalog(self, CIB_model, CIB_catalog_original, frequencies=[30,90,148,219,277,350], sigma_pix_frac=0.2, seed=0):
        """
        Build a CIB source catalog by placing model sources at a chosen resolution and extracting fluxes.
    
        Parameters
        ----------
        CIB_model : int
            Model index (1 or 2) selecting the resolution for source placement:
              - 1 : HEALPix-based placement using nside=8192 resolution
              - 2 : high-resolution grid with 0.25 arcmin pixels
        CIB_catalog_original : pandas.DataFrame
            Input catalog of CIB sources (should contain ra_deg, dec_deg, and '<freq>GHz_flux' in mJy).
        frequencies : list of int, optional
            List of frequencies (GHz) to include in the generated sims and final catalog. Default is [30, 90, 148, 219, 277, 350].
        sigma_pix_frac : float, optional
            Fraction of pixel size used as Gaussian scatter in source positions. Default 0.2.
        seed : int, optional
            Random seed for reproducibility. Default 0.
    
        Returns
        -------
        CIB_catalog : pandas.DataFrame
            Catalog DataFrame of extracted sources with columns ['ra_deg', 'dec_deg', '30GHz_flux', '90GHz_flux', ...], with flux in mJy.
        """
        
        CIB_resolutions = [hp.nside2resol(8192, arcmin=True), 0.25]
        
        CIB_sim = {
            freq: self.place_CIB_sources_in_largerPatch(
                catalog=CIB_catalog_original,
                frequency=freq,
                scaling_factor=0.75,
                CIB_resolution=CIB_resolutions[CIB_model - 1]
            )
            for freq in frequencies
        }
    
        CIB_catalog = self.make_catalog_from_sims(sims = CIB_sim, sigma_pix_frac=sigma_pix_frac, seed=seed)
        return CIB_catalog