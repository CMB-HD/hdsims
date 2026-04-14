"""Contains general functions for the simulated maps."""

import numpy as np
import healpy as hp
from pixell import enmap, utils as putils, wcsutils, curvedsky as cs, powspec, reproject
from pspy import so_map, so_window
from . import utils, fgcatalogs


def get_shape_wcs(res, ra_ctr, dec_ctr, width, height=None):
    """Return the `shape` and `wcs` for a `pixell.enmap.ndmap` of the
    given resolution `res`, centered at RA, dec = (`ra_ctr`,
    `dec_ctr`) with a width of `width` and height of `height`
    (or a square map if the height is not provided).

    Parameters
    ----------
    res : float
        The resolution (pixel size) of the map, in arcmin.
    ra_ctr, dec_ctr : float
        The RA and dec coordinates of the patch center, in degrees.
    width : float
        The width of the patch (in the "RA direction"), in degrees.
    height : float, optional
        The height of the patch (in the "dec direction"), in degrees.
        If not provided, the height is set to equal the width of the patch.

    Returns
    -------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec and RA
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the given patch of sky at the given resolution.

    See Also
    --------
    pixell.enmap.geometry
    """
    height = width if (height is None) else height
    ra_min = ra_ctr - (width / 2)
    ra_max = ra_ctr + (width / 2)
    dec_min = dec_ctr - (height / 2)
    dec_max = dec_ctr + (height / 2)
    coord_box = np.deg2rad(np.array([[dec_min, ra_max], [dec_max, ra_min]]))
    shape, wcs = enmap.geometry(pos=coord_box, res=res * putils.arcmin, proj='car')
    return shape, wcs


def get_pixel_positions(ra, dec, shape, wcs):
    """Return pairs of x and y pixel values corresponding to each pair of
    R.A. and dec. coordinates for a `pixell.enmap.ndmap` with the given 
    `shape` and `wcs`.

    Parameters
    ----------
    ra, dec : float or array_like of float
        The RA & dec coordinate pair(s), in degrees.
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    x, y : array_like of int
        The x (R.A. direction) and y (dec. direction) pixel values 
        corresponding to the given R.A. & dec. coordinates.

    See Also
    --------
    pixell.enmap.sky2pix
    pixell.enmap.pix2sky
    get_coord_positions : 
        R.A. and dec. pair(s) corresponding to pixel pair(s).
    coord2pix : 
        Single pair of x and y pixels given a single pair of R.A. and dec.
        coordinates.
    """
    ras = np.atleast_1d(ra)
    decs = np.atleast_1d(dec)
    if len(ras) != len(decs):
        errmsg = ("You must provide the same number of RA and dec values.")
        raise ValueError(errmsg)
    coords = np.deg2rad([decs, ras])
    pixels = np.round(enmap.sky2pix(shape, wcs, coords)).astype(int)
    y = pixels[0]
    x = pixels[1]
    return x, y


def get_coord_positions(x_pixels, y_pixels, shape, wcs):
    """Return pairs of R.A. and dec. coordinates corresponding to each
    pair of x and y pixel values for a `pixell.enmap.ndmap` with the
    given `shape` and `wcs`.

    Parameters
    ----------
    x, y : int or array_like of int
        The x (R.A. direction) and y (dec. direction) pixel values
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    ra, dec : array_like of float
        The R.A. and dec. coordinate pair(s), in degrees, corresponding to the
        given x and y pixel values.

    See Also
    --------
    pixell.enmap.sky2pix
    pixell.enmap.pix2sky
    get_pixel_positions :
        x and y pixel pair(s) corresponding to R.A. and dec. coordinate
        pair(s).
    pix2coord :
        Single pair of R.A. and dec. coordinates given a single pair of x
        and y pixels.
    """
    x = np.atleast_1d(x_pixels)
    y = np.atleast_1d(y_pixels)
    if len(x) != len(y):
        errmsg = ("You must provide the same number of `x_pixels` and `y_pixels`.")
        raise ValueError(errmsg)
    pixels = np.array([y, x])
    coords = np.rad2deg(enmap.pix2sky(shape, wcs, pixels))
    decs = coords[0]
    ras = coords[1]
    return ras, decs


def coord2pix(ra, dec, shape, wcs):
    """Return the values of the x and y pixel corresponding to the given
    R.A. and dec. coordinate pair for a `pixell.enmap.ndmap` with the
    given `shape` and `wcs`.

    Parameters
    ----------
    ra, dec : float
        The R.A. and dec. coordinate pair, in degrees.
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    x, y : int
        The x (R.A. direction) and y (dec. direction) pixel values
        corresponding to the given pair of R.A. and dec. coordinates.

    See Also
    --------
    pixell.enmap.sky2pix
    pixell.enmap.pix2sky
    pix2coord :
        Single pair of R.A. and dec. coordinates given a single pair of x
        and y pixels.
    get_pixel_positions :
        x and y pixel pair(s) corresponding to R.A. and dec. coordinate
        pair(s).
    """
    x_px, y_px = get_pixel_positions(ra, dec, shape, wcs)
    return x_px[0], y_px[0]


def pix2coord(x, y, shape, wcs):
    """Return a pair of R.A. and dec. coordinates corresponding to the
    pair of x and y pixel values for a `pixell.enmap.ndmap` with the given
    `shape` and `wcs`.

    Parameters
    ----------
    x, y : int
        The x (R.A. direction) and y (dec. direction) pixel values
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    ra, dec : float
        The RA & dec coordinate pair, in degrees, corresponding to the
        given x & y pixel values.

    See Also
    --------
    pixell.enmap.sky2pix
    pixell.enmap.pix2sky
    coord2pix :
        Single pair of x and y pixels given a single pair of R.A. and dec.
        coordinates.
    get_coord_positions :
        R.A. and dec. pair(s) corresponding to pixel pair(s).
    """
    ras, decs = get_coord_positions(x, y, shape, wcs)
    return ras[0], decs[0]


def get_patch_corner_coords(ra_ctr, dec_ctr, width, height=None):
    """Return the minimum and maximum R.A. and dec. coordinates for a
    patch of sky.

    Parameters
    ----------
    ra_ctr, dec_ctr : float
        The R.A. and dec. coordinates of the center of the patch.
    width : float
        The width of the patch, in the same units as `ra_ctr` and `dec_ctr`.
    height : float, optional
        The height of the patch, in the same units as `ra_ctr` and `dec_ctr`.
        By default, the `height` is set equal to the `width` (for a square
        patch of sky).

    Returns
    -------
    ra_min, ra_max, dec_min, dec_max : float
        The minimum and maximum R.A. and dec., respectively, in the same
        units as `ra_ctr` and `dec_ctr`.

    See Also
    --------
    get_map_corner_coords :
        Minimum and maximum R.A. and dec. coordinates for a 
        `pixell.enmap.ndmap` map.
    """
    height = width if (height is None) else height
    ra_min = ra_ctr - width / 2
    ra_max = ra_ctr + width / 2
    dec_min = dec_ctr - height / 2
    dec_max = dec_ctr + height / 2
    return ra_min, ra_max, dec_min, dec_max


def get_patch_ctr_extent(ra_min, ra_max, dec_min, dec_max):
    """Return the R.A. and dec. coordinates for the center of a patch of
    sky, and its width and height.

    Parameters
    ----------
    ra_min, ra_max, dec_min, dec_max : float
        The minimum and maximum R.A. and dec., respectively, of the patch
        of sky.

    Returns
    -------
    ra_ctr, dec_ctr : float
        The R.A. and dec. coordinates of the center of the patch, in the
        same units as the input coordinates.
    width, height : float
        The width and height of the patch, in the same units as the input
        coordinates.

    See Also
    --------
    get_map_ctr_extent :
        Returns the center R.A. and dec. coordinates and the width and
        height of a `pixell.enmap.ndmap` map.
    """
    ra_ctr = (ra_min + ra_max) / 2
    dec_ctr = (dec_min + dec_max) / 2
    width = abs(ra_max - ra_min)
    height = abs(dec_max - dec_min)
    return ra_ctr, dec_ctr, width, height


def get_map_corner_coords(shape, wcs):
    """Return the minimum and maximum R.A. and dec. coordinates for a
    `pixell.enmap.ndmap` map with the given `shape` and `wcs`.

    Parameters
    ----------
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    ra_min, ra_max, dec_min, dec_max : float
        The minimum and maximum R.A. and dec., respectively, in degrees.

    See Also
    --------
    pixell.enmap.corners
    get_patch_corner_coords :
        Return the minimum and maximum R.A. and dec. coordinates for a
        patch of sky defined by the R.A. and dec. of its center and its
        width and height.
    """
    corners = np.rad2deg(enmap.corners(shape, wcs))
    dec_min = corners[0][0]
    ra_max = corners[0][1]
    dec_max = corners[1][0]
    ra_min = corners[1][1]
    return ra_min, ra_max, dec_min, dec_max


def get_map_ctr_extent(shape, wcs):
    """Return the R.A. and dec. coordinates for the center of a
    `pixell.enmap.ndmap` map with the given `shape` and `wcs`, and its
    width and height.

    Parameters
    ----------
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    ra_ctr, dec_ctr : float
        The R.A. and dec. coordinates of the map center, in degrees.
    width, height : float
        The width and height of the map, in degrees.

    See Also
    --------
    get_patch_ctr_extent :
        Return the center R.A. and dec. coordinates and the width and
        height of a patch of sky defined by its minimum and maximum
        R.A. and dec. coordinates.
    """
    ra_min, ra_max, dec_min, dec_max = get_map_corner_coords(shape, wcs)
    ra_ctr, dec_ctr, width, height = get_patch_ctr_extent(ra_min, ra_max, dec_min, dec_max)
    return ra_ctr, dec_ctr, width, height


def get_map_resolution(shape, wcs):
    """Return the resolution, in arcminutes, of the pixels in a
    `pixell.enmap.ndmap` map with the given `shape` and `wcs`.

    Parameters
    ----------
    shape : tuple of int
        The `shape` attribute of the `pixell.enmap.ndmap`.
    wcs : astropy.wcs.wcs.WCS
        The `wcs` attribute of the `pixell.enmap.ndmap`.

    Returns
    -------
    res :  float
        The resolution of the map, in arcminutes.
    """
    x_pixels = list(range(shape[-1]))
    y_pixels = [round(shape[-2]/2)] * len(x_pixels)
    ras, _ = get_coord_positions(x_pixels, y_pixels, shape, wcs)
    res = utils.deg2arcmin(np.mean(np.abs(np.diff(ras))))
    return res


def get_cutout(imap, ra, dec, width, height=None):
    """Return a smaller patch cut out from the input map, centered at the
    given R.A. and dec. with the given width and height.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        The input map.
    ra, dec : float
        The R.A. and dec. of the center of the smaller patch, in degrees.
    width : float
        The width of the smaller patch, in degrees.
    height : float, optional
        The height of the smaller patch, in degrees. By default, the
        `height` is assumed to be equal to the `width`.

    Returns
    -------
    cutout : pixell.enmap.ndmap
        A cutout from the input map for the given patch of sky.
    """
    res = get_map_resolution(imap.shape, imap.wcs)
    cutout_shape, cutout_wcs = get_shape_wcs(res, ra, dec, width, height=height)
    cutout = enmap.project(imap.copy(), cutout_shape, cutout_wcs)
    return cutout


def map_shape_is_equal(shape1, shape2):
    """Compare the shape of two `pixell.enmap.ndmap` maps.

    Parameters
    ----------
    shape1, shape2 : tuple of int
        The shape each map. The last two elements should be
        `(Ny, Nx)` for the number of pixels along the dec. and R.A.
        directions, respectively.

    Returns
    -------
    bool
        Whether the two maps have the same number of pixels along the dec.
        and R.A. directions.

    Notes
    -----
    Only checks if the number of pixels along the dec. and R.A. directions
    are equal; doesn't compare the number of components (e.g. T, Q, U).
    """
    return (shape1[-1] == shape2[-1]) and (shape1[-2] == shape2[-2])


def map_wcs_is_equal(wcs1, wcs2):
    """Compare the World Coordinate System (`wcs`) attributes of two
    `pixell.enmap.ndmap` maps.

    Parameters
    ----------
    wcs1, wcs2 : astropy.wcs.wcs.WCS
        The `wcs` attribute of each map.

    Returns
    -------
    bool
        Whether the two maps have the same `wcs`.
    """
    return wcsutils.equal(wcs1, wcs2)


def map_geometry_is_equal(shape1, wcs1, shape2, wcs2):
    """Compare the `shape` and `wcs` attributes of two 
    `pixell.enmap.ndmap` maps.

    Parameters
    ----------
    shape1, shape2 : tuple of int
        The shape each map. The last two elements should be
        `(Ny, Nx)` for the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs1, wcs2 : astropy.wcs.wcs.WCS
        The `wcs` attribute of each map.

    Returns
    -------
    equal_geometry : bool
        Whether the two maps have the same `shape` and `wcs` attributes.
    """
    equal_shape = map_shape_is_equal(shape1, shape2)
    equal_wcs = map_wcs_is_equal(wcs1, wcs2)
    equal_geometry = equal_shape and equal_wcs
    return equal_geometry


def enmap2pspy(imap):
    """Return a `pspy.so_map.so_map` instance of the input 
    `pixell.enmap.ndmap` instance.
    """
    if len(imap.shape) > 2:
        ncomp = imap.shape[-3]
    else:
        ncomp = 1
    omap = so_map.car_template_from_shape_wcs(ncomp, imap.shape[-2:], imap.wcs)
    omap.data = imap.copy()
    return omap


def deconvolve_healpix_pixel_window(imap, nside):
    """Deconvolve the pixel window function from a HEALPix map.

    Parameters
    ----------
    imap : array_like of float
        The input HEALPix map.
    nside : int
        The HEALPix `nside` parameter for the input map.

    Returns
    -------
    array_like of float
        The HEALPix map that has been pixel-window-deconvolved.
    """
    pixwin = hp.pixwin(nside)
    fullsky_map = so_map.healpix_template(1, nside)
    fullsky_map.data[:] = imap.astype(np.float64)
    fullsky_map = fullsky_map.convolve_with_pixwin(pixwin=pixwin**-1)
    return fullsky_map.data


def cutout_car_patch_from_fullsky_healpix(fullsky_map, shape, wcs):
    """Reproject from a full-sky HEALPix map to a CAR map with the given
    geometry on a patch of the sky.

    Parameters
    ----------
    fullsky_map : array_like of float
        The full-sky HEALPix map.
    shape : tuple of int
        The shape `(Ny, Nx)` of the CAR map, where `Ny` and `Nx` are the
        number of pixels along the dec. and R.A. directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the given patch of sky at the given resolution.

    Returns
    -------
    pixell.enmap.ndmap
        The CAR map on the given patch of sky.

    See Also
    --------
    pixell.reproject.healpix2map
    """
    return reproject.healpix2map(fullsky_map, shape, wcs, spin=[0])


def make_apod_window(shape, wcs, apod_width_deg, map_type='pixell'):
    """Return an apodization window as a map with values between 0 and 1.
    
    The window will be filled with ones in the inner region, and gradually
    approach zero at the edges of the map, over a distance of 
    `apod_width_deg`.
    
    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec and RA
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the given patch of sky at the given resolution.
    apod_width_deg : float
        The width of the region (in degrees) along each edge of the map,
        over which the window should transition from 1 to 0.
    map_type : str, default='pixell'
        The options are `'pixell'` or `'enmap'` to return an instance of a 
        `pixell.enmap.ndmap` map; or `'so_map'`, `'pspy'`, or `'pspipe'`
        to return an instance of a `pspy.so_map.so_map`.
        
    Returns
    -------
    window : `pixell.enmap.ndmap` or `pspy.so_map.so_map`
        The apodization window. The type depends on `map_type`.
        
    Raises
    ------
    ValueError
        If the `map_type` is not recognized.
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


def convolve_sim_with_beam(imap, beam_fwhm, window=None):
    """Convolve a map with a Gaussian beam in Fourier space.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        The input map.
    beam_fwhm : float
        The beam full-width at half-maximum, in arcminutes.
    window : pixell.enmap.ndmap, optional
        The apodization window to apply to the map before beam convolution.
        This should be provided, unless the map has already been apodized.

    Returns
    -------
    pixell.enmap.ndmap
        A copy of the input map that has been convolved with the beam.
    """
    beam_sigma = utils.arcmin2rad(utils.fwhm2sigma(beam_fwhm))
    if window is not None:
        imap = imap.copy() * window
    return enmap.smooth_gauss(imap.copy(), beam_sigma)


def make_noise_map(shape, wcs, noise_level, seed=None):
    """Return a realization of white noise for a given map geometry
    (patch size and pixel size).

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the given patch of sky at the given resolution.
    noise_level : float
        The white noise level, in units of uK-arcmin.
    seed : int, optional
        The random seed to use when generating the noise. By default, no
        seed is set.

    Returns
    -------
    noise_map : pixell.enmap.ndmap
        The map of white noise.
    """
    np.random.seed(seed)
    noise_map = enmap.zeros(shape, wcs)
    pix_area = noise_map.pixsizemap() * (60 * 180 / np.pi)**2 # radians^2 -> arcmin^2
    noise_map[:] = np.random.randn(shape[-2], shape[-1]) * noise_level / np.sqrt(pix_area)
    return noise_map


def make_noise_maps(shape, wcs, noise_level, seeds=None):
    """Return a realization of temperature and polarization white noise
    maps for a given map geometry (patch size and pixel size).

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data for a
        single component (T, Q, or U), where `Ny` and `Nx` are the
        number of pixels along the dec. and R.A. directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the given patch of sky at the given resolution.
    noise_level : float
        The white noise level for temperature, in units of uK-arcmin.
        The noise level for polarization will be `sqrt(2) * noise_level`.
    seeds : list of int, optional
        A list of random seeds to use when generating the noise. By
        default, no seeds will be set. If the `seeds` are provided,
        each of the three temperature and polarization maps should
        typically have different seeds.

    Returns
    -------
    noise_map : pixell.enmap.ndmap
        The map of white noise for temperature and polarization, with
        a `shape` of `(3, Ny, Nx)` for the three T, Q, U components.
    """
    map_shape = (3, *shape[-2:]) # for T, Q, U
    noise_map = enmap.zeros(map_shape, wcs)
    seeds = [None, None, None] if (seeds is None) else seeds
    for i, seed in enumerate(seeds):
        map_noise_level = np.sqrt(2) * noise_level if (i > 0) else noise_level
        noise_map[i] = make_noise_map(shape[-2:], wcs, map_noise_level, seed=seed)
    return noise_map


def get_pixel_num(x, y, shape):
    """Assign a unique integer to the given pixel in a map of the given
    shape.

    Parameters
    ----------
    x, y : int or array_like of int
        The x (column index) and y (row index) pixel values.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.

    Returns
    -------
    pix_num : int or array_like of int
        The unique integer value(s) assigned to the given pixels.

    See Also
    --------
    get_pixel_from_num : 
        Calculate the location of pixel(s) in a map from the pixel number.
    """
    # y = row num, x = col num
    nx = shape[-1]
    pix_num = y * nx + x
    return pix_num


def get_pixel_from_num(pix_num, shape):
    """Calculate the position of pixel(s) in a map of the given shape from
    its unique integer pixel number(s).

    Parameters
    ----------
    pix_num : int or array_like of int
        The unique integer value(s) assigned to pixel(s) in the map.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.

    Returns
    -------
    x, y : int or array_like of int
        The x (column index) and y (row index) pixel value(s).

    See Also
    --------
    get_pixel_num : Assign a unique integer to each pixel in the map.
    """
    nx = shape[-1]
    x = pix_num % nx # col
    y = (pix_num - x) // nx # row
    return x, y


def add_pixel_coords_to_catalog(catalog, shape, wcs, add_pixel_num=False, 
                                ra_key='RADeg', dec_key='decDeg', 
                                x_key='x_pixel', y_key='y_pixel', 
                                pixel_num_key='pixel_num'):
    """Add columns for the x and y pixel values for a given map,
    corresponding to the R.A. and dec. columns in the given catalog.

    Parameters
    ----------
    catalog : pandas.DataFrame
        A data frame with columns for the R.A. and dec. coordinates, in 
        degrees.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.

    Returns
    -------
    ocat : pandas.DataFrame
        The catalog with additional columns holding the pixel information.

    Other Parameters
    ----------------
    add_pixel_num : bool, default=False
        If `True`, an additional column is added that assigns a single,
        unique integer to each pixel.
    ra_key, dec_key : str, optional
        The names of the columns in the catalog for the R.A. and dec.,
        respectively.
    x_key, y_key : str, optional
        The names of the columns that will be added for the x and y
        pixels, respectively.
    pixel_num_key : str, optional
        The name of the column holding the "pixel number" of each pixel.
        Only used if `add_pixel_num = True`.
    """
    ocat = catalog.copy()
    ras = ocat[ra_key].values
    decs = ocat[dec_key].values
    ocat[x_key], ocat[y_key] = get_pixel_positions(ras, decs, shape, wcs)
    if add_pixel_num:
        ocat[pixel_num_key] = get_pixel_num(ocat[x_key].values.astype(int), ocat[y_key].values.astype(int), shape)
    return ocat


def make_catalog_for_map_geometry(icat, shape, wcs, ra_key='RADeg', dec_key='decDeg', keep_pixel_cols=False):
    """Sum the fluxes of all sources in the catalog located in the same
    pixel of the map to produce a catalog with (at most) one source per
    pixel.
    
    Parameters
    ----------
    icat : pandas.DataFrame
        A catalog of sources with columns for their positions (R.A. and
        dec., in degrees), and column(s) for their flux(es). The flux
        column name(s) must contain `'flux'`.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
        
    Returns
    -------
    ocat : pandas.DataFrame
        A catalog that has, at most, one source per map pixel. The 
        location of each source is the R.A. and dec. of its pixel center,
        and its flux is the sum of the fluxes from all sources in the
        input catalog that fall in to that pixel.
        
        
    Other Parameters
    ----------------
    ra_key, dec_key : str, optional
        The name of the column in the catalogs containing the R.A. and dec.
        coordinates (in degrees) for each source. Defaults are `'RADeg'`
        and `'decDeg'`, respectively.
    keep_pixel_cols : bool, default=False
        Include columns for the x and y pixel positions (i.e., column and 
        row indices), named `'x_pixel'` and `'y_pixel'`, respectively,
        corresponding to each R.A. and dec.
        
    See Also
    --------
    make_catalog_for_map_pixels
    """
    catalog = add_pixel_coords_to_catalog(icat.copy(), shape, wcs, add_pixel_num=True, ra_key=ra_key, dec_key=dec_key)
    # make sure all sources in catalog fall into a pixel in the map:
    catalog = catalog[catalog['x_pixel'].between(0, shape[1]-1) & catalog['y_pixel'].between(0, shape[0]-1)]
    # only keep columns for the fluxes and the pixel positions of each source:
    flux_cols = [col for col in catalog.columns.values if ('flux' in col.lower())]
    catalog_cols = [*flux_cols, 'pixel_num']
    # separate sources that occupy their own pixel vs. sources that share a pixel with others:
    srcs_in_own_pixel = catalog[~catalog.duplicated(subset='pixel_num', keep=False)][catalog_cols].copy()
    srcs_in_shared_pixels = catalog[catalog.duplicated(subset='pixel_num', keep=False)][catalog_cols].copy()
    # sum the fluxes of the sources in shared pixels:
    combined_srcs_in_shared_pixels = srcs_in_shared_pixels.groupby('pixel_num', as_index=False).sum()
    # put the catalog back together, and add the RA, dec of each pixel center:
    ocat = fgcatalogs.combine_catalogs([srcs_in_own_pixel, combined_srcs_in_shared_pixels])
    ocat = ocat.sort_values('pixel_num').reset_index(drop=True)
    ocat['x_pixel'], ocat['y_pixel'] = get_pixel_from_num(ocat['pixel_num'].values, shape)
    ocat[ra_key], ocat[dec_key] = get_coord_positions(ocat['x_pixel'].values, ocat['y_pixel'].values, shape, wcs)
    ocat_cols = [ra_key, dec_key, *flux_cols]
    if keep_pixel_cols:
        ocat_cols = [*ocat_cols, 'x_pixel', 'y_pixel', 'pixel_num']
    ocat = ocat[ocat_cols].copy()
    return ocat


def make_catalog_for_map_pixels(icat, pixel_res, ra_ctr, dec_ctr, width, height, ra_key='RADeg', dec_key='decDeg', keep_pixel_cols=False):
    """Sum the fluxes of all sources in the catalog located in the same
    pixel of the map to produce a catalog with (at most) one source per
    pixel.
    
    Parameters
    ----------
    icat : pandas.DataFrame
        A catalog of sources with columns for their positions (R.A. and
        dec., in degrees), and column(s) for their flux(es). The flux
        column name(s) must contain `'flux'`.
    pixel_res : float
        The resolution of the map pixels, in arcminutes.
    ra_ctr, dec_ctr : float
        The R.A. and dec. coordinates of the map center, in degrees.
    width, height : float
        The width and height of the map, in degrees.
        
    Returns
    -------
    ocat : pandas.DataFrame
        A catalog that has, at most, one source per map pixel. The 
        location of each source is the R.A. and dec. of its pixel center,
        and its flux is the sum of the fluxes from all sources in the
        input catalog that fall in to that pixel.
        
        
    Other Parameters
    ----------------
    ra_key, dec_key : str, optional
        The name of the column in the catalogs containing the R.A. and dec.
        coordinates (in degrees) for each source. Defaults are `'RADeg'`
        and `'decDeg'`, respectively.
    keep_pixel_cols : bool, default=False
        Include columns for the x and y pixel positions (i.e., column and 
        row indices), named `'x_pixel'` and `'y_pixel'`, respectively,
        corresponding to each R.A. and dec.
        
    See Also
    --------
    make_catalog_for_map_geometry
    """
    shape, wcs = get_shape_wcs(pixel_res, ra_ctr, dec_ctr, width, height=height)
    ocat = make_catalog_for_map_geometry(icat, shape, wcs, ra_key=ra_key, dec_key=dec_key, keep_pixel_cols=keep_pixel_cols)
    return ocat


def make_src_map(shape, wcs, catalogs, freq, ra_key='RADeg', dec_key='decDeg', flux_key_root='flux', flux_unit='mJy'):
    """Return a map of point sources at a single frequency, in CMB 
    temperature units (uK).
    
    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec and RA
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    catalogs : list of pandas.DataFrame
        A list of catalogs of the point sources to place on the map. Each 
        catalog should have at least three columns for the R.A. and dec. 
        (in degrees) and the flux of each source.
    freq : int or float
        The frequency (in units of GHz) corresponding to the source fluxes
        in the catalogs.
    
    Returns
    -------
    src_map : pixell.enmap.ndmap
        A map of the point sources in the catalog, in units of uK.

    Other Parameters
    ----------------
    ra_key, dec_key : str, optional
        The name of the column in the catalogs containing the R.A. and dec. 
        coordinates (in degrees) for each source. Defaults are `'RADeg'`
        and `'decDeg'`, respectively.
    flux_key_root, flux_unit : str, optional
        The name of the column in the catalog holding the flux must begin
        with `flux_key_root` (default is `'flux'`), and end with the units
        of flux given by `flux_unit` (default is `'mJy'`). The full name
        of the column is `'{flux_key_root}{flux_unit}'`, e.g. `'fluxmJy'`
        by default. The `'flux_unit'` is also used to convert the map to
        units of uK; it must be either `'Jy'` or `'mJy'`.
    
    Raises
    -----
    ValueError
        If the `flux_unit` is not recognized.

    See Also
    --------
    make_src_maps : 
        Generate maps of point sources from a multi-frequency catalog.
    """
    if flux_unit.lower() not in ['jy', 'mjy']:
        errmsg = f"Invalid `flux_unit`: '{flux_unit}'. The `flux_unit` must be 'Jy' or 'mJy'."
        raise ValueError(errmsg)
    flux_unit_name = 'Jy' if (flux_unit.lower() == 'jy') else 'mJy'
    flux_unit_factor = 1 if (flux_unit_name == 'Jy') else 1e-3
    flux_key = f'{flux_key_root}{flux_unit_name}'
    # place sources on map in units of Jy:
    src_map = enmap.zeros(shape, wcs)
    for cat in catalogs:
        if len(cat) > 0:
            catalog = make_catalog_for_map_geometry(cat.copy(), shape, wcs, ra_key=ra_key, 
                                                    dec_key=dec_key, keep_pixel_cols=True)
            x_pixels = catalog['x_pixel'].values.astype(int)
            y_pixels = catalog['y_pixel'].values.astype(int)
            fluxes = catalog[flux_key].values * flux_unit_factor
            src_map[y_pixels, x_pixels] = fluxes
    src_map /= enmap.pixsizemap(shape, wcs) # convert to Jy / str
    src_map = utils.Jy_per_str_to_uK(src_map, freq) # convert to uK
    return src_map


def make_src_maps(shape, wcs, multifreq_catalog, freqs, 
                  ra_key='RADeg', dec_key='decDeg', 
                  flux_key_root='flux', flux_unit='mJy'):
    """Return maps of point sources at a multiple frequencies, in CMB 
    temperature units (uK).
    
    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec. and R.A.
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    multifreq_catalog : pandas.DataFrame
        A catalog of the point sources to place on the map. The catalog 
        should have columns for the RA and dec (in degrees) of each source,
        and a column for the fluxes at each frequency.
    freqs : list of float
        The frequencies (in units of GHz) corresponding to the source fluxes
        in the catalogs.  
    
    Returns
    -------
    src_maps : dict of pixell.enmap.ndmap
        A dictionary with keys given by the frequencies in `freqs` 
        containing the maps of point sources at each frequency, un units 
        of uK.

    Other Parameters
    ----------------
    ra_key, dec_key : str, optional
        The name of the column in the catalog containing the R.A. and dec. 
        coordinates (in degrees) for each source. Defaults are `'RADeg'`
        and `'decDeg'`, respectively.
    flux_key_root, flux_unit : str, optional
        The name of the columns in the catalog holding the fluxes must 
        begin with `flux_key_root` (default is `'flux'`), and end with
        the units of flux given by `flux_unit` (default is `'mJy'`). At a
        given frequency `freq`, the full name of the column is given by
        `'{flux_key_root}{flux_unit}_{freq}GHz'`, e.g. `'fluxmJy_90GHz'`
        by default at 90 GHz. The `'flux_unit'` is also used to convert 
        the maps to units of uK; it must be either `'Jy'` or `'mJy'`.
    
    Raises
    -----
    ValueError
        If the `flux_unit` is not recognized.

    See Also
    --------
    make_src_map : 
        Generate a single map of point sources from a catalog.
    """
    if flux_unit.lower() not in ['jy', 'mjy']:
        errmsg = f"Invalid `flux_unit`: '{flux_unit}'. The `flux_unit` must be 'Jy' or 'mJy'."
        raise ValueError(errmsg)
    flux_unit_name = 'Jy' if (flux_unit.lower() == 'jy') else 'mJy'
    flux_unit_factor = 1 if (flux_unit_name == 'Jy') else 1e-3
    flux_keys = {freq: f'{flux_key_root}{flux_unit_name}_{freq}GHz' for freq in freqs}
    # make empty maps:
    src_maps = {freq: enmap.zeros(shape, wcs) for freq in freqs}
    if len(multifreq_catalog) > 0:
        # get a catalog with one source per map pixel:
        catalog = make_catalog_for_map_geometry(multifreq_catalog.copy(), shape, wcs,
                                                ra_key=ra_key, dec_key=dec_key, keep_pixel_cols=True)
        x_pixels = catalog['x_pixel'].values.astype(int)
        y_pixels = catalog['y_pixel'].values.astype(int)
        # put the sources on the map at each frequency:
        for freq in freqs:
            flux_key = flux_keys[freq]
            fluxes = catalog[flux_key].values * flux_unit_factor
            src_maps[freq][y_pixels, x_pixels] = fluxes # units of Jy
            src_maps[freq] /= enmap.pixsizemap(shape, wcs) # convert to Jy / str
            src_maps[freq] = utils.Jy_per_str_to_uK(src_maps[freq], freq) # convert to uK
    return src_maps


def _format_cmb_theo_cls_from_file(theory_file, pol=True):
    """Load theory CMB temperature and polarization power spectra and
    place the power spectra at each multipole into a 3x3 (T, E, B) array,
    or just return the temperature power spectrum if `pol=False`.
    """
    theo_cls = powspec.read_spectrum(theory_file, inds=True, scale=False, expand=None, ncol=4)
    if not pol: # just load TT into 1d array
        ps = theo_cls[0]
    else: # put theory at each ell into a 3x3 array  (for T, E, B)
        nl = theo_cls.shape[-1]
        ps = np.zeros((3, 3, nl))
        ps[0, 0] = theo_cls[0] # TT
        ps[0, 1] = theo_cls[3] # TE
        ps[1, 0] = theo_cls[3] # ET = TE
        ps[1, 1] = theo_cls[1] # EE
        ps[2, 2] = theo_cls[2] # BB
    return ps


def _format_cmb_theo_cls_from_dict(theory_dict, pol=True):
    """Place the theory CMB temperature and polarization power spectra at
    each multipole into a 3x3 (T, E, B) array, or just return the
    temperature power spectrum if `pol=False`.
    """
    if not pol: # just load TT into 1d array
        ps = theory_dict['tt'].copy()
    else: # put theory at each ell into a 3x3 array  (for T, E, B)
        nl = theory_dict['tt'].shape[-1]
        ps = np.zeros((3, 3, nl))
        ps[0, 0] = theory_dict['tt']
        ps[0, 1] = theory_dict['te']
        ps[1, 0] = theory_dict['te'] # ET = TE
        ps[1, 1] = theory_dict['ee']
        ps[2, 2] = theory_dict['bb']
    return ps


def make_cmb_sim_from_theory(shape, wcs, theory_file=None, theory_dict=None, lmax=None, pol=True, seed=None):
    """Generate a realization of the CMB on a patch of the sky from
    theory CMB power spectra.

    The theory TT, EE, BB, and TE power spectra must be provided by
    passing either a `theory_file` or a `theory_dict`. The spectra should
    not be multiplied by any multipole factors.

    Parameters
    ----------
    shape : tuple of int
        The shape `(Ny, Nx)` of the map(s), where `Ny` and `Nx` are the
        number of pixels along the dec. and R.A. directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the given patch of sky at the given resolution.
    theory_file : str or None, default=None
        The path to the theory file. The first column should hold the
        multipoles, starting at zero, and the remaining columns should
        hold the power spectra in the order TT, EE, BB, TE. If the
        `theory_file` is not passed, a `theory_dict` must be passed.
    theory_dict : dict of array_like of float
        A dictionary with keys `'ells'`, `'tt'`, `'ee'`, `'bb'`, `'te'`
        for the arrays of the multipoles and TT, EE, BB, TE power spectra,
        respectively. If the `theory_dict` is not passed, a `theory_file`
        must be passed. If both are passed, the `theory_file` is used.
    lmax : int or None, default=None
        The maximum multipole to use when generating the CMB realization.
        By default, the maximum multipole of the theory spectra is used.
    pol : bool, default=True
        If `pol=True`, CMB temperature and polarization (T, Q, U) maps
        will be generated. Otherwise, only the temperature map is
        generated.
    seed : int or None, default=None
        The random seed used when generating the CMB realization.

    Returns
    -------
    sim : pixell.enmap.ndmap
        The CMB map(s). If `pol=True`, the shape will be `(3, Ny, Nx)`
        for the T, Q, and U maps, where `(Ny, Nx)` is given by the `shape`
        parameter. Otherwise, only the temperature map with shape
        `(Ny, Nx)` is returned.

    See Also
    --------
    pixell.curvedsky.rand_map
    """
    if theory_file is not None:
        ps = _format_cmb_theo_cls_from_file(theory_file, pol=pol)
    elif theory_dict is not None:
        ps = _format_cmb_theo_cls_from_dict(theory_dict, pol=pol)
    else:
        raise ValueError(f"`{theory_file = }` and `{theory_dict = }`. You must provide the CMB theory"
                         " by either passing the path to the saved theory file as `theory_file`,"
                         " or by passing a dictionary holding the theory (with keys for `'tt'`,"
                         " and `'te'`, `'ee'`, `'bb'` if `pol=True`) as `theory_dict`.")
    map_shape = shape[-2:]
    shape = (3, *map_shape) if pol else map_shape
    spin = [0, 2] if pol else [0]
    lmax = ps.shape[-1] - 1 if (lmax is None) else lmax
    sim = cs.rand_map(shape, wcs, ps, lmax=lmax, seed=seed, spin=spin)
    return sim


def replace_alms_within_ell_range(alms, alms_to_replace_with, lmin=None, lmax=None):
    """Replace a set of spherical harmonic coefficients ('alms') with a
    different set within some multipole range.

    Parameters
    ----------
    alms : array_like of float
        The set of alms that will be replaced within the multipole range.
    alms_to_replace_with : array_like of float
        The different set of alms used to replace the values in `alms`
        within the multipole range. Must be the same shape as `alms`.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipoles (`ell`s) that define the range.
        - If both `lmin` and `lmax` are passed, the values in `alms`
          corresponding to `lmin <= ell <= lmax` are replaced.
        - If only `lmin` is passed, the values in `alms` corresponding to
          `ell >= lmin` are replaced.
        - If only `lmax` is passed, the values in `alms` corresponding to
          `ell <= lmax` are replaced.
        - If both `lmin` and `lmax` are `None`, the `alms` are returned
          unchanged.

    Returns
    -------
    new_alms : array_like of float
        The new set of alms, with the values of `alms` outside the
        multipole range and the values of `alms_to_replace_with` within
        the multipole range.

    Raises
    ------
    ValueError
        If `lmin > lmax`, or if `alms` and `alms_to_replace_with` have
        different shapes.
    """
    alms_lmax = hp.Alm.getlmax(len(alms))
    alms_to_replace_with_lmax = hp.Alm.getlmax(len(alms_to_replace_with))
    if len(alms_to_replace_with) != len(alms):
        raise ValueError(f"`{len(alms_to_replace_with) = }` (max. multipole = {alms_to_replace_with_lmax}) and"
                         f" `{len(alms) = }` (max. multipole = {alms_lmax}). `alms_to_replace_with` must have"
                         " the same shape as `alms`.")

    # get an array (same shape as `alms`) where each element is the
    # multipole of the corresponding element in `alms`; use it to find
    # the elements in the `alms` array to be replaced:
    ell_vals, _ = hp.Alm.getlm(alms_lmax)
    if (lmin is not None) and (lmax is not None):
        if lmin > lmax:
            raise ValueError(f"`{lmin = }`, `{lmax = }`. `lmin` must be less than or equal to `lmax`.")
        ell_in_range = (ell_vals >= lmin) & (ell_vals <= lmax)
    elif lmin is not None:
        ell_in_range = ell_vals >= lmin
    elif lmax is not None:
        ell_in_range = ell_vals <= lmax
    else:
        ell_in_range = np.array([False] * len(alms)) # don't replace anything

    new_alms = np.where(ell_in_range, alms_to_replace_with, alms)
    return new_alms


def replace_power_in_map_with_theory(imap, theory_cls, lmax_for_alms,
                                     lmin_to_replace=None, lmax_to_replace=None,
                                     window=None, seed_for_alms=None):
    """Replace the power in a map within some multipole range with a
    Gaussian realization of a theory power spectrum.

    A Gaussian realization of spherical harmonic coefficients ('alms') is
    generated from the theory power spectrum, and the alms of the input
    map are replaced with this realization on scales between
    `lmin_to_replace` and `lmax_to_replace`.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        The input map.
    theory_cls : array_like of float
        The theory power spectrum. The units should be the units of `imap`
        squared (e.g., if `imap` has units of uK, `theory_cls` should have
        units of uK^2). The power spectrum should not be multiplied by any
        multipole factors.
    lmax_for_alms : int
        The maximum multipole to use when taking or generating the alms.
    lmin_to_replace, lmax_to_replace : int or None, default=None
        The minimum and maximum multipoles (`ell`s), respectively, that
        define the range where the power in the map should be replaced
        with the realization of the theory power spectrum.
        - If both `lmin_to_replace` and `lmax_to_replace` are passed, the
          map alms will be replaced in the range
          `lmin_to_replace <= ell <= lmax_to_replace`.
        - If `lmin_to_replace` is `None`, the map alms are replaced for
          all `ell <= lmax_to_replace`.
        - If `lmax_to_replace` is `None`, the map alms are replaced for
          all `ell >= lmin_to_replace`.
        - If both `lmin_to_replace` and `lmax_to_replace` are `None`, the
          `imap` is returned unchanged.
    window : pixell.enmap.ndmap or None, default=None
        An apodization window to apply to the map before taking its alms.
        The map should always be apodized before taking its alms, so the
        `window` should always be passed unless the `imap` is already
        apodized.
    seed_for_alms : int or None, default=None
        The random seed to use when generating the Gaussian realization of
        the theory power spectrum.

    Returns
    -------
    omap : pixell.enmap.ndmap
        The input map, with its power within the given range replaced with
        a Gaussian realization of the theory power spectrum.

    See Also
    --------
    make_apod_window : Apodization window.

    Notes
    -----
    Not implemented for polarization maps.
    """
    # get alms of the map:
    window = window if (window is not None) else enmap.ones(imap.shape, imap.wcs)
    imap_alms = cs.map2alm(imap * window, lmax=lmax_for_alms, spin=[0])
    # generate random realization of alms from the theory spectrum:
    theo_alms = cs.rand_alm(theory_cls, lmax=lmax_for_alms, seed=seed_for_alms)
    # make a new set of alms, with the map alms replaced
    # by theory alms in the given multipole range:
    alms = replace_alms_within_ell_range(imap_alms, theo_alms, lmin=lmin_to_replace, lmax=lmax_to_replace)
    # generate a new map from the alms:
    omap = cs.alm2map(alms, imap, spin=[0])
    return omap


def angular_distance(ra1, dec1, ra2, dec2, output_unit='arcminutes'):
    """Return the angular distance between two sets of pairs of R.A. and 
    dec. coordinates.

    Parameters
    ----------
    ra1, dec1 : float or arrary of float
        The first pair(s) of R.A. and dec. coordinates, in degrees.
    ra2, dec2 : float or array_like of float
        The second pair(s) of R.A. and dec. coordinates, in degrees.
    output_units : str, default='arcmin'
        The units of the returned angular distance. The options are
        `'radians'`, `'degrees'`, `'arcminutes'`, or `'arcseconds'`
        (common abbreviations such as `'deg'` are also recognized).

    Returns
    -------
    dist : float or array_like of float
        The angular distance(s) between the coordinate pairs. If `ra1`,
        `dec1`, `ra2`, `dec2` are all single values, a single value for
        the distance between `(ra1, dec1)` and `(ra2, dec2)` is returned.
        If `ra1` and `dec1` are single values and `ra2` and `dec2` are
        arrays (or vice versa), an array of distances between
        `(ra1, dec1)` and  each `(ra, dec)` pair in `(ra2, dec2)` is
        returned. If `ra1`, `dec1`, `ra2`, `dec2` are all arrays, an
        array of distances between each corresponding pair
        `(ra1[i], dec1[i])` and `(ra2[i], dec2[i])` is returned.

    Notes
    -----
    If `ra1`, `dec1`, `ra2`, `dec2` are all arrays, they must have the 
    same shape.
    """
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)
    cos_dist = np.cos(dec1) * np.cos(dec2) * (np.cos(ra1) * np.cos(ra2) + np.sin(ra1) * np.sin(ra2)) + np.sin(dec1) * np.sin(dec2)
    dist = np.arccos(np.round(cos_dist, 12)) # make sure values do not exceed +/- 1 (due to floating-point precision)
    # return the distance in the requested units
    if 'deg' in output_unit.lower():
        dist = np.rad2deg(dist)
    elif 'arcmin' in output_unit.lower():
        dist = utils.rad2arcmin(dist)
    elif 'arcsec' in output_unit.lower():
        dist = rad2arcsec(dist)
    elif 'rad' not in output_unit.lower():
        valid_output_units = ['radians', 'degrees', 'arcminutes', 'arcseconds']
        errmsg = (f"Invalid value of `output_units`: `'{output_units}'`. "
                 f"Valid options are: {valid_output_units}.")
        raise ValueError(errmsg)
    return dist

