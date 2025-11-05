"""Contains general functions for the extragalactic foreground catalogs."""

import os
from IPython.display import display
import numpy as np
import pandas as pd
from pixell import enmap
from . import utils, siminfo as si


def load_catalog(catalog_file):
    """Load a catalog from either a CSV or FITS file into a data frame.
    
    The file type is determined from the file extension. 
    
    If the catalog has already been saved from a `pandas.DataFrame` 
    instance, the column holding the data frame index (which is by default
    named `'Unnamed'`) will be removed, but the index will not be reset.
    
    Parameters
    ----------
    catalog_file : str
        The path to the catalog file.
        
    Returns
    -------
    catalog : pandas.DataFrame
        The catalog.
    """
    if catalog_file.endswith('fits'):
        from astropy.table import Table
        catalog = Table.read(catalog_file).to_pandas()
    else:
        catalog = drop_unnamed_columns(pd.read_csv(catalog_file))
    return catalog


def drop_unnamed_columns(df):
    """Remove any columns without a name from a data frame.

    By default, when saving a `pandas.DataFrame` instance as a CSV file,
    the data frame index is saved in the first column without a
    corresponding column name in the file header. When the file is loaded
    into a new data frame, there will be an column named `'Unnamed'`
    holding the index; this column (and any additional unnamed columns)
    will be removed.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame.

    Returns
    -------
    df : pandas.DataFrame
        The input data frame with all unnamed columns removed.
    """
    unnamed_cols = []
    for col in df.columns.values:
        if 'Unnamed' in col:
            unnamed_cols.append(col)
    if len(unnamed_cols) > 0:
        df = df.drop(columns=unnamed_cols)
    return df


def display_catalog(catalog, nrow=5):
    """Display a few rows at the beginning and end of a catalog.

    This can be used in Jupyter notebooks to quickly view a catalog
    without causing a significant increase in the notebook file size.

    Parameters
    ----------
    catalog : pandas.DataFrame
        The catalog to display.
    n : int, default=5
        The number of rows to display from the beginning and end of the
        catalog; a total of `2*n` rows will be displayed.
    """
    if len(catalog) <= 2*nrow:
        display(catalog)
    else:
        display(catalog.head(nrow))
        display(catalog.tail(nrow))
    print(f'{len(catalog)} rows, {len(catalog.columns.values)} columns')


def combine_catalogs(catalog_list):
    """Combine a list of catalogs into a single catalog.

    Assumes that each catalog in the list has the same set of columns.
    The index of each catalog data frame will not be preserved.

    Parameters
    ----------
    catalog_list : list of pandas.DataFrame
        A list of catalogs to combine. Each should have the same set of
        columns.

    Returns
    -------
    catalog : pandas.DataFrame
        A single catalog that has all rows of each catalog in the
        `catalog_list`.
    """
    catalogs = [catalog for catalog in catalog_list if (len(catalog) > 0)]
    if len(catalogs) > 0:
        catalog = pd.concat(catalogs, ignore_index=True, sort=False)
        catalog = catalog.reset_index(drop=True)
    else:
        # all are empty, so just return empty catalog w/ all columns:
        cols = list(set(np.concatenate([cat.columns.values for cat in catalog_list])))
        catalog = pd.DataFrame({col: [] for col in cols})
    return catalog


def trim_catalog_positions(input_catalog, ra_key='RADeg', dec_key='decDeg',
                           ra_ctr=None, dec_ctr=None, width=None, height=None,
                           ra_min=None, ra_max=None, dec_min=None, dec_max=None):
    """Given an input catalog with columns of R.A. and dec. coordinates,
    return a copy of the catalog that only contains rows with R.A. and dec.
    coordinates within some range of values.

    Parameters
    ----------
    input_catalog : pandas.DataFrame
        A catalog with columns for R.A. and dec. positions.
    ra_key, dec_key : str, optional
        The names of the columns holding the R.A. and dec., respectively.
        The defaults are `ra_key='RADeg'`, `dec_key='decDeg'`.
    ra_ctr, width : float, optional
        The center and width of the range in R.A. Both should be in the
        same units as the R.A. column of the catalog. If the `width` is not
        provided, then the value of `ra_ctr` will be ignored, and the
        values of `ra_min` and `ra_max` will be used to determine the
        range in R.A.
    dec_ctr, height : float, optional
        The center and height of the range in dec. Both should be in the
        same units as the dec. column of the catalog. If the `height` is
        not provided, then the value of `dec_ctr` will be ignored, and
        the values of `dec_min` and `dec_max` will be used to determine
        the range in dec.
    ra_min, ra_max : float, optional
        The minimum and maximum values of the range in R.A. Both should be
        in the same units as the R.A. column of the catalog. If `ra_min` is
        provided, the returned catalog will only contain rows with R.A.
        greater than or equal to `ra_min`. If `ra_max` is provided, the
        returned catalog will only contain rows with R.A. less than or
        equal to `ra_max`. Only used if the `width` is not provided.
    dec_min, dec_max : float, optional
        The minimum and maximum values of the range in dec. Both should be
        in the same units as the dec. column of the catalog. If `dec_min` is
        provided, the returned catalog will only contain rows with dec.
        greater than or equal to `dec_min`. If `dec_max` is provided, the
        returned catalog will only contain rows with dec. less than or
        equal to `dec_max`. Only used if the `height` is not provided.

    Returns
    -------
    catalog : pandas.DataFrame
        A copy of the `input_catalog`, only containing rows with R.A. and
        dec. coordinates within the given ranges.

    Raises
    ------
    ValueError
        If the `width` is provided but `ra_ctr=None`, or if the `height`
        is provided but `dec_ctr=None`.
    """
    catalog = input_catalog.copy()
    if width is None:
        if ra_min is not None:
            catalog = catalog[catalog[ra_key].ge(ra_min)]
        if ra_max is not None:
            catalog = catalog[catalog[ra_key].le(ra_max)]
    else:
        if ra_ctr is None:
            raise ValueError("You must pass `ra_ctr` if `width` is not `None`")
        else:
            catalog = catalog[catalog[ra_key].between(ra_ctr - width / 2, ra_ctr + width / 2)]
    if height is None:
        if dec_min is not None:
            catalog = catalog[catalog[dec_key].ge(dec_min)]
        if dec_max is not None:
            catalog = catalog[catalog[dec_key].le(dec_max)]
    else:
        if dec_ctr is None:
            raise ValueError("You must pass `dec_ctr` if `height` is not `None`")
        else:
            catalog = catalog[catalog[dec_key].between(dec_ctr - height / 2, dec_ctr + height / 2)]
    return catalog


def make_catalog_from_sims(sims):
    """Return a catalog of point sources measured from the given maps at
    different frequencies.

    The maps at each frequency must contain the same set of sources, i.e.
    the each source must be placed at the same location in each map.
    Each non-zero pixel in the maps will correspond to a single source in
    the returned catalog. The flux of the source at each frequency is
    measured by the value of its pixel in each map, and its position will
    be measured as the R.A., dec. of the center of the pixel.

    Parameters
    ----------
    sims : dict of pixell.enmap.ndmap
        A dictionary holding maps of point sources at different
        frequencies. The keys should be frequencies (`int` or `float`) in
        units of GHz, and the values should be the point-source maps at
        the corresponding frequency, in units of uK. The maps should NOT
        be convolved with any pixel window function, beam, etc.

    Returns
    -------
    catalog : pandas.DataFrame
        A catalog of sources in the maps. The catalog will have columns
        `'RADeg'` and `'decDeg'` holding the R.A. and dec. positions of 
        each source, in units of degrees. For each map frequency `freq` 
        (given by the keys of the `sims` dictionary), the catalog will 
        have a column named `'fluxmJy_{freq}GHz'` for the flux of each 
        source (in units of mJy) at that frequency.
    """
    freqs = sorted(list(sims.keys()))
    shape = sims[freqs[0]].shape
    wcs = sims[freqs[0]].wcs
    posmap = np.rad2deg(enmap.posmap(shape, wcs))
    ra_map = posmap[1]  # map with RA of center of each pixel
    dec_map = posmap[0] # map with dec of center of each pixel
    # convert maps to units of mJy / sr, then multiply by solid angle 
    # of each pixel to get units of mJy:
    flux_sims = {}
    for freq in freqs:
        flux_sims[freq] = utils.uK_to_mJy_per_str(sims[freq].copy(), freq) * sims[freq].pixsizemap()
    # make a catalog where each non-zero pixel corresponds to a single source:
    nonzero_mask = np.greater(flux_sims[freqs[0]], 0)
    catalog_dict = {'RADeg': ra_map[nonzero_mask], 'decDeg': dec_map[nonzero_mask]}
    for freq in freqs:
        catalog_dict[f'fluxmJy_{freq}GHz'] = flux_sims[freq][nonzero_mask]
    catalog = pd.DataFrame(catalog_dict)
    return catalog


def add_gauss_scatter_to_coords(ras, decs, shape, wcs, seed=si.cib_catalog_seed, 
                                sigma_pix_frac=si.cib_gauss_sigma_pix_frac):
    """Add some small Gaussian scatter to the R.A. and dec. coordinates, 
    such that the new coordinates will remain within the same or 
    neighboring pixel in a map on a given patch of sky.

    Parameters
    ----------
    ras, decs : array_like of float
        The R.A. and dec. coordinates, in degrees.
    shape : tuple of int
        The shape `(Ny, Nx)` of the array holding the map data, where
        `Ny` and `Nx` are the number of pixels along the dec and RA
        directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        An astropy World Coordinate System instance for the pixelization
        of the map.
    seed : int, default=0
        The random seed to use when generating the scatter.
    sigma_pix_frac : float, default=0.2
        The standard deviation to use when adding Gaussian scatter to the
        positions in the output catalog, defined as a fraction of the map
        pixel size. The value should fall between 0 and 1.

    Returns
    -------
    random_ras, random_decs : array_like of float
        The R.A. and dec. coordinates, in degrees, with the scatter added.
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

