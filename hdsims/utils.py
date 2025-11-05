"""Contains general utility functions."""

import os
import logging
import numpy as np
from scipy import sparse as sp


# ========== unit conversions ==========

def deg2arcmin(x):
    """Convert angles from units of degrees to arcminutes"""
    return x * 60


def arcmin2deg(x):
    """Convert angles from units of arcminutes to degrees"""
    return x / 60


def deg2arcsec(x):
    """Convert angles from units of degrees to arcseconds"""
    return x * 3600


def arcsec2deg(x):
    """Convert angles from units of arcseconds to degrees"""
    return x / 3600


def rad2arcmin(x):
    """Convert angles from units of radians to arcminutes"""
    return deg2arcmin(np.rad2deg(x))


def arcmin2rad(x):
    """Convert angles from units of arcminutes to radians"""
    return np.deg2rad(arcmin2deg(x))


def s10_temp_to_intensity_unit_conversion(freq, TCMB=2.7255e6):
    """Return the frequency-dependent factor, in units of (Jy/sr)/uK, to
    convert between temperature (units of uK) and intensity
    (Jy / steradian).

    The value is given by the derivative of the Planck intensity function
    with respect to temperature, evaluated at the CMB temperature today.
    These are the precomputed values for 30, 90, 148, 219, 277, and
    350 GHz that were used in the Sehgal et. al. (2010) simulations;
    see arXiv:0908.0540 and
    https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_info.html

    Parameters
    ----------
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The value of the CMB temperature to use, in uK.

    Returns
    -------
    unit_conversion : float
        The unit conversion factor, in units of (Jy/sr) / uK.
    """
    # divide map in Jy/str by this factor to get in delta T / T units:
    unit_conversions = {30: 7.364967e7, 90: 5.526540e8, 148: 1.072480e9, 
                        219: 1.318837e9, 277: 1.182877e9, 350: 8.247628e8} # Jy/sr
    if freq not in unit_conversions:
        raise NotImplementedError(f"The conversion factor for {freq = } GHz is not saved.")
    unit_conversion = unit_conversions[freq] / TCMB # (Jy/sr)/uK
    return unit_conversion


def Jy_per_str_to_uK(x, freq, TCMB=2.7255e6):
    """Convert from intensity units (Jy/sr) to temperature units (uK).
    
    Parameters
    ----------
    x : float or array_like of float
        The value(s), in units of Jy/sr.
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The value of the CMB temperature to use, in uK.
        
    Returns
    -------
    float or array_like of float
        The value(s), in units of uK.

    See Also
    --------
    uK_to_Jy_per_str
    """
    if freq in [30, 90, 148, 219, 277, 350]: 
        # use values provided on LAMBDA; see 
        # https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_info.html
        return x / s10_temp_to_intensity_unit_conversion(freq, TCMB=TCMB)
    else:
        c = 2.998e10 # cm / s
        h = 6.626e-27 # erg * s
        k_B = 1.381e-16 # erg / K
        Jy = 1e-23 # erg / cm^2
        nu = freq * 1e9 # Hz
        x_nu = h * nu / (k_B * TCMB * 1e-6) # dimensionless frequency
        # need units of uK / (Jy / sr):
        conversion_factor = 1e6 * (c**2 / (2 * k_B * nu**2)) * ((np.exp(x_nu) - 1) / x_nu)**2 / np.exp(x_nu) 
        return x * conversion_factor * Jy

    
def uK_to_Jy_per_str(x, freq, TCMB=2.7255e6):
    """Convert from temperature units (uK) to intensity units (Jy/sr).
    
    Parameters
    ----------
    x : float or array_like of float
        The value(s), in units of uK.
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The value of the CMB temperature to use, in uK.
        
    Returns
    -------
    float or array_like of float
        The value(s), in units of Jy / sr.

    See Also
    --------
    Jy_per_str_to_uK
    """
    return x / Jy_per_str_to_uK(1, freq, TCMB=TCMB)


def uK_to_mJy_per_str(x, freq, TCMB=2.7255e6):
    """Convert from temperature units (uK) to intensity units (mJy/sr).
    
    Parameters
    ----------
    x : float or array_like of float
        The value(s), in units of uK.
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The value of the CMB temperature to use, in uK.
        
    Returns
    -------
    float or array_like of float
        The value(s), in units of mJy / sr.

    See Also
    --------
    mJy_per_str_to_uK
    """
    flux_Jy_per_str = uK_to_Jy_per_str(x, freq, TCMB=TCMB)
    return flux_Jy_per_str * 1e3


def mJy_per_str_to_uK(x, freq, TCMB=2.7255e6):
    """Convert from intensity units (mJy/sr) to temperature units (uK).
    
    Parameters
    ----------
    x : float or array_like of float
        The value(s), in units of mJy/sr.
    freq : int or float
        The frequency, in GHz.
    TCMB : float, default=2.7255e6
        The value of the CMB temperature to use, in uK.
        
    Returns
    -------
    float or array_like of float
        The value(s), in units of uK.

    See Also
    --------
    uK_to_mJy_per_str
    """
    flux_Jy_per_str = x * 1e-3
    return Jy_per_str_to_uK(flux_Jy_per_str, freq, TCMB=TCMB)


# ========== loading and creating/saving files ==========

def mkdir(path):
    """Create a directory with the given `path` and any of its parent 
    directories.
    
    Parameters
    ----------
    path : str
        The absolute path of the directory.

    Returns
    -------
    path : str
        The absolute path of the directory (which now exists).

    See Also
    --------
    os.makedirs
    """
    os.makedirs(path, exist_ok=True)
    return path


def load_dict_from_file(fname, columns, skip_cols=[]):
    """Return dictionary from columns loaded from a `.txt` file.

    Parameters
    ----------
    fname : str
        The path to the file.
    columns : list of str
        The names, in the correct order, of each column; these will also 
        serve as the dictionary keys.
    skip_cols : list of str, default=[]
        The names of any columns that should not be included in the output
        dictionary.

    Returns
    -------
    data : dict of array_like
        A dictionary whose keys are the names in `columns` and values are
        one-dimensional arrays holding the data from the corresponding
        column in the file.

    See Also
    --------
    save_dict_to_file
    """
    data = {}
    data_array = np.loadtxt(fname)
    for i, col in enumerate(columns):
        if col not in skip_cols:
            data[col] = data_array[:,i].copy()
    return data


def save_dict_to_file(fname, data, keys=None, col_names=None, extra_header_info=None):
    """Save 1D arrays in a dictionary as columns in a `.txt` file.

    Parameters
    ----------
    fname : str
        The path to the file.
    data : dict of array_like
        The dictionary of arrays to save.
    keys : list or None, default=None
        A list of keys to use. Otherwise, all keys are used
        (and order of columns in the file is not pre-determined).
    col_names : list or None, default=None
        A list of names to use in the header for each column, in the same 
        order as `keys` (which must also be passed). Otherwise the `keys` 
        are used.
    extra_header_info : str or None, default=None
        If not `None`, add the `extra_header_info` to the end of the
        header, following the column names.

    See Also
    --------
    load_dict_from_file
    """
    if keys is None:
        col_names = None
        keys = list(data.keys())
    if col_names is None:
        col_names = keys.copy()
    header = ', '.join(col_names)
    if extra_header_info is not None:
        header = f'columns are: {header} ; {extra_header_info}'
    data_cols = []
    for key in keys:
        data_cols.append(data[key])
    np.savetxt(fname, np.column_stack(data_cols), header=header)


def save_yaml(fname, save_dict, overwrite=False):
    """Save a dictionary to a YAML file.

    Parameters
    ----------
    fname : str
        The file name, including the file extension.
    save_dict : dict
        The dictionary to save.
    overwrite : bool, default=False
        Whether to overwrite the file, if it already exists.

    See Also
    --------
    load_yaml
    """
    import yaml
    if overwrite or (not os.path.exists(fname)):
        with open(fname, 'w') as f:
            yaml.dump(save_dict, f)


def load_yaml(fname):
    """Return a dictionary loaded from a YAML file.

    Parameters
    ----------
    fname : str
        The YAML file name.

    Returns
    -------
    data : dict
        The dictionary loaded from the YAML file.

    See Also
    --------
    save_yaml
    """
    import yaml
    with open(fname, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_sparse_matrix_npz(filename, mat, tol=1e-12):
    """Save a dense matrix as a sparse CSR `.npz` file, after setting small
    values to zero.

    Parameters
    ----------
    filename : str
        Path for the saved sparse matrix (a `.npz` file will be written).
    mat : array_like
        Dense 2D array (or object convertible to `scipy.sparse.csr_matrix`)
        to save.
    tol : float, default=1e-12
        The tolerance used to determine which values are "small". Elements
        in the `mat` with an absolute value less than `tol` will be set to
        zero before saving.

    See Also
    --------
    load_sparse_matrix_npz
    scipy.sparse.csr_matrix
    """
    M = sp.csr_matrix(mat)
    M.data[np.abs(M.data) < tol] = 0.0
    M.eliminate_zeros()
    sp.save_npz(filename, M)


def load_sparse_matrix_npz(filename):
    """Load a dense matrix saved as a sparse CSR `.npz` file.

    Parameters
    ----------
    filename : str
        Path to the saved sparse matrix.

    Returns
    -------
    array_like
        The matrix that was saved.

    See Also
    --------
    save_sparse_matrix_npz
    scipy.sparse.csr_matrix
    """
    return sp.load_npz(filename).toarray()


# ========== coordinates on the sky: ==========

def dec_to_theta(dec):
    """Return the polar angle theta (radians) in spherical coordinates
    corresponding to the given declination (degrees).
    """
    return (np.pi / 2) - np.deg2rad(dec)


def ra_to_phi(ra):
    """Return the azimuthal angle phi (radians) in spherical coordinates
    corresponding to the given right ascension (degrees).
    """
    return np.deg2rad(ra)


def theta_to_dec(theta):
    """Return the declination (degrees) corresponding to the polar angle
    theta (radians) in spherical coordinates.
    """
    return  np.rad2deg((np.pi / 2) - theta)


def phi_to_ra(phi):
    """Return the right ascension (degrees) corresponding to the given
    azimuthal angle phi (radians) in spherical coordinates.
    """
    return np.rad2deg(phi)


def dec_ra_to_theta_phi(dec, ra):
    """Return the pair of spherical coordinates (theta, phi) corresponding
    to the given declination and right ascension coordinates.

    Parameters
    ----------
    dec, ra : float
        The declination and right ascension (in degrees), respectively.

    Returns
    -------
    theta, phi : float
        The polar and azimuthal angles (in radians), respectively,
        corresponding to the given declination and right ascension.
    """
    return dec_to_theta(dec), ra_to_phi(ra)


def theta_phi_to_dec_ra(theta, phi):
    """Return the pair of declination and right ascension coordinates
    corresponding to the given spherical coordinates..

    Parameters
    ----------
    theta, phi : float
        The polar and azimuthal angles (in radians), respectively.

    Returns
    -------
    dec, ra : float
        The declination and right ascension (in degrees), respectively,
        corresponding to the given spherical coordinates.
    """
    return theta_to_dec(theta), phi_to_ra(phi)


def ra_is_in_patch(ra, ra_min, ra_max):
    """Check whether the given right ascension coordinate is within a 
    patch of sky.
    
    Parameters
    ----------
    ra : float
        The right ascension.
    ra_min, ra_max : float
        The minimum and maximum right ascension coordinates of the patch 
        of sky. Must be in the same units as `ra`.
        
    Returns
    -------
    bool :
        `True` if the `ra` is in the range given by `ra_min` and `ra_max`,
        `False` otherwise.
    """
    return (ra >= ra_min) and (ra <= ra_max)


def dec_is_in_patch(dec, dec_min, dec_max):
    """Check whether the given declination coordinate is within a 
    patch of sky.
    
    Parameters
    ----------
    dec : float
        The declination.
    dec_min, dec_max : float
        The minimum and maximum declination coordinates of the patch 
        of sky. Must be in the same units as `dec`.
        
    Returns
    -------
    bool :
        `True` if the `dec` is in the range given by `dec_min` and 
        `dec_max`, `False` otherwise.
    """
    return (dec >= dec_min) and (dec <= dec_max)


def ra_dec_are_in_patch(ra, dec, ra_min, dec_min, ra_max, dec_max):
    """Check whether the given pair of right ascension and declination 
    coordinates are within a patch of sky.
    
    Parameters
    ----------
    ra, dec : float
        The right ascension and declination, respectively. Both should 
        have the same units.
    ra_min, ra_max : float
        The minimum and maximum right ascension coordinates of the patch 
        of sky. Must be in the same units as `ra`.
    dec_min, dec_max : float
        The minimum and maximum declination coordinates of the patch 
        of sky. Must be in the same units as `dec`.
        
    Returns
    -------
    bool :
        `True` if the `ra` is in the range given by `ra_min` and `ra_max`
        and the `dec` is in the range given by `dec_min` and  `dec_max`,
        `False` otherwise.
    """
    return ra_is_in_patch(ra, ra_min, ra_max) and dec_is_in_patch(dec, dec_min, dec_max)


# ========== misc. ==========

def tmsg(t):
    """Return a string that expresses the time `t` (`float`, units of 
    seconds) in minutes and seconds.
    """
    m = int(t // 60)
    s = int(t - (m * 60))
    return f'{m:>3d} min {s:>2d} sec'


def get_fdiff(x, y, percent=True):
    """Return the difference `x - y` between `x` and `y`, as a fraction of
    the second value `y`.

    Parameters
    ----------
    x, y : float or array_like of float
        The values to compare. If both `x` and `y` are arrays, their
        shapes must be compatible.
    percent : bool, default=True
        Whether to return the fractional difference as a percent. If
        `percent=True`, the fractional difference `(x-y)/y` is
        multiplied by 100.

    Returns
    -------
    fdiff : float or array_like of float
        The fractional difference.

    See Also
    --------
    get_frac_diff : 
        Fractional difference between arrays of different shapes.
    """
    fdiff = (x - y) / y
    if percent:
        fdiff *= 100
    return fdiff


def get_values_at_shared_points(x1, y1, x2, y2):
    """Find the elements of two arrays that correspond to the same points.

    The arrays `x1` and `x2` are the points at which the values in `y1`
    and `y2`, respectively, are evaluated.

    Parameters
    ----------
    x1, y1 : array_like
        The first pair of x and y values.
    x2, y2 : array_like
        The second pair of x and y values.

    Returns
    -------
    xvals : array_like
        The x-values that `x1` and `x2` have in common.
    yvals1, yvals2 : array_like
        The y-values in `y1` and `y2`, respectively, corresponding to the
        points in `xvals`.

    Examples
    --------
    >>> x1 = [0, 1, 2, 3]
    >>> y1 = [0, 1, 4, 9]
    >>> x2 = [0, 0.5, 1, 1.5, 2]
    >>> y2 = [0, 5, 10, 15, 20]
    >>> xvals, yvals1, yvals2 = get_values_at_shared_points(x1, y1, x2, y2)
    >>> xvals
    array([0, 1, 2])
    >>> yvals1
    array([0, 1, 4])
    >>> yvals2
    array([ 0, 10, 20])
    """
    # elements in `y1` corresponding to values in `x1` that are also in `x2`
    loc1 = np.isin(x1, x2)
    yvals1 = np.array(y1)[loc1].copy()
    xvals = np.array(x1)[loc1].copy() # only need one set of x values
    # elements in `y2` corresponding to values in `x2` that are also in `x1`
    loc2 = np.isin(x2, x1)
    yvals2 = np.array(y2)[loc2].copy()
    return xvals, yvals1, yvals2


def get_frac_diff(x1, y1, x2, y2, percent=True):
    """Calculate the fractional difference between two arrays.

    The fractional difference is the difference `y1 - y2` as a fraction
    of `y2`, i.e. `(y1-y2)/y2`. The arrays `x1` and `x2` are the points at
    which the values in `y1` and `y2`, respectively, are evaluated. If the
    two arrays are evaluated at different points, the fractional difference
    will be returned only at the points they share in common.

    Parameters
    ----------
    x1, y1 : array_like
        The first pair of x and y values.
    x2, y2 : array_like
        The second pair of x and y values.
    percent : bool, default=True
        Whether to return the fractional difference as a percent. If
        `percent=True`, the fractional difference `(y1-y2)/y2` is
        multiplied by 100.

    Returns
    -------
    xvals : array_like
        The points in common between `x1` and `x2`, at which the
        fractional difference is taken.
    frac_diff : array_like
        The fractional difference.

    See Also
    --------
    get_fdiff :
        Fractional difference of two arrays of the same shape and
        evaluated at the same points.
    """
    xvals, yvals1, yvals2 = get_values_at_shared_points(x1, y1, x2, y2)
    frac_diff = get_fdiff(yvals1, yvals2, percent=percent)
    return xvals, frac_diff


def trim_spectrum_ell_range(ells, spectrum, lmin=None, lmax=None):
    """Return a power spectrum and its corresponding multpoles within the
    multipole range.

    Parameters
    ----------
    ells : array_like of int or array_like of float
        The multipoles of the power spectrum
    spectrum : array_like of float
        The power spectrum.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipoles that define the range.
        - If both `lmin` and `lmax` are passed, the power spectrum in the
          range `lmin <= ell <= lmax` is returned.
        - If only `lmin` is passed, the the power spectrum in the range
          `ell >= lmin` is returned.
        - If only `lmax` is passed, the the power spectrum in the range
          `ell <= lmax` is returned.
        - If both `lmin` and `lmax` are `None`, the full power spectrum is
          returned unchanged.

    Returns
    -------
    ells, spectrum : array_like
        The multipoles and power spectrum within the given multipole range.
    """
    if lmin is None:
        lmin = ells[0]
    if lmax is None:
        lmax = ells[-1]
    loc = np.where((ells >= lmin) & (ells <= lmax))
    return ells[loc], spectrum[loc]


def fwhm2sigma(fwhm):
    """Return the standard deviation of a Gaussian profile from its
    full-width at half-maximum (FWHM).

    Parameters
    ----------
    fwhm: float
        The full-width at half-maximum for the Gaussian profile.

    Returns
    -------
    sigma : float
        The standard deviation of the Gaussian profile, in the same units
        as the `fwhm`.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma


def sigma2fwhm(sigma):
    """Return the full-width at half-maximum of a Gaussian profile from its
    standard deviation.

    Parameters
    ----------
    sigma: float
        The standard deviation of the Gaussian profile.

    Returns
    -------
    fwhm : float
        The full-width at half-maximum of the Gaussian profile, in the 
        same units as the `sigma`.
    """
    fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    return fwhm


def cl2dl(ells, cl):
    """Multiply a power spectrum C_ell by a factor of 
    ell * (ell + 1) / (2 * pi) at each multipole ell.

    Parameters
    ----------
    ells : array_like of int or array_like of float
        The multipoles at which the power spectrum is calculated.
    cl : array_like of float
        The power spectrum at each multipole in `ells`.

    Returns
    -------
    dl : array_like of float
        The power spectrum multiplied by a factor of 
        ell * (ell + 1) / (2 * pi). The power is set to zero at 
        multipoles ell < 2.
    """
    lfact = ells * (ells + 1) / (2 * np.pi)
    dl = cl.copy() * lfact
    loc = np.where(ells < 2)
    dl[loc] = 0
    return dl


def get_logger(log_file=None, level='debug', name=None, fmt=None, datefmt=None, use_simple_format=False):
    """Get a `logging.Logger` instance to log messages in a file.

    Parameters
    ----------
    log_file : str, default=None
        The full path to the file to be used for logging output.
        If None, logging output is printed to the terminal screen.
    level : str, default='debug'
        The minimum level of severity of the events passed to the logger.
        Valid options are `'debug'`, `'info'`, `'warning'`, `'error'`, and
        `'critical'`, in order of increasing severity ; you may also pass,
        e.g., `level=logging.DEBUG`.
    name : str, default=None
        An optional unique name to identify the logger. If None, the root 
        logger is used.
    fmt : str, default=None
        A format string used to format the messages written to the log.
        If None, a default is used. See the `logging` module documentation
        for details.
    datefmt : str, default=None
        A format string used to format the date and/or time of the message, if
        this information is included in the format string passed as `fmt`.
        If None, a default is used. See the `logging` module documentation
        for details.
    use_simple_format : bool, default=False
        If `fmt=None`, set `use_simple_format=True` to only include 
        information about the date and time in each message. By default,
        each logged message will include additional information (described
        in the "Notes" section below).
        
    Returns
    -------
    logger : `logging.Logger`
        An instance of the `logging.Logger` object.

    Raises
    ------
    ValueError
        If the `level` is not a valid logging level.

    See Also
    --------
    logging : The logging module in the Standard Python Library.

    Notes
    -----
    The `level` argument is case-insensitive.  Only messages at or above the
    specified `level` are written to the log file. If an invalid `level` is
    passed, the default is used.

    The default format of each log entry includes the date and time, the
    name of the file and function, along with the line number, from which the
    entry originated; and the level of severity of the message, followed
    by the message itself. The format is very long, but descriptive.

    If you pass your own format string, `style` (either '%' or '{') will be
    inferred by looking for one of the symbols, so you should not use both.
    
    Example
    -------
    In the following example, the file 'example.log' will contain a single line,
    "INFO: generating a random number".

    >>> fmt = '{levelname:s}: {message:s}' # choose a simple format
    >>> log = get_logger(log_file='example.log', level='info', fmt=fmt) # get the logger
    >>> log.info("generating a random number") # add a message, like an FYI
    >>> import random
    >>> x = random.uniform(-1,1)
    >>> if x < 0.0: log.debug("x is negative") # won't be logged b/c level='info'

    """
    
    # ---- set variables to get the logger ----

    # get (and create, if necessary) the log directory and file name
    if log_file is not None:
        log_dir = mkdir(os.path.dirname(log_file))

    # set the logging level
    levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 
              'error': logging.ERROR, 'critical': logging.CRITICAL}
    if level not in levels.values():
        if level.lower() not in levels.keys():
            raise ValueError(f"`{level = }`. Valid levels are: {list(levels.keys())}")
        level = levels[level.lower()]
    
    # set the formatting
    if use_simple_format:
        default_fmt = "[{asctime:s}] {message:s}"
    else:
        default_fmt = "[{asctime:s} | {funcName:s} in {filename:s} L {lineno:d} | {levelname:s}] {message:s}"
    default_style = '{'
    if fmt is None:
        fmt   = default_fmt
        style = default_style
    else:
        if '{' in fmt:
            style = '{'
        elif '%' in fmt:
            style = '%'
        else: # use the default and warn the user
            warnings.warn(f"You passed the following format string: {fmt = }, "
                          "but it does not contain any formatting placeholders, "
                          "so the default formatting will be used.")
            fmt = default_fmt
            style = default_style
            
    #default_datefmt = "%Y-%m-%d %H:%M:%S"
    if datefmt is None:
        #datefmt = default_datefmt
        datefmt = "%H:%M:%S" if use_simple_format else "%Y-%m-%d %H:%M:%S"

    # ---- get the logger ----
    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # get the handler
    if log_file is not None:
        # use `RotatingFileHandler` (instead of `FileHandler`) because it 
        # will automatically start over once log gets too large
        max_bytes = 100000000
        backup_count = 3
        handler = logging.handlers.RotatingFileHandler(log_file, mode='a', maxBytes=max_bytes, 
                                                       backupCount=backup_count)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)

    # get the formatter
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt, style=style)

    # put it all together
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

