"""Contains general functions to calculate the power spectra of 
simulations.

"""

import os
import numpy as np
import healpy as hp
from pixell import enmap
from pspy import so_mcm, so_spectra, sph_tools
from . import utils, maps


def get_uniform_bin_edges(lmax, bin_width):
    """Return an array of uniform bin edges.

    Parameters
    ----------
    lmax : int
        The maximum multipole of the bins.
    bin_width : int
        The uniform bin width.

    Returns
    -------
    bin_edges : array_like of int
        An array of bin edges. The first element is the lower edge of the
        first bin; the remaining elements are the upper edge of each bin.
    """
    bin_edges = np.arange(0, lmax, bin_width)
    # make sure bin edges start at ell = 2 and each bin has the same size:
    bin_edges[0] = 2
    bin_edges[1:] += 1
    return bin_edges


def get_bin_edges_and_centers(bin_edges, lmax=None):
    """Return arrays of the lower edge, upper edge, and center of each
    bin.

    Parameters
    ----------
    bin_edges : array_like of int
        An array of bin edges. The first element is the lower edge of the
        first bin; the remaining elements are the upper edge of each bin.
    lmax : int or None, default=None
        The maximum multipole of the bins. By default, the last element of
        `bin_edges` is used.

    Returns
    -------
    lower_edges, upper_edges : array_like of int or array_like of float
        Arrays of the lower and upper edges, respectively, of each bin.
    bin_centers : array_like of int or array_like of float
        The center of each bin.
    """
    if lmax is not None:
        bin_edges = bin_edges[bin_edges <= lmax]
    lower_edges = bin_edges[:-1].copy()
    lower_edges[1:] += 1
    upper_edges = bin_edges[1:].copy()
    bin_centers = (lower_edges + upper_edges) / 2
    return lower_edges, upper_edges, bin_centers


def save_uniform_binning_file(bin_file, lmax, bin_width):
    """Save a binning file for uniform binning. The file has columns for
    the lower edges, upper edges, and centers of each bin.

    Parameters
    ----------
    bin_file : str
        The path to the file.
    lmax : int
        The maximum multipole of the bins.
    bin_width : int
        The uniform bin width.
    """
    if not os.path.exists(bin_file):
        bin_edges = get_uniform_bin_edges(lmax, bin_width)
        lower_edges, upper_edges, bin_centers = get_bin_edges_and_centers(bin_edges, lmax=lmax)
        np.savetxt(bin_file, np.column_stack([lower_edges, upper_edges, bin_centers]))


def save_binning_file(bin_file, bin_edges=None, bin_width=None, lmax=None):
    """Save a binning file. The file has columns for the lower edges,
    upper edges, and centers of each bin.

    If no optional parameters are passed, the default uniform binning with
    a bin width of `200` is used.

    Parameters
    ----------
    bin_file : str
        The path to the file.
    bin_edges : array_like of int or array_like of float
        An array of bin edges. The first element is the lower edge of the
        first bin; the remaining elements are the upper edge of each bin.
    bin_width : int or None, default=None
        The bin width for uniform binning. If a `bin_width` is passed, the
        `lmax` parameter must be passed.
    lmax : int or None, default=None
        The maximum multipole of the bins. Must be passed if a `bin_width`
        is passed. Otherwise, by default, all bins are saved.
    """
    if bin_width is not None:
        if lmax is None:
            raise ValueError(f"`{lmax = }`. You must pass a maximum multipole to `lmax` when using uniform binning.")
        save_uniform_binning_file(bin_file, lmax, bin_width)
    else:
        edges = simutils.load_default_bin_edges() if (bin_edges is None) else bin_edges.copy()
        lower_edges, upper_edges, bin_centers = get_bin_edges_and_centers(edges, lmax=lmax)
        np.savetxt(bin_file, np.column_stack([lower_edges, upper_edges, bin_centers]))


def save_mode_coupling_files(mbb_inv, mbb_inv_fname, bbl=None, bbl_fname=None,
                             save_binning_matrix_as_npz=True, tol=1e-12):
    """Save the inverse mode-coupling and binning matrices.

    Parameters
    ----------
    mbb_inv : array_like of float or dict of array_like of float
        Either a single inverse mode-coupling matrix (applied to the power
        spectrum of a single map, i.e. without any polarization), or a
        dictionary of inverse mode-coupling matrices (applied to the power
        spectra of a set of temperature and polarization maps). The
        dictionary must have keys `'spin0xspin0'`, `'spin0xspin2'`,
        `'spin2xspin0'`, and `'spin2xspin2'`, where `'spin0'` and
        `'spin2'` refer to temperature and polarization maps,
        respectively.
    mbb_inv_fname : str or dict of str
        A single file name for the inverse mode-coupling matrix (if
        `mbb_inv` is a single array), or a dictionary of file names with
        the same keys as `mbb_inv` (if `mbb_inv` is a dictionary).
    bbl : array_like of float or dict of array_like of float or None, optional
        Either a single binning matrix (if `mbb_inv` is a single array) or
        a dictionary of binning matrices with the same keys as `mbb_inv`
        (if `mbb_inv` is a dictionary). If `bbl=None` (the default), the
        binning matrices are not saved.
    bbl_fname : str or dict of str or None, optional
         A single file name for the binning matrix (if `bbl` is a single
         array), or a dictionary of file names with the same keys as `bbl`
         (if `bbl` is a dictionary). Must be passed if `bbl` is not
         `None`; otherwise, `bbl_fname` will be ignored.

    Other Parameters
    ----------------
    save_binning_matrix_as_npz : bool, default=True
        Whether to save the binning matrices as sparse CSR `.npz` files
        to reduce the file sizes. If `save_binning_matrix_as_npz=False`,
        they will be saved as `.txt` files. Ignored if `bbl=None`.
    tol : float, default=1e-12
        The tolerance used when saving the binning matrices as sparse CSR
        `.npz` files. Ignored if `save_binning_matrix_as_npz=False` or
        `bbl=None`. See `hdsims.utils.save_sparse_matrix_npz` for more
        information.

    See Also
    --------
    load_mode_coupling_files : Load the files.
    calc_mode_coupling :
        Calculate the inverse mode-coupling and binning matrices.
    """
    if (bbl is not None) and (bbl_fname is None):
        raise ValueError(f"`{bbl_fname = }`. The `bbl_fname` must be provided if `bbl` is not `None`.")
    # if taking power of TQU maps, `mbb_inv` and `mbb_inv_fname`
    # will be dicts with a key for each spin pair:
    spin_pairs = ['spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2']
    # check if `mbb_inv` is dict with keys for each pair:
    pol = all([spin_pair in mbb_inv for spin_pair in spin_pairs])
    if (not pol) and ('spin0xspin0' not in mbb_inv):
        # put matrices and filenames into a dict anyway,
        # so we can loop through its keys:
        mbb_inv = {'spin0xspin0': mbb_inv.copy()}
        mbb_inv_fname = {'spin0xspin0': mbb_inv_fname}
        if bbl is not None:
            bbl = {'spin0xspin0': bbl.copy()}
            bbl_fname = {'spin0xspin0': bbl_fname}
    for spin_pair in mbb_inv.keys():
        np.savetxt(mbb_inv_fname[spin_pair], mbb_inv[spin_pair])
        if bbl is not None:
            if save_binning_matrix_as_npz:
                utils.save_sparse_matrix_npz(bbl_fname[spin_pair], bbl[spin_pair], tol=tol)
            else:
                np.savetxt(bbl_fname[spin_pair], bbl[spin_pair])


def load_mode_coupling_files(mbb_inv_fname, bbl_fname=None, save_binning_matrix_as_npz=True):
    """Load inverse mode-coupling and binning matrices.

    Parameters
    ----------
    mbb_inv_fname : str or dict of str
        A single file name for the inverse mode-coupling matrix, or a
        dictionary of file names with keys `'spin0xspin0'`,
        `'spin0xspin2'`, `'spin2xspin0'`, and `'spin2xspin2'`, where
        `'spin0'` and `'spin2'` refer to temperature and polarization
        maps, respectively.
    bbl_fname : str or dict of str or None, default=None
        A single file name for the binning matrix, or a dictionary of
        file names with the same keys as `mbb_inv_fname`. By default,
        the binning matrices won't be loaded.

    Returns
    -------
    mbb_inv : array_like of float or dict of array_like of float
        Either a single inverse mode-coupling matrix (if `mbb_inv_fname`
        is a single file name), or a dictionary of inverse mode-coupling
        matrices with the same keys as `mbb_inv_fname`.
    bbl : array_like of float or dict of array_like of float
        Only returned if `bbl_fname` is not `None`. Either a single
        binning matrix (if `bbl_fname` is a single file name), or a
        dictionary of binning matrices with the same keys as `bbl_fname`.


    Other Parameters
    ----------------
    save_binning_matrix_as_npz : bool, default=True
        Whether the binning matrices were saved as sparse CSR `.npz`
        files. Ignored if `bbl_fname=None`.

    See Also
    --------
    save_mode_coupling_files : Save the files.
    calc_mode_coupling :
        Calculate the inverse mode-coupling and binning matrices.
    """
    load_bbl = (bbl_fname is not None)
    # if taking power of TQU maps, `mbb_inv_fname` (and `bbl_fname`)
    # will be dicts with a key for each spin pair:
    spin_pairs = ['spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2']
    # check if `mbb_inv_fname` is dict with keys for each pair:
    pol = all([spin_pair in mbb_inv_fname for spin_pair in spin_pairs])
    if pol: # load in a matrix for each spin pair
        mbb_inv = {}
        bbl = {}
        for spin_pair in spin_pairs:
            mbb_inv[spin_pair] = np.loadtxt(mbb_inv_fname[spin_pair])
            if load_bbl:
                if save_binning_matrix_as_npz:
                    bbl[spin_pair] = utils.load_sparse_matrix_npz(bbl_fname[spin_pair])
                else:
                    bbl[spin_pair] = np.loadtxt(bbl_fname[spin_pair])
    else: # only one matrix
        mbb_inv = np.loadtxt(mbb_inv_fname)
        if load_bbl:
            if save_binning_matrix_as_npz:
                bbl = utils.load_sparse_matrix_npz(bbl_fname)
            else:
                bbl = np.loadtxt(bbl_fname)
    if load_bbl:
        return mbb_inv, bbl
    else:
        return mbb_inv



def bin_theory(theory_dict, binning_matrix, bin_edges, bin_dl=False, lmax=None):
    """Bin a dictionary of theory power spectra, using the same binning
    that is applied to the simulation power spectra.

    Parameters
    ----------
    theory_dict : dict of array_like of float
        The dictionary of theory power spectra. It should have a key
        `'ells'` for the multipoles, and either a key `'tt'` or `'kk'` for
        the temperature or lensing convergence power spectrum,
        respectively. In the latter case, the polarization power spectra
        may also be passed; if they are, they must have keys `'te'`,
        `'ee'`, and `'bb'` for the TE, EE, and BB power spectra,
        respectively.
        The CMB power spectra should not be multiplied by any multipole
        factors; if `bin_dl=True`, the power spectra will be multiplied by
        `ell * (ell + 1) / (2 * pi)` at each multipole `ell` before they
        are binned.
    binning_matrix : array_like of float or dict of array_like of float
        If the `theory_dict` has polarization power spectra,
        `binning_matrix` should be a dictionary of binning matrices with
        keys `'spin0xspin0'`, `'spin0xspin2'`, `'spin2xspin0'`, and
        `'spin2xspin2'`, where `'spin0'` and `'spin2'` refer to
        temperature and polarization maps, respectively.
    bin_edges : array_like of int or array_like of float
        An array of bin edges. The first element is the lower edge of the
        first bin; the remaining elements are the upper edge of each bin.
    bin_dl : bool, default=False
        If `bin_dl=True`, the power spectra will be multiplied by
        `ell * (ell + 1) / (2 * pi)` at each multipole `ell` before they
        are binned.
    lmax : int or None, default=None
        The maximum multipole of the theory power spectra to use.

    Returns
    -------
    binned_theory_dict : dict of array_like of float
        The dictionary of binned power spectra, with the same keys as
        `theory_dict`.

    See Also
    --------
    calc_mode_coupling : Calculate the inverse mode-coupling and
                         binning matrices.
    pspy.so_mcm.apply_Bbl : Apply the binning matrices to the power
                            spectra.
    """
    ells = theory_dict['ells']
    lfact = ells * (ells + 1) / (2 * np.pi) if bin_dl else np.ones(ells.shape)
    nspec = len(theory_dict.keys()) - 1

    # get array of bin centers:
    _, _, lbin = get_bin_edges_and_centers(bin_edges)

    # get the theory : trim ell-range, multiply by the `lfact`, etc.:
    if nspec > 1: # assume temperature & pol
        bbl = binning_matrix
        spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
        theo = {}
        for key in ['tt', 'te', 'ee', 'bb']:
            theo[key.upper()] = theory_dict[key][ells >= 2].copy()
            if bin_dl:
                theo[key.upper()] *= lfact[ells >= 2]
            if lmax is not None:
                theo[key.upper()] = theo[key.upper()][:lmax]
        theo['ET'] = theo['TE'].copy()
        for key in ['TB', 'BT', 'EB', 'BE']:
            theo[key] = np.zeros(theo['TT'].shape)
    else: # single spectrum
        # should have a single binning matrix:
        if 'spin0xspin0' in binning_matrix:
            bbl = binning_matrix['spin0xspin0']
        else:
            bbl = binning_matrix
        spectra = None
        theo_key = 'kk' if ('kk' in theory_dict.keys()) else 'tt'
        theo = theory_dict[theo_key][ells >= 2].copy()
        if bin_dl:
            theo *= lfact[ells >= 2]
        if lmax is not None:
            theo = theo[:lmax]

    # bin the theory
    binned_theo = so_mcm.apply_Bbl(bbl, theo, spectra=spectra)
    if nspec > 1: # return dict with lowercase keys
        binned_theory_dict = {key: binned_theo[key.upper()] for key in ['tt', 'ee', 'bb', 'te']}
        nbin = len(binned_theory_dict['tt'])
    else: # put the binned spectrum into a dict
        binned_theory_dict = {theo_key: binned_theo}
        nbin = len(binned_theo)
    binned_theory_dict['ells'] = lbin[:nbin]
    return binned_theory_dict


def calc_mode_coupling(window, lmax, binning_file, bin_dl=False, beam_fwhm=None, pol=True, pol_window=None, **kwargs):
    """Calculate the inverse mode-coupling and corresponding binning 
    matrices.
    
    Parameters
    ----------
    window : pixell.enmap.ndmap
        The apodization window applied to the maps before calculating 
        their power spectra.
    lmax : int
        The maximum multipole of the power spectra.
    binning_file : str
        The path to a binning file with columns for the upper edges, lower
        edges, and centers of each bin.
    bin_dl : bool, default=False
        Whether the binned power spectra will be multiplied by 
        `ell * (ell + 1) / (2 * pi)` at each multipole `ell`.
    beam_fwhm : int or float or None, default=None
        If the inverse mode-coupling matrix should also correct the power
        spectra for the effect of a Gaussian beam, pass the beam
        full-width at half-maximum (in arcminutes). Otherwise no
        correction for the beam is included.
    pol : bool, default=True
        If `pol=True`, inverse mode-coupling matrices and binning matrices
        are calculated for temperature and polarization power spectra. 
        Otherwise, polarization is not included.
    pol_window : pixell.enmap.ndmap, optional
        The apodization window applied to the polarization maps before 
        calculating their power spectra. By default, the same `window` is
        applied to each map (T, Q, and U). If `pol=True` and a 
        `pol_window` is passed, the `window` is only applied to the 
        temperature map, and the `pol_window` will be applied to the 
        polarization maps. Ignored if `pol=False`.
    
    Returns
    -------
    mbb_inv, bbl : array_like of float, or dict of array_like of float
        If `pol=False`, `mbb_inv` and `bbl` are the inverse mode-coupling 
        matrix and the corresponding binning matrix, respectively. The 
        mode-coupling matrix in this case is applied to the power spectrum
        of a single map (as opposed to T, Q, and U maps). The 
        mode-coupling matrix has shape `(nbin, nbin)`, and the binning 
        matrix has shape `(nbin, nl)`, where `nbin` and `nl` are the 
        number of bins or multipoles, respectively, up to `lmax`.

        If `pol=True`, `mbb_inv` and `bbl` are both dictionaries of
        arrays. The keys are `'spin0xspin0`, `'spin0xspin2'`,
        `'spin2xspin0`, `'spin2xspin2'`, where `'spin0'` and `'spin2'`
        refer to temperature and polarization maps, respectively.
    
    Other Parameters
    ----------------
    **kwargs : dict
        Additional keyword arguments passed to `so_mcm.mcm_and_bbl_spin0`
        or `so_mcm.mcm_and_bbl_spin0and2`. 
        
    See Also
    --------
    so_mcm.mcm_and_bbl_spin0, so_mcm.mcm_and_bbl_spin0and2
    """
    spec_type = 'dl' if bin_dl else 'cl'
    window = maps.enmap2pspy(window.copy())
    if beam_fwhm is not None:
        b_ell = hp.gauss_beam(utils.arcmin2rad(beam_fwhm), pol=False, lmax=lmax+500)
        bl1 = (b_ell, b_ell) if pol else b_ell
    else:
        bl1 = None
    kwargs = {**kwargs, 'lmax': lmax, 'bl1': bl1, 'type': spec_type.capitalize()}
    if 'niter' not in kwargs:
        kwargs['niter'] = 0
    if pol:
        if pol_window is None:
            pol_window = window
        else:
            pol_window = maps.enmap2pspy(pol_window.copy())
        mbb_inv, bbl = so_mcm.mcm_and_bbl_spin0and2((window, pol_window), binning_file, **kwargs)
    else:
        mbb_inv, bbl = so_mcm.mcm_and_bbl_spin0(window, binning_file, **kwargs)
    return mbb_inv, bbl



def take_sim_power(imap, window, lmax, binning_file, mbb_inv, bin_dl=False, deconvolve_pixwin=False, pol_window=None):
    """Calculate and bin the power spectra of a simulation, correcting
    for the effect of the apodization.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        The simulation.
    window : pixell.enmap.ndmap
        The apodization window applied to the simulation before taking its
        power.
    lmax : int
        The maximum multipole of the power spectra.
    binning_file : str
        The path to a binning file with columns for the upper edges, lower
        edges, and centers of each bin.
    mbb_inv : array_like of float, or dict of array_like of float
        If the simulation does not include polarization (Q and U maps),
        `mbb_inv` is a single inverse mode-coupling matrix, with shape
        `(nbin, nbin)`, where `nbin` is the number of bins up to `lmax`.
        Otherwise, `mbb_inv` should be a dictionary of inverse
        mode-coupling matrices with keys `'spin0xspin0`, `'spin0xspin2'`,
        `'spin2xspin0`, `'spin2xspin2'`, where `'spin0'` and `'spin2'`
        refer to temperature and polarization maps, respectively.
    bin_dl : bool, default=False
        Whether the binned power spectra will be multiplied by
        `ell * (ell + 1) / (2 * pi)` at each multipole `ell`.
    deconvolve_pixwin : bool, default=False
        If `deconvolve_pixwin=True`, the pixel window function will be
        deconvolved from the simulation before taking its power. If the
        simulation has been convolved with the pixel window, pass
        `deconvolve_pixwin=True`; otherwise, pass
        `deconvolve_pixwin=False`.
    pol_window : pixell.enmap.ndmap, optional
        The apodization window applied to the polarization maps before
        calculating their power spectra. If the `imap` contains T, Q,
        and U components and a `pol_window` is passed, the `window` is
        only applied to the temperature map, and the `pol_window` will be
        applied to the polarization maps. By default, the same `window`
        is applied to each map (T, Q, and U). Ignored if the `imap` only
        contains the temperature map.

    Returns
    -------
    sim_power : dict of array_like of float
        A dictionary of the binned power spectra of the simulation. It
        will always have a key `'ells'` for the binned multipoles. If the
        simulation includes polarization (T, Q, and U maps), the
        dictionary will have keys `'tt'`, `'te'`, `'ee'`, `'bb'` for the
        TT, TE, EE, and BB power spectra, respectively. Otherwise, it will
        only have the `'tt'` key.

    See Also
    --------
    pspy.sph_tools.get_alms
    pspy.so_spectra.get_spectra
    pspy.so_spectra.bin_spectra
    """
    spec_type = 'dl' if bin_dl else 'cl'
    # if taking power of TQU maps, `mbb_inv` will be a dict with a key 
    # for each spin pair:
    spin_pairs = ['spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2']
    # check if `mbb_inv` is dict with keys for each pair:
    pol = all([spin_pair in mbb_inv for spin_pair in spin_pairs]) 
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"] if pol else None

    if deconvolve_pixwin:
        if pol and (pol_window is not None):
            tqu_window = enmap.ones((3, *imap.shape[-2:]), imap.wcs)
            tqu_window[0] *= window
            tqu_window[1] *= pol_window
            tqu_window[2] *= pol_window
            imap = enmap.unapply_window(imap.copy() * tqu_window)
            pol_window = enmap.ones(imap.shape[-2:], imap.wcs)
        else:
            imap = enmap.unapply_window(imap.copy() * window)
        window = enmap.ones(imap.shape[-2:], imap.wcs)
    sim = maps.enmap2pspy(imap.copy())
    window = maps.enmap2pspy(window.copy())
    if pol:
        if pol_window is None:
            pol_window = window
        else:
            pol_window = maps.enmap2pspy(pol_window.copy())
        window = (window, pol_window) # (temp, pol)

    alms = sph_tools.get_alms(sim, window, niter=0, lmax=lmax)
    ells, cls = so_spectra.get_spectra(alms, spectra=spectra)
    lbin, binned_power = so_spectra.bin_spectra(ells, cls, binning_file, lmax, 
                                                type=spec_type.capitalize(), 
                                                mbb_inv=mbb_inv, spectra=spectra)
    if pol:
        # use lowercase keys:
        sim_power = {spec_name.lower(): spectrum for spec_name, spectrum in binned_power.items()}
        # add a key for the multipoles:
        sim_power['ells'] = lbin
    else:
        sim_power = {'ells': lbin, 'tt': binned_power}
    return sim_power


def calc_sim_power(imap, window, lmax, binning_file, mbb_inv_dict, 
                   bin_cl=None, bin_dl=None, deconvolve_pixwin=False, 
                   spectra=None, pol_window=None):
    """Calculate and bin the power spectra of a simulation, correcting
    for the effect of the apodization.

    Parameters
    ----------
    imap : pixell.enmap.ndmap
        The simulation.
    window : pixell.enmap.ndmap
        The apodization window applied to the simulation before taking its
        power.
    lmax : int
        The maximum multipole of the power spectra.
    binning_file : str
        The path to a binning file with columns for the upper edges, lower
        edges, and centers of each bin.
    mbb_inv_dict : nested dict of array_like of float
        A dictionary with keys for each `spec_type`: `'cl'` if
        `bin_cl=True`, and `'dl'` if `bin_dl=True`.
        If the simulation does not include polarization (Q and U maps),
        `mbb_inv_dict[spec_type]` is a single inverse mode-coupling
        matrix with shape `(nbin, nbin)`, where `nbin` is the number of
        bins up to `lmax`.
        Otherwise, `mbb_inv_dict[spec_type]` should be a dictionary of
        inverse mode-coupling matrices with keys `'spin0xspin0`,
        `'spin0xspin2'`, `'spin2xspin0`, `'spin2xspin2'`, where `'spin0'`
        and `'spin2'` refer to temperature and polarization maps,
        respectively.
    bin_cl : bool, default=False
        Whether to returned binned power spectra that have not been
        multiplied by any multipole factors.
    bin_dl : bool, default=False
        Whether to returned binned power spectra that have been multiplied
        by `ell * (ell + 1) / (2 * pi)` at each multipole `ell`.
    deconvolve_pixwin : bool, default=False
        If `deconvolve_pixwin=True`, the pixel window function will be
        deconvolved from the simulation before taking its power. If the
        simulation has been convolved with the pixel window, pass
        `deconvolve_pixwin=True`; otherwise, pass
        `deconvolve_pixwin=False`.
    spectra : list of str or None, optional
        A list of power spectra to return. The default is
        `['tt', 'te', 'ee', 'bb']` for the TT, TE, EE, and BB power
        spectra, respectively. Ignored if the simulation does not include
        polarization.
    pol_window : pixell.enmap.ndmap, optional
        The apodization window applied to the polarization maps before 
        calculating their power spectra. If the `imap` contains T, Q, 
        and U components and a `pol_window` is passed, the `window` is
        only applied to the temperature map, and the `pol_window` will be
        applied to the polarization maps. By default, the same `window` 
        is applied to each map (T, Q, and U). Ignored if the `imap` only
        contains the temperature map.

    Returns
    -------
    sim_power : dict of array_like of float
        A dictionary of the binned power spectra of the simulation, with
        the following keys and values:
        - `'ells'`: array of the binned multipoles.
        - `'cltt'` (if `bin_cl=True`), `'dltt'` (if `bin_dl=True`): The TT
            power spectrum (i.e., the power spectrum of a single map
            without any polarization).
        - `'clte'`, `'clee'`, `'clbb'` (if `bin_cl=True`),
          `'dlte'`, `'dlee'`, `'dlbb'` (if `bin_dl=True`): The TE, EE, and
            BB power spectra, respectively. Only calculated if the `imap`
            includes both temperature and polarization (T, Q, U maps).

    See Also
    --------
    pspy.sph_tools.get_alms
    pspy.so_spectra.get_spectra
    pspy.so_spectra.bin_spectra
    """
    # determine if binning C_ell's and/or D_ell's:
    if ('cl' not in mbb_inv_dict) and ('dl' not in mbb_inv_dict):
        raise ValueError(f"`mbb_inv_dict` must have a key for `'cl'` or a key for `'dl'`.")
    if (bin_cl is None) and ('cl' in mbb_inv_dict):
        bin_cl = True
    if (bin_dl is None) and ('dl' in mbb_inv_dict):
        bin_dl = True
    spec_types = []
    if bin_cl:
        spec_types.append('cl')
    if bin_dl:
        spec_types.append('dl')

    # take power of the `imap` and bin it:
    sim_power = {}
    if len(spec_types) > 0:
        # check if `mbb_inv_dict` for each `spec_type` is dict 
        # with keys for each spin pair:
        spin_pairs = ['spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2']
        pol = all([spin_pair in mbb_inv_dict[spec_types[0]] for spin_pair in spin_pairs]) 
        spectra_names = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"] if pol else None 
        if spectra is None: # list of spectra names to return
            spectra = ['tt', 'te', 'ee', 'bb'] if pol else ['tt']
    
        if deconvolve_pixwin:
            if pol and (pol_window is not None):
                tqu_window = enmap.ones((3, *imap.shape[-2:]), imap.wcs)
                tqu_window[0] *= window
                tqu_window[1] *= pol_window
                tqu_window[2] *= pol_window
                imap = enmap.unapply_window(imap.copy() * tqu_window)
                pol_window = enmap.ones(imap.shape[-2:], imap.wcs)
            else:
                imap = enmap.unapply_window(imap.copy() * window)
            window = enmap.ones(imap.shape[-2:], imap.wcs)
        sim = maps.enmap2pspy(imap.copy())
        window = maps.enmap2pspy(window.copy())
        if pol:
            if pol_window is None:
                pol_window = window
            else:
                pol_window = maps.enmap2pspy(pol_window.copy())
            window = (window, pol_window) # (temp, pol)

        alms = sph_tools.get_alms(sim, window, niter=0, lmax=lmax)
        ells, cls = so_spectra.get_spectra(alms, spectra=spectra_names)
        for spec_type in spec_types:
            lbin, binned_power = so_spectra.bin_spectra(ells, cls, binning_file, lmax, type=spec_type.capitalize(), 
                                                        mbb_inv=mbb_inv_dict[spec_type], spectra=spectra_names)
            sim_power['ells'] = lbin
            if pol: 
                # `binned_power` is a dict with keys for TT, TE, etc.
                # use lowercase keys and include `spec_type` (e.g. `'cltt'`):
                for spec_name, spectrum in binned_power.items():
                    if spec_name.lower() in spectra:
                        sim_power[f'{spec_type}{spec_name.lower()}'] = spectrum.copy()
            else: 
                # `binned_power` is a single array ; assume it's TT
                spec_name = spectra[0]
                sim_power[f'{spec_type}{spec_name}'] = binned_power.copy()

    return sim_power

