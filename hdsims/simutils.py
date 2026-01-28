"""Contains functions used when generating the simulations, and the `Sims`
class for simulations on a given patch of sky.

"""

import os
import numpy as np
from scipy.optimize import minimize
import camb
from . import utils, siminfo as si, maps


def hdsims_data_dir():
    """Return the path to the files needed to generate the simulations,
    provided with the `hdsims` package"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hdsims_data')


def load_default_bin_edges():
    """Load the bin edges for the default uniform binning with a bin width
    of 200.

    Returns
    -------
    bin_edges : array_like of int or array_like of float, optional
        An array of bin edges. The first element is the lower edge of the
        first bin (`bin_edges[0] = 2`), and the remaining elements are the
        upper edges of each bin.
    """
    fname = os.path.join(hdsims_data_dir(), 'bin_edges.txt')
    return np.loadtxt(fname)


def load_cl_ksz_template():
    """Load the kSZ template theory power spectrum.

    The power spectrum `C_ell` is in units of uK^2 without any
    multiplicative factors, i.e., not multiplied by a factor of
    `ell * (ell + 1) / (2*pi)` at each multipole `ell`. It is defined
    at each multipole starting from `ell=0`, and normalized such that
    `ell * (ell + 1) * C_ell / (2*pi) = 1` at `ell = 3000`.

    Returns
    -------
    ells, cls : array_like of float
        The multipoles and power spectrum, respectively.
    """
    fname = os.path.join(hdsims_data_dir(), 'cl_ksz_template.txt')
    ells, cls = np.loadtxt(fname, unpack=True)
    return ells, cls


def s10_camb_ini_file():
    """Return the path to the original Sehgal et. al. 2010
    (arXiv:0908.0540) CAMB input file.
    """
    fname = os.path.join(hdsims_data_dir(), 's10_camb_params.ini')
    return fname


def load_camb_theory():
    """Load the default CAMB theory unlensed CMB and lensing convergence
    power spectra.

    The power spectra were calculated using the same cosmology as the
    original Sehgal et. al. 2010 (arXiv:0908.0540) simulations and the
    CMB-HD accuracy settings.

    Returns
    -------
    theo : dict of array_like of float
        A dictionary of the theory power spectra with the following keys:
        - `'ells'`: array of each integer multipole between zero and
            40,000.
        - `'tt'`, `'te'`, `'ee'`, `'bb'`: unlensed CMB TT, TE, EE, and BB
            power spectra (in units of uK^2 as C_ell's, i.e. not
            multiplied by any multipole factors).
        - `'kk'`: the lensing convergence power spectrum, i.e.
            C_L^kappakappa = L^2 * (L+1)^2 * C_L^phiphi / 4.
    """
    camb_theory_fname = os.path.join(hdsims_data_dir(), 'camb_unlensed_cmb_and_clkk_theory.txt')
    theo = utils.load_dict_from_file(camb_theory_fname, si.spectra_col_names)
    return theo


def validate_sim_freq(freq, valid_freqs=si.freqs):
    """Verify that the given frequency is a valid frequency of the 
    simulations.
    
    The simulations are only generated at specific frequencies; by 
    default, these are 30, 90, 148, 219, 277, and 350 GHz.
    
    Parameters
    ----------
    freq : int or float
        The frequency (in GHz). 
    valid_freqs : list of int or list of float, optional
        A list of valid frequencies. By default, the list contains `30`,
        `90`, `148`, `219`, `277`, and `350` GHz.
        
    Returns
    -------
    freq : int or float
        The frequency.
        
    Raises
    ------
    ValueError
        If the `freq` is not equal to one of the frequencies in 
        `valid_freqs`.
        
    See Also
    --------
    validate_sim_freqs : Validate a list of frequencies.
    hdsims.siminfo.freqs : The default list of valid frequencies.
    """
    if freq not in valid_freqs:
        raise ValueError(f"`{freq = }`. The frequency `freq` (in units of GHz) must be one of {valid_freqs}.")
    return freq


def validate_sim_freqs(freqs, valid_freqs=si.freqs):
    """Verify that each given frequency is a valid frequency of the 
    simulations.
    
    The simulations are only generated at specific frequencies; by 
    default, these are 30, 90, 148, 219, 277, and 350 GHz.
    
    Parameters
    ----------
    freqs : list of int or list of float
        The list of frequencies (each in GHz). 
    valid_freqs : list of int or list of float, optional
        A list of valid frequencies. By default, the list contains `30`,
        `90`, `148`, `219`, `277`, and `350` GHz.
        
    Returns
    -------
    freqs : list of int or list of float
        The list of frequencies.
        
    Raises
    ------
    ValueError
        If the any of the frequencies in `freqs` are not equal to one of 
        the frequencies in `valid_freqs`.
        
    See Also
    --------
    validate_sim_freq : Validate a single frequency.
    hdsims.siminfo.freqs : The default list of valid frequencies.
    """
    # make sure `freqs` is a list instead of a single value:
    freqs = list(np.atleast_1d(freqs)) 
    invalid_freqs = [freq for freq in freqs if (freq not in valid_freqs)]
    if len(invalid_freqs) > 0:
        raise ValueError(f"`{freqs = }`. Each frequency (in units of GHz) in `freqs` must be one of {valid_freqs}.")
    return freqs


def validate_sim_component_name(component, valid_components=si.all_components):
    """Verify that the name of the given map component is a valid name.
    
    Parameters
    ----------
    component : str
        The name of the map component.
    valid_components : list of str, optional
        A list of valid component names. By default, the list contains
        `'tsz'` for the thermal SZ (tSZ); `'ksz'` for the kinetic SZ 
        (kSZ); `'cib'` for the cosmic infrared background (CIB); `'radio'` 
        for radio galaxies; `'cmb'` for the lensed, and  `'unlensed_cmb'`
        for the unlensed CMB.
        
    Returns
    -------
    component : str
        The name of the map component.
        
    Raises
    ------
    ValueError
        If the `component` is not equal to one of the component names in
        `valid_components`.
        
    See Also
    ---------
    validate_sim_component_names : Validate a list of component names.
    hdsims.siminfo.all_components : The default list of valid component 
                                    names.
    
    Notes
    -----
    The `component` name will always be converted to lowercase, so each
    name in `valid_components` should also be lowercase.
    """
    if component.lower() not in valid_components:
        raise ValueError(f"`{component = }`. The `component` must be one of {valid_components}.")
    return component.lower()


def validate_sim_component_names(components, valid_components=si.all_components):
    """Verify that each given map component name is a valid name.
    
    Parameters
    ----------
    components : list of str
        A list of map component names.
    valid_components : list of str, optional
        A list of valid component names. By default, the list contains
        `'tsz'` for the thermal SZ (tSZ); `'ksz'` for the kinetic SZ 
        (kSZ); `'cib'` for the cosmic infrared background (CIB); `'radio'`
        for radio galaxies; `'cmb'` for the lensed, and  `'unlensed_cmb'`
        for the unlensed CMB.
        
    Returns
    -------
    component : str
        The list of map component names.
        
    Raises
    ------
    ValueError
        If the any of the names in `components` are not equal to one of 
        the component names in `valid_components`.
        
    See Also
    ---------
    validate_sim_component_name : Validate a single component name.
    hdsims.siminfo.all_components : The default list of valid component 
                                    names.
    
    Notes
    -----
    The `component` name will always be converted to lowercase, so each
    name in `valid_components` should also be lowercase.
    """
    # make sure `components` is a list instead of a single value:
    components = [c.lower() for c in np.atleast_1d(components)] 
    invalid_components = [component for component in components if (component not in valid_components)]
    if len(invalid_components) > 0:
        raise ValueError(f"`{components = }`. Each component in `components` must be one of {valid_components}.")
    return components


def validate_map_components(components, valid_components=si.all_components):
    """Verify that all individual map components in the given list can be
    added to the same map.

    Parameters
    ----------
    components : list of str
        A list of individual map component names.
    valid_components : list of str, optional
        A list of all valid component names. By default, the list contains
        `'tsz'` for the thermal SZ (tSZ); `'ksz'` for the kinetic SZ
        (kSZ); `'cib'` for the cosmic infrared background (CIB); `'radio'`
        for radio galaxies; `'cmb'` for the lensed, and  `'unlensed_cmb'`
        for the unlensed CMB.

    Returns
    -------
    components : list of str
        The list of individual map component names.

    Raises
    ------
    ValueError
        If the list contains `'kappa'` in addition to other components
        (because the lensing convergence map is dimensionless, while the
        others are in units of uK); or if the list contains both `'cmb'`
        and `'unlensed_cmb'` (because we can't add the CMB twice to the
        same map).
    """
    # make sure each individial component in the list is a valid name:
    components = validate_sim_component_names(components, valid_components=valid_components)
    # make sure we're not adding other components to a kappa map:
    if ('kappa' in components) and (len(components) > 1):
        raise ValueError(f"`{components = }`. The only allowed `component` in the kappa map is `'kappa'`.")
    # make sure we're not trying to add both the lensed and unlensed cmb
    # to the same map:
    if ('cmb' in components) and ('unlensed_cmb' in components):
        raise ValueError(f"`{components = }`. You cannot add both the lensed CMB (`'cmb'`) "
                         "and the unlensed CMB (`'unlensed_cmb'`) to the same map.")
    return components


def validate_cib_model_name(cib_model, valid_model_names=[*si.cib_model_names, 's10']):
    """Verify that the name of the given CIB model is a valid name.
    
    The 'CIB model' refers to the the CIB catalog for the simulations is
    generated from the original full-sky catalog of the Sehgal et. al. 
    (arXiv:0908.0540) simulations. See arXiv:XXXX.XXXXX (!! TODO:LINK2PAPER !!)
    for further details.
    
    Parameters
    ----------
    cib_model : str
        The name of the CIB model.
    valid_model_names : list of str, optional
        A list of valid CIB model names. The list includes `'s10'` (which 
        refers to the case where the original S10 catalog is not modified)
        and the default list of CIB model names in 
        `hdsims.siminfo.cib_model_names`.
        
    Returns
    -------
    cib_model : str
        The name of the CIB model. Will be all lowercase.
        
    Raises
    ------
    ValueError
        If the CIB model name is not in the list of `valid_model_names`.
        
    See Also
    --------
    hdsims.siminfo.cib_model_names : The list of valid CIB model names.
    """
    if cib_model.lower() not in valid_model_names:
        raise ValueError(f"Unknown `{cib_model = }`. The CIB model name must be one of {valid_model_names}.")
    return cib_model.lower()


def separate_component_list(components):
    """Return separate lists of the CMB, foreground, and lensing 
    convergence component name(s) in the given list.
    
    Parameters
    ----------
    components : list of str
        A list of individual map component names. Each name in the list
        must be one of `'tsz'`, `'ksz'`, `'cib'`, `'radio'`, `'kappa'`,
        `'cmb'`, or `'unlensed_cmb'`.
        
    Returns
    -------
    cmb : list of str
        A list of all CMB map component names (`'cmb'`, `'unlensed_cmb'`)
        that are in the list of `components`.
    fgs : list of str
        A list of all extragalactic foreground map component names 
        (`'tsz'`, `'ksz'`, `'cib'`, `'radio'`) that are in the list of 
        `components`.
    kappa : list of str
        A list containing `'kappa'`, if `'kappa'` is in the list of 
        `components`.
    """
    components = validate_sim_component_names(components)
    cmb = [] # component name(s) for cmb
    fgs = [] # component name(s) for extragalactic foregrounds
    kappa = [] # component name for kappa
    for component in components:
        if 'cmb' in component:
            cmb.append(component)
        elif 'kappa' in component:
            kappa.append(component)
        else:
            fgs.append(component)
    return cmb, fgs, kappa


def has_cmb(components):
    """Check if the list of map component names contains the CMB.

    Parameters
    ----------
    components : list of str
        A list of individual map component names. Each name in the list
        must be one of `'tsz'`, `'ksz'`, `'cib'`, `'radio'`, `'kappa'`,
        `'cmb'`, or `'unlensed_cmb'`.

    Returns
    -------
    bool
        `True` if `'cmb'` or `'unlensed_cmb'` is in the list of
        `components`, `False` otherwise.
    """
    components_list = validate_sim_component_names(components)
    return any(['cmb' in c for c in components_list])


def cmb_component_name(components):
    """Return the name of the CMB (either lensed or unlensed) component in
    the given list.

    Parameters
    ----------
    components : list of str
        A list of individual component names. The list should contain only
        components that can be added to the same map; i.e., it cannot
        contain both the lensed and unlensed CMB. Each name in the list
        must be one of `'tsz'`, `'ksz'`, `'cib'`, `'radio'`, `'kappa'`,
        `'cmb'`, or `'unlensed_cmb'`.

    Returns
    -------
    str or None
        If the `components` list contains either `'cmb'` or
        `'unlensed_cmb', returns the name of the CMB component (`'cmb'` or
        `'unlensed_cmb'`). Otherwise, returns `None`.

    See Also
    --------
    validate_map_components :
        Check whether all individual map components in a list can be added
        to the same map.
    """
    # `validate_map_components` makes sure only lensed or unlensed CMB is
    # in the list (not both)
    components_list = validate_map_components(components)
    if has_cmb(components_list):
        cmb_components, _, _ =  separate_component_list(components_list)
        return cmb_components[0]


def has_pixwin(components):
    """Check whether the map made from the given individual components
    should be convolved with a pixel window.

    All maps except the lensing convergence map are convolved with the
    pixel window function.

    Parameters
    ----------
    components : list of str
        A list of individual component names. The list should contain only
        components that can be added to the same map; i.e., it cannot
        contain both the lensed and unlensed CMB. Each name in the list
        must be one of `'tsz'`, `'ksz'`, `'cib'`, `'radio'`, `'kappa'`,
        `'cmb'`, or `'unlensed_cmb'`.

    Returns
    -------
    bool
        `True` if the list of `components` does not contain `'kappa'`,
        `False` otherwise.

    See Also
    --------
    validate_map_components :
        Check whether all individual map components in a list can be added
        to the same map.
    """
    components_list = validate_map_components(components)
    return not any(['kappa' in c for c in components_list])


def has_freq_dependent_component(components):
    """Check whether any of the given individual map components depend
    on frequency.

    Parameters
    ----------
    components : list of str
        A list of individual component names. Each name in the list must
        be one of `'tsz'`, `'ksz'`, `'cib'`, `'radio'`, `'kappa'`,
        `'cmb'`, or `'unlensed_cmb'`.

    Returns
    -------
    bool
        `True` if the list of `components` contains `'tsz'`, `'cib'`, or
        `'radio'`; `False` otherwise.
    """
    components_list = validate_sim_component_names(components)
    return any([c in components_list for c in ['tsz', 'cib', 'radio']])


def get_spectra_keys(components, pol=True, cl=False, dl=False):
    """Return a list of column names, in the correct order, used when
    saving power spectra to a file; these are also used as the keys in a
    dictionary of the power spectra.

    For example, a file of the power spectra of a CMB simulation (or CMB
    theory power spectra) will have columns for the multipoles and the TT,
    EE, BB, and TE power spectra; the columns will always be in a fixed
    order. The dictionary of CMB power spectra will have keys `'ells'`,
    `'tt'`, `'ee'`, `'bb'`, and `'te'`.

    Parameters
    ----------
    components : list of str
        A list of the names of the individual components in the
        simulation.
    pol : bool, default=True
        Whether the simulation includes CMB polarization (i.e. has T, Q,
        and U maps).
    cl, dl : bool, default=False
        Whether the keys for the power spectra should include `'cl'` or
        `'dl'`, respectively. For example, the default key for the
        temperature power spectrum is `'tt'`; if `cl=True` or `dl=True`,
        it will be`'cltt'` or `'dltt'`, respectively. If both `cl=True`
        and `dl=True`, `cl` will be ignored.

    Returns
    -------
    keys : list of str
        A list of column names (in the correct order) and dictionary keys
        for the power spectra. The first element in the list will always
        be `'ells'` for the multipoles.
        - If the list of `components` includes `'cmb'` or `'unlensed_cmb'`
          and `pol=True`, the other keys are `'tt'`, `'ee'`, `'bb'`, and
          `'te'` (or `'cltt'`/`'dltt'`, etc. if `cl=True`/`dl=True`) for
          the TT, EE, BB, and TE power spectra. If `pol=False`, only the
          key for the TT power spectrum is included.
        - If the list of `components` contains `'kappa'`, the other key
          is `'kk'` for the lensing convergence power spectra.
        - Otherwise, the other key is `'tt'` for the temperature power
          spectrum.
        If `cl=True` or `dl=True`, each power spectrum key will start with
        `'cl'` or `'dl'` (e.g., `'cltt'` or `'dltt'` instead of `'tt'`),
        respectively.
    """
    components = validate_sim_component_names(components)
    if 'kappa' in components:
        if len(components) > 1 :
            # assume keys are for theory spectra instead of sim spectra
            key_names = si.spectra_col_names.copy()
        else:
            key_names = ['ells', 'kk']
    elif has_cmb(components) and pol:
        key_names = si.spectra_col_names[:-1]
    else:
        key_names = si.spectra_col_names[:2]
    if cl or dl:
        spec_type = 'dl' if dl else 'cl'
        keys = ['ells']
        for spec_name in key_names[1:]:
            keys.append(f'{spec_type}{spec_name}')
    else:
        keys = key_names
    return keys


def get_tqu_noise_seeds(temp_seed):
    """Return a list of seeds to use when generating temperature and
    polarization white noise maps.

    Parameters
    ----------
    temp_seed : int
        The seed for the temperature map.

    Returns
    -------
    list of int
        A list of three seeds for the T, Q, and U maps.

    Notes
    -----
    The seed for the Q and U maps are `temp_seed + 1000` and
    `temp_seed + 2000`, respectively.
    """
    return [temp_seed, temp_seed + 1000, temp_seed + 2000]


def get_default_cambparams_for_sim(lmax=si.lmax4theo):
    """Initialize the default CAMB parameters with the cosmology of the
    Sehgal et. al. 2010 (arXiv:0908.0540) simulations and the accuracy
    settings for CMB-HD.

    Parameters
    ----------
    lmax : int, default=40000
        The maximum multipole to use for the CAMB calculations.

    Returns
    -------
    pars : camb.model.CAMBparams
        The CAMB parameters instance.

    See Also
    --------
    get_cambparams_for_sim : CAMB parameters for any cosmology.
    s10_camb_ini_file : The default cosmology.
    hdsims.siminfo.hd_camb_accuracy_params : The CMB-HD accuracy
                                             parameters.
    """
    # read in the original S10 CAMB `.ini` file:
    pars = camb.read_ini(s10_camb_ini_file())
    # get the cosmo params and update with massive neutrinos
    cosmo_params = {'H0': pars.H0, 'ombh2': pars.ombh2, 'omch2': pars.omch2, 'nnu': pars.N_eff,
                    'Alens': pars.Alens, 'YHe': pars.YHe, 'TCMB': pars.TCMB, 'mnu': 0.06, 'num_massive_neutrinos': 3}
    # update accuracy:
    pars.set_accuracy(AccuracyBoost=1.1, lSampleBoost=3.0, lAccuracyBoost=3.0,
                      DoLateRadTruncation=False, min_l_logl_sampling=10000)
    pars.set_cosmology(**cosmo_params)
    pars.set_matter_power(kmax=10, k_per_logint=130)
    pars.set_for_lmax(lmax+500, lens_potential_accuracy=30, lens_margin=2050)
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params("mead2016")
    return pars


def get_cambparams_for_sim(lmax=si.lmax4theo, cosmo_params=None):
    """Initialize the CAMB parameters for the given cosmology with the
    accuracy settings for CMB-HD.

    Parameters
    ----------
    lmax : int, default=40000
        The maximum multipole to use for the CAMB calculations.
    cosmo_params : dict or None, default=None
        A dictionary of cosmological parameter names and their values
        that can be passed to the `camb.set_params` function. By default,
        the cosmology of the  Sehgal et. al. 2010 (arXiv:0908.0540)
        simulations is used.

    Returns
    -------
    pars : camb.model.CAMBparams
        The CAMB parameters instance.

    See Also
    --------
    get_default_cambparams_for_sim : Default CAMB parameters.
    camb.set_params
    """
    if cosmo_params is None:
        pars = get_default_cambparams_for_sim(lmax=lmax)
    else:
        camb_params_dict = {**cosmo_params, **si.hd_camb_accuracy_params, 'lmax': lmax + 500}
        pars = camb.set_params(**camb_params_dict)
    return pars


def calculate_camb_unlensed(lmax=si.lmax4theo, cosmo_params=None, camb_params=None, camb_results=None):
    """Calculate the CAMB theory unlensed CMB and lensing convergence
    power spectra.

    Parameters
    ----------
    lmax : int, default=40000
        The maximum multipole to use for the CAMB calculations.
    cosmo_params : dict or None, default=None
        A dictionary of cosmological parameter names and their values
        that can be passed to the `camb.set_params` function. By default,
        the cosmology of the  Sehgal et. al. 2010 (arXiv:0908.0540)
        simulations is used.
    camb_params : camb.model.CAMBparams or None, default=None
        A CAMB parameters instance to use for the calculation. If passed,
        the `cosmo_params` parameter will be ignored.
    camb_results : camb.results.CAMBdata
        A CAMB results instance to use for the calculation. If passed, the
        `cosmo_params` and `camb_params` parameters will be ignored.

    Returns
    -------
    theo : dict of array_like of float
        A dictionary of the theory power spectra with the following keys
        and values:
        - `'ells'`: array of each integer multipole between zero and the
            maximum given by the `lmax` parameter.
        - `'tt'`, `'te'`, `'ee'`, `'bb'`: unlensed CMB TT, TE, EE, and BB
            power spectra (in units of uK^2 as C_ell's, i.e. not
            multiplied by any multipole factors).
        - `'kk'`: the lensing convergence power spectrum, i.e.
              C_L^kappakappa = L^2 * (L+1)^2 * C_L^phiphi / 4.
    """
    if camb_results is None:
        if camb_params is None:
            camb_params = get_cambparams_for_sim(lmax=lmax, cosmo_params=cosmo_params)
        camb_results = camb.get_results(camb_params)
    theo = {'ells': np.arange(lmax+1)}
    # unlensed cmb:
    powers = camb_results.get_cmb_power_spectra(camb_params, CMB_unit='muK', raw_cl=True, lmax=lmax)
    for i, s in enumerate(['tt', 'ee', 'bb', 'te']):
        theo[s] = powers['unlensed_total'][:,i].copy()
        theo[s][:2] = 0
    # lensing convergence:
    theo['kk'] = camb_results.get_lens_potential_cls(lmax=lmax)[:,0] * 2 * np.pi / 4
    return theo


def get_cl_theory_from_template(template_ells, template_cls, amp=1, slope=0, ell0=3000):
    """Return the template theory power spectrum scaled by the given 
    amplitude and slope, with a pivot at the multipole `ell0`; 
    i.e., `theo_cls = amp * (template_ells / ell0)**n * template_cls`.
    
    Parameters
    ----------
    template_ells, template_cls : array_like of float
        The multipoles and template power spectrum, respectively.
    amp, slope : int or float, optional
        The amplitude and slope parameters, respectively. By default,
        `amp=1` and `slope=0`, which leaves the template power spectrum
        unchanged.
    ell0 : int, default=3000
        The pivot scale.
    
    Returns
    -------
    theo_cls : array_like of float
        The theory power spectrum.
    """
    loc = np.where(template_ells >= 2)
    theo_cls = np.zeros(template_cls.shape)
    theo_cls[loc] = amp * (template_ells[loc] / ell0)**slope * template_cls[loc]
    return theo_cls


def fit_theory_to_spectrum(ells, cls, template_ells, template_cls, ell_to_fit_slope, ell_to_fit_amp=3000):
    """Fit amplitude and slope parameters, used to scale a template 
    theory power spectrum, to a measured power spectrum.
    
    Parameters
    ----------
    ells, cls : array_like of float
        The measured multipoles and power spectrum, which may be binned.
    template_ells, template_cls : array_like of float
        The multipoles and template theory power spectrum, which should 
        not be binned.
    ell_to_fit_slope : int
        The multipole at which to match the measured and template power 
        spectra in order to fit the slope.
    ell_to_fit_amp : int, default=3000
        The multipole at which to match the measured and template power 
        spectra in order to fit the amplitude.
    
    Returns
    -------
    theo_ells, theo_cls : array_like of float
        The multipoles and theory power spectrum, obtained by scaling the
        template power spectrum by the amplitude and slope parameters.
    amp, slope : float
        The amplitude and slope parameters that were fit to the measured 
        power spectrum.
        
    See Also
    --------
    get_cl_theory_from_template : Scale a template power spectrum with 
                                  amplitude and slope parameters.
    """
    # get array indices corresponding to multipole `ell_to_fit_amp` 
    # to fit the amplitude:
    sim_amp_index = np.argmin(np.abs(ells - ell_to_fit_amp))
    sim_amp_bin_ctr = ells[sim_amp_index]
    template_amp_index = np.argmin(np.abs(template_ells - sim_amp_bin_ctr))
    amp = np.mean(cls[sim_amp_index-1:sim_amp_index+2]) / template_cls[template_amp_index]

    # minimize squared difference between theory template (using the 
    # fit amplitude) and sim power at `ell_to_fit_slope` to fit the slope:
    sim_slope_index = np.argmin(np.abs(ells - ell_to_fit_slope))
    sim_slope_bin_ctr = ells[sim_slope_index]
    avg_sim_power_at_ell4slope = np.mean(cls[sim_slope_index-1:sim_slope_index+2])
    template_slope_index = np.argmin(np.abs(template_ells - sim_slope_bin_ctr))
    # define the fit theory curve with fixed amplitude 
    # as a function of the slope:
    ell0 = round(sim_amp_bin_ctr)
    fit_theo_cls_at_ell4slope = lambda n: get_cl_theory_from_template(template_ells, template_cls, 
                                                                      amp=amp, slope=n, ell0=ell0)[template_slope_index]
    # define function to minimize:
    diff_to_minimize = lambda n: (avg_sim_power_at_ell4slope - fit_theo_cls_at_ell4slope(n))**2
    # fit the slope:
    slope_guess = [0]
    slope_bounds = [(-5, 5)]
    slope = minimize(diff_to_minimize, slope_guess, method='Nelder-Mead', bounds=slope_bounds).x[0]

    # get the theory curve using the fit params:
    theo_ells = template_ells.copy()
    theo_cls = get_cl_theory_from_template(template_ells, template_cls, amp=amp, slope=slope, ell0=ell0)
    return theo_ells, theo_cls, amp, slope


def round_str(x, n=2):
    """Return a string formatted with rounded value, removing any unnecessary zeros.

    This is used for file names.

    Parameters
    ----------
    x : int or float
        The value to round.
    n : int, default=2
        The number of digits to round `x` to.

    Returns
    -------
    str
        The rounded value as a string, with any extra zeros removed.

    Examples
    --------
    >>> f'{round(10.0, 2)}'
    '10.0'
    >>> round_str(10.0, n=2)
    '10'
    """
    s = f'{round(x, n)}'
    if s.endswith('.0'):
        s = s[:-2]
    return s


def get_output_dirs(hd_sims_dir, ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr,
                    width=si.width, height=si.height, res=si.hd_res, make_dirs=True):
    """Return a dictionary of output directory paths for the simulations
    on a given patch of the sky.

    Parameters
    ----------
    hd_sims_dir : str
        The path to the directory where all of the output files will
        be saved. All output files for simulations on a given patch of
        the sky will be saved to a sub-directory within the `hd_sims_dir`.
    ra_ctr, dec_ctr : int or float, default=6
        The right ascension (R.A.) and declination (dec.), in degrees, of
        the center of the patch of sky.
    width, height : int or float, default=10
        The width and height (in degrees) of the patch of sky.
    res : int or float, default=0.04
        The resolution of the simulations, in arcminutes.
    make_dirs : bool, default=True
        Whether to create the directories.

    Returns
    -------
    output_dirs : dict of str
        A dictionary of output directory paths with the following keys and
        values:
        - `'sims'` : Path to the 'main' output directory where the maps,
            catalogs, and theory power spectra are saved. All other
            directories are sub-directories of this one.
        - `'spectra'` : Path to the directory where the power spectra of
            the simulations are saved.
        - `'binning'` : Path to the directory where the binning files and
            inverse mode-coupling matrices are saved. This is a
            sub-directory of the simulation spectra directory.
        - `'intermediate'` : Path to the directory where all of the output
            from the intermediate, lower-resolution simulations is saved.
    """
    patch_info = f'ra{round_str(ra_ctr)}dec{round_str(dec_ctr)}_{round_str(width)}x{round_str(height)}deg'
    res_info = f'{round_str(res)}arcmin' if (round(res,2) != si.hd_res) else 'hdsims'
    output_dirs = {'sims': os.path.join(hd_sims_dir, f'{patch_info}_{res_info}')}
    output_dirs['spectra'] = os.path.join(output_dirs['sims'], 'spectra')
    output_dirs['binning'] = os.path.join(output_dirs['spectra'], 'binning_and_mode_decoupling')
    output_dirs['intermediate'] = os.path.join(output_dirs['sims'], 'intermediate_maps')
    output_dirs['plots'] = os.path.join(output_dirs['sims'], 'plots')
    if make_dirs:
        for dir_name, dir_path in output_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
    return output_dirs


def get_catalog_fname(component, width, height=None, catalog_dir=None,
                      file_ext='csv', cib_model=si.baseline_cib_model_name):
    """Return the file name of the simulation catalog for the given map
    component and patch of sky.

    Parameters
    ----------
    component : str
        The name of the map component. Must be `'cib'` for the CIB,
        `'radio'` for radio galaxies, or `'tsz'`, `'ksz'`, or `'sz'` for
        SZ clusters.
    width : int or float
        The width (in degrees) of the patch of sky.
    height : int or float or None, default=None
        The height (in degrees) of the patch of sky. If `height=None`, the
        value of the `width` is used.
    catalog_dir : str or None, default=None
        The path to the directory where the catalog is saved. If
        `catalog_dir` is passed, the absolute path to the file is
        returned; otherwise, only the file name (i.e. relative path)
        is returned.
    file_ext : str, default='csv'
        The file extension.
    cib_model : str or None, optional
        The name of a CIB model. Only used if `component='cib'`.

    Returns
    -------
    fname : str
        The catalog file name.
    """
    height = width if (height is None) else height
    patch_info = f'{round_str(width)}x{round_str(height)}deg'
    if 'sz' in component.lower():
        component_info = 'sz'
    elif (component.lower() == 'cib') and (cib_model is not None):
        component_info = f'cib_{cib_model}' if (cib_model.lower() != si.baseline_cib_model_name) else 'cib'
    else:
        component_info = component.lower()
    if file_ext.startswith('.'):
        file_ext = file_ext[1:]
    fname = f'{component_info}_{patch_info}.{file_ext}'
    if catalog_dir is not None:
        fname = os.path.join(catalog_dir, fname)
    return fname


def get_apod_window_fname(apod_width, width, height, maps_dir=None):
    """Return the file name of the apodization window.

    Parameters
    ----------
    apod_width : int or float
        The apodization width, in degrees.
    width, height : int or float
        The width and height of the apodization window, in degrees.
    maps_dir : str or None, default=None
        The path to the directory where the window is saved. If `maps_dir`
        is passed, the absolute path to the file is returned; otherwise,
        only the file name (i.e. relative path) is returned.

    Returns
    -------
    fname : str
        The file name.

    See Also
    --------
    hdsims.maps.make_apod_window : Create an apodization window.
    """
    patch_info = f'{round_str(width)}x{round_str(height)}deg'
    fname = f'window_{patch_info}_apod{round_str(apod_width)}deg.fits'
    if maps_dir is not None:
        fname = os.path.join(maps_dir, fname)
    return fname


def get_binning_file_name(bin_width=None, lmax=None, bin_info=None, binning_dir=None):
    """Return the file name of the binning file. By default, the file name
    of the default binning file is returned (without its absolute path).

    Parameters
    ----------
    bin_width : int or None, default=None
        The size of the bins, if using uniform binning. If a `bin_width`
        is passed, the `lmax` parameter must also be passed.
    lmax : int or None, default=None
        The maximum multipole in the binning file. Must be passed if
        `bin_width` is also passed.
    bin_info : str or None, default=None
        Any additional information to add to the file name.
    binning_dir : str or None, default=None
        The path to the directory where the binning file is saved. If
        `binning_dir` is passed, the absolute path to the file is
        returned; otherwise, only the file name (i.e. relative path) is
        returned.

    Returns
    -------
    fname : str
        The file name.
    """
    fname_info = ['binning']
    if bin_width is not None:
        if lmax is None:
            raise ValueError(f"`{lmax = }`. You must pass a maximum multipole to `lmax` when using uniform binning.")
        fname_info.append(f'delta{int(bin_width)}')
    if bin_info is not None:
        fname_info.append(bin_info)
    if lmax is not None:
        fname_info.append(f'lmax{int(lmax)}')
    fname_root = '_'.join(fname_info)
    fname = f'{fname_root}.txt'
    if binning_dir is not None:
        fname = os.path.join(binning_dir, fname)
    return fname


def get_mode_coupling_fnames(apod_width, lmax, bin_dl=False, pol=True, beam_fwhm=None,
                             fname_info=None, binning_dir=None, save_binning_matrix_as_npz=True):
    """Return the file names for the inverse mode-coupling and binning matrices.

    Parameters
    ----------
    apod_width : int or float
        The apodization width (in degrees) used when apodizing a map
        before calculating its power spectra.
    lmax : int
        The maximum multipole used for the calculation.
    bin_dl : bool, default=False
        Whether the power spectra being binned should be multiplied
        by a factor of  `ell * (ell + 1) / (2 pi)` at each multipole
        `ell`.
    pol : bool, default=True
        Whether the simulation whose power spectrum is being measured has
        both temperature and polarization maps.
    beam_fwhm : int or float or None, default=None
        If the inverse mode-coupling matrix should also correct the power
        spectra for the effect of a Gaussian beam, pass the beam
        full-width at half-maximum (in arcminutes). Otherwise no
        correction for the beam is included.
    binning_dir : str or None, default=None
        The path to the directory where the files are saved. If
        `binning_dir` is passed, the absolute paths to the files are
        returned; otherwise, only the file names (i.e. relative paths) are
        returned.

    Returns
    -------
    mcm_fname, bbl_fname : str or dict of str
            If `pol=False`, `mcm_fname` and `bbl_fname` are the file names
            of the inverse mode-coupling matrix and the corresponding
            binning matrix, respectively. The mode-coupling matrix in this
            case is applied to the power spectrum of a single map
            (as opposed to T, Q, and U maps).

            If `pol=True`, `mcm_fname` and `bbl_fname` are both
            dictionaries of file names. The keys are `'spin0xspin0`,
            `'spin0xspin2'`,  `'spin2xspin0`, `'spin2xspin2'`, where
            `'spin0'` and `'spin2'` refer to temperature and polarization
            maps, respectively.

    Other Parameters
    ----------------
    fname_info : str or None, default=None
        Any additional information to add to the file names.
    save_binning_matrix_as_npz : bool, default=True
        Whether to save the binning matrices as sparse CSR `.npz` files
        to reduce the file sizes. If `save_binning_matrix_as_npz=False`,
        they will be saved as `.txt` files.

    See Also
    --------
    hdsims.simpower.calc_mode_coupling
    hdsims.simpower.save_mode_coupling_files
    hdsims.simpower.load_mode_coupling_files
    """
    spec_type = 'dl' if bin_dl else 'cl'
    bbl_ext = 'npz' if save_binning_matrix_as_npz else 'txt'
    fname_parts = [f'apod{round_str(apod_width)}deg', f'lmax{int(lmax)}', spec_type]
    if beam_fwhm is not None:
        fname_parts.append(f'beam{round_str(beam_fwhm)}arcmin')
    if fname_info is not None:
        fname_parts.append(fname_info)
    fname_root = '_'.join(fname_parts)

    if pol:
        spin_pairs = ['spin0xspin0', 'spin0xspin2', 'spin2xspin0', 'spin2xspin2']
        mcm_fnames = {} # inv mode-coupling matrix
        bbl_fnames = {} # binning matrix
        for spin_pair in spin_pairs:
            mcm_fnames[spin_pair] = f'{fname_root}_{spin_pair}_inv_mcm.txt'
            bbl_fnames[spin_pair] = f'{fname_root}_{spin_pair}_binning_matrix.{bbl_ext}'
        if binning_dir is not None:
            mcm_fnames = {spin_pair: os.path.join(binning_dir, fname) for (spin_pair, fname) in mcm_fnames.items()}
            bbl_fnames = {spin_pair: os.path.join(binning_dir, fname) for (spin_pair, fname) in bbl_fnames.items()}
        return mcm_fnames, bbl_fnames

    else:
        mcm_fname = f'{fname_root}_spin0xspin0_inv_mcm.txt'
        bbl_fname = f'{fname_root}_spin0xspin0_binning_matrix.{bbl_ext}'
        if binning_dir is not None:
            mcm_fname = os.path.join(binning_dir, mcm_fname)
            bbl_fname = os.path.join(binning_dir, bbl_fname)
        return mcm_fname, bbl_fname



class Sims:
    """Base class for generating ultrahigh-resolution microwave sky
    simulations on a patch of the sky.
    
    Attributes for the specific patch of sky are defined here, and
    the output directories for the patch of sky are created. We also
    define default values for commonly-used keyword arguments.

    Attributes
    ----------
    ra_ctr, dec_ctr, width, height : int or float
    apod_width, map_apod_width : int or float
    res : float
    verbose : bool
    log : logging.Logger or None
    shape : tuple of int
        The shape `(Ny, Nx)` of the array of map pixels on the patch of 
        sky with area `width * height` at the given location and 
        resolution. `Ny` and `Nx` are the number of pixels along the dec. 
        and R.A. directions, respectively.
    wcs : astropy.wcs.wcs.WCS
        Specifies the astropy World Coordinate System for the pixelization
        of the patch of sky with area `width * height` at the given 
        location and resolution. A pair of `shape, wcs` values are 
        referred to as the "geometry" of a map.
    padded_width, padded_height : int or float
        The width and height (in degrees) of a slightly larger patch of
        sky, increased by the `map_apod_width` on each side.
    padded_shape, padded_wcs : tuple of int, astropy.wcs.wcs.WCS
        The geometry of the slightly larger patch of sky with a width and
        height of `padded_width` and `padded_height`, respectively. 
    padded2x_width, padded2x_height : int or float
        The width and height (in degrees) of an even larger patch of sky,
        increased by `2 * map_apod_width` on each side.
    padded2x_shape, padded2x_wcs : tuple of int, astropy.wcs.wcs.WCS
        The geometry of the even larger patch of sky with a width and
        height of `padded2x_width` and `padded2x_height`, respectively.
    default_kwargs : dict
        A dictionary of default values for commonly-used optional keyword
        arguments. It will include the `width`, `height`, `apod_width`, 
        `shape`, and `wcs` attributes.
    
    See Also
    --------
    pixell.enmap.geometry : Constructs the `shape`, `wcs` pair for a map.
    
    Notes
    -----
    Attributes listed above without a description will have the same value
    as the corresponding parameter passed when initializing the class, or
    the default value if it is not passed. 
    
    The patch of sky is defined by its area (width and height), 
    location (R.A. and dec. of its center), and resolution (pixel size).
    The simulated maps will be `pixell.enmap.ndmap` instances. 
    
    The `width` and `height` parameters/attributes define the area of the
    region of the maps used when calculating their power spectra, i.e., 
    this is the size of the "final" simulated map. The `apod_width` is the
    width of the region along each edge of the map that will be apodized
    before we take its power.
    
    When we need to apodize a map for any other reason (e.g., before
    taking it's Fourier or spherical harmonic transform), the
    `map_apod_width` is used instead of the `apod_width`. 
    
    The simulated maps are saved with a larger area (defined by 
    `padded_width` and `padded_height`) that is increased by the 
    `map_apod_width` on each side. This leaves room to apodize them if
    necessary and still leave the inner `width` x `height` region 
    unchanged. The simulations need to be apodized in the process of
    generating them, so they are initially generated on an even larger
    patch of sky (defined by `padded2x_width` and `padded2x_height`), and
    only the inner `padded_width` x `padded_height` "un-apodized" region
    is saved.
    
    The simulations have only been tested for a minimum resolution of
    0.04 arcminutes (the default value of `res`).
    """
    
    def __init__(self, hd_sims_dir, 
                 ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr, width=si.width, height=si.height,
                 apod_width=si.apod_width, map_apod_width=None, res=si.hd_res,
                 verbose=False, log=None, make_output_dirs=True):
        """Initialization for a given patch of sky.
        
        Parameters
        ----------
        hd_sims_dir : str
            The path to the directory where all of the output files will
            be saved. This directory will be created if it does not
            already exist.
        ra_ctr, dec_ctr : int or float, optional
            The right ascension (R.A.) and declination (dec.), in degrees,
            of the center of the patch of sky. The defaults are `ra_ctr=6`
            and `dec_ctr=6`.
        width : int or float, default=10
            The width (in degrees) of the region of the maps to be used
            for analysis, e.g. when taking the power spectrum of the
            simulations. 
        height : int or float, optional
            The height (in degrees) of the region of the maps to be used 
            for analysis. If the `height` is not provided, it is assumed
            to be equal to the `width`.
        apod_width : int or float, default=1
            The width (in degrees) of the region along each edge of the
            map that will be apodized before calculating its power
            spectrum.

        Other Parameters
        ----------------
        map_apod_width : int or float, optional
            The width (in degrees) of the region along each edge of the 
            map that will be apodized before taking any Fourier or
            spherical harmonic transforms. By default, `apod_width/2` will
            be used.
        res : int or float, default=0.04
            The resolution of the maps, in arcminutes. 
        verbose : bool, default=False
            Whether to print messages describing the progress of some
            calculations.
        log : logging.Logger, optional
            A `logging.Logger` instance to use when `verbose=True`. If a
            `log` is passed, any messages will be passed to `log.info`.
            Otherwise, messages will be passed to the `print` function.
        make_output_dirs : bool, default=True
            Whether to create the sub-directories under the `hd_sims_dir`
            where the output files will be saved. This should not be 
            changed, but it is provided to, e.g., allow you to check where
            the files will be saved before generating the simulations.

        Notes
        -----
        By default (if `make_output_dirs=True`), a sub-directory for this
        patch of sky will be created in the `hd_sims_dir`. The name of 
        this sub-directory depends on the `ra_ctr`, `dec_ctr`, `width`,
        `height`, and `res` parameters.
        """
        self.verbose = verbose
        self.log = log
        
        # define attributes for the patch of sky:
        self.res = res
        self.ra_ctr = ra_ctr
        self.dec_ctr = dec_ctr
        self.width = width
        self.height = width if (height is None) else height
        self.apod_width = apod_width
        self.map_apod_width = apod_width / 2 if (map_apod_width is None) else map_apod_width
        
        # dict of directories where output will be saved:
        self._output_dirs = get_output_dirs(hd_sims_dir, ra_ctr=self.ra_ctr, dec_ctr=self.dec_ctr, 
                                            width=self.width, height=self.height, res=self.res, 
                                            make_dirs=make_output_dirs)

        # the `width` and `height` define the patch of sky that will
        # ultimately be used for analysis (taking power spectra, etc.):
        self.shape, self.wcs = maps.get_shape_wcs(self.res, self.ra_ctr, self.dec_ctr, self.width, height=self.height)
        # the sims will be saved on a larger patch of sky, which is
        # extended by the `map_apod_width` on each side; this leaves room
        # to apodize them (if necessary) without affecting the inner
        # `width` x `height` region:
        self.padded_width = self.width + 2 * self.map_apod_width
        self.padded_height = self.height + 2 * self.map_apod_width
        self.padded_shape, self.padded_wcs = maps.get_shape_wcs(self.res, self.ra_ctr, self.dec_ctr, 
                                                                self.padded_width, height=self.padded_height)
        # we need to initially generate the sims on an even larger patch
        # of sky so the maps can be apodized during intermediate steps:
        self.padded2x_width = self.padded_width + 2 * self.map_apod_width
        self.padded2x_height = self.padded_height + 2 * self.map_apod_width
        self.padded2x_shape, self.padded2x_wcs = maps.get_shape_wcs(self.res, self.ra_ctr, self.dec_ctr, 
                                                                    self.padded2x_width, height=self.padded2x_height)
        
        # when `verbose=True`, we print out the dimensions (width and height)
        # of the maps a few times, so we define those strings here: 
        self._map_area_info = f'{round(self.width,2)} deg. x {round(self.height,2)} deg.'
        self._padded_map_area_info = f'{round(self.padded_width,2)} deg. x {round(self.padded_height,2)} deg.'
        self._padded2x_map_area_info = f'{round(self.padded2x_width,2)} deg. x {round(self.padded2x_height,2)} deg.'

        self.default_kwargs = {'width': self.width, 'height': self.height,
                               'shape': self.shape, 'wcs': self.wcs, 
                               'apod_width': self.apod_width}
    
    
    def infomsg(self, msg):
        """Print out a message.
        
        Parameters
        ----------
        msg : str
            The message to print.
            
        Notes
        -----
        This will only print out the message if `verbose=True` was passed
        during initialization. 
        """
        if self.verbose:
            if self.log is not None:
                self.log.info(msg, stacklevel=2)
            else:
                print(msg)


    def get_kwargs_with_defaults(self, defaults={}, **kwargs):
        """Return the dictionary of keyword arguments after adding
        default values for any missing keys.
        
        The default values are determined by the `default_kwargs`
        attribute that is defined during initialization. This method is
        used to ensure that the `kwargs` dictionary contains a key for all
        keyword arguments in `default_kwargs`.
        
        Parameters
        ----------
        defaults : dict, optional
            A dictionary of keyword argument names and their default
            values. This may be used to define a different default value
            than the one in the `default_kwargs` attribute, or to provide
            a default value for keyword arguments that are not in 
            `default_kwargs`.
        **kwargs : dict
            The dictionary of keyword arguments.
            
        Returns
        -------
        kwargs : dict
            The dictionary of keyword arguments that has been updated with
            default values for any missing keys.
            
        Raises
        ------
        ValueError
            If a `shape` was passed but no `wcs` was passed.
            
        Notes
        -----
        If the `shape` and `wcs` are passed, the corresponding `width` and
        `height` are calculated (for the map resolution `res` defined 
        during initialization) and added to the `kwargs` dictionary; any
        value of the `width` or `height` that was also passed along with
        `shape` and `wcs` will be ignored. If only the `width` and/or
        `height` are passed, the corresponding `shape` and `wcs` will be
        calculated and added to `kwargs`.
        """
        default_kwargs = {**self.default_kwargs, **defaults}
        # if the `shape` and `wcs` were passed, use them to set the 
        # correct `width` and `height`:
        if 'shape' in kwargs:
            if 'wcs' not in kwargs: # need both `shape` and `wcs`
                raise ValueError(f"`shape = {kwargs['shape']}`, but the `wcs` was not passed")
            _, _, width, height = maps.get_map_ctr_extent(kwargs['shape'], kwargs['wcs'])
            kwargs['width'] = round(width, 2)
            kwargs['height'] = round(height, 2)
        # if the `width` and/or `height` were passed but the `shape` and
        # `wcs` weren't, set the correct `shape` and `wcs` values:
        elif ('width' in kwargs) or ('height' in kwargs):
            width = default_kwargs['width'] if ('width' not in kwargs) else kwargs['width']
            height = default_kwargs['height'] if ('height' not in kwargs) else kwargs['height']
            kwargs['shape'], kwargs['wcs'] = maps.get_shape_wcs(self.res, self.ra_ctr, self.dec_ctr, width, height)
        # set the other values:
        for key, val in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val
        return kwargs
    
    
    def get_kwarg(self, key, default='none', **kwargs):
        """Return the value of a key word argument if it was passed, or
        the default value if it wasn't.
        
        Parameters
        ----------
        key : str
            The name of the keyword argument.
        default : optional
            The default value to use. If no `default` is provided and the
            `key` is in the dictionary of `default_kwargs`, the value in 
            `default_kwargs` is used. 
        **kwargs : dict
            The dictionary of keyword arguments.
            
        Raises
        ------
        KeyError
            If a `default` value wasn't provided and the `key` is not in
            the `default_kwargs` dictionary.
        """
        defaults = {}
        if default != 'none':
            defaults[key] = default
        return self.get_kwargs_with_defaults(defaults=defaults, **kwargs)[key]
    
    
    def sim_dir(self):
        """Return the path to the sub-directory where the simulated maps,
        catalogs, and theory curves for this patch of sky will be saved.
        """
        return self._output_dirs['sims']


    def spectra_dir(self):
        """Return the path to the sub-directory where the power spectra of
        the simulations will be saved.
        """
        return self._output_dirs['spectra']


    def binning_dir(self):
        """Return the path to the sub-directory where binning files and
        inverse mode-coupling matrices will be saved."""
        return self._output_dirs['binning']


    def intermediate_maps_dir(self):
        """Return the path to the sub-directory where any of the
        intermediate, lower-resolution output will be saved.
        """
        return self._output_dirs['intermediate']


    def plots_dir(self):
        """Return the path to the sub-directory where plots will be saved."""
        return self._output_dirs['plots']
        
    
    def _patch_info_for_filenames(self, **kwargs):
        """Return a string describing a patch of sky that is different
        from the patch of sky defined during initialization.
        
        The string is used in filenames for maps and catalogs that are
        saved for a larger patch of sky than the region defined by the
        `width` and `height` attributes set during initialization.
        
        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments used to specify the width and 
            height of a patch of sky. By default, the `width` and `height`
            attributes defined during initialization are used (so the 
            returned `patch_info` will be `None`). 
            
            Otherwise, the options are:
            (1) The `width` and `height` (`int` or `float`) of the patch
                of sky (centered at R.A. and dec. given by the `ra_ctr`
                and `dec_ctr` attributes); or
            (2) A `shape` (`tuple` of `int`) and `wcs` (instance of 
                `astropy.wcs.wcs.WCS`) pair to define the geometry of the
                patch of sky, which will be used to calculate its `width`
                and `height`.
        
        Returns
        -------
        patch_info : str or None
            A string describing the width and height of the patch of sky,
            or `None` if the width and height of the patch are the same as
            the `width` and `height` attributes defined during 
            initialization.
        """
        patch_info_list = []
        if ('shape' in kwargs) and ('wcs' in kwargs): 
            # compare the `shape` and `wcs` to our patch:
            if not maps.map_geometry_is_equal(kwargs['shape'], kwargs['wcs'], self.shape, self.wcs):
                ra_ctr, dec_ctr, width, height = maps.get_map_ctr_extent(kwargs['shape'], kwargs['wcs'])
                # compare center R.A. and dec with our patch (rounded 
                # to 2 digits, because that's what's used in filenames):
                ra_ctr = round(ra_ctr, 2) 
                dec_ctr = round(dec_ctr, 2)
                if not (np.isclose(ra_ctr, self.ra_ctr) and (dec_ctr, self.dec_ctr)):
                    patch_info_list.append(f'ra{round_str(ra_ctr)}dec{round_str(dec_ctr)}')
                # compare (rounded) width and height to our patch:
                width = round(width, 2)
                height = round(height, 2)
                if not (np.isclose(width, self.width) and np.isclose(height, self.height)):
                    patch_info_list.append(f'{round_str(width)}x{round_str(height)}deg')
        elif ('width' in kwargs) and ('height' in kwargs):
            # compare (rounded) width and height to our patch:
            width = round(kwargs['width'], 2)
            height = round(kwargs['height'], 2)
            if not (np.isclose(width, self.width) and np.isclose(height, self.height)):
                patch_info_list.append(f'{round_str(width)}x{round_str(height)}deg')
        patch_info = '_'.join(patch_info_list) if (len(patch_info_list) > 0) else None
        return patch_info
    
    


