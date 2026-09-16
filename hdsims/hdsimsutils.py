"""Contains functions used for the examples provided with `hdsims`."""

import os
import numpy as np
from pixell import enmap
from . import utils, simutils, siminfo as si, hdsims, plots


def precomputed_hdsims_output_dir():
    """Return the path to the precomputed `hdsims` output files."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'precomputed_hdsims_output')


def _get_hd_sims_dir(simlib):
    """Return the path to the main `hd_sims_dir` passed when initializing
    the `hdsims.HDSims` class.

    Parameters
    ----------
    simlib : hdsims.HDSims
        The instance of the `hdsims.HDSims` class.

    Returns
    -------
    hd_sims_dir : str
        Path to the `hd_sims_dir`.
    """
    hd_sims_dir, _ = os.path.split(simlib.sim_dir())
    return hd_sims_dir


def _github_url(fname, simlib):
    """Return the Github URL of the file corresponding to the file with
    absolute path `fname`.

    Parameters
    ----------
    fname : str
        The absolute path to the file. This should be a file name that is
        returned by a method of the `hdsims.HDSims` class.
    simlib : hdsims.HDSims
        An instance of the `hdsims.HDSims` class.

    Returns
    -------
    str
        The URL to the corresponding pre-computed file on Github.
    """
    url_root = 'https://raw.githubusercontent.com/CMB-HD/sim_files_for_example_notebooks/refs/heads/main'
    hd_sims_dir = _get_hd_sims_dir(simlib)
    return fname.replace(hd_sims_dir, url_root)


def _github_wget_command(fname, simlib):
    """Return a string with the `wget` command used to download
    pre-computed files used for the `hdsims` example.

    Parameters
    ----------
    fname : str
        The absolute path to the file. This should be a file name that is
        returned by a method of the `hdsims.HDSims` class.
    simlib : hdsims.HDSims
        An instance of the `hdsims.HDSims` class.

    Returns
    -------
    str
        The command to download the corresponding pre-computed file from
        Github.
    """
    return f'wget -c {_github_url(fname, simlib)} -O {fname}'


def _check_compatibility_with_precomputed_2x2_files(simlib, inv_mcm=False, sim_maps=False, spectra=False):
    """Verify that the settings used to initialize `simlib` (instance of
    `hdsims.HDSims`) are compatible with the pre-computed files available
    for the hdsims example.
    
    Raises
    ------
    ValueError
        If the settings for the `simlib` differ from those used to
        generate the pre-computed example files.
    """
    example2x2_hdsims_kwargs = {'res': si.hd_res, 'width': 2, 'height': 2, 'ra_ctr': si.ra_ctr, 'dec_ctr': si.dec_ctr, 
                                'apod_width': 0.2, 'map_apod_width': 0.2,  'cmb_seed': si.cmb_seed, 
                                'cib_model': si.baseline_cib_model_name, 'lmax': si.lmax4spectra, 'bin_info': None}
    inv_mcm_keys = ['res', 'width', 'height', 'apod_width', 'dec_ctr', 'lmax', 'bin_info']
    map_keys = ['res', 'width', 'height', 'ra_ctr', 'dec_ctr', 'apod_width', 'map_apod_width', 'cmb_seed', 'cib_model']
    # keep track of which keys correspond to floating point numerical values:
    numerical_keys = ['res', 'width', 'height', 'ra_ctr', 'dec_ctr', 'apod_width', 'map_apod_width']
    
    # list of keyword arguments to compare:
    if spectra or (inv_mcm and sim_maps):
        keys = list(example2x2_hdsims_kwargs.keys())
    elif inv_mcm:
        keys = inv_mcm_keys
    elif sim_maps:
        keys = map_keys
    else:
        keys = []
        
    # keep track of which settings are inconsistent:
    inconsistent_keys = {}
    simlib_kwargs = vars(simlib)
    for key in keys:
        value = simlib_kwargs[key]
        default_value = example2x2_hdsims_kwargs[key]
        if key in numerical_keys:
            if key == 'res':
                values_match = np.isclose(value, default_value)
            else:
                values_match = np.isclose(value, default_value, rtol=0, atol=utils.arcmin2deg(si.hd_res)/2)
        else:
            values_match = (value == default_value)
        if not values_match:
            inconsistent_keys[key] = {'value': value, 'default': default_value}
            
    if len(inconsistent_keys) > 0:
        errmsg_info = [f"`HDSims` was initialized with the following keyword arguments and values:"]
        for key, value_info in inconsistent_keys.items():
            errmsg_info.append(f"  {key} = {value_info['value']}")
        errmsg_info.append(f"Precomputed files are only available for:")
        for key, value_info in inconsistent_keys.items():
            errmsg_info.append(f"  {key} = {value_info['default']}")
        errmsg_info.append("To use the precomputed example files, you must initialize `HDSims` with"
                           " the default values listed above.")
        errmsg = '\n'.join(errmsg_info)
        raise ValueError(errmsg)


def _commands_to_download_inv_mcm(simlib):
    """Return a list of commands used to download the pre-computed
    inverse mode-coupling matrices and binning matrices used for the
    `hdsims` example from Github.
    
    `simlib` is the instance of `hdsims.HDSims` used for the example.
    """
    _check_compatibility_with_precomputed_2x2_files(simlib, inv_mcm=True)
    cmd_list = []
    for bin_dl in [False, True]:
        spec_type = 'dl' if bin_dl else 'cl'
        # inverse mode-coupling for high-resolution maps without any beam convolved:
        mcm_fname, bbl_fname = simlib.get_mode_coupling_fnames(bin_dl=bin_dl)
        if simlib.pol:
            mcm_saved = all([os.path.exists(mcm_fname[key]) for key in mcm_fname])
            bbl_saved = all([os.path.exists(bbl_fname[key]) for key in bbl_fname])
        else:
            mcm_saved = os.path.exists(mcm_fname)
            bbl_saved = os.path.exists(bbl_fname)
        if not (mcm_saved and bbl_saved):
            fname = os.path.join(simlib.binning_dir(), f'bin_{spec_type}s_nobeam.tar.gz')
            cmd_list.append(_github_wget_command(fname, simlib))
            cmd_list.append(f'tar -xzf {fname} -C {simlib.binning_dir()}')
            cmd_list.append(f'rm {fname}')
        # inverse mode-coupling for high-resolution 90 or 148 GHz beam-convolved maps:
        mcm_freqs = [freq for freq in simlib.freqs if (freq in [90, 148])]
        for freq in mcm_freqs:
            mcm_fname, bbl_fname = simlib.get_mode_coupling_fnames(bin_dl=bin_dl, beam=True, freq=freq)
            if simlib.pol:
                mcm_saved = all([os.path.exists(mcm_fname[key]) for key in mcm_fname])
                bbl_saved = all([os.path.exists(bbl_fname[key]) for key in bbl_fname])
            else:
                mcm_saved = os.path.exists(mcm_fname)
                bbl_saved = os.path.exists(bbl_fname)
            if not (mcm_saved and bbl_saved):
                fname = os.path.join(simlib.binning_dir(), f'bin_{spec_type}s_beam{simutils.round_str(si.beam_fwhm[freq])}arcmin.tar.gz')
                cmd_list.append(_github_wget_command(fname, simlib))
                cmd_list.append(f'tar -xzf {fname} -C {simlib.binning_dir()}')
                cmd_list.append(f'rm {fname}')
    if len(cmd_list) > 0:
        cmd_list = ['\n', '# download inverse mode-coupling matrices for high-resolution maps: ', *cmd_list]
    return cmd_list


def _commands_to_download_lowres_inv_mcm(simlib):
    """Return a list of commands used to download the pre-computed
    inverse mode-coupling matrices and binning matrices used for the
    lower-resolution maps in the `hdsims` example from Github.
    
    `simlib` is the instance of `hdsims.HDSims` used for the example.
    """
    _check_compatibility_with_precomputed_2x2_files(simlib, inv_mcm=True)
    cmd_list = []
    for bin_dl in [False, True]:
        # for S10-resolution temperature-only maps:
        mcm_fname, bbl_fname = simlib.get_intermediate_mode_coupling_fnames('tsz', bin_dl=bin_dl)
        if not os.path.exists(mcm_fname):
            url = _github_url(mcm_fname, simlib)
            cmd_list.append(f'wget -O {mcm_fname} -c {url}')
        if not bin_dl:
            # for S10-resolution kappa map:
            mcm_fname, bbl_fname = simlib.get_intermediate_mode_coupling_fnames('kappa', bin_dl=False)
            if not os.path.exists(mcm_fname):
                cmd_list.append(_github_wget_command(mcm_fname, simlib))
    if len(cmd_list) > 0:
        cmd_list = ['\n', '# download inverse mode-coupling matrices for lower-resolution maps: ', *cmd_list]
    return cmd_list


def _commands_to_download_lowres_maps(simlib):
    """Return a list of commands used to download the pre-computed
    lower-resolution CAR maps and corresponding catalogs that are used to
    generate the ultrahigh-resolution simulations in the `hdsims` example
    from Github.
    
    `simlib` is the instance of `hdsims.HDSims` used for the example.
    """
    _check_compatibility_with_precomputed_2x2_files(simlib, sim_maps=True)
    cmd_list = []
    # S10-resolution CAR maps:
    s10_components = [c for c in simlib.components if (c in si.s10_sim_components)]
    for component in s10_components:
        sim_freqs = simlib.freqs if simutils.has_freq_dependent_component(component) else [None]
        for freq in sim_freqs:
            fname = simlib.get_intermediate_sim_fname(component, freq=freq)
            if not os.path.exists(fname):
                cmd_list.append(_github_wget_command(fname, simlib))
    # catalogs for the simulated patch of sky:
    catalog_components = [c for c in simlib.components if (c in ['cib', 'radio'])] # we don't need the SZ catalog to generate the sims
    for component in catalog_components:
        fname = simlib.get_sim_catalog_fname(component, padded_ntimes=2)
        if not os.path.exists(fname):
            cmd_list.append(_github_wget_command(fname, simlib))
    if len(cmd_list) > 0:
        cmd_list = ['\n', '# download lower-resolution S10 CAR maps and catalogs for the patch of sky: ', *cmd_list]
    return cmd_list


def _commands_to_download_lensed_cmb(simlib):
    """Return a list of commands used to download the pre-computed
    ultrahigh-resolution lensed CMB maps and the corresponding theory 
    curves that are used  in the `hdsims` example from Github.
    
    `simlib` is the instance of `hdsims.HDSims` used for the example.
    """
    _check_compatibility_with_precomputed_2x2_files(simlib, sim_maps=True)
    cmd_list = []
    # lensed CMB sim:
    cmb_sim_fname = simlib.get_signal_sim_fname('cmb')
    if not os.path.exists(cmb_sim_fname):
        if simlib.pol: # download T, Q, and U maps separately
            tmap_info = simlib._map_component_list2str(components=['cmb'], pol=False)
            qmap_info = tmap_info.replace('T', 'Q')
            umap_info = tmap_info.replace('T', 'U')
            tmap_fname = simlib.get_signal_sim_fname('cmb', pol=False)
            qmap_fname = tmap_fname.replace(tmap_info, qmap_info)
            umap_fname = tmap_fname.replace(tmap_info, umap_info)
            for fname in [tmap_fname, qmap_fname, umap_fname]:
                if not os.path.exists(fname):
                    cmd_list.append(_github_wget_command(fname, simlib))
        else: # only need temperature map
            cmd_list.append(_github_wget_command(cmb_sim_fname, simlib))
    # lensed CMB sim theory power spectra:
    cmb_sim_theo_fname = simlib.get_sim_theory_fname('cmb')
    if not os.path.exists(cmb_sim_theo_fname):
        cmd_list.append(_github_wget_command(cmb_sim_theo_fname, simlib))
    if len(cmd_list) > 0:
        cmd_list = ['\n', '# download lensed CMB map(s) and corresponding theory power spectra : ', *cmd_list]
    return cmd_list


def _commands_to_download_all(simlib):
    """Return a list of commands used to download all pre-computed
    files that are used in the `hdsims` example from Github.
    
    `simlib` is the instance of `hdsims.HDSims` used for the example.
    """
    hd_sims_dir, sims_dir_name = os.path.split(simlib.sim_dir())
    git_repo_name = 'sim_files_for_example_notebooks'
    git_repo_url = f'https://github.com/CMB-HD/{git_repo_name}.git'
    repo_path = os.path.join(hd_sims_dir, git_repo_name)
    cmds = [f'git clone {git_repo_url} {hd_sims_dir}',
            f'mv {os.path.join(repo_path, sims_dir_name)} {simlib.sim_dir()}',
            f'rm -rf {repo_path}']
    return cmds


def save_commands_to_download_2x2example_files(simlib, output_dir=None, intermediate_maps=True, inv_mcm=True, lensed_cmb=True, download_all=False):
    """Return the name of the bash file containing the necessary commands
    to download pre-computed files that are used in the `hdsims` example 
    from Github. If there are no commands to run, returns `None` instead.
    
    `simlib` is the instance of `hdsims.HDSims` used for the example.
    """
    if download_all:
        cmd_list = _commands_to_download_all(simlib)
    else:
        cmd_list = []
        if intermediate_maps:
            cmd_list = [*_commands_to_download_lowres_maps(simlib)]
        if inv_mcm:
            cmd_list = [*cmd_list, *_commands_to_download_lowres_inv_mcm(simlib), *_commands_to_download_inv_mcm(simlib)]
        if lensed_cmb:
            cmd_list = [*cmd_list, *_commands_to_download_lensed_cmb(simlib)]
    commands = '\n'.join(cmd_list)
    if len(commands) > 0:
        output_dir = _get_hd_sims_dir(simlib) if (output_dir is None) else output_dir
        fname = os.path.join(output_dir, 'download_hdsims_2x2example_files.sh')
        with open(fname, 'w') as f:
            f.write(commands)
    else:
        fname = None
    return fname


def check_if_example_files_saved(simlib, bash_files_dir=None, download_all=False,
                                 use_precomputed_intermediate_sims=True, 
                                 use_precomputed_inv_mcm=True, use_precomputed_lensed_cmb=True):
    """Check if all files needed to run the `hdsims` example are saved."""
    _check_compatibility_with_precomputed_2x2_files(simlib, inv_mcm=use_precomputed_inv_mcm, 
                                                    sim_maps=(use_precomputed_intermediate_sims or use_precomputed_lensed_cmb),
                                                    spectra=use_precomputed_lensed_cmb)
    
    # initially, assume everything is saved
    intermediate_sim_files_saved = True
    inv_mcm_saved = True
    lensed_cmb_saved = True
    # keep track of info about any files that aren't saved
    intermediate_fnames = {'maps': {}, 'catalogs': {}}
    mcm_info = {'hd': {}, 's10': {}}
    cmb_info = {}
    
    # look for intermediate maps & catalogs that were generated from full-sky S10 sims:
    s10_components = [c for c in simlib.components if (c in si.s10_sim_components)]
    for component in s10_components:
        sim_freqs = simlib.freqs if simutils.has_freq_dependent_component(component) else [None]
        for freq in sim_freqs:
            fname = simlib.get_intermediate_sim_fname(component, freq=freq)
            if not os.path.exists(fname):
                intermediate_sim_files_saved = False
                if component not in intermediate_fnames['maps']:
                    intermediate_fnames['maps'][component] = {}
                intermediate_fnames['maps'][component] = {freq: fname}
    catalog_components = [c for c in simlib.components if (c in ['cib', 'radio'])] # we don't need the SZ catalog to generate the sims
    for component in catalog_components:
        fname = simlib.get_sim_catalog_fname(component, padded_ntimes=2)
        if not os.path.exists(fname):
            intermediate_sim_files_saved = False
            intermediate_fnames['catalogs'][component] = fname
            
    # look for inverse mode-coupling matrix & binning matrix files:
    for bin_dl in [False, True]:
        spec_type = 'dl' if bin_dl else 'cl'
        # for HD TQU maps:
        mcm_fname, bbl_fname = simlib.get_mode_coupling_fnames(bin_dl=bin_dl)
        if not simlib.pol: # make sure `mcm_fname` and `bbl_fname` are dictionaries
            mcm_fname = {'spin0xspin0': mcm_fname}
            bbl_fname = {'spin0xspin0': bbl_fname}
        for key in mcm_fname.keys():
            if not os.path.exists(mcm_fname[key]):
                inv_mcm_saved = False
                if spec_type not in mcm_info['hd']:
                    mcm_info['hd'][spec_type] = {}
                if key not in mcm_info['hd'][spec_type]:
                    mcm_info['hd'][spec_type][key] = {}
                mcm_info['hd'][spec_type][key]['mcm'] = mcm_fname[key]
            if not os.path.exists(bbl_fname[key]):
                inv_mcm_saved = False
                if spec_type not in mcm_info['hd']:
                    mcm_info['hd'][spec_type] = {}
                if key not in mcm_info['hd'][spec_type]:
                    mcm_info['hd'][spec_type][key] = {}
                mcm_info['hd'][spec_type][key]['bbl'] = bbl_fname[key]
        # for S10-res T-only maps (only need mcm):
        mcm_fname, bbl_fname = simlib.get_intermediate_mode_coupling_fnames('tsz', bin_dl=bin_dl)
        if not os.path.exists(mcm_fname):
            inv_mcm_saved = False
            mcm_info['s10'][spec_type] = mcm_fname
    # for S10-res kappa map:
    mcm_fname, bbl_fname = simlib.get_intermediate_mode_coupling_fnames('kappa', bin_dl=False)
    if not os.path.exists(mcm_fname):
        inv_mcm_saved = False
        mcm_info['s10_kappa'] = mcm_fname
        
    # lensed cmb sim & theory:
    cmb_sim_fname = simlib.get_signal_sim_fname('cmb')
    if not os.path.exists(cmb_sim_fname):
        if simlib.pol: # may have downloaded T, Q, and U maps separately
            tmap_info = simlib._map_component_list2str(components=['cmb'], pol=False)
            qmap_info = tmap_info.replace('T', 'Q')
            umap_info = tmap_info.replace('T', 'U')
            tmap_fname = simlib.get_signal_sim_fname('cmb', pol=False)
            qmap_fname = tmap_fname.replace(tmap_info, qmap_info)
            umap_fname = tmap_fname.replace(tmap_info, umap_info)
            if all([os.path.exists(fname) for fname in [tmap_fname, qmap_fname, umap_fname]]):
                # combine TQU into a single file (the way `HDSims` expects)
                cmb_sim = enmap.zeros((3, *simlib.padded_shape), simlib.padded_wcs)
                cmb_sim[0] = enmap.read_map(tmap_fname)
                cmb_sim[1] = enmap.read_map(qmap_fname)
                cmb_sim[2] = enmap.read_map(umap_fname)
                enmap.write_map(cmb_sim_fname, cmb_sim)
            else:
                lensed_cmb_saved = False
                cmb_info['sim'] = cmb_sim_fname
        else: 
            lensed_cmb_saved = False
            cmb_info['sim'] = cmb_sim_fname
    cmb_sim_theo_fname = simlib.get_sim_theory_fname('cmb')
    if not os.path.exists(cmb_sim_theo_fname):
        lensed_cmb_saved = False
        cmb_info['theo'] = cmb_sim_theo_fname

    # tell the user what must still be done (if anything):
    all_files_saved = all([intermediate_sim_files_saved, inv_mcm_saved, lensed_cmb_saved])

    if all_files_saved:
        print("You may proceed and run the rest of the example notebook: all files needed to generate the HD sims are saved")
    
    else: # print out info about what is missing:
        print("The following files were not found and must be saved before proceeding:")

        if len(intermediate_fnames['maps']) > 0:
            print("  Sims at the S10 resolution, cut out from the full-sky maps:")
            for component in intermediate_fnames['maps'].keys():
                for freq, fname in intermediate_fnames['maps'][component].items():
                    map_info = component if (freq is None) else f"{freq} GHz {component}"
                    print(f"    {map_info} : {fname}")

        if len(intermediate_fnames['catalogs']) > 0:
            print("  Catalogs for our patch of sky, generated from the full-sky S10 catalogs:")
            for component, fname in intermediate_fnames['catalogs'].items():
                print(f"    {component} : {fname}")

        if len(mcm_info['hd']) > 0:
            print("  Inverse mode-coupling and binning matrices for HD sims:")
            spin_info = {'spin0xspin0': 'temperature', 'spin0xspin2': 'temperature x polarization', 'spin2xspin0': 'polarization x temperature', 'spin2xspin2': 'polarization'}
            for spec_type in mcm_info['hd'].keys():
                for key in mcm_info['hd'][spec_type].keys():
                    print(f"    Files for {spin_info[key]} spectra (binned as {spec_type.capitalize()}'s):")
                    if 'mcm' in mcm_info['hd'][spec_type][key]:
                        print(f"      inverse mode-coupling matrix: {mcm_info['hd'][spec_type][key]['mcm']}")
                    if 'bbl' in mcm_info['hd'][spec_type][key]:
                        print(f"      binning matrix: {mcm_info['hd'][spec_type][key]['bbl']}")

        if (len(mcm_info['s10']) > 0) or ('s10_kappa' in mcm_info):
            print("  Inverse mode-coupling matrices for the intermediate, S10-resolution sims:")
            for spec_type in mcm_info['s10'].keys():
                print(f"    inverse mode-coulping matrix (for spectra binned as {spec_type.capitalize()}'s): {mcm_info['s10'][spec_type]}")
            if 's10_kappa' in mcm_info:
                print(f"    inverse mode-coulping matrix for S10-resolution lensing convergence sim: {mcm_info['s10_kappa']}")

        if not lensed_cmb_saved:
            if 'sim' in cmb_info:
                print(f"  The HD lensed CMB sim: {cmb_info['sim']}")
            if 'theo' in cmb_info:
                print(f"  The lensed CMB sim theory curves: {cmb_info['theo']}")

        print('\n')
        print_example_instructions(simlib, bash_files_dir=bash_files_dir, download_all=download_all,
                                   use_precomputed_intermediate_sims=use_precomputed_intermediate_sims,
                                   use_precomputed_inv_mcm=use_precomputed_inv_mcm,
                                   use_precomputed_lensed_cmb=use_precomputed_lensed_cmb)
        print("Then, re-run this cell to verify that everything has been saved.")
        
    return all_files_saved


def print_example_instructions(simlib, bash_files_dir=None, download_all=False,
                               generate_all_sims_and_calculate_spectra=False, 
                               use_precomputed_intermediate_sims=True,
                               use_precomputed_inv_mcm=True,
                               use_precomputed_lensed_cmb=True):
    """Print instructions to save the files needed to run the `hdsims` 
    example.
    """
    if (not use_precomputed_intermediate_sims) and (simlib.lowres_sims_dir in [None, '']):
        raise ValueError(f"`{use_precomputed_intermediate_sims = }` and `{simlib.lowres_sims_dir = }`."
                          " You must provide the path to the `lowres_sims_dir` when initializing `HDSims` in order to"
                          " use the full-sky S10 sims. Otherwise, pass `use_precomputed_intermediate_sims=True`"
                          " to instead use the pre-computed files that are provided.")
    # generate a single command and/or list of commands for each step that needs to be run:
    hd_sims_dir = _get_hd_sims_dir(simlib)
    freqs_list = ' '.join([str(int(freq)) for freq in simlib.freqs])
    args_list = [hd_sims_dir, f'--freqs {freqs_list}']
    cmd_list = []
    wget_cmd_list = []
    
    # instructions to download HD sims data:
    bash_fname = save_commands_to_download_2x2example_files(simlib, output_dir=bash_files_dir, download_all=download_all, 
                                                            intermediate_maps=use_precomputed_intermediate_sims, 
                                                            inv_mcm=use_precomputed_inv_mcm, lensed_cmb=use_precomputed_lensed_cmb)
    if bash_fname is not None:
        wget_cmd_list.append(f'bash {bash_fname}')
    # instructions to do all calculations outside of the example notebook:
    if generate_all_sims_and_calculate_spectra:
        if not use_precomputed_intermediate_sims:
            wget_cmd_list.append(f'bash download_90GHz_S10sims_data.sh {simlib.lowres_sims_dir}')
            args_list.append(f'--lowres_sims_dir {simlib.lowres_sims_dir}')
        args_list.append('--all')
        cmd_list.append(f"python prepare_2x2_example_files.py {hd_sims_dir} {' '.join(args_list)}")
    else: # instructions to only do some calculations outside of the example notebook:
        if not use_precomputed_intermediate_sims:
            wget_cmd_list.append(f'bash download_90GHz_S10sims_data.sh {simlib.lowres_sims_dir}')
            args_list.append(f'--lowres_sims_dir {simlib.lowres_sims_dir}')
            args_list.append('--intermediate_maps')
            cmd_list.append(f'python prepare_2x2_example_files.py {hd_sims_dir} --freqs {freqs_list} --lowres_sims_dir {simlib.lowres_sims_dir} --intermediate_maps')
        if not use_precomputed_inv_mcm:
            args_list.append('--inv_mcm')
            cmd_list.append(f'python prepare_2x2_example_files.py {hd_sims_dir} --freqs {freqs_list} --inv_mcm')
        if not use_precomputed_lensed_cmb:
            args_list.append('--cmb')
            cmd_list.append(f'python prepare_2x2_example_files.py {hd_sims_dir} --freqs {freqs_list} --cmb')
    
    example_cmd_args = ' '.join(args_list)
    example_cmd = f'python prepare_2x2_example_files.py {example_cmd_args}'
    if (len(cmd_list) > 0) or (len(wget_cmd_list) > 0):
        print(f'Before running the rest of this notebook, you must run the following command(s):\n')
        if len(wget_cmd_list) > 0:
            for cmd in wget_cmd_list:
                print('  ', cmd)
        if len(cmd_list) > 0:
            print('  ', example_cmd)
            if len(cmd_list) > 1:
                print('\n If you would prefer to run each step individually, you can replace the single python call shown above with the following:\n')
                for cmd in cmd_list:
                    print('  ', cmd)
        print('\n')


def compare_example_spectra(simlib, fdiff_tol=0.01):
    """Compare the power spectra calculated in the `hdsims` example with
    the corresponding precomputed power spectra.

    Check if the (absolute value of) the fractional difference (in percent)
    between the two sets of spectra is less than the `fdiff_tol`.
    """
    _check_compatibility_with_precomputed_2x2_files(simlib, spectra=True)
    components = simlib.components
    if ('cmb' in components) and ('kappa' not in components):
        components.append('kappa')
    precomputed_example_spectra_dir = simlib.spectra_dir().replace(_get_hd_sims_dir(simlib), precomputed_hdsims_output_dir())

    sim_spectra = {}
    precomputed_spectra = {}
    unmatched_info = [] # keep track of any components & freqs that don't match
    for component in components:
        print(f'{component = }:')
        sim_spectra[component] = {}
        precomputed_spectra[component] = {}

        sim_freqs = simlib.freqs if simutils.has_freq_dependent_component(component) else [None]
        bin_dl = False if (component in ['cmb', 'kappa']) else True
        for freq in sim_freqs:
            # load sim spectra
            sim_power = simlib.get_signal_sim_power(component, freq=freq, bin_dl=bin_dl)
            sim_spectra[component][freq] = {}  # use shorter names for keys
            for key in sim_power.keys():
                spec_key = key[2:] if ('ell' not in key) else key
                sim_spectra[component][freq][spec_key] = sim_power[key].copy()

            # load precomputed spectra
            spectra_fname = simlib.get_signal_sim_power_fname(component, freq=freq, bin_dl=bin_dl, pol=True)
            spectra_cols = simutils.get_spectra_keys(component, pol=True)
            precomputed_spectra_fname = spectra_fname.replace(simlib.spectra_dir(), precomputed_example_spectra_dir)
            precomputed_spectra[component][freq] = utils.load_dict_from_file(precomputed_spectra_fname, spectra_cols)

            # compare them
            if 'cmb' not in component:
                key = 'kk' if (component == 'kappa') else 'tt'
                freq_info = '' if (freq is None) else f'{freq:3d} GHz '
                fdiff = utils.get_fdiff(sim_spectra[component][freq][key], precomputed_spectra[component][freq][key])
                min_fdiff = np.min(fdiff)
                max_fdiff = np.max(fdiff)
                avg_fdiff = np.mean(fdiff)
                if abs(avg_fdiff) <= fdiff_tol:
                    print(f'  Success! Your {freq_info}power spectrum matches the precomputed spectrum'
                          f' (average fractional difference is {avg_fdiff:5.2f} %)')
                else:
                    unmatched_info.append([component, freq])
                    print(f'  Your {freq_info}power spectrum does not match the precomputed spectrum'
                          f' (average fractional difference is {avg_fdiff:5.2f} % ;'
                          f' min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')
            else:
                keys = ['tt', 'ee', 'bb'] if simlib.pol else ['tt']
                for key in keys:
                    fdiff = utils.get_fdiff(sim_spectra[component][freq][key], precomputed_spectra[component][freq][key])
                    min_fdiff = np.min(fdiff)
                    max_fdiff = np.max(fdiff)
                    avg_fdiff = np.mean(fdiff)
                    if abs(avg_fdiff) <= fdiff_tol:
                        print(f'  Success! Your {key.upper()} power spectrum matches the precomputed spectrum'
                              f' (average fractional difference is {avg_fdiff:5.2f} %)')
                    else:
                        if key == 'tt':
                            unmatched_info.append([component, freq])
                        print(f'  Your {key.upper()} power spectrum does not match the precomputed spectrum'
                              f' (average fractional difference is {avg_fdiff:5.2f} % ;'
                              f' min. = {min_fdiff:5.2f} %, max. = {max_fdiff:5.2f} %)')

    if len(unmatched_info) == 0:
        print('\nSuccess! All of your sim spectra match the precomputed spectra.')

    plots.plot_sim_spectra_comparison(sim_spectra, precomputed_spectra,  'Your sims', 'Precomputed',
                                      plot_fdiff=True, show=True,  lmin1=200, lmax1=20000, lmin2=200, lmax2=20000)


def print_instructions_for_10x10(hd_sims_dir, lowres_sims_dir=None,
                                 plot_maps=False, reproduce_sims_and_spectra=False,
                                 fig3_fname='fig3.png', fig4_fname='fig4.pdf', fig5_fname='fig5.pdf'):
    """Print instructions to reproduce the plots in arXiv:2609.16128, or
    to reproduce all of the simulation products for a 10 degree by
    10 degree patch of sky centered at R.A. = 6 degrees, dec. = 6 degrees.
    """
    if reproduce_sims_and_spectra:
        if lowres_sims_dir in [None, '']:
            raise ValueError("You must provide the path to the `lowres_sims_dir` in order to generate new simulations")
        print("To reproduce the HD sims and power spectra, you must download the full-sky S10 data by running the following command:\n")
        print(f"    bash download_all_S10sims_data.sh {lowres_sims_dir}\n")
        print("Then run the following command to generate the HD sims, sim power spectra, and plots:\n")
        print(f"    python reproduce_10x10.py {hd_sims_dir} {lowres_sims_dir}\n")
        print(f"The output files will be saved under: {hd_sims_dir}/ra6dec6_10x10deg_hdsims")
        print(f"The plots will be saved to: {hd_sims_dir}/ra6dec6_10x10deg_hdsims/plots")
    elif plot_maps:
        print("You must download the HD sim maps in order to plot them. This can be done by running the following command:\n")
        print(f"    bash download_90GHz_HDsims_maps.sh {hd_sims_dir}\n")
        print(f"Then, you can either plot the maps inside of this notebook, or you can run the following command to save all of the plots:\n")
        print(f"    python reproduce_10x10_plots.py {hd_sims_dir} --fig3fname {fig3_fname} --fig4fname {fig4_fname} --fig5fname {fig5_fname}")


def fig_maps_are_saved(simlib, freq=90):
    """Check if the maps needed to reproduce Figure 3 in arXiv:2609.16128 are saved."""
    all_files_saved = True # begin by assuming the maps are saved
    # we need the map of each component:
    for component in [*si.components, 'kappa']:
        sim_fname = simlib.get_signal_sim_fname(component, freq=freq)
        if not os.path.exists(sim_fname):
            all_files_saved = False
    # we also need the apodization window used when convolving the beam:
    window_fname = simlib.get_apod_window_fname(shape=simlib.padded_shape, wcs=simlib.padded_wcs, apod_width=simlib.map_apod_width) 
    if not os.path.exists(window_fname):
        all_files_saved = False
    # raise an error if the files weren't saved
    if not all_files_saved:
        raise FileNotFoundError(f"Cannot find the maps needed to reproduce Figure 3 in {simlib.sim_dir()}.")
    return all_files_saved


def reproduce_sim_maps_plot(hd_sims_dir, freq=90, show=True, fname=None):
    """Reproduce Figure 3 of arXiv:2609.16128."""
    simlib = hdsims.HDSims(hd_sims_dir, verbose=True)
    plt_output = simlib.plot_sim_maps(freq, show=show, save=(fname is not None), fname=fname)
    if not show:
        return plt_output


def reproduce_sim_spectra_comparison_plot(hd_sims_dir=None, show=True, fname=None):
    """Reproduce Figure 4 of arXiv:2609.16128."""
    if hd_sims_dir is None:
        hd_sims_dir = precomputed_hdsims_output_dir()
    simlib = hdsims.HDSims(hd_sims_dir, make_output_dirs=False)
    plt_output = simlib.plot_sim_spectra_comparison(show=show, save=(fname is not None), fname=fname, plot_fdiff=True, use_fig_settings=True)
    if not show:
        return plt_output


def reproduce_ksz_kappa_sim_spectra_plot(hd_sims_dir=None, show=True, fname=None):
    """Reproduce Figure 5 of arXiv:2609.16128."""
    if hd_sims_dir is None:
        hd_sims_dir = precomputed_hdsims_output_dir()
    simlib = hdsims.HDSims(hd_sims_dir, make_output_dirs=False)
    plt_output = simlib.plot_smallscale_ksz_kappa_spectra(show=show, save=(fname is not None), fname=fname, plot_fdiff=True)
    if not show:
        return plt_output


def make_run_hdsims_command(path_to_run_hdsims, hd_sims_dir, lowres_sims_dir,
                            ra_ctr=si.ra_ctr, dec_ctr=si.dec_ctr, width=si.width, height=si.height,
                            apod_width=si.apod_width, cmb_seed=si.cmb_seed, pol=True,
                            freqs=si.freqs, components=si.map_components,
                            calculate_power_spectra=True, save_plots=True):
    """Print out the command to use the `run_hdsims.py` file provided
    with the `hdsims` Github repository.
    """
    if hd_sims_dir in [None, '']:
        raise ValueError("You must provide the path to the `hd_sims_dir` where the simulations will be saved.")
    if lowres_sims_dir in [None, '']:
        raise ValueError("You must provide the path where the full-sky, lower-resolution S10 simulations will be (or already are) saved.")

    # check if S10 files are saved ; if not, need to download them
    need_s10_sims = False
    if 'tsz' in components:
        for freq in freqs:
            sim_fname = s10sims.get_fullsky_s10_sim_fname('tsz', freq=freq, s10_sims_dir=lowres_sims_dir)
            if not os.path.exists(sim_fname):
                need_s10_sims = True
    if ('ksz' in components) and (not os.path.exists(s10sims.get_fullsky_s10_sim_fname('ksz', s10_sims_dir=lowres_sims_dir))):
        need_s10_sims = True
    if (('kappa' in components) or ('cmb' in components)) and (not os.path.exists(s10sims.get_fullsky_s10_sim_fname('kappa', s10_sims_dir=lowres_sims_dir))):
        need_s10_sims = True
    if ('radio' in components) and (not os.path.exists(s10sims.get_fullsky_s10_catalog_fnames('radio', s10_sims_dir=lowres_sims_dir))):
        need_s10_sims = True
    if 'cib' in components:
        cib_catalog_fnames = s10sims.get_fullsky_s10_catalog_fnames('cib', s10_sims_dir=lowres_sims_dir)
        if not all([os.path.exists(fname) for fname in cib_catalog_fnames]):
            need_s10_sims = True

    # command to run hdsims:
    run_hdsims_file = os.path.join(path_to_run_hdsims, 'run_hdsims.py')
    run_hdsims_cmd_list = [f'python {run_hdsims_file} {hd_sims_dir} --lowres-sims-dir {lowres_sims_dir}']
    if not np.isclose(ra_ctr, si.ra_ctr, atol=utils.arcmin2deg(si.hd_res), rtol=0):
        run_hdsims_cmd_list.append(f'--ra {simutils.round_str(ra_ctr, n=5)}')
    if not np.isclose(dec_ctr, si.dec_ctr, atol=utils.arcmin2deg(si.hd_res), rtol=0):
        run_hdsims_cmd_list.append(f'--dec {simutils.round_str(dec_ctr, n=5)}')
    if not np.isclose(width, si.width, atol=utils.arcmin2deg(si.hd_res), rtol=0):
        run_hdsims_cmd_list.append(f'--width {simutils.round_str(width, n=5)}')
    if not np.isclose(height, si.height, atol=utils.arcmin2deg(si.hd_res), rtol=0):
        run_hdsims_cmd_list.append(f'--height {simutils.round_str(height, n=5)}')
    if not np.isclose(apod_width, si.apod_width, atol=utils.arcmin2deg(si.hd_res), rtol=0):
        run_hdsims_cmd_list.append(f'--apod_width {simutils.round_str(apod_width, n=5)}')
    if cmb_seed != si.cmb_seed:
        run_hdsims_cmd_list.append(f'--cmbseed {int(cmb_seed)}')
    if not pol:
        run_hdsims_cmd_list.append('--nopol')
    if set(freqs) != set(si.freqs):
        freqs_list = ' '.join([str(freq) for freq in freqs])
        run_hdsims_cmd_list.append(f'--freqs {freqs_list}')
    if (set(components) != set(si.components)) or (set(components) != set(si.map_components)):
        components_list = ' '.join(components)
        run_hdsims_cmd_list.append(f'--components {components_list}')
    if not calculate_power_spectra:
        run_hdsims_cmd_list.append('--nospectra')
    if not save_plots:
        run_hdsims_cmd_list.append('--noplots')
    run_hdsims_cmd = ' '.join(run_hdsims_cmd_list)


    if need_s10_sims:
        s10_download_file = os.path.join(path_to_run_hdsims, 'download_all_S10sims_data.sh')
        s10_download_cmd = f'bash {s10_download_file} {lowres_sims_dir}'
        print("Before generating the HD simulations, you must download the S10 simulation files ; you may do so by running the following command:\n")
        print("  ", s10_download_cmd, "\n")

    print("To generate the HD simulations, run the following command:\n")
    print("  ", run_hdsims_cmd, "\n")


# functions to load in some of the precomputed files we provide:

def load_precomputed_10x10_sim_spectra(component, freq=None, bin_cl=True, bin_dl=False):
    """Load the precomputed power spectra of the simulations used in
    arXiv:2609.16128 on a 10 degree by 10 degree patch of sky centered at
    R.A. = 6 degrees, dec. = 6 degrees.

    See `HDSims.load_signal_sim_power` for more information.
    """
    simlib = hdsims.HDSims(precomputed_hdsims_output_dir(), make_output_dirs=False)
    return simlib.load_signal_sim_power(component, freq=freq, bin_cl=bin_cl, bin_dl=bin_dl)


def load_precomputed_10x10_theory(component, binned=False, bin_dl=False):
    """Load the precomputed theory power spectra for the simulations on
    a 10 degree by 10 degree patch of sky centered at R.A. = 6 degrees,
    dec. = 6 degrees used in arXiv:2609.16128 .

    See `HDSims.get_sim_theory` for more information.
    """
    simlib = hdsims.HDSims(precomputed_hdsims_output_dir(), make_output_dirs=False)
    theo = simlib.get_sim_theory(component, binned=binned, dl=bin_dl)
    return theo


def load_precomputed_10x10_inv_mcm(bin_dl=False):
    """Load the precomputed inverse mode-coupling matrices for the
    simulations on a 10 degree by 10 degree patch of sky centered at
    R.A. = 6 degrees, dec. = 6 degrees used in arXiv:2609.16128 .

    See `HDSims.get_mode_coupling` for more information.
    """
    simlib = hdsims.HDSims(precomputed_hdsims_output_dir(), make_output_dirs=False)
    inv_mcm = simlib.get_mode_coupling(bin_dl=bin_dl, binning_matrix=False)
    return inv_mcm


def load_precomputed_10x10_binning_matrix(bin_dl=False):
    """Load the binning matrices calculated for the simulations on a
    10 degree by 10 degree patch of sky centered at R.A. = 6 degrees,
    dec. = 6 degrees used in arXiv:2609.16128

    See `HDSims.get_mode_coupling` for more information.
    """
    simlib = hdsims.HDSims(precomputed_hdsims_output_dir(), make_output_dirs=False)
    inv_mcm, bbl = simlib.get_mode_coupling(bin_dl=bin_dl, binning_matrix=True)
    return bbl

