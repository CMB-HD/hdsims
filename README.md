# Simulating Foregrounds for CMB-HD

This repository contains the methods to simulate and analyze ultra-high-resolution sky patches for CMB-HD, including
- Maps of diffuse foregrounds kSZ, tSZ, and lensing convergence
- Maps of discrete foregrounds CIB and radio
- Maps of Unlensed CMB and Lensed CMB based on lensing convergence maps
- Binning procedure and data covariance matrices

To do so, we take the fullsky maps for diffuse foregrounds and the discrete catalogs from [Sehgal et al. (2010)](https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html).

This work should reproduce the results in [!put the final paper here!](https://arxiv.org/).

# Installation

To load the files used in the Python functions in `CMB_HD_UltraHighRes/foreground_patches.py`, `CMB_HD_UltraHighRes/power_spectrum.py`, and `CMB_HD_UltraHighRes/CMB_patches.py`, first install python and the packages pixell, pspy, numpy, healpy, scipy, and pandas.

Then, simply clone this repository and import with `from CMB_HD_UltraHighRes import Foregrounds, Spectra, CMB`.

# Usage

The functions to generate foreground patches are located in the `Foregrounds` class of `CMB_HD_UltraHighRes/foreground_patches.py`. To generate your own patches, use the functions `generate_diffuse_foreground` and `generate_discrete_foreground`. The kSZ and lensing convergence foregrounds need to use a stitching procedure. For these, use the functions `get_theory_for_stitching` to construct your ideal patch theory, `generate_diffuse_foreground` to generate the foreground, `get_S10_for_stitching` to get the correct alms from your patches and then `stitch_alms` to generate the final, correct patch. (Need to add information here about the CIB)!

The functions to generate and lens the CMB in accordance with the lensing convergence foreground are located in the `CMB` class of `CMB_HD_UltraHighRes/CMB_patches.py`. Use `make_unlensed_patch`, `save_CMB_alms`, and `do_lensing` to construct accurate lensed CMB sky patches.

The functions to take power spectra are located in the `Spectra` class of `CMB_HD_UltraHighRes/power_spectrum.py`. To take a power spectrum, compute the binning files with the `make_binning_files` function (if the binning file is not already computed), then use the `get_foreground_power` function.

See the `example.ipynb` Jupyter notebook for a walkthrough example for small 1.2$^\circ$ by 1.2$^\circ$ sky patches. See also the plots for a larger 12$^\circ$ by 12$^\circ$ sky patch. For larger patches, many of the functions cannot be run in a Jupyter notebook, so we also provide `.py` files which generate the results used in [!put the final paper here!](https://arxiv.org/), and whose usage is also described in `example.ipynb`. File paths may need to be changed across these files.
