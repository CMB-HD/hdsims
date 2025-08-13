# Simulating Foregrounds for CMB-HD

This repository contains the methods to simulate and analyze ultra-high-resolution sky patches for CMB-HD, including
- Maps of diffuse foregrounds kSZ, tSZ, and lensing convergence
- Maps of discrete foregrounds CIB and radio
- Maps of Unlensed CMB and Lensed CMB using lensing convergence map
- Binning procedure and data covariance matrices

To do so, we take the fullsky maps for diffuse foregrounds and the discrete catalogs from [Sehgal et al. (2010)](https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html).

This work should reproduce the results in [!put the final paper here!](https://arxiv.org/).

# Installation Instructions

Clone this repository using the command:

```
git clone https://github.com/CMB-HD/hdsims
```
To generate CMB-HD simulations using this repository, you will need to install Python 3 and several Python packages. We make use of Python 3.11.6, pixell 0.23.14, pspy 1.7.5, numpy 1.26.0, healpy 1.17.1, scipy 1.11.3, and pandas 2.1.3. 

The full sky maps of the diffuse foregrounds and the discrete catalogs used in [Sehgal et al. (2010)](https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html) at all frequencies can be found here: (https://lambda.gsfc.nasa.gov/data/tools/simulations) where they can be downloaded directly from the website. Alternatively, these data can be downloaded directly from the command line by running

```
source download_all_S10_data.sh
```

in the `hdsims/S10_data` folder. Note that running this command will download data for all foregrounds and at all frequencies, which is just under 140 GB of data. 

# Usage

See the `example.ipynb` Jupyter notebook for a walkthrough example for small 2$^\circ$ by 2$^\circ$ sky patches. For larger patches, many of the functions cannot be run in a Jupyter notebook, so we also provide `.py` files which generate the results used in [!put the final paper here!](https://arxiv.org/).

***JA: Needs updating***
