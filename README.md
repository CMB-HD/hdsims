# Ultrahigh-resolution microwave sky simulations

This repository contains the methods to generate and analyze ultrahigh-resolution (0.04 arcminute) microwave simulations on a patch of the sky.

These simulations are used in [TODO:link to paper](https://arxiv.org/); if you use this code (or any of the [public products (TODO: link to HD sims on LAMBDA)](https://lambda.gsfc.nasa.gov) that it has generated), please cite that work and the lower-resolution counterpart on which the simulations you generate are based. By default, our simulations are based on the [publicly available](https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html) full-sky maps and catalogs used in [Sehgal et al. (2010)](https://arxiv.org/abs/0908.0540) (which we will refer to as "S10").



The `hdsims` code we provide here can be used to produce:
- Simulations of the thermal and kinetic SZ effects (tSZ and kSZ, respectively), the cosmic infrared background (CIB), radio galaxies, the lensing convergence field, and the lensed and unlensed CMB for both temperature and polarization
- Catalogs of the SZ clusters and point sources (CIB and radio) contained in the simulations
- Accurate power spectra of the simulations, corrected for effects such as the incomplete sky coverage or an instrumental beam.

We also provide examples that can reproduce the [HD simulations (TODO: link to HD sims on LAMBDA)](https://lambda.gsfc.nasa.gov) that we have released and Figures 2, 3, and 4 of [TODO:link to paper](https://arxiv.org/).



---


## Installation Instructions

To install the `hdsims` package, navigate to the directory where you would like to place this repository. Then clone and install it via `pip`:

```
git clone https://github.com/CMB-HD/hdsims
cd hdsims
pip install . --user
```


### Required packages


To use the `hdsims` code, you will need to install Python 3 and several Python packages. We make use of Python 3.11.6 and:
- [numpy](https://numpy.org/) 1.26.0
- [scipy](https://scipy.org/) 1.11.3
- [pandas](https://pandas.pydata.org/) 2.1.3
- [matplotlib](https://matplotlib.org/) 3.8.1
- [pixell](https://pixell.readthedocs.io/) 0.23.14
- [pspy](https://pspy.readthedocs.io/) 1.7.5
- [healpy](https://healpy.readthedocs.io/) 1.17.1
- [camb](https://camb.readthedocs.io/) 1.5.4


### Requirements to produce new simulations


To generate new simulations, you will need the full-sky maps and catalogs used in [Sehgal et al. (2010)](https://arxiv.org/abs/0908.0540), which can be downloaded from [LAMBDA](https://lambda.gsfc.nasa.gov/simulation/full_sky_sims_ov.html). We provide a command-line script, `download_all_S10sims_data.sh`, that will download the necessary files. From the `hdsims` directory (i.e., the directory where this readme file is located), run the command

```
bash download_all_S10sims_data.sh /path/to/myS10sims
```

where `/path/to/myS10sims` is the path to the directory where you would like to save the S10 files.


**Important notes** about the S10 simulations:
- Running this command will download the necessary data for all foregrounds and at all frequencies, which is nearly 130 GB of data.
- These files were last modified in 2009: if you are on a cluster and place the files in a "scratch" directory, they may be automatically deleted (depending on the policies on the cluster you're using).
- The full-sky S10 kSZ and lensing convergence sims have a discontinuity along dec. = $0^\circ$ and right ascension = $0^\circ$ and $90^\circ$ ; we recommend that you avoid these regions if you generate new simulations.

---

## How to download the public HD sims from LAMBDA

We provide a command-line script to download the HD simulations we provide on **[TODO:LINK2LAMBDA]()** for a $10^\circ \times 10^\circ$ patch of sky. If you would like to download these simulations, from the `hdsims` directory (i.e., the directory where this readme file is located), run the command


```
bash download_all_HDsims_data.sh /path/to/myHDsims
```

where `/path/to/myHDsims` is the path to the directory where you would like to save the simulations.

---

## Usage

The `HDSims` class in the `hdsims` module (i.e., `hdsims/hdsims.py`) can be used to generate a new set of simulations, take their power spectra, and plot the results. This can all be done with the `generate_and_powerspectra_hd_sims` method of `HDSims`. We provide a more detailed example (see the "Examples" section below) of how to use the code, but there are only two main steps: (1) initialize the `HDSims` class, and (2) call the `generate_and_powerspectra_hd_sims` method. This is shown in the python snippet below:

```
from hdsims import hdsims
hd_sims_dir = '/path/to/myHDsims'
lowres_sims_dir = '/path/to/myS10Sims'
simlib = hdsims.HDSims(hd_sims_dir, lowres_sims_dir=lowres_sims_dir)
simlib.generate_and_powerspectra_hd_sims()
```

By default, this will generate a set of 0.04 arcminute simulated maps for a $10^\circ \times 10^\circ$ patch of sky centered at R.A. = $6^\circ$, dec. = $6^\circ$ and take their power spectra.


If you download the HD sims we provide on **[TODO:LINK2LAMBDA]()**, then you can pass the same `hd_sims_dir` that you used when downloading the sims. In this case, it would not be necessary to download the S10 sims, and you could set `lowres_sims_dir=None` in the snippet above.


### Examples

We provide two examples in the `examples` directory:

1. `examples/example_2x2.ipynb`: This is a general example of how to use `hdsims` to generate a set of simulations on a given patch of sky and take their power spectra. In this example we generate 90 GHz maps for a $2^\circ \times 2^\circ$ patch of sky.
2. `examples/reproduce_10x10.ipynb`: An example to either reproduce *all* of the public simulation products we release on **[TODO:LINK2LAMBDA]()** (used in **[TODO:LINK2PAPER]()**) for a  $10^\circ \times 10^\circ$ patch of sky, or to *only* reproduce the plots in **[TODO:LINK2PAPER]()** without needing to generate any simulations.


