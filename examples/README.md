# Examples


## Running `hdsims` for a small patch of sky at a single frequency

The notebook `example_2x2.ipynb` will show you how to use the `hdsims` package.

- The `download_90GHz_S10sims_data.sh` script can be run to download the 90 GHz full-sky data, if necessary.
- The `prepare_2x2_example_files.py` script can be run to do any long calculations outside of the notebook.
- Alternatively, we provide instructions within the `example_2x2.ipynb` notebook to download all of the files needed to run the example.

We explain how to use these files (if necessary) within the `example_2x2.ipynb` notebook.


## Reproducing the sim plots in MacInnis et. al. (2026)

The notebook `reproduce_10x10.ipynb` will reproduce Figures 4 and 5 in [MacInnis et. al. (2026)](https://arxiv.org/abs/2609.16128) using the sim and theory power spectra we provide (i.e., you do not need to calculate anything).

- The `reproduce_10x10_plots.py` and `download_90GHz_HDsims_maps.sh` scripts can be used if you would also like to reproduce the plot of the 90 GHz maps (Figure 3 in MacInnis et. al. 2026).
- The `reproduce_10x10.py`, `download_all_HDsims_data.sh`, and `download_all_S10sims_data.sh` scripts can be used if you would like to reproduce *all* of the products we release (simulations, power spectra, etc.)

We explain how to use these files (if necessary) within the `reproduce_10x10.ipynb` notebook.

