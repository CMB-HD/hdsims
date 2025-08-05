from CMB_HD_UltraHighRes import Spectra
from pixell import enmap
import numpy as np

ps_og = Spectra(
    ra=6,
    dec=6,
    final_width=10,
    res=0.8588716029543515,
    apod_width=1,
    l_max = 6287)

ps_og.make_binning_files(delta_ell = 200, spin = 0, binning_output_path = "/gpfs/projects/SehgalGroup/jange/final/output/binning_files/")

ps_og = Spectra(
    ra=6,
    dec=6,
    final_width=10,
    res=0.42943580147717575,
    apod_width=1,
    output_folder = "/gpfs/projects/SehgalGroup/jange/final/output/",
    l_max = 12574)

ps_og.make_binning_files(delta_ell = 200, spin = 0, binning_output_path = "/gpfs/projects/SehgalGroup/jange/final/output/binning_files/")

ps_uhd = Spectra(
    ra=6,
    dec=6,
    final_width=10,
    res=0.04,
    apod_width=1,
    output_folder = "/gpfs/projects/SehgalGroup/jange/final/output/",
    l_max = 24000)

ps_uhd.make_binning_files(delta_ell = 200, spin = 0, binning_output_path = "/gpfs/projects/SehgalGroup/jange/final/output/binning_files/")
