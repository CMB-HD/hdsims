from CMB_HD_UltraHighRes import Spectra
import numpy as np
import healpy as hp
from pixell import enmap

############################################

res = 0.04
l_max = 24000

ps = Spectra(
    ra=6,
    dec=6,
    final_width=10,
    res=res,
    apod_width=1,
    l_max = l_max)
mbb_inv = np.load(f"/gpfs/projects/SehgalGroup/jange/final/output/binning_files/mbb_inv_lmax{l_max}_deltaEll200_6,6_10_1_{res}_spin0.npy", allow_pickle=True)
binning_file = f"/gpfs/projects/SehgalGroup/jange/final/output/binning_files/lmax{l_max}_deltaEll200_6,6_10_1_{res}"

component = 'kSZ'
print(component)
for frequency in ['030','090','148','219','277','350']:
    print(frequency)
    target_path = f"/gpfs/projects/SehgalGroup/jange/final/output/{component}_stitched/{frequency}/{res}/"
    patch = enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/final/output/{component}_stitched/{frequency}/{res}/inner_patch_stitched")
    
    ps.get_foreground_power(patch, mbb_inv, binning_file, deconvolve_pw = True,
                        output_path = target_path)

component = 'kappa'
frequency = 'none'
print(component)
print(frequency)
target_path = f"/gpfs/projects/SehgalGroup/jange/final/output/{component}_stitched/{frequency}/{res}/"
patch = enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/final/output/{component}_stitched/{frequency}/{res}/inner_patch_stitched")

ps.get_foreground_power(patch, mbb_inv, binning_file, deconvolve_pw = False,
                    output_path = target_path)

############################################
