from CMB_HD_UltraHighRes import CMB, Spectra
from pixell import enmap
import numpy as np

overall_path_example = '/gpfs/projects/SehgalGroup/jange/final/output/'
l_max = 24000
res=0.04

CMB_patch = CMB(
    ra=6,
    dec=6,
    final_width=12,
    res=res,
    apod_width=1,
    l_max = l_max)

print("Make unlensed CMB")
CMB_patch.make_unlensed_patch(theory_path = f"{overall_path_example}camb_full_output.txt", 
                              cmb_seed = 3, 
                              unlensed_output = f"{overall_path_example}CMB/unlensed/", 
                              convolve_pw = False)
print("Save CMB alms")
CMB_patch.save_CMB_alms(f"{overall_path_example}CMB/unlensed/polarization_unlensedcmb003_0.04arcmin_ra6dec6_12x12deg.fits", 
                        f"{overall_path_example}kappa_stitched/none/0.04/inner_patch_stitched",
                        f"{overall_path_example}CMB/lensed/")

print("Do lensing")
CMB_patch.do_lensing(f"{overall_path_example}CMB/lensed/")

ps = Spectra(
    ra=6,
    dec=6,
    final_width=10,
    res=res,
    apod_width=1,
    l_max = l_max)

print("Make binning files")
ps.make_binning_files(delta_ell = 200, spin = 2, binning_output_path = f"{overall_path_example}binning_files/")

print("Taking unlensed power")

mbb_inv = np.load(f"{overall_path_example}binning_files/mbb_inv_lmax{l_max}_deltaEll200_6,6_10_1_{res}_spin0and2.npy", allow_pickle=True)
binning_file = f"{overall_path_example}binning_files/lmax{l_max}_deltaEll200_6,6_10_1_{res}"

CMB_unlensed_patch = enmap.read_map(f"{overall_path_example}CMB/unlensed/polarization_unlensedcmb003_0.04arcmin_ra6dec6_12x12deg.fits")

ps.get_CMB_power(CMB_unlensed_patch[0], CMB_unlensed_patch[1], CMB_unlensed_patch[2], 
                 mbb_inv, binning_file, f"{overall_path_example}CMB/unlensed/", 
                 deconvolve_pw = False)

print("Taking lensed power")
shape12, wcs12 = ps.get_shape_wcs(0.04, 6, 6, 12, height=12)
patch_T = enmap.enmap(np.load(f"{overall_path_example}CMB/lensed/lensed_0output.npy"),wcs12)
patch_Q = enmap.enmap(np.load(f"{overall_path_example}CMB/lensed/lensed_1output.npy"),wcs12)
patch_U = enmap.enmap(np.load(f"{overall_path_example}CMB/lensed/lensed_2output.npy"),wcs12)

ps.get_CMB_power(patch_T, patch_Q, patch_U, 
                 mbb_inv, binning_file, f"{overall_path_example}CMB/lensed/", 
                 deconvolve_pw = True)