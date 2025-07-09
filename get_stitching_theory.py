from CMB_HD_UltraHighRes import Foregrounds
import healpy as hp
import pandas as pd
import numpy as np

res = 0.04

patcher = Foregrounds(
    ra=6,
    dec=6,
    final_width=12,
    new_res=res,
    apod_width=1,
    l_max = 24000)

############################################

ksz_template = pd.read_csv("raw_data/cmbhd_total_ksz_cls_v1.1.txt", delimiter=' ')
template_ells = ksz_template['ell']
template_cls = ksz_template['C_ell^kSZ']

for frequency in ['030','090','148','219','277','350']:
    patch_cls =  hp.read_cl(f"output/kSZ/{frequency}/{res}/cls.fits")
    patch_ells = hp.read_cl(f"output/kSZ/{frequency}/{res}/ells.fits")
    
    patcher.get_theory_for_stitching(f"/gpfs/projects/SehgalGroup/jange/final/output/kSZ_stitched/{frequency}/{res}/",
                                          template_cls,
                                          template_ells,
                                          patch_cls, 
                                          patch_ells, \
                                          template_minimization_index = 8102, 
                                          patch_minimization_index = 40)


############################################

hddata = np.loadtxt("output/camb_full_output.txt", comments="#")
template_ells =  hddata[:,0]
template_cls = hddata[:,9]
template_cls *= 2*np.pi/(template_ells * (template_ells + 1))**2

frequency = 'none'
patch_cls =  hp.read_cl(f"output/kappa/{frequency}/{res}/cls.fits")
patch_ells = hp.read_cl(f"output/kappa/{frequency}/{res}/ells.fits")

patcher.get_theory_for_stitching(f"/gpfs/projects/SehgalGroup/jange/final/output/kappa_stitched/{frequency}/{res}/",
                                      template_cls,
                                      template_ells,
                                      patch_cls, 
                                      patch_ells, \
                                      template_minimization_index = 3902, 
                                      patch_minimization_index = 19)
