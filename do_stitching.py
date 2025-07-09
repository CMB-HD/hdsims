from CMB_HD_UltraHighRes import Foregrounds
from pixell import enmap
import numpy as np

res = 0.04

patcher = Foregrounds(
    ra=6,
    dec=6,
    final_width=12,
    new_res=res,
    apod_width=1,
    l_max = 24000)

for frequency in ['030','090','148','219','277','350']:
    patch = enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/final/output/kSZ/{frequency}/{res}/inner_patch")
    
    kSZ_alms_for_stitching_resized = patcher.get_S10_for_stitching(f"/gpfs/projects/SehgalGroup/jange/final/output/kSZ_stitched/{frequency}/{res}/", 
                                                                   patch)

    kSZ_map_stitched = patcher.stitch_alms(alms_theory = np.load(f"/gpfs/projects/SehgalGroup/jange/final/output/kSZ_stitched/{frequency}/{res}/alms_stitchingTheory.npy"), 
                                            alms_S10 = np.load(f"/gpfs/projects/SehgalGroup/jange/final/output/kSZ_stitched/{frequency}/{res}/alms_for_stitching_resized.npy"), 
                                            output_path = f"/gpfs/projects/SehgalGroup/jange/final/output/kSZ_stitched/{frequency}/{res}/",
                                            l_cutoff = 8000)


frequency = 'none'
patch = enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/final/output/kappa/{frequency}/{res}/inner_patch")

kappa_alms_for_stitching_resized = patcher.get_S10_for_stitching(f"/gpfs/projects/SehgalGroup/jange/final/output/kappa_stitched/{frequency}/{res}/", 
                                                               patch)
kappa_map_stitched = patcher.stitch_alms(alms_theory = np.load(f"/gpfs/projects/SehgalGroup/jange/final/output/kappa_stitched/{frequency}/{res}/alms_stitchingTheory.npy"), 
                                        alms_S10 = np.load(f"/gpfs/projects/SehgalGroup/jange/final/output/kappa_stitched/{frequency}/{res}/alms_for_stitching_resized.npy"), 
                                        output_path = f"/gpfs/projects/SehgalGroup/jange/final/output/kappa_stitched/{frequency}/{res}/",
                                        l_cutoff = 4000)