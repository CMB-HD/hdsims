import healpy as hp
from hdsims import Foregrounds
from pspy import so_map
import os

#################################

try:
    os.makedirs("hdsims_output_from_example_notebook/S10_patches/", exist_ok=True)
    print(f"Made directory: hdsims_output_from_example_notebook/S10_patches/")
except FileExistsError:
    print(f"Directory already exists: hdsims_output_from_example_notebook/S10_patches/")

#################################

S10_resolution = hp.nside2resol(8192, arcmin=True) # Original S10 0.43' pixel size

foregrounds_S10 = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 2.0,
    new_res = S10_resolution,
    apod_width = 0.2,
    l_max = 24000)

#################################

component = 'radio'
catalog = foregrounds_S10.make_catalog(data_path = 'S10_data/', component = component)
catalog.to_csv(f"hdsims_output_from_example_notebook/S10_patches/{component}_2.4x2.4deg_source_catalog_ra=6_dec=6.csv")

#################################

component = 'CIB'
catalog = foregrounds_S10.make_catalog(data_path = 'S10_data/', component = component)
catalog.to_csv(f"hdsims_output_from_example_notebook/S10_patches/{component}_2.4x2.4deg_source_catalog_ra=6_dec=6.csv")

#################################