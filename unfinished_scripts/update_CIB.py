from CMB_HD_UltraHighRes import Foregrounds
import healpy as hp
from pixell import enmap
import pandas as pd

patcher = Foregrounds(
    ra=6,
    dec=6,
    final_width=12,
    new_res=0.04,
    apod_width=1,
    l_max = 24000,
    output_path = "/gpfs/projects/SehgalGroup/jange/hdsims/output/")

S10_CIB_sim = {
    30: enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB/030/0.42943580147717575/initial_patch"),
    90: enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB/090/0.42943580147717575/initial_patch"),
    148: enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB/148/0.42943580147717575/initial_patch"),
    219: enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB/219/0.42943580147717575/initial_patch"),
    277: enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB/277/0.42943580147717575/initial_patch"),
    350: enmap.read_map(f"/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB/350/0.42943580147717575/initial_patch")
}

new_CIB_catalog = patcher.make_catalog_from_sims(sims = S10_CIB_sim, component = "CIB_updated")

shape = S10_CIB_sim[90].shape
wcs = S10_CIB_sim[90].wcs
new_CIB_catalog['ra_deg'], new_CIB_catalog['dec_deg'] = patcher.add_gauss_scatter_to_coords(new_CIB_catalog['ra_deg'].values, new_CIB_catalog['dec_deg'].values, shape, wcs, sigma_pix_frac=0.2)

new_CIB_catalog.to_csv("/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB_updated/sources_in_14x14_6,6.csv")
new_CIB_catalog = pd.read_csv("/gpfs/projects/SehgalGroup/jange/hdsims/output/CIB_updated/sources_in_14x14_6,6.csv")

component = 'CIB_updated'
print(component)
for frequency in ['350','277','219','148','090','030']:
    print(frequency)
    patcher.generate_discrete_foreground_from_custom_catalog(
                component = component,
                frequency = frequency,
                catalog = new_CIB_catalog)