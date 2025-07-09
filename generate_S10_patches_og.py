from CMB_HD_UltraHighRes import Foregrounds
import healpy as hp

patcher = Foregrounds(
    ra=6,
    dec=6,
    final_width=12,
    new_res=hp.nside2resol(8192, arcmin=True),
    apod_width=1,
    l_max = 24000)

############################################

component = 'tSZ'
print(component)
for frequency in ['030','090','148','219','277','350']:
    print(frequency)
    patcher.generate_diffuse_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/{frequency}_tsz_healpix.fits",
                    do_fullsky_part = True)

############################################

component = 'kSZ'
print(component)
for frequency in ['030','090','148','219','277','350']:
    print(frequency)
    patcher.generate_diffuse_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/{frequency}_ksz_healpix.fits",
                    do_fullsky_part = True)

############################################

component = 'radio'
print(component)
for frequency in ['030','090','148','219','277','350']:
    print(frequency)
    patcher.generate_discrete_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/radio.cat",
                    make_catalog = False)

############################################

component = 'CIB'
print(component)
for frequency in ['350','277','219','148','090','030']:
    print(frequency)
    patcher.generate_discrete_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/",
                    make_catalog = False)

############################################

patcher = Foregrounds(
    ra=6,
    dec=6,
    final_width=12,
    new_res=hp.nside2resol(4096, arcmin=True),
    apod_width=1,
    output_folder = "/gpfs/projects/SehgalGroup/jange/final/output/",
    l_max = 24000)

############################################

component = 'kappa'
frequency = 'none'
print(component)
print(frequency)
patcher.generate_diffuse_foreground(
                component = component,
                frequency = frequency,
                data_path = f"raw_data/healpix_4096_KappaeffLSStoCMBfullsky.fits",
                do_fullsky_part = False)
