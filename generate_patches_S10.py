from CMB_HD_UltraHighRes import Foregrounds
import healpy as hp
S10_resolution = hp.nside2resol(8192, arcmin=True) # 0.43'
S10_resolution_kappa = hp.nside2resol(4096, arcmin=True) # 0.86'
freq_to_freqpath = {
        30: "030",
        90: "090",
        148: "148",
        219: "219",
        277: "277",
        350: "350",
        None: "none"
    }

############################################
'''
foregrounds_S10 = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12,
    new_res = S10_resolution,
    apod_width = 1,
    output_path = 'output/',
    l_max = 24000)

component = 'tSZ'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    foregrounds_S10.generate_diffuse_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/{freq_to_freqpath[frequency]}_tsz_healpix.fits",
                    do_fullsky_part = True)
component = 'kSZ'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    foregrounds_S10.generate_diffuse_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/{freq_to_freqpath[frequency]}_ksz_healpix.fits",
                    do_fullsky_part = True)

component = 'radio'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    if frequency == 30:
        foregrounds_S10.generate_discrete_foreground(
                        component = component,
                        frequency = frequency,
                        data_path = f"raw_data/",
                        make_catalog = True)
    else:
        foregrounds_S10.generate_discrete_foreground(
                        component = component,
                        frequency = frequency,
                        data_path = f"raw_data/",
                        make_catalog = False)

component = 'CIB'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    if frequency == 30:
        foregrounds_S10.generate_discrete_foreground(
                        component = component,
                        frequency = frequency,
                        data_path = f"raw_data/",
                        make_catalog = True)
    else:
        foregrounds_S10.generate_discrete_foreground(
                        component = component,
                        frequency = frequency,
                        data_path = f"raw_data/",
                        make_catalog = False)
'''
############################################

foregrounds_S10_kappa = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12,
    new_res = S10_resolution_kappa,
    apod_width = 1,
    output_path = 'output/',
    l_max = 24000)

component = 'kappa'
frequency = None
print(component)
print(frequency)
foregrounds_S10_kappa.generate_diffuse_foreground(
                component = component,
                frequency = frequency,
                data_path = f"raw_data/healpix_4096_KappaeffLSStoCMBfullsky.fits",
                do_fullsky_part = True)

############################################
