from CMB_HD_UltraHighRes import Foregrounds
import healpy as hp
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

foregrounds_HD = Foregrounds(
    ra = 6,
    dec = 6,
    final_width = 12,
    new_res = 0.04,
    apod_width = 1,
    output_path = 'output/',
    l_max = 24000)
'''
component = 'tSZ'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    foregrounds_HD.generate_diffuse_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/{freq_to_freqpath[frequency]}_tsz_healpix.fits",
                    do_fullsky_part = False)
component = 'kSZ'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    foregrounds_HD.generate_diffuse_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/{freq_to_freqpath[frequency]}_ksz_healpix.fits",
                    do_fullsky_part = False)
    
component = 'radio'
print(component)
for frequency in [30,90,148,219,277,350]:
    print(frequency)
    foregrounds_HD.generate_discrete_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/",
                    make_catalog = False)
'''
component = 'CIB'
print(component)
#for frequency in [30,90,148,219,277,350]:
for frequency in [219,277,350]:
    print(frequency)
    foregrounds_HD.generate_discrete_foreground(
                    component = component,
                    frequency = frequency,
                    data_path = f"raw_data/",
                    make_catalog = False)

component = 'kappa'
frequency = None
print(component)
print(frequency)
foregrounds_HD.generate_diffuse_foreground(
                component = component,
                frequency = frequency,
                data_path = f"raw_data/healpix_4096_KappaeffLSStoCMBfullsky.fits",
                do_fullsky_part = False)

############################################
