import camb
import numpy as np
import matplotlib.pyplot as plt

ini_file = 'raw_data/bode_almost_wmap5_params_highKeta.ini'
pars = camb.read_ini(ini_file)

# high-accuracy settings
pars.set_matter_power(kmax=10, k_per_logint=130)
pars.set_for_lmax (40000, \
    lens_potential_accuracy =30 , \
    lens_margin =2050)
pars.set_accuracy ( AccuracyBoost =1.1 , \
    lSampleBoost =3.0 , lAccuracyBoost =3.0 , \
    DoLateRadTruncation = False, min_l_logl_sampling=10000 )
pars.NonLinear = camb.model.NonLinear_both
pars.NonLinearModel.set_params("mead2016")

results = camb.get_results(pars)

lensed = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
unlensed = results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')['unlensed_scalar']

ells = np.arange(0, lensed.shape[0])
lensCL = results.get_lens_potential_cls(lmax=lensed.shape[0] + 2)
kk = (ells * (ells + 1))**2 * lensCL[2:len(ells)+2, 0] / 4.0

main_data = np.column_stack([
    ells,
    lensed[:,0], lensed[:,1], lensed[:,2], lensed[:,3],
    unlensed[:,0], unlensed[:,1], unlensed[:,2], unlensed[:,3],
    kk
])

np.savetxt(
    'output/camb_full_output.txt',
    main_data,
    header='ell    tt_lensed   ee_lensed   bb_lensed   te_lensed   '
           'tt_unlensed ee_unlensed bb_unlensed te_unlensed kk',
    fmt='%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e'
)