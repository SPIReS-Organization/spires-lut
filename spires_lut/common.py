import numpy as np

def generate_filter(wvl_m, wvl, wl_resol):
    """
    Generate norm array for wavelength resampling.

    :param wvl_m:    start wavelength grid
    :param wvl:      final wavelength grid
    :param wl_resol: resolution of final wavelength grid
    :return:         resampling norm
    """
    num_wvl_m = len(wvl_m)
    num_wvl = len(wvl)

    s_norm_m = np.zeros((num_wvl_m, num_wvl))
    exp_max = 2.
    exp_min = 2.
    exp_arr = exp_max + (exp_min - exp_max) * np.arange(0, 2100, 1) / (num_wvl - 1)
    c_arr = (1 / (2 ** exp_arr * np.log(2))) ** (1 / exp_arr)

    for bd in range(num_wvl):
        li1 = np.logical_and(wvl_m >= (wvl[bd] - 2. * wl_resol[bd]), wvl_m <= (wvl[bd] + 2. * wl_resol[bd]))
        li1 = np.where(li1)
        cnt = len(li1)

        if cnt > 0:
            tmp = np.abs(wvl[bd] - wvl_m[li1]) / (wl_resol[bd] * c_arr[bd])
            s = np.exp(-(tmp ** exp_arr[bd]))
            s_norm_m[li1, bd] = s / np.sum(s)

    return s_norm_m


def wvl_resample(specs, wvl_m, wvl, wl_resol):
    """
    Resample given spectra to desired wavelength grid.

    :param specs:    input spectra, can be single spectrum or array of spectra
    :param wvl_m:    input center wavelengths
    :param wvl:      output center wavelengths
    :param wl_resol: output fwhm
    :return:         input spectra resampled to desired wavelength grid
    """
    s_norm = generate_filter(wvl_m=wvl_m, wvl=wvl, wl_resol=wl_resol)
    specs_resample = specs @ s_norm

    return specs_resample
