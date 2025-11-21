import numpy as np

def spectrum_real(mode_intensities, wavenumber_range, peak_width, lambda_0, T):
    """
    Calculates and returns the intensity of the Raman spectrum in a given range.

    Arguments:
        mode_intensities: np.ndarray -- the intensities of each vibrational mode
        wavenumber_range: np.ndarray -- the range of frequencies
        peak_width: int -- The full width at half maximum (FWHM) of the Raman vibrational peaks, 
                                plotted as Lorentzian lines (unit: cm^-1)
        lambda_0: int -- incident wavelength (unit: nm)
        T: int -- Temperature, used to calculate the phonon population for each vibrational mode (unit: K)
    
    Returns:
        spectrum_total: np.ndarray -- the intensity of the spectrum in range wavenumber_range
    """

    parm = 0.6950302506112777  # Unit convert: 1K = 0.6950302506112777 cm-1
    vib_count = mode_intensities.shape[0]  # Number of vibrational modes
    spectrum = np.zeros((vib_count, wavenumber_range.size))
    
    # Calculating the spectrum
    for i in range(vib_count):
        if mode_intensities[i, 0] > 0:
            T_factor = 1 - np.exp(-mode_intensities[i, 0] / (parm * T))
            spectrum[i, :] = (
                mode_intensities[i, 1] * peak_width**2 / (2 * np.pi) *
                1.0 / ((wavenumber_range - mode_intensities[i, 0])**2 + 0.25 * peak_width**2) *
                (10**7 / lambda_0 - mode_intensities[i, 0])**4 / 
                (mode_intensities[i, 0] * T_factor * 1e9)
            )
        else:
            spectrum[i, :] = 0 * wavenumber_range
    
    spectrum_total = np.sum(spectrum, axis=0)

    return spectrum_total