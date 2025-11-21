import numpy as np
import matplotlib.pyplot as plt
from ters_img_simulator.core.read_gaussian import read_gaussian
from ters_img_simulator.core.load_molecule import load_molecule
from ters_img_simulator.core.analytic_field import analytic_field
from ters_img_simulator.core.spectrum_real import spectrum_real
import time as time
from pathlib import Path


def generate_spectrum(filename, tip_xyz, E_type, peak_width, lambda_0, T, tip_width):
    """
    Generate a spectrum for a given molecule.

    Parameters:
    filename (str): The filename of the molecule data.
    tip_xyz (tuple): The coordinates of the tip.
    E_type (int): The type of electric field.
    peak_width (float): The width of the spectral peaks.
    lambda_0 (float): The wavelength parameter.
    T (float): The temperature parameter.
    tip_width (float): The width of the tip.

    Returns:
    spectrum (array): The generated spectrum.
    wavenumber_range (array): The range of wavenumbers.
    """
    
    atom_polarizabilities, atoms, frequencies, atom_pos, R = load_molecule(filename, 0, 0, 90)

    wavenumber_range = np.arange(150, 4000, 1)

    if (E_type == 0):
        mode_intensities, dipoles = \
        analytic_field(atom_polarizabilities, atoms, frequencies, E_type)
    elif (E_type == 1 or E_type == 2):
        mode_intensities, dipoles = \
            analytic_field(atom_polarizabilities, atoms, frequencies, E_type, atom_pos, R, tip_xyz, tip_width)
    spectrum = spectrum_real(mode_intensities, wavenumber_range, peak_width, lambda_0, T)

    return spectrum, wavenumber_range