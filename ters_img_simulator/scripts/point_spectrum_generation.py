"""
This module generates data that can be used to create simulated TERS images for molecules.
The TERS data in saved in a folder in .npz format.
The data contains the positions of atoms, atomic numbers, grid positions, normal modes and the intensity values for every normal mode for every grid position.

Arguments:
    directory_path: str -- the path to the directory containing the .fchk files.
    save_path: str -- the path to the directory to save the image data.
    log_file: str -- the name of the log file (Not the whole path).
    rotation_angles: list -- list of rotation angles of the molecule. Default is [0].
"""
# python point_spectrum_generation.py /scratch/phys/sin/sethih1/data_files/all_group /scratch/phys/sin/sethih1/data_files/all_group_images_freq /scratch/phys/sin/sethih1/data_files/all_group_freq_log

# Ensure the project root is in sys.path for module imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import numpy as np
from ters_img_simulator.core.load_molecule import load_molecule
from ters_img_simulator.core.analytic_field import analytic_field
from ters_img_simulator.core.spectrum_real import spectrum_real
import time as time
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import logging
import argparse
from datetime import datetime

from ters_img_simulator.utils.utils import *


# Defining constants
CUTOFF_FREQUENCY = 0
E_TYPE = 2
PEAK_WIDTH = 5  # Default 5, in matlab 7
LAMBDA_0 = 532  # Default 532
T = 1e-6  # Temperature, K
TIP_WIDTH = np.array([1, 1, 1])*5
# PHI, THETA, PSI = 0, 0, 0
# PHI, THETA = 0, 0
X_COUNT, Y_COUNT = 64, 64  # Frame resolution
X_COUNT, Y_COUNT = 256, 256  # Frame resolution
#X_COUNT, Y_COUNT = 32, 32
X_WIDTH, Y_WIDTH = 18, 18  # Frame width and height, A

# Check if slurm CPUs are detected
if 'SLURM_CPUS_PER_TASK' in os.environ:
    cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    cpus = os.cpu_count()


def generate_ters_data(filename, molecule_rotation, plot_spectrum):
    """
    Calculates the Raman spectra for a grid on top of a molecule with desired resolution.

    Arguments:
        filename: Path object -- the .fchk file that contains molecule data.
        molecule_rotation: list -- list of angles that defines the rotation of the molecule. (unit: degrees)
        plot_spectrum: list -- Lis of frequencies where spectrum is calculated. If None, the spectrum is calculated at normal modes of the molecule.

    Returns:
        atom_pos_rotated: np.ndarray -- positions of atoms in the molecule with wanted rotation applied on them (unit: Angstrom)
        atomic_numbers: np.ndarray -- atomic numbers of atoms in the molecule
        x_pos, y_pos: np.ndarray --  grid_positions on the x- and y-axis (unit: Angstrom)
        wavenumber_range: np.ndarray -- the normal modes of the molecule
        spectrums: np.ndarray -- contains the intensity for every normal mode for every grid position
        filename: str -- the .fchk file that contains molecule data.
    """
    PHI, THETA, PSI = molecule_rotation

    result = load_molecule(filename, PHI, THETA, PSI)
    if result is None:
        return None
    atom_polarizabilities, atoms, frequencies, atom_pos, atom_pos_rotated, atomic_numbers, R = result
    
    if plot_spectrum is not None:
        wavenumber_range = np.array(plot_spectrum)
    else:
        wavenumber_range = frequencies[frequencies >= CUTOFF_FREQUENCY]

    x_pos = np.linspace(-X_WIDTH/2, X_WIDTH/2, X_COUNT)
    y_pos = np.linspace(-Y_WIDTH/2, Y_WIDTH/2, Y_COUNT)
    z = atom_pos[:, 2].max() + 3  # Tips distance from top atom of a molecule, A

    spectrums = np.zeros((X_COUNT, Y_COUNT, wavenumber_range.size))

    for dx in range(X_COUNT):
        for dy in range(Y_COUNT):
            
            # Currently field types 0 and 1 are not used, so condition statement is commented out
            """
            if (E_TYPE == 0):
                mode_intensities, dipoles = \
                    analytic_field(atom_polarizabilities, atoms, frequencies, E_TYPE)
            elif (E_TYPE == 1 or E_TYPE == 2):
                tip_xyz = np.array([x_pos[dx], y_pos[dy], 3])
                mode_intensities, dipoles = \
                    analytic_field(atom_polarizabilities, atoms, frequencies, E_TYPE, atom_pos, R, tip_xyz, tip_width)
            """
            tip_xyz = np.array([x_pos[dx], y_pos[dy], z])
            mode_intensities, dipoles = \
                analytic_field(atom_polarizabilities, atoms, frequencies, E_TYPE, atom_pos, R, tip_xyz, TIP_WIDTH)

            spectrums[dx, dy, :] = spectrum_real(mode_intensities, wavenumber_range, PEAK_WIDTH, LAMBDA_0, T)
    
    for i in range(len(wavenumber_range)):
        spectrums[:, :, i] = spectrums[:, :, i].T

    return atom_pos_rotated, atomic_numbers, x_pos, y_pos, wavenumber_range, spectrums, filename


def save_ters_data(atom_pos, atomic_numbers, x_pos, y_pos, frequencies, spectrums, filename, save_path):
    """
    Saves the data to be able to create a simulated TERS image for a molecule in a text file.
    The data contains more data besides intensity values and one file contains data for all
    normal modes of a molecule.
    """
    data_save_path = save_path / f"{filename.stem}.npz"
    np.savez(data_save_path, atom_pos=atom_pos, atomic_numbers=atomic_numbers, x_pos=x_pos, y_pos=y_pos, frequencies=frequencies, spectrums=spectrums)


def process_fchk_file(args):
    file_path, save_path, log_file, molecule_rotation, plot_spectrum = args
    """
    Generates data to be able to create a simulated TERS image for a molecule and saves the 
    data in a file.
    """

    configure_logging(log_file)
    log_status(f"Started processing file: {file_path.name}")

    
    if file_path.is_file():
        current_rotation = molecule_rotation
        coords, atomic_numbers = read_fchk(file_path)
        if current_rotation is None:
            eigvals, eigvecs, X = pca(coords)
            normal = eigvecs[:, 0]
            current_rotation = normal_to_zyz_ters(normal)
            #_, R = rotation(normal, X)
            #current_rotation = rotation_to_zyz_euler(R)

        image_data = generate_ters_data(file_path, current_rotation, plot_spectrum)
    else:
        image_data = None
        log_status(f"File {file_path.name} does not exist.")

    if image_data is not None:
        atom_pos, atomic_numbers, x_pos, y_pos, frequencies, spectrums, filename = image_data
        save_ters_data(atom_pos, atomic_numbers, x_pos, y_pos, frequencies, spectrums, filename, save_path)
    else:
        log_status(f"Error processing file: {file_path.name}")

    log_status(f"Finished processing file: {file_path.name}")


def generate_data_from_fchk_files(directory_path, save_path, log_file, molecule_rotation, plot_spectrum):
    """
    Generating and saving image data for a group of molecules.
    """

    file_paths = list(directory_path.glob("*.fchk"))
    num_files = len(file_paths)

    t_0 = time.time()
    with multiprocessing.Pool(processes=cpus) as pool:
        with tqdm(total=num_files) as pbar:
            for _ in pool.imap_unordered(process_fchk_file, [(file_path, save_path, log_file, molecule_rotation, plot_spectrum) for file_path in file_paths]):
                pbar.update(1)

    t_1 = time.time()

    print(f"Time for generating data for all the molecules: {t_1 - t_0}")


def configure_logging(log_file=None):
    """
    Configures the logging settings.
    
    Parameters:
    log_file (str): Path to the log file. If None, a default log file will be created in the current directory.
    """
    if not log_file:
        # Create a default log file name with a timestamp
        log_file = os.path.join(os.getcwd(), f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    elif os.path.isdir(log_file):
        # If the provided path is a directory, add a default log file name
        log_file = os.path.join(log_file, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Extract the directory from the log file path
    log_directory = os.path.dirname(log_file)

    if log_directory:  # Check if the directory part of the path is valid
        try:
            # Create the directory if it doesn't exist
            os.makedirs(log_directory, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory {log_directory}: {e}")
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_status(message):
    """
    Updates the log file.
    """
    logging.info(message)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("directory_path", type=str, help="The path to the directory containing the .fchk files.")
    parser.add_argument("save_path", type=str, help="The path to the directory to save the image data.")
    parser.add_argument("log_file", type=str, help="The path of the log file.")
    parser.add_argument(
        "--molecule_rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("PHI", "THETA", "PSI"),
        help="Euler angles in degrees. If omitted, PCA-based auto-rotation is used.",
    )
    parser.add_argument("--plot_spectrum", type=float, nargs='+', default=None, help="List of wavenumbers to plot the spectrum. Using default value calculates spectrum at normal modes of the molecule.")
    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file     
    configure_logging(log_file)

    # Convert string paths to Path objects
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    directory_path = Path(args.directory_path)

    molecule_rotation = args.molecule_rotation
    plot_spectrum = args.plot_spectrum

    # Generating the data for molecules
    generate_data_from_fchk_files(directory_path, save_path, log_file, molecule_rotation, plot_spectrum)
