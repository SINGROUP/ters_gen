# Raman Spectral Image Generator

## Overview
The repository provides python code converted from original MATLAB based code[J Raman Spectrosc. 52:296–309 (2021)]

The repository can be used to simulate the off-Resonant Stokes Raman spectra of a single-molecule in any inhomogeneous field, typically in a field whose spatial distribution is approximated by a Gaussian function. Thus, it is also an approximate model to simulate tip-enhanced Raman scattering (TERS) spectra of a single-molecule in a nanocavity plasmon (NCP) field induced by the tip-substrate system. 


In this program, the inhomogeneous field can be described by a Gaussian function or any other user-defined real field. It should be noticed that the scattered field is processed in a simple way following the optical reciprocity theorem, and the "true" radiation process can be introduced through the farfield Green's function, which will be included in future versions of the program. 

The necessary information on the single-molecule vibrations is read from the output file (.fch or .fchk) calculated by the Gaussian package. The program reads molecular data files, computes Raman spectra over a spatial grid, and saves the resulting spectral images. 

## Features

- **Molecular Spectral Analysis**: Compute Raman spectra using atomic polarizabilities and vibrational frequencies
- **Grid Mapping**: Generate spectral data over a configurable spatial grid
- **Batch Processing**: Automatically process multiple molecular files in a directory
- **Visualization**: Save spectral data as contour plots for each vibrational frequency
- **Customizable Parameters**: Easily adjust grid size, spatial resolution, and peak widths

## File Structure

```
.
├── core/                        # Core simulation logic
│   ├── analytic_field.py        # Computes vibrational mode intensities
│   ├── generate_spectrum.py     # Wrapper for spectrum generation
│   ├── load_molecule.py         # Loads molecular attributes
│   ├── read_gaussian.py         # Module to parse molecular data from .fchk files
│   └── spectrum_real.py         # Calculates Raman spectra
├── scripts/                     # Executable Python scripts
│   ├── point_spectrum_generation.py # Main script for generating spectral images
│   └── log_reading.py           # Utility to parse log files
├── jobs/                        # Shell scripts for batch execution
│   ├── generalized_run.sh
│   ├── multiple_run_generation.bash
│   └── ...
├── notebooks/                   # Jupyter notebooks for analysis and testing
│   └── testing.ipynb
├── extra_functions/             # Additional utilities and notebooks
│   ├── molecule_rotation.py
│   └── ...
└── utils/                       # Shared utilities
    └── utils.py
```

## Usage

### 1. Prepare Input Files

Store molecular `.fchk` files in a directory.

### 2. Run the Script

Execute the main script from the `scripts` directory:

```bash
python scripts/point_spectrum_generation.py <directory_path> <save_path> <log_file>
```

Arguments:
- `directory_path`: Path to the directory containing the `.fchk` files.
- `save_path`: Path to the directory to save the image data.
- `log_file`: Name of the log file (not the whole path).

### 3. Output

The script generates data that can be used to create simulated TERS images.

## Customization

You can adjust parameters in the `core/` modules or the `scripts/point_spectrum_generation.py` file.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `x_count` | 51 | Number of grid points along the x-axis |
| `y_count` | 51 | Number of grid points along the y-axis |
| `x_width` | 18 | Spatial extent (Å) along the x-axis |
| `y_width` | 18 | Spatial extent (Å) along the y-axis |
| `peak_width` | 5 | Width of spectral peaks |
| `lambda_0` | 532 | Wavelength of the incident light (nm) |
| `T` | 1e-6 | Temperature in Kelvin |

## Performance Optimization

### Parallel Processing

The code supports parallelizing grid computations.

### Logging

For better control over output, replace `print` statements with the `logging` module.


