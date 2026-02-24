# TERS Image Simulator

Python tools for generating simulated TERS image data from Gaussian `.fchk` files.

## Overview

This package reads vibrational and polarizability information from Gaussian formatted checkpoint files, computes spatially resolved Raman intensity on an `(x, y)` grid, and writes one `.npz` output per molecule.

This implementation is adapted from an original MATLAB workflow for single-molecule Raman/TERS simulation[J Raman Spectrosc. 52:296–309 (2021)].

The simulator targets off-resonant Stokes Raman response in an inhomogeneous near field and is commonly used as an approximate TERS image generator for tip-substrate nanocavity conditions.

The current model combines:

- Vibrational normal mode and polarizability-derivative data parsed from Gaussian `.fchk` files.
- A near-field treatment driven by `analytic_field.py` with `E_TYPE`-controlled behavior (default in current script: `E_TYPE=2`).
- Spectrum construction through mode intensities and broadening in `spectrum_real.py`.

Important assumptions/scope in current implementation:

- Near-field spatial dependence is approximated in code through the selected field model and `TIP_WIDTH`.
- Scattering/radiation handling is simplified relative to full electrodynamic far-field Green-function treatments.

## What this package can do

- Batch process folders of Gaussian `.fchk` files.
- Generate dense spatial spectral tensors on a configurable `(x, y)` grid.
- Save per-molecule simulation outputs in `.npz` for downstream ML/data-analysis pipelines.
- Run in parallel across molecules via multiprocessing (`SLURM_CPUS_PER_TASK` aware).


## Current project structure

```text
ters_img_simulator/
├── core/
│   ├── analytic_field.py
│   ├── generate_spectrum.py
│   ├── load_molecule.py
│   ├── read_gaussian.py
│   └── spectrum_real.py
├── scripts/
│   ├── log_reading.py
│   └── point_spectrum_generation.py
└── utils/
    └── utils.py
```


## Usage

Run from repository root (`/scratch/work/sethih1/ters_gen`):

```bash
python ters_img_simulator/scripts/point_spectrum_generation.py \
  <directory_path> \
  <save_path> \
  <log_file_or_log_dir>
```

Arguments:
- `directory_path`: folder containing `.fchk` files.
- `save_path`: folder where `.npz` outputs are written.
- `log_file`: either
  - a log filename (e.g. `run.log`),
  - a full/relative log path (e.g. `logs/run.log`), or
  - a directory path (e.g. `logs/`) to auto-create timestamped log files.

Optional arguments:

```bash
--molecule_rotation <phi theta psi>
--plot_spectrum <w1 w2 w3 ...>
```

Rotation behavior:

- If `--molecule_rotation` is provided, those Euler angles are used directly.
- If `--molecule_rotation` is omitted, the script computes a PCA-based normal from atomic coordinates and auto-rotates the molecule so that normal aligns with the tip-axis convention used by the simulator.

Example:

```bash
python ters_img_simulator/scripts/point_spectrum_generation.py \
  /path/to/fchk_dir \
  /path/to/output_npz \
  /path/to/logs/
```

## Output format

For each input `<name>.fchk`, the script writes:

- `<save_path>/<name>.npz`

Each `.npz` contains:
- `atom_pos`: rotated atomic positions
- `atomic_numbers`
- `x_pos`
- `y_pos`
- `frequencies`
- `spectrums` (shape: `X_COUNT x Y_COUNT x N_modes`)

## Simulation defaults (current code)

Defined in `scripts/point_spectrum_generation.py`:

- `X_COUNT, Y_COUNT = 256, 256`
- `X_WIDTH, Y_WIDTH = 18, 18` (Angstrom)
- `TIP_WIDTH = [5, 5, 5]`
- `PEAK_WIDTH = 5`
- `LAMBDA_0 = 532`
- `T = 1e-6`
- `E_TYPE = 2`
- Multiprocessing workers:
  - `SLURM_CPUS_PER_TASK` when available
  - otherwise `os.cpu_count()`

## Logging and troubleshooting

- Use `scripts/log_reading.py` to inspect unfinished or failed files:

```bash
python ters_img_simulator/scripts/log_reading.py 0 <log_file>  # unfinished
python ters_img_simulator/scripts/log_reading.py 1 <log_file>  # errors
```
