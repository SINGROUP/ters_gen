#!/usr/bin/env python3
"""
Compute RMSD-based planarity for all .fchk files in a directory and save results to CSV.
Usage:
    python scripts/fchk_rmsd_to_csv.py /path/to/FCHK -o /path/to/out.csv

This re-uses the project's `read_fchk`, `pca`, and `planarity` implementations where possible.
"""
import argparse
import csv
from pathlib import Path
import sys

# Try to import utilities from the repository (notebooks package)
try:
    from notebooks.utils.read_files import read_fchk
    from notebooks.utils.planarity import pca, planarity
except Exception:
    # fallback: local minimal implementations
    import numpy as np

    def read_fchk(fchk_file):
        atomic_numbers = []
        coordinates = []
        num_atoms = 0
        with open(fchk_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('Number of atoms'):
                    num_atoms = int(line.split()[-1])
                elif line.startswith('Atomic numbers'):
                    start_line = i + 1
                    values = []
                    while len(values) < num_atoms and start_line < len(lines):
                        values.extend([int(x) for x in lines[start_line].split()])
                        start_line += 1
                    atomic_numbers = values[:num_atoms]
                elif line.startswith('Current cartesian coordinates'):
                    start_line = i + 1
                    values = []
                    while len(values) < 3 * num_atoms and start_line < len(lines):
                        values.extend([float(x) for x in lines[start_line].split()])
                        start_line += 1
                    coordinates = values[:3 * num_atoms]
        import numpy as _np
        coordinates = _np.array(coordinates).reshape(num_atoms, 3)
        return coordinates, atomic_numbers

    def pca(coords):
        import numpy as _np
        N = coords.shape[0]
        centroid = coords.mean(axis=0)
        X = coords - centroid
        C = (X.T @ X) / N
        eigvals, eigvecs = _np.linalg.eigh(C)
        return eigvals, eigvecs, X

    def planarity(eigvals, eigvecs, X):
        import numpy as _np
        idx = _np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        planarity_pca = 100.0 * (eigvals[0] + eigvals[1]) / eigvals.sum()
        normal = eigvecs[:, 2]
        d = (X @ normal) / _np.linalg.norm(normal)
        rmsd = _np.sqrt(_np.mean(d ** 2))
        L = _np.sqrt(eigvals[0] + eigvals[1])
        planarity_rms = 100.0 * (1 - rmsd / L)
        return planarity_pca, planarity_rms, rmsd


def find_fchk_files(directory: Path):
    return sorted(directory.rglob('*.fchk'))


def process_directory(directory: Path, out_csv: Path, recursive: bool = True):
    files = find_fchk_files(directory)
    if len(files) == 0:
        print(f"No .fchk files found in {directory}")
        return 1

    rows = []
    for f in files:
        try:
            coords, atomic_numbers = read_fchk(str(f))
        except Exception as e:
            rows.append({
                'path': str(f), 'name': f.name, 'n_atoms': None,
                'rmsd_A': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': str(e)
            })
            continue

        if coords is None or coords.size == 0:
            rows.append({
                'path': str(f), 'name': f.name, 'n_atoms': 0,
                'rmsd_A': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': 'no-coordinates'
            })
            continue

        try:
            eigvals, eigvecs, X = pca(coords)
            plan_pca, plan_rms, rmsd = planarity(eigvals, eigvecs, X)
            rows.append({
                'path': str(f), 'name': f.name, 'n_atoms': int(coords.shape[0]),
                'rmsd_A': float(rmsd), 'planarity_pca_pct': float(plan_pca),
                'planarity_rms_pct': float(plan_rms), 'error': ''
            })
        except Exception as e:
            rows.append({
                'path': str(f), 'name': f.name, 'n_atoms': int(coords.shape[0]),
                'rmsd_A': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': str(e)
            })

    # write CSV
    fieldnames = ['path', 'name', 'n_atoms', 'rmsd_A', 'planarity_pca_pct', 'planarity_rms_pct', 'error']
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} records to {out_csv}")
    return 0


def compute_row(fchk_file: Path):
    """Return a dict (row) with computed values for a single .fchk file (no I/O).

    This is safe for multiprocessing workers.
    """
    try:
        coords, atomic_numbers = read_fchk(str(fchk_file))
    except Exception as e:
        return {'path': str(fchk_file), 'name': fchk_file.name, 'n_atoms': None,
                'rmsd_A': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': str(e)}

    if coords is None or coords.size == 0:
        return {'path': str(fchk_file), 'name': fchk_file.name, 'n_atoms': 0,
                'rmsd_A': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': 'no-coordinates'}

    try:
        eigvals, eigvecs, X = pca(coords)
        plan_pca, plan_rms, rmsd = planarity(eigvals, eigvecs, X)
        return {'path': str(fchk_file), 'name': fchk_file.name, 'n_atoms': int(coords.shape[0]),
                'rmsd_A': float(rmsd), 'planarity_pca_pct': float(plan_pca),
                'planarity_rms_pct': float(plan_rms), 'error': ''}
    except Exception as e:
        return {'path': str(fchk_file), 'name': fchk_file.name, 'n_atoms': int(coords.shape[0]),
                'rmsd_A': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': str(e)}


def process_file(fchk_file: Path, out_csv: Path):
    """Process a single .fchk file and append a one-line CSV (creates parent dir)."""
    row = compute_row(Path(fchk_file))

    fieldnames = ['path', 'name', 'n_atoms', 'rmsd_A', 'planarity_pca_pct', 'planarity_rms_pct', 'error']
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not out_csv.exists()
    with open(out_csv, 'a', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)
    return 0


def process_files(file_paths, out_csv: Path, workers: int = 1):
    """Compute rows for a list of files in parallel (workers) and write a single CSV."""
    import multiprocessing as mp

    file_paths = [Path(f) for f in file_paths]
    if len(file_paths) == 0:
        print("No files to process")
        return 1

    if workers is None or workers <= 1:
        rows = [compute_row(f) for f in file_paths]
    else:
        with mp.Pool(processes=workers) as pool:
            rows = pool.map(compute_row, file_paths)

    # write CSV (single header)
    fieldnames = ['path', 'name', 'n_atoms', 'rmsd_A', 'planarity_pca_pct', 'planarity_rms_pct', 'error']
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} records to {out_csv}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Compute RMSD planarity for .fchk files and save CSV')
    parser.add_argument('directory', nargs='?', type=Path, help='Directory containing .fchk files (positional)')
    parser.add_argument('--file', type=Path, help='Single .fchk file to process')
    parser.add_argument('--file-list', type=Path, help='Text file with one .fchk path per line')
    parser.add_argument('--num-chunks', type=int, default=1, help='Split file-list into this many chunks')
    parser.add_argument('--chunk-index', type=int, default=1, help='1-based chunk index to process (1..num-chunks)')
    parser.add_argument('--workers', type=int, default=1, help='Number of local worker processes to use (multiprocessing)')
    parser.add_argument('-o', '--out', type=Path, default=None, help='Output CSV path')
    args = parser.parse_args()

    # single-file mode
    if args.file:
        out_csv = args.out or (Path(args.file).parent / (Path(args.file).stem + '.rmsd.csv'))
        return process_file(args.file, out_csv)

    # file-list mode (useful for chunked SLURM jobs)
    if args.file_list:
        lines = args.file_list.read_text().splitlines()
        files = [l.strip() for l in lines if l.strip()]
        if len(files) == 0:
            print(f"No files listed in {args.file_list}")
            return 1
        num_chunks = max(1, args.num_chunks)
        if args.chunk_index < 1 or args.chunk_index > num_chunks:
            raise SystemExit(f"chunk-index must be in 1..{num_chunks}")
        # distribute by modulo to evenly spread work across chunks
        selected = [f for i, f in enumerate(files) if (i % num_chunks) == (args.chunk_index - 1)]
        out_csv = args.out or (Path(args.file_list).parent / f'task_{args.chunk_index}.csv')
        return process_files(selected, out_csv, workers=args.workers)

    # directory mode (can also be chunked)
    if args.directory:
        files = [str(p) for p in find_fchk_files(args.directory)]
        if len(files) == 0:
            print(f"No .fchk files found in {args.directory}")
            return 1
        num_chunks = max(1, args.num_chunks)
        if num_chunks == 1:
            out_csv = args.out or (args.directory / 'rmsd_planarity_summary.csv')
            return process_files(files, out_csv, workers=args.workers)
        if args.chunk_index < 1 or args.chunk_index > num_chunks:
            raise SystemExit(f"chunk-index must be in 1..{num_chunks}")
        selected = [f for i, f in enumerate(files) if (i % num_chunks) == (args.chunk_index - 1)]
        out_csv = args.out or (Path(args.directory) / f'task_{args.chunk_index}.csv')
        return process_files(selected, out_csv, workers=args.workers)

    parser.print_usage()
    return 2


if __name__ == '__main__':
    sys.exit(main())
