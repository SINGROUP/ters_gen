#!/usr/bin/env python3
"""Simple parallel RMSD/planarity exporter for .fchk files

- Uses multiprocessing.Pool (workers) to process files in parallel.
- Reuses repo helpers when available; falls back to local readers.
- Example:
    python3 scripts/fchk_rmsd_parallel_simple.py /path/to/FCHK -o /path/to/out.csv --workers 16 --bohr
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from functools import partial
from pathlib import Path
import sys
import time
from typing import Dict

import numpy as np

# --- try to reuse repository utilities if available ---
try:
    from notebooks.utils.read_files import read_fchk
    from notebooks.utils.planarity import pca, planarity
except Exception:
    # minimal fallbacks (copied from project implementations)
    def read_fchk(fchk_file: str):
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
        coordinates = np.array(coordinates).reshape(num_atoms, 3)
        return coordinates, atomic_numbers

    def pca(coords: np.ndarray):
        N = coords.shape[0]
        centroid = coords.mean(axis=0)
        X = coords - centroid
        C = (X.T @ X) / N
        eigvals, eigvecs = np.linalg.eigh(C)
        return eigvals, eigvecs, X

    def planarity(eigvals: np.ndarray, eigvecs: np.ndarray, X: np.ndarray):
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        planarity_pca = 100.0 * (eigvals[0] + eigvals[1]) / eigvals.sum()
        normal = eigvecs[:, 2]
        d = (X @ normal) / np.linalg.norm(normal)
        rmsd = np.sqrt(np.mean(d ** 2))
        L = np.sqrt(eigvals[0] + eigvals[1])
        planarity_rms = 100.0 * (1 - rmsd / L)
        return planarity_pca, planarity_rms, rmsd


BOHR_TO_ANG = 0.52917721092

# global flag initialized in main (read-only for workers)
_GLOBAL_BOHR = False


def _compute_row(fpath: str) -> Dict:
    """Compute one row given a file path (string). Safe for Pool workers."""
    try:
        coords, atomic_numbers = read_fchk(fpath)
    except Exception as e:
        return {'path': fpath, 'name': Path(fpath).name, 'n_atoms': None,
                'rmsd': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': f'read-error: {e}'}

    if coords is None or coords.size == 0:
        return {'path': fpath, 'name': Path(fpath).name, 'n_atoms': 0,
                'rmsd': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': 'no-coordinates'}

    coords = np.asarray(coords, dtype=float)
    if _GLOBAL_BOHR:
        coords = coords * BOHR_TO_ANG

    try:
        eigvals, eigvecs, X = pca(coords)
        plan_pca, plan_rms, rmsd = planarity(eigvals, eigvecs, X)
        return {'path': fpath, 'name': Path(fpath).name, 'n_atoms': int(coords.shape[0]),
                'rmsd': float(rmsd), 'planarity_pca_pct': float(plan_pca),
                'planarity_rms_pct': float(plan_rms), 'error': ''}
    except Exception as e:
        return {'path': fpath, 'name': Path(fpath).name, 'n_atoms': int(coords.shape[0]),
                'rmsd': None, 'planarity_pca_pct': None, 'planarity_rms_pct': None,
                'error': f'proc-error: {e}'}


def main():
    global _GLOBAL_BOHR
    parser = argparse.ArgumentParser(description='Simple multiprocessing RMSD/planarity exporter for .fchk')
    parser.add_argument('directory', type=Path, help='Directory containing .fchk files')
    parser.add_argument('-o', '--out', type=Path, default=None, help='Output CSV path')
    parser.add_argument('-w', '--workers', type=int, default=max(1, mp.cpu_count() // 2), help='Number of worker processes')
    parser.add_argument('--bohr', action='store_true', help='Convert coordinates from Bohr to Angstrom')
    parser.add_argument('--chunksize', type=int, default=16, help='Pool.imap_unordered chunksize')
    args = parser.parse_args()

    fdir = args.directory
    if not fdir.exists():
        print(f'Directory not found: {fdir}', file=sys.stderr); return 2

    files = sorted([str(p) for p in fdir.rglob('*.fchk')])
    if not files:
        print(f'No .fchk files found in {fdir}')
        return 1

    out_csv = args.out or (fdir / 'rmsd_planarity_parallel.csv')
    workers = max(1, args.workers)
    _GLOBAL_BOHR = bool(args.bohr)

    total = len(files)
    status_every = max(1, total // 100)

    print(f'Processing {total} files with {workers} workers (bohr_to_ang={_GLOBAL_BOHR})')
    start = time.time()

    rows = []
    try:
        with mp.Pool(processes=workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_compute_row, files, chunksize=args.chunksize), start=1):
                rows.append(row)
                if (i % status_every) == 0 or i == total:
                    elapsed = time.time() - start
                    print(f'[{i}/{total}] elapsed={elapsed:.0f}s')
    except KeyboardInterrupt:
        print('Interrupted by user, terminating workers...', file=sys.stderr)
        pool.terminate()
        pool.join()
        raise

    # write CSV
    fieldnames = ['path', 'name', 'n_atoms', 'rmsd', 'planarity_pca_pct', 'planarity_rms_pct', 'error']
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    elapsed = time.time() - start
    print(f'Done — wrote {len(rows)} rows to {out_csv} (elapsed {elapsed:.0f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
