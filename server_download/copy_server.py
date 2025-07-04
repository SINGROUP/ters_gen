import os
import sys
import ctypes
import time
from multiprocessing import Pool, Value

import numpy as np
import requests

# Disable SSL warnings (not recommended for production)
requests.packages.urllib3.disable_warnings()

# Shared global counter for logging progress
screened_counter = None

# Map atomic numbers to chemical symbols
ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
    7: 'N', 8: 'O', 9: 'F', 10: 'Ne'
    # Extend as needed
}

class CIDHelper:
    """
    Helper class to interact with remote CCSD database and fetch molecules.
    Implements retry logic on timeouts.
    """
    SERVER = 'https://69.12.4.38/CCSD'
    CIDs = []

    @staticmethod
    def process_response(rsp):
        if rsp.status_code != 200:
            print(f"Error: Unable to connect to server (status {rsp.status_code})")
            return None
        data = rsp.json()
        if data.get('type') == 'error':
            print("Error response from server:", data.get('message', '<no message>'))
            return None
        return data

    @classmethod
    def init(cls, timeout=10, retries=2, backoff=2):
        """Fetch and store all available CIDs with retry on failure."""
        url = f"{cls.SERVER}/CIDs"
        for attempt in range(retries + 1):
            try:
                rsp = requests.get(url, verify=False, timeout=timeout)
                data = cls.process_response(rsp)
                if data and 'CIDs' in data:
                    cls.CIDs = data['CIDs']
                    return
                print("No CIDs retrieved.")
                return
            except requests.RequestException as e:
                print(f"Attempt {attempt+1}/{retries+1} failed: {e}")
                if attempt < retries:
                    time.sleep(backoff)
        print("Failed to fetch CIDs after retries.")

    @staticmethod
    def get_xyz(cid, timeout=10, retries=2, backoff=2):
        """Download atomic numbers (Z) and coordinates (xyz) for the given CID with retries."""
        url = f"{CIDHelper.SERVER}/molecule/xyz/{cid}"
        for attempt in range(retries + 1):
            try:
                rsp = requests.get(url, verify=False, timeout=timeout)
                data = CIDHelper.process_response(rsp)
                if not data:
                    return None
                data['Z'] = np.asarray(data['Z'], dtype=int)
                data['xyz'] = np.asarray(data['xyz'], dtype=float)
                data['CID'] = cid
                return data
            except (requests.RequestException, ValueError) as e:
                print(f"CID {cid} attempt {attempt+1}/{retries+1} failed: {e}")
                if attempt < retries:
                    time.sleep(backoff)
        print(f"Failed to fetch data for CID {cid} after retries.")
        return None


def pca(coords):
    N = coords.shape[0]
    centroid = coords.mean(axis=0)
    X = coords - centroid
    cov = (X.T @ X) / N
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs, X


def planarity(eigvals, eigvecs, X):
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    planarity_pca = 100 * (eigvals[0] + eigvals[1]) / eigvals.sum()
    normal = eigvecs[:, 2]
    distances = (X @ normal) / np.linalg.norm(normal)
    rmsd = np.sqrt(np.mean(distances**2))
    span = np.sqrt(eigvals[0] + eigvals[1])
    planarity_rms = 100 * (1 - rmsd / span)

    return planarity_pca, planarity_rms, rmsd


def print_gaussian_input(cid, molecule_data, nproc=32, method_line="# PBEPBE/6-31G(d) NoSymm freq=raman"):
    Z = molecule_data['Z']
    xyz = molecule_data['xyz']
    lines = [
        f"%nproc={nproc}",
        f"%chk={cid}.chk",
        method_line,
        "",
        f"This is molecule {cid}",
        "",
        "0 1"
    ]
    for atomic_num, coord in zip(Z, xyz):
        symbol = ATOMIC_SYMBOLS.get(atomic_num, str(atomic_num))
        x, y, z = coord
        lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
    lines.append("")
    return "\n".join(lines)


def _init_worker(counter):
    global screened_counter
    screened_counter = counter


def process_cid(args):
    cid, max_rmsd, max_atoms, min_atoms, print_interval = args
    global screened_counter

    with screened_counter.get_lock():
        screened_counter.value += 1
        if screened_counter.value % print_interval == 0:
            print(f"Screened {screened_counter.value} molecules...")

    mol = CIDHelper.get_xyz(cid)
    if mol is None:
        return (cid, False, False, None, None)

    n_atoms = mol['Z'].shape[0]
    in_range = (min_atoms <= n_atoms <= max_atoms)
    if not in_range:
        return (cid, False, False, None, None)

    eigvals, eigvecs, X = pca(mol['xyz'])
    _, _, rmsd = planarity(eigvals, eigvecs, X)

    if rmsd <= max_rmsd:
        return (cid, True, True, None, rmsd)
    return (cid, True, False, None, None)


def main(max_rmsd=0.1, max_atoms=60, min_atoms=50,
         outdir="/scratch/phys/sin/sethih1/planar_molecules_gaussian",
         nproc=os.cpu_count(), print_interval=100):
    global screened_counter
    screened_counter = Value(ctypes.c_int, 0)

    CIDHelper.init()
    if not CIDHelper.CIDs:
        return

    total_molecules = len(CIDHelper.CIDs)
    os.makedirs(outdir, exist_ok=True)
    print(f"Using {nproc} CPUs to screen {total_molecules} molecules with atom range [{min_atoms}, {max_atoms}] and RMSD ≤ {max_rmsd}")

    args = [
        (cid, max_rmsd, max_atoms, min_atoms, print_interval)
        for cid in CIDHelper.CIDs
    ]

    with Pool(processes=nproc, initializer=_init_worker, initargs=(screened_counter,)) as pool:
        results = pool.map(process_cid, args)

    in_range_count = sum(1 for _, in_range, _, _, _ in results if in_range)
    planar = [r for r in results if r[2]]

    print(f"\nMolecules in atom range: {in_range_count}")
    print(f"Total molecules screened: {total_molecules}")
    print(f"Total planar molecules (RMSD ≤ {max_rmsd}): {len(planar)}")

    for cid, _, _, _, rmsd in planar:
        print(f"Found planar CID: {cid} | RMSD={rmsd:.4f}")
        mol = CIDHelper.get_xyz(cid)
        out_path = os.path.join(outdir, f"{cid}.com")
        with open(out_path, "w") as f:
            f.write(print_gaussian_input(cid, mol))

if __name__ == "__main__":
    main()
