import base64
import io
import os
import sys
import numpy as np
import requests

# Disable SSL warnings (not recommended for production)
requests.packages.urllib3.disable_warnings()

# Map atomic numbers to chemical symbols
ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
    7: 'N', 8: 'O', 9: 'F', 10: 'Ne'
    # Extend as needed
}

class CIDHelper:
    """
    Helper class to interact with remote CCSD database and fetch molecules.
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
    def init(cls):
        """Fetch and store all available CIDs."""
        url = f"{cls.SERVER}/CIDs"
        rsp = requests.get(url, verify=False)
        data = cls.process_response(rsp)
        if data and 'CIDs' in data:
            cls.CIDs = data['CIDs']
        else:
            print("No CIDs retrieved.")

    @classmethod
    def get_xyz(cls, cid):
        """Download atomic numbers (Z) and coordinates (xyz) for the given CID."""
        if cid not in cls.CIDs:
            print(f"CID {cid} not available.")
            return None
        url = f"{cls.SERVER}/molecule/xyz/{cid}"
        rsp = requests.get(url, verify=False)
        data = cls.process_response(rsp)
        if not data:
            return None
        try:
            data['Z'] = np.asarray(data['Z'], dtype=int)
            data['xyz'] = np.asarray(data['xyz'], dtype=float)
        except Exception as e:
            print(f"Error parsing XYZ data for CID {cid}: {e}")
            return None
        data['CID'] = cid
        return data


def pca(coords):
    """
    Perform PCA on centered coordinates.
    Returns eigenvalues, eigenvectors, and centered data matrix X.
    """
    N = coords.shape[0]
    centroid = coords.mean(axis=0)
    X = coords - centroid
    cov = (X.T @ X) / N
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs, X


def planarity(eigvals, eigvecs, X):
    """
    Compute planarity metrics:
      - planarity_pca: % variance in first two PCs
      - planarity_rms: % flatness relative to planar span
      - rmsd: root-mean-square deviation from best-fit plane
    """
    # Sort eigenvalues/vectors descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # PCA-based planarity: variance explained by first two components
    planarity_pca = 100 * (eigvals[0] + eigvals[1]) / eigvals.sum()

    # Compute RMSD from the best-fit plane
    normal = eigvecs[:, 2]
    distances = (X @ normal) / np.linalg.norm(normal)
    rmsd = np.sqrt(np.mean(distances**2))
    span = np.sqrt(eigvals[0] + eigvals[1])
    planarity_rms = 100 * (1 - rmsd / span)

    return planarity_pca, planarity_rms, rmsd


def print_gaussian_input(cid, molecule_data, nproc=32, method_line="# PBEPBE/6-31G(d) NoSymm freq=raman"):
    """
    Print Gaussian .com input for a molecule to stdout.
    """
    Z = molecule_data['Z']
    xyz = molecule_data['xyz']
    print(f"%nproc={nproc}")
    print(f"%chk={cid}.chk")
    print(method_line, end="\n\n")
    print(f"This is molecule {cid}\n")
    print("0 1")  # charge and multiplicity
    for atomic_num, coord in zip(Z, xyz):
        symbol = ATOMIC_SYMBOLS.get(atomic_num, str(atomic_num))
        x, y, z = coord
        print(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
    print()


def main(threshold_pca=99.0, threshold_rms=98.0, max_rmsd=0.1, max_atoms=50, min_atoms=30):
    """
    Main routine to download, filter for planar molecules, and write Gaussian input files.
    Filters out molecules with <= max_atoms atoms.
    """
    CIDHelper.init()
    if not CIDHelper.CIDs:
        return

    #os.makedirs("planar_inputs", exist_ok=True)
    planar_cids = []

    for cid in CIDHelper.CIDs:
        mol = CIDHelper.get_xyz(cid)
        if mol is None:
            continue
        # Exclude small molecules (<= max_atoms)
        if mol['Z'].shape[0] > max_atoms and mol['Z'].shape[0] < min_atoms:
            continue
        # Only light atoms (H through O)
        if not all(num < 9 for num in mol['Z']):
            continue

        eigvals, eigvecs, X = pca(mol['xyz'])
        pca_score, rms_score, rmsd = planarity(eigvals, eigvecs, X)

        if pca_score >= threshold_pca and rms_score >= threshold_rms and rmsd <= max_rmsd:
            planar_cids.append(cid)
            print(f"Found planar CID: {cid} | PCA={pca_score:.2f}% | RMSD={rmsd:.4f}")
            out_path = os.path.join("/scratch/phys/sin/sethih1/planar_molecules_gaussian/", f"{cid}.com")
            with open(out_path, "w") as f:
                sys.stdout = f
                print_gaussian_input(cid, mol)
                sys.stdout = sys.__stdout__

    print(f"\nTotal planar molecules: {len(planar_cids)}. Files saved in '/scratch/phys/sin/sethih1/planar_molecules_gaussian/' directory.")

if __name__ == "__main__":
    main()
