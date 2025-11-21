import sys
sys.path.append('/home/sethih1/masque_new/ters_gen')


from notebooks.utils.planarity import pca,  planarity
from notebooks.utils.read_files import read_fchk, read_npz


from pathlib import Path
import os
import shutil




def planar_molecule_finder(input_dir, output_dir, rmsd_thr, ext = 'npz'):
    

    read_func = read_npz if ext == 'npz' else read_fchk
    dir = Path(input_dir)
    files = list(dir.rglob("*." + ext))

    count = 0

    os.makedirs(output_dir, exist_ok=True)

    for file in files:

        coords, atomic_numbers = read_func(file)
        if len(atomic_numbers) == 0 or len(coords) == 0:
            print(file)
            continue

        eigvals, eigvecs, X = pca(coords)
        planarity_pca, planarity_rms, rmsd = planarity(eigvals, eigvecs, X)

        if rmsd <= rmsd_thr:
            shutil.copy(file, os.path.join(output_dir, os.path.basename(file)))
            count += 1

        
    print("Planar molecules found")
    print(f"Total_files: {len(files)}, Planar files: {count}")




if __name__ == "__main__":


    input_dir = "/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_again/planar_npz_1.0_again"
    ext = 'npz'

    
    rmsd_thrs = [0.05, 0.1]

    for rmsd_thr in rmsd_thrs:

        output_dir = f'/scratch/phys/sin/sethih1/Extended_TERS_data/planar_oct_2025/planar_again/planar_npz_{rmsd_thr}'

        planar_molecule_finder(input_dir, output_dir, rmsd_thr, ext)