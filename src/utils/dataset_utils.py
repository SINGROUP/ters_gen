import numpy as np

from src.covalent_radii_new import covalent_radii_new

def compute_bonds_new(atom_pos, atomic_numbers, cutoff_scale=1.24):  # Correct 1.24
        """
        Compute bonds between atoms based on a cutoff determined from covalent radii.

        Args:
            atom_pos (np.ndarray): Positions of atoms.
            atomic_numbers (np.ndarray): Atomic numbers for each atom.
            cutoff_scale (float): Scale factor for determining cutoff distance.

        Returns:
            set: A set of tuples representing bonds (atomic_number_i, atomic_number_j).
        """
        num_atoms = len(atomic_numbers)
        bonds = set()
                
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = np.linalg.norm(atom_pos[i] - atom_pos[j])
                
                # Calculate cutoff distance based on covalent radii
                cutoff = cutoff_scale * (covalent_radii_new[atomic_numbers[i]] + covalent_radii_new[atomic_numbers[j]])
                
                if distance <= cutoff:
                    bonds.add((atomic_numbers[i], atomic_numbers[j]))
        
        return bonds
    

def _get_element_flags(bonds: set) -> tuple:
    """
    Determine element presence based on the bonds.
    Returns:
        tuple: (contains_carbon, contains_hydrogen, contains_nitrogen, contains_oxygen)
    """
    contains_c = contains_h = contains_n = contains_o = False
    for bond in bonds:
        if bond == (6, 1):
            contains_c = True
            contains_h = True
        elif bond == (6, 6):
            contains_c = True
        elif bond == (7, 1):
            contains_n = True
            contains_h = True
        elif bond == (7, 6):
            contains_n = True
            contains_c = True
        elif bond == (7, 7):
            contains_n = True
        elif bond == (8, 1):
            contains_o = True
            contains_h = True
        elif bond == (8, 6):
            contains_o = True
            contains_c = True
        elif bond == (8, 7):
            contains_o = True
            contains_n = True
        elif bond == (8, 8):
            contains_o = True
    return contains_c, contains_h, contains_n, contains_o