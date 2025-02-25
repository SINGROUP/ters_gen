import os
import glob
import numpy as np
import time as time

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torch.nn as nn


from src.covalent_radii import covalent_radii_new



# For visualization image
import matplotlib.pyplot as plt
from ase import Atoms
from ase.visualize.plot import plot_atoms
from io import BytesIO
from PIL import Image


# Dictionary to convert atomic number to atomic symbols
atomic_symbols = {
    1: "H",   # Hydrogen
    2: "He",  # Helium
    3: "Li",  # Lithium
    4: "Be",  # Beryllium
    5: "B",   # Boron
    6: "C",   # Carbon
    7: "N",   # Nitrogen
    8: "O",   # Oxygen
    9: "F",   # Fluorine
    10: "Ne", # Neon
    11: "Na", # Sodium
    12: "Mg", # Magnesium
    13: "Al", # Aluminum
    14: "Si", # Silicon
    15: "P",  # Phosphorus
    16: "S",  # Sulfur
    17: "Cl", # Chlorine
    18: "Ar", # Argon
    19: "K",  # Potassium
    20: "Ca", # Calcium
    21: "Sc", # Scandium
    22: "Ti", # Titanium
    23: "V",  # Vanadium
    24: "Cr", # Chromium
    25: "Mn", # Manganese
    26: "Fe", # Iron
    27: "Co", # Cobalt
    28: "Ni", # Nickel
    29: "Cu", # Copper
    30: "Zn", # Zinc
    31: "Ga", # Gallium
    32: "Ge", # Germanium
    33: "As", # Arsenic
    34: "Se", # Selenium
    35: "Br", # Bromine
    36: "Kr", # Krypton
    37: "Rb", # Rubidium
    38: "Sr", # Strontium
    39: "Y",  # Yttrium
    40: "Zr", # Zirconium
}


# Define a fixed color map for the four elements
color_map = {
    "C": "black",    # Carbon
    "H": "lightgray", # Hydrogen
    "O": "red",      # Oxygen
    "N": "blue"      # Nitrogen
}


def padding(spectrums, frequencies):
    freq_pad = torch.zeros(54) # 54 is the maximum number of frequency modes possible for a molecule with 20 atoms
    img_pad = torch.zeros((54, 64, 64)) # 54 images with 64x64 pixels

    # Pad the images and frequencies to have the same length
    img_pad[:spectrums.shape[0], :, :] = spectrums
    freq_pad[:len(frequencies)] = frequencies

    return img_pad, freq_pad

# Useful function
def xyz_string_to_image_array(xyz_string, img_size=(64, 64)):
    """Convert XYZ data (string) to a NumPy image array of size 224×224."""
    lines = xyz_string.strip().split("\n")[2:]  # Skip first two lines (comment and atom count)
    
    elements = []
    positions = []
    
    for line in lines:
        parts = line.split()
        elements.append(parts[0])  # First entry is the element
        positions.append([float(x) for x in parts[1:4]])  # Next three are coordinates
    
    positions = np.array(positions)
    mol = Atoms(elements, positions=positions)
    
    atom_colors = [color_map.get(atom, "gray") for atom in mol.get_chemical_symbols()]  # Assign colors
    
    # Plot molecule
    fig, ax = plt.subplots(figsize=(3, 3))
    plot_atoms(mol, ax, radii=0.3, colors=atom_colors)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Save image to an in-memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Convert image to 224×224 NumPy array
    buf.seek(0)
    image = Image.open(buf).convert("RGB")  # Ensure it's RGB
    image = image.resize(img_size, Image.LANCZOS)  # Resize to 224×224
    image_array = np.array(image)  # Convert to NumPy array (H, W, C)
    image_array = image_array/255
    
    return image_array  # Directly return NumPy array



class Ters_dataset_filtered_skip(Dataset):
    def __init__(self, filename, frequency_range, std_deviation_multiplier=2, t_image=None, t_freq=None):
        super(Ters_dataset_filtered_skip, self).__init__()
        self.filename = filename
        self.frequency_range = frequency_range
        self.t_image = t_image
        self.t_freq = t_freq

        # Use glob to get a list of all .npz files in the directory
        file_pattern = os.path.join(filename, '*.npz')
        npz_files = sorted(glob.glob(file_pattern))

        # self.length = len(npz_files)
        self.length_counter = 0

        self.molecules = len(npz_files)
        self.unique_bonds = set()
        self.bonds = []
        self.frequencies = []
        self.images = []

        self.skipped_count = 0
        self.atom_pos = []
        self.atomic_numbers = []
        self.names = []

        self.min_length = float('inf')  # Initialize the minimum length
        self.max_length = 0
        lengths = []  # List to store the lengths of images for each molecule
        self.target_images = []

        for npz_file in npz_files:
            self.names.append(os.path.basename(npz_file))
            with np.load(npz_file) as data:
                atom_pos = data['atom_pos']
                atomic_numbers = data['atomic_numbers']
                frequencies = data['frequencies']
                spectrums = data['spectrums']

            # Filter images based on the given frequency range
            indices = [i for i, freq in enumerate(frequencies) if self.frequency_range[0] <= freq <= self.frequency_range[1]]
            filtered_frequencies = [frequencies[i] for i in indices]
            filtered_spectrums = spectrums[:, :, indices]
            
            self.frequencies.append(filtered_frequencies)
            self.images.append(filtered_spectrums)

            self.atom_pos.append(atom_pos)
            self.atomic_numbers.append(atomic_numbers)



            t = list(zip(atomic_numbers, atom_pos))

            text = f"{len(t)}\nComment\n"
            for i in range(len(t)):
                atom, pos = t[i]
                pos_str = "\t".join(f"{coord:.6f}" for coord in pos)
                text += atomic_symbols[atom] + "\t" +  pos_str + "\n"

            image = xyz_string_to_image_array(text)
            self.target_images.append(image)

            

        self.mean_atoms = np.mean([len(atoms) for atoms in self.atom_pos])
        self.std_atoms = np.std([len(atoms) for atoms in self.atom_pos])

        self.num_molecules_sizes = 10*[0]
        self.num_molecules_elements = 4*[0]

        indices_to_delete = []
        for i, molecule in enumerate(self.atom_pos):

            # Check if the molecule is too small and break out of the loop if so
            loop_stopped = len(molecule) <= self.mean_atoms - std_deviation_multiplier*self.std_atoms
            if loop_stopped:
                indices_to_delete.append(i)
                self.skipped_count += 1
                continue

            if loop_stopped:
                raise ValueError("Loop stopped but still in the loop")

            self.length_counter += 1

            length = len(self.frequencies[i])
            self.min_length = min(self.min_length, length)
            self.max_length = max(self.max_length, length)
            lengths.append(length)

            # Same with compute_bonds_new
            bonds = self.compute_bonds_new(molecule, self.atomic_numbers[i])
            self.bonds.append(bonds)
            self.unique_bonds.update(bonds)

            # Collecting the number of molecules of different sizes, crude approach implemented aferwards just to get the data
            self.num_molecules_sizes[len(molecule) - 10] += 1  # 10 is the smallest molecule size

            # Collecting the number of molecules containing different elements
            # Bond classes: (6, 1), (6, 6), (7, 1), (7, 6), (7, 7), (8, 1), (8, 6), (8, 7), (8, 8)
            contains_carbon = False
            contains_hydrogen = False
            contains_nitrogen = False
            contains_oxygen = False

            for i, bond in enumerate(bonds):
                if bond == (6, 1):
                    contains_carbon = True
                    contains_hydrogen = True
                elif bond == (6, 6):
                    contains_carbon = True
                elif bond == (7, 1):
                    contains_nitrogen = True
                    contains_hydrogen = True
                elif bond == (7, 6):
                    contains_nitrogen = True
                    contains_carbon = True
                elif bond == (7, 7):
                    contains_nitrogen = True
                elif bond == (8, 1):
                    contains_oxygen = True
                    contains_hydrogen = True
                elif bond == (8, 6):
                    contains_oxygen = True
                    contains_carbon = True
                elif bond == (8, 7):
                    contains_oxygen = True
                    contains_nitrogen = True
                else:  # bond == (8, 8)
                    contains_oxygen = True

            if contains_carbon:
                self.num_molecules_elements[0] += 1 # Carbon
            if contains_hydrogen:    
                self.num_molecules_elements[1] += 1 # Hydrogen
            if contains_nitrogen:    
                self.num_molecules_elements[2] += 1 # Nitrogen
            if contains_oxygen:
                self.num_molecules_elements[3] += 1 # Oxygen

        # Remove marked molecules
        for i in sorted(indices_to_delete, reverse=True):
            del self.atom_pos[i]
            del self.frequencies[i]
            del self.images[i]
            del self.atomic_numbers[i]
            del self.names[i]

        self.length = len(self.atom_pos)

        self.unique_bonds = sorted(self.unique_bonds)
        self.bond_count = len(self.unique_bonds)

        self.mean_length = np.mean(lengths)
        self.std_length = np.std(lengths)

        self.mean_atoms = np.mean([len(atoms) for atoms in self.atom_pos])
        self.std_atoms = np.std([len(atoms) for atoms in self.atom_pos])
        self.min_atoms = np.min([len(atoms) for atoms in self.atom_pos])
        self.max_atoms = np.max([len(atoms) for atoms in self.atom_pos])

        # unique_bonds are already sortred
        self.bond_map = {bond: idx for idx, bond in enumerate(self.unique_bonds)}


        


    def compute_bonds_new(self, atom_pos, atomic_numbers, cutoff_scale=1.24):  # Correct 1.24
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


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        images = self.images[idx] # (64, 64, N_modes)
        frequencies = self.frequencies[idx] # (N_modes,)

        target_image = self.target_images[idx] # (224, 224, 3)
        
        target_image = torch.from_numpy(target_image).float()

        target_image = target_image.permute(2, 0, 1)

        selected_images = [images[:,:, i] for i in range(images.shape[2])]
        selected_images = [torch.from_numpy(image).float() for image in selected_images]
        selected_frequencies = torch.tensor(frequencies).float()

        # Applying transformations to the images (and frequencies)
        if self.t_image:
            selected_images = [self.t_image(image) for image in selected_images]

        if self.t_freq:
            selected_frequencies = self.t_freq(selected_frequencies)

        

        
        selected_images  = torch.stack(selected_images, dim = 0)



        selected_images, selected_frequencies = padding(selected_images, selected_frequencies)
        mol_images_tensor = selected_images

        bonds = torch.zeros(self.bond_count)
        bond_indices = [self.bond_map[bond] for bond in self.bonds[idx]]
        bonds[bond_indices] = 1

        # selected_frequencies = selected_frequencies / self.max_freq

        return mol_images_tensor, selected_frequencies, target_image
