import os
import glob
import numpy as np
import time as time

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torch.nn as nn


from src.covalent_radii import covalent_radii


from src.utils.molecule_gaussian_image import molecule_gaussian_image
from src.utils.xyz_to_label import molecule_circular_image



# For augmentations

from src.transforms import AugmentTransform
import albumentations as A 
from albumentations.pytorch import ToTensorV2


#import kornia.augmentation as k






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
}


def padding(spectrums, frequencies):
    
    freq_pad = torch.zeros(100) # 54 is the maximum number of frequency modes possible for a molecule with 20 atoms
    img_pad = torch.zeros((54, 64, 64)) # 54 images with 64x64 pixels

    # Pad the images and frequencies to have the same length
    #img_pad[:spectrums.shape[0], :, :] = spectrums
    #freq_pad[:len(frequencies)] = frequencies

    return img_pad, freq_pad


def uniform_channels(spectrums, frequencies, num_channels=400):
    '''Function to create uniform channels for the spectrums'''
    
    

    max_freq = 4000
    #num_channels = 400
    step = max_freq // num_channels

    grid_size = spectrums.shape[1]
    channels = np.zeros((grid_size, grid_size, num_channels))

    mean = np.mean(spectrums, axis = (0,1))
    std = np.std(spectrums, axis=(0,1))
    spectrums = (spectrums - mean)/std
    

    count = 0 
    for i in range(1, max_freq, step):
        indices =  (frequencies > i) & (frequencies < i+step)
        selected_spectrums = spectrums[:, :, indices]
        if np.all(selected_spectrums == 0):
            count += 1
            continue
        if selected_spectrums.size == 0:
            count += 1
            continue
        channels[:, :, count] = np.mean(selected_spectrums, axis = 2)
        count += 1

    return channels


class Ters_dataset_filtered_skip(Dataset):
    def __init__(self, filename, frequency_range, num_channels, std_deviation_multiplier=2, sg_ch = True, circle_radius = 5, t_image=None, t_freq=None, flag = False, train_aug = False):
        super(Ters_dataset_filtered_skip, self).__init__()
        self.filename = filename
        self.frequency_range = frequency_range
        self.num_channels = num_channels
        self.t_image = t_image
        self.t_freq = t_freq
        self.flag = flag

        self.sg_ch = sg_ch
        self.circle_radius = circle_radius

        # Use glob to get a list of all .npz files in the directory
        file_pattern = os.path.join(filename, '*.npz')
        npz_files = sorted(glob.glob(file_pattern))

        self.length = len(npz_files)


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

        self.train_aug = train_aug



        # Adding augmentations (noise, rotation, etc.) to the images
        self.aug_image = AugmentTransform(gauss_std_range=(0.01, 0.1))

        '''self.aug_image = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            ToTensorV2(p=1.0),
        ])

        self.aug_image = torch.nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.random.VerticalFlip(p=0.5),
            K.RandomRotat
        )'''



        for npz_file in npz_files:
            self.names.append(npz_file)


            

        


    def compute_bonds_new(self, atom_pos, atomic_numbers, cutoff_scale=1.24):  # Correct 1.24
        num_atoms = len(atomic_numbers)
        bonds = set()
                
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = np.linalg.norm(atom_pos[i] - atom_pos[j])
                
                # Calculate cutoff distance based on covalent radii
                cutoff = cutoff_scale * (covalent_radii[atomic_numbers[i]] + covalent_radii[atomic_numbers[j]])
                
                if distance <= cutoff:
                    bonds.add((atomic_numbers[i], atomic_numbers[j]))
        
        return bonds


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        npz_file = self.names[idx]

        filename = os.path.splitext(os.path.basename(npz_file))[0]

        with np.load(npz_file, mmap_mode='r') as data:
            atom_pos = data['atom_pos']
            atomic_numbers = data['atomic_numbers']
            frequencies = data['frequencies']
            spectrums = data['spectrums']
                

        # Filter images based on the given frequency range
        mask = (frequencies >= self.frequency_range[0]) & (frequencies <= self.frequency_range[1])
        filtered_frequencies = frequencies[mask]
        filtered_spectrums = spectrums[:, :, mask]
        
        #channels = uniform_channels(spectrums, frequencies)
        filtered_spectrums = uniform_channels(filtered_spectrums, filtered_frequencies, num_channels=self.num_channels)
        #filtered_spectrums = channels
        t = list(zip(atomic_numbers, atom_pos))
        text = f"{len(t)}\nComment\n"
        for i in range(len(t)):
            atom, pos = t[i]
            pos_str = "\t".join(f"{coord:.6f}" for coord in pos)
            text += atomic_symbols[atom] + "\t" +  pos_str + "\n"

        target_image = molecule_circular_image(text, flag=self.sg_ch, circle_radius=self.circle_radius) # (224, 224, 3)
        
            



        images = filtered_spectrums # (64, 64, N_modes)
        frequencies = filtered_frequencies # (N_modes,)

        target_image = torch.from_numpy(target_image).float()

        #target_image = target_image.permute(2, 0, 1)

        #selected_images = torch.from_numpy(images).permute(2, 0, 1).float() # (N_modes, 64, 64)

        selected_images = [torch.from_numpy(images[:,:, i]).float() for i in range(images.shape[2])]


        #selected_images = [images[:,:, i] for i in range(images.shape[2])]
        #selected_images = [torch.from_numpy(image).float() for image in selected_images]
        selected_frequencies = torch.tensor(frequencies).float()

        # Applying transformations to the images (and frequencies)
        if self.t_image:
            selected_images = [self.t_image(image) for image in selected_images]

        if self.t_freq:
            selected_frequencies = self.t_freq(selected_frequencies)

        

        
        selected_images  = torch.stack(selected_images, dim = 0)

        if self.train_aug:
            selected_images, target_image = self.aug_image(img=selected_images, mask = target_image)
            '''
            augmented = self.aug_image(image=selected_images.permute(1,2,0).cpu().numpy(), mask = target_image.permute(1,2,0).cpu().numpy())
            selected_images = augmented["image"]
            target_image = augmented["mask"]
            '''


        _, selected_frequencies = padding(selected_images, selected_frequencies)
        mol_images_tensor = selected_images


        # selected_frequencies = selected_frequencies / self.max_freq

        if self.flag: 
            return filename, _, _ , mol_images_tensor, selected_frequencies, target_image
            #return filename, atom_pos, atomic_numbers, mol_images_tensor, selected_frequencies, target_image


        return mol_images_tensor, selected_frequencies, target_image

