import torch
import argparse
import itertools
import os
import time
import random
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Keep your original imports
from src.models import *
from src.trainer.trainer_image_to_image import Trainer
from src.datasets.ters_image_to_image import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero

from collections import defaultdict
from collections import Counter

import numpy as np

def split_dataset(dataset, test_ratio=0.2, val_ratio=0.1, seed=0):
    # Ensure reproducibility
    torch.manual_seed(seed)
    
    # Calculate sizes for each set
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - test_size - val_size
    
    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset


def split_dataset_by_moleculeid(dataset, test_ratio=0.2, val_ratio=0.1, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)  # Ensure reproducibility for random.shuffle

    # Function to extract moleculeid from the full name
    def extract_moleculeid(name):
        return name.split('_')[0]
    
    # Group indices by moleculeid
    id_to_indices = defaultdict(list)
    for idx, name in enumerate(dataset.names):
        moleculeid = extract_moleculeid(name)
        id_to_indices[moleculeid].append(idx)
    
    # Get all unique molecule IDs
    all_ids = list(id_to_indices.keys())
    
    # Shuffle the molecule IDs to randomize the splitting
    random.shuffle(all_ids)
    
    # Calculate the number of molecule IDs in each set
    total_ids = len(all_ids)
    test_size = int(test_ratio * total_ids)
    val_size = int(val_ratio * total_ids)
    train_size = total_ids - test_size - val_size
    
    # Split molecule IDs into train, validation, and test sets
    train_ids = all_ids[:train_size]
    val_ids = all_ids[train_size:train_size + val_size]
    test_ids = all_ids[train_size + val_size:]
    
    # Collect indices for each set
    train_indices = [idx for moleculeid in train_ids for idx in id_to_indices[moleculeid]]
    val_indices = [idx for moleculeid in val_ids for idx in id_to_indices[moleculeid]]
    test_indices = [idx for moleculeid in test_ids for idx in id_to_indices[moleculeid]]
    
    # Create Subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('data_path', type=str, help='Path to the data folder containing the npz files of the molecules')
    argparser.add_argument('--save_path', type=str, help='Path to save the model')
    argparser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    argparser.add_argument('--batch_size', type=int, nargs='+', default=[4, 8, 16, 32], help='Batch size used for training')
    argparser.add_argument('--learning_rate', type=float, nargs='+', default=[0.001, 0.005], help='Learning rate used for training')
    argparser.add_argument('--split_by_id', type=int, default=0, help='Split the dataset by molecule ID')
    args = argparser.parse_args()

    # Set up the hyperparameter grid
    hyperparameter_grid = {
        'batch_size': args.batch_size,
        'lr': args.learning_rate,
    }


    split_by_id = args.split_by_id

    # ---------------------------------------------------------------------
    # 1) Define separate transforms for training vs. validation/testing
    train_transform = transforms.Compose([
        Normalize(),
        MinimumToZero(),
    ])
    val_test_transform = transforms.Compose([
        Normalize(),
        MinimumToZero()
    ])
    
    train_transform = val_test_transform

    # 2) Load your dataset (no transforms here, they are assigned after splitting)
    ters_set = Ters_dataset_filtered_skip(
        filename=args.data_path,
        frequency_range=[0, 4000],  # adjust as needed
        std_deviation_multiplier=2,
        t_image=None,
        t_freq=None
    )

    # 3) Split into train/val/test
    if split_by_id:
        train_dataset, validation_dataset, test_dataset = split_dataset_by_moleculeid(ters_set, seed=0)
    else:
        train_dataset, validation_dataset, test_dataset = split_dataset(ters_set, seed=0)

    # 4) Assign transforms: augmentation only on training set
    train_dataset.dataset.t_image = train_transform
    validation_dataset.dataset.t_image = val_test_transform
    test_dataset.dataset.t_image = val_test_transform

    print("Data Loaded")
    # ---------------------------------------------------------------------

  
    
    combinations = list(itertools.product(*hyperparameter_grid.values()))
    results = []

    epochs = args.epochs
    data_path = args.data_path
    save_path = args.save_path

    for combination in combinations:
        batch_size, lr = combination


        bond_dataset = []
        print(train_dataset.dataset.bond_map)
    
    
        for i in train_dataset.indices:
            bonds = test_dataset.dataset.bonds[i]
            for bond in bonds:
                bond_dataset.append(bond)
    

        counter = Counter(bond_dataset)
        weights = []
        print(counter)
    
        N = len(bond_dataset) - counter[(np.int64(8), np.int64(8))]
    
        for c in train_dataset.dataset.bond_map:
            value = counter[c]
            print(f'{c}: {value}')
            if c == (np.int64(8), np.int64(8)):
                weights.append(0)
            else:
                weights.append(N/value)
    
    
        weights = torch.tensor(weights).float().to(device)
        weights = weights/weights.sum()
    
        print("Class weights:", weights)
    
    
        criterion = nn.BCEWithLogitsLoss(weight=weights)
        #criterion = nn.BCEWithLogitsLoss()

        dataloader_args = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 8
        }
        print("Training with paramaeters: ", combination)
        trainer = Trainer(lr = lr,
                          train_set=train_dataset,
                          validation_set=validation_dataset,
                          test_set=test_dataset,
                          save_path=save_path,
                          dataloader_args=dataloader_args,
                          device=device,
                          print_interval=100,
                          dataset_bonds=ters_set.unique_bonds)

        trainer.train(epochs=epochs)

        trainer.save_image()


        results.append({
            'params': combination,
            'lowest_val_loss': trainer.lowest_val_loss,
            'lowest_val_loss_epoch': trainer.lowest_val_loss_epoch
        })

        trainer.save_final_model("_bs_{batch_size}_lr_{lr}")

    sorted_results = sorted(results, key=lambda x: x['lowest_val_loss'])
    print("\nResults sorted by validation loss:")
    for result in sorted_results:
        print(f"Params: {result['params']} | Lowest val Loss: {result['lowest_val_loss']:.3f} at epoch {result['lowest_val_loss_epoch']}")

if __name__ == '__main__':
    main()