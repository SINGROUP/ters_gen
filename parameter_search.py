import torch
import numpy as np
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
#from src.datasets.ters_image_to_image import Ters_dataset_filtered_skip
from src.datasets.ters_image_to_image_sh import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero

from src.configs.base import get_config

from collections import defaultdict
from collections import Counter


def get_model(model_type, params):
    
    if model_type == "AttentionUNet":
        return AttentionUNet(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
    argparser.add_argument('--config', type = str, default=None, help='Path to YAML config file')
    args = argparser.parse_args()


    # Load configuration
    config = get_config(args.config)

    # Extract configuration parameters
    model_type = config.model.type
    model_params = {k: config.model[k] for k in config.model if k!= 'type'}
    batch_sizes = config.training.batch_sizes
    learning_rates = config.training.learning_rates
    loss_fn = config.training.loss_functions
    epochs = config.training.epochs
    data_path = config.data.path
    split_by_id = config.data.split_by_id
    save_path = config.save_path
    log_path = config.log_path
    num_channels = config.model.in_channels
    out_channels = config.model.out_channels

    sg_ch = True
    if out_channels > 1:
        sg_ch = False
    


    print("Configuration")
    print(config)
    # Set up the hyperparameter grid
    hyperparameter_grid = {
        'batch_size': batch_sizes,
        'lr': learning_rates,
        'loss_fn': loss_fn
    }


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
        filename=data_path,
        frequency_range=[0, 4000],  # adjust as needed
        num_channels=num_channels,
        std_deviation_multiplier=2,
        sg_ch=sg_ch,
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

    #model = AttentionUNet(in_channels=400, out_channels=4, filters=[16, 32, 64, 128], att_channels=32, kernel_size=[3, 3, 3, 3], return_att_map=False)
    
    print('Data loaded')

    for combination in combinations:
        batch_size, lr, loss_fn = combination

        model = get_model(model_type, model_params)
        model.to(device)


        dataloader_args = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 4
        }
        print("Training with paramaeters: ", combination)
        trainer = Trainer(model,
                          lr = lr,
                          loss_fn = loss_fn, 
                          train_set=train_dataset,
                          validation_set=validation_dataset,
                          test_set=test_dataset,
                          save_path=save_path,
                          log_path=log_path,
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

        trainer.save_final_model(f"_bs_{batch_size}_lr_{lr}_loss_{loss_fn}")

    sorted_results = sorted(results, key=lambda x: x['lowest_val_loss'])
    print("\nResults sorted by validation loss:")
    for result in sorted_results:
        print(f"Params: {result['params']} | Lowest val Loss: {result['lowest_val_loss']:.3f} at epoch {result['lowest_val_loss_epoch']}")

if __name__ == '__main__':
    main()