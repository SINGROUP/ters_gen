import os
import shutil
from sklearn.model_selection import train_test_split


def split_data(input_dir, output_dir, val_ratio = 0.2, seed = 42):

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # Create directories for train and val
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all files in the input directory
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    
    train_files, val_files = train_test_split(files, test_size=val_ratio, random_state=seed)

    for f in train_files:
        shutil.copy(os.path.join(input_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(input_dir, f), os.path.join(val_dir, f))
    

    print(f"Split {len(files)} files into {len(train_files)} train and {len(val_files)} val files.")