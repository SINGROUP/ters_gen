"""
Simple evaluation script for a single model on a single dataset.
Uses the same Metrics class as training to compute global dice score.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from src.datasets import Ters_dataset_filtered_skip
from src.metrics import Metrics
from src.transforms import NormalizeVectorized, MinimumToZeroVectorized


def evaluate(model_path: str, data_path: str, batch_size: int = 32):
    """
    Evaluate a model on a dataset using the same metrics as training.
    
    Args:
        model_path: Path to the saved model (.pt file)
        data_path: Path to the data directory containing .npz files
        batch_size: Batch size for evaluation
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Get num_channels from model
    num_channels = model.conv.weight.shape[1]
    print(f"Detected num_channels: {num_channels}")
    
    # Setup transforms (same as training)
    data_transform = transforms.Compose([NormalizeVectorized(), MinimumToZeroVectorized()])
    
    # Setup dataset
    print(f"Loading data from: {data_path}")
    dataset = Ters_dataset_filtered_skip(
        filename=data_path,
        frequency_range=[0, 4000],
        num_channels=num_channels,
        std_deviation_multiplier=2,
        sg_ch=True,
        t_image=data_transform,
        t_freq=None,
        flag=False  # Returns (images, frequencies, target_image)
    )
    print(f"Number of samples: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Collect predictions and ground truths (same as training)
    all_ground_truths = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images, frequencies, tgt_image = batch
            images = images.to(device)
            tgt_image = tgt_image.to(device)
            
            # Threshold ground truth (same as training)
            tgt_image = (tgt_image > 0.01).int()
            
            # Get model predictions (same as training)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).int()
            
            # Collect ground truths and predictions
            all_ground_truths.append(tgt_image.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
    
    # Concatenate all batches (same as training - global metrics)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"Ground truth shape: {all_ground_truths.shape}")
    print(f"Predictions shape: {all_predictions.shape}")
    
    # Compute metrics using the same Metrics class as training
    metrics = Metrics(
        model=model,
        data={"pred": all_predictions, "ground_truth": all_ground_truths},
        config={}
    )
    results = metrics.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    evaluate(args.model, args.data, args.batch_size)
