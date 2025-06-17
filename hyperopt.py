import os
import torch
import argparse
import pprint
import optuna
from optuna.exceptions import TrialPruned
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from optuna.visualization import (
    plot_param_importances,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf,
    plot_intermediate_values
)

# Optional Weights & Biases
import wandb

# project imports
from src.models import AttentionUNet, UNet
from src.trainer.trainer_image_to_image import Trainer
from src.datasets.ters_image_to_image_sh import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero
from src.configs.base import get_config

def get_model(model_type, params):
    """Instantiate model based on type."""
    if model_type == "AttentionUNet":
        return AttentionUNet(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def sample_model_params(trial, config):
    """Sample model hyperparameters, keeping filters & kernels aligned."""
    filters = trial.suggest_categorical("filters", config.model.filters_options)
    idx = config.model.filters_options.index(filters)
    kernel_size = config.model.kernel_size_options[idx]
    return {
        "in_channels": trial.suggest_categorical("in_channels", config.model.in_channels),
        "out_channels": config.model.out_channels,
        "filters": filters,
        "att_channels": trial.suggest_categorical("att_channels", config.model.att_channels_options),
        "kernel_size": kernel_size,
        #"activation": trial.suggest_categorical("activation", config.model.activation_options)
    }

def objective(trial, config, device, use_wandb=False):
    """
    Optuna objective:
    1) Sample hyperparams
    2) Train epoch-by-epoch and report intermediate metrics for pruning
    3) Return final dice coefficient
    """
    # Sample training parameters
    batch_size = trial.suggest_categorical("batch_size", config.training.batch_sizes)
    lr = trial.suggest_float("lr", config.training.learning_rates[0], config.training.learning_rates[-1], log=True)
    loss_name = trial.suggest_categorical("loss_fn", config.training.loss_functions)
    augmentation = trial.suggest_categorical("augmentation", config.data.augmentation)

    # W&B config update
    if use_wandb:
        wandb.config.update({"batch_size": batch_size, "lr": lr, "loss_fn": loss_name}, allow_val_change=True)

    # Data transforms
    transform = transforms.Compose([Normalize(), MinimumToZero()])

    # Prepare datasets
    train_ds = Ters_dataset_filtered_skip(
        filename=config.data.train_path,
        frequency_range=[0,4000],
        num_channels=config.model.in_channels[0],
        std_deviation_multiplier=2,
        sg_ch=(config.model.out_channels==1),
        circle_radius=config.data.circle_radius,
        t_image=transform, 
        train_aug=augmentation
    )
    val_ds = Ters_dataset_filtered_skip(
        filename=config.data.val_path,
        frequency_range=[0,4000],
        num_channels=config.model.in_channels[0],
        std_deviation_multiplier=2,
        sg_ch=(config.model.out_channels==1),
        circle_radius=config.data.circle_radius,
        t_image=transform
    )

    # Instantiate model
    model_params = sample_model_params(trial, config)
    model = get_model(config.model.type, model_params).to(device)

    # Initialize trainer
    trainer = Trainer(
        model,
        lr=lr,
        loss_fn=loss_name,
        train_set=train_ds,
        validation_set=val_ds,
        test_set=None,
        save_path=config.save_path,
        log_path=config.log_path,
        dataloader_args={"batch_size": batch_size, "shuffle": True, "num_workers": 8},
        device=device,
        print_interval=0,
        dataset_bonds=train_ds.unique_bonds
    )

    # Epoch-wise training for pruning
    for epoch in range(config.training.epochs):
        # Train one epoch (assumes Trainer has this method)
        trainer.train_one_epoch()
        # Validate after epoch (assumes validate() returns dice)
        dice = trainer.validate()
        # Report to Optuna
        trial.report(dice, epoch)
        if trial.should_prune():
            raise TrialPruned(f"Trial {trial.number} pruned at epoch {epoch}")

    # Save final model
    trainer.save_final_model(f"_trial{trial.number}_bs{batch_size}_lr{lr:.0e}_loss{loss_name}")
    final_dice = trainer.final_metrics()

    # W&B logging
    if use_wandb:
        wandb.log({"final_dice": final_dice, "trial": trial.number})

    return final_dice

def visualize_study(study, output_dir):
    """Save all Optuna visualizations to HTML."""
    os.makedirs(output_dir, exist_ok=True)
    plot_optimization_history(study).write_html(os.path.join(output_dir, "optimization_history.html"))
    plot_param_importances(study).write_html(os.path.join(output_dir, "param_importances.html"))
    plot_parallel_coordinate(study).write_html(os.path.join(output_dir, "parallel_coordinates.html"))
    plot_slice(study).write_html(os.path.join(output_dir, "slice_plot.html"))
    plot_contour(study).write_html(os.path.join(output_dir, "contour_plot.html"))
    plot_edf(study).write_html(os.path.join(output_dir, "edf_plot.html"))
    plot_intermediate_values(study).write_html(os.path.join(output_dir, "intermediate_values.html"))
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # W&B login if needed
    if args.use_wandb:
        wandb.login()  # ensure you're logged in before init
        wandb.init(project="Posnet", config={})

    config = get_config(args.config)
    pprint.pprint(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create study with pruning
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(t, config, device, args.use_wandb),
                   n_trials=config.training.n_trials,
                   n_jobs=config.training.n_jobs)

    # Save trial dataframe
    df = study.trials_dataframe()
    os.makedirs(config.save_path, exist_ok=True)
    df.to_csv(os.path.join(config.save_path, "optuna_trials.csv"), index=False)
    print(f"Trials logged to {config.save_path}/optuna_trials.csv")

    # Best results
    print("Best params:", study.best_params)
    print("Best dice:", study.best_value)

    # Visualize
    visualize_study(study, config.save_path)

    # Finish W&B
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
