import os
import torch
import argparse
import pprint
import optuna
import wandb
import shutil
import subprocess
import numpy as np
from multiprocessing import Manager, Lock
import torchvision.transforms as transforms

# Project imports (you must adjust these if your structure changes)
from src.models import AttentionUNet
from src.trainer.trainer_image_to_image import Trainer
from src.datasets.ters_image_to_image_sh import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero
from src.configs.base import get_config

def get_free_gpu_id(threshold_mb=10000, locked_gpus=None):
    """Return GPU id with most free memory, ignoring locked_gpus."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    free_mem = np.array([int(x) for x in result.stdout.strip().split('\n')])
    mask = np.ones_like(free_mem, dtype=bool)
    if locked_gpus:
        for g in locked_gpus:
            mask[g] = False
    available = np.where((free_mem > threshold_mb) & mask)[0]
    if len(available) == 0:
        return None
    return int(available[np.argmax(free_mem[available])])

def get_model(model_type, params):
    if model_type == "AttentionUNet":
        return AttentionUNet(**params)
    raise ValueError(f"Unknown model type: {model_type}")

def sample_model_params(trial, config):
    filters = trial.suggest_categorical("filters", config.model.filters_options)
    idx = config.model.filters_options.index(filters)
    kernel_size = config.model.kernel_size_options[idx]
    return {
        "in_channels": trial.suggest_categorical("in_channels", config.model.in_channels),
        "out_channels": config.model.out_channels,
        "filters": filters,
        "att_channels": trial.suggest_categorical("att_channels", config.model.att_channels_options),
        "kernel_size": kernel_size,
    }

def objective(trial, config, gpu_state, lock, use_wandb=False, threshold_mb=7500):
    # --- GPU assignment with lock (critical section)
    with lock:
        locked_gpus = list(gpu_state['locked'])
        gpu_id = get_free_gpu_id(threshold_mb=threshold_mb, locked_gpus=locked_gpus)
        if gpu_id is None:
            raise RuntimeError("No free GPU available for this trial!")
        gpu_state['locked'].append(gpu_id)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # --- Hyperparameter sampling
        batch_size = trial.suggest_categorical("batch_size", config.training.batch_sizes)
        lr = trial.suggest_float("lr", config.training.learning_rates[0], config.training.learning_rates[-1], log=True)
        loss_name = trial.suggest_categorical("loss_fn", config.training.loss_functions)
        augmentation = trial.suggest_categorical("augmentation", config.data.augmentation)
        run_name = f"trial_{trial.number}_bs_{batch_size}_lr_{lr:.0e}_{loss_name}"

        # --- Weights & Biases logging
        if use_wandb:
            wandb.init(
                project="Composnet_multi_class",
                reinit=True,
                name=run_name,
                config={**vars(config), "batch_size": batch_size, "lr": lr, "loss_fn": loss_name}
            )

        # --- Data preparation
        transform = transforms.Compose([Normalize(), MinimumToZero()])
        model_params = sample_model_params(trial, config)
        model = get_model(config.model.type, model_params).to(device)
        in_channels = model_params["in_channels"]
        train_ds = Ters_dataset_filtered_skip(
            filename=config.data.train_path,
            frequency_range=[0,4000],
            num_channels=in_channels,
            std_deviation_multiplier=2,
            sg_ch=(config.model.out_channels==1),
            circle_radius=config.data.circle_radius,
            t_image=transform,
            train_aug=augmentation
        )
        val_ds = Ters_dataset_filtered_skip(
            filename=config.data.val_path,
            frequency_range=[0,4000],
            num_channels=in_channels,
            std_deviation_multiplier=2,
            sg_ch=(config.model.out_channels==1),
            circle_radius=config.data.circle_radius,
            t_image=transform
        )
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
        trainer.train(epochs=config.training.epochs)
        trainer.save_final_model(f"_trial{trial.number}_bs{batch_size}_lr{lr:.0e}_loss{loss_name}.pt")
        final_dice = trainer.final_metrics()
        if use_wandb:
            wandb.log({"final_dice": final_dice, "trial": trial.number})
        return final_dice
    finally:
        with lock:
            gpu_state['locked'].remove(gpu_id)

def visualize_study(study, output_dir):
    from optuna.visualization import (
        plot_param_importances,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
        plot_edf,
        plot_intermediate_values
    )
    import os
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
    parser.add_argument("--n_gpus", type=int, default=4, help="How many GPUs to use in parallel")
    args = parser.parse_args()

    config = get_config(args.config)
    pprint.pprint(config)
    n_jobs = args.n_gpus

    # multiprocessing manager for gpu lock state
    manager = Manager()
    gpu_state = manager.dict()
    gpu_state['locked'] = manager.list()
    lock = Lock()

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda t: objective(t, config, gpu_state, lock, args.use_wandb),
        n_trials=config.training.n_trials,
        n_jobs=n_jobs
    )

    df = study.trials_dataframe()
    os.makedirs(config.save_path, exist_ok=True)
    df.to_csv(os.path.join(config.save_path, "optuna_trials.csv"), index=False)
    print(f"Trials logged to {config.save_path}/optuna_trials.csv")
    print("Best params:", study.best_params)
    print("Best dice:", study.best_value)
    best_trial = study.best_trial
    best_model_path = best_trial.user_attrs.get("model_path")
    if best_model_path:
        src_path = os.path.join(config.save_path, best_model_path)
        dst_path = os.path.join(config.save_path, "best_model.pt")
        shutil.copy(src_path, dst_path)
        print(f"Best model saved as {dst_path}")
    visualize_study(study, config.save_path)
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()