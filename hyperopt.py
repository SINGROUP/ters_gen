import os
import torch
import argparse
import pprint
import optuna
import wandb
import shutil
import numpy as np
import multiprocessing
from contextlib import contextmanager
import torchvision.transforms as transforms

# Project imports (adjust paths as necessary)
from src.models import AttentionUNet
from src.trainer.trainer_image_to_image import Trainer
from src.datasets.ters_image_to_image_sh import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero
from src.configs.base import get_config

class GpuQueue:
    def __init__(self, n_gpus, manager):
        self.n_gpus = n_gpus
        self.queue = manager.Queue()
        if n_gpus > 0:
            for idx in range(n_gpus):
                self.queue.put(idx)
        else:
            self.queue.put(None)

    @contextmanager
    def one_gpu_per_process(self):
        gpu_idx = self.queue.get()
        try:
            yield gpu_idx
        finally:
            self.queue.put(gpu_idx)

def get_model(model_type, params):
    if model_type == "AttentionUNet":
        return AttentionUNet(**params)
    raise ValueError(f"Unknown model type: {model_type}")

def sample_model_params(trial, config):
    idx = trial.suggest_int("filters_idx", 0, len(config.model.filters_options) - 1)
    return {
        "in_channels": trial.suggest_categorical("in_channels", config.model.in_channels),
        "out_channels": config.model.out_channels,
        "filters": config.model.filters_options[idx],
        "att_channels": trial.suggest_categorical("att_channels", config.model.att_channels_options),
        "kernel_size": config.model.kernel_size_options[idx],
    }

def objective(trial, config, gpu_queue, use_wandb=False):
    batch_size = trial.suggest_categorical("batch_size", config.training.batch_sizes)
    lr = trial.suggest_float("lr", config.training.learning_rates[0], config.training.learning_rates[-1], log=True)
    loss_name = trial.suggest_categorical("loss_fn", config.training.loss_functions)
    augmentation = trial.suggest_categorical("augmentation", config.data.augmentation)
    run_name = f"trial_{trial.number}_bs{batch_size}_lr{lr:.0e}_{loss_name}"

    run = None
    if use_wandb:
        run = wandb.init(
            project="Posnet_50epochs_just_aug_val_32x32",
            name=run_name,
            config={
                **vars(config),
                "batch_size": batch_size,
                "lr": lr,
                "loss_fn": loss_name
            },
            reinit=True
        )

    final_dice = None
    with gpu_queue.one_gpu_per_process() as gpu_idx:
        device = torch.device(f"cuda:{gpu_idx}" if gpu_idx is not None and torch.cuda.is_available() else "cpu")
        try:
            transform = transforms.Compose([Normalize(), MinimumToZero()])
            model_params = sample_model_params(trial, config)
            model = get_model(config.model.type, model_params).to(device)

            train_ds = Ters_dataset_filtered_skip(
                filename=config.data.train_path,
                frequency_range=[0, 4000],
                num_channels=model_params["in_channels"],
                std_deviation_multiplier=2,
                sg_ch=(config.model.out_channels == 1),
                circle_radius=config.data.circle_radius,
                t_image=transform,
                train_aug=augmentation
            )
            val_ds = Ters_dataset_filtered_skip(
                filename=config.data.val_path,
                frequency_range=[0, 4000],
                num_channels=model_params["in_channels"],
                std_deviation_multiplier=2,
                sg_ch=(config.model.out_channels == 1),
                circle_radius=config.data.circle_radius,
                t_image=transform
            )

            trainer = Trainer(
                model=model,
                lr=lr,
                loss_fn=loss_name,
                train_set=train_ds,
                validation_set=val_ds,
                test_set=None,
                save_path=config.save_path,
                log_path=config.log_path,
                dataloader_args={"batch_size": batch_size, "shuffle": True, "num_workers": 7},  # set to 0 for debug, increase later
                device=device,
                print_interval=0,
                dataset_bonds=train_ds.unique_bonds
            )

            trainer.train(epochs=config.training.epochs)
            model_file = f"_trial{trial.number}_bs{batch_size}_lr{lr:.0e}_loss{loss_name}.pt"
            trainer.save_final_model(model_file)
            final_dice = trainer.final_metrics()
            trial.set_user_attr("model_path", model_file)
        except Exception as e:
            import traceback
            print(f"Exception in trial {trial.number}: {e}")
            traceback.print_exc()
            final_dice = 0.0  # Penalize failed trials
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if use_wandb and run is not None:
        wandb.log({"final_dice": final_dice, "trial": trial.number})
        run.finish()

    return final_dice

def visualize_study(study, output_dir):
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
        plot_edf,
        plot_intermediate_values
    )
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
    parser.add_argument("--n_gpus", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()

    config = get_config(args.config)
    pprint.pprint(config)

    with multiprocessing.Manager() as manager:
        gpu_queue = GpuQueue(args.n_gpus, manager)
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(
            lambda t: objective(t, config, gpu_queue, args.use_wandb),
            n_trials=config.training.n_trials,
            n_jobs=args.n_gpus
        )

        print("Optuna results saving")

    df = study.trials_dataframe()
    os.makedirs(config.log_path, exist_ok=True)
    df.to_csv(os.path.join(config.log_path, "optuna_trials.csv"), index=False)
    print(f"Trials logged to {config.log_path}/optuna_trials.csv")
    print("Best params:", study.best_params)
    print("Best dice:", study.best_value)
    best_trial = study.best_trial
    if "model_path" in best_trial.user_attrs:
        best_model = best_trial.user_attrs["model_path"]
        shutil.copy(
            os.path.join(config.save_path, "seg" + best_model),  # fixed here: no "seg" prefix
            os.path.join(config.save_path, "best_model.pt")
        )
        print("Best model saved to", os.path.join(config.save_path, "best_model.pt"))
    visualize_study(study, os.path.join(config.log_path, "visualizations"))

if __name__ == "__main__":
    main()