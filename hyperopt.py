import torch
import argparse
import random 
import optuna
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms


# project imports
from src.models import AttentionUNet, UNet
from src.trainer.trainer_image_to_image import Trainer
from src.datasets.ters_image_to_image_sh import Ters_dataset_filtered_skip
from src.transforms import Normalize, MinimumToZero
from src.configs.base import get_config

from collections import defaultdict


def get_model(model_type, params):
    
    if model_type == "AttentionUNet":
        return AttentionUNet(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def objective(trial, config, device):

    # Sample hyperparameters
    #batch_size = trial.suggest_int('batch_size', config.training.batch_sizes[0], config.training.batch_sizes[1], step=16)
    batch_size = trial.suggest_categorical("batch_size", config.training.batch_sizes)
    lr = trial.suggest_loguniform('lr', config.training.learning_rates[0], config.training.learning_rates[-1])
    loss_name = trial.suggest_categorical('loss_fn', config.training.loss_functions)

    # build dataset + transforms
    data_transform = transforms.Compose([Normalize(), MinimumToZero()])


    print(f"[Trial {trial.number}] Trying batch_size={batch_size}, lr={lr}, loss_fn={loss_name}")

    ## Make it train, val and test dataset here!
    train_ds = Ters_dataset_filtered_skip(
        filename=config.data.train_path,
        frequency_range=[0, 4000],
        num_channels=config.model.in_channels,
        std_deviation_multiplier=2, 
        sg_ch=(config.model.out_channels == 1), 
        t_image=data_transform
    )


    val_ds = Ters_dataset_filtered_skip(
        filename=config.data.val_path,
        frequency_range=[0, 4000],
        num_channels=config.model.in_channels,
        std_deviation_multiplier=2, 
        sg_ch=(config.model.out_channels == 1), 
        t_image=data_transform
    )



    # loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)


    # model

    model_type = config.model.type
    model_params = {k: config.model[k] for k in config.model if k!= 'type'}
    #model = AttentionUNet(**config.model).to(device)

    model = get_model(model_type, model_params).to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr = lr)


    log_path = config.log_path
    save_path = config.save_path
    # trainer
    trainer = Trainer(
        model, lr=lr, loss_fn=loss_name,
        train_set=train_ds, validation_set=val_ds, test_set=None,
        save_path=save_path, log_path=log_path,
        dataloader_args={'batch_size': batch_size, 'shuffle': True, 'num_workers': 8},
        device=device, print_interval=0,
        dataset_bonds=train_ds.unique_bonds
    )

    # run training
    trainer.train(epochs=config.training.epochs)
    trainer.save_final_model(f"_bs_{batch_size}_lr_{lr}_loss_{loss_name}")
    dice_coeff = trainer.final_metrics()

    print(f"[Trial {trial.number}] Finished with val_loss={trainer.lowest_val_loss}")
    print(f"[Trial {trial.number}] Finished with dice_coeff={dice_coeff}")

    #return trainer.lowest_val_loss
    return dice_coeff

    # return validation metric (lower is better)
    #return dice_coeff




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required=True, help = 'Path to YAML config file')
    args = parser.parse_args()


    config = get_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, config, device), n_trials = config.training.n_trials, n_jobs=config.training.n_jobs)


    print('Best params:', study.best_params)
    print('Best val loss:', study.best_value)



if __name__ == '__main__':
    main()









