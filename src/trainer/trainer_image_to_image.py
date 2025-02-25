import os
import glob
import numpy as np
import time as time

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torch.nn as nn

from torchvision.utils import make_grid

from datetime import datetime

# Use tensorboard
from torch.utils.tensorboard import SummaryWriter



from src.models import UNet, Discriminator



class Trainer():
    def __init__(self, lr,
                 train_set,
                 validation_set, 
                 test_set,
                 save_path=None,
                 dataloader_args={
                        'batch_size': 4,
                        'shuffle': True,
                        'num_workers': 0
                    },
                 device='cuda',
                 print_interval=10,
                 dataset_bonds=None):

        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set

        self.train_loader = DataLoader(self.train_set, **dataloader_args)
        self.validation_loader = DataLoader(self.validation_set, **dataloader_args)
        testloader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
        # self.test_loader = DataLoader(self.test_set, **dataloader_args)
        self.test_loader = DataLoader(self.test_set, **testloader_args)
                
        self.device = device
        self.print_interval = print_interval
        self.dataset_bonds = dataset_bonds
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir=f"/scratch/phys/sin/sethih1/runs_ters/TERS_ML_gans/batch{dataloader_args['batch_size']}_LR{lr}/") 





        # New parameters 
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
        self.lambda_recon = 200

        input_dim = 54
        real_dim = 3

        target_shape = 64


        device = 'cuda'

        lr = lr
        self.gen = UNet(input_dim, real_dim).to(device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.disc = Discriminator(input_dim + real_dim).to(device) 
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)

    def train(self, epochs, early_stop_value=0.01):
        self.lowest_val_loss = float('inf')
        self.lowest_val_loss_epoch = 0
        for epoch in range(epochs):
            start = time.time()
            gen_train_loss, disc_train_loss = self.train_epoch()

            
            print(f"Epoch {epoch+1:4}/{epochs}, time: {time.time()-start:.2f} s, gen_loss: {gen_train_loss:.3f}, disc_loss: {disc_train_loss:.3f}")
            self.writer.add_scalar(f"Loss/Generator",gen_train_loss,epoch)
            self.writer.add_scalar(f"Loss/Discriminator", disc_train_loss, epoch)



        self.writer.close()

    def train_epoch(self):
        self.disc.train()
        self.gen.train()
        epoch_gen_loss = []
        epoch_disc_loss = []

        for i, batch in enumerate(self.train_loader):
            images, frequencies, tgt_image = batch
            condition = images.to(self.device)
            frequencies = frequencies.to(self.device)
            real = tgt_image.to(self.device)
            # images, bonds = self.batch_to_device(batch)


            # Train Discriminator
            self.disc_opt.zero_grad()
            with torch.no_grad():
                fake = self.gen(condition)

            disc_fake_hat = self.disc(fake.detach(), condition)
            disc_fake_loss = self.adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = self.disc(real, condition)
            disc_real_loss = self.adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            self.disc_opt.step()


            # Train Generator
            self.gen_opt.zero_grad()
            gen_loss = self.get_gen_loss(real, condition)
            gen_loss.backward()
            self.gen_opt.step()




            epoch_disc_loss.append(disc_loss.item())
            epoch_gen_loss.append(gen_loss.item())



        gen_loss = np.mean(epoch_gen_loss)
        disc_loss = np.mean(epoch_disc_loss)
        return gen_loss, disc_loss

    
    def get_gen_loss(self, real, condition):

        fake = self.gen(condition)
        disc_fake_pred = self.disc(fake, condition)
        gen_adv_loss = self.adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_rec_loss = self.recon_criterion(fake, real)
        gen_loss = gen_adv_loss + self.lambda_recon * gen_rec_loss

        return gen_loss
    


    def save_final_model(self, model_name):
        """Saving the parameters of the model after training."""
        if self.save_path:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            torch.save(self.gen, os.path.join(self.save_path, "gen" + model_name))
            torch.save(self.disc, os.path.join(self.save_path, "disc" + model_name))


    def save_image(self):

        condition, _, real = next(iter(self.test_loader))
        condition = condition.to(self.device)
        self.gen.eval()
        with torch.no_grad():
            fake = self.gen(condition)

        grid_condition = make_grid(condition.cpu())
        grid_fake = make_grid(real.cpu())

        self.writer.add_image('Real', grid_condition)
        self.writer.add_image('Generated', grid_fake)
        self.writer.close()