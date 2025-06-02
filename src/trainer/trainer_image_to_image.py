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

from src.losses import get_loss_function

from src.metrics import Metrics




class Trainer():
    def __init__(self, model,
                 lr,
                 loss_fn, 
                 train_set,
                 validation_set, 
                 test_set,
                 save_path=None,
                 log_path = None, 
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

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = get_loss_function(loss_fn)
        #self.loss_fn = get_loss_function('dice_loss')
        #self.loss_fn = get_loss_function('focal_loss')
        self.model = model

        val_loader_args = {'batch_size': 64, 'shuffle': False, 'num_workers': 8}

        self.train_loader = DataLoader(self.train_set, **dataloader_args)
        self.validation_loader = DataLoader(self.validation_set, **dataloader_args)
        testloader_args = {'batch_size': 4, 'shuffle': False, 'num_workers': 0}
        # self.test_loader = DataLoader(self.test_set, **dataloader_args)
        self.test_loader = DataLoader(self.test_set, **testloader_args)
                
        self.device = device
        self.print_interval = print_interval
        self.dataset_bonds = dataset_bonds
        self.save_path = save_path
        log_path = os.path.join(log_path, f"batch{dataloader_args['batch_size']}_LR{lr}_{loss_fn}/")
        self.writer = SummaryWriter(log_dir=log_path) 


    def train(self, epochs, early_stop_value=0.01):
        self.lowest_val_loss = float('inf')
        self.lowest_val_loss_epoch = 0
        for epoch in range(epochs):
            start = time.time()
            epoch_loss = self.train_epoch()

            
            print(f"Epoch {epoch+1:4}/{epochs}, time: {time.time()-start:.2f} s, training_loss: {epoch_loss:.3f}")
            self.writer.add_scalar(f"Training Loss",epoch_loss, epoch)

            epoch_loss = self.evaluate()

            
            print(f"Epoch {epoch+1:4}/{epochs}, time: {time.time()-start:.2f} s, val_loss: {epoch_loss:.3f}")
            self.writer.add_scalar(f"Validation Loss",epoch_loss, epoch)

            # Computing model metrics
            self.evaluate_model_metrics(self.model, epoch)



        self.writer.close()

    def train_epoch(self):
        
        
        self.model.train()

        total_loss = []
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images, frequencies, tgt_image = batch
            images = images.to(self.device)
            tgt_image = tgt_image.to(self.device)

            tgt_image = (tgt_image > 0.01).long()
            outputs = self.model(images)
            #tgt_image_new = torch.zeros((tgt_image.shape[0], tgt_image.shape[1] + 1, tgt_image.shape[2], tgt_image.shape[3]), device=self.device)
            #tgt_image_new[:, 1:, :, :] = tgt_image
            #tgt_image = tgt_image_new
            #tgt_image = torch.argmax(tgt_image, dim=1)

            loss = self.loss_fn(outputs, tgt_image)
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())


        
        return np.mean(total_loss)
    

    def evaluate(self):

        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, batch in enumerate(self.validation_loader):
                
                images, frequencies, tgt_image = batch
                images = images.to(self.device)
                tgt_image = tgt_image.to(self.device)
    
                tgt_image = (tgt_image > 0.01).long()
                outputs = self.model(images)

                #tgt_image_new = torch.zeros((tgt_image.shape[0], tgt_image.shape[1] + 1, tgt_image.shape[2], tgt_image.shape[3]), device=self.device)
                #tgt_image_new[:, 1:, :, :] = tgt_image
                #tgt_image = tgt_image_new
                #tgt_image = torch.argmax(tgt_image, dim=1)
                loss = self.loss_fn(outputs, tgt_image)
                
                total_loss.append(loss.item())

        return np.mean(total_loss)
    



    def compute_metrics(self,model, data_loader):
        """
        Compute metrics for a given data loader (training or validation).
        """
        all_inputs = []
        all_ground_truths = []
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                images, frequencies, tgt_image = batch
                images = images.to(self.device)
                tgt_image = tgt_image.to(self.device)

                # Threshold and convert target image to long
                tgt_image = (tgt_image > 0.01).int()

                # Get model predictions
                outputs = model(images)
                #preds = torch.argmax(outputs, dim=1)

                preds = (outputs > 0.5).int()

                # Collect inputs, ground truths, and predictions
                all_inputs.append(images.cpu().numpy())
                all_ground_truths.append(tgt_image.cpu().numpy())
                all_predictions.append(preds.cpu().numpy())

        # Convert lists to numpy arrays
        all_ground_truths = np.concatenate(all_ground_truths, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        #print("Ground Truth: ", all_ground_truths)
        #print("Predictions: ", all_predictions)

        # Initialize Metrics class
        metrics = Metrics(model=model, data={"pred": all_predictions, "ground_truth": all_ground_truths}, config={})

        # Compute metrics
        results = metrics.evaluate()

        

        return results
    

    def evaluate_model_metrics(self, model, step):


        model.eval()
        train_loader = self.train_loader
        val_loader = self.validation_loader

        # Compute metrics for training data
        train_metrics = self.compute_metrics(model, train_loader)

        # Compute metrics for validation data
        val_metrics = self.compute_metrics(model, val_loader)

        # Print metrics
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
            self.writer.add_scalar(f"Train/{metric}", value, step)

        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
            self.writer.add_scalar(f"Validation/{metric}", value, step)

        
    

    def final_metrics(self):
        model = self.model.eval()
        model.eval()
        val_loader = self.validation_loader

        # Compute metrics for training data
        val_metrics = self.compute_metrics(model, val_loader)

        # Print metrics
        print("Test Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
            self.writer.add_scalar(f"Test/{metric}", value, 0)

        dice_coeff = val_metrics["Dice Coefficient"]

        return dice_coeff

            

    

    def save_final_model(self, model_name):
        """Saving the parameters of the model after training."""
        if self.save_path:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            torch.save(self.model, os.path.join(self.save_path, "seg" + model_name))




    def save_image(self):
        input_img, _, tgt_img = next(iter(self.test_loader))
        input_img = input_img[:4]
        tgt_img = tgt_img[:4]
        input_img = input_img.to(self.device)

        tgt_image = tgt_img

        #tgt_img = (tgt_img > 0.01).long().float()
        '''
        tgt_image_new = torch.zeros((tgt_image.shape[0], tgt_image.shape[1] + 1, tgt_image.shape[2], tgt_image.shape[3]))
        tgt_image_new[:, 1:, :, :] = tgt_image
        tgt_img = tgt_image_new
        '''

        self.model.eval()
        with torch.no_grad():
            seg = self.model(input_img)
   
        '''num_channels = 5  # H, C, N, O

        # Reshape so that channels remain contiguous per image:
        # This changes the shape from [B, 4, H, W] to [B*4, 1, H, W].
        seg_reshaped = seg.view(-1, 1, seg.shape[2], seg.shape[3])
        tgt_reshaped = tgt_img.view(-1, 1, tgt_img.shape[2], tgt_img.shape[3])

        # Use make_grid with nrow=4: 
        # Each row in the grid will have 4 images (i.e. channels) corresponding to one image.
        grid_fake = make_grid(seg_reshaped.cpu(), nrow=num_channels, normalize=True, padding=2)
        grid_real = make_grid(tgt_reshaped.cpu(), nrow=num_channels, normalize=True, padding=2)'''

        #self.writer.add_image("Segmented Image", grid_fake)
        #self.writer.add_image("True Label", grid_real)

        if seg.shape[1] > 1:
            num_channels = seg.shape[1] 
            seg_reshaped = seg.view(-1, 1, seg.shape[2], seg.shape[3])
            tgt_image_reshaped = tgt_image.view(-1, 1, tgt_image.shape[2], tgt_image.shape[3])

            grid_fake = make_grid(seg_reshaped.cpu(), nrow=num_channels, normalize=True, padding=2)
            grid_real = make_grid(tgt_image_reshaped.cpu(), nrow=num_channels, normalize=True, padding=2)

        else:
            

        #seg = seg[:, 1:, :, :]
        #tgt_img = tgt_img[:, 1:, :, :]

            #s = (seg.cpu()*255)
            #t = (tgt_img.cpu()*255)
            s = seg
            t = tgt_img
            grid_fake = make_grid(s, normalize=True, padding=2)
            grid_real = make_grid(t, normalize=True, padding=2)

        self.writer.add_image("Segmented Image", grid_fake)
        self.writer.add_image("True Label", grid_real)


        self.writer.close()
