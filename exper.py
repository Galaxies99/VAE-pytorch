import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
import torchvision
import torchvision.utils as tvutils
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class CelebAExperiment(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super(CelebAExperiment, self).__init__()
        self.model = model
        self.cur_device = None
        self.params = kwargs
        self.base_dir = None
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def train_dataloader(self):
        dataset = CelebA(
            root = self.params['data_path'], 
            split = 'train',
            transform = self.get_transform(),
            download = False
        )
        self.num_train_images = len(dataset)
        return DataLoader(
            dataset,
            batch_size = self.params['batch_size'],
            num_workers = 4,
            shuffle = True,
            drop_last = True
        )
    
    def val_dataloader(self):
        dataset = CelebA(
            root = self.params['data_path'], 
            split = 'test',
            transform = self.get_transform(),
            download = False
        )
        self.num_val_images = len(dataset)
        self.sample_dataloader = DataLoader(
            dataset,
            batch_size = self.params['batch_size'],
            num_workers = 4,
            drop_last = True
        )
        return self.sample_dataloader
    
    def training_step(self, data, batch_idx, optimizer_idx = 0):
        x, labels = data
        self.cur_device = x.device
        res = self.model.forward(x, labels = labels)
        loss = self.model.loss(
            *res, 
            kl_weight = self.params['batch_size'] / self.num_train_images, 
            batch_idx = batch_idx,
            optimizer_idx = optimizer_idx
        )
        self.logger.experiment.log({key: val.item() for key, val in loss.items()})
        return loss
    
    def validation_step(self, data, batch_idx, optimizer_idx = 0):
        x, labels = data
        self.cur_device = x.device
        res = self.model.forward(x, labels = labels)
        loss = self.model.loss(
            *res, 
            kl_weight = self.params['batch_size'] / self.num_val_images, 
            batch_idx = batch_idx,
            optimizer_idx = optimizer_idx
        )
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.sample_images()
        return {'val_loss': avg_loss}

    def sample_images(self):
        x, labels = next(iter(self.sample_dataloader))
        x = x.to(self.cur_device)
        labels = labels.to(self.cur_device)
        recon = self.model.reconstruct(x, labels = labels)
        nrow = int(np.ceil(math.sqrt(self.params['batch_size'])))
        
        if self.base_dir is None:
            self.base_dir = os.path.join(self.logger.save_dir, self.logger.name, "version_" + str(self.logger.version))
            self.reconstructed_dir = os.path.join(self.base_dir, "reconstructed_images")
            self.generated_dir = os.path.join(self.base_dir, "generated_images")
            if os.path.exists(self.reconstructed_dir) is False:
                os.makedirs(self.reconstructed_dir)
            if os.path.exists(self.generated_dir) is False:
                os.makedirs(self.generated_dir)

        tvutils.save_image(
            recon.data,
            os.path.join(
                self.reconstructed_dir,
                "reconstructed_{}_{}.png".format(self.logger.name, self.current_epoch)
            ),
            normalize = True,
            nrow = nrow
        )
        tvutils.save_image(
            x.data,
            os.path.join(
                self.reconstructed_dir,
                "original_{}_{}.png".format(self.logger.name, self.current_epoch)
            ),
            normalize = True,
            nrow = nrow
        )
        samples = self.model.sample(self.params['sample_num'], self.cur_device, labels = labels)
        tvutils.save_image(
            samples.data,
            os.path.join(
                self.generated_dir,
                "generated_{}_{}.png".format(self.logger.name, self.current_epoch)
            ),
            normalize = True,
            nrow = nrow
        )
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.params['learning_rate'],
            weight_decay = self.params['weight_decay'],
            eps = self.params['eps']
        )
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma = self.params['scheduler_gamma']
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]
    
    def get_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.params['center_crop']),
            transforms.Resize(self.params['img_size']),
            transforms.ToTensor(),
            transforms.Lambda(lambda X: 2 * X - 1.)
        ])
    

