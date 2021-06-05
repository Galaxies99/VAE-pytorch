import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange


class BetaTCVAE(nn.Module):
    def __init__(self, image_size, in_channels, beta, capacity_max_iter, **kwargs):
        super(BetaTCVAE, self).__init__()
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.hidden_dim = kwargs.get('hidden_dim', [32, 64, 128, 256, 512])
        self.alpha = kwargs.get('alpha', 1)
        self.beta = beta
        self.gamma = kwargs.get('gamma', 1)
        self.capacity_max_iter = capacity_max_iter
        self.iter = 0
        self.encoder_layer_num = len(self.hidden_dim)
        self.zipped_size = 2 ** self.encoder_layer_num

        if isinstance(image_size, int):
            self.image_H, self.image_W = image_size, image_size
        elif isinstance(image_size, tuple) and len(image_size) == 2 \
            and isinstance(image_size[0], int) and isinstance(image_size[1], int):
            self.image_H, self.image_W = image_size[0], image_size[1]
        else:
            raise AttributeError('Invalid attribute of image_size, image_size should be int or tuple of (int, int).')
        
        if isinstance(in_channels, int):
            self.in_channels = in_channels
        else:
            raise AttributeError('Invalid attribute of in_channels, in_channels should be int.')

        if self.image_H % self.zipped_size != 0 or self.image_W % self.zipped_size != 0:
            raise AttributeError('The size of image should be divided by {}'.format(self.zipped_size))

        # Encoder of VAE
        encoder_layers = []
        last_channels = self.in_channels

        for channels in self.hidden_dim:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(last_channels, channels, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU()
                )
            )
            last_channels = channels
        
        self.flatten_H = self.image_H // self.zipped_size
        self.flatten_W = self.image_W // self.zipped_size
        self.flatten_size = self.flatten_H * self.flatten_W

        self.encoder = nn.Sequential(
            *encoder_layers, 
            Rearrange('b c h w -> b (c h w)')
        )
        self.mu = nn.Linear(self.hidden_dim[-1] * self.flatten_size, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_dim[-1] * self.flatten_size, self.latent_dim)

        # Decoder of VAE
        decoder_layers = []
        last_channels = self.hidden_dim[-1]

        for i in range(len(self.hidden_dim) - 1, 0, -1):
            prev_channels = self.hidden_dim[i - 1]
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(last_channels, prev_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(prev_channels),
                    nn.LeakyReLU()
                )
            )
            last_channels = prev_channels
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim[-1] * self.flatten_size),
            Rearrange('b (c h w) -> b c h w', h = self.flatten_H, w = self.flatten_W),
            *decoder_layers
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(last_channels, last_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(last_channels),
            nn.LeakyReLU(),
            nn.Conv2d(last_channels, self.in_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.Tanh()
        )

    def encode(self, x):
        # x: B * C * H * W, double check
        _, C, H, W = x.shape
        assert C == self.in_channels and H == self.image_H and W == self.image_W
        latent_var = self.encoder(x)
        return [self.mu(latent_var), self.log_var(latent_var)]
    
    def decode(self, z):
        return self.final(self.decoder(z))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x, **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var, z]
    
    @staticmethod
    def log_gaussian_density(x, mu, log_var):
        return -0.5 * (np.log(2 * np.pi) + log_var) - 0.5 * ((x - mu) ** 2 * torch.exp(- log_var))

    def loss(self, recon, x, mu, log_var, z, **kwargs):
        if 'batch_size' not in kwargs.keys():
            raise AttributeError('Please pass parameter "batch_size" into the loss function.')
        if 'dataset_size' not in kwargs.keys():
            raise AttributeError('Please pass parameter "dataset_size" into the loss function.')
        batch_size = kwargs['batch_size']
        dataset_size = kwargs['dataset_size']
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        recon_loss = F.mse_loss(recon, x)
        log_q_zx = self.log_gaussian_density(z, mu, log_var).sum(dim = 1)
        log_p_z = self.log_gaussian_density(z, torch.zeros_like(mu), torch.zeros_like(log_var)).sum(dim = 1)
        # Mini-batch weighted sampling
        mat_log_q_z = self.log_gaussian_density(
            rearrange(z, 'b d -> b 1 d'),
            rearrange(mu, 'b d -> 1 b d'),
            rearrange(log_var, 'b d -> 1 b d')
        )
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        imp_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(x.device)
        imp_weights.view(-1)[::batch_size] = 1 / dataset_size
        imp_weights.view(-1)[1::batch_size] = strat_weight
        imp_weights[batch_size - 2, 0] = strat_weight
        log_iw_mat = torch.log(imp_weights)
        mat_log_q_z += rearrange(log_iw_mat, 'b bb -> b bb 1')
        log_q_z = torch.logsumexp(mat_log_q_z.sum(dim = 2), dim = 1)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim = 1).sum(dim = 1)
        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()
        if self.training:
            rate = min(self.iter, self.capacity_max_iter) / self.capacity_max_iter
            self.iter += 1
        else:
            rate = 1
        loss = recon_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * rate * kld_loss
        return {
            'loss': loss,
            'reconstruction loss': recon_loss, 
            'kl loss': kld_loss,
            'tc loss': tc_loss,
            'mi loss': mi_loss
        }

    def sample(self, num, device, **kwargs):
        z = torch.randn(num, self.latent_dim).to(device)
        return self.decode(z)
    
    def reconstruct(self, x, **kwargs):
        return self.forward(x, **kwargs)[0]
