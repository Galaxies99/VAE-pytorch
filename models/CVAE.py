import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange


class CVAE(nn.Module):
    def __init__(self, image_size, in_channels, **kwargs):
        super(CVAE, self).__init__()
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.hidden_dim = kwargs.get('hidden_dim', [32, 64, 128, 256, 512])
        self.num_classes = kwargs.get('num_classes', 40)
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
        self.image_embedding = nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 1)
        self.label_embedding = nn.Linear(self.num_classes, self.image_H * self.image_W)

        encoder_layers = []
        last_channels = self.in_channels + 1

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
            nn.Linear(self.latent_dim + self.num_classes, self.hidden_dim[-1] * self.flatten_size),
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
        assert C == self.in_channels + 1 and H == self.image_H and W == self.image_W
        latent_var = self.encoder(x)
        return [self.mu(latent_var), self.log_var(latent_var)]
    
    def decode(self, z):
        return self.final(self.decoder(z))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x, **kwargs):
        y = kwargs['labels'].float()
        input = torch.cat([
            self.image_embedding(x),
            rearrange(self.label_embedding(y), 'b (c h w) -> b c h w', c = 1, h = self.image_H, w = self.image_W)
        ], dim = 1)
        mu, log_var = self.encode(input)
        z = torch.cat([
            self.reparameterize(mu, log_var),
            y
        ], dim = 1)
        return [self.decode(z), x, mu, log_var]
    
    def loss(self, recon, x, mu, log_var, **kwargs):
        if 'kl_weight' not in kwargs.keys():
            raise AttributeError('Please pass parameter "kl_weight" into the loss function.')
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        recon_loss = F.mse_loss(recon, x)
        kl_weight = kwargs['kl_weight']
        loss = kl_loss * kl_weight + recon_loss
        return {
            'loss': loss,
            'reconstruction loss': recon_loss, 
            'kl loss': kl_loss
        }

    def sample(self, num, device, **kwargs):
        y = kwargs['labels'].float()
        z = torch.randn(num, self.latent_dim).to(device)
        z = torch.cat([
            torch.randn(num, self.latent_dim).to(device),
            y
        ], dim = 1)
        return self.decode(z)
    
    def reconstruct(self, x, **kwargs):
        return self.forward(x, **kwargs)[0]
