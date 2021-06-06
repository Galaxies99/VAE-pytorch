import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn
from einops.layers.torch import Rearrange


class DFCVAE(nn.Module):
    def __init__(self, image_size, in_channels, alpha = 1, beta = 0.5, **kwargs):
        super(DFCVAE, self).__init__()
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.hidden_dim = kwargs.get('hidden_dim', [32, 64, 128, 256, 512])
        self.encoder_layer_num = len(self.hidden_dim)
        self.zipped_size = 2 ** self.encoder_layer_num
        self.alpha = alpha
        self.beta = beta

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

        self.feature_extractor = vgg19_bn(pretrained = True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

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
    
    def extract_feature(self, x, feature_layers = ['14', '24', '34', '43']):
        features = []
        for (key, module) in self.feature_extractor.features._modules.items():
            x = module(x)
            if key in feature_layers:
                features.append(x)
        return features
    
    def forward(self, x, **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)

        x_feature = self.extract_feature(x)
        recon_feature = self.extract_feature(recon)
        return [recon, x, mu, log_var, x_feature, recon_feature]
    
    def loss(self, recon, x, mu, log_var, x_feature, recon_feature, **kwargs):
        if 'kl_weight' not in kwargs.keys():
            raise AttributeError('Please pass parameter "kl_weight" into the loss function.')
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        recon_loss = F.mse_loss(recon, x)
        feature_loss = 0
        for (recon_f, x_f) in zip(recon_feature, x_feature):
            feature_loss += F.mse_loss(recon_f, x_f)
        kl_weight = kwargs['kl_weight']
        loss = self.alpha * kl_loss * kl_weight + self.beta * (recon_loss + feature_loss)
        return {
            'loss': loss,
            'reconstruction loss': recon_loss, 
            'feature loss': feature_loss,
            'kl loss': kl_loss
        }

    def sample(self, num, device, **kwargs):
        z = torch.randn(num, self.latent_dim).to(device)
        return self.decode(z)
    
    def reconstruct(self, x, **kwargs):
        return self.forward(x, **kwargs)[0]
