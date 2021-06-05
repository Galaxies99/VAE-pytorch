import os
import yaml
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from utils.logger import ColoredLogger
from datasets.celeba import CelebADataset
from torch.utils.data import DataLoader
import torchvision.utils as tuitls
from models.VAE import VAE
from models.CVAE import CVAE
from models.BetaVAE import BetaVAE
from models.DisentangledBetaVAE import DisentangledBetaVAE


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default = 'train', help = 'the running mode, "train" or "inference"', type = str)
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'VAE.yaml'), help = 'Config File', type = str)
parser.add_argument('--clean_cache', '-cc', action = 'store_true', help = 'whether to clean the cache of GPU while training, evaluation and testing')
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
CLEAN_CACHE = FLAGS.clean_cache
MODE = FLAGS.mode

if MODE not in ['train', 'test']:
    raise AttributeError('mode should be either "train" or "inference".')

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
model_params = cfg_dict.get('model', {})
dataset_params = cfg_dict.get('dataset', {})
optimizer_params = cfg_dict.get('optimizer', {})
scheduler_params = cfg_dict.get('scheduler', {})
trainer_params = cfg_dict.get('trainer', {})
inferencer_params = cfg_dict.get('inferencer', {})
stats_params = cfg_dict.get('stats', {})

logger.info('Building Models ...')
model_name = model_params.get('name', 'VAE')
if model_name == 'VAE':
    model = VAE(**model_params)
elif model_name == 'CVAE':
    model = CVAE(**model_params)
elif model_name == 'BetaVAE':
    model = BetaVAE(**model_params)
elif model_name == 'DisentangledBetaVAE':
    model = DisentangledBetaVAE(**model_params)
else:
    raise NotImplementedError('Invalid model name.')

multigpu = trainer_params.get('multigpu', False)
if multigpu:
    logger.info('Initialize multi-gpu training ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

if dataset_params.get('type', 'CelebA') == 'CelebA':
    logger.info('Building datasets ...')
    train_dataset = CelebADataset(
        root = dataset_params.get('path', 'data'),
        split = 'train',
        img_size = dataset_params.get('img_size', 64),
        center_crop = dataset_params.get('center_crop', 148),
        download = False
    )
    val_dataset = CelebADataset(
        root = dataset_params.get('path', 'data'),
        split = 'test',
        img_size = dataset_params.get('img_size', 64),
        center_crop = dataset_params.get('center_crop', 148),
        download = False
    )

logger.info('Building dataloader ...')
batch_size = dataset_params.get('batch_size', 144)
train_dataloader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
    drop_last = True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
    drop_last = True
)

total_train_samples = len(train_dataset)
total_val_samples = len(val_dataset)

logger.info('Building optimizer and learning rate scheduler ...')
optimizer = torch.optim.AdamW(
    model.parameters(),
    betas = (optimizer_params.get('adam_beta1', 0.9), optimizer_params.get('adam_beta2', 0.999)),
    lr = optimizer_params.get('learning_rate', 0.005),
    weight_decay = optimizer_params.get('weight_decay', 0.01),
    eps = optimizer_params.get('eps', 1e-6)
)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma = scheduler_params.get('gamma', 0.95)
)

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = trainer_params.get('max_epoch', 50)
stats_dir = os.path.join(stats_params.get('stats_dir', 'stats'), stats_params.get('stats_folder', 'temp'))
if os.path.exists(stats_dir) == False:
    os.makedirs(stats_dir)
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    lr_scheduler.last_epoch = start_epoch - 1
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
elif MODE == "inference":
    raise AttributeError('There should be a checkpoint file for inference.')

multigpu = trainer_params.get('multigpu', False)
if multigpu:
    model = torch.nn.DataParallel(model)


def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    with tqdm(train_dataloader) as pbar:
        for data in pbar:
            if CLEAN_CACHE and device != torch.device('cpu'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            res = model(x, labels = labels)
            loss_dict = model.loss(
                *res,
                kl_weight = batch_size / total_train_samples
            )
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}.'.format(epoch + 1, mean_loss))


def eval_one_epoch(epoch):
    logger.info('Start evaluation process in epoch {}.'.format(epoch + 1))
    model.eval()
    losses = []
    with tqdm(val_dataloader) as pbar:
        for data in pbar:
            if CLEAN_CACHE and device != torch.device('cpu'):
                torch.cuda.empty_cache()
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                res = model(x, labels = labels)
                loss_dict = model.loss(
                    *res,
                    kl_weight = batch_size / total_val_samples
                )
                loss = loss_dict['loss']
            pbar.set_description('Eval epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    logger.info('Finish evaluation process in epoch {}, mean evaluation loss: {:.8f}'.format(epoch + 1, mean_loss))
    return mean_loss


def inference(epoch = -1):
    suffix = ""
    if 0 <= epoch < max_epoch:
        logger.info('Begin inference on checkpoint of epoch {} ...'.format(epoch + 1))
        suffix = "epoch_{}".format(epoch)
    else:
        logger.info('Begin inference ...')
    x, labels = next(iter(val_dataloader))
    x = x.to(device)
    labels = labels.to(device)
    recon = model.reconstruct(x, labels = labels)
    nrow = int(np.ceil(np.sqrt(batch_size)))
    reconstructed_dir = os.path.join(stats_dir, 'reconstructed_images')
    generated_dir = os.path.join(stats_dir, 'generated_images')
    if os.path.exists(reconstructed_dir) == False:
        os.makedirs(reconstructed_dir)
    if os.path.exists(generated_dir) == False:
        os.makedirs(generated_dir)
    tuitls.save_image(
        x.data,
        os.path.join(reconstructed_dir, "original_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )
    tuitls.save_image(
        recon.data,
        os.path.join(reconstructed_dir, "reconstructed_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )
    samples = model.sample(inferencer_params.get('sample_num', 144), device, labels = labels)
    tuitls.save_image(
        samples.data,
        os.path.join(generated_dir, "generated_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )
    

def train(start_epoch):
    global cur_epoch
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch
        logger.info('--> Epoch {}/{}, learning rate: {}'.format(epoch + 1, max_epoch, lr_scheduler.get_last_lr()[0]))
        train_one_epoch(epoch)
        loss = eval_one_epoch(epoch)
        lr_scheduler.step()
        if multigpu is False:
            save_dict = {
                'epoch': epoch + 1, 
                'loss': loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch + 1, 
                'loss': loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.module.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))
        inference(epoch)


if __name__ == '__main__':
    if MODE == "train":
        train(start_epoch)
    else:
        inference()