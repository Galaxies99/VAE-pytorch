import os
import argparse
import yaml
import numpy as np
from models.VAE import VAE
from exper import CelebAExperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'default.yaml'), help = 'Config File', type = str)

FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

logger = TestTubeLogger(
    save_dir = cfg_dict['logger']['save_dir'],
    name = cfg_dict['logger']['name']
)

model = VAE(**cfg_dict['model'])

experiment = CelebAExperiment(
    model,
    **cfg_dict['experiment']
)

trainer = Trainer(
    default_root_dir = logger.save_dir,
    min_epochs = 1,
    logger = logger,
    **cfg_dict['trainer']
)

print('--> Training VAE ...')
trainer.fit(experiment)
