#!/usr/bin/env python3
import os
import json
import argparse
from utils.configurator import Configurator
from rnn_scripts.dataset import SMILESDataset
from rnn_scripts.model import CLM
from torch.utils.data import DataLoader, random_split


def main(template_path):
    print(f'Configure the model based on {os.path.basename(template_path)} ...')
    config = Configurator(template_path)
    config.configurate()
    print('Configuration done.')

    print('Prepare the torch datasets for training and validating ...')
    pretrain_set = SMILESDataset(config, pattern="pretrain")
    train_set_size = int(len(pretrain_set) * config.train_split)
    valid_set_size = len(pretrain_set) - train_set_size
    train_set, valid_set = random_split(pretrain_set, [train_set_size, valid_set_size])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    
    config = pretrain_set.get_config()
    print('Preparation done.')

    print('Initialize the model ...')
    DEVICE = config.check_device()
    clm = CLM(config, DEVICE, pattern='pretrain')
    clm.model.to(DEVICE)
    print('Initialization done.')

    print('Start pretraining ...')
    clm.pretrain(train_loader, valid_loader)
    print('Pretraining done.')

    print('Saving config ...')
    clm.config.config_path = os.path.join(clm.config.prj_dir, 'config.json')
    with open(clm.config.config_path, 'w') as f:
        json.dump(clm.config.__dict__, f, indent=2)
    print('Config saved.')

if __name__ == '__main__':
    print('############ MODEL-PRETRAINING ############')
    parser = argparse.ArgumentParser(description="Model Pretraining")
    parser.add_argument("-t", "--template", type=str, default="template/template_pretrain.json", help="input pretrain template")
    args = parser.parse_args()
    main(args.template)
    print('############ MODEL-PRETRAINING ############')