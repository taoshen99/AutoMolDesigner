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

    print('Prepare the torch dataset for finetuning ...')
    finetune_set = SMILESDataset(config, pattern="finetune")
    train_set_size = int(len(finetune_set) * config.train_split)
    valid_set_size = len(finetune_set) - train_set_size
    train_set, valid_set = random_split(finetune_set, [train_set_size, valid_set_size])
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
    
    config = finetune_set.get_config()
    print('Preparation done.')

    print('Initialize the model ...')
    DEVICE = config.check_device()
    clm = CLM(config, DEVICE, pattern='finetune')
    clm.model.to(DEVICE)
    print('Initialization done.')

    print('Start finetuning ...')
    clm.finetune(train_loader, valid_loader)
    print('Finetuning done.')

    print('Saving config ...')
    clm.config.config_path = os.path.join(clm.config.prj_dir, 'config.json')
    with open(clm.config.config_path, 'w') as f:
        json.dump(clm.config.__dict__, f, indent=2)
    print('Config saved.')

if __name__ == "__main__":
    print('############ MODEL-FINETUNING ############')
    parser = argparse.ArgumentParser(description="Model Finetuning")
    parser.add_argument("-t", "--template", type=str, default="template/template_finetune.json", help="input finetune template")
    args = parser.parse_args()
    main(args.template)
    print('############ MODEL-FINETUNING ############')