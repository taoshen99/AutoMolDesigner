#!/usr/bin/env python3
import os
import json
import argparse
from utils.configurator import Configurator
from rnn_scripts.model import CLM

def main(template_path):
    print(f'Configure the model based on {os.path.basename(template_path)} ...')
    config = Configurator(template_path)
    config.configurate()
    print('Configuration done.')

    print('Initialize the model ...')
    DEVICE = config.check_device()
    clm = CLM(config, DEVICE, pattern='sample')
    clm.model.to(DEVICE)
    print('Initialization done.')

    print('Start sampling ...')
    sampled_smis = clm.sample()
    with open(os.path.join(clm.config.prj_dir, 'sampled_SMILES.smi'), "w") as f:
        for smi in sampled_smis:
            f.write(smi + "\n")
    print('Sampling done.')

    print('Saving config ...')
    clm.config.config_path = os.path.join(clm.config.prj_dir, 'config.json')
    with open(clm.config.config_path, 'w') as f:
        json.dump(clm.config.__dict__, f, indent=2)
    print('Config saved.')

if __name__ == '__main__':
    print('############ MODEL-SAMPLING ############')
    parser = argparse.ArgumentParser(description="Model Sampling")
    parser.add_argument("-t", "--template", type=str, default="template/template_sample.json", help="input sample template")
    args = parser.parse_args()
    main(args.template)
    print('############ MODEL-SAMPLING ############')