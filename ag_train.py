import os
import json
import platform
import pathlib
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from utils.features import gen_features
from utils.configurator import Configurator
from autogluon.tabular import TabularPredictor


def build_df(train_data_path: str):
    _, sfx = os.path.splitext(train_data_path)
    if sfx == '.csv':
        df_train = pd.read_csv(train_data_path)
    else:
        df_train = pd.read_excel(train_data_path)
    return df_train

def smis2fps(smis: list, features='ecfp4'):
    fps = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        fp = gen_features(mol, features=features)
        fps.append(fp)
    return np.array(fps)

def gen_train_data(full_df: pd.DataFrame, config):
    if config.test_size != 0:
        print(f'{int(config.test_size * 100)}% of full data will be will be held out for testing.')
        X, y = list(full_df['SMILES']), list(full_df['label'])
        if len(set(y)) == 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=42, shuffle=True, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=42, shuffle=True)            
        y_train_s, y_test_s = pd.Series(y_train, name='label'), pd.Series(y_test, name='label')
        X_train_smis, X_test_smis = pd.Series(X_train, name='SMILES'), pd.Series(X_test, name='SMILES')    

        df_train = pd.concat([X_train_smis, y_train_s], axis=1)
        df_train.to_csv(os.path.join(config.prj_dir, 'ag_train.csv'), index=False)
        df_test = pd.concat([X_test_smis, y_test_s], axis=1)
        df_test.to_csv(os.path.join(config.prj_dir, 'ag_test.csv'), index=False)
    else:
        print(f'You are using AutoGluon to train a complete model. Full data will be used for training.')
        df_train = full_df.sample(frac=1.0).reset_index(drop=True)
    train_smis = list(df_train['SMILES'])
    fps_np = smis2fps(train_smis, features=config.features)
    df_fps_np = pd.DataFrame(fps_np)
    df_fps_np_label = pd.concat([df_fps_np, df_train['label']], axis=1)
    return df_fps_np_label

if __name__ == "__main__":
    print('############ AUTOGLUON-TRAINING ############')
    parser = argparse.ArgumentParser(description="perform model training using AutoGluon")
    parser.add_argument("-t", "--template", type=str, default="template/template_ag_train.json",
                         help="input AutoGluon training template")
    args = parser.parse_args()
    if platform.system() == "Windows":
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    print(f'Configure the training settings based on {os.path.basename(args.template)} ...')
    config = Configurator(args.template)
    config.configurate()
    print('Configuration done.')
    print('Prepare the datasets for training ...')
    df_all = build_df(config.train_data)
    df_all.to_csv(os.path.join(config.prj_dir, 'ag_train_full.csv'), index=False)
    df_train_fps_label = gen_train_data(df_all, config)
    print('Preparation done.')
    print('Start training ...')
    if config.opt_deploy:
        predictor = TabularPredictor(
            label='label', 
            path=config.model_dir, 
            eval_metric=config.eval_metric).fit(
            train_data=df_train_fps_label,
            time_limit=config.time_limit,
            presets=[config.quality, "optimize_for_deployment"]
        )
    else:
        predictor = TabularPredictor(
            label='label', 
            path=config.model_dir, 
            eval_metric=config.eval_metric).fit(
            train_data=df_train_fps_label,
            time_limit=config.time_limit,
            presets=config.quality
        )
    print('Training done.')
    print('Saving config ...')
    config.config_path = os.path.join(config.prj_dir, 'config.json')
    with open(config.config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print('Config saved.')
    print('############ AUTOGLUON-TRAINING ############')