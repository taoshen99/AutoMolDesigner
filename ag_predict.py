#!/usr/bin/env python
import os
import time
import json
import platform
import argparse
import pathlib
import numpy as np
import pandas as pd
from rdkit import Chem
from utils.configurator import Configurator
from utils.features import gen_features
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, accuracy_score, \
                            f1_score, confusion_matrix, \
                            matthews_corrcoef


def read_smis2mols(predict_data_path: str):
    file_type = os.path.splitext(predict_data_path)[1]
    if file_type == ".smi":
        mols = []
        with open(predict_data_path, "r") as f:
            for l in f:
                smi = l.rstrip()
                mol = Chem.MolFromSmiles(smi)
                mols.append(mol)
        return mols, []
    elif file_type == ".csv":
        df_test = pd.read_csv(predict_data_path)
        test_smis = list(df_test['SMILES'])
        mols = [Chem.MolFromSmiles(smi) for smi in test_smis]
        labels = list(df_test['Type'])
        return mols, labels
    else:
        df_test = pd.read_excel(predict_data_path)
        test_smis = list(df_test['SMILES'])
        mols = [Chem.MolFromSmiles(smi) for smi in test_smis]
        labels = list(df_test['label'])
        return mols, labels

def gen_predict_df(mols: list, labels: list, features='ecfp4'):
    fps = [gen_features(m, features=features) for m in mols]
    fps_np = np.array(fps)
    df_fps = pd.DataFrame(fps_np)
    if labels:
        s_label = pd.Series(labels, name='label')
        df_fps_label = pd.concat([df_fps, s_label], axis=1)
        return df_fps_label
    else:
        smis = [Chem.MolToSmiles(x) for x in mols]
        s_smis = pd.Series(smis, name='SMILES')
        df_fps_smis = pd.concat([df_fps, s_smis], axis=1)
        return df_fps_smis 

def metric_cls(y_true: list, y_pred: list):
    thre_f1 = []
    for thre in np.arange(0, 1.01, 0.01):
        preds_bi = [1 if x >=thre else 0 for x in y_pred]
        f1 = f1_score(y_true, preds_bi)
        thre_f1.append([thre, f1])
    thre_f1 = np.array(thre_f1)
    maxthre_idx = np.argmax(thre_f1, axis=0)[1]
    maxthre = thre_f1[maxthre_idx, 0]
    preds_bi = [1 if x >=maxthre else 0 for x in y_pred]

    acc_score = accuracy_score(y_true, preds_bi)
    roc_auc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, preds_bi)
    f1 = f1_score(y_true, preds_bi)
    conf_mtx = confusion_matrix(y_true, preds_bi)

    tn, fp, fn, tp = conf_mtx[0][0], conf_mtx[0][1], conf_mtx[1][0], conf_mtx[1][1]
    eval_metric = {'TN':tn,'FP':fp,'FN':fn,'TP':tp,
    'Accuracy':acc_score, 'AUROC':roc_auc, 'MCC':mcc, 'F1 score':f1}

    return eval_metric

if __name__ == "__main__":
    print('############ AUTOGLUON-PREDICTING ############')
    parser = argparse.ArgumentParser(description="predicting new data (or benchmarking) using AutoGluon")
    parser.add_argument("-t", "--template", type=str, default="template/template_ag_predict.json",
                         help="input AutoGluon predicting template")
    args = parser.parse_args()
    if platform.system() == "Windows":
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    print(f'Configure the predicting settings based on {os.path.basename(args.template)} ...')
    config = Configurator(args.template)
    config.configurate()
    print('Configuration done.')
    print('Prepare the datasets for predicting ...')
    ms, labels = read_smis2mols(config.smiles_dir)
    predict_df = gen_predict_df(ms, labels, config.features)
    print('Preparation done.')
    print(f'Load AutoGluon models from {config.model_dir} ...')
    predictor = TabularPredictor.load(config.model_dir, require_py_version_match=False, require_version_match=False)
    print('Loading done.')
    if 'label' in predict_df.columns:
        print(f'You are using AutoGluon for benchmarking. The benchmarking results will be saved as {os.path.join(config.prj_dir,"metrics.csv")}.')
        if len(set(list(predict_df['label']))) == 2:
            print('Your task is classification.')
            predict_df_droplast = predict_df.drop(columns=['label'])
            time_start = time.time()
            preds_df = predictor.predict_proba(predict_df_droplast)
            time_done = time.time()
            print(f'Predicting done. Time elapsed: {(time_done - time_start):.3f} seconds.\n')
            y_true = list(predict_df['label'])
            preds = list(preds_df.iloc[:, 1])
            perf_metric = metric_cls(y_true=y_true, y_pred=preds)       
            df_perf_metric = pd.DataFrame(perf_metric, index=[0])
        else:
            print('Your task is regression.')                
            time_start = time.time()
            perf = predictor.evaluate(predict_df)
            time_done = time.time()
            print(f'Predicting done. Time elapsed: {(time_done - time_start):.3f} seconds.\n')
            perf['mean_absolute_error'] = -perf['mean_absolute_error']
            perf['root_mean_squared_error'] = -perf['root_mean_squared_error']
            perf['mean_squared_error'] = -perf['mean_squared_error']
            perf['median_absolute_error'] = -perf['median_absolute_error']
            df_perf_metric = pd.DataFrame(perf, index=[0])
        df_perf_metric.to_csv(os.path.join(config.prj_dir,"metrics.csv"), index=False)
    else:
        print(f'You are using AutoGluon to predict new data. The predicting results will be saved as {os.path.join(config.prj_dir, "pred_results.csv")} and {os.path.join(config.prj_dir, "pred_results.sdf")}.')
        if predictor.can_predict_proba:
            print('Your task is classification.')
            predict_df_droplast = predict_df.drop(columns=['SMILES'])
            time_start = time.time()
            preds_df = predictor.predict_proba(predict_df_droplast)
            time_done = time.time()
            print(f'Predicting done. Time elapsed: {(time_done - time_start):.3f} seconds.\n')
            df_pred_smis = pd.concat([predict_df['SMILES'], preds_df.iloc[:, 1].rename('Score')], axis=1)
            df_pred_smis.to_csv(os.path.join(config.prj_dir,'pred_results.csv'), index=False)
            with Chem.SDWriter(os.path.join(config.prj_dir,'pred_results.sdf')) as w:
                for idx, row in df_pred_smis.iterrows():
                    smi = row['SMILES']
                    m = Chem.MolFromSmiles(smi)
                    score = round(row['Score'], 3)
                    m.SetProp('_Name', 'mol'+str(idx))
                    m.SetProp('SMILES', smi)
                    m.SetDoubleProp('Score', score)
                    w.write(m)
        else:
            print('Your task is regression.')
            predict_df_droplast = predict_df.drop(columns=['SMILES'])
            time_start = time.time()
            preds_df = predictor.predict(predict_df_droplast)
            time_done = time.time()
            print(f'Predicting done. Time elapsed: {(time_done - time_start):.3f} seconds.\n')
            df_pred_smis = pd.concat([predict_df['SMILES'], preds_df.rename('Score')], axis=1)
            df_pred_smis.to_csv(os.path.join(config.prj_dir,'pred_results.csv'), index=False)
            with Chem.SDWriter(os.path.join(config.prj_dir,'pred_results.sdf')) as w:
                for idx, row in df_pred_smis.iterrows():
                    smi = row['SMILES']
                    m = Chem.MolFromSmiles(smi)
                    score = round(row['Score'], 3)
                    m.SetProp('_Name', 'mol'+str(idx))
                    m.SetProp('SMILES', smi)
                    m.SetDoubleProp('Score', score)
                    w.write(m)   
    
    print('Saving config ...')
    config.config_path = os.path.join(config.prj_dir, 'config.json')
    with open(config.config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print('Config saved.')
    print('############ AUTOGLUON-PREDICTING ############')