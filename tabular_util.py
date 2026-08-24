import os
# import time
# import pdb
# from glob import glob
import torch
# from torch.nn.parameter import Parameter
# import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import scipy.misc as sci
# import scipy.ndimage
import shutil
# from skimage.measure import compare_psnr, compare_ssim
import sklearn.metrics
# import matplotlib as mpl
# import nibabel as nib
import h5py
import pandas as pd
import yaml
import copy


class LongitudinalPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = df.columns[-68:]

        # Group by subject
        self.groups = []
        for rid, subdf in self.df.groupby("RID"):
            subdf_sorted = subdf.sort_values("AGE")
            if len(subdf_sorted) > 1:
                # Generate all directed forward pairs (age1 < age2)
                rows = subdf_sorted.to_dict("records")
                for i in range(len(rows)):
                    for j in range(i+1, len(rows)):
                        self.groups.append((rid, rows[i], rows[j]))

    def __len__(self):
        return len(self.groups)
        # return 140

    def __getitem__(self, idx):
        rid, r1, r2 = self.groups[idx]

        tab1 = np.array([r1[c] for c in self.feature_cols], dtype=np.float32)
        tab2 = np.array([r2[c] for c in self.feature_cols], dtype=np.float32)

        return {
            "rid": str(rid),
            "tab1": tab1,
            "tab2": tab2,
            "lb1": int(r1["DXGrp"]),
            "lb2": int(r2["DXGrp"]),
            "age1": float(r1["AGE"]),
            "age2": float(r2["AGE"]),
        }


class LongitudinalSingleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = df.columns[-68:]
        # Convert to records so we can index directly
        self.records = self.df.to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        rid = r["RID"]
        tab = np.array([r[c] for c in self.feature_cols], dtype=np.float32)

        return {
            "rid": str(rid),
            "tab": tab,
            "lb": int(r["DXGrp"]),
            "age": float(r["AGE"]),
        }


class LongitudinalData(object):
    def __init__(self, data_dir, batch_size, shuffle):
        num_workers = 0

        train_df = pd.read_csv(os.path.join(data_dir, 'preadj_train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'preadj_val.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'preadj_test.csv'))

        self.train_dataset = LongitudinalPairDataset(train_df)
        self.val_dataset = LongitudinalPairDataset(val_df)
        self.test_dataset = LongitudinalPairDataset(test_df)

        self.trainLoader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.valLoader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.testLoader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# load config file from ckpt
def load_config_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return True, config
    else:
        return False, None

# save config file at the beginning of the training
def save_config_yaml(ckpt_path, config):
    yaml_path = os.path.join(ckpt_path, 'config.yaml')
    remove_key = []
    for key in config.keys():
        if isinstance(config[key], int) or isinstance(config[key], float) or isinstance(config[key], str) or isinstance(config[key], list)  or isinstance(config[key], dict):
            continue
        remove_key.append(key)
    config_copy = copy.deepcopy(config)
    for key in remove_key:
        config_copy.pop(key, None)
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(config_copy, file)
    print('Saved yaml file')

# load model/scheduler
def load_checkpoint_by_key(values, checkpoint_dir, keys, device, ckpt_name='model_best.pth.tar'):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = os.path.join(checkpoint_dir, ckpt_name)
    print(filename)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            try:
                if key == 'model':
                    values[i] = load_checkpoint_model(values[i], checkpoint[key])
                else:
                    values[i].load_state_dict(checkpoint[key])
                print('loading ' + key + ' success!')
            except:
                print('loading ' + key + ' failed!')
        print("loaded checkpoint from '{}' (epoch: {}, monitor metric: {})".format(filename, \
                epoch, checkpoint['monitor_metric']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch

# load each part of the model
def load_checkpoint_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

# save results statistics
def save_result_stat(stat, config, info='Default'):
    stat_path = os.path.join(config['ckpt_path'], 'stat.csv')
    columns=['info',] + sorted(stat.keys())
    if not os.path.exists(stat_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(stat_path, mode='a', header=True)

    stat['info'] = info
    for key, value in stat.items():
        stat[key] = [value]
    df = pd.DataFrame.from_dict(stat)
    df = df[columns]
    df.to_csv(stat_path, mode='a', header=False)

def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')
        print('save best')

def compute_classification_metrics(label, pred, dataset_name='ADNI', postfix='NC_AD', task='classification'):
    if task == 'age':
        r2 = sklearn.metrics.r2_score(label, pred)
        if dataset_name == 'LAB':
            label = label * 17.6 + 47.3
            pred = pred * 17.6 + 47.3
        if dataset_name == 'NCANDA':
            label = label * 3.4 + 19.5
            pred = pred * 3.4 + 19.5
        mse = sklearn.metrics.mean_squared_error(label, pred, squared=False)
        mae = np.abs(pred - label).mean()
        print(mse, r2, mae)
        return r2
    else:
        pred_bi = (pred>0.5).squeeze(1)
        if dataset_name == 'ADNI':
            if 'NC_AD' in postfix:
                classes = [0,2]
            elif 'pMCI_sMCI' in postfix:
                classes = [3,4]
        elif dataset_name == 'tabular':
            if 'NC_AD' in postfix:
                classes = [0,1]
            elif 'pMCI_sMCI' in postfix:
                classes = [2,3]
        elif dataset_name == 'LAB':
            if 'C_E_HE' in postfix:
                label = (label > 0)
                classes = [0,1]
        elif dataset_name == 'NCANDA':
            label = (label > 0)
            classes = [0,1]
        tp = np.sum(np.logical_and(label==classes[1], pred_bi==1))
        fp = np.sum(np.logical_and(label==classes[0], pred_bi==1))
        tn = np.sum(np.logical_and(label==classes[0], pred_bi==0))
        fn = np.sum(np.logical_and(label==classes[1], pred_bi==0))
        auc = sklearn.metrics.roc_auc_score(label==classes[1], pred.squeeze(1))
        sen = tp/(tp+fn)
        spe = tn/(tn+fp)
        bacc = 0.5 * (sen + spe)
        print(auc, bacc, sen, spe)
        return bacc
