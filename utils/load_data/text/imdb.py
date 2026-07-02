# -*-coding:utf-8-*-
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils.datasets import CustomTensorDataset
from .text_utils import *

def split_sms(dataset_dir):
    """Split sms dataset to train and test"""
    x_total_raw_path = os.path.join(dataset_dir, 'raw/original/sms_total_data.npy')
    y_total_raw_path = os.path.join(dataset_dir, 'raw/original/sms_total_label.npy')

    x_total = np.load(x_total_raw_path, allow_pickle=True)
    y_total = np.load(y_total_raw_path, allow_pickle=True)

    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=42)

    x_train_raw_path = os.path.join(dataset_dir, 'raw/train_data.npy')
    y_train_raw_path = os.path.join(dataset_dir, 'raw/train_label.npy')
    x_test_raw_path = os.path.join(dataset_dir, 'raw/test_data.npy')
    y_test_raw_path = os.path.join(dataset_dir, 'raw/test_label.npy')

    np.save(x_train_raw_path, x_train)
    np.save(y_train_raw_path, y_train)
    np.save(x_test_raw_path, x_test)
    np.save(y_test_raw_path, y_test)

    return

def load_imdb(path_to_data, dataset_name, load_type='raw'):
    """
    Load IMDB and SMS Dataset (modified from tdpr)
    load_type:
        raw: load the raw text.
    """
    # (1) load raw text

    if dataset_name == 'imdb':
        # download and save the raw text from datasets lib from huggingface
        dataset_dir = os.path.join(path_to_data, f'imdb')
        padding_size = 500
    elif dataset_name == 'sms':
        # download the raw text from
        dataset_dir = os.path.join(path_to_data, f'sms')
        padding_size = 55
        # split sms to train and test set
        if not os.path.exists(os.path.join(dataset_dir, 'raw/train_data.npy')):
            split_sms(dataset_dir)


    if load_type in ['raw']:
        x_train_raw_path = os.path.join(dataset_dir, 'raw/train_data.npy')
        y_train_raw_path = os.path.join(dataset_dir, 'raw/train_label.npy')
        x_test_raw_path = os.path.join(dataset_dir, 'raw/test_data.npy')
        y_test_raw_path = os.path.join(dataset_dir, 'raw/test_label.npy')
        print(f"===> Loading raw text from \n    {x_train_raw_path}, \n    {y_train_raw_path}"
              f"\n    {x_test_raw_path}, \n    {y_test_raw_path}")

        x_train = np.load(x_train_raw_path, allow_pickle=True)
        y_train = np.load(y_train_raw_path, allow_pickle=True)
        x_test = np.load(x_test_raw_path, allow_pickle=True)
        y_test = np.load(y_test_raw_path, allow_pickle=True)

    # (2) Process data
    x_train_pad_path = os.path.join(dataset_dir, 'processed/train_data_pad.npy')
    y_train_pad_path = os.path.join(dataset_dir, 'processed/train_label_pad.npy')
    x_test_pad_path = os.path.join(dataset_dir, 'processed/test_data_pad.npy')
    y_test_pad_path = os.path.join(dataset_dir, 'processed/test_label_pad.npy')

    if os.path.exists(x_test_pad_path) or load_type == 'processed':
        # Load tockenized data
        x_train_pad = np.load(x_train_pad_path, allow_pickle=True)
        y_train_pad = np.load(y_train_pad_path, allow_pickle=True)
        x_test_pad = np.load(x_test_pad_path, allow_pickle=True)
        y_test_pad = np.load(y_test_pad_path, allow_pickle=True)

    else:  # Process and save tockenized data
        x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)

        x_train_pad = padding_(x_train, padding_size).astype(np.int32)
        y_train_pad = y_train
        x_test_pad = padding_(x_test, padding_size).astype(np.int32)
        y_test_pad = y_test

        np.save(x_train_pad_path, x_train_pad)
        np.save(y_train_pad_path, y_train_pad)
        np.save(x_test_pad_path, x_test_pad)
        np.save(y_test_pad_path, y_test_pad)

    train_set = CustomTensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train_pad).long())
    test_set = CustomTensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test_pad).long())

    return train_set, test_set