import os, pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def load_us_airline(path_to_data, dataset_name, load_type='raw'):
    """
    Load load_us_airline Dataset
    load_type:
        raw: load the raw text.
    """

    """
    加载文本数据
    load_type:
        ndarray: 直接加载ndarray
        process_to_tensor: 将ndarray处理为tensor, 并存储
        processed: 直接加载存储的tensor
    """

    # (1) load pkl data
    dataset_dir = os.path.join(path_to_data, dataset_name)
    train_pt = os.path.join(dataset_dir, 'processed_tensor/train_data.pt')
    test_pt = os.path.join(dataset_dir, 'processed_tensor/test_data.pt')

    if load_type in ['ndarray', 'process_to_tensor']:
        train_pkl = os.path.join(dataset_dir, 'processed_ndarray/train_data.pkl')
        test_pkl = os.path.join(dataset_dir, 'processed_ndarray/test_data.pkl')
        print(f"===> Loading processed ndarray from {train_pkl}, {test_pkl}")

        with open(train_pkl, 'rb') as f:
            train_data = pickle.load(f)
        if isinstance(train_data, dict):
            train_data = [train_data['data'], train_data['label']]
        with open(test_pkl, 'rb') as f:
            test_data = pickle.load(f)
        if isinstance(test_data, dict):
            test_data = [test_data['data'], test_data['label']]

        if load_type == 'ndarray':
            return train_data, test_data

    if load_type == 'process_to_tensor':
        print(f"===> Processing ndarray to Tensors")
        X_train, y_train = train_data
        X_test, y_test = test_data

        print("Shape of raw train_data:", X_train.shape)
        print("Shape of raw test_data:", X_test.shape)

        # (3) 转化为tensor
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        # (4) 存储到 .pt 文件
        torch.save((X_train_t, y_train_t), train_pt)
        torch.save((X_test_t, y_test_t), test_pt)

    print(f"===> Loading processed Tensors from {train_pt}, {test_pt}")

    X_train_t, y_train_t = torch.load(train_pt)
    X_test_t, y_test_t = torch.load(test_pt)

    train_set = US_Airline_Dataset(X_train_t, y_train_t)
    test_set = US_Airline_Dataset(X_test_t, y_test_t)

    return train_set, test_set


class US_Airline_Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = np.array(targets)
        # self.class_to_idx = {'0': 0, '1': 1, '2': 2}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]