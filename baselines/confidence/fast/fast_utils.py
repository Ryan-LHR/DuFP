from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from utils.load_data.load_data import load_loader
from utils.model_utils import extract_layer_output, flatten_layer_output
from utils.data_utils import VectorDataset

def construct_trusted_subset(train_set, train_truths, train_vectors, train_correct,
                             conf_thres, batch_size, args):
    """Construct subset of train set with high confidence for Contribution Measurement"""
    class_to_indices = {}

    sample = train_set[0]
    sample_length = len(sample)
    if sample_length == 2:
        IS_HF_DATASET = False
    else:
        IS_HF_DATASET = True

    if args.dataset in ['imagenet_100']:
        IS_IN_DATASET = True
    else:
        IS_IN_DATASET = False

    for idx, (vector, correct) in tqdm(enumerate(zip(train_vectors, train_correct)),
                                       total=len(train_vectors), desc="Extract labels"):
        if correct:
            confidence = torch.max(vector).item() if isinstance(vector, torch.Tensor) else max(vector)
            if confidence > conf_thres:
                if not IS_HF_DATASET:  # for standard image datasets
                    _, label = train_set[idx]

                else:  # for hf text datasets
                    label = train_truths[idx]
                # class_to_indices[label].append(idx)
                class_to_indices.setdefault(label, []).append(idx)

    loaders_dict = {}
    for label, indices in tqdm(class_to_indices.items(), desc="Split subset"):
        subset = Subset(train_set, indices)
        if not IS_HF_DATASET and not IS_IN_DATASET:  # for standard image datasets
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        else:  # for hf text datasets
            kwargs = {'model_file': args.model_file, 'method': 'fast'}
            dataloader = load_loader(args.dataset, subset, batch_size, args.n_workers, **kwargs)
        loaders_dict[label] = dataloader

    return loaders_dict, class_to_indices

def construct_prob_dict(class_to_indices, train_prob):
    """Construct probabilities ndarray dict of loaders_dict"""
    prob_dict = {}
    for label, indices in tqdm(class_to_indices.items(), desc="Collect probabilities"):
        prob_dict[label] = train_prob[indices]

    return prob_dict


class FAST:
    """
    FAST (official implementation of FAST, modified for torch framework)
    """
    def __init__(self, model_name, model, layer_index, layer_name, classes, device):
        self.model_name = model_name
        self.classes = classes
        # self.layer_index = layer_index
        self.layer_name = layer_name
        self.class_freqs = []
        self.class_ns = []
        self.model = model
        self.device = device

    def get_remaining_model(self, model):
        """manually add remaining layers"""
        remaining_layers = OrderedDict()

        if self.model_name in ["MNIST-LeNet1", 'FM-LeNet1']:
            remaining_layers['softmax'] = model.softmax

        elif self.model_name in ["MNIST-LeNet5", "SVHN-LeNet5"]:
            remaining_layers['relu1'] = model.relu
            remaining_layers['fc2'] = model.fc2
            remaining_layers['relu2'] = model.relu
            remaining_layers['fc3'] = model.fc3
            remaining_layers['softmax'] = model.softmax

        elif self.model_name in ["C10-VGG16", "SVHN-VGG16"]:
            remaining_layers['relu1'] = model.relu
            remaining_layers['fc2'] = model.fc2
            remaining_layers['relu2'] = model.relu
            remaining_layers['fc3'] = model.fc3
            remaining_layers['softmax'] = model.softmax

        elif self.model_name in ["FM-ResNet20", "C10-ResNet20"]:
            remaining_layers['softmax'] = model.softmax

        elif self.model_name in ["IMDB-Transformer"]:
            remaining_layers['fc'] = model.fc
            remaining_layers['softmax'] = model.softmax

        elif self.model_name in ["IM100-FastViT_T8", "IM100-FastViT_S12"]:
            remaining_layers['head'] = model.head

        elif self.model_name in ["SMS-BiLSTM"]:
            remaining_layers['fc'] = model.fc

        elif self.model_name in ["C100-ShuffleNetv2_x1_5"]:
            # remaining_layers['globalpool'] = nn.AdaptiveAvgPool2d((1, 1))
            # remaining_layers['flatten'] = nn.Flatten(start_dim=1)
            remaining_layers['fc'] = model.fc

        elif self.model_name in ["DBPedia-Bert_base_uncased", "AGNews-Bert"]:
            remaining_layers['classifier'] = model.hf_model.classifier
            remaining_layers['softmax'] = nn.Softmax(dim=-1)

        elif self.model_name in ["IM100Test-deit_base_patch16_224"]:
            remaining_layers['softmax'] = nn.Softmax(dim=-1)
        else:
            raise ValueError("Model not found")

        return nn.Sequential(remaining_layers)

    def fetch_ns_torch(self, loaders_dict, prob_dict):
        """Contribution Measurement"""
        remaining_layers = self.get_remaining_model(self.model)
        for c in tqdm(range(self.classes), desc="fetch ns torch"):

            # debug for key error
            if c not in loaders_dict or c not in prob_dict:
                if len(self.class_ns) > 0:
                    self.class_ns.append(np.zeros_like(self.class_ns[-1]))
                continue

            # get intermediate outputs
            dataloader = loaders_dict[c]
            hidden_layer_values = extract_layer_output(self.model, dataloader, self.layer_name)

            hidden_layer_values = process_feature(self.model_name, hidden_layer_values)

            # if len(hidden_layer_values.shape) != 2:  # especially for cifar100
            #     hidden_layer_values = flatten_layer_output(hidden_layer_values)
            num_features = hidden_layer_values.shape[1]

            # calculate the difference of confidence score
            ns = []
            org_prob = prob_dict[c]
            org_conf = np.max(org_prob, axis=1)

            try:
                batch_size = dataloader.batch_size
            except Exception:
                first_batch = next(iter(dataloader))
                batch_size = first_batch[0].size(0)

            # iterate for each intermediate feature dimension
            for i in tqdm(range(num_features), desc="Iterate each feature dimension: "):
                class_pattern = np.ones(num_features)
                class_pattern[i] = 0
                # mask i-th latent feature
                layer_outputs = hidden_layer_values * class_pattern
                layer_outputs_torch = torch.tensor(layer_outputs, dtype=torch.float32).to(self.device)

                vec_loader = DataLoader(VectorDataset(layer_outputs_torch),
                                         batch_size=batch_size, shuffle=False, num_workers=0)

                # get confidence outputs after feature mask
                after_conf_list = []
                with torch.no_grad():
                    for batch in vec_loader:
                        batch = batch.to(self.device)
                        batch_conf = remaining_layers(batch)
                        batch_conf = torch.max(batch_conf, dim=1)[0]
                        after_conf_list.append(batch_conf.cpu().numpy())

                # get contribution of each feature dimension
                after_conf = np.concatenate(after_conf_list, axis=0)
                ns.append(org_conf.mean() - after_conf.mean())
            self.class_ns.append(np.array(ns))

    def fetch_class_patterns_ns(self, p):
        self.class_patterns = []
        # for c in range(self.classes):
        for c in tqdm(range(self.classes), desc="fetch_class_patterns_ns"):
            class_freq = self.class_ns[c]
            class_pattern = np.zeros(class_freq.shape, dtype=np.int8)
            NUMS = int(class_freq.shape[0] * p)  # 保留的特征数量
            if class_freq.shape[0]-NUMS == 0:
                NUMS -= 1  # ensure as least filter 1 feature
            ranks = np.argsort(-class_freq)
            class_pattern[ranks[:NUMS]] = 1
            self.class_patterns.append(class_pattern)

    def get_purified_dataloader(self, cand_loader, cand_labels):
        remaining_layers = self.get_remaining_model(self.model)
        predict_label = np.array([t.cpu() for t in cand_labels])
        # get intermediate outputs
        hidden_layer_values = extract_layer_output(self.model, cand_loader, self.layer_name)

        hidden_layer_values = process_feature(self.model_name, hidden_layer_values)

        # mask i-th latent feature
        class_pattern = np.array([self.class_patterns[c] for c in predict_label])

        if hidden_layer_values.ndim == 3 and class_pattern.ndim == 2:
            hidden_layer_values = hidden_layer_values.squeeze(-1)
        layer_outputs = hidden_layer_values * class_pattern

        try:
            batch_size = cand_loader.batch_size
        except Exception:
            first_batch = next(iter(cand_loader))
            batch_size = first_batch[0].size(0)

        # get vec dataloader
        layer_outputs_torch = torch.tensor(layer_outputs, dtype=torch.float32).to(self.device)
        vec_loader = DataLoader(VectorDataset(layer_outputs_torch),
                                batch_size=batch_size, shuffle=False)

        # get confidence outputs after feature mask
        after_prob_list = []
        with torch.no_grad():
            for batch in vec_loader:
                batch = batch.to(self.device)
                batch_prob = remaining_layers(batch)
                after_prob_list.append(batch_prob.cpu().numpy())

        after_prob = np.concatenate(after_prob_list, axis=0)
        prob_new_loader = DataLoader(VectorDataset(after_prob),
                                batch_size=batch_size, shuffle=False)
        return prob_new_loader

def process_feature(model_name, x):
    # extra computation for forward
    if model_name == "C100-ShuffleNetv2_x1_5":
        x = x.mean(axis=(2, 3))
    elif model_name in ["IM100-FastViT_T8", "IM100-FastViT_S12"]:
        x = x.reshape(x.shape[0], -1)

    return x


