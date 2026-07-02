import os

import numpy as np
import torch
import corrupted_text
from corrupted_text import CorruptionWeights
try:
    from datasets import load_from_disk
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

from utils.candidate.corrupted.corrupted_utils import make_corruption_weights
from utils.load_data.text.dbpedia import load_dbpedia
from utils.load_data.text.hg_dataset_utils import ensure_labels_column, guess_text_column


def gen_corrupted_dbpedia(path_to_data, load_type, weight_values,
                       corruption_type, severity, dataset_name, seed=42):
    """
    Corrupt text dataset, and random select the same size samples with original test set
    """
    '(1) Setup'
    dataset_dir = os.path.join(path_to_data, f'{dataset_name}')
    save_dir = os.path.join(dataset_dir,
                            f'corrupted/corrupted_test_severity_{severity}_weights{weight_values}_seed{seed}')


    if load_type == 'processed':
        print(f"===> Loading processed corrupted dataset from: {save_dir}")
        test_set_corrupted = load_from_disk(save_dir)
        return test_set_corrupted

    print(f"===> Loading original {dataset_name} dataset:")
    train_set, test_set = load_dbpedia(path_to_data, dataset_name)

    '(2) Load raw text from test set'
    print(f'\ngenerate corrupted dataset: {dataset_name}-c')

    train_set, label_col = ensure_labels_column(train_set)
    test_set, label_col = ensure_labels_column(test_set)
    text_col = guess_text_column(test_set)

    x_train = train_set[text_col]
    x_test = test_set[text_col]

    # y_test = test_set[label_col]

    '(3) Corrupted'
    # x_train, x_test = x_train.tolist(), x_test.tolist()
    corruptor = corrupted_text.TextCorruptor(base_dataset=x_test + x_train,
                                             cache_dir=os.path.join(dataset_dir, '.mycache'))
    # weights for 4 corrupted operator
    weights = make_corruption_weights(
        typo=weight_values[0],
        autocomplete=weight_values[1],
        autocorrect=weight_values[2],
        synonym=weight_values[3],
    )
    # weights = make_corruption_weights()  # use default weights config

    # weights.set_weights(corruption)
    x_corrupted = corruptor.corrupt(x_test, severity=severity, seed=seed, weights=weights)

    # no consider weights
    x_corrupted = np.array(x_corrupted)

    '(4) Wrapped corrupted test_set'
    def replace_text(example, idx):
        example[text_col] = x_corrupted[idx]
        return example

    test_set_corrupted = test_set.map(
        replace_text,
        with_indices=True
    )

    '(5) Save and load'
    test_set_corrupted.save_to_disk(save_dir)

    test_set_corrupted = load_from_disk(save_dir)

    return test_set_corrupted

