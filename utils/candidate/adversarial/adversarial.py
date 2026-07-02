import torch
from torch.utils.data import ConcatDataset, Subset

# from utils import sample_subset
from utils.load_data.load_data import IMAGE_DATASETS, TEXT_DATASETS
from utils.candidate.adversarial.adv_image import generate_adv_image, \
    load_adversarial_tensor_dataset, load_adversarial_im100
from utils.candidate.adversarial.adv_text import generate_adv_text, load_adversarial_text

def load_adversarial_data(dataset_name, model_name, path_to_data, load_type, model, test_set, test_loader,
                          model_file, mix=True):
    """Load adversarial dataset"""
    print('\nloading adversarial dataset...')

    if dataset_name in IMAGE_DATASETS:
        attack_types = ["fgsm", "bim", "pgd", "cw", "deepfool"]
    elif dataset_name in TEXT_DATASETS:
        attack_types = ["TextFooler", "PWWS"]
    else:
        raise ValueError("Dataset not found!")

    "(1) Generate"
    if load_type == 'not_generated':
        if dataset_name in IMAGE_DATASETS:
            generate_adv_image(dataset_name, model_name, attack_types, model, test_loader, path_to_data)
        elif dataset_name in TEXT_DATASETS:
            generate_adv_text(dataset_name, model_name, attack_types, model, test_loader, path_to_data, model_file)

        else:
            raise ValueError("Dataset not found!")

    "(2) Load and Mix"
    if dataset_name in IMAGE_DATASETS:
        if dataset_name == "imagenet_100":
            dataset = load_adversarial_im100(path_to_data, model_name, attack_types, mix)
        else:
            dataset = load_adversarial_tensor_dataset(dataset_name, model_name, path_to_data)
            if mix:
                dataset = concat_dataset(dataset, test_set)
    elif dataset_name in TEXT_DATASETS:
        dataset = load_adversarial_text(dataset_name, path_to_data, model_name, attack_types, mix)
    return dataset

def concat_dataset(cand_set, test_set, seed=42):
    """
    Construct a mixed dataset by taking half from test_set
    and half from cand_set
    """
    assert len(cand_set) == len(test_set), \
        f"Dataset size mismatch: cand_set={len(cand_set)}, test_set={len(test_set)}"
    total_size = len(test_set)
    split_a_size = total_size // 2
    # split_b_size = total_size - split_a_size

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=gen)

    idx_test = perm[:split_a_size]
    idx_cand = perm[split_a_size:]

    test_half = Subset(test_set, idx_test)
    cand_half = Subset(cand_set, idx_cand)

    cand_set_concat = ConcatDataset([test_half, cand_half])
    return cand_set_concat