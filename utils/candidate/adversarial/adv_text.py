import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from datasets import load_from_disk
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

from utils import get_device
from utils.load_data.text.dbpedia import load_dbpedia
from utils.load_data.text.hg_dataset_utils import ensure_labels_column, guess_text_column


def generate_adv_text(dataset_name, model_name, attack_types, model, dataloader, path_to_data, token_path):
    """

    """
    "(0) Lazy imports (TextAttack is heavy)"
    import datasets
    import torch
    import textattack
    from textattack import Attacker
    from textattack.attack_args import AttackArgs
    from textattack.attack_results import SuccessfulAttackResult
    from textattack.datasets import Dataset as TA_Dataset
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.shared.utils import set_seed as ta_set_seed

    "(1) Setup"
    attack_tag = "+".join(attack_types) if isinstance(attack_types, (list, tuple)) else str(attack_types)
    ta_set_seed(0)
    # device = get_device()

    "(2) Load model and dataset"
    # model and tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    model_wrapper = HuggingFaceModelWrapper(model.hf_model, tokenizer)

    print(f"===> Loading original {dataset_name} dataset:")
    _, test_set = load_dbpedia(path_to_data, dataset_name)
    # --sample for test--
    # N = 5
    # test_set = test_set.select(range(min(N, len(test_set))))
    test_set, label_col = ensure_labels_column(test_set)
    text_col = guess_text_column(test_set)

    texts: List[str] = test_set[text_col]
    labels = test_set[label_col]
    if not isinstance(labels, list):
        labels = labels.tolist()

    "(3) Attack Setup"
    recipe_names = list(attack_types) if isinstance(attack_types, (list, tuple)) else [str(attack_types)]
    adv_texts = list(texts)

    # Reproducible per-sample random assignment of a single recipe
    rng = np.random.default_rng(0)
    assigned_recipe = rng.choice(recipe_names, size=len(texts), replace=True).tolist()

    # Save
    out_root = os.path.join(
        path_to_data,
        dataset_name,
        "adv",
        f"{model_name}_{attack_tag}",
    )
    os.makedirs(out_root, exist_ok=True)

    "(4) Attack"
    for rname in recipe_names:
        # [MOD] Only attack samples assigned to this recipe
        rem_list = [i for i, rr in enumerate(assigned_recipe) if rr == rname]
        if len(rem_list) == 0:
            continue

        attack = _build_textattack_recipe(rname, model_wrapper)

        # build TA dataset from assigned samples (keep a mapping)
        ta_samples = [(adv_texts[i], labels[i]) for i in rem_list]
        ta_dataset = TA_Dataset(ta_samples)

        # checkpoint folder per recipe (so partial progress can resume/debug)
        ckpt_dir = os.path.join(out_root, "_checkpoints", rname)

        attack_args = AttackArgs(
            num_examples=-1,
            random_seed=0,
            checkpoint_interval=200,
            checkpoint_dir=ckpt_dir,
            disable_stdout=True,
            silent=True,
            log_to_csv=os.path.join(out_root, f"_textattack_log_{rname}.csv"),
            csv_coloring_style="plain",
        )

        attacker = Attacker(attack, ta_dataset, attack_args)
        results = attacker.attack_dataset()  # list aligned with ta_dataset order

        # write back results to global arrays
        for local_j, res in tqdm(enumerate(results), desc=f"Write results ({rname})"):
            global_i = rem_list[local_j]

            if isinstance(res, SuccessfulAttackResult):
                try:
                    perturbed_plain = res.perturbed_result.attacked_text.text
                    original_plain = res.original_result.attacked_text.text
                except Exception:
                    perturbed_plain = res.perturbed_text()
                    original_plain = res.original_text()

                adv_texts[global_i] = perturbed_plain

    adv_set = test_set
    adv_set = _safe_add_or_replace_column(adv_set, text_col, adv_texts)

    "(4) Save and Load"
    adv_set.save_to_disk(out_root)
    dataset = load_from_disk(out_root)
    return dataset


def _build_textattack_recipe(recipe_name: str, model_wrapper):
    """Map a short recipe name to TextAttack attack recipe build()."""
    from textattack.attack_recipes import (
        TextFoolerJin2019,
        PWWSRen2019,
        BAEGarg2019,
    )

    name = str(recipe_name).strip().lower()
    table = {
        "textfooler": TextFoolerJin2019,
        "pwws": PWWSRen2019,
        "bae": BAEGarg2019,
    }

    if name not in table:
        raise ValueError(
            f"Unknown TextAttack recipe: {recipe_name}. "
            f"Supported: {sorted(list(table.keys()))}"
        )

    recipe_cls = table[name]
    attack = recipe_cls.build(model_wrapper)

    # ---- Remove USE (Universal Sentence Encoder) constraint to avoid TF/tensorflow_hub ----
    # Your stack shows it comes from:
    # textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder
    new_constraints = []
    for c in attack.constraints:
        mod = c.__class__.__module__
        cname = c.__class__.__name__

        # Only drop the TFHub USE-based sentence encoder constraint
        if ("universal_sentence_encoder" in mod) or ("UniversalSentenceEncoder" in cname):
            continue
        new_constraints.append(c)

    attack.constraints = new_constraints
    return attack

def _count_word_changes(a: str, b: str) -> int:
    """Count changed word positions (rough metric)."""
    a_words = a.split()
    b_words = b.split()
    n = min(len(a_words), len(b_words))
    diff = sum(1 for i in range(n) if a_words[i] != b_words[i])
    diff += abs(len(a_words) - len(b_words))
    return int(diff)

def _safe_add_or_replace_column(hf_ds, col_name: str, values: List[Any]):
    """Add a column; if exists, remove and re-add (datasets doesn't allow overwrite in-place)."""
    if col_name in hf_ds.column_names:
        hf_ds = hf_ds.remove_columns([col_name])
    return hf_ds.add_column(col_name, values)


def load_adversarial_text(dataset_name, path_to_data, model_name, attack_types, mix):
    attack_tag = "+".join(attack_types) if isinstance(attack_types, (list, tuple)) else str(attack_types)
    save_dir = os.path.join(
        path_to_data,
        dataset_name,
        "adv",
        f"{model_name}_{attack_tag}",
    )
    mixed_save_dir = os.path.join(
        path_to_data,
        dataset_name,
        "adv",
        f"{model_name}_{attack_tag}_mix",
    )
    if mix:
        if not os.path.exists(mixed_save_dir):
            make_mixed_text_dataset(
                save_dir,
                mixed_save_dir,
                dataset_name,
                path_to_data,
                seed=42,
            )
        dataset = load_from_disk(mixed_save_dir)
    else:
        dataset = load_from_disk(save_dir)
    return dataset


def make_mixed_text_dataset(save_dir, mixed_save_dir, dataset_name, path_to_data, seed=42):

    # Load
    print(f"===> Loading adv {dataset_name} dataset:")
    adv_set = load_from_disk(save_dir)
    print(f"===> Loading original {dataset_name} dataset:")
    _, test_set = load_dbpedia(path_to_data, dataset_name)
    # --sample for test--
    # N = 5
    # test_set = test_set.select(range(min(N, len(test_set))))
    test_set, label_col = ensure_labels_column(test_set)
    text_col = guess_text_column(test_set)

    # Setup
    assert len(adv_set) == len(test_set), \
        f"Dataset size mismatch: adv_set={len(adv_set)} vs test_set={len(test_set)}"

    assert text_col in adv_set.column_names, \
        f"text_col='{text_col}' not found in adv_set columns: {adv_set.column_names}"

    n = len(test_set)
    half = n // 2

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    idx_adv = set(perm[:half].tolist())

    adv_texts = adv_set[text_col]
    org_texts = test_set[text_col]

    mixed_texts = [adv_texts[i] if i in idx_adv else org_texts[i] for i in range(n)]

    print(f"Save mixed dataset to {mixed_save_dir}")
    mix_set = _safe_add_or_replace_column(test_set, text_col, mixed_texts)
    os.makedirs(mixed_save_dir, exist_ok=True)
    mix_set.save_to_disk(mixed_save_dir)
    return