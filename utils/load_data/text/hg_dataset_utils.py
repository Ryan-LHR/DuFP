import torch
import torch.nn as nn
from tqdm import tqdm
try:
    from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


from utils.model_utils import get_predictions
from torch.utils.data import DataLoader, Subset


def guess_text_column(ds):
    """
    Get text column
    """

    # 优先用 content，其次 title，否则选第一个字符串列
    if "content" in ds.column_names:
        return "content"
    if "text" in ds.column_names:
        return "text"
    if "title" in ds.column_names:
        return "title"

    # fallback: 找第一个可能是文本的列
    for c in ds.column_names:
        v = ds[0][c]
        if isinstance(v, str):
            return c

    raise ValueError(f"Cannot find a text column in: {ds.column_names}")

def ensure_labels_column(ds):
    """
    Ensure the label column is named 'labels'.
    Returns:
        ds: possibly renamed dataset
        label_col: the final label column name (always 'labels')
    """
    if "labels" not in ds.column_names:
        if "label" in ds.column_names:
            ds = ds.rename_column("label", "labels")
        elif "topic" in ds.column_names:  # for yahoo
            ds = ds.rename_column("topic", "labels")
        else:
            raise ValueError(f"Cannot find a label column in: {ds.column_names}")
    return ds, "labels"

def is_tokenized_hf_dataset(dataset):
    """
    Check whether given dataset is a tokenized dataset

    True  -> ds looks like a tokenized HF dataset (has input_ids/attention_mask)
    False -> ds looks like a raw-text HF dataset
    """
    cols = set(dataset.column_names)
    # 只要出现这些字段，基本就确定是 tokenized
    if "input_ids" in cols and "attention_mask" in cols:
        return True
    return False

def build_text_dataloader(dataset, tokenizer, batch_size=32, shuffle=False, max_length=256, **common_loader_args):
    """
    Build text dataloader for HF text datasets.
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
        )

    if isinstance(dataset, Subset):  # convert a Subset object to a hg dataset
        base = dataset.dataset
        indices = dataset.indices
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        indices = list(map(int, indices))
        dataset = base.select(indices)

    if not is_tokenized_hf_dataset(dataset):  # original dataset
        text_col = guess_text_column(dataset)

        # label 列名统一成 labels（transformers 常用）
        dataset, label_col = ensure_labels_column(dataset)

        dataset_tok = dataset.map(tokenize_fn, batched=True)

    else:  # tokenized dataset
        dataset_tok = dataset

    # 只保留模型需要的字段：input_ids/attention_mask/token_type_ids(若有)/labels
    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset_tok.column_names:
        keep_cols.append("token_type_ids")

    remove_cols = [c for c in dataset_tok.column_names if c not in keep_cols]
    dataset_tok = dataset_tok.remove_columns(remove_cols)

    # 动态 padding 的 collate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = DeviceSplitInputsLabelsCollator(tokenizer, device)

    dataloader = DataLoader(dataset_tok, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, **common_loader_args)

    return dataloader


class DeviceSplitInputsLabelsCollator:
    """
    Wrapper for text classification tasks (for Collator and Dataloader)
    (directly aggregate inputs)
    """
    def __init__(self, tokenizer, device):
        self.inner = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        self.device = device

    def __call__(self, features):
        batch = self.inner(features)  # dict: input_ids, attention_mask, (token_type_ids), labels
        labels = batch.pop("labels")

        batch = {k: v.to(self.device) for k, v in batch.items()}
        labels = labels.to(self.device)
        return batch, labels


class HFSeqClsWrapper(nn.Module):
    """
    Wrapper for text classification tasks (for DNN models)
    (directly return softmax probabilities for forward)
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, inputs):
        # 返回概率张量 [B, C]，可以直接 torch.max(., 1)
        out = self.hf_model(**inputs)
        return torch.softmax(out.logits, dim=-1)


# def test_prediction(dataset, tokenizer):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     path = r"E:\python学习\20241110 ResearchProject-Sci-2\ResearchProject-Sci-2\data\models\DBPedia\bert-base-uncased-dbpedia_14"
#     path = "/root/autodl-tmp/Projects/dimp/data/models/DBPedia/bert-base-uncased-dbpedia_14"
#     model = AutoModelForSequenceClassification.from_pretrained(path)
#
#     model = HFSeqClsWrapper(model)
#     model = model.to(device)
#     model.eval()
#
#     dataloader = build_text_dataloader(dataset, tokenizer, batch_size=32, shuffle=False)
#
#     get_predictions(model, dataloader, return_type='Tensor', is_print=True,
#                     data_name=None, device=None)
