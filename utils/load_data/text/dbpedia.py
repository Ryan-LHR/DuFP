import os

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False


# from utils.load_data.text.hg_dataset_utils import test_prediction


def load_dbpedia(path_to_data, dataset_name, load_type='raw', token_path=None, only_test=False):
    """
    Load DBPedia Dataset and other huggingface text dataset.
    load_type:
        raw: load the raw text.
    """
    # (1) load raw text
    # download and save the raw text from datasets lib from huggingface
    dataset_dir = os.path.join(path_to_data, dataset_name)
    raw_path = os.path.join(dataset_dir, f'original')

    train_set = load_dataset(raw_path, split="train")
    test_set = load_dataset(raw_path, split="test")

    # if dataset_name == 'dbpedia_14':
    #     # web_path = "fancyzhx/dbpedia_14"
    #     None
    #
    # elif dataset_name == 'ag_news':
    #     # web_path = "fancyzhx/ag_news"
    #     None
    #     # dataset_dir = os.path.join(path_to_data, f'ag_news')
    #
    # elif dataset_name == 'yahoo_answers_topics':
    #     # web_path = "community-datasets/yahoo_answers_topics"
    #     None
    #     # dataset_dir = os.path.join(path_to_data, f'yahoo_answers_topics')

    if only_test:
        return test_set

    return train_set, test_set
