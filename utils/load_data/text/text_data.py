
from utils.load_data.text.imdb import load_imdb
from utils.load_data.text.us_airline import load_us_airline
from utils.load_data.text.dbpedia import load_dbpedia


def load_text_data(dataset_name, path_to_data):
    """
    Load Text Dataset
    """
    if dataset_name == 'imdb':
        train_set, test_set = load_imdb(path_to_data, dataset_name, load_type='processed')

    elif dataset_name == 'sms':
        # reuse the load function of imdb
        train_set, test_set = load_imdb(path_to_data, dataset_name, load_type='raw')

    elif dataset_name == 'us_airline':
        train_set, test_set = load_us_airline(path_to_data, dataset_name, load_type='processed')

    elif dataset_name in ['dbpedia_14', 'ag_news', 'yahoo_answers_topics']:
        train_set, test_set = load_dbpedia(path_to_data, dataset_name, load_type='raw')

    else:
        raise ValueError("Dataset Not Found")


    return train_set, test_set
