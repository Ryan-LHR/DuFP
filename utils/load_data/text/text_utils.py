# -*-coding:utf-8-*-
import re
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

def padding_(sentences, seq_len):
    """
    Padding word vector (from tdpr)
    i.e., Unify the length of each data piece
    """
    # padding with zero 0
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def preprocess_string(s):
    """preprocess of tdpr"""
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s


def tockenize(x_train, y_train, x_val, y_val):
    """
    Tockenize given input (from tdpr)
    i.e., convert word to vector
    """

    word_list = []

    from tqdm import tqdm
    stop_words = set(stopwords.words('english'))
    for i, sent in tqdm(enumerate(x_train), total=x_train.size):
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for i, sent in tqdm(enumerate(x_train), total=x_train.size):
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for i, sent in tqdm(enumerate(x_val), total=x_val.size):
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    # encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    # encoded_test = [1 if label == 'positive' else 0 for label in y_val]

    # final_list_train = np.array(final_list_train)
    encoded_train = np.array(y_train)
    # final_list_test = np.array(final_list_test)
    encoded_test = np.array(y_val)

    return final_list_train, encoded_train, final_list_test, encoded_test, onehot_dict