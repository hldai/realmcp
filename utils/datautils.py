import pickle
import json


def load_vocab_file(filename):
    with open(filename, encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    return vocab, {t: i for i, t in enumerate(vocab)}


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, 4)
