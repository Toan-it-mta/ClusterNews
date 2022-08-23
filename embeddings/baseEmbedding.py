import json
from multiprocessing import parent_process
from xml.sax import parseString
from sklearn.feature_extraction.text import CountVectorizer


class BaseEmbedding:
    def __init__(self):
        pass

    def get_vocab_from_text(self, text):
        pass

    def get_embedding_for_text(self, text):
        pass

    def save_vocab(self, path_vocab=None):
        pass

    def load_vocab(self, path_vocab=None):
        pass
