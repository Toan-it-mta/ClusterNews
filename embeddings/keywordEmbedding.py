from ast import keyword
import json
from sklearn.feature_extraction.text import CountVectorizer
from embeddings.baseEmbedding import BaseEmbedding
from yake import KeywordExtractor

# Lớp thực hiện Embbeding


class KeywordExtractionEmbedding(BaseEmbedding):
    def __init__(self):
        # Khởi tạo trình trích chọn Keyword
        self.keyWordsExtractor = KeywordExtractor(
            lan='vi', dedupLim=0.7, n=3, top=20, windowsSize=3)
        self.vocab = []
        self.vectorizer = CountVectorizer(
            lowercase=True, vocabulary=self.vocab)

    def get_vocab_from_text(self, text):
        words = text.split()
        for word in words:
            if word not in self.vocab:
                self.vocab.append(word)
        self.vectorizer.set_params(vocabulary=self.vocab)

    def get_embedding_for_text(self, text):
        # Trích xuất từ khóa
        keywords_scores = self.keyWordsExtractor.extract_keywords(text)
        keywords = [item[0] for item in keywords_scores]
        text_keyword = " ".join(keywords).lower()
        # Lấy Vocab
        self.get_vocab_from_text(text_keyword)
        # Chuyển đổi từ khóa thành véc-tơ
        return self.vectorizer.fit_transform([text_keyword]).toarray()[0]

    def save_vocab(self, path_vocab='./vocab.json'):
        _obj = {}
        _obj['vocab'] = self.vocab
        _file_ = open(path_vocab, 'w', encoding='utf-8')
        json.dump(_obj, _file_, ensure_ascii=False)
        _file_.close()

    def load_vocab(self, path_vocab='./vocab.json'):
        _file_ = open(path_vocab, 'r', encoding='utf-8')
        _obj = json.load(_file_)
        _file_.close()
        self.vocab = _obj['vocab']
