from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import requests
import numpy as np
import re
import unicodedata

# Hàm lấy ner embedding


def get_embedding(vocab, entities):
    embedding = np.zeros(len(vocab))
    for i in range(0, len(vocab)):
        for entity in entities:
            if entity == vocab[i]:
                embedding[i] = 1
                break
    return embedding

# Hàm tính điểm tương đồng giữa 2 véc-tơ


def get_score_similarity(X, Y, alg='entities'):
    if alg == 'entities':
        count = 0
        for i in range(0, len(X)):
            if X[i] == Y[i]:
                count += 1
        return count/len(X)
    else:
        return cosine_similarity(np.asarray(X).reshape(1, -1), np.asarray(Y).reshape(1, -1))


def get_ner(text):
    URL = 'http://118.70.52.237:8001/ner'
    PARAMS = {
        "sentences": [
            {
                "text": text
            }
        ]
    }
    r = requests.post(url=URL, json=PARAMS)
    return r.json()


def clean_str(str):
    # Encode NFC string
    regex = "[^àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝA-Za-z0-9(),!?\'\`-]"
    str = unicodedata.normalize('NFC', str)
    str = str.strip()
    str = re.sub(regex, " ", str)
    str = re.sub(r"-", " - ", str)
    str = re.sub(r",", " , ", str)
    str = re.sub(r"!", " ! ", str)
    str = re.sub(r"'", " ' ", str)
    str = re.sub(r"\"", " \" ", str)
    str = re.sub(r"\(", " \( ", str)
    str = re.sub(r"\)", " \) ", str)
    str = re.sub(r"\?", " \? ", str)
    str = re.sub(r"\s{2,}", " ", str)
    return str
