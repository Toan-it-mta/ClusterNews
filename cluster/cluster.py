from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity


def get_score_similarity(X, Y, alg='entities'):
    if alg == 'entities':
        count = 0
        for i in range(0, len(X)):
            if X[i] == Y[i]:
                count += 1
        return count/len(X)
    else:
        return cosine_similarity([X], [Y])


class Cluster():
    def __init__(self):
        self.embeddings = dict()
        self.labels = dict()
        # self.embeddings = None
        self.count = 0
        self.cluster = 0
        # self.nbrs = NearestNeighbors(n_neighbors=1)

    def clustering(self, embedding, alg='eculic', threshold=0.8):
        try:
            # Khởi tạo bộ dữ liệu
            # _embedding = np.array([embedding])
            # if self.embeddings is None:
            #     self.embeddings = np.array(_embedding)
            # else:
            #     self.embeddings = np.append(
            #         self.embeddings, _embedding, axis=0)

            # nbrs = self.nbrs.fit(self.embeddings)
            # distances, indices = nbrs.kneighbors(_embedding)
            # print(distances)
            # print(indices)

            max_score = 0
            max_cluster = None
            if self.count != 0:
                # lấy embedidng các bài viết đã lưu
                for i in range(1, self.count+1):
                    embeddingCurrent = self.embeddings[i]
                    # Tính khoảng cách giữa bài viết hiện tại và bài viết đã lưu trữ
                    score = get_score_similarity(
                        embedding, embeddingCurrent, alg)

                    if score >= max_score:
                        max_score = score
                        max_cluster = self.labels[i]

            if max_score < threshold:
                self.cluster += 1
                max_cluster = self.cluster
                print('Create new clsuter: ', max_cluster)

            self.count += 1
            self.embeddings[self.count] = embedding
            self.labels[self.count] = max_cluster
            return max_cluster

        except Exception as e:
            print(e)


def test():
    cluster = Cluster()

    data = np.load(
        '/media/nguyenphuctoan/AI_Academy1/Project_VOSINT/NER_Event_Classify_News/news_cluster/10000x15000.npy')
    start = time.time()
    for i in range(0, 1000):
        cluster.clustering(data[i])
    print('Time: ', time.time()-start)


if __name__ == "__main__":
    test()
    # a = np.load('./10000x15000.npy')
    # print(a.shape)
    # print(np.random.rand(10))
