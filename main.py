from cluster.cluster import Cluster
import embeddings
from utils import *
from embeddings.keywordEmbedding import KeywordExtractionEmbedding
import json


class Pipeline:
    def __init__(self) -> None:
        self.embeddingKeyWordExtractor = KeywordExtractionEmbedding()
        self.embeddingKeyWordExtractor.load_vocab()

    def pipeline(self, text, alg='keywords'):
        embeddingExtractor = None
        clusterFactory = Cluster()
        if alg == 'keywords':
            embeddingExtractor = self.embeddingKeyWordExtractor
        # Tiền xử lý dũ liệu
        clean_text = clean_str(text)
        # Lấy embedding của text
        _embedding = embeddingExtractor.get_embedding_for_text(clean_text)
        # Thực hiện phân cụm
        # numberCluster = clusterFactory.clustering(
        #     embedding=_embedding, alg='keywords', threshold=0.8)
        return "numberCluster"


def test():
    pipeline = Pipeline()
    f = open('./data/100k_news_vi.json', 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    for item in data:
        text = data['content']
        pipeline.pipeline(text)
    pipeline.embeddingKeyWordExtractor.save_vocab()


if __name__ == "__main__":

    text = "Tờ Bưu điện Hoa Nam Buổi sáng (Hong Kong, Trung Quốc) đưa tin máy bay ném bom B-2A có thể mang theo bom hạt nhân vào ngày 11/8 đã cất cánh từ căn cứ không quân Whiteman ở Missouri (Mỹ), băng qua Australia và hạ cánh tại Diego Garcia. Những chiếc B-2A này được tiếp liệu trên không vài lần.  Lần gần đây nhất máy bay ném bom được triển khai tới Diego Garcia, cách Maldives 1.200 km về phía Nam, là cách đây 4 năm.  Các nhà quan sát cho biết mặc dù 3 máy bay ném bom này không đi qua những khu vực nhạy cảm như Tây Thái Bình Dương, Biển Đông hoặc eo biển Đài Loan nhưng sự hiện diện của chúng tại Diego Garcia là dấu hiệu cho thấy Mỹ thể hiện sức mạnh quân sự ở Ấn Độ-Thái Bình Dương.  Ông Zhao Tong tại Quỹ Carnegie vì hòa bình quốc tế ở Bắc Kinh phân tích: “Hướng di chuyển của các máy bay ném bom là để thể hiện sức mạnh”.  B-2 là máy bay ném bom chiến lược tiên tiến nhất trên thế giới với khả năng tàng hình và có thể xuyên thủng hệ thống phòng không.  Nhà phân tích quân sự Zhou Chenming tại Bắc Kinh thừa nhận: “Máy bay ném bom tàng hình khó có thể phát hiện hoặc đánh chặn”.  Chiến đấu cơ Nga đã ra mặt chặn máy bay ném bom Mỹ B-1B Lancer đang hướng tới không phận nước này trên bầu trời Biển Đen."
    pipeline = Pipeline()
    number = pipeline.pipeline(text)
    print(number)
    pipeline.embeddingKeyWordExtractor.save_vocab()
