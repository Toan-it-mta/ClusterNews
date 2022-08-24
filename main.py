from cluster.cluster import Cluster
import embeddings
from utils import *
from embeddings.keywordEmbedding import KeywordExtractionEmbedding
import json


class Pipeline:
    def __init__(self) -> None:
        self.embeddingKeyWordExtractor = KeywordExtractionEmbedding()
        self.embeddingKeyWordExtractor.load_vocab()
        self.clusterFactory = Cluster()

    def pipeline(self, text, alg='keywords'):
        embeddingExtractor = None
        if alg == 'keywords':
            embeddingExtractor = self.embeddingKeyWordExtractor
        # Tiền xử lý dũ liệu
        clean_text = clean_str(text)
        # Lấy embedding của text
        _embedding = embeddingExtractor.get_embedding_for_text(clean_text)
        # Thực hiện phân cụm
        numberCluster = self.clusterFactory.clustering(
            embedding=_embedding, alg='keywords', threshold=0.8)
        return numberCluster


def test():
    pipeline = Pipeline()
    f = open('./data/100k_news_vi.json', 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    for item in data:
        text = item['content']
        cluster = pipeline.pipeline(text)
        print('cluster: ', cluster)
    # pipeline.embeddingKeyWordExtractor.save_vocab()


if __name__ == "__main__":
    test()
    # text = "Theo CNN, bão Laura đổ bộ vào miền nam nước Mỹ hôm 27/8, khiến ít nhất 6 người thiệt mạng tại bang Louisiana. Các nạn nhân trong độ tuổi từ 14 đến 68. (Nguồn ảnh: Reuters/AJ/CNN)Nhiều tòa nhà bị sập, cây cối gãy đổ do sức mạnh của cơn bão. Những mảnh vỡ từ các tòa nhà bị sập nằm rải rác trên con đường ngập nước.Các cửa sổ của tòa nhà 22 tầng Capital One Tower ở thành phố Lake Charles bị thổi tung.Thống đốc bang Louisiana John Bel Edward cho biết tại cuộc họp báo: \"Chúng tôi đã phải hứng chịu thiệt hại to lớn\".Ông Edward khuyến cáo người dân nên ở trong nhà để đảm bảo an toàn.Al Jazeera đưa tin, khoảng 650.000 ngôi nhà và cơ sở kinh doanh tại Louisiana và Texas bị mất điện sáng 27/8 do ảnh hưởng của bão Laura.CNN đưa tin, New Jersey sẽ cử lực lượng tới Louisiana để hỗ trợ bang này trong hoạt động giải cứu và khắc phục hậu quả sau bão Laura.Trong khi đó, bang Arkansas cũng đã chuẩn bị tinh thần ứng phó với bão Laura.Tổng thống Donald Trump hôm 27/8 phê chuẩn tuyên bố khẩn cấp cho Arkansas, ủy quyền cho các quan chức liên bang giải ngân quỹ và điều phối nỗ lực cứu trợ bang này.Được biết, trước khi đổ bộ vào Mỹ, bão Laura đã tàn phá các quốc gia ở vùng Caribe, trong đó có Cộng hòa Dominica, Haiti và Cuba.Phần mái của một tòa nhà bị thổi bay sau khi bão Laura càn quét qua Orange, Texas, ngày 27/8.Đường phố ngập lụt ở Lake Charles, Louisiana.Cảnh tượng ngổn ngang bên trong một cửa hàng ở Lake Charles sau khi bão quét qua.Những cây cột điện bị đổ xuống đường tại Sulphur, Louisiana, ngày 27/8. Mời độc giả xem thêm video: Siêu bão Harvey đổ bộ vào Mỹ, người dân Texas đi tránh bão (Nguồn video: VTC1)  Theo CNN, bão Laura đổ bộ vào miền nam nước Mỹ hôm 27/8, khiến ít nhất 6 người thiệt mạng tại bang Louisiana. Các nạn nhân trong độ tuổi từ 14 đến 68. (Nguồn ảnh: Reuters/AJ/CNN)  Nhiều tòa nhà bị sập, cây cối gãy đổ do sức mạnh của cơn bão. Những mảnh vỡ từ các tòa nhà bị sập nằm rải rác trên con đường ngập nước.  Các cửa sổ của tòa nhà 22 tầng Capital One Tower ở thành phố Lake Charles bị thổi tung.  Thống đốc bang Louisiana John Bel Edward cho biết tại cuộc họp báo: \"Chúng tôi đã phải hứng chịu thiệt hại to lớn\".  Ông Edward khuyến cáo người dân nên ở trong nhà để đảm bảo an toàn.  Al Jazeera đưa tin, khoảng 650.000 ngôi nhà và cơ sở kinh doanh tại Louisiana và Texas bị mất điện sáng 27/8 do ảnh hưởng của bão Laura.  CNN đưa tin, New Jersey sẽ cử lực lượng tới Louisiana để hỗ trợ bang này trong hoạt động giải cứu và khắc phục hậu quả sau bão Laura.  Trong khi đó, bang Arkansas cũng đã chuẩn bị tinh thần ứng phó với bão Laura.  Tổng thống Donald Trump hôm 27/8 phê chuẩn tuyên bố khẩn cấp cho Arkansas, ủy quyền cho các quan chức liên bang giải ngân quỹ và điều phối nỗ lực cứu trợ bang này.  Được biết, trước khi đổ bộ vào Mỹ, bão Laura đã tàn phá các quốc gia ở vùng Caribe, trong đó có Cộng hòa Dominica, Haiti và Cuba.  Phần mái của một tòa nhà bị thổi bay sau khi bão Laura càn quét qua Orange, Texas, ngày 27/8.  Đường phố ngập lụt ở Lake Charles, Louisiana.  Cảnh tượng ngổn ngang bên trong một cửa hàng ở Lake Charles sau khi bão quét qua.  Những cây cột điện bị đổ xuống đường tại Sulphur, Louisiana, ngày 27/8.  Mời độc giả xem thêm video: Siêu bão Harvey đổ bộ vào Mỹ, người dân Texas đi tránh bão (Nguồn video: VTC1)"
    # pipeline = Pipeline()
    # number = pipeline.pipeline(text)
    # pipeline.embeddingKeyWordExtractor.save_vocab()
