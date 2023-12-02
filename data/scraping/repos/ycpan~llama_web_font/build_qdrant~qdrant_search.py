from qdrant_client import models
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import numpy as np
from qdrant_client.models import PointStruct
import uuid

namespace = uuid.UUID('{00010203-0405-0607-0809-0a0b0c0d0e0f}')
model_name = "/home/user/panyongcan/project/big_model/m3e-base"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
class qdrant:
    def __init__(self):
        self.client = QdrantClient("10.0.0.14", port=6333)
    def _search_by_vectory(self,query_vector,collection_name='patent_key_word'):
        cnt = 0
        while cnt <= 3:
            try:
                hits = self.client.search(
                collection_name=collection_name ,
                query_vector = query_vector,
                with_vectors=False,
                with_payload=True,
                #score_threshold=0.9799999,
                limit = 10, 
                search_params=models.SearchParams(
                #hnsw_ef=256,
                hnsw_ef=32,
                exact=False
                    ),
                
                )
                print(hits)
                return hits
            except Exception as e:
                cnt += 1
                print(f'time out,正在进行第{cnt}次重新链接')
                time.sleep(1 * cnt)
                try:
                    self.client = QdrantClient("10.0.0.14", port=6333)
                    print(f'第{cnt}次链接成功')
                except Exception as e:
                    print(f'time out,第{cnt}次重新链接失败')
    def search_by_vector_from_name(self,name,collection_name):
        name_vector = hf_embeddings.embed_query(name)
        hists = self._search_by_vectory(name_vector,collection_name)
        return hists
                    
    def search_with_filter(self,query_vector,filter_value,score_threshold=0.5,filter_key='key',collection_name="patent_key_word"):
        cnt = 0
        while cnt <= 3:
            try:
                hits = self.client.search(
                collection_name=collection_name,
                query_vector = query_vector,
                with_vectors=False,
                with_payload=True,
                score_threshold=score_threshold,
                #limit = 1000, 
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=filter_key,
                            match=models.MatchValue(
                                value=filter_value,
                            ),
                        )
                    ]
                ),

                search_params=models.SearchParams(
                #hnsw_ef=256,
                hnsw_ef=6,
                exact=False
                    ),
                
                )
                return hits
            except Exception as e:
                cnt += 1
                print(f'time out,正在进行第{cnt}次重新链接')
                time.sleep(1 * cnt)
                try:
                    self.client = QdrantClient("10.0.0.14", port=6333)
                    print(f'第{cnt}次链接成功')
                except Exception as e:
                    print(f'time out,第{cnt}次重新链接失败')
                    
    def search_by_pn(self,pn):
        pn_vid = str(uuid.uuid3(namespace, pn))
        res = self.search_by_id(pn_vid)
        return res
    def search_by_id(self,vid):

        if not isinstance(vid,list):
            vid = [vid]
        res = self.client.retrieve(
            #collection_name=self.collect_name,
            collection_name="patent_parent_children",
            ids=vid,
            )
        return res
    def process(self,df,vectors):
        data = []
        import ipdb
        ipdb.set_trace()
        for idx,row in tqdm(df[30:60].iterrows(),total=len(df)):
            hits = self.search(vectors[idx])
            for i in hits:
            #   if i.score >= 0.95:
                    data.append([row['0'],row['1'],i.score,i.payload['name'],i.payload['key']])
            #data.append([row['0'],row['1'],i.score,i.payload['name'],i.payload['key']])
        return data
        
if __name__ == '__main__':
    #df = pd.read_csv('/home/zhangshao/zhangshao/Similarity/examples/data/200_vid_words.csv')
    ##df =df[475:]
    #vectors = np.load('/home/zhangshao/zhangshao/Similarity/examples/data/200_vid_words_0.npy',allow_pickle=True)
    ##vectors=vectors[475:]
    #query_vector = np.random.rand(100)
    #assert len(df) == len(vectors),"len(df) != len(vectors)"
    #assert len(df) == len(vectors),"len(df) != len(vectors)"
    
    #import ipdb
    #ipdb.set_trace()
    qd = qdrant()
    #hists = qd.search_with_filter(vectors[59],filter_key='pn',filter_value='CN200820093321.0',collection_name='patent_key_word1')
    #hists = qd.search_with_filter(vectors[59],filter_key='name',filter_value='CN202110094980.6',collection_name='test_query')
    #hists = qd.search_with_filter(vectors[59],filter_key='content',filter_value='CN202110094980.6',collection_name='test_query')
    import ipdb
    ipdb.set_trace()
    hists = qd.search_by_vector_from_name('南阳市',collection_name='test_query')
    print(hists)
    #data = qd.process(df,vectors)

    #df_result = pd.DataFrame(data)
    #df_result.columns = ['vid','name','得分','pn','word']
    ##df_result.sort_values(by='得分',ascending=False,inplace=True)
    #df_result.to_csv('data/检索专利words.csv',index=0)
