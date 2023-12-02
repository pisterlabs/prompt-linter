from qdrant_client.models import Distance, VectorParams,FieldCondition,Filter,MatchValue
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient
import time
import numpy as np
from qdrant_client.models import PointStruct
import uuid
import pandas as pd
import numpy as np
from tqdm import tqdm
namespace = uuid.UUID('{00010203-0405-0607-0809-0a0b0c0d0e0f}')
model_name = "/home/user/panyongcan/project/big_model/m3e-base"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
class zhuanliKey:
    def __init__(self,host='10.0.0.14',collect_name="patent_key_word1"):
        self.collect_name = collect_name
        self.host = host
        self.client = QdrantClient(self.host, port=6333)
        #client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
        collect_li = self.client.get_collections()
        self.collect_li = [x.name for x in collect_li.collections]
        if self.collect_name not in self.collect_li:
            self.client.recreate_collection(
                #collection_name=self.collect_name,
                collection_name=self.collect_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            self.client.create_payload_index(collection_name=self.collect_name, 
                                        field_name="content", 
                                        field_schema="keyword")
            
    def build_keys(self,csv_path,vec_path=None):
        #import ipdb
        #ipdb.set_trace()
        #loader = TextLoader(csv_path)
        loader = DirectoryLoader(csv_path, glob="**/*.txt",show_progress=True,use_multithreading=True,loader_cls=TextLoader)
        #loader = DirectoryLoader('./', glob="**/*.txt",show_progress=True,use_multithreading=True,loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200,separators=["\n\n", "\n","。","\r","\u3000"])
        #text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200,separators=["\n\n", "\n", " ", ""])
        texts = text_splitter.split_documents(documents)
        #df_reader = pd.read_csv(csv_path,chunksize=20)
        ps = []
        bucket = []
        for text in tqdm(texts):
            text = text.page_content.strip()
            if len(text) < 50:
                continue
            bucket.append(text)
            if len(bucket) >= 500:
                vecs = hf_embeddings.embed_documents(bucket)
                assert len(vecs) == len(bucket)
                for vec,content in zip(vecs,bucket):
                    #key = str(row['pn'].strip())
                    key = str(content)
                    ps.append(
                        PointStruct(
                            #id=idx,
                            id=str(uuid.uuid3(namespace, key)),
                            vector=vec,
                            #payload={"content": content, "key": key}
                            payload={"content": content}
                        ))
                bucket = []
            if len(ps)>= 500:
                cnt = 0
                while cnt <= 3:
                    try:
                        self.client.upsert(collection_name=self.collect_name,points=ps)
                        ps = []
                        break
                    except Exception as e:
                        print(e)
                        try:
                            print('现在尝试进行第{}次链接'.format(cnt))
                            time.sleep(5)
                            self.client = QdrantClient(self.host, port=6333)
                            self.client.upsert(collection_name=self.collect_name,points=ps)
                            ps = []
                            print('第{}次链接成功，写入成功'.format(cnt))
                            break
                        except:
                            print('第{}次链接失败，休息{}s'.format(cnt,10*cnt))
                            time.sleep(10*cnt)
                            cnt += 1
                #if len(ps) > 0 and cnt >= 3:
                if cnt >= 3:
                    raise TimeoutError('qdrants insert timeout')
    def search_by_vertix(self,vector):
        hits = self.client.search(
            collection_name=self.collect_name,
            query_vector=vector,
            limit=1  # Return 5 closest points
        )
        return hits
    def search_by_name(self,q_str,limit=10):
        try:
            res = self.client.scroll(
                collection_name=self.collect_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="content", 
                            #key="key", 
                            match=MatchValue(value=q_str)
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
                #with_vector=False,
                #with_vector=True,
            )
            return res
        except Exception as e:
            print(e)
        return None
    def search_by_pn(self,pn):
        pn_vid = str(uuid.uuid3(namespace, pn))
        res = self.search_by_id(pn_vid)
        return res
    def search_by_id(self,vid):

        if not isinstance(vid,list):
            vid = [vid]
        res = self.client.retrieve(
            collection_name=self.collect_name,
            ids=vid,
            )
        return res
    def delete_collcet_by_keyword(self,keyword):
        if not keyword:
            print('你输入的keyword为空，不删除任何collect')
            return None
        for collect_name in self.collect_li:
            if keyword in collect_name:
                input_str = input(f'警告⚠️：你要删除的collcet是\033[31m{collect_name}\033[0m,确认删除请按Y，不删除请按N')
                if 'Y'== input_str:
                    self.client.delete_collection(collection_name=collect_name)
                    print(f'删除{collect_name}成功')
                else:
                    print(f'你输入的是{input_str},不是"Y",不对{collect_name}进行删除')

    
        
if __name__ == '__main__':
    import sys
    #argv = sys.argv
    #csv_path = 'data.txt'
    csv_path = '/home/user/panyongcan/project/big_data/pretrain/chanye'
    #csv_path = argv[1]
    #if not csv_path:
    #    print('请加参数')
    #vec_path = '/home/zhangshao/zhangshao/Similarity/examples/data/patent_word_pn_3.npy'
    #zlk = zhuanliKey(collect_name="patent_key_word")
    zlk = zhuanliKey(collect_name="test_query")
    zlk.build_keys(csv_path)
    #import ipdb
    #ipdb.set_trace()
    res = zlk.search_by_name('南阳市')
    #res = zlk.search_by_id('985b95d0-69a1-3f25-b3bc-6779a3d08266')
    #res = zlk.search_by_pn('CN201520086793.3')
    #res = zlk.search_by_pn('CN201510463993.0')
    print(res)
#import ipdb
#ipdb.set_trace()
