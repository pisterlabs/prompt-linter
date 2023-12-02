import re
import time
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from typing import Dict, List, Optional, Tuple, Union
from plugins.common import settings, allowCROS
from bottle import route, request, static_file

MetadataFilter = Dict[str, Union[str, int, bool]]
COLLECTION_NAME = settings.librarys.qdrant.collection
divider = "\n"


model_kwargs = {'device': settings.librarys.qdrant.device}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=settings.librarys.qdrant.model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
class QdrantIndex(object):
    def __init__(self, embedding_model):
        if(settings.librarys.qdrant.qdrant_path):
            self.qdrant_client = QdrantClient(
                path=settings.librarys.qdrant.qdrant_path,
            )
        elif(settings.librarys.qdrant.qdrant_host):
            self.qdrant_client = QdrantClient(
                url=settings.librarys.qdrant.qdrant_host,
            )

        self.embedding_model = embedding_model
        self.collection_name = COLLECTION_NAME

    def similarity_search_with_score(
            self, query, k=settings.librarys.qdrant.count
    ):
        embedding = self.embedding_model.encode(query)
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            with_payload=True,
            limit=k,
        )

        return results

    def retrieve_from_id(self, _id):
        return self.qdrant_client.retrieve(self.collection_name, [_id])[0]


def find(s, step=0):
    try:
        original_results = qdrant.similarity_search_with_score(s)
        docs = []
        #import ipdb
        #ipdb.set_trace()
        for sample in original_results:
            if sample.score < settings.librarys.qdrant.similarity_threshold:
                continue
            #docs.append(get_doc(sample, step))
            docs.append(get_my_doc(sample))
        print('docs length is {}'.format(len(docs)))
        content = '\n'.join(docs)
        #content = get_content(docs)
        related_res = get_related_content(s,content)
        return related_res
        #return content
    except Exception as e:
        print(e)
        #return []
        return ''


def get_content(res_li):
    res = []
    len_str = 0
    for da in res_li:
        link = da['link']
        article = g.extract(url=link)
        title = article.title
        cleaned_text = article.cleaned_text
        len_str += len(title)
        len_str += len(cleaned_text)
        res.append(title)
        res.append(cleaned_text)
        if len_str > 2000:
            break
    res =  '\n'.join(res)
    return res

def get_related_content(query,content):
    def clean_text(content):
        res = []
        for txt in content.split('\n'):
            if not txt:
                continue
            if len(txt) < 50:
                continue
            res.append(txt.strip())
        return '\n'.join(res)

    #content_li = content.split('。')
    #import ipdb
    #ipdb.set_trace()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50,separators=["\n\n", "\n","。","\r","\u3000"])
    #texts = text_splitter.split_documents(content)
    content = clean_text(content)
    content_li = text_splitter.split_text(content)
    content_li.append(query)
    embedding = hf_embeddings.embed_documents(content_li)
    score = cosine_similarity([embedding[-1]],embedding[0:-1])
    idxs = score.argsort()
    idxs = idxs[0][::-1]
    res = ['']*len(content_li)
    len_str = 0
    for idx in idxs:
        sub_content = content_li[idx]
        if not sub_content:
            continue
        if len(sub_content) < 15:
            continue
        len_str += len(sub_content)
        res[idx]=sub_content
        #if len_str > 1200:
        if len_str > 2800:
            break
    final_res = []
    #len_res = len(res)
    for idx,txt in enumerate(res):
        #start = idx - 3 if idx - 3 >= 0 else 0
        #end = idx + 3 if idx + 3 < len_res else len_res - 1
        #useful_count = len([i for i in res[start:end+1] if i])
        #ratio = useful_count / len(res[start:end+1])
        #if ratio > 0.28:
        #    final_res.append(content_li[idx])
        if txt.strip() and len(txt) > 10:
            final_res.append(txt)

    res = '\n\n'.join(final_res)
    #res = '\n'.join(res)
    return res[0:3700]
    #return res[0:4500]
def get_my_doc(doc):
    final_content = doc.payload["content"]
    return final_content
    
def get_doc(doc, step):
    #final_content = doc.payload["page_content"]
    final_content = doc.payload["content"]
    #doc_source = doc.payload["metadata"]["source"]
    doc_source = "qdrants"
    print("文段分数: ", doc.score, final_content)

    # 当前文段在对应文档中的分段数
    _id = int(doc.id[-3:], 16)
    if step > 0:
        for i in range(1, step+1):
            try:
                doc_before = qdrant.retrieve_from_id(doc.id[:-3] + str(hex(_id-i))[2:].zfill(3))
                # 可能出现哈希碰撞
                if doc_source == doc_before.payload["metadata"]["source"]:
                    final_content = process_strings(doc_before.payload["page_content"], divider, final_content)
            except:
                pass
            try:
                doc_after = qdrant.retrieve_from_id(doc.id[:-3] + str(hex(_id+i))[2:].zfill(3))
                # 可能出现哈希碰撞
                if doc_source == doc_after.payload["metadata"]["source"]:
                    final_content = process_strings(final_content, divider, doc_after.payload["page_content"])
            except:
                pass
    if doc_source.endswith(".pdf") or doc_source.endswith(".txt"):
        title = f"[{doc_source}](/{settings.librarys.qdrant.path}/{doc_source})"
    else:
        title = doc_source
    return {'title': title, 'content': re.sub(r'\n+', "\n", final_content), "score": doc.score}


def process_strings(A, C, B):
    """
    find the longest common suffix of A and prefix of B
    """
    common = ""
    for i in range(1, min(len(A), len(B)) + 1):
        if A[-i:] == B[:i]:
            common = A[-i:]
    # if there is a common substring, replace one of them with C and concatenate
    if common:
        return A[:-len(common)] + C + B
    # otherwise, just return A + B
    else:
        return A + B


embedding_model = SentenceTransformer(settings.librarys.qdrant.model_path, device=settings.librarys.qdrant.device)
qdrant = QdrantIndex(embedding_model)
