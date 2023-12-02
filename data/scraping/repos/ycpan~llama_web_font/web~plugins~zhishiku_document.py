import http.client
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import json
import os
import fitz
import docx
from goose3 import Goose
from goose3.text import StopWordsChinese
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from plugins.common import settings, allowCROS
model_name = settings.librarys.qdrant.model_path
document_path = settings.librarys.document.knowledge_path
#SERPER_API_KEY=os.getenv("d50d0b2ff04a3bc6ed8101333204d3d0c3281039")
model_kwargs = {'device': settings.librarys.qdrant.device}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
def read_txt_file(file_path):
    f = open(file_path,'r')
    data = f.readlines()
    return data
def read_doc_file(file_path):
    res = []
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        txt = para.text
        res.append(txt)
    return res
def read_pdf_file(file_path):
    res = []
    doc = fitz.open(file_path) # open a document
    for page in doc: # iterate the document pages
        #text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
        text = page.get_text()
        res.append(text)
    return res
def get_clean_data():
    res = []
    for file_name in os.listdir(document_path):
        data = []
        file_path = os.path.join(document_path,file_name)
        if 'txt' in file_path[-4:]: 
            data = read_txt_file(file_path)
        if 'pdf' in file_path[-4:]: 
            data = read_pdf_file(file_path)
        if 'doc' in file_path[-4:]: 
            data = read_doc_file(file_path)
        res.extend(data)
    return '\n'.join(res)

clean_data = get_clean_data()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
#texts = text_splitter.split_documents(content)
content_li = text_splitter.split_text(clean_data)
document_embedding = hf_embeddings.embed_documents(content_li)
def find(search_query,step = 0):
    #clean_data = get_clean_data()
    #content = get_content(clean_data)
    content = clean_data
    related_res = get_related_content(search_query,content)
    #print(data)
    #l=[{'title': "["+organic["title"]+"]("+organic["link"]+")", 'content':organic["snippet"]} for organic in data['organic']]
    #try:
    #    if data.get("answerBox"):
    #        answer_box = data.get("answerBox", {})
    #        l.insert(0,{'title': "[answer："+answer_box["title"]+"]("+answer_box["link"]+")", 'content':answer_box["snippet"]})
    #except:
    #    pass
    #return l,data
    #return content
    return related_res
def get_content(res_li):
    """
    {'title': '《南阳市“十四五”现代服务业发展规划》政策解读', 'link': 'http://henan
    .kjzch.com/nanyang/2022-06-17/819566.html', 'snippet': '顺应产业融合需求，秉承“
    两业并举”，以服务产业升级、提高流通效率为导向，大力发展现代金融、现代物流、科技
    服务、信息服务、中介服务、节能环保服务、 ...', 'date': 'Jun 17, 2022', 'attribu
    tes': {'南阳市人民政府办公室': '2022-06-17'}, 'position': 1}
    """
    res = []
    len_str = 0
    for da in res_li:
        link = da['link']
        try:
            article = g.extract(url=link)
            title = article.title
            cleaned_text = article.cleaned_text
            len_str += len(title)
            len_str += len(cleaned_text)
            res.append(title)
            res.append(cleaned_text)
        except Exception as e:
            print(e)
        if len_str > 6000:
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
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25,separators=["\n\n", "\n","。","\r","\u3000"])
    ##texts = text_splitter.split_documents(content)
    #content = clean_text(content)
    #content_li = text_splitter.split_text(content)
    #content_li.append(query)
    query_embedding = hf_embeddings.embed_documents([query])
    #embedding = hf_embeddings.embed_documents(content_li)
    #score = cosine_similarity([embedding[-1]],embedding[0:-1])
    score = cosine_similarity(query_embedding,document_embedding)
    idxs = score.argsort()
    idxs = idxs[0][::-1]
    res = ['']*len(content_li)
    len_str = 0
    for idx in idxs:
        s = score[0][idx]
        if s < 0.72:
            continue
        sub_content = content_li[idx]
        if not sub_content:
            continue
        if len(sub_content) < 15:
            continue
        len_str += len(sub_content)
        res[idx]=sub_content
        #if len_str > 1000:
        if len_str > 6000:
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
    return res[0:1700]
    #return res[0:2500]


