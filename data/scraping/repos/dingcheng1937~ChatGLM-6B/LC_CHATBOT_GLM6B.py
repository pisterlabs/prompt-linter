from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.huggingface import \
    HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import \
    UnstructuredFileLoader
import sentence_transformers
import torch
from chatglm_llm import ChatGLM
import requests
import re
from plugins import settings
from langchain.schema import Document

session = requests.Session()
# 正则提取摘要和链接
title_pattern = re.compile(
    '<a.target=..blank..target..(.*?)</a>')
brief_pattern = re.compile('K=.SERP(.*?)</p>')
link_pattern = re.compile(
    '(?<=(a.target=._blank..target=._blank..href=.))(.*?)(?=(..h=))')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
proxies = {"http": None, "https": None, }


def find(search_query, headers=headers, proxies=proxies):
    url = 'https://cn.bing.com/search?q={}'.format(
        search_query)
    res = session.get(url, headers=headers, proxies=proxies)
    r = res.text

    title = title_pattern.findall(r)
    brief = brief_pattern.findall(r)
    link = link_pattern.findall(r)

    # 数据清洗
    clear_brief = []
    for i in brief:
        tmp = re.sub('<[^<]+?>', '', i).replace('\n',
                                                '').strip()
        tmp1 = re.sub('^.*&ensp;', '', tmp).replace('\n',
                                                    '').strip()
        tmp2 = re.sub('^.*>', '', tmp1).replace('\n',
                                                '').strip()
        clear_brief.append(tmp2)

    clear_title = []
    for i in title:
        tmp = re.sub('^.*?>', '', i).replace('\n',
                                             '').strip()
        tmp2 = re.sub('<[^<]+?>', '', tmp).replace('\n',
                                                   '').strip()
        clear_title.append(tmp2)
    return [{'title': "[" + clear_title[i] + "]", 'content': clear_brief[i]}
            for i in
            range(min(settings.chunk_count, len(brief)))]


# Global Parameters
EMBEDDING_MODEL = "text2vec"
VECTOR_SEARCH_TOP_K = 3
LLM_MODEL = "chatglm-6b"
LLM_HISTORY_LEN = 3
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

llm_model_dict = {
    "chatglm-6b-int4-qe": "./model/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "./model/chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
}


def init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN,
             V_SEARCH_TOP_K=6):
    global chatglm, embeddings, VECTOR_SEARCH_TOP_K
    VECTOR_SEARCH_TOP_K = V_SEARCH_TOP_K

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[EMBEDDING_MODEL], )
    embeddings.client = sentence_transformers.SentenceTransformer(
        embeddings.model_name, device=DEVICE)

    chatglm = ChatGLM()
    chatglm.load_model(
        model_name_or_path=llm_model_dict[LLM_MODEL])
    chatglm.history_len = LLM_HISTORY_LEN


def AskLLM(query, vector_store):
    global chatglm, embeddings

    # Prompt template for knowledge chain
    prompt_template = """基于以下已知信息，请扮演小雀儿来回答。
                        如果无法从中得到答案，请说 
                        "小雀儿也不懂呢"。 答案请使用中文。

                        已知内容:
                        {context}

                        问题:
                        {question}"""
    # Instantiate the prompt template
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context",
                                             "question"])

    # print(vector_store)
    # Fetch the searchtool result
    search_result = find(query)
    # print(search_result)
    documents = [Document(page_content=result['content'])
                 for result in search_result]
    vector_store.add_documents(documents)

    # Update the retriever in knowledge_chain with the new vector store

    # Instantiate the knowledge chain
    knowledge_chain = RetrievalQA.from_llm(
        llm=chatglm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        prompt=prompt,
    )
    # Set the document prompt for the combine documents chain
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Enable returning source documents
    knowledge_chain.return_source_documents = True

    # Call the knowledge chain with a query
    result = knowledge_chain({"query": query})

    # Print the response
    # print(f"debug: {result}")
    # print(f"提问： {result['query']}, 回答： {result['result']}")
    print(f"[debug] search result: {search_result}\n")
    return result

def debugVS(vector_store):
    for doc_id, doc in vector_store.docstore.doc_dict.items():
        print(doc_id, doc)



init_cfg(LLM_MODEL, EMBEDDING_MODEL, LLM_HISTORY_LEN)
loader = UnstructuredFileLoader("output2.txt",
                                mode="elements")
docs = []
docs += loader.load()
vector_store = FAISS.from_documents(docs, embeddings)
#debugVS(vector_store)
# print(find("How to learn Python?"))
search_query = "今天几月几号?"
QueryList1 = ["原神",
              "神里绫华",
              "今天上海星期几?",
              "今天几月几号?",
              "今天上海天气如何?",

              "你是谁?",
              "你来自哪里",
              "讲个冷笑话",
              "你好可爱",
              "我不喜欢你"]
for query in QueryList1:
    print(f"[用户]提问：{query}\n")
    resp = AskLLM(query, vector_store)
    print(f"[小雀儿]回答：{resp['result']}\n")
while True:
    query = input("[用户]Input your question 请输入问题：")
    resp = AskLLM(query, vector_store)
    print(f"[小雀儿]回答：{resp['result']}\n")